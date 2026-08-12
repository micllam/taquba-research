//! [`ResearchStepRunner`]: the [`StepRunner`] that drives a research run
//! through its six phases.

use std::collections::HashSet;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use rig_agent::agent::{PromptResponse, TypedPromptResponse};
use rig_agent::client::AgentClientExt;
use rig_agent::completion::{Prompt, PromptError, StructuredOutputError, TypedPrompt};
use rig_core::OneOrMany;
use rig_core::client::CompletionClient;
use rig_core::completion::{
    CompletionError, Usage,
    message::{self, AssistantContent, UserContent},
};
use rig_core::http_client;
use rig_core::providers::anthropic::completion as anthropic_completion;
use rig_core::providers::{anthropic, ollama, openai};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use taquba::{LeaseHandle, Queue};
use taquba_jobs::{JobRunner, JoinError};
use taquba_workflow::{Memo, Step, StepError, StepOutcome, StepRunner};
use url::Url;

use crate::fetch_job::FetchPage;
use crate::report::{Citation, Report, RunStats, render_markdown};
use crate::search::{SearchBackend, SearchError};
use crate::state::{
    Phase, ResearchConfig, ResearchState, SourceQuote, Summary, SynthesisOutput, TokenUsage,
};
use crate::store::RunStore;

/// Preamble applied to every Rig agent built by the runner. Kept
/// terse: the per-phase prompts carry the task-specific instructions.
const AGENT_PREAMBLE: &str = "Be precise and concise.";
/// Cadence at which a step polls its cancellation sentinel while
/// phase work is in flight. Sets the upper bound on how long an LLM
/// or HTTP call keeps running after the CLI's `cancel` lands.
const CANCEL_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Upper bound on a single LLM completion call. The step's lease is
/// extended by this much before the call is issued, so the delivery
/// is not re-queued mid-call. Sized for slow local models.
const LLM_CALL_TIMEOUT: Duration = Duration::from_secs(600);
/// Upper bound on a single search-backend call, covered by the lease
/// the same way.
const SEARCH_CALL_TIMEOUT: Duration = Duration::from_secs(60);
/// Lease extension applied before each fetch-handle await: one
/// FetchPage job's full retry cycle (3 attempts of 20s each plus
/// backoff) plus the wait for a runner slot behind 16 concurrent
/// jobs, rounded up. Each completed handle re-extends, so the lease
/// stays within one job completion of live progress.
const FETCH_JOIN_LEASE: Duration = Duration::from_secs(150);

/// Memo user-keys for each phase's cached LLM response. Each is
/// scoped per `(run_id, step_number)` by [`taquba_workflow::Memo`],
/// so a plain string suffices.
const MEMO_KEY_PLANNING: &str = "planning";
const MEMO_KEY_SUMMARIZING: &str = "summarizing";
const MEMO_KEY_SYNTHESIZING: &str = "synthesizing";
const MEMO_KEY_WRITING: &str = "writing";

/// Drives a research run through plan -> search -> fetch -> summarize ->
/// synthesize -> write phases.
///
/// `ResearchStepRunner` is cheap to clone (internal `Arc`s). One instance
/// is shared across all worker tasks of a single [`WorkflowRuntime`].
///
/// [`WorkflowRuntime`]: taquba_workflow::WorkflowRuntime
#[derive(Clone)]
pub struct ResearchStepRunner {
    provider: Arc<ProviderClient>,
    search: Arc<dyn SearchBackend>,
    run_store: Option<RunStore>,
    job_runner: Option<Arc<JobRunner>>,
    queue: Option<Arc<Queue>>,
}

/// Per-provider LLM client.
pub(crate) enum ProviderClient {
    OpenAi(openai::Client),
    Anthropic(anthropic::Client),
    Ollama(ollama::Client),
}

/// What the terminal hook persists for a finished run. Distinct from the
/// in-flight state because terminal records only need the final report
/// plus enough metadata to render it. Failure reasons live on
/// [`taquba_workflow::RunOutcome::error`] instead of here, since the
/// runner builds a `RunRecord` only on success.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    /// Final report.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report: Option<Report>,
    /// Run identifier.
    pub run_id: String,
    /// Original query.
    pub query: String,
}

impl ResearchStepRunner {
    /// Build a runner from a Rig OpenAI client and a search backend.
    pub fn new_openai(client: openai::Client, search: Arc<dyn SearchBackend>) -> Self {
        Self::from_provider(ProviderClient::OpenAi(client), search)
    }

    /// Build a runner from a Rig Anthropic client and a search backend.
    pub fn new_anthropic(client: anthropic::Client, search: Arc<dyn SearchBackend>) -> Self {
        Self::from_provider(ProviderClient::Anthropic(client), search)
    }

    /// Build a runner from a Rig Ollama client and a search backend, for
    /// local models.
    pub fn new_ollama(client: ollama::Client, search: Arc<dyn SearchBackend>) -> Self {
        Self::from_provider(ProviderClient::Ollama(client), search)
    }

    pub(crate) fn from_provider(provider: ProviderClient, search: Arc<dyn SearchBackend>) -> Self {
        Self {
            provider: Arc::new(provider),
            search,
            run_store: None,
            job_runner: None,
            queue: None,
        }
    }

    /// Attach a [`RunStore`] used to check cancellation sentinels
    /// throughout each step. The runner polls the sentinel concurrently
    /// with the phase work, so a long-running LLM or HTTP call is
    /// dropped within ~1 second of the sentinel appearing. When unset,
    /// the runner ignores cancellation.
    pub fn with_run_store(mut self, store: RunStore) -> Self {
        self.run_store = Some(store);
        self
    }

    /// Attach the [`JobRunner`] the fetching phase submits per-URL
    /// `FetchPage` jobs to. Required when the workflow advances into
    /// `Phase::Fetching`: the phase submits one job per URL and
    /// `try_join_all`s the handles. The runner must already have
    /// `FetchPage` registered and an `Arc<reqwest::Client>` on its
    /// state; use [`crate::spawn_fetch_runner`] to build one with
    /// both already attached.
    pub fn with_job_runner(mut self, job_runner: Arc<JobRunner>) -> Self {
        self.job_runner = Some(job_runner);
        self
    }

    /// Attach the underlying [`Queue`] so the fetching phase can call
    /// `Queue::cancel(job_id)` on in-flight `FetchPage` jobs when the
    /// surrounding run is cancelled. Without this, those jobs run to
    /// completion (or to their reqwest timeout) after the run has
    /// already terminated.
    pub fn with_queue(mut self, queue: Arc<Queue>) -> Self {
        self.queue = Some(queue);
        self
    }

    /// Encode the initial state for a research run, ready to hand to
    /// [`taquba_workflow::RunSpec::input`].
    pub fn initial_state(query: impl Into<String>, config: ResearchConfig) -> Vec<u8> {
        ResearchState::new(query, config).to_bytes()
    }
}

impl StepRunner for ResearchStepRunner {
    #[tracing::instrument(
        skip_all,
        fields(run_id = %step.run_id, step = step.step_number)
    )]
    async fn run_step(&self, step: &Step) -> Result<StepOutcome, StepError> {
        let mut state = ResearchState::from_bytes(&step.payload)
            .map_err(|e| StepError::permanent(format!("malformed research state: {e}")))?;

        // Cross-process cancellation: the CLI's `cancel` subcommand
        // writes a sentinel object the runner watches for. The
        // workflow runtime's in-process registry doesn't see the
        // cross-process cancel, so we surface it as `StepOutcome::Cancel`
        // here. The watcher races the phase work, so a long LLM or HTTP
        // call is dropped within `CANCEL_POLL_INTERVAL` of the sentinel
        // appearing instead of blocking until the call completes.
        let work = self.dispatch_phase(step, &mut state);
        match &self.run_store {
            Some(store) => {
                tokio::select! {
                    res = work => res,
                    () = poll_cancelled(store, &step.run_id) => Ok(StepOutcome::Cancel {
                        reason: "Cancelled by user".to_string(),
                    }),
                }
            }
            None => work.await,
        }
    }
}

/// Resolves when the run's cancellation sentinel appears. Polls at
/// [`CANCEL_POLL_INTERVAL`] cadence; the initial check fires immediately
/// so an already-cancelled run short-circuits before any phase work runs.
async fn poll_cancelled(store: &RunStore, run_id: &str) {
    loop {
        if store.is_cancelled(run_id).await {
            return;
        }
        tokio::time::sleep(CANCEL_POLL_INTERVAL).await;
    }
}

/// Drop-guard that fires `Queue::cancel` on a non-empty set of
/// in-flight job IDs when dropped. Used by `run_fetching` to make
/// FetchPage jobs stop running when the surrounding research run
/// is cancelled mid-step, rather than running to the reqwest
/// timeout after the run has already terminated. Call
/// [`Self::disarm`] (which empties the ID list) on every controlled
/// exit so normal flows don't issue spurious `Queue::cancel` calls.
///
/// `Queue::cancel` is fire-and-forget here: the surrounding step is
/// being cancelled anyway, and a cancel call against an already-
/// terminal job is a no-op, so any individual failure to cancel is
/// logged at `warn` and ignored.
struct PendingJobsGuard {
    queue: Arc<Queue>,
    job_ids: Vec<String>,
}

impl PendingJobsGuard {
    fn new(queue: Arc<Queue>, job_ids: impl IntoIterator<Item = String>) -> Self {
        Self {
            queue,
            job_ids: job_ids.into_iter().collect(),
        }
    }

    fn disarm(&mut self) {
        self.job_ids.clear();
    }
}

impl Drop for PendingJobsGuard {
    fn drop(&mut self) {
        if self.job_ids.is_empty() {
            return;
        }
        let queue = self.queue.clone();
        let ids = std::mem::take(&mut self.job_ids);
        // Drop is sync; defer the awaits onto a background task.
        // The task survives the current step's cancellation and
        // gets the cancels to taquba before in-flight handlers
        // can run to completion.
        tokio::spawn(async move {
            for id in ids {
                if let Err(e) = queue.cancel(&id).await {
                    tracing::warn!(job_id = %id, error = %e, "failed to cancel fetch job");
                }
            }
        });
    }
}

#[derive(Debug, Clone)]
struct SynthesisDocument {
    /// The numbered source this document backs.
    citation: Citation,
    /// Full fetched body fed to the model for synthesis.
    text: String,
}

impl ResearchStepRunner {
    /// Runs the phase indicated by `state.phase` and returns the
    /// resulting `StepOutcome`. Separated from `run_step` so it can be
    /// raced against the cancellation watcher via `tokio::select!`.
    async fn dispatch_phase(
        &self,
        step: &Step,
        state: &mut ResearchState,
    ) -> Result<StepOutcome, StepError> {
        match state.phase {
            Phase::Planning => self.run_planning(step, state).await?,
            Phase::Searching => self.run_searching(step, state).await?,
            Phase::Fetching => self.run_fetching(step, state).await?,
            Phase::Summarizing => self.run_summarizing(step, state).await?,
            Phase::Synthesizing => self.run_synthesizing(step, state).await?,
            Phase::Writing => {
                let report = self.run_writing(step, state).await?;
                let record = RunRecord {
                    report: Some(report),
                    run_id: step.run_id.clone(),
                    query: state.query.clone(),
                };
                let bytes = serde_json::to_vec(&record)
                    .map_err(|e| StepError::permanent(format!("encoding report: {e}")))?;
                return Ok(StepOutcome::Succeed { result: bytes });
            }
        }

        state.steps_completed += 1;
        Ok(StepOutcome::continue_now(state.to_bytes()))
    }

    async fn run_planning(&self, step: &Step, state: &mut ResearchState) -> Result<(), StepError> {
        tracing::info!("planning sub-questions");
        let prompt = format!(
            "You are a research planner. Decompose the user's query into at most {} \
             distinct, web-searchable sub-questions that together cover the topic.\n\n\
             Query: {}",
            state.config.depth, state.query
        );

        // Memoize the LLM call so an at-least-once retry of this step
        // (lease expiry, worker restart) reuses the prior attempt's
        // response instead of re-paying for the same prompt.
        let plan: Plan = memoized(&step.memo, MEMO_KEY_PLANNING, async {
            self.llm_prompt_typed(&step.lease, &prompt, state).await
        })
        .await?;

        let questions: Vec<String> = plan
            .sub_questions
            .into_iter()
            .filter(|q| !q.trim().is_empty())
            .take(state.config.depth)
            .collect();

        if questions.is_empty() {
            return Err(StepError::transient(
                "plan returned no sub-questions; will retry",
            ));
        }

        state.sub_questions = questions;
        state.search_queue = (0..state.sub_questions.len()).collect();
        state.phase = Phase::Searching;
        tracing::info!("planned {} sub-questions", state.sub_questions.len());
        Ok(())
    }

    async fn run_searching(&self, step: &Step, state: &mut ResearchState) -> Result<(), StepError> {
        let Some(idx) = state.search_queue.pop_front() else {
            // Defensive: shouldn't happen because the transition below
            // moves us out of Searching as soon as the queue empties.
            state.phase = Phase::Fetching;
            return Ok(());
        };
        let q = state.sub_questions[idx].clone();
        let total = state.sub_questions.len();
        let done = total - state.search_queue.len(); // already-popped count
        tracing::info!("searching ({done}/{total}): {q}");

        // 5 results per sub-question is a reasonable starting point;
        // the total is capped later by max_sources.
        let results = under_lease(&step.lease, SEARCH_CALL_TIMEOUT, "search call", async {
            self.search.search(&q, 5).await.map_err(StepError::from)
        })
        .await?;

        let known: HashSet<Url> = state.fetched.keys().cloned().collect();
        let mut queued: HashSet<Url> = state.fetch_queue.iter().cloned().collect();

        for h in &results {
            if known.contains(&h.url) || queued.contains(&h.url) {
                continue;
            }
            if state.fetch_queue.len() >= state.config.max_sources {
                break;
            }
            state.fetch_queue.push_back(h.url.clone());
            queued.insert(h.url.clone());
        }
        state.search_results.insert(idx, results);

        if state.search_queue.is_empty() {
            state.phase = if state.fetch_queue.is_empty() {
                // No sources to fetch; skip ahead. The synthesis step
                // will gracefully produce a "nothing found" report.
                Phase::Synthesizing
            } else {
                Phase::Fetching
            };
        }
        Ok(())
    }

    async fn run_fetching(&self, step: &Step, state: &mut ResearchState) -> Result<(), StepError> {
        if state.fetch_queue.is_empty() {
            state.phase = Phase::Summarizing;
            return Ok(());
        }
        let job_runner = self
            .job_runner
            .as_ref()
            .ok_or_else(|| StepError::permanent("fetching phase requires a JobRunner"))?;
        let queue = self
            .queue
            .as_ref()
            .ok_or_else(|| StepError::permanent("fetching phase requires a Queue"))?;

        // Submit one FetchPage job per URL. Result-aware idempotent
        // submit means a retry of this step re-submits the same
        // payloads and either dedup-hits a still-pending submission
        // or short-circuits to a cached result blob; either way no
        // URL is fetched twice.
        let urls: Vec<Url> = state.fetch_queue.iter().cloned().collect();
        tracing::info!("fetching {} URLs in parallel", urls.len());
        let mut handles = Vec::with_capacity(urls.len());
        for url in &urls {
            let job = FetchPage {
                run_id: step.run_id.clone(),
                url: url.clone(),
                max_chars: state.config.max_page_chars,
            };
            let handle = job_runner
                .submit(job)
                .await
                .map_err(|e| StepError::transient(format!("fetch submit: {e}")))?;
            handles.push(handle);
        }

        // Arm a guard that cancels any still-in-flight jobs if this
        // future is dropped before completing, i.e. the surrounding
        // run was cancelled and the outer `run_step` is propagating
        // `StepOutcome::Cancel`. Disarmed on every controlled exit
        // (success, infra error) so normal flows don't issue
        // spurious `Queue::cancel` calls.
        let mut guard =
            PendingJobsGuard::new(queue.clone(), handles.iter().map(|h| h.id().to_string()));

        // Await all the handles. A per-URL handler failure
        // (`JoinError::Job`) is logged and skipped; an infrastructure
        // error (`JoinError::Infra`) fails the step transiently so
        // taquba re-delivers it.
        for (url, handle) in urls.iter().zip(handles) {
            // Each completed handle is a progress point: re-extend the
            // lease to cover the next job's worst-case completion.
            if let Err(e) = step.lease.ensure_at_least(FETCH_JOIN_LEASE) {
                // Let the still-in-flight jobs finish: a superseding
                // delivery's idempotent re-submits await these same
                // job ids.
                guard.disarm();
                return Err(lease_step_err(e));
            }
            match handle.await {
                Ok(page) => {
                    state.summarize_queue.push_back(url.clone());
                    state.fetched.insert(url.clone(), page);
                }
                Err(JoinError::Job(je)) => {
                    tracing::warn!(url = %url, error = %je, "fetch failed, skipping page");
                }
                Err(JoinError::Infra(infra)) => {
                    // Step will be retried; let the still-in-flight
                    // jobs finish so the retry's idempotent submits
                    // short-circuit to their cached results.
                    guard.disarm();
                    return Err(StepError::transient(format!("fetch infra: {infra}")));
                }
            }
        }

        guard.disarm();
        state.fetch_queue.clear();
        state.phase = Phase::Summarizing;
        Ok(())
    }

    async fn run_summarizing(
        &self,
        step: &Step,
        state: &mut ResearchState,
    ) -> Result<(), StepError> {
        let Some(url) = state.summarize_queue.pop_front() else {
            state.phase = Phase::Synthesizing;
            return Ok(());
        };
        let Some(page) = state.fetched.get(&url).cloned() else {
            // Shouldn't happen as summarize_queue is populated only after
            // a successful fetch.
            if state.summarize_queue.is_empty() {
                state.phase = Phase::Synthesizing;
            }
            return Ok(());
        };
        let total = state.summaries.len() + state.summarize_queue.len() + 1;
        let done = state.summaries.len() + 1;
        tracing::info!("summarizing ({done}/{total}): {url}");

        let prompt = format!(
            "You are a research assistant. The user is investigating:\n\n  {query}\n\n\
             Below is text extracted from a single web page (title: {title}). \
             Produce: (1) a 2-4 sentence summary focused on what is relevant to \
             the user's query, and (2) a relevance score from 0.0 (off-topic) to \
             1.0 (highly relevant).\n\n\
             Page text:\n{text}",
            query = state.query,
            title = page.title,
            text = page.text,
        );
        let parsed: SummaryResp = memoized(&step.memo, MEMO_KEY_SUMMARIZING, async {
            self.llm_prompt_typed(&step.lease, &prompt, state).await
        })
        .await?;

        state.summaries.insert(
            url,
            Summary {
                title: page.title,
                text: parsed.summary,
                relevance: parsed.relevance.clamp(0.0, 1.0),
            },
        );

        if state.summarize_queue.is_empty() {
            state.phase = Phase::Synthesizing;
        }
        Ok(())
    }

    async fn run_synthesizing(
        &self,
        step: &Step,
        state: &mut ResearchState,
    ) -> Result<(), StepError> {
        tracing::info!("synthesizing from {} sources", state.summaries.len());
        let mut sources = String::new();
        let mut source_documents = Vec::new();
        let mut citations: Vec<Citation> = Vec::new();
        let mut sorted: Vec<(&Url, &Summary)> = state.summaries.iter().collect();
        sorted.sort_by(|a, b| {
            b.1.relevance
                .partial_cmp(&a.1.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // The 1-based source number is the position in this sorted list and
        // is shared by the prompt text, the citation list, and any cited
        // excerpts. Building all three here keeps that numbering consistent.
        for (idx, &(url, s)) in sorted.iter().enumerate() {
            let index = idx + 1;
            sources.push_str(&format!(
                "Source {n} (relevance {r:.2}, title: {t}, URL: {u}):\n{x}\n\n",
                n = index,
                r = s.relevance,
                t = s.title,
                u = url,
                x = s.text,
            ));
            let citation = Citation {
                index,
                url: (*url).clone(),
                title: s.title.clone(),
            };
            if let Some(page) = state.fetched.get(url) {
                source_documents.push(SynthesisDocument {
                    citation: citation.clone(),
                    text: page.text.clone(),
                });
            }
            citations.push(citation);
        }
        if sources.is_empty() {
            sources.push_str("(no sources gathered)\n");
        }

        let prompt = format!(
            "You are a research analyst. Synthesize the following per-source \
             summaries into a coherent multi-paragraph narrative that directly \
             answers the user's query. Stay factual, cite sources by their \
             number in square brackets like [1], [2]. Do NOT fabricate \
             citations. If sources conflict, acknowledge the conflict.\n\n\
             User query: {q}\n\nSources:\n\n{s}",
            q = state.query,
            s = sources,
        );
        let synthesis: SynthesisOutput = memoized(&step.memo, MEMO_KEY_SYNTHESIZING, async {
            let (narrative, evidence) = self
                .llm_synthesis_prompt(&step.lease, &prompt, &source_documents, state)
                .await?;
            Ok(SynthesisOutput {
                narrative,
                citations,
                evidence,
            })
        })
        .await?;
        state.synthesis = Some(synthesis);
        state.phase = Phase::Writing;
        Ok(())
    }

    async fn run_writing(
        &self,
        step: &Step,
        state: &mut ResearchState,
    ) -> Result<Report, StepError> {
        tracing::info!("writing final report");
        let synthesis = state.synthesis.clone().unwrap_or_default();

        let prompt = format!(
            "You are a technical writer. Produce a polished markdown research \
             report answering the user's query, structured as: a one-sentence \
             TL;DR, then 2-5 sections with `## ` headers, then a `## Sources \
             cited` section listing the citations referenced in the body. \
             Preserve numeric citation markers like [1] from the synthesis. \
             Do not add citations beyond what is supplied. Do not include a \
             top-level `# ` header (it will be added by the runner).\n\n\
             User query: {q}\n\nSynthesis to expand:\n\n{syn}",
            q = state.query,
            syn = synthesis.narrative,
        );

        let body: String = memoized(&step.memo, MEMO_KEY_WRITING, async {
            self.llm_prompt(&step.lease, &prompt, state).await
        })
        .await?;

        let finished_at = Utc::now();
        let wall_time = (finished_at - state.started_at)
            .to_std()
            .unwrap_or_default();
        let stats = RunStats {
            steps_completed: state.steps_completed + 1,
            wall_time,
            started_at: state.started_at,
            finished_at,
            token_usage: state.token_usage,
        };

        let markdown = render_markdown(&state.query, &step.run_id, &body, &synthesis, &stats);
        Ok(Report {
            run_id: step.run_id.clone(),
            query: state.query.clone(),
            markdown,
            citations: synthesis.citations,
            stats,
        })
    }

    /// Run a single completion via Rig, dispatched to the configured
    /// provider, with the delivery lease extended to cover the call's
    /// [`LLM_CALL_TIMEOUT`] bound. Records this call's token usage on
    /// `state.token_usage` and logs the per-call counts at info level.
    async fn llm_prompt(
        &self,
        lease: &LeaseHandle,
        prompt: &str,
        state: &mut ResearchState,
    ) -> Result<String, StepError> {
        let model = &state.config.model;
        let max_tokens = state.config.max_tokens_per_call;
        let response = under_lease(lease, LLM_CALL_TIMEOUT, "LLM call", async {
            match self.provider.as_ref() {
                ProviderClient::OpenAi(client) => {
                    prompt_extended(client, model, max_tokens, prompt).await
                }
                ProviderClient::Anthropic(client) => {
                    prompt_extended(client, model, max_tokens, prompt).await
                }
                ProviderClient::Ollama(client) => {
                    prompt_extended(client, model, max_tokens, prompt).await
                }
            }
        })
        .await?;
        record_usage(&mut state.token_usage, &response.usage);
        Ok(response.output)
    }

    async fn llm_synthesis_prompt(
        &self,
        lease: &LeaseHandle,
        prompt: &str,
        source_documents: &[SynthesisDocument],
        state: &mut ResearchState,
    ) -> Result<(String, Vec<SourceQuote>), StepError> {
        match self.provider.as_ref() {
            ProviderClient::Anthropic(client) if !source_documents.is_empty() => {
                let message = anthropic_document_message(prompt, source_documents)?;
                let response = under_lease(lease, LLM_CALL_TIMEOUT, "LLM call", async {
                    prompt_extended(
                        client,
                        &state.config.model,
                        state.config.max_tokens_per_call,
                        message,
                    )
                    .await
                })
                .await?;
                record_usage(&mut state.token_usage, &response.usage);
                let evidence = response
                    .messages
                    .as_deref()
                    .map(|messages| extract_anthropic_source_quotes(messages, source_documents))
                    .transpose()?
                    .unwrap_or_default();
                Ok((response.output, evidence))
            }
            _ => Ok((self.llm_prompt(lease, prompt, state).await?, Vec::new())),
        }
    }

    /// Run a structured completion via Rig's `prompt_typed`, dispatched
    /// to the configured provider. Same lease and usage-tracking
    /// behaviour as [`Self::llm_prompt`].
    async fn llm_prompt_typed<T>(
        &self,
        lease: &LeaseHandle,
        prompt: &str,
        state: &mut ResearchState,
    ) -> Result<T, StepError>
    where
        T: JsonSchema + DeserializeOwned + Send + 'static,
    {
        let model = &state.config.model;
        let max_tokens = state.config.max_tokens_per_call;
        let response = under_lease(lease, LLM_CALL_TIMEOUT, "LLM call", async {
            match self.provider.as_ref() {
                ProviderClient::OpenAi(client) => {
                    prompt_typed_extended::<_, T>(client, model, max_tokens, prompt).await
                }
                ProviderClient::Anthropic(client) => {
                    prompt_typed_extended::<_, T>(client, model, max_tokens, prompt).await
                }
                ProviderClient::Ollama(client) => {
                    prompt_typed_extended::<_, T>(client, model, max_tokens, prompt).await
                }
            }
        })
        .await?;
        record_usage(&mut state.token_usage, &response.usage);
        Ok(response.output)
    }
}

/// Build a Rig agent for `model` with the shared preamble and per-call
/// token cap, then run `prompt` with extended details. The concrete agent
/// type differs per provider, so this is generic over the client; the
/// returned [`PromptResponse`] is provider-independent.
async fn prompt_extended<C>(
    client: &C,
    model: &str,
    max_tokens: u64,
    prompt: impl Into<message::Message> + Send,
) -> Result<PromptResponse, StepError>
where
    C: CompletionClient + AgentClientExt,
    C::CompletionModel: 'static,
{
    client
        .agent(model)
        .preamble(AGENT_PREAMBLE)
        .max_tokens(max_tokens)
        .build()
        .prompt(prompt)
        .extended_details()
        .await
        .map_err(classify_rig_err)
}

/// Structured counterpart to [`prompt_extended`], running Rig's
/// `prompt_typed` for the schema `T`.
async fn prompt_typed_extended<C, T>(
    client: &C,
    model: &str,
    max_tokens: u64,
    prompt: &str,
) -> Result<TypedPromptResponse<T>, StepError>
where
    C: CompletionClient + AgentClientExt,
    C::CompletionModel: 'static,
    T: JsonSchema + DeserializeOwned + Send + 'static,
{
    client
        .agent(model)
        .preamble(AGENT_PREAMBLE)
        .max_tokens(max_tokens)
        .build()
        .prompt_typed::<T>(prompt)
        .extended_details()
        .await
        .map_err(classify_structured_err)
}

/// Accumulate one call's `Usage` into the run-aggregate `TokenUsage`,
/// logging the per-call counts at info level.
fn record_usage(total: &mut TokenUsage, call: &Usage) {
    tracing::info!(
        input = call.input_tokens,
        output = call.output_tokens,
        total = call.total_tokens,
        cached_input = call.cached_input_tokens,
        reasoning = call.reasoning_tokens,
        "LLM call usage",
    );
    total.input_tokens = total.input_tokens.saturating_add(call.input_tokens);
    total.output_tokens = total.output_tokens.saturating_add(call.output_tokens);
    total.total_tokens = total.total_tokens.saturating_add(call.total_tokens);
    total.cached_input_tokens = total
        .cached_input_tokens
        .saturating_add(call.cached_input_tokens);
    total.cache_creation_input_tokens = total
        .cache_creation_input_tokens
        .saturating_add(call.cache_creation_input_tokens);
    total.reasoning_tokens = total.reasoning_tokens.saturating_add(call.reasoning_tokens);
}

/// Extend `lease` to cover `bound`, then run `work` under a timeout
/// of the same bound, so the delivery cannot outlive its lease
/// mid-call. A timeout is a transient step error; `what` names the
/// call in the error message.
async fn under_lease<T, F>(
    lease: &LeaseHandle,
    bound: Duration,
    what: &str,
    work: F,
) -> Result<T, StepError>
where
    F: Future<Output = Result<T, StepError>>,
{
    lease.ensure_at_least(bound).map_err(lease_step_err)?;
    match tokio::time::timeout(bound, work).await {
        Ok(result) => result,
        Err(_) => Err(StepError::transient(format!(
            "{what} timed out after {}s",
            bound.as_secs()
        ))),
    }
}

/// Map a failed lease extension to a transient step error. The
/// extension fails only when the claim was lost or the job's
/// cancellation was requested; in both cases this delivery's
/// settlement is rejected or its run terminated, so retrying is safe.
fn lease_step_err(e: taquba::Error) -> StepError {
    StepError::transient(format!("lease extension failed: {e}"))
}

/// Returns the JSON-decoded value previously written to `memo`
/// under `key`. If none exists, awaits `compute`, JSON-encodes its
/// result into the memo, and returns it; an at-least-once retry of
/// the surrounding step then finds the cached value and skips the
/// compute call entirely.
async fn memoized<T, F>(memo: &Memo, key: &str, compute: F) -> Result<T, StepError>
where
    T: Serialize + DeserializeOwned,
    F: Future<Output = Result<T, StepError>>,
{
    if let Some(bytes) = memo.get(key).await? {
        return serde_json::from_slice(&bytes)
            .map_err(|e| StepError::permanent(format!("memo[{key}] decode: {e}")));
    }
    let fresh = compute.await?;
    let bytes = serde_json::to_vec(&fresh)
        .map_err(|e| StepError::permanent(format!("memo[{key}] encode: {e}")))?;
    memo.put(key, &bytes).await?;
    Ok(fresh)
}

/// Classify a Rig prompt error as transient or permanent.
fn classify_rig_err(err: PromptError) -> StepError {
    let msg = err.to_string();
    match err {
        // Configuration / runner-policy issues.
        PromptError::MaxTurnsError { .. }
        | PromptError::PromptCancelled { .. }
        | PromptError::UnknownToolCall { .. } => {
            StepError::permanent(format!("LLM permanent failure: {msg}"))
        }

        PromptError::CompletionError(CompletionError::HttpError(http_err)) => {
            classify_http(&http_err, &msg)
        }

        // Anything else from the completion layer (ProviderError,
        // JsonError, UrlError, RequestError, ResponseError), plus
        // `MemoryError` and any future variant (`PromptError` is
        // non-exhaustive). Auth failures come through
        // `HttpError(InvalidStatusCode(401|403))` above, so we default
        // the rest to transient. Persistent ones get dead-lettered
        // after `max_attempts`.
        _ => StepError::transient(format!("LLM call failed: {msg}")),
    }
}

fn classify_http(err: &http_client::Error, msg: &str) -> StepError {
    let status = match err {
        http_client::Error::InvalidStatusCode(code)
        | http_client::Error::InvalidStatusCodeWithMessage(code, _) => Some(code.as_u16()),
        _ => None,
    };
    match status {
        Some(401 | 403) => StepError::permanent(format!("LLM authentication failure: {msg}")),
        Some(code) if !is_transient_status(code) => {
            StepError::permanent(format!("LLM client error {code}: {msg}"))
        }
        _ => StepError::transient(format!("LLM HTTP error: {msg}")),
    }
}

/// HTTP retry policy.
/// Transient: 429 (rate limit), 5xx (server errors), and any
/// non-4xx code.
/// Permanent: 4xx other than 429.
fn is_transient_status(code: u16) -> bool {
    code == 429 || !(400..500).contains(&code)
}

impl From<SearchError> for StepError {
    fn from(e: SearchError) -> Self {
        match e {
            SearchError::RateLimit { retry_after } => {
                StepError::transient(format!("search rate-limited (retry after {retry_after:?})"))
            }
            SearchError::Transport(err) => {
                StepError::transient(format!("search transport error: {err}"))
            }
            SearchError::AuthFailed => {
                StepError::permanent("search authentication failed".to_string())
            }
            SearchError::Other(msg) => StepError::transient(format!("search error: {msg}")),
        }
    }
}

/// Planning step response schema.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Plan {
    sub_questions: Vec<String>,
}

/// Summarizing step response schema.
#[derive(Serialize, Deserialize, JsonSchema)]
struct SummaryResp {
    summary: String,
    relevance: f32,
}

fn anthropic_document_message(
    prompt: &str,
    source_documents: &[SynthesisDocument],
) -> Result<message::Message, StepError> {
    let mut content = Vec::with_capacity(source_documents.len() + 1);
    content.push(UserContent::text(format!(
        "{prompt}\n\nUse the attached source documents as the authoritative \
         citation corpus. The documents are in the same order as the numbered \
         sources above. Cite relevant claims with the source numbers already \
         assigned above."
    )));

    for document in source_documents {
        content.push(UserContent::Document(message::Document {
            data: message::DocumentSourceKind::String(document.text.clone()),
            media_type: Some(message::DocumentMediaType::TXT),
            additional_params: Some(serde_json::json!({
                "title": format!("Source {}: {}", document.citation.index, document.citation.title),
                "context": format!("URL: {}", document.citation.url),
                "citations": { "enabled": true },
            })),
        }));
    }

    OneOrMany::many(content)
        .map(message::Message::from)
        .map_err(|e| StepError::permanent(format!("anthropic document prompt: {e}")))
}

fn extract_anthropic_source_quotes(
    messages: &[message::Message],
    source_documents: &[SynthesisDocument],
) -> Result<Vec<SourceQuote>, StepError> {
    let mut evidence = Vec::new();
    let mut seen = HashSet::new();

    for message in messages {
        let message::Message::Assistant { content, .. } = message else {
            continue;
        };
        for block in content.iter() {
            let AssistantContent::Text(text) = block else {
                continue;
            };
            let citations = anthropic_completion::anthropic_citations(text)
                .map_err(|e| StepError::permanent(format!("anthropic citation decode: {e}")))?;
            for citation in citations {
                let Some((document_index, cited_text)) =
                    anthropic_citation_document_span(&citation)
                else {
                    continue;
                };
                let Some(document) = source_documents.get(document_index) else {
                    continue;
                };
                if !seen.insert((document.citation.index, cited_text.to_string())) {
                    continue;
                }
                evidence.push(SourceQuote {
                    citation_index: document.citation.index,
                    excerpt: cited_text.to_string(),
                });
            }
        }
    }

    Ok(evidence)
}

fn anthropic_citation_document_span(
    citation: &anthropic_completion::Citation,
) -> Option<(usize, &str)> {
    match citation {
        anthropic_completion::Citation::CharLocation {
            document_index,
            cited_text,
            ..
        }
        | anthropic_completion::Citation::PageLocation {
            document_index,
            cited_text,
            ..
        }
        | anthropic_completion::Citation::ContentBlockLocation {
            document_index,
            cited_text,
            ..
        } => Some((*document_index, cited_text.as_str())),
        anthropic_completion::Citation::SearchResultLocation { .. }
        | anthropic_completion::Citation::WebSearchResultLocation { .. }
        | anthropic_completion::Citation::Unknown(_) => None,
    }
}

/// Classify a typed-prompt error. Delegates the wrapped `PromptError`
/// case to [`classify_rig_err`]; structured-output-specific variants
/// are mapped according to whether retrying would plausibly help.
fn classify_structured_err(err: StructuredOutputError) -> StepError {
    match err {
        StructuredOutputError::PromptError(inner) => classify_rig_err(*inner),
        StructuredOutputError::DeserializationError(e) => {
            StepError::permanent(format!("typed prompt: schema deserialize failed: {e}"))
        }
        // Empty response is the one variant that's typically a one-off
        // transient provider failure; retrying is appropriate.
        StructuredOutputError::EmptyResponse => {
            StepError::transient("typed prompt: model returned an empty response".to_string())
        }
        // `StructuredOutputError` is non-exhaustive; default future
        // variants to transient, matching `classify_rig_err`.
        other => StepError::transient(format!("typed prompt failed: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::StatusCode;
    use std::collections::HashMap;
    use taquba::object_store::memory::InMemory;
    use taquba::object_store::path::Path;
    use taquba_workflow::{MemoStore, StepErrorKind};
    use tokio_util::sync::CancellationToken;

    fn test_store() -> RunStore {
        RunStore::new(Arc::new(InMemory::new()), &Path::default())
    }

    /// Build a `Step` with a fresh in-memory `Memo` and otherwise
    /// inert fields, suitable for exercising memo-using helpers.
    fn test_step(run_id: &str, step_number: u32) -> Step {
        let memo =
            MemoStore::new(Arc::new(InMemory::new()), "test-memo").new_memo(run_id, step_number);
        Step {
            run_id: run_id.to_string(),
            step_number,
            payload: Vec::new(),
            headers: HashMap::new(),
            job_id: String::new(),
            attempts: 1,
            cancel_token: CancellationToken::new(),
            lease: taquba::LeaseHandle::detached(),
            memo,
            signal: None,
        }
    }

    fn assert_transient(err: &StepError) {
        assert!(
            matches!(err.kind, StepErrorKind::Transient),
            "expected transient, got {:?}: {}",
            err.kind,
            err.message
        );
    }

    fn assert_permanent(err: &StepError) {
        assert!(
            matches!(err.kind, StepErrorKind::Permanent),
            "expected permanent, got {:?}: {}",
            err.kind,
            err.message
        );
    }

    fn http_status(code: u16) -> http_client::Error {
        http_client::Error::InvalidStatusCode(StatusCode::from_u16(code).unwrap())
    }

    #[tokio::test]
    async fn poll_cancelled_returns_immediately_if_already_cancelled() {
        let store = test_store();
        store.mark_cancelled("run-1").await.unwrap();

        // Initial check fires before any sleep, so this returns
        // without yielding to the timer.
        poll_cancelled(&store, "run-1").await;
    }

    #[tokio::test(start_paused = true)]
    async fn poll_cancelled_resolves_after_sentinel_appears() {
        let store = test_store();
        let writer = store.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            writer.mark_cancelled("run-2").await.unwrap();
        });

        // With virtual time the polling loop's sleep auto-advances
        // when the runtime is idle.
        poll_cancelled(&store, "run-2").await;
    }

    #[test]
    fn is_transient_status_retries_rate_limit_and_5xx() {
        assert!(is_transient_status(429));
        assert!(is_transient_status(500));
        assert!(is_transient_status(502));
        assert!(is_transient_status(503));
        assert!(is_transient_status(504));
    }

    #[test]
    fn is_transient_status_marks_non_429_4xx_as_permanent() {
        assert!(!is_transient_status(400));
        assert!(!is_transient_status(401));
        assert!(!is_transient_status(403));
        assert!(!is_transient_status(404));
        assert!(!is_transient_status(422));
    }

    #[test]
    fn is_transient_status_defaults_outside_4xx_to_transient() {
        // Codes outside the standard 4xx range fall back to transient.
        assert!(is_transient_status(200));
        assert!(is_transient_status(0));
    }

    #[test]
    fn classify_http_routes_401_403_to_permanent() {
        assert_permanent(&classify_http(&http_status(401), "unauthorized"));
        assert_permanent(&classify_http(&http_status(403), "forbidden"));
    }

    #[test]
    fn classify_http_routes_other_4xx_to_permanent() {
        assert_permanent(&classify_http(&http_status(400), "bad request"));
        assert_permanent(&classify_http(&http_status(404), "not found"));
        assert_permanent(&classify_http(&http_status(422), "unprocessable"));
    }

    #[test]
    fn classify_http_routes_rate_limit_and_5xx_to_transient() {
        assert_transient(&classify_http(&http_status(429), "rate limited"));
        assert_transient(&classify_http(&http_status(500), "server error"));
        assert_transient(&classify_http(&http_status(503), "unavailable"));
    }

    #[test]
    fn classify_structured_err_empty_response_is_transient() {
        assert_transient(&classify_structured_err(
            StructuredOutputError::EmptyResponse,
        ));
    }

    #[test]
    fn classify_structured_err_deserialize_failure_is_permanent() {
        let json_err: serde_json::Error = serde_json::from_str::<i32>("nope").unwrap_err();
        assert_permanent(&classify_structured_err(
            StructuredOutputError::DeserializationError(json_err),
        ));
    }

    #[test]
    fn classify_rig_err_http_401_is_permanent() {
        let err = PromptError::CompletionError(CompletionError::HttpError(http_status(401)));
        assert_permanent(&classify_rig_err(err));
    }

    #[test]
    fn classify_rig_err_http_429_is_transient() {
        let err = PromptError::CompletionError(CompletionError::HttpError(http_status(429)));
        assert_transient(&classify_rig_err(err));
    }

    #[test]
    fn classify_rig_err_unknown_tool_call_is_permanent() {
        let err = PromptError::UnknownToolCall {
            tool_name: "lookup".to_string(),
            available_tools: Vec::new(),
            allowed_tools: Vec::new(),
            chat_history: Box::new(Vec::new()),
        };
        assert_permanent(&classify_rig_err(err));
    }

    #[tokio::test(start_paused = true)]
    async fn under_lease_times_out_transiently() {
        let lease = LeaseHandle::detached();
        let err = under_lease(
            &lease,
            Duration::from_secs(5),
            "test call",
            std::future::pending::<Result<(), StepError>>(),
        )
        .await
        .expect_err("pending work must time out");
        assert_transient(&err);
    }

    #[tokio::test]
    async fn under_lease_returns_the_inner_result() {
        let lease = LeaseHandle::detached();
        let value = under_lease(&lease, Duration::from_secs(5), "test call", async { Ok(7) })
            .await
            .unwrap();
        assert_eq!(value, 7);
    }

    #[test]
    fn lease_step_err_is_transient() {
        assert_transient(&lease_step_err(taquba::Error::ClaimLost));
    }

    #[test]
    fn extract_anthropic_source_quotes_maps_document_indices_to_sources() {
        let documents = vec![
            SynthesisDocument {
                citation: Citation {
                    index: 1,
                    url: "https://example.com/one".parse().unwrap(),
                    title: "One".to_string(),
                },
                text: "first source text".to_string(),
            },
            SynthesisDocument {
                citation: Citation {
                    index: 2,
                    url: "https://example.com/two".parse().unwrap(),
                    title: "Two".to_string(),
                },
                text: "second source text".to_string(),
            },
        ];
        let citations = serde_json::to_value(vec![anthropic_completion::Citation::CharLocation {
            cited_text: "second source text".to_string(),
            document_index: 1,
            document_title: Some("Source 2: Two".to_string()),
            start_char_index: 0,
            end_char_index: 18,
        }])
        .unwrap();
        let messages = vec![message::Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::Text(message::Text {
                text: "summary".to_string(),
                additional_params: Some(serde_json::json!({ "citations": citations })),
            })),
        }];

        let evidence = extract_anthropic_source_quotes(&messages, &documents).unwrap();

        assert_eq!(
            evidence,
            vec![SourceQuote {
                citation_index: 2,
                excerpt: "second source text".to_string(),
            }]
        );
    }

    #[test]
    fn extract_anthropic_source_quotes_deduplicates_repeated_spans() {
        let documents = vec![SynthesisDocument {
            citation: Citation {
                index: 1,
                url: "https://example.com/one".parse().unwrap(),
                title: "One".to_string(),
            },
            text: "first source text".to_string(),
        }];
        let citation = anthropic_completion::Citation::CharLocation {
            cited_text: "first source text".to_string(),
            document_index: 0,
            document_title: Some("Source 1: One".to_string()),
            start_char_index: 0,
            end_char_index: 17,
        };
        let citations = serde_json::to_value(vec![citation.clone(), citation]).unwrap();
        let messages = vec![message::Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::Text(message::Text {
                text: "summary".to_string(),
                additional_params: Some(serde_json::json!({ "citations": citations })),
            })),
        }];

        let evidence = extract_anthropic_source_quotes(&messages, &documents).unwrap();

        assert_eq!(evidence.len(), 1);
    }

    #[tokio::test]
    async fn memoized_invokes_compute_once_and_caches_the_result() {
        let step = test_step("run-1", 3);
        let calls = std::cell::Cell::new(0);
        let make_plan = || Plan {
            sub_questions: vec!["q1".into(), "q2".into()],
        };

        let first: Plan = memoized(&step.memo, MEMO_KEY_PLANNING, async {
            calls.set(calls.get() + 1);
            Ok(make_plan())
        })
        .await
        .unwrap();

        let second: Plan = memoized(&step.memo, MEMO_KEY_PLANNING, async {
            calls.set(calls.get() + 1);
            panic!("compute must not run on a cache hit");
        })
        .await
        .unwrap();

        assert_eq!(calls.get(), 1);
        assert_eq!(first.sub_questions, make_plan().sub_questions);
        assert_eq!(second.sub_questions, make_plan().sub_questions);
    }

    #[tokio::test]
    async fn memoized_rejects_corrupt_cached_bytes_as_permanent() {
        let step = test_step("run-1", 0);
        step.memo.put(MEMO_KEY_PLANNING, b"not json").await.unwrap();

        let err = memoized::<Plan, _>(&step.memo, MEMO_KEY_PLANNING, async {
            panic!("compute must not run when cached bytes exist");
        })
        .await
        .expect_err("corrupt bytes must surface as an error");
        assert_permanent(&err);
    }
}
