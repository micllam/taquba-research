//! [`ResearchStepRunner`]: the [`StepRunner`] that advances a research run
//! through its seven phases.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use rig_agent::agent::{AgentBuilder, CompletionCall, PromptResponse, TypedPromptResponse};
use rig_agent::completion::{PromptError, StructuredOutputError};
use rig_core::DynModel;
use rig_core::completion::{
    FinishReason, Usage,
    message::{self, AssistantContent, UserContent, citation},
};
use rig_core::operation::Completion;
use rig_core::providers::{anthropic, ollama, openai};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use taquba::object_store::path::Path;
use taquba::object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use taquba::{LeaseHandle, Queue};
use taquba_workflow::jobs::{GroupResult, Job, JobRunner, RunnerHandle};
use taquba_workflow::{
    Memo, RunId, RunOptions, Step, StepError, StepErrorKind, StepOutcome, StepRunner,
};
use url::Url;

use crate::host::{Ask, Host, TurnEnd};
use crate::investigate::{self, Investigation, ResearchTools};
use crate::report::{Citation, Report, RunStats, render_markdown};
use crate::search::{SearchBackend, SearchError};
use crate::state::{
    Clarification, Phase, Reply, ResearchConfig, ResearchState, SourceQuote, Summary,
    SynthesisOutput, TokenUsage,
};
use crate::store::{PROGRESS_KEY, RunProgress};
use crate::tool_job::{Artifact, ToolJob, fetch_group_id, search_kind};

/// Preamble applied to every Rig agent built by the runner. The per-phase
/// prompts contain the task-specific instructions.
pub(crate) const AGENT_PREAMBLE: &str = "Be precise and concise.";

/// Upper bound on a single LLM completion call. The step's lease is extended by
/// this much before the call is issued, so the delivery is not re-queued
/// mid-call. Sized for slow local models.
pub(crate) const LLM_CALL_TIMEOUT: Duration = Duration::from_secs(600);
/// Upper bound on a single search-backend call, covered by the lease the same
/// way.
pub(crate) const SEARCH_CALL_TIMEOUT: Duration = Duration::from_secs(60);
/// Upper bound on the canonical report blob write, covered by the lease the
/// same way.
const REPORT_PUT_TIMEOUT: Duration = Duration::from_secs(20);
/// Lease extension applied before a wait for `ToolJob` jobs: one fetch's full
/// retry cycle (3 attempts of 20s each plus backoff) plus the wait for a runner
/// slot while 16 concurrent jobs run, rounded up.
pub(crate) const FETCH_JOIN_LEASE: Duration = Duration::from_secs(150);
/// Upper bound on the wait of a run for a group of `ToolJob` jobs.
const GROUP_WAIT: Duration = Duration::from_secs(15 * 60);

/// Memo user-keys for each phase's cached LLM response. Each is scoped per
/// `(run_id, step_number)` by [`taquba_workflow::Memo`], so a plain string
/// suffices.
const MEMO_KEY_PLANNING: &str = "planning";
const MEMO_KEY_SUMMARIZING: &str = "summarizing";
const MEMO_KEY_SYNTHESIZING: &str = "synthesizing";
const MEMO_KEY_WRITING: &str = "writing";

/// Advances a research run through the plan -> search -> fetch -> summarise ->
/// investigate -> synthesise -> write phases.
///
/// Cloning a `ResearchStepRunner` copies its internal `Arc`s. One instance is
/// shared across all worker tasks of a single [`WorkflowRuntime`].
///
/// [`WorkflowRuntime`]: taquba_workflow::WorkflowRuntime
#[derive(Clone)]
pub struct ResearchStepRunner {
    models: Models,
    search: Arc<dyn SearchBackend>,
    job_runner: Option<JobRunner>,
    report_store: Option<(Arc<dyn ObjectStore>, Path)>,
}

/// The lookup of a completion model by the model identifier of a run.
pub(crate) type Models = Arc<dyn Fn(&str) -> DynModel<Completion> + Send + Sync>;

/// The completion models of an OpenAI client.
pub(crate) fn openai_models(client: openai::OpenAI) -> Models {
    Arc::new(move |model| client.completion(model).into())
}

/// The completion models of an Anthropic client.
pub(crate) fn anthropic_models(client: anthropic::Anthropic) -> Models {
    Arc::new(move |model| client.completion(model).into())
}

/// The native completion models of an Ollama client.
pub(crate) fn ollama_models(client: ollama::Ollama) -> Models {
    Arc::new(move |model| client.native_completion(model).into())
}

/// The result of a succeeded run: the final report and the metadata to render
/// it. The runner builds a `RunRecord` only on success, and a failure reason is
/// stored in [`taquba_workflow::RunTermination::error`].
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
    /// Build a runner from a lookup of completion models and a search backend.
    /// Each step calls `models` with [`ResearchConfig::model`] of its run.
    pub fn new(
        models: impl Fn(&str) -> DynModel<Completion> + Send + Sync + 'static,
        search: Arc<dyn SearchBackend>,
    ) -> Self {
        Self::from_models(Arc::new(models), search)
    }

    /// Build a runner from a Rig OpenAI client and a search backend.
    pub fn new_openai(client: openai::OpenAI, search: Arc<dyn SearchBackend>) -> Self {
        Self::from_models(openai_models(client), search)
    }

    /// Build a runner from a Rig Anthropic client and a search backend.
    pub fn new_anthropic(client: anthropic::Anthropic, search: Arc<dyn SearchBackend>) -> Self {
        Self::from_models(anthropic_models(client), search)
    }

    /// Build a runner from a Rig Ollama client and a search backend, for local
    /// models.
    pub fn new_ollama(client: ollama::Ollama, search: Arc<dyn SearchBackend>) -> Self {
        Self::from_models(ollama_models(client), search)
    }

    pub(crate) fn from_models(models: Models, search: Arc<dyn SearchBackend>) -> Self {
        Self {
            models,
            search,
            job_runner: None,
            report_store: None,
        }
    }

    /// Build the [`JobRunner`] of the fetches of the fetching phase and of the
    /// tool calls of the investigating phase, with the search backend of this
    /// runner, and spawn its worker. Returns the `JobRunner` for
    /// [`Self::with_job_runner`] and a [`RunnerHandle`] for graceful shutdown.
    ///
    /// The job runner shares `queue` and `object_store` with the surrounding
    /// workflow runtime, on the queue [`crate::TOOL_QUEUE_NAME`]. It must shut
    /// down after the workflow worker.
    pub fn spawn_tool_runner(
        &self,
        queue: &Arc<Queue>,
        object_store: &Arc<dyn ObjectStore>,
    ) -> (JobRunner, RunnerHandle) {
        crate::tool_job::spawn_tool_runner(queue, object_store, self.search.clone())
    }

    /// Attach the [`JobRunner`] that [`Self::spawn_tool_runner`] builds.
    /// Required when the workflow advances into `Phase::Fetching` or
    /// `Phase::Investigating`: each phase submits its jobs as a group and waits
    /// for the group.
    pub fn with_job_runner(mut self, job_runner: JobRunner) -> Self {
        self.job_runner = Some(job_runner);
        self
    }

    /// Attach the object store and key prefix under which the Writing step
    /// writes the canonical report blob (`<prefix>/reports/<run_id>.md`),
    /// before the step settles. When unset, no canonical report blob is
    /// written.
    pub fn with_report_store(mut self, object_store: Arc<dyn ObjectStore>, prefix: &Path) -> Self {
        self.report_store = Some((object_store, prefix.clone()));
        self
    }

    /// Encode the initial state for a research run, ready to pass to
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

        // The token fires when the runtime cancels the run, so a slow LLM or
        // HTTP call is dropped at once. A redelivery of a cancelled step starts
        // with the token cancelled and skips the phase work.
        let outcome = tokio::select! {
            biased;
            () = step.cancel_token.cancelled() => StepOutcome::Cancel {
                reason: "Cancelled by user".to_string(),
            },
            res = self.dispatch_phase(step, &mut state) => res?,
        };
        stage_progress(step, &state)?;
        Ok(outcome)
    }
}

/// The reply that the signal of the step after a wait contains. A step without
/// a signal follows the end of the wait, so the agent proceeds on an
/// assumption.
fn resolve_reply(signal: Option<&[u8]>) -> Reply {
    match signal {
        Some(payload) => Reply::Given(String::from_utf8_lossy(payload).into_owned()),
        None => Reply::Assumed,
    }
}

/// Stage the progress of `state` on the step's effects handle, so it commits
/// with the settlement of the step.
fn stage_progress(step: &Step, state: &ResearchState) -> Result<(), StepError> {
    let progress = RunProgress {
        steps_completed: state.steps_completed,
        token_usage: state.token_usage,
    };
    step.effects
        .put_run_kv(PROGRESS_KEY, progress.to_bytes())
        .map_err(|e| StepError::permanent(format!("staging run progress: {e}")))
}

/// How a step of the investigating phase waits.
enum Wait {
    /// For the reply of the user, at most this long.
    Reply(Duration),
    /// For the tool calls of the reply of the step, as the members of this
    /// group.
    Group(RunId),
}

#[derive(Debug, Clone)]
struct SynthesisDocument {
    /// The numbered source this document backs.
    citation: Citation,
    /// Full fetched body fed to the model for synthesis.
    text: String,
}

impl ResearchStepRunner {
    /// Runs the phase indicated by `state.phase` and returns the resulting
    /// `StepOutcome`. Separated from `run_step` so it can be raced against the
    /// cancellation token of the step via `tokio::select!`.
    async fn dispatch_phase(
        &self,
        step: &Step,
        state: &mut ResearchState,
    ) -> Result<StepOutcome, StepError> {
        match state.phase {
            Phase::Planning => self.run_planning(step, state).await?,
            Phase::Searching => self.run_searching(step, state).await?,
            Phase::Fetching => {
                // A wait is not a completed step.
                if let Some(group_id) = self.run_fetching(step, state).await? {
                    return Ok(StepOutcome::continue_on_group(
                        state.to_bytes(),
                        group_id,
                        GROUP_WAIT,
                    ));
                }
            }
            Phase::Summarizing => self.run_summarizing(step, state).await?,
            Phase::Investigating => match self.run_investigating(step, state).await? {
                None => {}
                // A wait for the user is not a completed step.
                Some(Wait::Reply(wait)) => {
                    return Ok(StepOutcome::continue_on_signal(state.to_bytes(), wait));
                }
                // A wait for the tool calls follows the model call of the step.
                Some(Wait::Group(group_id)) => {
                    state.steps_completed += 1;
                    return Ok(StepOutcome::continue_on_group(
                        state.to_bytes(),
                        group_id,
                        GROUP_WAIT,
                    ));
                }
            },
            Phase::Synthesizing => self.run_synthesizing(step, state).await?,
            Phase::Writing => {
                let report = self.run_writing(step, state).await?;
                // The blob is written before the step settles, so a terminal
                // record saying succeeded implies the report exists. A
                // redelivered step rewrites the same memoized bytes.
                self.put_report(step, &report).await?;
                state.steps_completed = report.stats.steps_completed;
                let record = RunRecord {
                    report: Some(report),
                    run_id: step.run_id.to_string(),
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

    /// Write the canonical report blob at `<prefix>/reports/<run_id>.md`. A
    /// no-op without a report store. A failed write is a transient step error,
    /// so the Writing step never settles without the blob.
    async fn put_report(&self, step: &Step, report: &Report) -> Result<(), StepError> {
        let Some((object_store, prefix)) = &self.report_store else {
            return Ok(());
        };
        let path = crate::store::report_path(prefix, &step.run_id);
        under_lease(&step.lease, REPORT_PUT_TIMEOUT, "report write", async {
            object_store
                .put(&path, PutPayload::from(report.markdown.as_bytes().to_vec()))
                .await
                .map(|_| ())
                .map_err(|e| StepError::transient(format!("writing report blob: {e}")))
        })
        .await
    }

    async fn run_planning(&self, step: &Step, state: &mut ResearchState) -> Result<(), StepError> {
        tracing::info!("planning sub-questions");
        let prompt = format!(
            "You are a research planner. Decompose the user's query into at most {} \
             distinct, web-searchable sub-questions that together cover the topic.\n\n\
             Query: {}",
            state.config.depth, state.query
        );

        // Memoize the LLM call so an at-least-once retry of this step (lease
        // expiry, worker restart) reuses the prior attempt's response in place
        // of a second request for the same prompt.
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
            // Unreachable in normal flow: the phase leaves Searching as soon as
            // the queue empties.
            state.phase = Phase::Fetching;
            return Ok(());
        };
        let q = state.sub_questions[idx].clone();
        let total = state.sub_questions.len();
        let done = total - state.search_queue.len(); // already-popped count
        tracing::info!("searching ({done}/{total}): {q}");

        // 5 results per sub-question. `max_sources` caps the total.
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
                // No sources to fetch. The investigating step searches again.
                Phase::Investigating
            } else {
                Phase::Fetching
            };
        }
        Ok(())
    }

    /// Submit the fetch group of the run and return its id for the wait, or, in
    /// the step after the wait, record the results of the group.
    async fn run_fetching(
        &self,
        step: &Step,
        state: &mut ResearchState,
    ) -> Result<Option<RunId>, StepError> {
        if state.fetch_queue.is_empty() {
            state.phase = Phase::Summarizing;
            return Ok(None);
        }
        let job_runner = self
            .job_runner
            .as_ref()
            .ok_or_else(|| StepError::permanent("fetching phase requires a JobRunner"))?;
        let group = job_runner.group::<ToolJob>(fetch_group_id(&step.run_id)?);
        let jobs: Vec<ToolJob> = state
            .fetch_queue
            .iter()
            .map(|url| ToolJob::Fetch { url: url.clone() })
            .collect();

        // A retry of this step submits the same members again, which runs only
        // the members without a success.
        if !state.fetch_submitted {
            tracing::info!("fetching {} URLs in parallel", jobs.len());
            group.submit_with(jobs, owned_by(step)).await?;
            state.fetch_submitted = true;
            return Ok(Some(group.id().clone()));
        }

        // The wait ends at the timeout with members still active.
        let status = group.status().await?;
        if status.pending > 0 {
            tracing::warn!(pending = status.pending, "fetch wait elapsed, cancelling");
            group
                .cancel(Some("the fetch wait elapsed".to_string()))
                .await?;
        }
        let results = under_lease(&step.lease, FETCH_JOIN_LEASE, "fetch join", async {
            Ok(group.join().await?)
        })
        .await?;
        let urls: HashMap<String, Url> = state
            .fetch_queue
            .iter()
            .filter_map(|url| {
                let key = ToolJob::Fetch { url: url.clone() }.idempotency_key()?;
                Some((key, url.clone()))
            })
            .collect();
        for GroupResult { key, result } in results {
            let Some(url) = urls.get(&key) else {
                continue;
            };
            match result {
                Ok(Artifact::Page(page)) => {
                    state.summarize_queue.push_back(url.clone());
                    state
                        .fetched
                        .insert(url.clone(), page.first_chunk(state.config.max_page_chars));
                }
                Ok(Artifact::Results(_)) => {}
                Err(e) => tracing::warn!(url = %url, error = %e, "fetch failed, skipping page"),
            }
        }

        state.fetch_queue.clear();
        state.fetch_submitted = false;
        state.phase = Phase::Summarizing;
        Ok(None)
    }

    async fn run_summarizing(
        &self,
        step: &Step,
        state: &mut ResearchState,
    ) -> Result<(), StepError> {
        let Some(url) = state.summarize_queue.pop_front() else {
            state.phase = Phase::Investigating;
            return Ok(());
        };
        let Some(page) = state.fetched.get(&url).cloned() else {
            // Unreachable in normal flow: `summarize_queue` receives a URL only
            // after a successful fetch.
            if state.summarize_queue.is_empty() {
                state.phase = Phase::Investigating;
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
            state.phase = Phase::Investigating;
        }
        Ok(())
    }

    async fn run_investigating(
        &self,
        step: &Step,
        state: &mut ResearchState,
    ) -> Result<Option<Wait>, StepError> {
        let turns = state.config.investigation_turns;
        if turns == 0 {
            state.phase = Phase::Synthesizing;
            return Ok(None);
        }
        let jobs = self
            .job_runner
            .as_ref()
            .ok_or_else(|| StepError::permanent("investigating phase requires a JobRunner"))?;
        let run_id = &step.run_id;
        let max_chars = state.config.max_page_chars;

        let ask = match (
            state.config.clarification_wait_secs,
            &mut state.clarification,
        ) {
            (None, _) => Ask::Off,
            (Some(_), None) => Ask::Open,
            (Some(_), Some(clarification)) => {
                let reply = clarification
                    .reply
                    .get_or_insert_with(|| resolve_reply(step.signal.as_deref()));
                Ask::Replied(investigate::reply_text(reply))
            }
        };
        tracing::info!("investigating gaps in {} sources", state.summaries.len());

        let can_ask = !matches!(ask, Ask::Off);
        let tools = ResearchTools {
            jobs: jobs.clone(),
            run_id: run_id.clone(),
            fetching: state.fetched.keys().cloned().collect(),
            chunk_chars: max_chars,
        };
        let mut definitions = investigate::definitions(max_chars);
        if can_ask {
            definitions.push(Arc::new(investigate::ask_user()));
        }
        let spec = investigate::spec(turns, state.config.max_tokens_per_call);
        let host = Host {
            model: (self.models)(&state.config.model),
            spec: &spec,
            tools: &tools,
            definitions,
            ask,
            jobs,
            group_options: owned_by(step),
            run_id,
            memo: &step.memo,
            lease: &step.lease,
        };
        let prompt = investigate::prompt(&state.query, &state.summaries, turns, can_ask);
        let agent = state.investigation.get_or_insert_default();
        let findings = match host.step(agent, &prompt, &mut state.token_usage).await? {
            TurnEnd::Continue => return Ok(None),
            TurnEnd::Group { group_id } => return Ok(Some(Wait::Group(group_id))),
            TurnEnd::Wait { question } => {
                tracing::info!(
                    %question,
                    "waiting for a reply: taquba-research reply {run_id} \"<text>\""
                );
                state.clarification = Some(Clarification {
                    question,
                    asked_at: Utc::now(),
                    reply: None,
                });
                let wait = state.config.clarification_wait_secs.unwrap_or_default();
                return Ok(Some(Wait::Reply(Duration::from_secs(wait))));
            }
            TurnEnd::Done { output } => {
                serde_json::from_str::<Investigation>(&output)
                    .map_err(|e| StepError::permanent(format!("investigation output: {e}")))?
                    .findings
            }
            TurnEnd::OutOfTurns => {
                tracing::warn!(turns, "investigation ran out of model calls");
                Vec::new()
            }
        };
        let transcript = state
            .investigation
            .take()
            .map(|agent| agent.transcript)
            .unwrap_or_default();
        let fetched = investigate::fetched(&transcript);

        // A finding enters the run only for a page that `fetch_page` returned.
        // The page text comes from the recorded result of its member.
        let mut added = 0;
        for finding in findings {
            let Ok(url) = Url::parse(&finding.url) else {
                continue;
            };
            if !fetched.contains_key(&url) || state.summaries.contains_key(&url) {
                continue;
            }
            let Some(page) = tools.page(&url, &transcript).await? else {
                tracing::warn!(url = %url, "page of a finding not found, skipping finding");
                continue;
            };
            state.summaries.insert(
                url.clone(),
                Summary {
                    title: page.title.clone(),
                    text: finding.summary,
                    relevance: finding.relevance.clamp(0.0, 1.0),
                },
            );
            state.fetched.insert(url, page.first_chunk(max_chars));
            added += 1;
        }
        tracing::info!("investigation added {added} sources");
        state.phase = Phase::Synthesizing;
        Ok(None)
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
        // The 1-based source number is the position in this sorted list and is
        // shared by the prompt text, the citation list and any cited excerpts.
        // Building all three here keeps that numbering consistent.
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

        let output: PromptOutput = memoized(&step.memo, MEMO_KEY_WRITING, async {
            self.llm_prompt(&step.lease, &prompt, state).await
        })
        .await?;
        if output.truncated {
            tracing::warn!("report body truncated at the output-token limit");
        }

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
            output_truncated: output.truncated,
        };

        let markdown =
            render_markdown(&state.query, &step.run_id, &output.text, &synthesis, &stats);
        Ok(Report {
            run_id: step.run_id.to_string(),
            query: state.query.clone(),
            markdown,
            citations: synthesis.citations,
            stats,
        })
    }

    /// Run a single completion via Rig, dispatched to the configured provider,
    /// with the delivery lease extended to cover the call's
    /// [`LLM_CALL_TIMEOUT`] bound. Records this call's token usage on
    /// `state.token_usage` and logs the per-call counts at info level.
    async fn llm_prompt(
        &self,
        lease: &LeaseHandle,
        prompt: &str,
        state: &mut ResearchState,
    ) -> Result<PromptOutput, StepError> {
        let model = (self.models)(&state.config.model);
        let max_tokens = state.config.max_tokens_per_call;
        let response = under_lease(
            lease,
            LLM_CALL_TIMEOUT,
            "LLM call",
            prompt_extended(model, max_tokens, prompt),
        )
        .await?;
        record_usage(&mut state.token_usage, &response.usage);
        Ok(PromptOutput {
            truncated: length_truncated(&response.completion_calls),
            text: response.output(),
        })
    }

    async fn llm_synthesis_prompt(
        &self,
        lease: &LeaseHandle,
        prompt: &str,
        source_documents: &[SynthesisDocument],
        state: &mut ResearchState,
    ) -> Result<(String, Vec<SourceQuote>), StepError> {
        let model = (self.models)(&state.config.model);
        match model.name() {
            anthropic::PROVIDER_NAME if !source_documents.is_empty() => {
                let message = anthropic_document_message(prompt, source_documents);
                let response = under_lease(
                    lease,
                    LLM_CALL_TIMEOUT,
                    "LLM call",
                    prompt_extended(model, state.config.max_tokens_per_call, message),
                )
                .await?;
                record_usage(&mut state.token_usage, &response.usage);
                let evidence = extract_anthropic_source_quotes(&response.content, source_documents);
                Ok((response.output(), evidence))
            }
            _ => Ok((
                self.llm_prompt(lease, prompt, state).await?.text,
                Vec::new(),
            )),
        }
    }

    /// Run a structured completion via Rig's `prompt_typed`, dispatched to the
    /// configured provider. Same lease and usage-tracking behaviour as
    /// [`Self::llm_prompt`].
    async fn llm_prompt_typed<T>(
        &self,
        lease: &LeaseHandle,
        prompt: &str,
        state: &mut ResearchState,
    ) -> Result<T, StepError>
    where
        T: JsonSchema + DeserializeOwned + Send + 'static,
    {
        let model = (self.models)(&state.config.model);
        let max_tokens = state.config.max_tokens_per_call;
        let response = under_lease(
            lease,
            LLM_CALL_TIMEOUT,
            "LLM call",
            prompt_typed_extended::<T>(model, max_tokens, prompt),
        )
        .await?;
        record_usage(&mut state.token_usage, &response.usage);
        Ok(response.output)
    }
}

/// A Rig agent on `model` with the shared preamble and per-call token cap.
fn build_agent(model: DynModel<Completion>, max_tokens: Option<u64>) -> rig_agent::Agent {
    let mut builder = AgentBuilder::new(model).preamble(AGENT_PREAMBLE);
    if let Some(max_tokens) = max_tokens {
        builder = builder.max_tokens(max_tokens);
    }
    builder.build()
}

/// Run `prompt` on a Rig agent for `model`.
async fn prompt_extended(
    model: DynModel<Completion>,
    max_tokens: Option<u64>,
    prompt: impl Into<message::Message>,
) -> Result<PromptResponse, StepError> {
    build_agent(model, max_tokens)
        .prompt(prompt)
        .await
        .map_err(classify_rig_err)
}

/// Structured counterpart to [`prompt_extended`], running Rig's `prompt_typed`
/// for the schema `T`.
async fn prompt_typed_extended<T>(
    model: DynModel<Completion>,
    max_tokens: Option<u64>,
    prompt: &str,
) -> Result<TypedPromptResponse<T>, StepError>
where
    T: JsonSchema + DeserializeOwned + Send + 'static,
{
    build_agent(model, max_tokens)
        .prompt_typed::<T>(prompt)
        .await
        .map_err(classify_structured_err)
}

/// Accumulate one call's `Usage` into the run-aggregate `TokenUsage`, logging
/// the per-call counts at info level.
pub(crate) fn record_usage(total: &mut TokenUsage, call: &Usage) {
    tracing::info!(
        input = call.input_tokens,
        output = call.output_tokens,
        total = call.total_tokens,
        cached_input = call.cached_input_tokens,
        reasoning = call.reasoning_tokens,
        "LLM call usage",
    );
    let add = |sum: &mut u64, counter: Option<u64>| {
        *sum = sum.saturating_add(counter.unwrap_or(0));
    };
    add(&mut total.input_tokens, call.input_tokens);
    add(&mut total.output_tokens, call.output_tokens);
    add(&mut total.total_tokens, call.total_tokens);
    add(&mut total.cached_input_tokens, call.cached_input_tokens);
    add(
        &mut total.cache_creation_input_tokens,
        call.cache_creation_input_tokens,
    );
    add(
        &mut total.tool_use_prompt_tokens,
        call.tool_use_prompt_tokens,
    );
    add(&mut total.reasoning_tokens, call.reasoning_tokens);
}

/// The options of a group that `step` submits: the run of `step` owns its
/// members, so the termination of the run cancels them.
fn owned_by(step: &Step) -> RunOptions {
    RunOptions {
        owner: Some(step.owner.clone()),
        ..RunOptions::default()
    }
}

/// Extend `lease` to cover `bound`, then run `work` under a timeout of the same
/// bound, so the delivery cannot outlive its lease mid-call. A timeout is a
/// transient step error, and `what` identifies the call in the error message.
pub(crate) async fn under_lease<T, F>(
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

/// Map a failed lease extension to a transient step error. The extension fails
/// only when the claim was lost or the job's cancellation was requested. In
/// both cases this delivery's settlement is rejected or its run terminated, so
/// a retry is safe.
pub(crate) fn lease_step_err(e: taquba::Error) -> StepError {
    StepError::transient(format!("lease extension failed: {e}"))
}
/// Returns the JSON-decoded value previously written to `memo` under `key`. If
/// none exists, awaits `compute`, JSON-encodes its result into the memo, and
/// returns it. An at-least-once retry of the surrounding step then finds the
/// cached value and skips the compute call entirely.
pub(crate) async fn memoized<T, F>(memo: &Memo, key: &str, compute: F) -> Result<T, StepError>
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

/// Classify a Rig prompt error as transient or permanent. A provider failure is
/// transient when Rig reports it as retryable.
pub(crate) fn classify_rig_err(err: PromptError) -> StepError {
    let retryable = match &err {
        PromptError::Provider(error) => error.is_retryable(),
        PromptError::Report(report) => report.is_retryable(),
        PromptError::MaxTurns { .. }
        | PromptError::Cancelled { .. }
        | PromptError::UnknownToolCall { .. } => false,
        // `PromptError::Memory` and a future variant.
        _ => true,
    };
    if retryable {
        StepError::transient(format!("LLM call failed: {err}"))
    } else {
        StepError::permanent(format!("LLM permanent failure: {err}"))
    }
}

impl From<SearchError> for StepError {
    fn from(e: SearchError) -> Self {
        match search_kind(&e) {
            StepErrorKind::Transient => StepError::transient(e.to_string()),
            StepErrorKind::Permanent => StepError::permanent(e.to_string()),
        }
    }
}

/// Planning step response schema.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Plan {
    sub_questions: Vec<String>,
}

/// `Summarizing` step response schema.
#[derive(Serialize, Deserialize, JsonSchema)]
struct SummaryResp {
    summary: String,
    relevance: f32,
}

/// A plain completion's text plus whether its final completion call stopped at
/// the output-token limit. The Writing step memoizes the whole value, so the
/// truncation flag is preserved across a step retry.
#[derive(Serialize, Deserialize)]
struct PromptOutput {
    text: String,
    truncated: bool,
}

/// Whether the final completion call of an agent run stopped generating at the
/// output-token limit.
fn length_truncated(calls: &[CompletionCall]) -> bool {
    calls
        .last()
        .is_some_and(|call| matches!(call.finish_reason, Some(FinishReason::Length)))
}

fn anthropic_document_message(
    prompt: &str,
    source_documents: &[SynthesisDocument],
) -> message::Message {
    let mut content = Vec::with_capacity(source_documents.len() + 1);
    content.push(UserContent::text(format!(
        "{prompt}\n\nUse the attached source documents as the authoritative \
         citation corpus. The documents are in the same order as the numbered \
         sources above. Cite relevant claims with the source numbers already \
         assigned above."
    )));

    for document in source_documents {
        content.push(UserContent::Document(message::Document {
            data: message::DocumentData::Text(document.text.clone()),
            media_type: Some(message::DocumentMediaType::TXT),
            additional_params: Some(serde_json::json!({
                "title": format!(
                    "Source {}: {}",
                    document.citation.index, document.citation.title
                ),
                "context": format!("URL: {}", document.citation.url),
                "citations": { "enabled": true },
            })),
        }));
    }

    message::Message::from(content)
}

fn extract_anthropic_source_quotes(
    content: &[AssistantContent],
    source_documents: &[SynthesisDocument],
) -> Vec<SourceQuote> {
    let mut evidence = Vec::new();
    let mut seen = HashSet::new();

    for block in content {
        let AssistantContent::Text(text) = block else {
            continue;
        };
        for source in text.citations().iter().flat_map(|c| &c.sources) {
            let Some((document_index, cited_text)) = cited_document_span(source) else {
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

    evidence
}

/// The request position of the document that `source` cites, and the passage it
/// quotes.
fn cited_document_span(source: &citation::Source) -> Option<(usize, &str)> {
    let citation::SourceLocation::Document {
        index: Some(index), ..
    } = &source.location
    else {
        return None;
    };
    Some((usize::try_from(*index).ok()?, source.cited_text.as_deref()?))
}

/// Classify a typed-prompt error. Delegates the wrapped `PromptError` case to
/// [`classify_rig_err`]. A deserialization error is permanent and an empty
/// response is transient.
pub(crate) fn classify_structured_err(err: StructuredOutputError) -> StepError {
    match err {
        StructuredOutputError::Prompt(inner) => classify_rig_err(inner),
        StructuredOutputError::Deserialization { error, .. } => {
            StepError::permanent(format!("typed prompt: schema deserialize failed: {error}"))
        }
        // An empty response is typically a one-off provider failure.
        StructuredOutputError::EmptyResponse => {
            StepError::transient("typed prompt: model returned an empty response".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::{Plan as CallPlan, Record, RecordedCall, TurnTools};
    use reqwest::StatusCode;
    use rig_core::http_client;
    use rig_core::test_utils::MockCompletionModel;
    use rig_core::{ErrorKind, ErrorReport, ProviderError, ProviderResponseError};
    use taquba::Queue;
    use taquba::object_store::memory::InMemory;
    use taquba::object_store::path::Path;
    use taquba_workflow::MemoStore;

    /// Build a `Step` with a fresh in-memory `Memo` and otherwise inert fields,
    /// suitable for exercising memo-using helpers.
    fn test_step(run_id: &str, step_number: u32) -> Step {
        let run_id = RunId::new(run_id).unwrap();
        let mut step = Step::detached(Vec::new());
        step.memo =
            MemoStore::new(Arc::new(InMemory::new()), "test-memo").new_memo(&run_id, step_number);
        step.run_id = run_id;
        step.step_number = step_number;
        step
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

    fn http_status(code: u16) -> ProviderError {
        ProviderError::from_transport_error(http_client::Error::non_success_with_details(
            StatusCode::from_u16(code).unwrap(),
            Default::default(),
            String::new(),
        ))
    }

    #[tokio::test]
    async fn a_cancelled_step_returns_cancel_without_phase_work() {
        let runner = ResearchStepRunner::new_ollama(
            ollama::Ollama::from_env().unwrap(),
            Arc::new(crate::search::Tavily::new("unused")),
        );
        let mut step = test_step("run-1", 0);
        step.payload = ResearchState::new("q", ResearchConfig::new("m")).to_bytes();
        step.cancel_token.cancel();

        // The planning phase would call the model, and the runner does not
        // reach it.
        let outcome = runner.run_step(&step).await.unwrap();
        assert!(matches!(outcome, StepOutcome::Cancel { .. }));
    }

    #[tokio::test]
    async fn a_step_looks_up_the_model_of_its_run() {
        let plan = r#"{"sub_questions":["a sub-question"]}"#;
        let model = MockCompletionModel::text(plan);
        let looked_up = Arc::new(std::sync::Mutex::new(Vec::new()));
        let models = {
            let looked_up = looked_up.clone();
            move |id: &str| {
                looked_up.lock().unwrap().push(id.to_string());
                model.clone().into()
            }
        };
        let runner =
            ResearchStepRunner::new(models, Arc::new(crate::search::Tavily::new("unused")));
        let mut step = test_step("run-1", 0);
        step.payload = ResearchState::new("q", ResearchConfig::new("run-model")).to_bytes();

        runner.run_step(&step).await.unwrap();
        assert_eq!(*looked_up.lock().unwrap(), ["run-model"]);
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
            StructuredOutputError::Deserialization {
                output: "nope".to_string(),
                error: json_err,
            },
        ));
    }

    #[test]
    fn classify_rig_err_follows_the_retry_verdict_of_rig() {
        let failures = [
            http_status(401),
            http_status(408),
            http_status(429),
            ProviderError::ProviderResponse(ProviderResponseError::new(
                StatusCode::OK,
                "{\"error\":{\"type\":\"overloaded_error\"}}",
            )),
            ProviderError::ProviderResponse(ProviderResponseError::new(
                StatusCode::OK,
                "{\"error\":{\"message\":\"no code\"}}",
            )),
            ProviderError::Truncated,
        ];
        for failure in failures {
            let retryable = failure.is_retryable();
            let err = classify_rig_err(PromptError::Provider(failure));
            assert_eq!(
                matches!(err.kind, StepErrorKind::Transient),
                retryable,
                "{}",
                err.message
            );
        }
        for retryable in [true, false] {
            let report = ErrorReport::new(ErrorKind::Http, "relayed").with_retryable(retryable);
            let err = classify_rig_err(PromptError::Report(report));
            assert_eq!(
                matches!(err.kind, StepErrorKind::Transient),
                retryable,
                "{}",
                err.message
            );
        }
    }

    #[test]
    fn classify_rig_err_unknown_tool_call_is_permanent() {
        let err = PromptError::UnknownToolCall {
            tool_name: "lookup".to_string(),
            available_tools: Vec::new(),
            allowed_tools: Vec::new(),
            chat_history: Vec::new(),
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
    fn length_truncated_reads_the_final_completion_call() {
        assert!(!length_truncated(&[]));

        let mut cut = CompletionCall::new(0, Usage::default(), serde_json::Value::Null);
        cut.finish_reason = Some(FinishReason::Length);
        assert!(length_truncated(&[cut.clone()]));

        let complete = CompletionCall::new(1, Usage::default(), serde_json::Value::Null);
        assert!(!length_truncated(&[cut, complete]));
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
        let content = vec![AssistantContent::Text(
            message::Text::new("summary")
                .with_citations([document_citation(1, "second source text")]),
        )];

        let evidence = extract_anthropic_source_quotes(&content, &documents);

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
        let citation = document_citation(0, "first source text");
        let content = vec![AssistantContent::Text(
            message::Text::new("summary").with_citations([citation.clone(), citation]),
        )];

        let evidence = extract_anthropic_source_quotes(&content, &documents);

        assert_eq!(evidence.len(), 1);
    }

    /// A citation of the request document at `index` that quotes `cited_text`.
    fn document_citation(index: u32, cited_text: &str) -> citation::Citation {
        citation::Citation::new([citation::Source::new(citation::SourceLocation::Document {
            index: Some(index),
            id: None,
            within: None,
        })
        .cited_text(cited_text)])
    }

    struct NoSearch;

    #[async_trait::async_trait]
    impl SearchBackend for NoSearch {
        async fn search(
            &self,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<crate::search::SearchResult>, SearchError> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn the_writing_step_commits_its_progress_and_writes_the_report() {
        use crate::store::run_progress;
        use taquba_workflow::{RunSpec, TerminalStatus, WorkflowRuntime};

        let object_store: Arc<dyn taquba::object_store::ObjectStore> = Arc::new(InMemory::new());
        let queue = Arc::new(Queue::open(object_store.clone(), "q").await.unwrap());
        let run_id = RunId::new("01RUNIDX").unwrap();

        // Pre-populate the writing memo so the terminal step completes without
        // an LLM call.
        MemoStore::new(object_store.clone(), "test-memo")
            .new_memo(&run_id, 0)
            .put(
                MEMO_KEY_WRITING,
                &serde_json::to_vec(&PromptOutput {
                    text: "report body".to_string(),
                    truncated: false,
                })
                .unwrap(),
            )
            .await
            .unwrap();

        let mut state = ResearchState::new("a query", ResearchConfig::new("m"));
        state.phase = Phase::Writing;

        let runner = ResearchStepRunner::new_ollama(
            rig_core::providers::ollama::Ollama::from_env().unwrap(),
            Arc::new(NoSearch),
        )
        .with_report_store(object_store.clone(), &Path::default());
        let runtime = WorkflowRuntime::builder(queue.clone(), object_store.clone(), runner)
            .queue_name("test-workflow")
            .memo_prefix("test-memo")
            .max_concurrent_steps(1)
            .build()
            .unwrap();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let worker_runtime = runtime.clone();
        let worker = tokio::spawn(async move {
            worker_runtime
                .run(async move {
                    let _ = shutdown_rx.await;
                })
                .await
        });

        runtime
            .submit(RunSpec {
                run_id: Some(run_id.clone()),
                input: state.to_bytes(),
                ..Default::default()
            })
            .await
            .unwrap();

        let outcome = runtime.wait(&run_id).await.unwrap();
        assert_eq!(outcome.termination.status, TerminalStatus::Succeeded);
        let _ = shutdown_tx.send(());
        let _ = worker.await;

        let progress = run_progress(runtime.view(), &run_id)
            .await
            .unwrap()
            .expect("the Succeed settlement commits the progress");
        assert_eq!(progress.steps_completed, 1);

        // The Writing step wrote the canonical report blob before it settled.
        let blob = object_store
            .get(&crate::store::report_path(&Path::default(), &run_id))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert!(!blob.is_empty());
    }

    /// A local HTTP server that returns an HTML page at `/page` and 404 at
    /// every other path, with the count of requests for `/page`.
    async fn page_server() -> (Url, Arc<std::sync::atomic::AtomicUsize>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = hits.clone();
        tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let mut buf = [0u8; 1024];
                while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                    let n = socket.read(&mut buf).await.unwrap();
                    if n == 0 {
                        break;
                    }
                    request.extend_from_slice(&buf[..n]);
                }
                let response = if request.starts_with(b"GET /page ") {
                    counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let body = "<html><head><title>Page</title></head>\
                                <body><p>The whole text of the page.</p></body></html>";
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/html\r\n\
                         content-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    )
                } else {
                    "HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                        .to_string()
                };
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        (base, hits)
    }

    /// Runs the research runner while the run is in the fetching phase, and
    /// succeeds with the state as the result once the run leaves it.
    struct UntilFetched(ResearchStepRunner);

    impl StepRunner for UntilFetched {
        async fn run_step(&self, step: &Step) -> Result<StepOutcome, StepError> {
            let state = ResearchState::from_bytes(&step.payload).unwrap();
            if state.phase != Phase::Fetching {
                return Ok(StepOutcome::Succeed {
                    result: step.payload.clone(),
                });
            }
            self.0.run_step(step).await
        }
    }

    /// Run the fetching phase of a run over `urls` through a workflow runtime
    /// and the fetch runner, and return the fetch runner, the run id and the
    /// state after the phase.
    async fn run_fetching_phase(urls: Vec<Url>) -> (JobRunner, RunId, ResearchState) {
        use taquba_workflow::{RunSpec, TerminalStatus, WorkflowRuntime};

        let object_store: Arc<dyn taquba::object_store::ObjectStore> = Arc::new(InMemory::new());
        let queue = Arc::new(Queue::open(object_store.clone(), "q").await.unwrap());
        let run_id = RunId::new("01RUNIDX").unwrap();
        let mut state = ResearchState::new("a query", ResearchConfig::new("m"));
        state.phase = Phase::Fetching;
        state.fetch_queue = urls.into();

        let runner = ResearchStepRunner::new_ollama(
            rig_core::providers::ollama::Ollama::from_env().unwrap(),
            Arc::new(NoSearch),
        );
        let (jobs, jobs_handle) = runner.spawn_tool_runner(&queue, &object_store);
        let runner = runner.with_job_runner(jobs.clone());
        let runtime =
            WorkflowRuntime::builder(queue.clone(), object_store.clone(), UntilFetched(runner))
                .queue_name("test-workflow")
                .memo_prefix("test-memo")
                .build()
                .unwrap();
        let worker = runtime.spawn(std::future::pending::<()>());
        runtime
            .submit(RunSpec {
                run_id: Some(run_id.clone()),
                input: state.to_bytes(),
                ..Default::default()
            })
            .await
            .unwrap();

        let outcome = tokio::time::timeout(Duration::from_secs(30), runtime.wait(&run_id))
            .await
            .expect("the run terminates in time")
            .unwrap();
        assert_eq!(outcome.termination.status, TerminalStatus::Succeeded);
        worker.shutdown().await.unwrap();
        jobs_handle.shutdown().await.unwrap();
        let state = ResearchState::from_bytes(&outcome.result.unwrap()).unwrap();
        (jobs, run_id, state)
    }

    #[tokio::test]
    async fn the_fetching_step_waits_for_its_group_and_keeps_the_fetched_pages() {
        let (base, _) = page_server().await;
        let page = base.join("page").unwrap();
        let missing = base.join("missing").unwrap();

        let (_, _, state) = run_fetching_phase(vec![page.clone(), missing.clone()]).await;

        assert_eq!(state.phase, Phase::Summarizing);
        assert!(state.fetch_queue.is_empty());
        assert!(!state.fetch_submitted);
        assert_eq!(state.summarize_queue, std::slice::from_ref(&page));
        assert_eq!(state.fetched[&page].title, "Page");
        assert!(!state.fetched.contains_key(&missing));
    }

    #[tokio::test]
    async fn a_cancel_of_a_run_during_the_fetch_wait_cancels_its_fetches() {
        use taquba_workflow::{RunSpec, RunState, TerminalStatus, WorkflowRuntime};

        // A server that accepts each request and never replies.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = Url::parse(&format!("http://{}/page", listener.local_addr().unwrap())).unwrap();
        tokio::spawn(async move {
            let mut sockets = Vec::new();
            loop {
                sockets.push(listener.accept().await.unwrap());
            }
        });

        let object_store: Arc<dyn taquba::object_store::ObjectStore> = Arc::new(InMemory::new());
        let queue = Arc::new(Queue::open(object_store.clone(), "q").await.unwrap());
        let run_id = RunId::new("01RUNIDX").unwrap();
        let mut state = ResearchState::new("a query", ResearchConfig::new("m"));
        state.phase = Phase::Fetching;
        state.fetch_queue = vec![url].into();

        let runner = ResearchStepRunner::new_ollama(
            rig_core::providers::ollama::Ollama::from_env().unwrap(),
            Arc::new(NoSearch),
        );
        let (jobs, jobs_handle) = runner.spawn_tool_runner(&queue, &object_store);
        let runner = runner.with_job_runner(jobs.clone());
        let runtime =
            WorkflowRuntime::builder(queue.clone(), object_store.clone(), UntilFetched(runner))
                .queue_name("test-workflow")
                .memo_prefix("test-memo")
                .build()
                .unwrap();
        let worker = runtime.spawn(std::future::pending::<()>());
        runtime
            .submit(RunSpec {
                run_id: Some(run_id.clone()),
                input: state.to_bytes(),
                ..Default::default()
            })
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(30), async {
            while !matches!(
                runtime
                    .view()
                    .status(&run_id)
                    .await
                    .unwrap()
                    .map(|s| s.state),
                Some(RunState::WaitingForGroup { .. })
            ) {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the run waits for its fetch group");

        runtime.cancel(&run_id, None).await.unwrap();
        let outcome = runtime.wait(&run_id).await.unwrap();
        assert_eq!(outcome.termination.status, TerminalStatus::Cancelled);
        // The fetch ends well before its timeout of 20 seconds.
        let group = jobs.group::<ToolJob>(fetch_group_id(&run_id).unwrap());
        let results = tokio::time::timeout(Duration::from_secs(10), group.join())
            .await
            .expect("the cancel of the run ends its fetch")
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].result.is_err());
        worker.shutdown().await.unwrap();
        jobs_handle.shutdown().await.unwrap();
    }

    fn fetch_call(id: &str, url: &Url, chunk: usize) -> RecordedCall {
        RecordedCall {
            id: id.to_string(),
            name: "fetch_page".to_string(),
            arguments: serde_json::json!({ "url": url, "chunk": chunk }),
        }
    }

    /// The chunk of `plan`, which must be a result without a member.
    fn planned_chunk(plan: CallPlan<ToolJob>) -> serde_json::Value {
        match plan {
            CallPlan::Result(result) => result.output().as_json().cloned().expect("a JSON chunk"),
            CallPlan::Member(job) => panic!("a member for a fetched page: {job:?}"),
        }
    }

    #[tokio::test]
    async fn a_fetch_page_call_reads_a_page_of_the_fetch_group_without_a_member() {
        let (base, hits) = page_server().await;
        let page = base.join("page").unwrap();
        let (jobs, run_id, _) = run_fetching_phase(vec![page.clone()]).await;
        let tools = ResearchTools {
            jobs,
            run_id,
            fetching: HashSet::from([page.clone()]),
            chunk_chars: 1000,
        };

        let plan = tools
            .plan(&fetch_call("call_1", &page, 1), &[])
            .await
            .unwrap();

        let chunk = planned_chunk(plan);
        assert!(chunk["text"].as_str().unwrap().contains("The whole text"));
        assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_fetch_page_call_reads_a_page_of_an_earlier_turn_without_a_member() {
        let (base, hits) = page_server().await;
        let page = base.join("page").unwrap();
        let object_store: Arc<dyn taquba::object_store::ObjectStore> = Arc::new(InMemory::new());
        let queue = Arc::new(Queue::open(object_store.clone(), "q").await.unwrap());
        let (jobs, _handle) =
            crate::tool_job::spawn_tool_runner(&queue, &object_store, Arc::new(NoSearch));
        let run_id = RunId::new("01RUNIDX").unwrap();
        let group = jobs.group::<ToolJob>(crate::host::turn_group_id(&run_id, 1).unwrap());
        group
            .submit([ToolJob::Fetch { url: page.clone() }])
            .await
            .unwrap();
        group.join().await.unwrap();
        let earlier = [
            Record::User {
                text: "query".to_string(),
            },
            Record::Assistant {
                text: String::new(),
                calls: vec![fetch_call("call_1", &page, 1)],
            },
            Record::Results {
                results: vec![crate::host::RecordedResult {
                    id: "call_1".to_string(),
                    name: "fetch_page".to_string(),
                    text: String::new(),
                    is_error: false,
                }],
            },
        ];
        let tools = ResearchTools {
            jobs,
            run_id,
            fetching: HashSet::new(),
            chunk_chars: 10,
        };

        let plan = tools
            .plan(&fetch_call("call_2", &page, 2), &earlier)
            .await
            .unwrap();

        assert_eq!(planned_chunk(plan)["chunk"], 2);
        assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 1);
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

    #[tokio::test]
    async fn investigating_with_zero_turns_skips_to_synthesizing() {
        let runner = ResearchStepRunner::new_ollama(
            ollama::Ollama::from_env().unwrap(),
            Arc::new(crate::search::Tavily::new("unused")),
        );
        let mut config = ResearchConfig::new("model");
        config.investigation_turns = 0;
        let mut state = ResearchState::new("query", config);
        state.phase = Phase::Investigating;

        // The runner does not have a job runner, so only the skip completes the
        // step.
        let wait = runner
            .run_investigating(&test_step("run", 4), &mut state)
            .await
            .unwrap();
        assert!(wait.is_none());
        assert_eq!(state.phase, Phase::Synthesizing);
    }

    #[test]
    fn resolve_reply_reads_the_signal_of_the_wait() {
        assert_eq!(
            resolve_reply(Some(b"the 2024 edition")),
            Reply::Given("the 2024 edition".to_string())
        );
        assert_eq!(resolve_reply(None), Reply::Assumed);
    }
}
