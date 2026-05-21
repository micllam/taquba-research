//! [`ResearchStepRunner`]: the [`StepRunner`] that drives a research run
//! through its six phases.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use rig_core::client::CompletionClient;
use rig_core::completion::{
    CompletionError, Prompt, PromptError, StructuredOutputError, TypedPrompt, Usage,
};
use rig_core::http_client;
use rig_core::providers::{anthropic, openai};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use taquba_workflow::{Step, StepError, StepOutcome, StepRunner};
use url::Url;

use crate::report::{Citation, Report, RunStats, render_markdown};
use crate::search::{SearchBackend, SearchError};
use crate::state::{FetchedPage, Phase, ResearchConfig, ResearchState, Summary, TokenUsage};
use crate::store::RunStore;

/// Maximum bytes we'll read from a single fetch response, before
/// applying the further `max_page_chars` cap on extracted text.
const FETCH_RESPONSE_BYTE_CAP: usize = 2 * 1024 * 1024;
/// Per-fetch HTTP timeout.
const FETCH_TIMEOUT: Duration = Duration::from_secs(20);
/// Preamble applied to every Rig agent built by the runner. Kept
/// terse: the per-phase prompts carry the task-specific instructions.
const AGENT_PREAMBLE: &str = "Be precise and concise.";
/// Cadence at which a step polls its cancellation sentinel while
/// phase work is in flight. Sets the upper bound on how long an LLM
/// or HTTP call keeps running after the CLI's `cancel` lands.
const CANCEL_POLL_INTERVAL: Duration = Duration::from_secs(1);

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
    http: reqwest::Client,
    run_store: Option<RunStore>,
}

/// Per-provider LLM client.
pub(crate) enum ProviderClient {
    OpenAi(openai::Client),
    Anthropic(anthropic::Client),
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

    pub(crate) fn from_provider(provider: ProviderClient, search: Arc<dyn SearchBackend>) -> Self {
        let http = reqwest::Client::builder()
            .timeout(FETCH_TIMEOUT)
            .user_agent(concat!("taquba-research/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("reqwest client builder cannot fail with default config");
        Self {
            provider: Arc::new(provider),
            search,
            http,
            run_store: None,
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
            Phase::Planning => self.run_planning(state).await?,
            Phase::Searching => self.run_searching(state).await?,
            Phase::Fetching => self.run_fetching(state).await?,
            Phase::Summarizing => self.run_summarizing(state).await?,
            Phase::Synthesizing => self.run_synthesizing(state).await?,
            Phase::Writing => {
                let report = self.run_writing(state, &step.run_id).await?;
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
        Ok(StepOutcome::Continue {
            payload: state.to_bytes(),
        })
    }

    async fn run_planning(&self, state: &mut ResearchState) -> Result<(), StepError> {
        tracing::info!("planning sub-questions");
        let prompt = format!(
            "You are a research planner. Decompose the user's query into at most {} \
             distinct, web-searchable sub-questions that together cover the topic.\n\n\
             Query: {}",
            state.config.depth, state.query
        );
        let plan: Plan = self.llm_prompt_typed(&prompt, state).await?;

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

    async fn run_searching(&self, state: &mut ResearchState) -> Result<(), StepError> {
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
        let results = self.search.search(&q, 5).await?;

        let known: HashSet<String> = state.fetched.keys().cloned().collect();
        let mut queued: HashSet<String> = state
            .fetch_queue
            .iter()
            .map(|u| u.as_str().to_string())
            .collect();

        for h in &results {
            let s = h.url.as_str().to_string();
            if known.contains(&s) || queued.contains(&s) {
                continue;
            }
            if state.fetch_queue.len() >= state.config.max_sources {
                break;
            }
            state.fetch_queue.push_back(h.url.clone());
            queued.insert(s);
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

    async fn run_fetching(&self, state: &mut ResearchState) -> Result<(), StepError> {
        let Some(url) = state.fetch_queue.pop_front() else {
            state.phase = Phase::Summarizing;
            return Ok(());
        };
        let total = state.fetched.len() + state.fetch_queue.len() + 1;
        let done = state.fetched.len() + 1;
        tracing::info!("fetching ({done}/{total}): {url}");

        match fetch_and_extract(&self.http, &url, state.config.max_page_chars).await {
            Ok(page) => {
                state.summarize_queue.push_back(url.clone());
                state.fetched.insert(url.as_str().to_string(), page);
            }
            Err(msg) => {
                // "Best effort" page fetch: skip flaky sources rather than
                // blocking the whole run on one. A multi-source agent
                // can drop one URL.
                tracing::warn!(url = %url, error = %msg, "fetch error, skipping page");
            }
        }

        if state.fetch_queue.is_empty() {
            state.phase = Phase::Summarizing;
        }
        Ok(())
    }

    async fn run_summarizing(&self, state: &mut ResearchState) -> Result<(), StepError> {
        let Some(url) = state.summarize_queue.pop_front() else {
            state.phase = Phase::Synthesizing;
            return Ok(());
        };
        let url_key = url.as_str().to_string();
        let Some(page) = state.fetched.get(&url_key).cloned() else {
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
        let parsed: SummaryResp = self.llm_prompt_typed(&prompt, state).await?;

        state.summaries.insert(
            url_key,
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

    async fn run_synthesizing(&self, state: &mut ResearchState) -> Result<(), StepError> {
        tracing::info!("synthesizing from {} sources", state.summaries.len());
        let mut sources = String::new();
        let mut sorted: Vec<(&String, &Summary)> = state.summaries.iter().collect();
        sorted.sort_by(|a, b| {
            b.1.relevance
                .partial_cmp(&a.1.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (idx, (_url, s)) in sorted.iter().enumerate() {
            sources.push_str(&format!(
                "Source {n} (relevance {r:.2}, title: {t}):\n{x}\n\n",
                n = idx + 1,
                r = s.relevance,
                t = s.title,
                x = s.text,
            ));
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
        let synthesis = self.llm_prompt(&prompt, state).await?;
        state.synthesis = Some(synthesis);
        state.phase = Phase::Writing;
        Ok(())
    }

    async fn run_writing(
        &self,
        state: &mut ResearchState,
        run_id: &str,
    ) -> Result<Report, StepError> {
        tracing::info!("writing final report");
        let synthesis = state.synthesis.clone().unwrap_or_default();

        let mut sorted: Vec<(&String, &Summary)> = state.summaries.iter().collect();
        sorted.sort_by(|a, b| {
            b.1.relevance
                .partial_cmp(&a.1.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let citations: Vec<Citation> = sorted
            .iter()
            .enumerate()
            .filter_map(|(i, (url, s))| {
                Url::parse(url).ok().map(|u| Citation {
                    index: i + 1,
                    url: u,
                    title: s.title.clone(),
                })
            })
            .collect();

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
            syn = synthesis,
        );

        let body = self.llm_prompt(&prompt, state).await?;

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

        let markdown = render_markdown(&state.query, run_id, &body, &citations, &stats);
        Ok(Report {
            run_id: run_id.to_string(),
            query: state.query.clone(),
            markdown,
            citations,
            stats,
        })
    }

    /// Run a single completion via Rig, dispatched to the configured
    /// provider. Records this call's token usage on `state.token_usage`
    /// and logs the per-call counts at info level.
    async fn llm_prompt(
        &self,
        prompt: &str,
        state: &mut ResearchState,
    ) -> Result<String, StepError> {
        let response = match self.provider.as_ref() {
            ProviderClient::OpenAi(client) => {
                let agent = client
                    .agent(&state.config.model)
                    .preamble(AGENT_PREAMBLE)
                    .max_tokens(state.config.max_tokens_per_call)
                    .build();
                agent
                    .prompt(prompt)
                    .extended_details()
                    .await
                    .map_err(classify_rig_err)?
            }
            ProviderClient::Anthropic(client) => {
                let agent = client
                    .agent(&state.config.model)
                    .preamble(AGENT_PREAMBLE)
                    .max_tokens(state.config.max_tokens_per_call)
                    .build();
                agent
                    .prompt(prompt)
                    .extended_details()
                    .await
                    .map_err(classify_rig_err)?
            }
        };
        record_usage(&mut state.token_usage, &response.usage);
        Ok(response.output)
    }

    /// Run a structured completion via Rig's `prompt_typed`, dispatched
    /// to the configured provider. Same usage-tracking behaviour as
    /// [`Self::llm_prompt`].
    async fn llm_prompt_typed<T>(
        &self,
        prompt: &str,
        state: &mut ResearchState,
    ) -> Result<T, StepError>
    where
        T: JsonSchema + DeserializeOwned + Send + 'static,
    {
        let (output, usage) = match self.provider.as_ref() {
            ProviderClient::OpenAi(client) => {
                let agent = client
                    .agent(&state.config.model)
                    .preamble(AGENT_PREAMBLE)
                    .max_tokens(state.config.max_tokens_per_call)
                    .build();
                let response = agent
                    .prompt_typed::<T>(prompt)
                    .extended_details()
                    .await
                    .map_err(classify_structured_err)?;
                (response.output, response.usage)
            }
            ProviderClient::Anthropic(client) => {
                let agent = client
                    .agent(&state.config.model)
                    .preamble(AGENT_PREAMBLE)
                    .max_tokens(state.config.max_tokens_per_call)
                    .build();
                let response = agent
                    .prompt_typed::<T>(prompt)
                    .extended_details()
                    .await
                    .map_err(classify_structured_err)?;
                (response.output, response.usage)
            }
        };
        record_usage(&mut state.token_usage, &usage);
        Ok(output)
    }
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

/// Classify a Rig prompt error as transient or permanent.
fn classify_rig_err(err: PromptError) -> StepError {
    let msg = err.to_string();
    match err {
        // Configuration / runner-policy issues.
        PromptError::MaxTurnsError { .. }
        | PromptError::ToolError(_)
        | PromptError::ToolServerError(_)
        | PromptError::PromptCancelled { .. } => {
            StepError::permanent(format!("LLM permanent failure: {msg}"))
        }

        PromptError::CompletionError(CompletionError::HttpError(http_err)) => {
            classify_http(&http_err, &msg)
        }

        // Anything else from the completion layer (ProviderError,
        // JsonError, UrlError, RequestError, ResponseError). Auth
        // failures come through `HttpError(InvalidStatusCode(401|403))`
        // above, so we default the rest to transient.
        // Persistent ones get dead-lettered after `max_attempts`.
        PromptError::CompletionError(_) => StepError::transient(format!("LLM call failed: {msg}")),
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
#[derive(Deserialize, JsonSchema)]
struct Plan {
    sub_questions: Vec<String>,
}

#[derive(Deserialize, JsonSchema)]
struct SummaryResp {
    summary: String,
    relevance: f32,
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
    }
}

/// Fetch a URL and extract title + plain-text body. Errors are returned
/// as a single message string; the runner logs them and skips the page
/// rather than failing the step, so the transient/permanent distinction
/// the HTTP layer makes isn't acted on at this boundary (a multi-source
/// agent loses little by dropping one flaky URL).
async fn fetch_and_extract(
    http: &reqwest::Client,
    url: &Url,
    max_chars: usize,
) -> Result<FetchedPage, String> {
    let resp = http
        .get(url.as_str())
        .send()
        .await
        .map_err(|e| format!("send: {e}"))?;

    let status = resp.status();
    if !status.is_success() {
        return Err(format!("HTTP {status}"));
    }

    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    let is_html = content_type.contains("html") || content_type.is_empty();
    let is_text = content_type.contains("text") || is_html;
    if !is_text {
        return Err(format!("non-text content-type: {content_type}"));
    }

    let bytes = resp.bytes().await.map_err(|e| format!("read body: {e}"))?;
    let raw = String::from_utf8_lossy(&bytes[..bytes.len().min(FETCH_RESPONSE_BYTE_CAP)]);

    let (title, text) = if is_html {
        extract_html(&raw)
    } else {
        (String::new(), raw.to_string())
    };
    let text: String = text.chars().take(max_chars).collect();
    if text.trim().is_empty() {
        return Err("empty extracted text".into());
    }
    Ok(FetchedPage { title, text })
}

/// Extract the page's `<title>` and a plain-text
/// rendering of the body.
fn extract_html(html: &str) -> (String, String) {
    let title = extract_tag_content(html, "title").unwrap_or_default();
    // Width is intentionally large to avoid injecting newlines
    // mid-sentence and bloating tokens.
    let text = html2text::from_read(html.as_bytes(), 100_000).unwrap_or_default();
    (title.trim().to_string(), text)
}

fn extract_tag_content(html: &str, tag: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let i = lower.find(&open)?;
    let after_open = i + html[i..].find('>')? + 1;
    let j = lower[after_open..].find(&close)?;
    Some(html[after_open..after_open + j].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::StatusCode;
    use taquba::object_store::memory::InMemory;
    use taquba::object_store::path::Path;
    use taquba_workflow::StepErrorKind;

    fn test_store() -> RunStore {
        RunStore::new(Arc::new(InMemory::new()), &Path::default())
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

    #[test]
    fn extract_html_yields_title_and_visible_body() {
        let html = r#"<html><head><title>Hi</title><style>body{}</style></head>
        <body><script>alert(1)</script><p>Hello &amp; welcome</p></body></html>"#;
        let (title, text) = extract_html(html);
        assert_eq!(title, "Hi");
        assert!(text.contains("Hello & welcome"));
        assert!(!text.contains("alert"));
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
}
