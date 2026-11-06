//! `ToolJob`: the durable [`Job`] of a tool call, a web search or the fetch of
//! a page. Its output is the artifact of the call, the search results or the
//! title and the whole extracted plain text of the page, which the run result
//! record of the job keeps.
//!
//! The [`Phase::Fetching`](crate::state::Phase::Fetching) step submits a fetch
//! per URL as the members of the fetch group of the run, and each step of the
//! investigating agent submits the tool calls of its reply as the members of
//! the group of the turn. The run waits for a group without a worker. A retried
//! step submits the same members again, which runs only the members without a
//! success.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use taquba::Queue;
use taquba::object_store::ObjectStore;
use taquba_workflow::jobs::{Job, JobContext, JobRunner, RunnerHandle};
use taquba_workflow::{RunId, StepError, StepErrorKind};
use thiserror::Error;
use url::Url;

use crate::runner::SEARCH_CALL_TIMEOUT;
use crate::search::{SearchBackend, SearchError, SearchResult};
use crate::state::FetchedPage;

/// Maximum bytes read from a single fetch response.
const FETCH_RESPONSE_BYTE_CAP: usize = 2 * 1024 * 1024;
/// Per-fetch HTTP timeout.
const FETCH_TIMEOUT: Duration = Duration::from_secs(20);
/// Results per web search of a tool call.
pub(crate) const SEARCH_LIMIT: usize = 5;
/// Logical queue name for tool jobs, distinct from the workflow runtime's
/// queue.
pub const TOOL_QUEUE_NAME: &str = "research-tool-jobs";
/// How long a `ToolJob` run result record and a group are retained after the
/// job or the last member reaches a terminal state. A retried step reads the
/// recorded results until this window elapses. The week exceeds every realistic
/// run wall-time plus an inspection gap.
const TOOL_RESULT_RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// Build a [`JobRunner`] with `ToolJob` registered and `search` and a
/// `reqwest::Client` on its state, then spawn its worker. Returns the
/// `JobRunner` for submission and a [`RunnerHandle`] for graceful shutdown.
///
/// The runner shares the supplied `queue` and `object_store` with the
/// surrounding workflow runtime. Jobs are enqueued with the
/// `research-tool-jobs` queue name, and their memo and run result records are
/// stored at a sibling prefix in the object store.
pub(crate) fn spawn_tool_runner(
    queue: &Arc<Queue>,
    object_store: &Arc<dyn ObjectStore>,
    search: Arc<dyn SearchBackend>,
) -> (JobRunner, RunnerHandle) {
    let http = Arc::new(
        reqwest::Client::builder()
            .timeout(FETCH_TIMEOUT)
            .user_agent(concat!("taquba-research/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("reqwest client builder cannot fail with default config"),
    );
    let job_runner = JobRunner::builder(queue.clone(), object_store.clone())
        .queue_name(TOOL_QUEUE_NAME)
        .state(http)
        .state(search)
        .retention(TOOL_RESULT_RETENTION)
        .group_retention(TOOL_RESULT_RETENTION)
        .register::<ToolJob>()
        .build();
    let handle = job_runner.spawn(std::future::pending::<()>());
    (job_runner, handle)
}

/// The id of the fetch group of the run `run_id`.
pub(crate) fn fetch_group_id(run_id: &RunId) -> Result<RunId, StepError> {
    RunId::new(format!("fetch-{run_id}"))
        .map_err(|e| StepError::permanent(format!("group id: {e}")))
}

/// One durable tool call. The idempotency key is the member key in a group: the
/// URL of a fetch, so that the calls of a reply for one page share a member,
/// and the call id of a search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tool", rename_all = "snake_case")]
pub(crate) enum ToolJob {
    /// A `web_search` call.
    Search {
        /// The call id of the provider.
        call: String,
        /// The search query.
        query: String,
    },
    /// The fetch of a page.
    Fetch {
        /// The URL to fetch.
        url: Url,
    },
}

/// The output of a [`ToolJob`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub(crate) enum Artifact {
    /// The results of a search.
    Results(Vec<SearchResult>),
    /// A fetched page.
    Page(FetchedPage),
}

/// Failure modes of a [`ToolJob`].
#[derive(Debug, Error)]
pub(crate) enum ToolError {
    /// The fetch failed.
    #[error(transparent)]
    Fetch(#[from] FetchError),
    /// The search failed.
    #[error(transparent)]
    Search(#[from] SearchError),
    /// The call outlasted its bound, in seconds. Retryable.
    #[error("timed out after {0}s")]
    Timeout(u64),
    /// The job's cancel token fired, at a cancellation of the group. Treated as
    /// permanent, so the job dead-letters and does not retry.
    #[error("cancelled")]
    Cancelled,
    /// Extending the delivery's lease to cover the call failed: the claim was
    /// lost to a re-delivery. Retryable.
    #[error("lease: {0}")]
    Lease(String),
}

/// Failure modes of a fetch.
#[derive(Debug, Error)]
pub(crate) enum FetchError {
    /// Transport-layer failure: connect timeout, DNS, TLS, broken stream.
    /// Retryable.
    #[error("send: {0}")]
    Transport(String),
    /// Reading the response body failed mid-stream.
    #[error("read body: {0}")]
    ReadBody(String),
    /// The server returned a non-success status. Classification depends on the
    /// code: 5xx and 429 retry, other 4xx fail fast.
    #[error("HTTP {0}")]
    HttpStatus(u16),
    /// `Content-Type` is set to a type that is not text-like.
    #[error("non-text content-type: {0}")]
    NonText(String),
    /// The fetch succeeded but extraction did not yield readable text. Treated
    /// as permanent.
    #[error("empty extracted text")]
    Empty,
}

impl ToolJob {
    /// The time bound of the call.
    fn bound(&self) -> Duration {
        match self {
            Self::Search { .. } => SEARCH_CALL_TIMEOUT,
            Self::Fetch { .. } => FETCH_TIMEOUT,
        }
    }

    async fn call(&self, ctx: &JobContext<'_>) -> Result<Artifact, ToolError> {
        match self {
            Self::Search { query, .. } => {
                let search = ctx.state::<Arc<dyn SearchBackend>>();
                Ok(Artifact::Results(search.search(query, SEARCH_LIMIT).await?))
            }
            Self::Fetch { url } => {
                let http = ctx.state::<Arc<reqwest::Client>>();
                Ok(Artifact::Page(fetch_and_extract(http, url).await?))
            }
        }
    }
}

impl Job for ToolJob {
    const NAME: &'static str = "taquba-research.tool";
    type Output = Artifact;
    type Error = ToolError;

    async fn run(&self, ctx: JobContext<'_>) -> Result<Artifact, ToolError> {
        // Extend the lease to cover the bound of the call before issuing it, so
        // a slow call cannot outlive the lease.
        let bound = self.bound();
        if let Err(e) = ctx.lease.ensure_at_least(bound) {
            return Err(match e {
                taquba::Error::CancelRequested => ToolError::Cancelled,
                other => ToolError::Lease(other.to_string()),
            });
        }
        // A cancellation of the group fires the token, which aborts the call
        // before its bound.
        tokio::select! {
            result = tokio::time::timeout(bound, self.call(&ctx)) => {
                result.unwrap_or(Err(ToolError::Timeout(bound.as_secs())))
            }
            _ = ctx.cancel_token.cancelled() => Err(ToolError::Cancelled),
        }
    }

    fn idempotency_key(&self) -> Option<String> {
        Some(match self {
            Self::Search { call, .. } => format!("search:{call}"),
            Self::Fetch { url } => format!("fetch:{url}"),
        })
    }

    fn classify(&self, error: &ToolError) -> StepErrorKind {
        match error {
            ToolError::Fetch(error) => fetch_kind(error),
            ToolError::Search(error) => search_kind(error),
            ToolError::Timeout(_) | ToolError::Lease(_) => StepErrorKind::Transient,
            ToolError::Cancelled => StepErrorKind::Permanent,
        }
    }
}

/// The retry class of a failed fetch.
fn fetch_kind(error: &FetchError) -> StepErrorKind {
    match error {
        FetchError::Transport(_) | FetchError::ReadBody(_) => StepErrorKind::Transient,
        FetchError::HttpStatus(code) if is_transient_status(*code) => StepErrorKind::Transient,
        FetchError::HttpStatus(_) | FetchError::NonText(_) | FetchError::Empty => {
            StepErrorKind::Permanent
        }
    }
}

/// The retry class of a failed search: a rejected credential is permanent.
pub(crate) fn search_kind(error: &SearchError) -> StepErrorKind {
    match error {
        SearchError::AuthFailed => StepErrorKind::Permanent,
        SearchError::RateLimit { .. } | SearchError::Transport(_) | SearchError::Other(_) => {
            StepErrorKind::Transient
        }
    }
}

/// HTTP retry policy: 5xx server errors, 429 rate-limit and every non-4xx code
/// are transient. The rest of 4xx (404, 401, 422, …) is permanent.
fn is_transient_status(code: u16) -> bool {
    code == 429 || !(400..500).contains(&code)
}

/// Fetch `url`, decode the response and return the page title and a plain-text
/// rendering of the body. Returns a structured [`FetchError`] that
/// [`fetch_kind`] maps to a retry or a permanent failure.
async fn fetch_and_extract(http: &reqwest::Client, url: &Url) -> Result<FetchedPage, FetchError> {
    let resp = http
        .get(url.as_str())
        .send()
        .await
        .map_err(|e| FetchError::Transport(e.to_string()))?;

    let status = resp.status();
    if !status.is_success() {
        return Err(FetchError::HttpStatus(status.as_u16()));
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
        return Err(FetchError::NonText(content_type));
    }

    let bytes = resp
        .bytes()
        .await
        .map_err(|e| FetchError::ReadBody(e.to_string()))?;
    let raw = String::from_utf8_lossy(&bytes[..bytes.len().min(FETCH_RESPONSE_BYTE_CAP)]);

    let (title, text) = if is_html {
        extract_html(&raw)
    } else {
        (String::new(), raw.to_string())
    };
    if text.trim().is_empty() {
        return Err(FetchError::Empty);
    }
    Ok(FetchedPage { title, text })
}

/// Extract the page's `<title>` and a plain-text rendering of the body.
fn extract_html(html: &str) -> (String, String) {
    let title = extract_tag_content(html, "title").unwrap_or_default();
    // The large width prevents newlines mid-sentence, which add tokens.
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

    #[test]
    fn extract_html_yields_title_and_visible_body() {
        let html = r#"<html><head><title>Hi</title><style>body{}</style></head>
        <body><script>alert(1)</script><p>Hello &amp; welcome</p></body></html>"#;
        let (title, text) = extract_html(html);
        assert_eq!(title, "Hi");
        assert!(text.contains("Hello & welcome"));
        assert!(!text.contains("alert"));
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

    fn fetch_job() -> ToolJob {
        ToolJob::Fetch {
            url: Url::parse("http://example.com/").unwrap(),
        }
    }

    #[test]
    fn classify_routes_transport_5xx_rate_limits_and_timeouts_to_transient() {
        let job = fetch_job();
        for error in [
            ToolError::Fetch(FetchError::Transport("dns".into())),
            ToolError::Fetch(FetchError::HttpStatus(503)),
            ToolError::Fetch(FetchError::HttpStatus(429)),
            ToolError::Search(SearchError::RateLimit { retry_after: None }),
            ToolError::Search(SearchError::Other("busy".into())),
            ToolError::Timeout(20),
            ToolError::Lease("claim lost".into()),
        ] {
            assert_eq!(job.classify(&error), StepErrorKind::Transient, "{error}");
        }
    }

    #[test]
    fn classify_routes_4xx_non_text_auth_and_cancellation_to_permanent() {
        let job = fetch_job();
        for error in [
            ToolError::Fetch(FetchError::HttpStatus(404)),
            ToolError::Fetch(FetchError::NonText("application/pdf".into())),
            ToolError::Fetch(FetchError::Empty),
            ToolError::Search(SearchError::AuthFailed),
            ToolError::Cancelled,
        ] {
            assert_eq!(job.classify(&error), StepErrorKind::Permanent, "{error}");
        }
    }
}
