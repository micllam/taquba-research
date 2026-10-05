//! `FetchPage`: the durable [`Job`] that fetches a single URL and returns its
//! title and extracted plain text.
//!
//! The [`Phase::Fetching`](crate::state::Phase::Fetching) step parallelises its
//! work by submitting a `FetchPage` per URL with a deterministic
//! `idempotency_key` and awaiting the handles with `try_join_all`. A retried
//! step re-submits the same payloads, the job runner's result-aware idempotent
//! submit short-circuits to the recorded results, and the awaits resolve
//! without a second HTTP request.
//!
//! The fetching phase is a single workflow step for all URLs, and the
//! idempotency key derived from `(run_id, url)` replaces a per-step `Memo`.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use taquba::Queue;
use taquba::object_store::ObjectStore;
use taquba_workflow::StepErrorKind;
use taquba_workflow::jobs::{Job, JobContext, JobRunner, RunnerHandle};
use thiserror::Error;
use url::Url;

use crate::state::FetchedPage;

/// Maximum bytes read from a single fetch response, before the `max_chars` cap
/// on extracted text applies.
const FETCH_RESPONSE_BYTE_CAP: usize = 2 * 1024 * 1024;
/// Per-fetch HTTP timeout.
const FETCH_TIMEOUT: Duration = Duration::from_secs(20);
/// Logical queue name for fetch jobs, distinct from the workflow runtime's
/// queue.
pub const FETCH_QUEUE_NAME: &str = "research-fetch-jobs";
/// How long a `FetchPage` run result record is retained after the job reaches a
/// terminal state. An in-process idempotent re-submission (a workflow-step
/// retry) of the same `(run_id, url)` short-circuits to the recorded result
/// until this window elapses. After that the record is swept and a
/// re-submission fetches the page again. The week exceeds every realistic run
/// wall-time plus an inspection gap.
const FETCH_RESULT_RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// Build a [`JobRunner`] with the internal `FetchPage` job registered and an
/// `Arc<reqwest::Client>` on its state, then spawn its worker. Returns the
/// `JobRunner` for submission and a [`RunnerHandle`] for graceful shutdown.
///
/// The runner shares the supplied `queue` and `object_store` with the
/// surrounding workflow runtime. Jobs are enqueued with the
/// `research-fetch-jobs` queue name, and their memo and run result records are
/// stored at a sibling prefix in the object store.
pub fn spawn_fetch_runner(
    queue: &Arc<Queue>,
    object_store: &Arc<dyn ObjectStore>,
) -> (JobRunner, RunnerHandle) {
    let http = Arc::new(
        reqwest::Client::builder()
            .timeout(FETCH_TIMEOUT)
            .user_agent(concat!("taquba-research/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("reqwest client builder cannot fail with default config"),
    );
    let job_runner = JobRunner::builder(queue.clone(), object_store.clone())
        .queue_name(FETCH_QUEUE_NAME)
        .state(http)
        .retention(FETCH_RESULT_RETENTION)
        .register::<FetchPage>()
        .build();
    let handle = job_runner.spawn(std::future::pending::<()>());
    (job_runner, handle)
}

/// One durable HTTP fetch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct FetchPage {
    /// The owning research run. Combined with `url` into the idempotency key,
    /// so a retry is deduplicated or short-circuits to the recorded result.
    pub run_id: String,
    /// The URL to fetch.
    pub url: Url,
    /// Per-page text cap applied after HTML extraction.
    pub max_chars: usize,
}

/// Failure modes for [`FetchPage`].
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
    /// The job's cancel-token fired mid-fetch (via `Queue::cancel`, issued when
    /// the surrounding research run is cancelled). Treated as permanent, so the
    /// job dead-letters and does not retry after the owning run ends.
    #[error("cancelled")]
    Cancelled,
    /// Extending the delivery's lease to cover the fetch failed: the claim was
    /// lost to a re-delivery. Retryable.
    #[error("lease: {0}")]
    Lease(String),
}

impl Job for FetchPage {
    const NAME: &'static str = "taquba-research.fetch-page";
    type Output = FetchedPage;
    type Error = FetchError;

    async fn run(&self, ctx: JobContext<'_>) -> Result<FetchedPage, FetchError> {
        let http = ctx.state::<Arc<reqwest::Client>>();
        // Extend the lease to cover the fetch's timeout before issuing it, so a
        // slow page cannot outlive the lease.
        if let Err(e) = ctx.lease.ensure_at_least(FETCH_TIMEOUT) {
            return Err(match e {
                taquba::Error::CancelRequested => FetchError::Cancelled,
                other => FetchError::Lease(other.to_string()),
            });
        }
        // Race the HTTP fetch against the job's cooperative cancellation. When
        // the surrounding run is cancelled, run_fetching calls
        // `Queue::cancel(job_id)`, which fires this token. The in-flight HTTP
        // request is then aborted before the reqwest timeout.
        tokio::select! {
            result = fetch_and_extract(http, &self.url, self.max_chars) => result,
            _ = ctx.cancel_token.cancelled() => Err(FetchError::Cancelled),
        }
    }

    fn idempotency_key(&self) -> Option<String> {
        Some(format!("fetch:{}:{}", self.run_id, self.url))
    }

    fn classify(&self, error: &FetchError) -> StepErrorKind {
        match error {
            FetchError::Transport(_) | FetchError::ReadBody(_) | FetchError::Lease(_) => {
                StepErrorKind::Transient
            }
            FetchError::HttpStatus(code) if is_transient_status(*code) => StepErrorKind::Transient,
            FetchError::HttpStatus(_)
            | FetchError::NonText(_)
            | FetchError::Empty
            | FetchError::Cancelled => StepErrorKind::Permanent,
        }
    }
}

/// HTTP retry policy: 5xx server errors, 429 rate-limit and every non-4xx code
/// are transient. The rest of 4xx (404, 401, 422, …) is permanent.
fn is_transient_status(code: u16) -> bool {
    code == 429 || !(400..500).contains(&code)
}

/// Fetch `url`, decode the response and return the page title and a plain-text
/// rendering of the body capped at `max_chars`. Returns a structured
/// [`FetchError`] that [`FetchPage::classify`] maps to a retry or a permanent
/// failure.
async fn fetch_and_extract(
    http: &reqwest::Client,
    url: &Url,
    max_chars: usize,
) -> Result<FetchedPage, FetchError> {
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
    let text: String = text.chars().take(max_chars).collect();
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

    #[test]
    fn classify_routes_transport_and_5xx_to_transient() {
        let job = FetchPage {
            run_id: "r".into(),
            url: Url::parse("http://example.com/").unwrap(),
            max_chars: 1000,
        };
        assert_eq!(
            job.classify(&FetchError::Transport("dns".into())),
            StepErrorKind::Transient
        );
        assert_eq!(
            job.classify(&FetchError::HttpStatus(503)),
            StepErrorKind::Transient
        );
        assert_eq!(
            job.classify(&FetchError::HttpStatus(429)),
            StepErrorKind::Transient
        );
        assert_eq!(
            job.classify(&FetchError::Lease("claim lost".into())),
            StepErrorKind::Transient
        );
    }

    #[test]
    fn classify_routes_4xx_and_non_text_to_permanent() {
        let job = FetchPage {
            run_id: "r".into(),
            url: Url::parse("http://example.com/").unwrap(),
            max_chars: 1000,
        };
        assert_eq!(
            job.classify(&FetchError::HttpStatus(404)),
            StepErrorKind::Permanent
        );
        assert_eq!(
            job.classify(&FetchError::NonText("application/pdf".into())),
            StepErrorKind::Permanent
        );
        assert_eq!(job.classify(&FetchError::Empty), StepErrorKind::Permanent);
        assert_eq!(
            job.classify(&FetchError::Cancelled),
            StepErrorKind::Permanent
        );
    }
}
