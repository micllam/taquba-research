//! `FetchPage`: the durable [`taquba_jobs::Job`] that fetches a single
//! URL and returns its title + extracted plain text.
//!
//! Submitting one `FetchPage` per URL with a deterministic
//! `idempotency_key` and `try_join_all`-ing the handles is how the
//! [`Phase::Fetching`](crate::state::Phase::Fetching) step parallelises
//! its work while staying correct under at-least-once retries. A
//! retried step re-submits the same payloads, taquba-jobs's
//! result-aware idempotent submit short-circuits to the cached
//! result blobs, and the awaits resolve without re-running any HTTP.
//!
//! Per-URL work has no LLM cost, so saving the bytes is incidental
//! here; what matters is that the surrounding step becomes a single
//! workflow step rather than N (one-per-URL), and the
//! `(run_id, url)`-keyed idempotency replaces the per-step `Memo` we'd
//! otherwise need.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use taquba::Queue;
use taquba::object_store::ObjectStore;
use taquba_jobs::{ErrorKind, Job, JobContext, JobRunner, RunnerHandle};
use thiserror::Error;
use url::Url;

use crate::state::FetchedPage;

/// Maximum bytes we'll read from a single fetch response, before
/// applying the further `max_chars` cap on extracted text.
const FETCH_RESPONSE_BYTE_CAP: usize = 2 * 1024 * 1024;
/// Per-fetch HTTP timeout.
const FETCH_TIMEOUT: Duration = Duration::from_secs(20);
/// Logical queue name for fetch jobs. Distinct from the workflow
/// runtime's queue so retention policies can diverge if needed.
const FETCH_QUEUE_NAME: &str = "research-fetch-jobs";

/// Build a [`JobRunner`] with the internal `FetchPage` job
/// registered and an `Arc<reqwest::Client>` on its state, then spawn
/// its worker. Returns an `Arc<JobRunner>` for submission and a
/// [`RunnerHandle`] for graceful shutdown.
///
/// The runner shares the supplied `queue` and `object_store` with
/// the surrounding workflow runtime; jobs are enqueued under the
/// `research-fetch-jobs` queue-name and their result blobs live
/// under a sibling prefix in the object store.
pub fn spawn_fetch_runner(
    queue: &Arc<Queue>,
    object_store: &Arc<dyn ObjectStore>,
) -> Result<(Arc<JobRunner>, RunnerHandle)> {
    let http = Arc::new(
        reqwest::Client::builder()
            .timeout(FETCH_TIMEOUT)
            .user_agent(concat!("taquba-research/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("reqwest client builder cannot fail with default config"),
    );
    let mut job_runner = JobRunner::builder()
        .queue(queue.clone())
        .object_store(object_store.clone())
        .queue_name(FETCH_QUEUE_NAME)
        .state(http)
        .build()
        .context("building fetch JobRunner")?;
    job_runner.register::<FetchPage>();
    let handle = job_runner.spawn(std::future::pending::<()>());
    Ok((Arc::new(job_runner), handle))
}

/// One durable HTTP fetch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct FetchPage {
    /// The owning research run; combined with `url` into the
    /// idempotency key so retries dedup or short-circuit to a
    /// cached result.
    pub run_id: String,
    /// The URL to fetch.
    pub url: Url,
    /// Per-page text cap applied after HTML extraction.
    pub max_chars: usize,
}

/// Failure modes for [`FetchPage`].
#[derive(Debug, Error)]
pub(crate) enum FetchError {
    /// Transport-layer failure: connect timeout, DNS, TLS, broken
    /// stream. Retryable.
    #[error("send: {0}")]
    Transport(String),
    /// Reading the response body failed mid-stream.
    #[error("read body: {0}")]
    ReadBody(String),
    /// The server returned a non-success status. Classification
    /// depends on the code: 5xx and 429 retry, other 4xx fail fast.
    #[error("HTTP {0}")]
    HttpStatus(u16),
    /// `Content-Type` was set to something that isn't text-like.
    #[error("non-text content-type: {0}")]
    NonText(String),
    /// The page fetched fine but extraction yielded no readable
    /// text. Treated as permanent.
    #[error("empty extracted text")]
    Empty,
}

impl Job for FetchPage {
    const NAME: &'static str = "taquba-research.fetch-page";
    type Output = FetchedPage;
    type Error = FetchError;

    async fn run(&self, ctx: JobContext<'_>) -> Result<FetchedPage, FetchError> {
        let http = ctx.state::<Arc<reqwest::Client>>();
        fetch_and_extract(http, &self.url, self.max_chars).await
    }

    fn idempotency_key(&self) -> Option<String> {
        Some(format!("fetch:{}:{}", self.run_id, self.url))
    }

    fn classify(&self, error: &FetchError) -> ErrorKind {
        match error {
            FetchError::Transport(_) | FetchError::ReadBody(_) => ErrorKind::Transient,
            FetchError::HttpStatus(code) if is_transient_status(*code) => ErrorKind::Transient,
            FetchError::HttpStatus(_) | FetchError::NonText(_) | FetchError::Empty => {
                ErrorKind::Permanent
            }
        }
    }
}

/// HTTP retry policy: 5xx server errors, 429 rate-limit, and any
/// non-4xx code are transient; the rest of 4xx (404, 401, 422, …)
/// will not improve on retry.
fn is_transient_status(code: u16) -> bool {
    code == 429 || !(400..500).contains(&code)
}

/// Fetch `url`, decode the response, and return the page title +
/// plain-text rendering of the body capped at `max_chars`. Returns a
/// structured [`FetchError`] so [`FetchPage::classify`] can route
/// retryable failures back through the queue.
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
            ErrorKind::Transient
        );
        assert_eq!(
            job.classify(&FetchError::HttpStatus(503)),
            ErrorKind::Transient
        );
        assert_eq!(
            job.classify(&FetchError::HttpStatus(429)),
            ErrorKind::Transient
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
            ErrorKind::Permanent
        );
        assert_eq!(
            job.classify(&FetchError::NonText("application/pdf".into())),
            ErrorKind::Permanent
        );
        assert_eq!(job.classify(&FetchError::Empty), ErrorKind::Permanent);
    }
}
