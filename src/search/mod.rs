//! Pluggable web-search backends.
//!
//! Implementations of [`SearchBackend`](crate::search::SearchBackend) return a
//! list of [`SearchResult`](crate::search::SearchResult)s for a query. The
//! research runner uses the backend during the searching phase, then passes
//! every result's URL to the fetching phase.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use url::Url;

mod tavily;
pub use tavily::Tavily;

/// A single search result returned to the research runner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Result URL.
    pub url: Url,
    /// Result title.
    pub title: String,
    /// Snippet / summary returned by the search backend.
    pub snippet: String,
}

/// Failure modes for `from_env`-style constructors that read process
/// environment.
#[derive(Debug, Error)]
pub enum EnvError {
    /// The expected variable was not set in the process environment.
    #[error("environment variable `{0}` is not set")]
    Missing(&'static str),
    /// The variable was set but empty after trimming.
    #[error("environment variable `{0}` is set but empty")]
    Empty(&'static str),
}

/// Failure modes of a search, which the runner classifies as transient or
/// permanent. Conversion to [`taquba_workflow::StepError`] happens in
/// [`crate::ResearchStepRunner`].
#[derive(Debug, Error)]
pub enum SearchError {
    /// The backend returned a 429 or equivalent.
    #[error("search rate-limited (retry after {retry_after:?})")]
    RateLimit {
        /// Server-suggested retry delay, when known.
        retry_after: Option<Duration>,
    },
    /// Underlying HTTP transport failed (DNS, connect, read, parse).
    #[error("search transport error: {0}")]
    Transport(#[from] reqwest::Error),
    /// The API rejected the credential (401/403). Treated as permanent: the run
    /// is dead-lettered.
    #[error("search authentication failed")]
    AuthFailed,
    /// Any other backend-reported failure. Surface verbatim in tracing.
    #[error("search backend error: {0}")]
    Other(String),
}

/// A backend that returns up to N search results for a query. [`Tavily`] is the
/// default implementation.
#[async_trait]
pub trait SearchBackend: Send + Sync + 'static {
    /// Search for `query` and return up to `limit` results. An implementation
    /// must treat `limit` as an upper bound and can return fewer results, for
    /// example when the backend itself returns fewer.
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>, SearchError>;
}
