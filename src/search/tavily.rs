//! [Tavily](https://tavily.com) search backend. Default for v0.1 because
//! of its generous free tier and clean POST-JSON API.

use std::env;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use url::Url;

use super::{EnvError, SearchBackend, SearchError, SearchResult};

const TAVILY_ENDPOINT: &str = "https://api.tavily.com/search";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const API_KEY_ENV: &str = "TAVILY_API_KEY";
const MAX_RESULTS_PER_REQUEST: usize = 20;

/// Tavily-backed [`SearchBackend`] implementation.
#[derive(Debug, Clone)]
pub struct Tavily {
    api_key: String,
    client: reqwest::Client,
    endpoint: String,
}

impl Tavily {
    /// Build a Tavily client with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(DEFAULT_TIMEOUT)
            .build()
            .expect("reqwest client builder cannot fail with default config");
        Self {
            api_key: api_key.into(),
            client,
            endpoint: TAVILY_ENDPOINT.to_string(),
        }
    }

    /// Read the API key from the `TAVILY_API_KEY` environment variable.
    pub fn from_env() -> Result<Self, EnvError> {
        let key = env::var(API_KEY_ENV).map_err(|_| EnvError::Missing(API_KEY_ENV))?;
        if key.trim().is_empty() {
            return Err(EnvError::Empty(API_KEY_ENV));
        }
        Ok(Self::new(key))
    }

    /// Override the API endpoint. Mainly useful for testing against a
    /// recorded fixture or a self-hosted proxy.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }
}

#[async_trait]
impl SearchBackend for Tavily {
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>, SearchError> {
        let body = TavilyRequest {
            api_key: &self.api_key,
            query,
            max_results: limit.min(MAX_RESULTS_PER_REQUEST),
            search_depth: "basic",
            include_answer: false,
            include_raw_content: false,
        };
        let resp = self.client.post(&self.endpoint).json(&body).send().await?;

        match resp.status() {
            StatusCode::OK => {}
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                return Err(SearchError::AuthFailed);
            }
            StatusCode::TOO_MANY_REQUESTS => {
                let retry_after = resp
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok())
                    .map(Duration::from_secs);
                return Err(SearchError::RateLimit { retry_after });
            }
            other => {
                let snippet = resp.text().await.unwrap_or_default();
                let truncated: String = snippet.chars().take(256).collect();
                return Err(SearchError::Other(format!(
                    "tavily returned HTTP {other}: {truncated}"
                )));
            }
        }

        let parsed: TavilyResponse = resp.json().await?;
        let results = parsed
            .results
            .into_iter()
            .filter_map(|r| {
                let url = Url::parse(&r.url).ok()?;
                Some(SearchResult {
                    url,
                    title: r.title.unwrap_or_default(),
                    snippet: r.content.unwrap_or_default(),
                })
            })
            .collect();
        Ok(results)
    }
}

#[derive(Serialize)]
struct TavilyRequest<'a> {
    api_key: &'a str,
    query: &'a str,
    max_results: usize,
    search_depth: &'static str,
    include_answer: bool,
    include_raw_content: bool,
}

#[derive(Deserialize)]
struct TavilyResponse {
    #[serde(default)]
    results: Vec<TavilyResult>,
}

#[derive(Deserialize)]
struct TavilyResult {
    url: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    content: Option<String>,
}
