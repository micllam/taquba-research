//! The tools, the settings and the output schema of the investigating phase, in
//! which a Rig agent searches and fetches pages for the gaps that the summaries
//! of a run leave.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use rig_agent::run::RunSpec;
use rig_agent::tool::ToolSet;
use rig_core::completion::ToolDefinition;
use rig_core::message::ToolName;
use rig_core::schemars::schema_for;
use rig_core::tool::{Tool, ToolContext};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use taquba_workflow::jobs::JobRunner;
use thiserror::Error;
use url::Url;

use crate::fetch_job::FetchPage;
use crate::host::{ASK_USER, Record};
use crate::runner::{AGENT_PREAMBLE, FETCH_JOIN_LEASE, SEARCH_CALL_TIMEOUT};
use crate::search::{SearchBackend, SearchError, SearchResult};
use crate::state::{FetchedPage, Reply, Summary, TextChunk};

/// Results per `web_search` call.
const SEARCH_LIMIT: usize = 5;

/// The name of the tool through which the agent returns its findings.
const OUTPUT_TOOL: &str = "submit_findings";

/// Text of the `ask_user` result when the wait for a reply ran out.
const ASSUMED: &str = "The user did not reply in time. Proceed on the most likely \
     interpretation, and state that assumption in the summary of each finding it affects.";

/// The request settings of the agent, for at most `turns` model calls.
pub(crate) fn spec(turns: usize, max_tokens: Option<u64>) -> RunSpec {
    let mut spec = RunSpec::new();
    spec.preamble = Some(AGENT_PREAMBLE.to_string());
    spec.max_turns = Some(turns);
    spec.max_tokens = max_tokens;
    spec.output_schema = Some(schema_for!(Investigation));
    spec.output_tool_name = Some(OUTPUT_TOOL.to_string());
    spec
}

/// The `web_search` and `fetch_page` tools of the agent.
pub(crate) fn tools(search: Arc<dyn SearchBackend>, fetch: FetchPageTool) -> ToolSet {
    let mut tools = ToolSet::default();
    tools.add_tool(WebSearch { search });
    tools.add_tool(fetch);
    tools
}

/// The time bound of a call of the tool `name`.
pub(crate) fn bound(name: &str) -> Duration {
    match name {
        FetchPageTool::NAME => FETCH_JOIN_LEASE,
        _ => SEARCH_CALL_TIMEOUT,
    }
}

/// The definition of the `ask_user` tool.
pub(crate) fn ask_user() -> ToolDefinition {
    let name = ToolName::new(ASK_USER).expect("a valid tool name");
    ToolDefinition::new(
        name,
        "Ask the user one short question, and wait for the reply. Use it once at \
         most, when the sources conflict or the query is ambiguous in a way that \
         changes the findings.",
        json!({
            "type": "object",
            "properties": {
                "question": { "type": "string", "description": "The question to the user." }
            },
            "required": ["question"]
        }),
    )
}

/// The text of the `ask_user` result for `reply`.
pub(crate) fn reply_text(reply: &Reply) -> String {
    match reply {
        Reply::Given(text) => text.clone(),
        Reply::Assumed => ASSUMED.to_string(),
    }
}

/// The URLs that a successful `fetch_page` call of `transcript` returned.
pub(crate) fn fetched(transcript: &[Record]) -> HashSet<Url> {
    let mut calls = BTreeMap::new();
    let mut fetched = HashSet::new();
    for record in transcript {
        match record {
            Record::Assistant { calls: made, .. } => {
                for call in made.iter().filter(|call| call.name == FetchPageTool::NAME) {
                    calls.insert(call.id.clone(), call.arguments.clone());
                }
            }
            Record::Results { results } => {
                for result in results.iter().filter(|result| !result.is_error) {
                    let url = calls.get(&result.id).and_then(|arguments| {
                        serde_json::from_value::<FetchPageArgs>(arguments.clone()).ok()
                    });
                    if let Some(args) = url {
                        fetched.insert(args.url);
                    }
                }
            }
            Record::User { .. } => {}
        }
    }
    fetched
}

/// The `web_search` tool over the search backend of the run.
pub(crate) struct WebSearch {
    pub(crate) search: Arc<dyn SearchBackend>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct WebSearchArgs {
    query: String,
}

impl Tool for WebSearch {
    const NAME: &'static str = "web_search";
    type Args = WebSearchArgs;
    type Output = Vec<SearchResult>;
    type Error = SearchError;

    fn description(&self) -> String {
        "Search the web. Returns the URL, title and snippet of each result.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "The search query." }
            },
            "required": ["query"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: WebSearchArgs,
    ) -> Result<Vec<SearchResult>, SearchError> {
        self.search.search(&args.query, SEARCH_LIMIT).await
    }
}

/// The `fetch_page` tool, which fetches a page through a `FetchPage` job of the
/// run and returns one chunk of its text. The recorded result of the job keeps
/// the whole text, so a call for a later chunk does not fetch the page again.
pub(crate) struct FetchPageTool {
    pub(crate) jobs: JobRunner,
    pub(crate) run_id: String,
    /// The length of a chunk in characters.
    pub(crate) chunk_chars: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct FetchPageArgs {
    url: Url,
    /// The chunk to return, counted from 1, or the first chunk when absent.
    #[serde(default)]
    chunk: Option<usize>,
}

/// A failed `FetchPage` job, or a chunk past the last one, as the model reads
/// it.
#[derive(Debug, Error)]
#[error("{0}")]
pub(crate) struct FetchPageError(String);

impl Tool for FetchPageTool {
    const NAME: &'static str = "fetch_page";
    type Args = FetchPageArgs;
    type Output = TextChunk;
    type Error = FetchPageError;

    fn description(&self) -> String {
        format!(
            "Fetch a web page. Returns its title and one chunk of at most {} characters of \
             its text, with the chunk number, the number of chunks and the length of the \
             whole text. Pass `chunk` to read a later chunk.",
            self.chunk_chars
        )
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "The absolute URL of the page." },
                "chunk": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "The chunk of the text to return, 1 by default."
                }
            },
            "required": ["url"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: FetchPageArgs,
    ) -> Result<TextChunk, FetchPageError> {
        let fetched = fetch_page(&self.jobs, &self.run_id, args.url)
            .await
            .map_err(FetchPageError)?;
        let chunk = args.chunk.unwrap_or(1);
        fetched.chunk(chunk, self.chunk_chars).ok_or_else(|| {
            FetchPageError(format!("chunk {chunk} is past the last chunk of the text"))
        })
    }
}

/// Fetch `url` through a `FetchPage` job, with its whole text. A job of the
/// same run and URL returns its recorded result.
pub(crate) async fn fetch_page(
    jobs: &JobRunner,
    run_id: &str,
    url: Url,
) -> Result<FetchedPage, String> {
    let job = FetchPage {
        run_id: run_id.to_string(),
        url,
    };
    let handle = jobs
        .submit(job)
        .await
        .map_err(|e| format!("fetch submit: {e}"))?;
    handle.await.map_err(|e| e.to_string())
}

/// The structured output of the investigating agent.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub(crate) struct Investigation {
    /// One finding per fetched page that fills a gap.
    pub(crate) findings: Vec<Finding>,
}

/// A page that the agent fetched, with its summary.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub(crate) struct Finding {
    /// The URL passed to `fetch_page`.
    pub(crate) url: String,
    /// A summary of what the page says about the query.
    pub(crate) summary: String,
    /// Relevance to the query, from 0.0 to 1.0.
    pub(crate) relevance: f32,
}

/// The prompt of the investigating agent, listing each summarised source.
pub(crate) fn prompt(
    query: &str,
    summaries: &BTreeMap<Url, Summary>,
    turns: usize,
    can_ask: bool,
) -> String {
    let asking = if can_ask {
        " When the sources conflict, or the query is ambiguous in a way that \
         changes the findings, call `ask_user` once with one short question."
    } else {
        ""
    };
    let mut sources = String::new();
    for (url, summary) in summaries {
        sources.push_str(&format!(
            "- {title} ({url}): {text}\n",
            title = summary.title,
            text = summary.text,
        ));
    }
    if sources.is_empty() {
        sources.push_str("(no sources gathered)\n");
    }
    format!(
        "You are a research investigator. The user is investigating:\n\n  {query}\n\n\
         The sources gathered so far are listed below with their summaries. \
         Identify the important gaps: aspects of the query that the sources do \
         not cover, or on which they conflict. Use `web_search` to find pages \
         that address the gaps and `fetch_page` to read them. You can make at \
         most {turns} model calls, the final answer included.{asking}\n\n\
         Return one finding for each fetched page that fills a gap: its URL, a \
         2-4 sentence summary of what is relevant to the query, and a relevance \
         score from 0.0 (off-topic) to 1.0 (highly relevant). Never return a \
         page you did not fetch or a source from the list. Return no findings \
         when the sources already cover the query.\n\n\
         Sources:\n{sources}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::{RecordedCall, RecordedResult};

    fn call(id: &str, name: &str, url: &str) -> RecordedCall {
        RecordedCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: json!({ "url": url }),
        }
    }

    fn result(id: &str, name: &str, is_error: bool) -> RecordedResult {
        RecordedResult {
            id: id.to_string(),
            name: name.to_string(),
            text: String::new(),
            is_error,
        }
    }

    #[test]
    fn fetched_lists_the_urls_of_successful_fetch_page_calls() {
        let transcript = [
            Record::User {
                text: "query".to_string(),
            },
            Record::Assistant {
                text: String::new(),
                calls: vec![
                    call("1", "fetch_page", "https://example.com/ok"),
                    call("2", "fetch_page", "https://example.com/failed"),
                    call("3", "web_search", "https://example.com/search"),
                ],
            },
            Record::Results {
                results: vec![
                    result("1", "fetch_page", false),
                    result("2", "fetch_page", true),
                    result("3", "web_search", false),
                ],
            },
        ];
        let urls: Vec<String> = fetched(&transcript).iter().map(Url::to_string).collect();
        assert_eq!(urls, ["https://example.com/ok"]);
    }
}
