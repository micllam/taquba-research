//! The tools, the settings and the output schema of the investigating phase, in
//! which a Rig agent searches and fetches pages for the gaps that the summaries
//! of a run leave.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use rig_agent::run::RunSpec;
use rig_core::completion::ToolDefinition;
use rig_core::message::ToolName;
use rig_core::schemars::schema_for;
use rig_core::tool::{ToolExecutionError, ToolOutput, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use taquba_workflow::jobs::{Job, JobError, JobRunner};
use taquba_workflow::{Error as WorkflowError, RunId, StepError};
use url::Url;

use crate::host::{ASK_USER, Plan, Record, RecordedCall, TurnTools, turn_group_id};
use crate::runner::AGENT_PREAMBLE;
use crate::state::{FetchedPage, Reply, Summary};
use crate::tool_job::{Artifact, ToolJob, fetch_group_id};

/// The name of the `web_search` tool.
const WEB_SEARCH: &str = "web_search";

/// The name of the `fetch_page` tool.
const FETCH_PAGE: &str = "fetch_page";

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

/// The definitions of the `web_search` and `fetch_page` tools, with chunks of
/// `chunk_chars` characters of a page.
pub(crate) fn definitions(chunk_chars: usize) -> Vec<Arc<ToolDefinition>> {
    let web_search = ToolDefinition::new(
        ToolName::new(WEB_SEARCH).expect("a valid tool name"),
        "Search the web. Returns the URL, title and snippet of each result.",
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "The search query." }
            },
            "required": ["query"]
        }),
    );
    let fetch_page = ToolDefinition::new(
        ToolName::new(FETCH_PAGE).expect("a valid tool name"),
        format!(
            "Fetch a web page. Returns its title and one chunk of at most {chunk_chars} \
             characters of its text, with the chunk number, the number of chunks and the \
             length of the whole text. Pass `chunk` to read a later chunk."
        ),
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
        }),
    );
    vec![Arc::new(web_search), Arc::new(fetch_page)]
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

/// The turn of the first successful `fetch_page` call of each URL of
/// `transcript`, counted from 1. A later call for the URL reads the page of
/// that call.
pub(crate) fn fetched(transcript: &[Record]) -> HashMap<Url, usize> {
    let mut turn = 0;
    let mut calls = BTreeMap::new();
    let mut fetched = HashMap::new();
    for record in transcript {
        match record {
            Record::Assistant { calls: made, .. } => {
                turn += 1;
                for call in made.iter().filter(|call| call.name == FETCH_PAGE) {
                    calls.insert(call.id.clone(), call.arguments.clone());
                }
            }
            Record::Results { results } => {
                for result in results.iter().filter(|result| !result.is_error) {
                    let url = calls.get(&result.id).and_then(|arguments| {
                        serde_json::from_value::<FetchPageArgs>(arguments.clone()).ok()
                    });
                    if let Some(args) = url {
                        fetched.entry(args.url).or_insert(turn);
                    }
                }
            }
            Record::User { .. } => {}
        }
    }
    fetched
}

#[derive(Debug, Deserialize)]
struct WebSearchArgs {
    query: String,
}

#[derive(Debug, Deserialize)]
struct FetchPageArgs {
    url: Url,
    /// The chunk to return, counted from 1, or the first chunk when absent.
    chunk: Option<usize>,
}

/// The tools of the investigating agent of a run. A `fetch_page` call for a
/// page that an earlier group fetched reads the page from that group, so a
/// later chunk does not fetch the page again.
pub(crate) struct ResearchTools {
    pub(crate) jobs: JobRunner,
    pub(crate) run_id: RunId,
    /// The pages of the fetch group of the run.
    pub(crate) fetching: HashSet<Url>,
    /// The length of a chunk in characters.
    pub(crate) chunk_chars: usize,
}

impl ResearchTools {
    /// The page `url` from an earlier group: the fetch group of the run, or the
    /// group of the turn of `transcript` that fetched it first.
    pub(crate) async fn page(
        &self,
        url: &Url,
        transcript: &[Record],
    ) -> Result<Option<FetchedPage>, StepError> {
        let group_id = if self.fetching.contains(url) {
            fetch_group_id(&self.run_id)?
        } else {
            match fetched(transcript).get(url) {
                Some(turn) => turn_group_id(&self.run_id, *turn)?,
                None => return Ok(None),
            }
        };
        read_page(&self.jobs, group_id, url).await
    }

    fn chunk(&self, page: &FetchedPage, chunk: Option<usize>) -> ToolResult {
        let chunk = chunk.unwrap_or(1);
        match page.chunk(chunk, self.chunk_chars) {
            Some(text) => json_result(&text),
            None => ToolResult::failed(ToolExecutionError::invalid_args(format!(
                "chunk {chunk} is past the last chunk of the text"
            ))),
        }
    }
}

impl TurnTools for ResearchTools {
    type Job = ToolJob;

    async fn plan(
        &self,
        call: &RecordedCall,
        earlier: &[Record],
    ) -> Result<Plan<ToolJob>, StepError> {
        let invalid = |e: serde_json::Error| {
            Plan::Result(ToolResult::failed(ToolExecutionError::invalid_args(
                format!("invalid arguments: {e}"),
            )))
        };
        match call.name.as_str() {
            WEB_SEARCH => Ok(
                match serde_json::from_value::<WebSearchArgs>(call.arguments.clone()) {
                    Ok(args) => Plan::Member(ToolJob::Search {
                        call: call.id.clone(),
                        query: args.query,
                    }),
                    Err(e) => invalid(e),
                },
            ),
            FETCH_PAGE => {
                let args = match serde_json::from_value::<FetchPageArgs>(call.arguments.clone()) {
                    Ok(args) => args,
                    Err(e) => return Ok(invalid(e)),
                };
                Ok(match self.page(&args.url, earlier).await? {
                    Some(page) => Plan::Result(self.chunk(&page, args.chunk)),
                    None => Plan::Member(ToolJob::Fetch { url: args.url }),
                })
            }
            name => Ok(Plan::Result(ToolResult::failed(
                ToolExecutionError::not_found(format!("unknown tool `{name}`")),
            ))),
        }
    }

    fn key(&self, call: &RecordedCall) -> Option<String> {
        let job = match call.name.as_str() {
            WEB_SEARCH => ToolJob::Search {
                call: call.id.clone(),
                query: String::new(),
            },
            FETCH_PAGE => ToolJob::Fetch {
                url: serde_json::from_value::<FetchPageArgs>(call.arguments.clone())
                    .ok()?
                    .url,
            },
            _ => return None,
        };
        job.idempotency_key()
    }

    fn render(&self, call: &RecordedCall, member: &Result<Artifact, JobError>) -> ToolResult {
        match member {
            Ok(Artifact::Results(results)) => json_result(results),
            Ok(Artifact::Page(page)) => {
                let chunk = serde_json::from_value::<FetchPageArgs>(call.arguments.clone())
                    .ok()
                    .and_then(|args| args.chunk);
                self.chunk(page, chunk)
            }
            Err(e) => ToolResult::failed(ToolExecutionError::other(e.message.clone())),
        }
    }
}

/// The result of a call that returned `value`, as JSON.
fn json_result(value: &impl Serialize) -> ToolResult {
    match serde_json::to_value(value) {
        Ok(value) => ToolResult::success(ToolOutput::json(value)),
        Err(e) => ToolResult::failed(ToolExecutionError::other(format!("encoding: {e}"))),
    }
}

/// The page `url` of the group `group_id`, or `None` for a group that does not
/// exist or did not fetch the page.
async fn read_page(
    jobs: &JobRunner,
    group_id: RunId,
    url: &Url,
) -> Result<Option<FetchedPage>, StepError> {
    let job = ToolJob::Fetch { url: url.clone() };
    let key = job.idempotency_key().expect("a fetch has a member key");
    match jobs.group::<ToolJob>(group_id).result(&key).await {
        Ok(member) => Ok(match member.result {
            Ok(Artifact::Page(page)) => Some(page),
            _ => None,
        }),
        Err(
            WorkflowError::GroupNotFound(_)
            | WorkflowError::MemberNotFound { .. }
            | WorkflowError::MemberNotSubmitted { .. },
        ) => Ok(None),
        Err(e) => Err(e.into()),
    }
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
    fn fetched_maps_each_successfully_fetched_url_to_the_turn_of_its_first_fetch() {
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
            Record::Assistant {
                text: String::new(),
                calls: vec![
                    call("4", "fetch_page", "https://example.com/later"),
                    call("5", "fetch_page", "https://example.com/ok"),
                ],
            },
            Record::Results {
                results: vec![
                    result("4", "fetch_page", false),
                    result("5", "fetch_page", false),
                ],
            },
        ];
        let mut urls: Vec<(String, usize)> = fetched(&transcript)
            .into_iter()
            .map(|(url, turn)| (url.to_string(), turn))
            .collect();
        urls.sort();
        assert_eq!(
            urls,
            [
                ("https://example.com/later".to_string(), 2),
                ("https://example.com/ok".to_string(), 1),
            ]
        );
    }
}
