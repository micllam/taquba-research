//! The tools, the hook and the output schema of the investigating phase, in
//! which a Rig agent searches and fetches pages for the gaps that the summaries
//! of a run leave.

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use rig_agent::agent::{
    AgentBuilder, AgentHook, DispatchAction, DispatchEvent, HookContext, OutcomeAction,
    OutcomeEvent,
};
use rig_agent::bus::Bus;
use rig_agent::completion::{PromptError, StructuredOutputError};
use rig_agent::tool::RegisteredTool;
use rig_agent::tool::server::ToolServer;
use rig_core::DynModel;
use rig_core::effect::{EffectKind, Outcome, model_key};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::operation::Completion;
use rig_core::serve::adapters::{ModelAdapter, ToolAdapter};
use rig_core::tool::PortableTool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use taquba_workflow::StepError;
use taquba_workflow::jobs::JobRunner;
use thiserror::Error;
use url::Url;

use crate::fetch_job::FetchPage;
use crate::journal::Journal;
use crate::runner::{
    AGENT_PREAMBLE, FETCH_JOIN_LEASE, LLM_CALL_TIMEOUT, SEARCH_CALL_TIMEOUT,
    classify_structured_err,
};
use crate::search::{SearchBackend, SearchError, SearchResult};
use crate::state::{Clarification, FetchedPage, Reply, Summary, TokenUsage};

/// Results per `web_search` call.
const SEARCH_LIMIT: usize = 5;

/// Prefix of the keys of the journal of the investigating step in the run memo.
pub(crate) const JOURNAL_PREFIX: &str = "investigating/";

/// Owner of the bus keys of the investigating agent, and the label of its
/// model.
const INVESTIGATOR: &str = "investigator";

/// Text of the `ask_user` result when the wait for a reply ran out.
const ASSUMED: &str = "The user did not reply in time. Proceed on the most likely \
     interpretation, and state that assumption in the summary of each finding it affects.";

/// Text of the skipped result of a second `ask_user` call.
const ONE_QUESTION: &str = "The run allows one question, and the user was already asked.";

/// Upper bound on an `ask_user` call, which returns a reply that exists.
const ASK_TIMEOUT: Duration = Duration::from_secs(5);

/// The investigating agent of a step: a model with the `web_search` and
/// `fetch_page` tools, and the `ask_user` tool unless `asking` is `Off`.
pub(crate) struct Investigator {
    pub(crate) model: DynModel<Completion>,
    pub(crate) search: Arc<dyn SearchBackend>,
    pub(crate) fetch: FetchPageTool,
    pub(crate) asking: Asking,
    pub(crate) max_tokens: Option<u64>,
}

/// Whether the agent can ask the user a question.
pub(crate) enum Asking {
    /// The agent cannot ask.
    Off,
    /// The agent can ask a question.
    Open,
    /// The agent asked this question, and the user replied or the wait ended.
    Replied(Clarification),
}

/// How a run of the investigating agent ended.
pub(crate) enum Investigated {
    /// The agent returned its findings.
    Finished {
        findings: Vec<Finding>,
        /// The URLs that `fetch_page` returned.
        fetched: HashSet<Url>,
        usage: TokenUsage,
    },
    /// The agent called `ask_user`, and the run must wait for a reply.
    Asked {
        /// The journal key of the call.
        key: String,
        question: String,
    },
}

impl Investigator {
    /// Run the agent over a bus whose model and tools are under `journal`, for
    /// at most `turns` model calls, adding the usage of each completion to
    /// `usage`. An agent that reaches `turns` returns an empty list of
    /// findings. An agent that asks a question stops at the call, and a later
    /// run over the same journal replays the agent up to the call.
    pub(crate) async fn run(
        self,
        journal: &Journal,
        prompt: String,
        turns: usize,
        usage: TokenUsage,
    ) -> Result<Investigated, StepError> {
        let (dispatcher, registrar, mut driver) = Bus::channel();
        let model_key = model_key(INVESTIGATOR);
        let model = ModelAdapter::new(INVESTIGATOR, self.model);
        driver
            .register(model_key.clone(), journal.wrap(model, LLM_CALL_TIMEOUT))
            .map_err(bus_step_err)?;
        let search = WebSearch {
            search: self.search,
        };
        let mut tools = ToolServer::new()
            .registered_tool(
                RegisteredTool::from_handler(
                    journal.wrap(ToolAdapter::new(search), SEARCH_CALL_TIMEOUT),
                )
                .map_err(bus_step_err)?,
            )
            .registered_tool(
                RegisteredTool::from_handler(
                    journal.wrap(ToolAdapter::new(self.fetch), FETCH_JOIN_LEASE),
                )
                .map_err(bus_step_err)?,
            );
        let (gate, ask) = match self.asking {
            Asking::Off => (None, None),
            Asking::Open => (
                Some(ClarificationGate::new(journal, None)),
                Some(AskUser { reply: None }),
            ),
            Asking::Replied(clarification) => (
                Some(ClarificationGate::new(journal, Some(clarification.key))),
                Some(AskUser {
                    reply: clarification.reply,
                }),
            ),
        };
        if let Some(ask) = ask {
            tools = tools.registered_tool(
                RegisteredTool::from_handler(journal.wrap(ToolAdapter::new(ask), ASK_TIMEOUT))
                    .map_err(bus_step_err)?,
            );
        }
        let observed = Observed::new(usage);
        let mut builder = AgentBuilder::over_bus(dispatcher, registrar, INVESTIGATOR, model_key)
            .preamble(AGENT_PREAMBLE)
            .add_hook(journal.call_keys())
            .add_hook(observed.clone());
        let pending = gate.as_ref().map(|gate| gate.pending.clone());
        if let Some(gate) = gate {
            builder = builder.add_hook(gate);
        }
        if let Some(max_tokens) = self.max_tokens {
            builder = builder.max_tokens(max_tokens);
        }
        let agent = builder.tool_server_handle(tools.run()).build();

        let run = agent
            .prompt_typed::<Investigation>(prompt)
            .max_turns(turns)
            .into_future();
        // The driver ends only after the agent drops its dispatcher.
        let result = tokio::select! {
            result = run => result,
            () = driver => return Err(StepError::transient("investigating: the bus closed")),
        };
        if let Some(fault) = journal.take_fault() {
            return Err(fault);
        }
        // The gate cancels the run at the `ask_user` call.
        if let Some((key, question)) = pending.and_then(|pending| lock(&pending).take()) {
            return Ok(Investigated::Asked { key, question });
        }
        let findings = match result {
            Ok(response) => response.output.findings,
            Err(StructuredOutputError::PromptError(PromptError::MaxTurnsError {
                max_turns,
                ..
            })) => {
                tracing::warn!(max_turns, "investigation ran out of model calls");
                Vec::new()
            }
            Err(e) => return Err(classify_structured_err(e)),
        };
        let observations = observed.take();
        Ok(Investigated::Finished {
            findings,
            fetched: observations.fetched,
            usage: observations.usage,
        })
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The hook that stops the `ask_user` call until the user replies. Without a
/// question, it records the first call and cancels the run. With a question, it
/// lets that call through and skips any other `ask_user` call.
struct ClarificationGate {
    journal: Journal,
    /// The journal key of the call that asked.
    asked: Option<String>,
    /// The key and the question of a call that the gate held.
    pending: Arc<Mutex<Option<(String, String)>>>,
}

impl ClarificationGate {
    fn new(journal: &Journal, asked: Option<String>) -> Self {
        Self {
            journal: journal.clone(),
            asked,
            pending: Arc::default(),
        }
    }
}

impl AgentHook for ClarificationGate {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        let (EffectKind::ToolCall { name, args }, Some(call_id)) = (event.kind, event.call_id)
        else {
            return DispatchAction::Proceed;
        };
        if name != AskUser::NAME {
            return DispatchAction::Proceed;
        }
        let key = self.journal.tool_key(call_id);
        match &self.asked {
            Some(asked) if *asked == key => DispatchAction::Proceed,
            Some(_) => DispatchAction::Deny(
                ErrorReport::new(ErrorKind::Denied, ONE_QUESTION).with_retryable(false),
            ),
            None => {
                let question = serde_json::from_str::<AskUserArgs>(args)
                    .map_or_else(|_| args.clone(), |args| args.question);
                *lock(&self.pending) = Some((key, question));
                DispatchAction::Deny(ErrorReport::new(
                    ErrorKind::Cancelled,
                    "waiting for the reply of the user",
                ))
            }
        }
    }
}

/// The `ask_user` tool, which returns the reply to the question of the run. The
/// [`ClarificationGate`] lets a call through only once the reply exists.
pub(crate) struct AskUser {
    reply: Option<Reply>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AskUserArgs {
    question: String,
}

/// An `ask_user` call that ran before the reply existed.
#[derive(Debug, Error)]
#[error("the reply of the user is missing")]
pub(crate) struct ReplyMissing;

impl PortableTool for AskUser {
    const NAME: &'static str = "ask_user";
    type Args = AskUserArgs;
    type Output = String;
    type Error = ReplyMissing;

    fn description(&self) -> String {
        "Ask the user one short question, and wait for the reply. Use it once at \
         most, when the sources conflict or the query is ambiguous in a way that \
         changes the findings."
            .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "question": { "type": "string", "description": "The question to the user." }
            },
            "required": ["question"]
        })
    }

    async fn call(&self, _args: AskUserArgs) -> Result<String, ReplyMissing> {
        match &self.reply {
            Some(Reply::Given(text)) => Ok(text.clone()),
            Some(Reply::Assumed) => Ok(ASSUMED.to_string()),
            None => Err(ReplyMissing),
        }
    }
}

/// Map a refused handler registration on a Rig bus to a permanent step error.
fn bus_step_err(report: ErrorReport) -> StepError {
    StepError::permanent(format!("bus registration refused: {}", report.message))
}

/// The `web_search` tool over the search backend of the run.
pub(crate) struct WebSearch {
    pub(crate) search: Arc<dyn SearchBackend>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct WebSearchArgs {
    query: String,
}

impl PortableTool for WebSearch {
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

    async fn call(&self, args: WebSearchArgs) -> Result<Vec<SearchResult>, SearchError> {
        self.search.search(&args.query, SEARCH_LIMIT).await
    }
}

/// The `fetch_page` tool, which fetches a page through a `FetchPage` job of the
/// run.
pub(crate) struct FetchPageTool {
    pub(crate) jobs: JobRunner,
    pub(crate) run_id: String,
    pub(crate) max_chars: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct FetchPageArgs {
    url: Url,
}

/// A failed `FetchPage` job, as the model reads it.
#[derive(Debug, Error)]
#[error("{0}")]
pub(crate) struct FetchPageError(String);

impl PortableTool for FetchPageTool {
    const NAME: &'static str = "fetch_page";
    type Args = FetchPageArgs;
    type Output = FetchedPage;
    type Error = FetchPageError;

    fn description(&self) -> String {
        "Fetch a web page. Returns its title and its text.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "url": { "type": "string", "description": "The absolute URL of the page." }
            },
            "required": ["url"]
        })
    }

    async fn call(&self, args: FetchPageArgs) -> Result<FetchedPage, FetchPageError> {
        fetch_page(&self.jobs, &self.run_id, args.url, self.max_chars)
            .await
            .map_err(FetchPageError)
    }
}

/// Fetch `url` through a `FetchPage` job. A job of the same run and URL returns
/// its recorded result.
pub(crate) async fn fetch_page(
    jobs: &JobRunner,
    run_id: &str,
    url: Url,
    max_chars: usize,
) -> Result<FetchedPage, String> {
    let job = FetchPage {
        run_id: run_id.to_string(),
        url,
        max_chars,
    };
    let handle = jobs
        .submit(job)
        .await
        .map_err(|e| format!("fetch submit: {e}"))?;
    handle.await.map_err(|e| e.to_string())
}

/// The hook that collects, from every completion and tool outcome of the agent,
/// the token usage and the URLs that `fetch_page` returned. A replayed outcome
/// passes the hook like a fresh one.
#[derive(Clone)]
pub(crate) struct Observed(Arc<Mutex<Observations>>);

#[derive(Default)]
pub(crate) struct Observations {
    pub(crate) fetched: HashSet<Url>,
    pub(crate) usage: TokenUsage,
}

impl Observed {
    /// A hook that adds the usage of each completion to `usage`.
    pub(crate) fn new(usage: TokenUsage) -> Self {
        Self(Arc::new(Mutex::new(Observations {
            fetched: HashSet::new(),
            usage,
        })))
    }

    pub(crate) fn take(&self) -> Observations {
        std::mem::take(&mut *self.0.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

impl AgentHook for Observed {
    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        let mut observations = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        match (event.kind, event.outcome) {
            (_, Ok(Outcome::Completion(response))) => {
                crate::runner::record_usage(&mut observations.usage, &response.usage);
            }
            (EffectKind::ToolCall { name, args }, Ok(Outcome::ToolResult { result }))
                if name == FetchPageTool::NAME && result.is_success() =>
            {
                if let Ok(args) = serde_json::from_str::<FetchPageArgs>(args) {
                    observations.fetched.insert(args.url);
                }
            }
            _ => {}
        }
        OutcomeAction::Proceed
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use rig_core::completion::Usage;
    use rig_core::test_utils::{MockCompletionModel, MockTurn};
    use taquba::object_store::ObjectStore;
    use taquba::object_store::memory::InMemory;
    use taquba::{LeaseHandle, Queue};
    use taquba_workflow::{Memo, MemoStore, RunId, StepErrorKind};

    use crate::journal::{JournalEntry, read_entries};
    use crate::store::{WORKFLOW_MEMO_PREFIX, journal_entries};

    const FINDINGS: &str = r#"{"findings":[{"url":"https://example.com/gap",
        "summary":"The page fills the gap.","relevance":0.8}]}"#;

    #[derive(Default)]
    struct CountingSearch(AtomicUsize);

    #[async_trait]
    impl SearchBackend for CountingSearch {
        async fn search(
            &self,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<SearchResult>, SearchError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(vec![SearchResult {
                url: "https://example.com/gap".parse().unwrap(),
                title: "Gap".to_string(),
                snippet: "fills the gap".to_string(),
            }])
        }
    }

    struct StalledSearch;

    #[async_trait]
    impl SearchBackend for StalledSearch {
        async fn search(
            &self,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<SearchResult>, SearchError> {
            std::future::pending().await
        }
    }

    fn step_memo() -> Memo {
        MemoStore::new(Arc::new(InMemory::new()), "test-memo")
            .new_memo(&RunId::new("run").unwrap(), 4)
    }

    /// An investigator over `model` and `search`, with a job runner whose
    /// worker never starts.
    async fn investigator(
        model: &MockCompletionModel,
        search: Arc<dyn SearchBackend>,
        asking: Asking,
    ) -> Investigator {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let queue = Arc::new(Queue::open(store.clone(), "q").await.unwrap());
        let jobs = JobRunner::builder(queue, store)
            .register::<FetchPage>()
            .build();
        Investigator {
            model: model.clone().into(),
            search,
            fetch: FetchPageTool {
                jobs,
                run_id: "run".to_string(),
                max_chars: 100,
            },
            asking,
            max_tokens: None,
        }
    }

    async fn run_asking(
        model: &MockCompletionModel,
        search: Arc<dyn SearchBackend>,
        memo: &Memo,
        turns: usize,
        asking: Asking,
    ) -> Result<Investigated, StepError> {
        let journal = Journal::new(memo.clone(), JOURNAL_PREFIX, LeaseHandle::detached());
        investigator(model, search, asking)
            .await
            .run(&journal, "query".to_string(), turns, TokenUsage::default())
            .await
    }

    async fn run(
        model: &MockCompletionModel,
        search: Arc<dyn SearchBackend>,
        memo: &Memo,
        turns: usize,
    ) -> Result<Investigated, StepError> {
        run_asking(model, search, memo, turns, Asking::Off).await
    }

    /// The findings and the usage of a finished run.
    fn finished(investigated: Investigated) -> (Vec<Finding>, TokenUsage) {
        match investigated {
            Investigated::Finished {
                findings, usage, ..
            } => (findings, usage),
            Investigated::Asked { question, .. } => panic!("the agent asked: {question}"),
        }
    }

    fn usage(input_tokens: u64) -> Usage {
        Usage {
            input_tokens: Some(input_tokens),
            ..Usage::default()
        }
    }

    #[tokio::test]
    async fn investigator_replays_recorded_outcomes() {
        let memo = step_memo();
        let model = MockCompletionModel::from_turns([
            MockTurn::tool_call("call_1", "web_search", json!({ "query": "gaps" }))
                .with_usage(usage(10)),
            MockTurn::text(FINDINGS).with_usage(usage(20)),
        ]);
        let search = Arc::new(CountingSearch::default());
        let (_, first_usage) = finished(run(&model, search.clone(), &memo, 4).await.unwrap());

        // A model without a script fails each call, so the retry must replay.
        let replay_model = MockCompletionModel::from_turns(Vec::<MockTurn>::new());
        let replay_search = Arc::new(CountingSearch::default());
        let (findings, usage) = finished(
            run(&replay_model, replay_search.clone(), &memo, 4)
                .await
                .unwrap(),
        );

        assert_eq!(model.request_count(), 2);
        assert_eq!(search.0.load(Ordering::SeqCst), 1);
        assert_eq!(replay_model.request_count(), 0);
        assert_eq!(replay_search.0.load(Ordering::SeqCst), 0);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].url, "https://example.com/gap");
        assert_eq!(usage.input_tokens, 30);
        assert_eq!(usage, first_usage);
    }

    #[tokio::test]
    async fn investigator_stops_at_a_question_and_replays_up_to_it_after_the_reply() {
        let memo = step_memo();
        let asking = MockCompletionModel::from_turns([MockTurn::tool_call(
            "call_1",
            "ask_user",
            json!({ "question": "Which edition?" }),
        )]);
        let Investigated::Asked { key, question } = run_asking(
            &asking,
            Arc::new(CountingSearch::default()),
            &memo,
            4,
            Asking::Open,
        )
        .await
        .unwrap() else {
            panic!("the agent must stop at the question");
        };
        assert_eq!(question, "Which edition?");

        let replied = Clarification {
            key,
            question,
            asked_at: chrono::Utc::now(),
            reply: Some(Reply::Given("the 2024 edition".to_string())),
        };
        let model = MockCompletionModel::from_turns([MockTurn::text(FINDINGS)]);
        let (findings, _) = finished(
            run_asking(
                &model,
                Arc::new(CountingSearch::default()),
                &memo,
                4,
                Asking::Replied(replied),
            )
            .await
            .unwrap(),
        );
        // The completion that asked replays, so the model receives one request,
        // with the reply as the result of `ask_user`.
        assert_eq!(model.request_count(), 1);
        assert!(format!("{:?}", model.requests()[0]).contains("the 2024 edition"));
        assert_eq!(findings.len(), 1);
    }

    #[tokio::test]
    async fn investigator_skips_a_second_question() {
        let memo = step_memo();
        let asking = MockCompletionModel::from_turns([MockTurn::tool_call(
            "call_1",
            "ask_user",
            json!({ "question": "Which edition?" }),
        )]);
        let Investigated::Asked { key, question } = run_asking(
            &asking,
            Arc::new(CountingSearch::default()),
            &memo,
            4,
            Asking::Open,
        )
        .await
        .unwrap() else {
            panic!("the agent must stop at the question");
        };

        let replied = Clarification {
            key,
            question,
            asked_at: chrono::Utc::now(),
            reply: Some(Reply::Assumed),
        };
        let model = MockCompletionModel::from_turns([
            MockTurn::tool_call("call_2", "ask_user", json!({ "question": "Which year?" })),
            MockTurn::text(FINDINGS),
        ]);
        finished(
            run_asking(
                &model,
                Arc::new(CountingSearch::default()),
                &memo,
                4,
                Asking::Replied(replied),
            )
            .await
            .unwrap(),
        );
        assert!(format!("{:?}", model.requests()[1]).contains(ONE_QUESTION));
    }

    #[tokio::test]
    async fn journal_entries_list_completions_and_their_tool_calls() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let run_id = RunId::new("run").unwrap();
        // The runtime's run memo, which `store::journal_entries` reads.
        let memo = MemoStore::new(object_store.clone(), WORKFLOW_MEMO_PREFIX).new_run_memo(&run_id);
        let model = MockCompletionModel::from_turns([
            MockTurn::tool_call("call_1", "web_search", json!({ "query": "gaps" })),
            MockTurn::text(FINDINGS),
        ]);
        run(&model, Arc::new(CountingSearch::default()), &memo, 4)
            .await
            .unwrap();

        let entries = journal_entries(object_store.clone(), &run_id).await;
        assert_eq!(
            entries.unwrap(),
            vec![
                JournalEntry::Completion,
                JournalEntry::ToolCall {
                    name: "web_search".to_string(),
                    args: r#"{"query":"gaps"}"#.to_string(),
                    recorded: true,
                },
                JournalEntry::Completion,
            ]
        );
        let other = journal_entries(object_store, &RunId::new("other").unwrap()).await;
        assert!(other.unwrap().is_empty());
    }

    #[tokio::test]
    async fn investigator_calls_the_model_again_after_a_failed_call() {
        let memo = step_memo();
        let failing = MockCompletionModel::from_turns([MockTurn::error("overloaded")]);
        let err = run(&failing, Arc::new(CountingSearch::default()), &memo, 4)
            .await
            .err()
            .expect("a failed completion must fail the step");
        assert!(matches!(err.kind, StepErrorKind::Transient), "{err:?}");

        let model = MockCompletionModel::from_turns([MockTurn::text(FINDINGS)]);
        let (findings, _) = finished(
            run(&model, Arc::new(CountingSearch::default()), &memo, 4)
                .await
                .unwrap(),
        );
        assert_eq!(model.request_count(), 1);
        assert_eq!(findings.len(), 1);
    }

    #[tokio::test]
    async fn investigator_returns_empty_findings_after_its_model_calls() {
        let model = MockCompletionModel::from_turns([MockTurn::tool_call(
            "call_1",
            "web_search",
            json!({ "query": "gaps" }),
        )]);
        let (findings, _) = finished(
            run(&model, Arc::new(CountingSearch::default()), &step_memo(), 1)
                .await
                .unwrap(),
        );
        assert!(findings.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn investigator_fails_a_tool_call_after_its_bound() {
        let model = MockCompletionModel::from_turns([
            MockTurn::tool_call("call_1", "web_search", json!({ "query": "gaps" })),
            MockTurn::text(FINDINGS),
        ]);
        let memo = step_memo();
        run(&model, Arc::new(StalledSearch), &memo, 4)
            .await
            .unwrap();
        let follow_up = format!("{:?}", model.requests()[1]);
        assert!(
            follow_up.contains("timed out after 60s"),
            "the model must read the timeout: {follow_up}"
        );
        let entries = read_entries(&memo, JOURNAL_PREFIX).await.unwrap();
        assert!(matches!(
            entries[1],
            JournalEntry::ToolCall {
                recorded: false,
                ..
            }
        ));
    }
}
