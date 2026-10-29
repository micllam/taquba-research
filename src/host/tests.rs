use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use rig_core::completion::Usage;
use rig_core::message::{AssistantContent, ToolCall, ToolFunction, ToolName};
use rig_core::test_utils::{MockCompletionModel, MockTurn};
use serde_json::json;
use taquba::object_store::memory::InMemory;
use taquba_workflow::{MemoStore, RunId, StepErrorKind};

use super::*;
use crate::investigate::{self, WebSearch};
use crate::search::{SearchBackend, SearchError, SearchResult};

#[derive(Default)]
struct CountingSearch(AtomicUsize);

#[async_trait]
impl SearchBackend for CountingSearch {
    async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<SearchResult>, SearchError> {
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
    async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<SearchResult>, SearchError> {
        std::future::pending().await
    }
}

fn step_memo(step_number: u32) -> Memo {
    MemoStore::new(Arc::new(InMemory::new()), "test-memo")
        .new_memo(&RunId::new("run").unwrap(), step_number)
}

fn spec(turns: usize) -> RunSpec {
    let mut spec = RunSpec::new();
    spec.max_turns = Some(turns);
    spec
}

fn tools(search: Arc<dyn SearchBackend>) -> ToolSet {
    let mut tools = ToolSet::default();
    tools.add_tool(WebSearch { search });
    tools
}

fn search_call(id: &str) -> MockTurn {
    MockTurn::tool_call(id, "web_search", json!({ "query": "gaps" }))
}

fn call(id: &str, name: &str, arguments: serde_json::Value) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall::from_wire(
        id,
        ToolFunction::new(ToolName::new(name).unwrap(), arguments),
    ))
}

fn ask_call(id: &str, question: &str) -> MockTurn {
    MockTurn::tool_call(id, ASK_USER, json!({ "question": question }))
}

/// A step of a host over `model` and `tools`, with `ask_user` unless `ask` is
/// `Off`.
async fn step(
    model: &MockCompletionModel,
    spec: &RunSpec,
    tools: &ToolSet,
    ask: Ask,
    memo: &Memo,
    state: &mut AgentState,
    usage: &mut TokenUsage,
) -> Result<TurnEnd, StepError> {
    let mut definitions = tools.tool_definitions();
    if !matches!(ask, Ask::Off) {
        definitions.push(Arc::new(investigate::ask_user()));
    }
    let lease = LeaseHandle::detached();
    let host = Host {
        model: model.clone().into(),
        spec,
        tools,
        definitions,
        bound: investigate::bound,
        ask,
        memo,
        lease: &lease,
    };
    host.step(state, "query", usage).await
}

/// The debug form of the request `index` that `model` received.
fn request(model: &MockCompletionModel, index: usize) -> String {
    format!("{:?}", model.requests()[index])
}

#[tokio::test]
async fn a_step_makes_one_model_call_and_its_tool_calls() {
    let model = MockCompletionModel::from_turns([search_call("call_1"), MockTurn::text("done")]);
    let search = Arc::new(CountingSearch::default());
    let tools = tools(search.clone());
    let (spec, mut state, mut usage) = (spec(4), AgentState::default(), TokenUsage::default());

    let first = step(
        &model,
        &spec,
        &tools,
        Ask::Off,
        &step_memo(1),
        &mut state,
        &mut usage,
    )
    .await;
    assert_eq!(first.unwrap(), TurnEnd::Continue);
    assert_eq!(model.request_count(), 1);
    assert_eq!(search.0.load(Ordering::SeqCst), 1);

    let second = step(
        &model,
        &spec,
        &tools,
        Ask::Off,
        &step_memo(2),
        &mut state,
        &mut usage,
    )
    .await;
    let output = "done".to_string();
    assert_eq!(second.unwrap(), TurnEnd::Done { output });
    let roles: Vec<&str> = state
        .transcript
        .iter()
        .map(|record| match record {
            Record::User { .. } => "user",
            Record::Assistant { .. } => "assistant",
            Record::Results { .. } => "results",
        })
        .collect();
    assert_eq!(roles, ["user", "assistant", "results", "assistant"]);
}

#[tokio::test]
async fn a_redelivered_step_reuses_the_model_reply_and_the_tool_results() {
    let memo = step_memo(1);
    let model = MockCompletionModel::from_turns([
        search_call("call_1").with_usage(Usage::new().input_tokens(10))
    ]);
    let search = Arc::new(CountingSearch::default());
    let spec = spec(4);
    let mut first_state = AgentState::default();
    let mut first_usage = TokenUsage::default();
    let tools_first = tools(search.clone());
    step(
        &model,
        &spec,
        &tools_first,
        Ask::Off,
        &memo,
        &mut first_state,
        &mut first_usage,
    )
    .await
    .unwrap();

    // A model without a script fails each call, so the redelivery must reuse.
    let replay = MockCompletionModel::from_turns(Vec::<MockTurn>::new());
    let replay_search = Arc::new(CountingSearch::default());
    let replay_tools = tools(replay_search.clone());
    let (mut state, mut usage) = (AgentState::default(), TokenUsage::default());
    let end = step(
        &replay,
        &spec,
        &replay_tools,
        Ask::Off,
        &memo,
        &mut state,
        &mut usage,
    )
    .await;
    assert_eq!(end.unwrap(), TurnEnd::Continue);
    assert_eq!(replay.request_count(), 0);
    assert_eq!(replay_search.0.load(Ordering::SeqCst), 0);
    assert_eq!(state.transcript, first_state.transcript);
    assert_eq!(usage.input_tokens, 10);
}

#[tokio::test]
async fn a_question_waits_after_the_other_calls_and_the_reply_resolves_it() {
    let model = MockCompletionModel::from_turns([
        MockTurn::from_contents([
            call("call_1", "web_search", json!({ "query": "gaps" })),
            call("call_2", ASK_USER, json!({ "question": "Which edition?" })),
        ]),
        MockTurn::text("done"),
    ]);
    let search = Arc::new(CountingSearch::default());
    let tools = tools(search.clone());
    let (spec, mut state, mut usage) = (spec(4), AgentState::default(), TokenUsage::default());

    let asked = step(
        &model,
        &spec,
        &tools,
        Ask::Open,
        &step_memo(1),
        &mut state,
        &mut usage,
    )
    .await;
    let question = "Which edition?".to_string();
    assert_eq!(asked.unwrap(), TurnEnd::Wait { question });
    assert_eq!(search.0.load(Ordering::SeqCst), 1);

    let reply = Ask::Replied("the 2024 edition".to_string());
    let after = step(
        &model,
        &spec,
        &tools,
        reply,
        &step_memo(2),
        &mut state,
        &mut usage,
    )
    .await;
    let output = "done".to_string();
    assert_eq!(after.unwrap(), TurnEnd::Done { output });
    assert!(request(&model, 1).contains("the 2024 edition"));
}

#[tokio::test]
async fn a_second_question_receives_a_skipped_result() {
    let model = MockCompletionModel::from_turns([
        ask_call("call_1", "Which edition?"),
        ask_call("call_2", "Which year?"),
        MockTurn::text("done"),
    ]);
    let tools = tools(Arc::new(CountingSearch::default()));
    let (spec, mut state, mut usage) = (spec(4), AgentState::default(), TokenUsage::default());
    step(
        &model,
        &spec,
        &tools,
        Ask::Open,
        &step_memo(1),
        &mut state,
        &mut usage,
    )
    .await
    .unwrap();
    let reply = Ask::Replied("the 2024 edition".to_string());
    let after = step(
        &model,
        &spec,
        &tools,
        reply,
        &step_memo(2),
        &mut state,
        &mut usage,
    )
    .await;
    assert_eq!(after.unwrap(), TurnEnd::Continue);
    let reply = Ask::Replied("the 2024 edition".to_string());
    step(
        &model,
        &spec,
        &tools,
        reply,
        &step_memo(3),
        &mut state,
        &mut usage,
    )
    .await
    .unwrap();
    assert!(request(&model, 2).contains(ONE_QUESTION));
}

#[tokio::test]
async fn a_call_to_a_tool_without_a_definition_fails_the_step_permanently() {
    let model = MockCompletionModel::from_turns([ask_call("call_1", "Which edition?")]);
    let tools = tools(Arc::new(CountingSearch::default()));
    let (spec, mut state, mut usage) = (spec(4), AgentState::default(), TokenUsage::default());
    let end = step(
        &model,
        &spec,
        &tools,
        Ask::Off,
        &step_memo(1),
        &mut state,
        &mut usage,
    )
    .await;
    let err = end.expect_err("an undefined tool must fail the step");
    assert!(matches!(err.kind, StepErrorKind::Permanent), "{err:?}");
}

#[tokio::test]
async fn a_failed_model_call_fails_the_step_transiently() {
    let model = MockCompletionModel::from_turns([MockTurn::error("overloaded")]);
    let tools = tools(Arc::new(CountingSearch::default()));
    let (spec, mut state, mut usage) = (spec(4), AgentState::default(), TokenUsage::default());
    let memo = step_memo(1);
    let err = step(
        &model,
        &spec,
        &tools,
        Ask::Off,
        &memo,
        &mut state,
        &mut usage,
    )
    .await
    .expect_err("a failed call must fail the step");
    assert!(matches!(err.kind, StepErrorKind::Transient), "{err:?}");

    // The failure is not in the memo, so the retry calls the model.
    let retry = MockCompletionModel::from_turns([MockTurn::text("done")]);
    let end = step(
        &retry,
        &spec,
        &tools,
        Ask::Off,
        &memo,
        &mut state,
        &mut usage,
    )
    .await;
    let output = "done".to_string();
    assert_eq!(end.unwrap(), TurnEnd::Done { output });
}

#[tokio::test]
async fn a_run_past_its_model_calls_ends_out_of_turns() {
    let model = MockCompletionModel::from_turns([search_call("call_1")]);
    let tools = tools(Arc::new(CountingSearch::default()));
    let (spec, mut state, mut usage) = (spec(1), AgentState::default(), TokenUsage::default());
    step(
        &model,
        &spec,
        &tools,
        Ask::Off,
        &step_memo(1),
        &mut state,
        &mut usage,
    )
    .await
    .unwrap();
    let end = step(
        &model,
        &spec,
        &tools,
        Ask::Off,
        &step_memo(2),
        &mut state,
        &mut usage,
    )
    .await;
    assert_eq!(end.unwrap(), TurnEnd::OutOfTurns);
}

#[tokio::test(start_paused = true)]
async fn a_tool_call_past_its_bound_returns_a_failed_result() {
    let model = MockCompletionModel::from_turns([search_call("call_1"), MockTurn::text("done")]);
    let tools = tools(Arc::new(StalledSearch));
    let (spec, mut state, mut usage) = (spec(4), AgentState::default(), TokenUsage::default());
    for number in 1..=2 {
        step(
            &model,
            &spec,
            &tools,
            Ask::Off,
            &step_memo(number),
            &mut state,
            &mut usage,
        )
        .await
        .unwrap();
    }
    let follow_up = request(&model, 1);
    assert!(
        follow_up.contains("timed out after 60s"),
        "the model must read the timeout: {follow_up}"
    );
}

/// Replace the format of the serialized run of `state`, so the run does not
/// decode.
fn refuse_decode(state: &mut AgentState) {
    let run = state.run.as_mut().unwrap();
    run["format"] = json!(0);
}

#[tokio::test]
async fn a_run_that_does_not_decode_continues_from_its_transcript() {
    let model = MockCompletionModel::from_turns([search_call("call_1"), MockTurn::text("done")]);
    let tools = tools(Arc::new(CountingSearch::default()));
    let (spec, mut state, mut usage) = (spec(4), AgentState::default(), TokenUsage::default());
    step(
        &model,
        &spec,
        &tools,
        Ask::Off,
        &step_memo(1),
        &mut state,
        &mut usage,
    )
    .await
    .unwrap();
    let resumed = model.requests();
    refuse_decode(&mut state);

    let end = step(
        &model,
        &spec,
        &tools,
        Ask::Off,
        &step_memo(2),
        &mut state,
        &mut usage,
    )
    .await;
    let output = "done".to_string();
    assert_eq!(end.unwrap(), TurnEnd::Done { output });
    let rebuilt = &model.requests()[1];
    assert_eq!(rebuilt.chat_history.len(), 3);
    assert!(request(&model, 1).contains("fills the gap"));
    assert_eq!(resumed.len(), 1);
}

#[tokio::test]
async fn a_rebuild_during_a_wait_resolves_the_question_with_the_reply() {
    let model = MockCompletionModel::from_turns([
        ask_call("call_1", "Which edition?"),
        MockTurn::text("done"),
    ]);
    let tools = tools(Arc::new(CountingSearch::default()));
    let (spec, mut state, mut usage) = (spec(4), AgentState::default(), TokenUsage::default());
    step(
        &model,
        &spec,
        &tools,
        Ask::Open,
        &step_memo(1),
        &mut state,
        &mut usage,
    )
    .await
    .unwrap();
    refuse_decode(&mut state);

    let reply = Ask::Replied("the 2024 edition".to_string());
    let end = step(
        &model,
        &spec,
        &tools,
        reply,
        &step_memo(2),
        &mut state,
        &mut usage,
    )
    .await;
    let output = "done".to_string();
    assert_eq!(end.unwrap(), TurnEnd::Done { output });
    assert!(request(&model, 1).contains("the 2024 edition"));
}
