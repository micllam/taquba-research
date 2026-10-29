//! The host of an agent with one model turn per workflow step. A step advances
//! a Rig [`AgentRun`] by one model call and its tool calls, and the state
//! between the steps contains the serialized run and its transcript.

#[cfg(test)]
mod tests;
mod transcript;

use std::sync::Arc;
use std::time::Duration;

use rig_agent::completion::PromptError;
use rig_agent::run::prepare::prepare_request;
use rig_agent::run::{
    AgentRun, AgentRunStep, ExecCall, ModelTurn, ModelTurnOutcome, PendingToolCall, RunSpec,
    ToolAnswer,
};
use rig_agent::tool::ToolSet;
use rig_core::DynModel;
use rig_core::completion::{CompletionRequest, Message, ToolDefinition};
use rig_core::operation::Completion;
use rig_core::tool::{ToolContext, ToolExecutionError, ToolOutput, ToolResult};
use serde::{Deserialize, Serialize};
use taquba::LeaseHandle;
use taquba_workflow::{Memo, StepError};

pub use transcript::{Record, RecordedCall, RecordedResult};

use crate::runner::{LLM_CALL_TIMEOUT, classify_rig_err, lease_step_err, memoized, record_usage};
use crate::state::TokenUsage;

/// The name of the tool through which the agent asks the user a question.
pub(crate) const ASK_USER: &str = "ask_user";

/// The text of the skipped result of a second `ask_user` call.
const ONE_QUESTION: &str = "The run allows one question, and the user was already asked.";

/// The text of the result of a call that a rebuild of the run cannot recover.
const LOST: &str = "The result of this call was lost. Call the tool again if you need it.";

/// The step-memo key of the model reply of the step.
const MODEL_KEY: &str = "host/model";

/// The state of a host run between its steps.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentState {
    /// The serialized [`AgentRun`], absent before the first step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    run: Option<serde_json::Value>,
    /// The messages of the run.
    pub transcript: Vec<Record>,
}

/// Whether the agent can ask the user a question.
pub(crate) enum Ask {
    /// The agent cannot ask.
    Off,
    /// The agent can ask a question.
    Open,
    /// The agent asked, and this text resolves the question.
    Replied(String),
}

/// How a step of the host ended.
#[derive(Debug, PartialEq)]
pub(crate) enum TurnEnd {
    /// The run continues in the next step.
    Continue,
    /// The agent asked this question, and the run waits for the reply.
    Wait { question: String },
    /// The agent returned this output.
    Done { output: String },
    /// The run reached its limit of model calls.
    OutOfTurns,
}

/// One step of a host run: a model, its tools and the step it runs in.
pub(crate) struct Host<'a> {
    pub(crate) model: DynModel<Completion>,
    /// The request settings, with the model calls of the whole run in
    /// `max_turns`.
    pub(crate) spec: &'a RunSpec,
    pub(crate) tools: &'a ToolSet,
    /// The definitions that the model receives: the tools and, unless `ask` is
    /// `Off`, `ask_user`.
    pub(crate) definitions: Vec<Arc<ToolDefinition>>,
    /// The time bound of a call of the named tool.
    pub(crate) bound: fn(&str) -> Duration,
    pub(crate) ask: Ask,
    pub(crate) memo: &'a Memo,
    pub(crate) lease: &'a LeaseHandle,
}

impl Host<'_> {
    /// Advance the run of `state` by one model call and its tool calls, adding
    /// the usage of the call to `usage`. The first step starts the run on
    /// `prompt`. A step after a wait first resolves the pending `ask_user`
    /// call. A redelivered step reuses the model reply and the tool results of
    /// the step memo.
    pub(crate) async fn step(
        &self,
        state: &mut AgentState,
        prompt: &str,
        usage: &mut TokenUsage,
    ) -> Result<TurnEnd, StepError> {
        let mut run = self.restore(state, prompt)?;
        let (end, waiting) = self.advance(&mut run, usage).await?;
        state.run = Some(
            serde_json::to_value(&run)
                .map_err(|e| StepError::permanent(format!("encoding the agent run: {e}")))?,
        );
        // The full history closes the call that the run waits for with a
        // placeholder result.
        state.transcript = transcript::records(&run.full_history());
        if let Some(id) = waiting {
            transcript::remove_result(&mut state.transcript, &id);
        }
        Ok(end)
    }

    /// The run of `state`, or a run rebuilt from its transcript when the run
    /// does not decode, such as after an upgrade of Rig.
    fn restore(&self, state: &AgentState, prompt: &str) -> Result<AgentRun, StepError> {
        let Some(run) = &state.run else {
            return Ok(AgentRun::from_spec(self.spec, prompt, None));
        };
        match serde_json::from_value(run.clone()) {
            Ok(run) => Ok(run),
            Err(e) => {
                tracing::warn!(error = %e, "rebuilding the agent run from its transcript");
                self.rebuild(&state.transcript)
            }
        }
    }

    /// A run from `records`, with the remaining model calls. Each call of the
    /// last reply without a result receives the reply of the user or the text
    /// of a lost result, and the results become the prompt.
    fn rebuild(&self, records: &[Record]) -> Result<AgentRun, StepError> {
        let mut records = records.to_vec();
        let answered: Vec<String> = match records.last() {
            Some(Record::Results { results }) => results.iter().map(|r| r.id.clone()).collect(),
            _ => Vec::new(),
        };
        let replied = records.iter().rev().find_map(|record| match record {
            Record::Assistant { calls, .. } => Some(calls.clone()),
            _ => None,
        });
        let missing: Vec<RecordedResult> = replied
            .unwrap_or_default()
            .into_iter()
            .filter(|call| !answered.contains(&call.id))
            .map(|call| {
                let text = match (&self.ask, call.name.as_str()) {
                    (Ask::Replied(reply), ASK_USER) => reply.clone(),
                    _ => LOST.to_string(),
                };
                RecordedResult {
                    id: call.id,
                    name: call.name,
                    text,
                    is_error: false,
                }
            })
            .collect();
        if !missing.is_empty() {
            match records.last_mut() {
                Some(Record::Results { results }) => results.extend(missing),
                _ => records.push(Record::Results { results: missing }),
            }
        }
        let used = records
            .iter()
            .filter(|record| matches!(record, Record::Assistant { .. }))
            .count();
        let mut spec = self.spec.clone();
        spec.max_turns = Some(self.spec.effective_max_turns().saturating_sub(used));
        let mut messages = transcript::messages(&records)?;
        let prompt = messages
            .pop()
            .ok_or_else(|| StepError::permanent("rebuilding an agent run without a transcript"))?;
        let history = messages.into_iter().map(Arc::new).collect();
        Ok(AgentRun::from_spec(&spec, prompt, Some(history)))
    }

    /// Advance `run` to the end of the step: after the tool calls of its model
    /// call, at a question or at the end of the run. At a question, the call id
    /// of the `ask_user` call follows the end.
    async fn advance(
        &self,
        run: &mut AgentRun,
        usage: &mut TokenUsage,
    ) -> Result<(TurnEnd, Option<String>), StepError> {
        let mut called = false;
        loop {
            let next = match run.next_step() {
                Ok(next) => next,
                Err(PromptError::MaxTurns { .. }) => return Ok((TurnEnd::OutOfTurns, None)),
                Err(e) => return Err(classify_rig_err(e)),
            };
            match next {
                AgentRunStep::CallModel {
                    prompt,
                    history,
                    turn,
                } => {
                    if called {
                        return Ok((TurnEnd::Continue, None));
                    }
                    self.call_model(run, prompt, &history, turn, usage).await?;
                    called = true;
                }
                AgentRunStep::CallTools { calls } => {
                    if let Some((question, id)) = self.call_tools(run, calls, called).await? {
                        return Ok((TurnEnd::Wait { question }, Some(id)));
                    }
                    if called {
                        return Ok((TurnEnd::Continue, None));
                    }
                }
                AgentRunStep::Done(response) => {
                    let output = response.output();
                    return Ok((TurnEnd::Done { output }, None));
                }
            }
        }
    }

    /// Call the model under the lease, or reuse the reply of the step memo, and
    /// record the reply in `run`. An invalid tool call resolves by the policy
    /// of the spec.
    async fn call_model(
        &self,
        run: &mut AgentRun,
        prompt: Arc<Message>,
        history: &[Arc<Message>],
        turn: usize,
        usage: &mut TokenUsage,
    ) -> Result<(), StepError> {
        let capabilities = self.model.capabilities().completion;
        let prepared = prepare_request(
            self.spec,
            &capabilities,
            history,
            self.definitions.clone(),
            run.output_tool_name(),
            None,
        )
        .map_err(|e| StepError::permanent(format!("preparing the model request: {e}")))?;
        run.advertise_tools(turn, prepared.tools.clone());
        let policy = prepared.policy.clone();
        let request = prepared.apply(CompletionRequest::new((*prompt).clone()));
        let reply: ModelTurn = memoized(self.memo, MODEL_KEY, async {
            self.lease
                .ensure_at_least(LLM_CALL_TIMEOUT)
                .map_err(lease_step_err)?;
            let response = tokio::time::timeout(LLM_CALL_TIMEOUT, self.model.call(request))
                .await
                .map_err(|_| {
                    StepError::transient(format!(
                        "LLM call timed out after {}s",
                        LLM_CALL_TIMEOUT.as_secs()
                    ))
                })?
                .map_err(|e| classify_rig_err(e.into()))?;
            Ok(ModelTurn::from_policy(&response, policy))
        })
        .await?;
        record_usage(usage, &reply.usage);
        let mut outcome = run.model_response(reply).map_err(classify_rig_err)?;
        while let ModelTurnOutcome::NeedsResolution(_) = outcome {
            outcome = run
                .resolve_unhandled_invalid_tool_call()
                .map_err(classify_rig_err)?;
        }
        Ok(())
    }

    /// Pass a result to each tool call of a reply, and return the question and
    /// the call id of an `ask_user` call that the run must wait for. `called`
    /// tells whether the step called the model, so that the calls are not those
    /// of a reply before a wait.
    async fn call_tools(
        &self,
        run: &mut AgentRun,
        calls: Vec<PendingToolCall>,
        called: bool,
    ) -> Result<Option<(String, String)>, StepError> {
        let mut question = None;
        let mut answers: Vec<ToolAnswer> = Vec::with_capacity(calls.len());
        for call in calls {
            match call {
                PendingToolCall::Execute(call) if call.name().as_str() == ASK_USER => {
                    match (&self.ask, called, &question) {
                        (Ask::Replied(reply), false, _) => answers.push(
                            call.answer(ToolResult::success(ToolOutput::text(reply.clone()))),
                        ),
                        (Ask::Open, _, None) => {
                            question = Some((question_of(&call), call.id().wire().into_owned()));
                        }
                        _ => answers.push(call.answer(ToolResult::skipped(ONE_QUESTION))),
                    }
                }
                PendingToolCall::Execute(call) => {
                    let key = format!("host/call/{}", call.id().wire());
                    let result = memoized(self.memo, &key, self.execute(&call)).await?;
                    answers.push(call.answer(result));
                }
                PendingToolCall::Malformed(call) => answers.push(call.answer(None)),
                other => {
                    return Err(StepError::permanent(format!(
                        "unsupported pending tool call: {other:?}"
                    )));
                }
            }
        }
        for answer in answers {
            run.answer(answer).map_err(classify_rig_err)?;
        }
        Ok(question)
    }

    /// Run `call` under the lease. A call past the bound of its tool returns a
    /// failed result, which the model reads.
    async fn execute(&self, call: &ExecCall) -> Result<ToolResult, StepError> {
        let name = call.name().as_str();
        let bound = (self.bound)(name);
        self.lease.ensure_at_least(bound).map_err(lease_step_err)?;
        let mut context = ToolContext::default();
        let execution = self
            .tools
            .execute(name, call.arguments().to_string(), &mut context);
        Ok(tokio::time::timeout(bound, execution)
            .await
            .unwrap_or_else(|_| {
                ToolResult::failed(ToolExecutionError::timeout(format!(
                    "{name} timed out after {}s",
                    bound.as_secs()
                )))
            }))
    }
}

/// The question of an `ask_user` call, or its raw arguments without a
/// `question` field.
fn question_of(call: &ExecCall) -> String {
    let arguments = call.arguments();
    match arguments
        .get("question")
        .and_then(serde_json::Value::as_str)
    {
        Some(question) => question.to_string(),
        None => arguments.to_string(),
    }
}
