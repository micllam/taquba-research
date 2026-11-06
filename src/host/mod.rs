//! The host of an agent with one model turn per workflow step. A step advances
//! a Rig [`AgentRun`] by one model call, and the tool calls of the reply run as
//! the members of a job group of the turn, which the run waits for without a
//! worker. The next step passes each call the result of its member. The state
//! between the steps contains the serialized run and its transcript.

#[cfg(test)]
mod tests;
mod transcript;

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::Arc;

use rig_agent::completion::PromptError;
use rig_agent::run::prepare::prepare_request;
use rig_agent::run::{
    AgentRun, AgentRunStep, ExecCall, ModelTurn, ModelTurnOutcome, PendingToolCall, RunSpec,
    ToolAnswer,
};
use rig_core::DynModel;
use rig_core::completion::{CompletionRequest, Message, ToolDefinition};
use rig_core::operation::Completion;
use rig_core::tool::{ToolExecutionError, ToolOutput, ToolResult};
use serde::{Deserialize, Serialize};
use taquba::LeaseHandle;
use taquba_workflow::jobs::{Job, JobError, JobRunner};
use taquba_workflow::{Error as WorkflowError, Memo, RunId, RunOptions, StepError};

pub use transcript::{Record, RecordedCall, RecordedResult};

use crate::runner::{
    FETCH_JOIN_LEASE, LLM_CALL_TIMEOUT, classify_rig_err, lease_step_err, memoized, record_usage,
    under_lease,
};
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
    /// The tool calls of the reply run as the members of this group, and the
    /// run waits for the group.
    Group { group_id: RunId },
    /// The agent asked this question, and the run waits for the reply.
    Wait { question: String },
    /// The agent returned this output.
    Done { output: String },
    /// The run reached its limit of model calls.
    OutOfTurns,
}

/// What executes a tool call of a reply.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum Plan<J> {
    /// The member of the group of the turn.
    Member(J),
    /// The result, without a job.
    Result(ToolResult),
}

/// The tools of a host run. The tool calls of a reply run as the members of a
/// job group, and the calls of the reply with the key of a member share it.
pub(crate) trait TurnTools: Sync {
    /// The job of a member.
    type Job: Job + Serialize + serde::de::DeserializeOwned + Send + Sync;

    /// What executes `call`. `earlier` contains the records before the reply of
    /// `call`.
    fn plan(
        &self,
        call: &RecordedCall,
        earlier: &[Record],
    ) -> impl Future<Output = Result<Plan<Self::Job>, StepError>> + Send;

    /// The key of the member that executes `call`.
    fn key(&self, call: &RecordedCall) -> Option<String>;

    /// The result of `call` from the result of its member.
    fn render(
        &self,
        call: &RecordedCall,
        member: &Result<<Self::Job as Job>::Output, JobError>,
    ) -> ToolResult;
}

/// The id of the group of the tool calls of the reply `turn`, counted from 1,
/// of the host run of the workflow run `run_id`.
pub(crate) fn turn_group_id(run_id: &RunId, turn: usize) -> Result<RunId, StepError> {
    RunId::new(format!("turn-{run_id}-{turn}"))
        .map_err(|e| StepError::permanent(format!("group id: {e}")))
}

/// The number of replies in `records`, which is the turn of the last reply.
pub(crate) fn replies(records: &[Record]) -> usize {
    records
        .iter()
        .filter(|record| matches!(record, Record::Assistant { .. }))
        .count()
}

/// One step of a host run: a model, its tools and the step it runs in.
pub(crate) struct Host<'a, T> {
    pub(crate) model: DynModel<Completion>,
    /// The request settings, with the model calls of the whole run in
    /// `max_turns`.
    pub(crate) spec: &'a RunSpec,
    pub(crate) tools: &'a T,
    /// The definitions that the model receives: the tools and, unless `ask` is
    /// `Off`, `ask_user`.
    pub(crate) definitions: Vec<Arc<ToolDefinition>>,
    pub(crate) ask: Ask,
    /// The runner of the groups of the turns.
    pub(crate) jobs: &'a JobRunner,
    /// The options of the members of each group of a turn.
    pub(crate) group_options: RunOptions,
    /// The workflow run, whose id is part of the id of each group of a turn.
    pub(crate) run_id: &'a RunId,
    pub(crate) memo: &'a Memo,
    pub(crate) lease: &'a LeaseHandle,
}

impl<T: TurnTools> Host<'_, T> {
    /// Advance the run of `state` by one model call, adding the usage of the
    /// call to `usage`. The first step starts the run on `prompt`. A step after
    /// a wait first resolves the open calls of the previous reply. A
    /// redelivered step reuses the model reply and the plan of each tool call
    /// of the step memo.
    pub(crate) async fn step(
        &self,
        state: &mut AgentState,
        prompt: &str,
        usage: &mut TokenUsage,
    ) -> Result<TurnEnd, StepError> {
        let mut run = self.restore(state, prompt).await?;
        let (end, open) = self.advance(&mut run, &state.transcript, usage).await?;
        state.run = Some(
            serde_json::to_value(&run)
                .map_err(|e| StepError::permanent(format!("encoding the agent run: {e}")))?,
        );
        // The full history closes each open call with a placeholder result.
        state.transcript = transcript::records(&run.full_history());
        for id in &open {
            transcript::remove_result(&mut state.transcript, id);
        }
        Ok(end)
    }

    /// The run of `state`, or a run rebuilt from its transcript when the run
    /// does not decode, such as after an upgrade of Rig.
    async fn restore(&self, state: &AgentState, prompt: &str) -> Result<AgentRun, StepError> {
        let Some(run) = &state.run else {
            return Ok(AgentRun::from_spec(self.spec, prompt, None));
        };
        match serde_json::from_value(run.clone()) {
            Ok(run) => Ok(run),
            Err(e) => {
                tracing::warn!(error = %e, "rebuilding the agent run from its transcript");
                self.rebuild(&state.transcript).await
            }
        }
    }

    /// A run from `records`, with the remaining model calls. Each call of the
    /// last reply without a result receives the result of its member, the reply
    /// of the user or the text of a lost result, and the results become the
    /// prompt.
    async fn rebuild(&self, records: &[Record]) -> Result<AgentRun, StepError> {
        let mut records = records.to_vec();
        let answered: Vec<String> = match records.last() {
            Some(Record::Results { results }) => results.iter().map(|r| r.id.clone()).collect(),
            _ => Vec::new(),
        };
        let replied = records.iter().rev().find_map(|record| match record {
            Record::Assistant { calls, .. } => Some(calls.clone()),
            _ => None,
        });
        let open: Vec<RecordedCall> = replied
            .unwrap_or_default()
            .into_iter()
            .filter(|call| !answered.contains(&call.id))
            .collect();
        let members = if open.iter().any(|call| call.name != ASK_USER) {
            self.join(replies(&records)).await?
        } else {
            HashMap::new()
        };
        let missing: Vec<RecordedResult> = open
            .into_iter()
            .map(|call| {
                let text = match (&self.ask, call.name.as_str()) {
                    (Ask::Replied(reply), ASK_USER) => reply.clone(),
                    _ => match self.member_result(&call, &members) {
                        Some(result) => {
                            return transcript::recorded(&call, &result);
                        }
                        None => LOST.to_string(),
                    },
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
        let used = replies(&records);
        let mut spec = self.spec.clone();
        spec.max_turns = Some(self.spec.effective_max_turns().saturating_sub(used));
        let mut messages = transcript::messages(&records)?;
        let prompt = messages
            .pop()
            .ok_or_else(|| StepError::permanent("rebuilding an agent run without a transcript"))?;
        let history = messages.into_iter().map(Arc::new).collect();
        Ok(AgentRun::from_spec(&spec, prompt, Some(history)))
    }

    /// Advance `run` to the end of the step: after the dispatch of the tool
    /// calls of its model call, at a question or at the end of the run.
    /// `stored` contains the transcript at the start of the step. The ids of
    /// the calls that stay open follow the end.
    async fn advance(
        &self,
        run: &mut AgentRun,
        stored: &[Record],
        usage: &mut TokenUsage,
    ) -> Result<(TurnEnd, Vec<String>), StepError> {
        let mut called = false;
        loop {
            let next = match run.next_step() {
                Ok(next) => next,
                Err(PromptError::MaxTurns { .. }) => return Ok((TurnEnd::OutOfTurns, Vec::new())),
                Err(e) => return Err(classify_rig_err(e)),
            };
            match next {
                AgentRunStep::CallModel {
                    prompt,
                    history,
                    turn,
                } => {
                    if called {
                        return Ok((TurnEnd::Continue, Vec::new()));
                    }
                    self.call_model(run, prompt, &history, turn, usage).await?;
                    called = true;
                }
                AgentRunStep::CallTools { calls } if called => {
                    return self.dispatch(run, calls).await;
                }
                AgentRunStep::CallTools { calls } => {
                    if let Some((question, id)) = self.resolve(run, calls, stored).await? {
                        return Ok((TurnEnd::Wait { question }, vec![id]));
                    }
                }
                AgentRunStep::Done(response) => {
                    let output = response.output();
                    return Ok((TurnEnd::Done { output }, Vec::new()));
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

    /// Dispatch the tool calls of the reply of this step. The calls with a
    /// member are submitted as the group of the turn, an `ask_user` call stays
    /// open for a question, and the other calls receive their results.
    async fn dispatch(
        &self,
        run: &mut AgentRun,
        calls: Vec<PendingToolCall>,
    ) -> Result<(TurnEnd, Vec<String>), StepError> {
        let records = transcript::records(&run.full_history());
        let turn = replies(&records);
        let earlier = &records[..records
            .iter()
            .rposition(|record| matches!(record, Record::Assistant { .. }))
            .unwrap_or(0)];
        let mut question = None;
        let mut open = Vec::new();
        let mut members = Vec::new();
        let mut keys = HashSet::new();
        let mut answers: Vec<ToolAnswer> = Vec::with_capacity(calls.len());
        for call in calls {
            match call {
                PendingToolCall::Execute(call) if call.name().as_str() == ASK_USER => {
                    match (&self.ask, &question) {
                        (Ask::Open, None) => {
                            open.push(call.id().wire().into_owned());
                            question = Some(question_of(&call));
                        }
                        _ => answers.push(call.answer(ToolResult::skipped(ONE_QUESTION))),
                    }
                }
                PendingToolCall::Execute(call) => {
                    let recorded = recorded_call(&call);
                    let key = format!("host/call/{}", recorded.id);
                    let plan = memoized(self.memo, &key, self.tools.plan(&recorded, earlier));
                    match plan.await? {
                        Plan::Member(job) => {
                            open.push(recorded.id);
                            if keys.insert(job.idempotency_key()) {
                                members.push(job);
                            }
                        }
                        Plan::Result(result) => answers.push(call.answer(result)),
                    }
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
        if !members.is_empty() {
            let group_id = turn_group_id(self.run_id, turn)?;
            self.jobs
                .group::<T::Job>(group_id.clone())
                .submit_with(members, self.group_options.clone())
                .await?;
            return Ok((TurnEnd::Group { group_id }, open));
        }
        Ok(match question {
            Some(question) => (TurnEnd::Wait { question }, open),
            None => (TurnEnd::Continue, Vec::new()),
        })
    }

    /// Pass a result to each open call of the reply before this step, and
    /// return the question and the call id of an `ask_user` call that the run
    /// must wait for. A call with a member receives the result of its member in
    /// the group of the turn of `stored`.
    async fn resolve(
        &self,
        run: &mut AgentRun,
        calls: Vec<PendingToolCall>,
        stored: &[Record],
    ) -> Result<Option<(String, String)>, StepError> {
        let mut question = None;
        let mut members = None;
        let mut answers: Vec<ToolAnswer> = Vec::with_capacity(calls.len());
        for call in calls {
            match call {
                PendingToolCall::Execute(call) if call.name().as_str() == ASK_USER => {
                    match (&self.ask, &question) {
                        (Ask::Replied(reply), _) => answers.push(
                            call.answer(ToolResult::success(ToolOutput::text(reply.clone()))),
                        ),
                        (Ask::Open, None) => {
                            question = Some((question_of(&call), call.id().wire().into_owned()));
                        }
                        _ => answers.push(call.answer(ToolResult::skipped(ONE_QUESTION))),
                    }
                }
                PendingToolCall::Execute(call) => {
                    if members.is_none() {
                        members = Some(self.join(replies(stored)).await?);
                    }
                    let members = members.as_ref().expect("joined above");
                    let recorded = recorded_call(&call);
                    let result = self
                        .member_result(&recorded, members)
                        .unwrap_or_else(|| ToolResult::failed(ToolExecutionError::other(LOST)));
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

    /// The result of each member of the group of `turn` by key, or no results
    /// for a turn without a group. A group with active members has outlasted
    /// its wait, so its members are cancelled first, and a cancelled member
    /// reports a failure.
    async fn join(
        &self,
        turn: usize,
    ) -> Result<HashMap<String, Result<<T::Job as Job>::Output, JobError>>, StepError> {
        let group = self.jobs.group::<T::Job>(turn_group_id(self.run_id, turn)?);
        let status = match group.status().await {
            Ok(status) => status,
            Err(WorkflowError::GroupNotFound(_)) => return Ok(HashMap::new()),
            Err(e) => return Err(e.into()),
        };
        if status.pending > 0 {
            tracing::warn!(pending = status.pending, "tool wait elapsed, cancelling");
            group
                .cancel(Some("the wait for the tool calls elapsed".to_string()))
                .await?;
        }
        let results = under_lease(self.lease, FETCH_JOIN_LEASE, "tool join", async {
            Ok(group.join().await?)
        })
        .await?;
        Ok(results
            .into_iter()
            .map(|member| (member.key, member.result))
            .collect())
    }

    /// The result of `call` from its member in `members`.
    fn member_result(
        &self,
        call: &RecordedCall,
        members: &HashMap<String, Result<<T::Job as Job>::Output, JobError>>,
    ) -> Option<ToolResult> {
        let member = members.get(&self.tools.key(call)?)?;
        Some(self.tools.render(call, member))
    }
}

/// The recorded form of `call`.
fn recorded_call(call: &ExecCall) -> RecordedCall {
    RecordedCall {
        id: call.id().wire().into_owned(),
        name: call.name().as_str().to_string(),
        arguments: call.arguments().clone(),
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
