//! The effect journal: handlers on a rig bus that record each completion and
//! tool-call outcome in the step [`Memo`], and return the recorded outcome when
//! a retried step dispatches the same effect.
//!
//! A completion uses its ordinal within the step as its key, and a tool call
//! uses its turn and the call id of the provider. A replayed completion
//! contains the call ids of its tool calls, so both keys are stable across a
//! retry. The call id reaches only the hooks of the agent, so
//! [`Journal::call_keys`] must be added to the agent that dispatches the tool
//! calls.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use rig_agent::agent::{AgentHook, DispatchAction, DispatchEvent, HookContext};
use rig_core::effect::{EffectId, EffectKind, HandlerDescriptor};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::serve::{Dispatch, Reply, Serve};
use taquba::LeaseHandle;
use taquba_workflow::{Memo, StepError};

/// The journal of one step attempt. A clone shares the same journal.
#[derive(Clone)]
pub(crate) struct Journal {
    inner: Arc<JournalInner>,
}

struct JournalInner {
    memo: Memo,
    lease: LeaseHandle,
    completions: AtomicUsize,
    /// Journal keys of dispatched tool calls, by effect id.
    calls: Mutex<HashMap<EffectId, String>>,
    /// The first failure of the journal itself.
    fault: Mutex<Option<StepError>>,
}

impl Journal {
    pub(crate) fn new(memo: Memo, lease: LeaseHandle) -> Self {
        Self {
            inner: Arc::new(JournalInner {
                memo,
                lease,
                completions: AtomicUsize::new(0),
                calls: Mutex::new(HashMap::new()),
                fault: Mutex::new(None),
            }),
        }
    }

    /// `handler` under this journal. The journal extends the lease by `bound`
    /// before each call of `handler` and fails the call after `bound`.
    pub(crate) fn wrap<H: Serve>(&self, handler: H, bound: Duration) -> Journalled<H> {
        Journalled {
            inner: handler,
            journal: self.clone(),
            bound,
        }
    }

    /// The hook that gives each tool call its journal key.
    pub(crate) fn call_keys(&self) -> CallKeys {
        CallKeys(self.clone())
    }

    /// The first failure to read or write the journal, or to extend the lease.
    /// A tool handler's error reaches the model as a failed result, so the step
    /// must return this error after the agent finishes.
    pub(crate) fn take_fault(&self) -> Option<StepError> {
        lock(&self.inner.fault).take()
    }

    fn key(&self, kind: &EffectKind, id: EffectId) -> Result<String, StepError> {
        match kind {
            EffectKind::Completion { .. } => {
                let ordinal = self.inner.completions.fetch_add(1, Ordering::Relaxed);
                Ok(format!("completion/{ordinal}"))
            }
            EffectKind::ToolCall { name, .. } => {
                lock(&self.inner.calls).remove(&id).ok_or_else(|| {
                    StepError::permanent(format!(
                        "journal: tool call `{name}` was dispatched without a call id"
                    ))
                })
            }
            other => Err(StepError::permanent(format!(
                "journal: unsupported effect `{}`",
                other.family()
            ))),
        }
    }

    async fn replay_or_run(
        &self,
        key: &str,
        bound: Duration,
        call: impl Future<Output = Reply>,
    ) -> Reply {
        let outcome = async {
            match self.inner.memo.get(key).await {
                Ok(Some(bytes)) => {
                    return serde_json::from_slice(&bytes).map_err(|e| {
                        self.fault(StepError::permanent(format!("journal[{key}] decode: {e}")))
                    });
                }
                Ok(None) => {}
                Err(e) => {
                    return Err(
                        self.fault(StepError::transient(format!("journal[{key}] read: {e}")))
                    );
                }
            }
            if let Err(e) = self.inner.lease.ensure_at_least(bound) {
                return Err(
                    self.fault(StepError::transient(format!("lease extension failed: {e}")))
                );
            }
            let Ok(outcome) =
                tokio::time::timeout(bound, async { call.await.into_outcome().await }).await
            else {
                return Err(ErrorReport::new(
                    ErrorKind::Timeout,
                    format!("timed out after {}s", bound.as_secs()),
                )
                .with_retryable(true));
            };
            // An error is not recorded, so a retried step calls the handler
            // again.
            if let Ok(recorded) = &outcome {
                let bytes = serde_json::to_vec(recorded).map_err(|e| {
                    self.fault(StepError::permanent(format!("journal[{key}] encode: {e}")))
                })?;
                if let Err(e) = self.inner.memo.put(key, &bytes).await {
                    return Err(
                        self.fault(StepError::transient(format!("journal[{key}] write: {e}")))
                    );
                }
            }
            outcome
        };
        Reply::Outcome(outcome.await)
    }

    /// Keep the first fault and return the report the handler replies with.
    fn fault(&self, error: StepError) -> ErrorReport {
        let report = ErrorReport::new(ErrorKind::Internal, error.to_string()).with_retryable(false);
        lock(&self.inner.fault).get_or_insert(error);
        report
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A handler whose outcomes the [`Journal`] records and replays.
pub(crate) struct Journalled<H> {
    inner: H,
    journal: Journal,
    bound: Duration,
}

impl<H: Serve> Serve for Journalled<H> {
    type Family = H::Family;

    fn descriptor(&self) -> HandlerDescriptor {
        self.inner.descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let key = match self.journal.key(&kind, dispatch.id()) {
            Ok(key) => key,
            Err(error) => return Reply::Outcome(Err(self.journal.fault(error))),
        };
        self.journal
            .replay_or_run(&key, self.bound, self.inner.serve(kind, dispatch))
            .await
    }
}

/// The hook that records the journal key of each tool call at dispatch.
pub(crate) struct CallKeys(Journal);

impl AgentHook for CallKeys {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if let Some(call_id) = event.call_id {
            lock(&self.0.inner.calls).insert(event.id, format!("tool/{}/{call_id}", event.turn));
        }
        DispatchAction::Proceed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use rig_agent::bus::Bus;
    use rig_core::effect::tool_key;
    use rig_core::serve::adapters::ToolAdapter;
    use taquba::object_store::memory::InMemory;
    use taquba_workflow::{MemoStore, RunId, StepErrorKind};

    use crate::investigate::WebSearch;
    use crate::search::Tavily;

    #[tokio::test]
    async fn journal_faults_a_tool_call_without_a_call_id() {
        let memo = MemoStore::new(Arc::new(InMemory::new()), "test-memo")
            .new_memo(&RunId::new("run").unwrap(), 4);
        let journal = Journal::new(memo, LeaseHandle::detached());
        let (dispatcher, _registrar, mut driver) = Bus::channel();
        let key = tool_key("web_search");
        let search = WebSearch {
            search: Arc::new(Tavily::new("unused")),
        };
        driver
            .register(
                key.clone(),
                journal.wrap(ToolAdapter::new(search), Duration::from_secs(5)),
            )
            .unwrap();
        tokio::spawn(driver);

        // Without the `CallKeys` hook, the journal lacks a key for the call.
        let outcome = dispatcher
            .dispatch(
                &key,
                EffectKind::ToolCall {
                    name: "web_search".to_string(),
                    args: r#"{"query":"gaps"}"#.to_string(),
                },
            )
            .await;
        assert!(outcome.is_err());
        let fault = journal
            .take_fault()
            .expect("the journal must keep the fault");
        assert!(matches!(fault.kind, StepErrorKind::Permanent), "{fault:?}");
    }
}
