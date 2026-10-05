//! The reply watcher: a task in the process of a workflow runtime that wakes a
//! run waiting for the reply of the user once its reply or its cancellation is
//! in the store.

use std::sync::Arc;
use std::time::Duration;

use taquba::{JobStatus, Queue, QueueView};
use taquba_workflow::{
    HEADER_RUN_ID, HEADER_SIGNAL_WAIT, StepRunner, TerminalHook, WorkflowRuntime,
};
use tokio::task::JoinHandle;

use crate::store::{CancelSentinel, ReplyBox, WORKFLOW_QUEUE_NAME, clarification_signal_key};

/// Interval between two passes over the waiting runs.
const WATCH_INTERVAL: Duration = Duration::from_secs(2);
/// Page size of the scan of scheduled step jobs.
const SCAN_PAGE: usize = 256;

/// Spawn the task that wakes each run of `runtime` that waits for the reply of
/// the user: with its reply from `replies`, or with an empty signal once
/// `cancel` includes its cancellation. A pass runs every 2 seconds. Dropping
/// the returned handle stops the task.
///
/// The CLI and [`crate::ResearchAgent`] spawn it. A caller with its own
/// [`WorkflowRuntime`] must spawn it, or a reply from another process reaches a
/// waiting run only through [`WorkflowRuntime::signal`].
pub fn spawn_reply_watcher<R, H>(
    runtime: WorkflowRuntime<R, H>,
    queue: Arc<Queue>,
    replies: ReplyBox,
    cancel: Option<CancelSentinel>,
) -> ReplyWatcher
where
    R: StepRunner + 'static,
    H: TerminalHook + 'static,
{
    let task = tokio::spawn(async move {
        loop {
            match due_signals(queue.view(), &replies, cancel.as_ref()).await {
                Ok(signals) => {
                    for (key, payload) in signals {
                        if let Err(e) = runtime.signal(&key, payload).await {
                            tracing::warn!(%key, error = %e, "waking a waiting run failed");
                        }
                    }
                }
                Err(e) => tracing::warn!(error = %e, "scanning waiting runs failed"),
            }
            tokio::time::sleep(WATCH_INTERVAL).await;
        }
    });
    ReplyWatcher(task)
}

/// Handle of the task of [`spawn_reply_watcher`]. Dropping it stops the task.
#[derive(Debug)]
pub struct ReplyWatcher(JoinHandle<()>);

impl Drop for ReplyWatcher {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// The signals that wake the waiting runs of `view`, as correlation key and
/// payload: the reply of a run with a stored reply, else an empty payload for a
/// run with a cancellation sentinel. A run waits while its step job is
/// scheduled and unclaimed, with the clarification key of the run in
/// [`HEADER_SIGNAL_WAIT`].
pub(crate) async fn due_signals(
    view: &QueueView,
    replies: &ReplyBox,
    cancel: Option<&CancelSentinel>,
) -> anyhow::Result<Vec<(String, Vec<u8>)>> {
    let mut signals = Vec::new();
    let mut cursor: Option<Vec<u8>> = None;
    loop {
        let page = view
            .list_jobs(
                WORKFLOW_QUEUE_NAME,
                JobStatus::Scheduled,
                cursor.as_deref(),
                SCAN_PAGE,
            )
            .await?;
        for job in &page.jobs {
            let (Some(key), Some(run_id)) = (
                job.headers.get(HEADER_SIGNAL_WAIT),
                job.headers.get(HEADER_RUN_ID),
            ) else {
                continue;
            };
            // A retried step keeps the header of the wait that woke it.
            if job.attempts > 0 || *key != clarification_signal_key(run_id) {
                continue;
            }
            if let Some(reply) = replies.get(run_id).await? {
                signals.push((key.clone(), reply.into_bytes()));
            } else if let Some(cancel) = cancel
                && cancel.is_set(run_id).await?
            {
                signals.push((key.clone(), Vec::new()));
            }
        }
        match page.next_cursor {
            Some(next) => cursor = Some(next),
            None => return Ok(signals),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    use taquba::EnqueueOptions;
    use taquba::object_store::ObjectStore;
    use taquba::object_store::memory::InMemory;
    use taquba::object_store::path::Path;

    #[tokio::test]
    async fn due_signals_wake_waiting_runs_with_a_reply_or_a_cancellation() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let queue = Queue::open(object_store.clone(), "q").await.unwrap();
        let replies = ReplyBox::new(object_store.clone(), &Path::default());
        let cancel = CancelSentinel::new(object_store, &Path::default());
        let later = SystemTime::now() + Duration::from_secs(3600);
        for (run, wait) in [
            ("01A", Some(clarification_signal_key("01A"))),
            ("01B", Some(clarification_signal_key("01B"))),
            ("01C", Some(clarification_signal_key("01C"))),
            ("01D", None),
            ("01E", Some("other/01E".to_string())),
        ] {
            let mut options = EnqueueOptions::default()
                .header(HEADER_RUN_ID, run)
                .run_at(later);
            if let Some(wait) = wait {
                options = options.header(HEADER_SIGNAL_WAIT, wait);
            }
            queue
                .enqueue_with(WORKFLOW_QUEUE_NAME, Vec::new(), options)
                .await
                .unwrap();
        }
        replies.put("01A", "the 2024 edition").await.unwrap();
        cancel.mark("01B").await.unwrap();
        // A run without a wait on its clarification key does not wait for a
        // reply, even with a stored reply.
        replies.put("01D", "unused").await.unwrap();
        replies.put("01E", "unused").await.unwrap();

        let mut signals = due_signals(queue.view(), &replies, Some(&cancel))
            .await
            .unwrap();
        signals.sort();
        assert_eq!(
            signals,
            vec![
                (
                    clarification_signal_key("01A"),
                    b"the 2024 edition".to_vec()
                ),
                (clarification_signal_key("01B"), Vec::new()),
            ]
        );
    }
}
