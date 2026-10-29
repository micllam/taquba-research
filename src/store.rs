//! Runs of the research workflow, read through a
//! [`WorkflowView`](taquba_workflow::WorkflowView).
//!
//! The run listing of the view contains the submission time and the headers of
//! each run, and [`QUERY_HEADER`](crate::store::QUERY_HEADER) contains the
//! query. Each step settlement writes the
//! [`RunProgress`](crate::store::RunProgress) of the run under
//! [`PROGRESS_KEY`](crate::store::PROGRESS_KEY) in its run KV space, which
//! outlasts the payload of the step. A display status is derived at read time,
//! see [`derive_display_status`](crate::store::derive_display_status).

use std::sync::Arc;

use futures_util::TryStreamExt;
use object_store::ObjectStore;
use object_store::path::Path;
use serde::{Deserialize, Serialize};
use taquba::object_store;
use taquba::{JobStatus, KvOrder, QueueReader, QueueView};
use taquba_workflow::{
    HEADER_TERMINAL, ListedRun, MemoStore, RunFilter, RunId, RunState, TerminalStatus, WorkflowView,
};

use crate::state::TokenUsage;

/// Name of the workflow queue. The CLI and [`crate::ResearchAgent`] configure
/// the workflow runtime with it, and the reader-side queries in this module
/// target the same queue.
pub const WORKFLOW_QUEUE_NAME: &str = "research-workflow";

/// Name of the control queue of the workflow runtime, on which another process
/// enqueues a `taquba_workflow::Command` through the inbox. A runtime that
/// applies `reply` and `cancel` from another process requires a queue opened
/// with [`taquba::OpenOptions::inbox`] for this queue and
/// [`WORKFLOW_QUEUE_NAME`], with a [`taquba::QueueConfig`] of this queue that
/// keeps done jobs for the record retention of the inbox. The runtime requires
/// [`taquba_workflow::WorkflowRuntimeBuilder::control_queue`] of this queue.
pub const WORKFLOW_CONTROL_QUEUE_NAME: &str = "research-workflow.control";

/// Memo prefix the CLI and [`crate::ResearchAgent`] configure on the workflow
/// runtime. Set explicitly so [`workflow_view`] reads the memo store the
/// runtime writes to.
pub const WORKFLOW_MEMO_PREFIX: &str = "research-workflow-memo";

/// Header of a run that contains its query. The CLI and
/// [`crate::ResearchAgent`] submit each run with it.
pub const QUERY_HEADER: &str = "research.query";

/// The read-only workflow queries over `reader` and the runtime's memo store in
/// `object_store`, for a process without a runtime.
pub fn workflow_view(reader: &QueueReader, object_store: Arc<dyn ObjectStore>) -> WorkflowView {
    WorkflowView::new(
        reader.view().clone(),
        MemoStore::new(object_store, WORKFLOW_MEMO_PREFIX),
    )
}

/// Key of the [`RunProgress`] in the run KV space of a run, which the memo
/// sweep deletes with the run.
pub const PROGRESS_KEY: &[u8] = b"progress";

/// The progress of a run at its latest step settlement, stored under
/// [`PROGRESS_KEY`]. JSON-encoded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunProgress {
    /// Number of steps the runner completed.
    pub steps_completed: u32,
    /// Aggregate token usage across every LLM call in the run.
    pub token_usage: TokenUsage,
}

impl RunProgress {
    /// Encode the entry for storage under [`PROGRESS_KEY`].
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("RunProgress is serde-derivable")
    }

    /// Decode an entry read from the run KV space.
    pub fn from_bytes(bytes: &[u8]) -> serde_json::Result<Self> {
        serde_json::from_slice(bytes)
    }
}

/// The progress entry of `run_id` through `view`. `Ok(None)` before the first
/// step settles and after the run expires, and an error for a malformed entry.
pub async fn run_progress(
    view: &WorkflowView,
    run_id: &RunId,
) -> anyhow::Result<Option<RunProgress>> {
    let Some(bytes) = view.run_kv(run_id, PROGRESS_KEY).await? else {
        return Ok(None);
    };
    let progress = RunProgress::from_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("malformed progress entry for {run_id}: {e}"))?;
    Ok(Some(progress))
}

/// Prefix, under the store's key prefix, of canonical report blobs.
pub const REPORTS_PREFIX: &str = "reports";

/// Object key of `run_id`'s canonical report blob.
pub fn report_path(prefix: &Path, run_id: &str) -> Path {
    prefix
        .clone()
        .join(REPORTS_PREFIX)
        .join(format!("{run_id}.md"))
}

/// Display status of a run, computed at read time from the run state of the
/// workflow view. See [`derive_display_status`] for the precedence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunDisplayStatus {
    /// The view's termination says succeeded.
    Succeeded,
    /// The view's termination says failed. The view also reports a step that
    /// the queue dead-lettered outside the worker, whose run the next worker
    /// terminates as failed.
    Failed,
    /// The view's termination says cancelled.
    Cancelled,
    /// The view reports [`RunState::Cancelling`], or a cancel command awaits a
    /// writer.
    CancellationRequested,
    /// The view reports [`RunState::Waiting`]: the run waits for the reply of
    /// the user.
    Waiting,
    /// The view reports [`RunState::Running`]: a worker claimed the current
    /// step. A claim abandoned by a killed worker process also derives this
    /// state: lease expiry is writer-process state, so the job stays claimed
    /// until the next writer open requeues it.
    Running,
    /// The view reports [`RunState::Pending`] or [`RunState::Scheduled`]: an
    /// interrupted run that awaits `resume`, the interval between an
    /// acknowledgement and the next claim, or the backoff of a retry.
    Queued,
}

impl RunDisplayStatus {
    /// Human-readable label used in CLI output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::CancellationRequested => "cancellation requested",
            Self::Waiting => "waiting for reply",
            Self::Running => "running",
            Self::Queued => "queued",
        }
    }
}

impl std::fmt::Display for RunDisplayStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Compute a run's display status from the run state that the workflow view
/// reports and whether a cancel command of the run awaits a writer. Precedence:
/// a terminal state of the view, then the cancellation, then the view's other
/// states.
pub fn derive_display_status(state: &RunState, cancel_requested: bool) -> RunDisplayStatus {
    match state {
        RunState::Terminated(termination) => match termination.status {
            TerminalStatus::Succeeded => RunDisplayStatus::Succeeded,
            TerminalStatus::Failed => RunDisplayStatus::Failed,
            TerminalStatus::Cancelled => RunDisplayStatus::Cancelled,
        },
        RunState::DeadLettered => RunDisplayStatus::Failed,
        _ if cancel_requested => RunDisplayStatus::CancellationRequested,
        RunState::Cancelling { .. } => RunDisplayStatus::CancellationRequested,
        RunState::Waiting { .. } => RunDisplayStatus::Waiting,
        RunState::Running => RunDisplayStatus::Running,
        RunState::Pending | RunState::Scheduled { .. } => RunDisplayStatus::Queued,
    }
}

/// Page size for KV and job-listing scans.
const SCAN_PAGE: usize = 256;

/// Every run of `view`, newest first.
pub async fn list_runs(view: &WorkflowView) -> taquba_workflow::Result<Vec<ListedRun>> {
    view.runs(RunFilter::All, KvOrder::Descending, SCAN_PAGE)
        .try_collect()
        .await
}

/// Number of step jobs waiting in the workflow queue (pending or scheduled,
/// terminal notifications excluded). The workflow is sequential, so each
/// waiting run has exactly one step job and the count equals the number of
/// interrupted or queued runs. `view` is the view of the writer or of a reader.
pub async fn count_waiting_step_jobs(view: &QueueView) -> taquba::Result<usize> {
    let mut count = 0usize;
    for status in [JobStatus::Pending, JobStatus::Scheduled] {
        let mut cursor: Option<Vec<u8>> = None;
        loop {
            let page = view
                .list_jobs(WORKFLOW_QUEUE_NAME, status, cursor.as_deref(), SCAN_PAGE)
                .await?;
            count += page
                .jobs
                .iter()
                .filter(|j| !j.headers.contains_key(HEADER_TERMINAL))
                .count();
            match page.next_cursor {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use taquba::object_store::memory::InMemory;
    use taquba::{Queue, ReaderMode, ReaderOptions};
    use taquba_workflow::{
        HEADER_RUN_ID, NoopTerminalHook, RunOptions, RunSpec, StepError, StepRef, WorkflowRuntime,
    };

    fn terminated(status: TerminalStatus) -> RunState {
        RunState::Terminated(taquba_workflow::RunTermination {
            status,
            error: None,
            error_kind: None,
            cancelled_by: None,
            final_step: StepRef {
                number: 0,
                job_id: String::new(),
            },
            terminated_at_ms: 0,
        })
    }

    #[test]
    fn the_states_of_the_view_and_a_pending_cancellation_derive_the_display_status() {
        use RunDisplayStatus::{CancellationRequested, Queued};
        // The third column is the display status with a pending cancellation.
        for (state, display, cancel_display) in [
            (
                terminated(TerminalStatus::Succeeded),
                RunDisplayStatus::Succeeded,
                RunDisplayStatus::Succeeded,
            ),
            (
                terminated(TerminalStatus::Failed),
                RunDisplayStatus::Failed,
                RunDisplayStatus::Failed,
            ),
            (
                terminated(TerminalStatus::Cancelled),
                RunDisplayStatus::Cancelled,
                RunDisplayStatus::Cancelled,
            ),
            (
                RunState::DeadLettered,
                RunDisplayStatus::Failed,
                RunDisplayStatus::Failed,
            ),
            (
                RunState::Cancelling {
                    reason: None,
                    requested_at_ms: 0,
                },
                CancellationRequested,
                CancellationRequested,
            ),
            (
                RunState::Waiting { timeout_at_ms: 0 },
                RunDisplayStatus::Waiting,
                CancellationRequested,
            ),
            (
                RunState::Running,
                RunDisplayStatus::Running,
                CancellationRequested,
            ),
            (RunState::Pending, Queued, CancellationRequested),
            (
                RunState::Scheduled { run_at_ms: 0 },
                Queued,
                CancellationRequested,
            ),
        ] {
            assert_eq!(derive_display_status(&state, false), display, "{state:?}");
            assert_eq!(
                derive_display_status(&state, true),
                cancel_display,
                "{state:?} with a pending cancellation"
            );
        }
    }

    /// A reader over a store with the runs `queries` submitted under their run
    /// ids, which the worker does not claim.
    async fn reader_with_runs(queries: &[(&str, &str)]) -> (QueueReader, Arc<dyn ObjectStore>) {
        use crate::state::{ResearchConfig, ResearchState};

        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let queue = Arc::new(Queue::open(object_store.clone(), "q").await.unwrap());
        // The worker is not started, so the submitted steps stay pending.
        let runtime = WorkflowRuntime::builder(
            queue.clone(),
            object_store.clone(),
            AlwaysFail,
            NoopTerminalHook,
        )
        .queue_name(WORKFLOW_QUEUE_NAME)
        .memo_prefix(WORKFLOW_MEMO_PREFIX)
        .build()
        .unwrap();
        for (run_id, query) in queries {
            runtime
                .submit(RunSpec {
                    run_id: Some(RunId::new(*run_id).unwrap()),
                    input: ResearchState::new(*query, ResearchConfig::new("m")).to_bytes(),
                    options: RunOptions {
                        headers: HashMap::from([(QUERY_HEADER.to_string(), query.to_string())]),
                        ..Default::default()
                    },
                    ..Default::default()
                })
                .await
                .unwrap();
        }

        // Opened after the writes, so the reader's initial view contains them
        // without waiting for a manifest poll.
        let reader = QueueReader::open_with_options(
            object_store.clone(),
            "q",
            ReaderOptions::default().mode(ReaderMode::FollowLatest),
        )
        .await
        .unwrap();
        (reader, object_store)
    }

    #[tokio::test]
    async fn list_runs_returns_the_newest_run_first() {
        let (reader, object_store) =
            reader_with_runs(&[("01A", "first query"), ("01B", "second query")]).await;

        let runs = list_runs(&workflow_view(&reader, object_store))
            .await
            .unwrap();
        let listed: Vec<(&str, Option<&str>)> = runs
            .iter()
            .map(|run| {
                (
                    run.status.run_id.as_str(),
                    run.headers.get(QUERY_HEADER).map(String::as_str),
                )
            })
            .collect();
        assert_eq!(
            listed,
            [("01B", Some("second query")), ("01A", Some("first query"))]
        );

        reader.close().await.unwrap();
    }

    #[tokio::test]
    async fn count_waiting_excludes_terminal_notifications() {
        use taquba::EnqueueOptions;

        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let queue = Queue::open(object_store, "q").await.unwrap();
        for run in ["01A", "01B"] {
            queue
                .enqueue_with(
                    WORKFLOW_QUEUE_NAME,
                    Vec::new(),
                    EnqueueOptions::default().header(HEADER_RUN_ID, run),
                )
                .await
                .unwrap();
        }
        queue
            .enqueue_with(
                WORKFLOW_QUEUE_NAME,
                Vec::new(),
                EnqueueOptions::default()
                    .header(HEADER_RUN_ID, "01A")
                    .header(HEADER_TERMINAL, "true"),
            )
            .await
            .unwrap();

        assert_eq!(count_waiting_step_jobs(queue.view()).await.unwrap(), 2);
    }

    struct AlwaysFail;

    impl taquba_workflow::StepRunner for AlwaysFail {
        async fn run_step(
            &self,
            _step: &taquba_workflow::Step,
        ) -> Result<taquba_workflow::StepOutcome, StepError> {
            Err(StepError::permanent("simulated permanent failure"))
        }
    }
}
