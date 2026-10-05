//! Run-level index of submitted runs, stored in the queue's user KV namespace,
//! plus the cross-process cancellation sentinel.
//!
//! The index stores only what cannot be derived from the queue. An entry is
//! written at most twice per run:
//!
//! - **At submission** (query, submit time), joining the submit transaction via
//!   [`RunSpec::effects`](taquba_workflow::RunSpec::effects), so a run cannot
//!   exist without an entry or an entry without a run.
//! - **At termination**: for runner-issued outcomes (`Succeed` / `Cancel`) the
//!   terminal record joins the terminal step's settlement transaction via
//!   [`Step::effects`](taquba_workflow::Delivery::effects). For terminations
//!   that do not apply step effects (a dead-lettered step, an external
//!   cancellation) it joins the terminal notification's settlement, staged by
//!   [`TerminalReconciler`].
//!
//! Every in-flight status is derived at read time from the run state that a
//! [`WorkflowView`](taquba_workflow::WorkflowView) reports through a
//! [`QueueReader`](taquba::QueueReader), see
//! [`derive_display_status`](crate::store::derive_display_status). The
//! cancellation sentinel remains a plain object at
//! `<store>/runs/<run_id>.cancel`, written by the `cancel` command and polled
//! by the runner concurrently with phase work.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use futures_util::TryStreamExt;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use serde::{Deserialize, Serialize};
use taquba::object_store;
use taquba::{JobRecord, JobStatus, Queue, QueueReader, QueueView};
use taquba_workflow::{
    HEADER_RUN_ID, HEADER_TERMINAL, MemoStore, RunId, RunOutcome, RunState, StepError,
    TerminalEffects, TerminalHook, TerminalStatus, WorkflowView,
};

pub use crate::journal::JournalEntry;
use crate::state::{ResearchState, TokenUsage};

/// Name of the workflow queue. The CLI and [`crate::ResearchAgent`] configure
/// the workflow runtime with it, and the reader-side queries in this module
/// target the same queue.
pub const WORKFLOW_QUEUE_NAME: &str = "research-workflow";

/// Memo prefix the CLI and [`crate::ResearchAgent`] configure on the workflow
/// runtime. Set explicitly so [`workflow_view`] reads the memo store the
/// runtime writes to.
pub const WORKFLOW_MEMO_PREFIX: &str = "research-workflow-memo";

/// The read-only workflow queries over `reader` and the runtime's memo store in
/// `object_store`, for a process without a runtime.
pub fn workflow_view(reader: &QueueReader, object_store: Arc<dyn ObjectStore>) -> WorkflowView {
    WorkflowView::new(
        reader.view().clone(),
        MemoStore::new(object_store, WORKFLOW_MEMO_PREFIX),
    )
}

/// The records of the journal of the investigating step of `run_id`, from the
/// run memo in the runtime's memo store in `object_store`: each completion in
/// order, followed by the tool calls that it requested. The list is empty
/// before the investigating step starts.
pub async fn journal_entries(
    object_store: Arc<dyn ObjectStore>,
    run_id: &RunId,
) -> anyhow::Result<Vec<JournalEntry>> {
    let memo = MemoStore::new(object_store, WORKFLOW_MEMO_PREFIX).new_run_memo(run_id);
    crate::journal::read_entries(&memo, crate::investigate::JOURNAL_PREFIX).await
}

/// Prefix of run index entries in the queue's user KV namespace.
pub const RUNS_KV_PREFIX: &str = "research/runs/";

/// Prefix, under the store's key prefix, of canonical report blobs.
pub const REPORTS_PREFIX: &str = "reports";

/// Object key of `run_id`'s canonical report blob.
pub fn report_path(prefix: &Path, run_id: &str) -> Path {
    prefix
        .clone()
        .join(REPORTS_PREFIX)
        .join(format!("{run_id}.md"))
}

/// KV key of `run_id`'s index entry. Run ids are ULIDs assigned at submission,
/// so a scan over [`RUNS_KV_PREFIX`] returns entries in submission order.
pub fn run_entry_key(run_id: &str) -> Vec<u8> {
    format!("{RUNS_KV_PREFIX}{run_id}").into_bytes()
}

/// Run index entry stored under [`run_entry_key`]. JSON-encoded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunIndexEntry {
    /// Workflow runtime run identifier.
    pub run_id: String,
    /// Original query passed to the runner.
    pub query: String,
    /// Wall-clock submission time.
    pub submitted_at: DateTime<Utc>,
    /// Terminal facts, present once the run terminated. Runner-issued outcomes
    /// (`Succeed`, `Cancel`) stage the record in the terminal step's
    /// settlement. For outcomes that do not apply step effects, the record is
    /// staged by [`TerminalReconciler`] on the terminal notification's
    /// settlement. Absent while the run is in flight.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal: Option<TerminalRecord>,
}

impl RunIndexEntry {
    /// Encode the entry for storage under [`run_entry_key`].
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("RunIndexEntry is serde-derivable")
    }

    /// Decode an entry read from the KV namespace.
    pub fn from_bytes(bytes: &[u8]) -> serde_json::Result<Self> {
        serde_json::from_slice(bytes)
    }
}

/// Terminal facts recorded on a [`RunIndexEntry`] at termination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalRecord {
    /// How the run terminated.
    pub status: StoredStatus,
    /// Cancellation reason or failure message, when the run has either.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Wall-clock instant of the terminal outcome.
    pub finished_at: DateTime<Utc>,
    /// Summary statistics, so `status` prints without fetching the report.
    pub summary: RunSummary,
}

/// Terminal status stored in a [`TerminalRecord`]. Only terminal outcomes are
/// stored. The full display set is [`RunDisplayStatus`], derived at read time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredStatus {
    /// Reached `StepOutcome::Succeed` and produced a report.
    Succeeded,
    /// The run terminated as failed (a dead-lettered step). Recorded by
    /// [`TerminalReconciler`].
    Failed,
    /// The runner observed the cancellation sentinel and terminated the run.
    Cancelled,
}

impl StoredStatus {
    /// Stable lower-case identifier, matching the serde encoding.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

impl std::fmt::Display for StoredStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Summary statistics recorded on a [`TerminalRecord`].
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct RunSummary {
    /// Number of steps the runner completed.
    pub steps_completed: u32,
    /// Wall-clock seconds from submission to termination.
    pub wall_time_secs: u64,
    /// Aggregate token usage across every LLM call in the run.
    pub token_usage: TokenUsage,
}

/// Display status of a run, computed at read time from the stored entry, the
/// run state of the workflow view and the cancellation sentinel. See
/// [`derive_display_status`] for the precedence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunDisplayStatus {
    /// The terminal record, or the view's termination, says succeeded.
    Succeeded,
    /// The terminal record, or the view's termination, says failed. A
    /// dead-lettered step terminates its run as failed.
    Failed,
    /// The terminal record, or the view's termination, says cancelled.
    Cancelled,
    /// The cancellation sentinel exists and no terminal record does, or the
    /// view reports [`RunState::Cancelling`]. The cancellation takes effect on
    /// the runner's next step.
    CancellationRequested,
    /// The view reports [`RunState::Running`]: a worker claimed the current
    /// step. A claim abandoned by a killed worker process also derives this
    /// state: lease expiry is writer-process state, so the job stays claimed
    /// until the next writer open requeues it.
    Running,
    /// The view reports [`RunState::Pending`]: an interrupted run that awaits
    /// `resume`, or the interval between an acknowledgement and the next claim.
    /// A step dead-lettered outside the worker also reports pending, until the
    /// next worker terminates its run as failed.
    Queued,
    /// The run is absent from the view and the entry does not have a terminal
    /// record: the memo sweep removed the run's terminal record before a worker
    /// processed its notification, store corruption or a version mismatch.
    /// Collectable via the CLI's `gc --status unknown`.
    Unknown,
}

impl RunDisplayStatus {
    /// Human-readable label used in CLI output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::CancellationRequested => "cancellation requested",
            Self::Running => "running",
            Self::Queued => "queued",
            Self::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for RunDisplayStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The payload of `job`, a record of a job listing. An offloaded payload is
/// read through `view`, and `Ok(None)` means the job no longer exists.
pub async fn job_payload(view: &QueueView, job: &JobRecord) -> taquba::Result<Option<Vec<u8>>> {
    if job.payload_ref.is_none() {
        return Ok(Some(job.payload.clone()));
    }
    Ok(view.get_job(&job.id).await?.map(|job| job.payload))
}

/// Compute a run's display status from its index entry, the run state that
/// [`WorkflowView::status`] reports and the cancellation sentinel. Precedence:
/// the stored terminal record, then a termination of the view, then the
/// cancellation sentinel, then the view's cancelling, running and pending
/// states, then unknown.
pub fn derive_display_status(
    entry: &RunIndexEntry,
    state: Option<&RunState>,
    cancel_requested: bool,
) -> RunDisplayStatus {
    if let Some(terminal) = &entry.terminal {
        return match terminal.status {
            StoredStatus::Succeeded => RunDisplayStatus::Succeeded,
            StoredStatus::Failed => RunDisplayStatus::Failed,
            StoredStatus::Cancelled => RunDisplayStatus::Cancelled,
        };
    }
    match state {
        Some(RunState::Terminated(termination)) => match termination.status {
            TerminalStatus::Succeeded => RunDisplayStatus::Succeeded,
            TerminalStatus::Failed => RunDisplayStatus::Failed,
            TerminalStatus::Cancelled => RunDisplayStatus::Cancelled,
        },
        _ if cancel_requested => RunDisplayStatus::CancellationRequested,
        Some(RunState::Cancelling) => RunDisplayStatus::CancellationRequested,
        Some(RunState::Running) => RunDisplayStatus::Running,
        Some(RunState::Pending) => RunDisplayStatus::Queued,
        None => RunDisplayStatus::Unknown,
    }
}

/// Page size for KV and job-listing scans.
const SCAN_PAGE: usize = 256;

/// Enumerate every run index entry, oldest first (run ids are ULIDs, so key
/// order is submission order). Malformed entries are logged and skipped.
pub async fn list_runs(reader: &QueueReader) -> taquba::Result<Vec<RunIndexEntry>> {
    let mut out = Vec::new();
    let entries = reader
        .view()
        .kv_entries(RUNS_KV_PREFIX.as_bytes(), .., SCAN_PAGE);
    let mut entries = std::pin::pin!(entries);
    while let Some((key, value)) = entries.try_next().await? {
        match RunIndexEntry::from_bytes(&value) {
            Ok(entry) => out.push(entry),
            Err(e) => tracing::warn!(
                key = %String::from_utf8_lossy(&key),
                error = %e,
                "skipping malformed run index entry"
            ),
        }
    }
    Ok(out)
}

/// Load a run's index entry. `Ok(None)` when the run is unknown, and an error
/// for a malformed entry.
pub async fn get_run(reader: &QueueReader, run_id: &str) -> anyhow::Result<Option<RunIndexEntry>> {
    let Some(bytes) = reader.view().kv_get(&run_entry_key(run_id)).await? else {
        return Ok(None);
    };
    let entry = RunIndexEntry::from_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("malformed run index entry for {run_id}: {e}"))?;
    Ok(Some(entry))
}

/// The step job of `run_id` among the jobs of the workflow queue in `status`,
/// in the stored form of [`QueueView::list_jobs`]. `None` when the listing does
/// not have a step job of the run. A terminal notification (reserved
/// `workflow.terminal` header) is not a step job. The listing is a scan of
/// every job in `status`, so a caller passes the status that
/// [`WorkflowView::status`] reports for the run.
pub async fn find_step_job(
    view: &QueueView,
    run_id: &str,
    status: JobStatus,
) -> taquba::Result<Option<JobRecord>> {
    let mut cursor: Option<Vec<u8>> = None;
    loop {
        let page = view
            .list_jobs(WORKFLOW_QUEUE_NAME, status, cursor.as_deref(), SCAN_PAGE)
            .await?;
        let found = page.jobs.into_iter().find(|job| {
            !job.headers.contains_key(HEADER_TERMINAL)
                && job.headers.get(HEADER_RUN_ID).map(String::as_str) == Some(run_id)
        });
        if found.is_some() {
            return Ok(found);
        }
        match page.next_cursor {
            Some(next) => cursor = Some(next),
            None => return Ok(None),
        }
    }
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

/// Terminal-hook decorator reconciling the run index with outcomes that did not
/// apply step effects. A dead-lettered step (and an external
/// [`WorkflowRuntime::cancel`](taquba_workflow::WorkflowRuntime::cancel))
/// terminates a run without staging its terminal record. This hook stages the
/// missing record on the notification's [`TerminalEffects`], so it commits
/// atomically with the notification's acknowledgement. Entries whose record was
/// staged step-side are left unchanged, and a retried notification stages the
/// record again.
///
/// Wraps the host's own hook: reconciliation runs first and `inner` is invoked
/// only after it succeeds. Every notification is processed, including one for a
/// run another process submitted.
pub struct TerminalReconciler<H> {
    queue: Arc<Queue>,
    inner: H,
}

impl<H> TerminalReconciler<H> {
    /// Wrap `inner`, reading and staging index state through `queue`.
    pub fn new(queue: Arc<Queue>, inner: H) -> Self {
        Self { queue, inner }
    }

    async fn reconcile(
        &self,
        outcome: &RunOutcome,
        effects: &TerminalEffects,
    ) -> Result<(), StepError> {
        let key = run_entry_key(&outcome.run_id);
        let bytes = self
            .queue
            .view()
            .kv_get(&key)
            .await
            .map_err(|e| StepError::transient(format!("reading run index entry: {e}")))?;
        let Some(bytes) = bytes else {
            // Not a run this index manages, or the entry was already collected.
            return Ok(());
        };
        let mut entry = match RunIndexEntry::from_bytes(&bytes) {
            Ok(entry) => entry,
            Err(e) => {
                tracing::warn!(
                    run_id = %outcome.run_id,
                    error = %e,
                    "skipping reconciliation of malformed run index entry"
                );
                return Ok(());
            }
        };
        if entry.terminal.is_some() {
            return Ok(());
        }
        let status = match outcome.status {
            TerminalStatus::Succeeded => StoredStatus::Succeeded,
            TerminalStatus::Failed => StoredStatus::Failed,
            TerminalStatus::Cancelled => StoredStatus::Cancelled,
        };
        let finished_at = Utc::now();
        let summary = self
            .summary_for(outcome, entry.submitted_at, finished_at)
            .await;
        entry.terminal = Some(TerminalRecord {
            status,
            error: outcome.error.clone(),
            finished_at,
            summary,
        });
        effects
            .put(key, entry.to_bytes())
            .map_err(|e| StepError::permanent(format!("staging reconciled run index entry: {e}")))
    }

    /// Best-effort summary for a reconciled record. A failed run's progress is
    /// decoded from its dead-letter job's payload, and a succeeded outcome's
    /// from its `RunRecord` result. When neither source is available the
    /// summary contains the wall time alone.
    async fn summary_for(
        &self,
        outcome: &RunOutcome,
        submitted_at: DateTime<Utc>,
        finished_at: DateTime<Utc>,
    ) -> RunSummary {
        let mut summary = RunSummary {
            wall_time_secs: (finished_at - submitted_at)
                .to_std()
                .map(|d| d.as_secs())
                .unwrap_or_default(),
            ..RunSummary::default()
        };
        match outcome.status {
            TerminalStatus::Succeeded => {
                if let Some(result) = &outcome.result
                    && let Ok(record) = serde_json::from_slice::<crate::runner::RunRecord>(result)
                    && let Some(report) = record.report
                {
                    summary.steps_completed = report.stats.steps_completed;
                    summary.token_usage = report.stats.token_usage;
                }
            }
            TerminalStatus::Failed => {
                if let Some(state) = self.dead_job_state(&outcome.run_id).await {
                    summary.steps_completed = state.steps_completed;
                    summary.token_usage = state.token_usage;
                }
            }
            _ => {}
        }
        summary
    }

    /// Final persisted state of `run_id`'s dead-lettered step, when its
    /// dead-letter job is still present and its payload decodes.
    async fn dead_job_state(&self, run_id: &str) -> Option<ResearchState> {
        let view = self.queue.view();
        let job = find_step_job(view, run_id, JobStatus::Dead).await.ok()??;
        let payload = job_payload(view, &job).await.ok()??;
        ResearchState::from_bytes(&payload).ok()
    }
}

impl<H: TerminalHook> TerminalHook for TerminalReconciler<H> {
    async fn on_termination(
        &self,
        outcome: &RunOutcome,
        effects: &TerminalEffects,
    ) -> Result<(), StepError> {
        self.reconcile(outcome, effects).await?;
        self.inner.on_termination(outcome, effects).await
    }

    // Reconciliation needs every notification, regardless of what `inner`
    // observes.
    fn observes(&self, _outcome: &RunOutcome) -> bool {
        true
    }
}

/// Handle to the per-run cancellation sentinels inside the configured object
/// store, at `<prefix>/runs/<run_id>.cancel`.
///
/// A clone copies an internal `Arc<dyn ObjectStore>`.
#[derive(Clone)]
pub struct CancelSentinel {
    object_store: Arc<dyn ObjectStore>,
    runs_prefix: Path,
}

impl std::fmt::Debug for CancelSentinel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancelSentinel")
            .field("runs_prefix", &self.runs_prefix.as_ref())
            .field("object_store", &self.object_store.to_string())
            .finish()
    }
}

impl CancelSentinel {
    /// Build a handle rooted at `<prefix>/runs/` inside `object_store`.
    /// `prefix` is the key prefix within the store under which queue state and
    /// sentinels live. Pass [`Path::default()`] to use the store's bucket root.
    pub fn new(object_store: Arc<dyn ObjectStore>, prefix: &Path) -> Self {
        let runs_prefix = prefix.clone().join("runs");
        Self {
            object_store,
            runs_prefix,
        }
    }

    /// Object key for `run_id`'s cancellation sentinel.
    pub fn path(&self, run_id: &str) -> Path {
        self.runs_prefix.clone().join(format!("{run_id}.cancel"))
    }

    /// Write the cancellation sentinel for `run_id`.
    pub async fn mark(&self, run_id: &str) -> object_store::Result<()> {
        self.object_store
            .put(&self.path(run_id), PutPayload::from_static(b""))
            .await
            .map(|_| ())
    }

    /// Whether the cancellation sentinel exists. `Ok(false)` only for a missing
    /// sentinel. Any other `head` failure is returned.
    pub async fn is_set(&self, run_id: &str) -> object_store::Result<bool> {
        match self.object_store.head(&self.path(run_id)).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { .. }) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Instant the sentinel was written (its object's `last_modified`).
    /// `Ok(None)` only when no sentinel exists. Any other `head` failure is
    /// returned.
    pub async fn requested_at(&self, run_id: &str) -> object_store::Result<Option<DateTime<Utc>>> {
        match self.object_store.head(&self.path(run_id)).await {
            Ok(meta) => Ok(Some(meta.last_modified)),
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Remove the sentinel. Missing sentinels are not an error.
    pub async fn clear(&self, run_id: &str) -> object_store::Result<()> {
        match self.object_store.delete(&self.path(run_id)).await {
            Ok(()) => Ok(()),
            Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(e) => Err(e),
        }
    }
}

/// Correlation key of the signal that ends the wait of `run_id` for the reply
/// of the user. A [`WorkflowRuntime::signal`] with the reply as its payload
/// delivers it, and an empty payload wakes the run to check its cancellation.
///
/// [`WorkflowRuntime::signal`]: taquba_workflow::WorkflowRuntime::signal
pub fn clarification_signal_key(run_id: &str) -> String {
    format!("research/clarify/{run_id}")
}

/// Handle to the reply of the user to the question of a run, inside the
/// configured object store at `<prefix>/runs/<run_id>.reply`. The reply watcher
/// of a worker ([`crate::spawn_reply_watcher`]) delivers it to the waiting run.
///
/// A clone copies an internal `Arc<dyn ObjectStore>`.
#[derive(Clone)]
pub struct ReplyBox {
    object_store: Arc<dyn ObjectStore>,
    runs_prefix: Path,
}

impl std::fmt::Debug for ReplyBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReplyBox")
            .field("runs_prefix", &self.runs_prefix.as_ref())
            .field("object_store", &self.object_store.to_string())
            .finish()
    }
}

impl ReplyBox {
    /// Build a handle rooted at `<prefix>/runs/` inside `object_store`, the
    /// prefix of the cancellation sentinels.
    pub fn new(object_store: Arc<dyn ObjectStore>, prefix: &Path) -> Self {
        Self {
            object_store,
            runs_prefix: prefix.clone().join("runs"),
        }
    }

    /// Object key of the reply of `run_id`.
    pub fn path(&self, run_id: &str) -> Path {
        self.runs_prefix.clone().join(format!("{run_id}.reply"))
    }

    /// Write `reply` for `run_id`, replacing an earlier reply.
    pub async fn put(&self, run_id: &str, reply: &str) -> object_store::Result<()> {
        self.object_store
            .put(
                &self.path(run_id),
                PutPayload::from(reply.as_bytes().to_vec()),
            )
            .await
            .map(|_| ())
    }

    /// The reply of `run_id`. `Ok(None)` only for a missing reply. A reply that
    /// is not UTF-8 is decoded lossily.
    pub async fn get(&self, run_id: &str) -> object_store::Result<Option<String>> {
        let read = match self.object_store.get(&self.path(run_id)).await {
            Ok(resp) => resp.bytes().await,
            Err(e) => Err(e),
        };
        match read {
            Ok(bytes) => Ok(Some(String::from_utf8_lossy(&bytes).into_owned())),
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Remove the reply. A missing reply is not an error.
    pub async fn clear(&self, run_id: &str) -> object_store::Result<()> {
        match self.object_store.delete(&self.path(run_id)).await {
            Ok(()) | Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(terminal: Option<TerminalRecord>) -> RunIndexEntry {
        RunIndexEntry {
            run_id: "01TESTRUN".to_string(),
            query: "a query".to_string(),
            submitted_at: Utc::now(),
            terminal,
        }
    }

    fn terminal(status: StoredStatus, error: Option<&str>) -> TerminalRecord {
        TerminalRecord {
            status,
            error: error.map(str::to_string),
            finished_at: Utc::now(),
            summary: RunSummary {
                steps_completed: 7,
                wall_time_secs: 42,
                token_usage: TokenUsage::default(),
            },
        }
    }

    fn terminated(status: TerminalStatus) -> RunState {
        RunState::Terminated(taquba_workflow::RunTermination {
            status,
            error: None,
            error_kind: None,
            final_step: 0,
            terminated_at_ms: 0,
        })
    }

    #[test]
    fn entry_round_trips_with_terminal_record() {
        let e = entry(Some(terminal(StoredStatus::Cancelled, Some("by user"))));
        let back = RunIndexEntry::from_bytes(&e.to_bytes()).unwrap();
        let t = back.terminal.expect("terminal record");
        assert_eq!(t.status, StoredStatus::Cancelled);
        assert_eq!(t.error.as_deref(), Some("by user"));
        assert_eq!(t.summary.steps_completed, 7);
        assert_eq!(t.summary.wall_time_secs, 42);
    }

    #[test]
    fn entry_round_trips_without_terminal_record() {
        let e = entry(None);
        let bytes = e.to_bytes();
        // A submit-time entry serializes without a terminal key.
        assert!(!String::from_utf8_lossy(&bytes).contains("terminal"));
        let back = RunIndexEntry::from_bytes(&bytes).unwrap();
        assert!(back.terminal.is_none());
    }

    #[test]
    fn terminal_record_wins_over_the_view_state_and_the_sentinel() {
        let e = entry(Some(terminal(StoredStatus::Succeeded, None)));
        assert_eq!(
            derive_display_status(&e, Some(&terminated(TerminalStatus::Failed)), true),
            RunDisplayStatus::Succeeded
        );
        let e = entry(Some(terminal(StoredStatus::Failed, Some("boom"))));
        assert_eq!(
            derive_display_status(&e, Some(&RunState::Running), true),
            RunDisplayStatus::Failed
        );
        let e = entry(Some(terminal(StoredStatus::Cancelled, None)));
        assert_eq!(
            derive_display_status(&e, None, false),
            RunDisplayStatus::Cancelled
        );
    }

    #[test]
    fn a_termination_of_the_view_wins_over_the_sentinel() {
        let e = entry(None);
        assert_eq!(
            derive_display_status(&e, Some(&terminated(TerminalStatus::Failed)), true),
            RunDisplayStatus::Failed
        );
        assert_eq!(
            derive_display_status(&e, Some(&terminated(TerminalStatus::Succeeded)), true),
            RunDisplayStatus::Succeeded
        );
        assert_eq!(
            derive_display_status(&e, Some(&terminated(TerminalStatus::Cancelled)), true),
            RunDisplayStatus::Cancelled
        );
    }

    #[test]
    fn sentinel_wins_over_running_and_pending() {
        let e = entry(None);
        assert_eq!(
            derive_display_status(&e, Some(&RunState::Running), true),
            RunDisplayStatus::CancellationRequested
        );
        assert_eq!(
            derive_display_status(&e, Some(&RunState::Pending), true),
            RunDisplayStatus::CancellationRequested
        );
        assert_eq!(
            derive_display_status(&e, None, true),
            RunDisplayStatus::CancellationRequested
        );
    }

    #[test]
    fn the_active_states_of_the_view_derive_their_display_status() {
        let e = entry(None);
        assert_eq!(
            derive_display_status(&e, Some(&RunState::Cancelling), false),
            RunDisplayStatus::CancellationRequested
        );
        assert_eq!(
            derive_display_status(&e, Some(&RunState::Running), false),
            RunDisplayStatus::Running
        );
        assert_eq!(
            derive_display_status(&e, Some(&RunState::Pending), false),
            RunDisplayStatus::Queued
        );
    }

    #[test]
    fn no_run_state_and_no_terminal_record_is_unknown() {
        let e = entry(None);
        assert_eq!(
            derive_display_status(&e, None, false),
            RunDisplayStatus::Unknown
        );
    }

    #[tokio::test]
    async fn reader_serves_entries_and_the_run_state_of_a_submitted_run() {
        use crate::state::ResearchConfig;
        use taquba::object_store::memory::InMemory;
        use taquba::{Queue, ReaderMode, ReaderOptions};
        use taquba_workflow::{NoopTerminalHook, RunId, RunSpec, WorkflowRuntime};

        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let queue = Arc::new(Queue::open(object_store.clone(), "q").await.unwrap());
        // The worker is not started, so the submitted step stays pending.
        let runtime = WorkflowRuntime::builder(
            queue.clone(),
            object_store.clone(),
            AlwaysFail,
            NoopTerminalHook,
        )
        .queue_name(WORKFLOW_QUEUE_NAME)
        .memo_prefix(WORKFLOW_MEMO_PREFIX)
        .build();

        let e = entry(None);
        let run_id = RunId::new(&e.run_id).unwrap();
        runtime
            .submit(RunSpec {
                run_id: Some(run_id.clone()),
                input: ResearchState::new("a query", ResearchConfig::new("m")).to_bytes(),
                effects: taquba::SettlementEffects::default()
                    .kv_put(run_entry_key(&e.run_id), e.to_bytes()),
                ..Default::default()
            })
            .await
            .unwrap();

        // Opened after the writes, so the reader's initial view contains them
        // without waiting for a manifest poll.
        let reader = QueueReader::open_with_options(
            object_store.clone(),
            "q",
            ReaderOptions::default().mode(ReaderMode::FollowLatest),
        )
        .await
        .unwrap();

        let runs = list_runs(&reader).await.unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].run_id, e.run_id);

        let status = workflow_view(&reader, object_store)
            .status(&run_id)
            .await
            .unwrap()
            .expect("the view reports the submitted run");
        assert_eq!(status.state, RunState::Pending);
        assert_eq!(
            derive_display_status(&runs[0], Some(&status.state), false),
            RunDisplayStatus::Queued
        );
        let job = find_step_job(reader.view(), &e.run_id, JobStatus::Pending)
            .await
            .unwrap()
            .expect("the pending listing contains the step job");
        assert_eq!(job.headers.get(HEADER_RUN_ID), Some(&e.run_id));
        assert!(
            find_step_job(reader.view(), &e.run_id, JobStatus::Claimed)
                .await
                .unwrap()
                .is_none()
        );

        reader.close().await.unwrap();
    }

    #[tokio::test]
    async fn job_payload_reads_an_offloaded_payload() {
        use taquba::object_store::memory::InMemory;
        use taquba::{OpenOptions, Queue};

        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let queue = Queue::open_with_options(
            object_store,
            "q",
            OpenOptions::default().payload_offload_threshold(4),
        )
        .await
        .unwrap();
        let payload = b"an offloaded payload".to_vec();
        queue
            .enqueue(WORKFLOW_QUEUE_NAME, payload.clone())
            .await
            .unwrap();

        let page = queue
            .view()
            .list_jobs(WORKFLOW_QUEUE_NAME, JobStatus::Pending, None, 1)
            .await
            .unwrap();
        let job = &page.jobs[0];
        assert!(job.payload.is_empty());
        assert_eq!(job_payload(queue.view(), job).await.unwrap(), Some(payload));
    }

    #[tokio::test]
    async fn count_waiting_excludes_terminal_notifications() {
        use taquba::object_store::memory::InMemory;
        use taquba::{EnqueueOptions, Queue};

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

    struct Signal {
        tx: std::sync::Mutex<Option<tokio::sync::oneshot::Sender<RunOutcome>>>,
    }

    impl TerminalHook for Signal {
        async fn on_termination(
            &self,
            outcome: &RunOutcome,
            _effects: &TerminalEffects,
        ) -> Result<(), StepError> {
            if let Some(tx) = self.tx.lock().unwrap().take() {
                let _ = tx.send(outcome.clone());
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn reconciler_records_failed_for_dead_lettered_run() {
        use crate::state::ResearchConfig;
        use taquba::object_store::memory::InMemory;
        use taquba_workflow::{RunId, RunSpec, WorkflowRuntime};

        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let queue = Arc::new(Queue::open(object_store.clone(), "q").await.unwrap());
        let run_id = RunId::new("01RECONCILE").unwrap();

        let (tx, rx) = tokio::sync::oneshot::channel();
        let hook = TerminalReconciler::new(
            queue.clone(),
            Signal {
                tx: std::sync::Mutex::new(Some(tx)),
            },
        );
        // The reconciler's dead-job scan targets WORKFLOW_QUEUE_NAME, and the
        // runtime must use the same queue name.
        let runtime =
            WorkflowRuntime::builder(queue.clone(), object_store.clone(), AlwaysFail, hook)
                .queue_name(WORKFLOW_QUEUE_NAME)
                .max_concurrent_steps(1)
                .build();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let worker_runtime = runtime.clone();
        let worker = tokio::spawn(async move {
            worker_runtime
                .run(async move {
                    let _ = shutdown_rx.await;
                })
                .await
        });

        let mut state = ResearchState::new("a query", ResearchConfig::new("m"));
        state.steps_completed = 3;
        state.token_usage.total_tokens = 123;
        let e = RunIndexEntry {
            run_id: run_id.to_string(),
            query: "a query".to_string(),
            submitted_at: state.started_at,
            terminal: None,
        };
        runtime
            .submit(RunSpec {
                run_id: Some(run_id.clone()),
                input: state.to_bytes(),
                effects: taquba::SettlementEffects::default()
                    .kv_put(run_entry_key(&run_id), e.to_bytes()),
                ..Default::default()
            })
            .await
            .unwrap();

        // A permanent step error dead-letters immediately, without a retry
        // backoff.
        let outcome = rx.await.unwrap();
        assert_eq!(outcome.status, TerminalStatus::Failed);
        let _ = shutdown_tx.send(());
        let _ = worker.await;

        // The reconciled record committed with the notification's
        // acknowledgement. The summary comes from the dead job's payload.
        let bytes = queue
            .view()
            .kv_get(&run_entry_key(&run_id))
            .await
            .unwrap()
            .unwrap();
        let stored = RunIndexEntry::from_bytes(&bytes).unwrap();
        let terminal = stored.terminal.expect("reconciled terminal record");
        assert_eq!(terminal.status, StoredStatus::Failed);
        assert!(
            terminal
                .error
                .as_deref()
                .is_some_and(|e| e.contains("simulated permanent failure"))
        );
        assert_eq!(terminal.summary.steps_completed, 3);
        assert_eq!(terminal.summary.token_usage.total_tokens, 123);
    }

    #[tokio::test]
    async fn cancel_sentinel_round_trip() {
        use taquba::object_store::memory::InMemory;
        let sentinel = CancelSentinel::new(Arc::new(InMemory::new()), &Path::default());
        assert!(!sentinel.is_set("run-1").await.unwrap());
        assert!(sentinel.requested_at("run-1").await.unwrap().is_none());
        sentinel.mark("run-1").await.unwrap();
        assert!(sentinel.is_set("run-1").await.unwrap());
        assert!(sentinel.requested_at("run-1").await.unwrap().is_some());
        sentinel.clear("run-1").await.unwrap();
        assert!(!sentinel.is_set("run-1").await.unwrap());
        // Clearing an absent sentinel is not an error.
        sentinel.clear("run-1").await.unwrap();
    }

    #[tokio::test]
    async fn reply_box_round_trip() {
        use taquba::object_store::memory::InMemory;
        let replies = ReplyBox::new(Arc::new(InMemory::new()), &Path::default());
        assert!(replies.get("run-1").await.unwrap().is_none());
        replies.put("run-1", "the 2024 edition").await.unwrap();
        assert_eq!(
            replies.get("run-1").await.unwrap().as_deref(),
            Some("the 2024 edition")
        );
        replies.clear("run-1").await.unwrap();
        assert!(replies.get("run-1").await.unwrap().is_none());
        // Clearing an absent reply is not an error.
        replies.clear("run-1").await.unwrap();
    }
}
