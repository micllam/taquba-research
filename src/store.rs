//! Run-level index of submitted runs, stored in the queue's user KV
//! namespace, plus the cross-process cancellation sentinel.
//!
//! The index stores only what cannot be derived from the queue. An
//! entry is written at most twice per run:
//!
//! - **At submission** (query, submit time), joining the submit
//!   transaction via
//!   [`RunSpec::kv_writes`](taquba_workflow::RunSpec::kv_writes), so a
//!   run cannot exist without an entry or an entry without a run.
//! - **At voluntary termination** (`Succeed` / `Cancel` outcomes),
//!   joining the terminal step's settlement transaction via
//!   [`Step::effects`](taquba_workflow::Step::effects): the terminal
//!   status, an error for cancellations and a small summary.
//!
//! Every in-flight status is derived at read time from
//! [`QueueReader`](taquba::QueueReader)-visible state; see
//! [`derive_display_status`](crate::store::derive_display_status). The
//! cancellation sentinel remains a plain object at
//! `<store>/runs/<run_id>.cancel`, written by the `cancel` command and
//! polled by the runner concurrently with phase work.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use serde::{Deserialize, Serialize};
use taquba::object_store;
use taquba::{JobRecord, JobStatus, QueueReader};
use taquba_workflow::{HEADER_RUN_ID, HEADER_TERMINAL};

use crate::state::TokenUsage;

/// Queue name the CLI and [`crate::ResearchAgent`] configure on the
/// workflow runtime. Set explicitly so the reader-side queries in this
/// module target the same queue as the runtime.
pub const WORKFLOW_QUEUE_NAME: &str = "research-workflow";

/// Prefix of run index entries in the queue's user KV namespace.
pub const RUNS_KV_PREFIX: &str = "research/runs/";

/// KV key of `run_id`'s index entry. Run ids are ULIDs assigned at
/// submission, so a scan over [`RUNS_KV_PREFIX`] returns entries in
/// submission order.
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
    /// Terminal facts, present once the run terminated voluntarily
    /// (`Succeed` or `Cancel`). Absent while the run is in flight and
    /// for dead-lettered runs, whose failure is derived from the
    /// queue's dead-letter set.
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

/// Terminal facts recorded on a [`RunIndexEntry`] at voluntary
/// termination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalRecord {
    /// How the run terminated.
    pub status: StoredStatus,
    /// Cancellation reason or failure message, when there is one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Wall-clock instant of the terminal outcome.
    pub finished_at: DateTime<Utc>,
    /// Summary statistics, so `status` prints without fetching the
    /// report.
    pub summary: RunSummary,
}

/// Terminal status stored in a [`TerminalRecord`]. Only voluntary
/// terminations are stored; the full display set is
/// [`RunDisplayStatus`], derived at read time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredStatus {
    /// Reached `StepOutcome::Succeed` and produced a report.
    Succeeded,
    /// Reached a runner-verdict terminal failure.
    Failed,
    /// The runner observed the cancellation sentinel and terminated
    /// the run.
    Cancelled,
}

impl StoredStatus {
    /// Stable lowercase identifier, matching the serde encoding.
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

/// Display status of a run, computed at read time from the stored
/// entry, the queue's job state and the cancellation sentinel. See
/// [`derive_display_status`] for the precedence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunDisplayStatus {
    /// Terminal record says succeeded.
    Succeeded,
    /// Terminal record says failed (runner verdict).
    Failed,
    /// Terminal record says cancelled.
    Cancelled,
    /// A step job for the run is in the dead-letter set.
    DeadLettered,
    /// The cancellation sentinel exists but no terminal record does;
    /// the cancellation takes effect on the runner's next step.
    CancellationRequested,
    /// A step job for the run is claimed by a worker.
    Running,
    /// A step job is pending or scheduled: either an interrupted run
    /// awaiting `resume`, or the interval between an acknowledgement
    /// and the next claim.
    Queued,
    /// No step job and no terminal record. Reachable through store
    /// corruption or a version mismatch.
    Unknown,
}

impl RunDisplayStatus {
    /// Human-readable label used in CLI output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::DeadLettered => "failed (dead-lettered)",
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

/// A run's step job as observed in the queue, in decreasing precedence
/// order for status derivation.
#[derive(Debug, Clone)]
pub enum StepJobState {
    /// A step job is in the dead-letter set.
    Dead(JobRecord),
    /// A step job is claimed by a worker.
    Claimed(JobRecord),
    /// A step job is pending or scheduled.
    Waiting(JobRecord),
}

impl StepJobState {
    /// The observed job record, regardless of state.
    pub fn job(&self) -> &JobRecord {
        match self {
            Self::Dead(j) | Self::Claimed(j) | Self::Waiting(j) => j,
        }
    }
}

/// Compute a run's display status. Precedence: stored terminal record,
/// then dead-lettered step job, then cancellation sentinel, then
/// claimed step job, then pending/scheduled step job, then unknown.
pub fn derive_display_status(
    entry: &RunIndexEntry,
    job: Option<&StepJobState>,
    cancel_requested: bool,
) -> RunDisplayStatus {
    if let Some(terminal) = &entry.terminal {
        return match terminal.status {
            StoredStatus::Succeeded => RunDisplayStatus::Succeeded,
            StoredStatus::Failed => RunDisplayStatus::Failed,
            StoredStatus::Cancelled => RunDisplayStatus::Cancelled,
        };
    }
    match job {
        Some(StepJobState::Dead(_)) => RunDisplayStatus::DeadLettered,
        _ if cancel_requested => RunDisplayStatus::CancellationRequested,
        Some(StepJobState::Claimed(_)) => RunDisplayStatus::Running,
        Some(StepJobState::Waiting(_)) => RunDisplayStatus::Queued,
        None => RunDisplayStatus::Unknown,
    }
}

/// Page size for KV and job-listing scans.
const SCAN_PAGE: usize = 256;

/// Enumerate every run index entry, oldest first (run ids are ULIDs,
/// so key order is submission order). Malformed entries are logged and
/// skipped.
pub async fn list_runs(reader: &QueueReader) -> taquba::Result<Vec<RunIndexEntry>> {
    let mut out = Vec::new();
    let mut cursor: Option<Vec<u8>> = None;
    loop {
        let page = reader
            .kv_scan(RUNS_KV_PREFIX.as_bytes(), cursor.as_deref(), SCAN_PAGE)
            .await?;
        for (key, value) in page.entries {
            match RunIndexEntry::from_bytes(&value) {
                Ok(entry) => out.push(entry),
                Err(e) => tracing::warn!(
                    key = %String::from_utf8_lossy(&key),
                    error = %e,
                    "skipping malformed run index entry"
                ),
            }
        }
        match page.next_cursor {
            Some(next) => cursor = Some(next),
            None => break,
        }
    }
    Ok(out)
}

/// Load one run's index entry. `Ok(None)` when the run is unknown; a
/// malformed entry is an error.
pub async fn get_run(reader: &QueueReader, run_id: &str) -> anyhow::Result<Option<RunIndexEntry>> {
    let Some(bytes) = reader.kv_get(&run_entry_key(run_id)).await? else {
        return Ok(None);
    };
    let entry = RunIndexEntry::from_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("malformed run index entry for {run_id}: {e}"))?;
    Ok(Some(entry))
}

/// Snapshot every step job of `queue` keyed by run id, for status
/// derivation. Terminal-notification jobs (reserved
/// `workflow.terminal` header) are excluded: they record a run's
/// termination. When a run has jobs in several states, the
/// highest-precedence one wins (dead, then claimed, then waiting).
pub async fn snapshot_step_jobs(
    reader: &QueueReader,
    queue: &str,
) -> taquba::Result<HashMap<String, StepJobState>> {
    let mut map: HashMap<String, StepJobState> = HashMap::new();

    let mut insert = |job: JobRecord, make: fn(JobRecord) -> StepJobState| {
        if job.headers.contains_key(HEADER_TERMINAL) {
            return;
        }
        if let Some(run_id) = job.headers.get(HEADER_RUN_ID) {
            map.entry(run_id.clone()).or_insert_with(|| make(job));
        }
    };

    // Every scan is driven by the page cursor: a page can be shorter
    // than the limit without being the last one (records whose
    // offloaded payload was removed mid-scan are dropped after the
    // cursor is computed), so a short page must not end the loop.
    for (status, make) in [
        (JobStatus::Dead, StepJobState::Dead as fn(_) -> _),
        (JobStatus::Claimed, StepJobState::Claimed),
        (JobStatus::Pending, StepJobState::Waiting),
        (JobStatus::Scheduled, StepJobState::Waiting),
    ] {
        let mut cursor: Option<Vec<u8>> = None;
        loop {
            let page = reader
                .list_jobs(queue, status, cursor.as_deref(), SCAN_PAGE)
                .await?;
            for job in page.jobs {
                insert(job, make);
            }
            match page.next_cursor {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }
    }
    Ok(map)
}

/// Handle to the per-run cancellation sentinels inside the configured
/// object store, at `<prefix>/runs/<run_id>.cancel`.
///
/// Cheap to clone (internal `Arc<dyn ObjectStore>`).
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
    /// `prefix` is the key prefix within the store under which queue
    /// state and sentinels live; pass [`Path::default()`] to use the
    /// store's bucket root.
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

    /// Whether the cancellation sentinel exists.
    pub async fn is_set(&self, run_id: &str) -> bool {
        self.object_store.head(&self.path(run_id)).await.is_ok()
    }

    /// Instant the sentinel was written (its object's
    /// `last_modified`), or `None` when no sentinel exists.
    pub async fn requested_at(&self, run_id: &str) -> Option<DateTime<Utc>> {
        self.object_store
            .head(&self.path(run_id))
            .await
            .ok()
            .map(|meta| meta.last_modified)
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

    fn job(headers: &[(&str, &str)]) -> JobRecord {
        let encoded = serde_json::json!({
            "id": "01JOB",
            "queue": WORKFLOW_QUEUE_NAME,
            "payload": [],
            "status": "Pending",
            "attempts": 0,
            "max_attempts": 3,
            "enqueued_at": 0,
            "priority": 1000,
            "headers": headers
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<HashMap<_, _>>(),
        });
        serde_json::from_value(encoded).expect("JobRecord fields")
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
    fn terminal_record_wins_over_everything() {
        let e = entry(Some(terminal(StoredStatus::Succeeded, None)));
        let dead = StepJobState::Dead(job(&[(HEADER_RUN_ID, "01TESTRUN")]));
        assert_eq!(
            derive_display_status(&e, Some(&dead), true),
            RunDisplayStatus::Succeeded
        );
    }

    #[test]
    fn dead_letter_wins_over_sentinel_and_live_jobs() {
        let e = entry(None);
        let dead = StepJobState::Dead(job(&[(HEADER_RUN_ID, "01TESTRUN")]));
        assert_eq!(
            derive_display_status(&e, Some(&dead), true),
            RunDisplayStatus::DeadLettered
        );
    }

    #[test]
    fn sentinel_wins_over_claimed_and_waiting() {
        let e = entry(None);
        let claimed = StepJobState::Claimed(job(&[(HEADER_RUN_ID, "01TESTRUN")]));
        assert_eq!(
            derive_display_status(&e, Some(&claimed), true),
            RunDisplayStatus::CancellationRequested
        );
        assert_eq!(
            derive_display_status(&e, None, true),
            RunDisplayStatus::CancellationRequested
        );
    }

    #[test]
    fn claimed_and_waiting_derive_running_and_queued() {
        let e = entry(None);
        let claimed = StepJobState::Claimed(job(&[(HEADER_RUN_ID, "01TESTRUN")]));
        let waiting = StepJobState::Waiting(job(&[(HEADER_RUN_ID, "01TESTRUN")]));
        assert_eq!(
            derive_display_status(&e, Some(&claimed), false),
            RunDisplayStatus::Running
        );
        assert_eq!(
            derive_display_status(&e, Some(&waiting), false),
            RunDisplayStatus::Queued
        );
    }

    #[test]
    fn no_job_and_no_terminal_record_is_unknown() {
        let e = entry(None);
        assert_eq!(
            derive_display_status(&e, None, false),
            RunDisplayStatus::Unknown
        );
    }

    #[tokio::test]
    async fn reader_serves_entries_and_step_job_snapshot() {
        use taquba::object_store::memory::InMemory;
        use taquba::{EnqueueOptions, Queue, ReaderMode, ReaderOptions};

        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let queue = Queue::open(object_store.clone(), "q").await.unwrap();

        let e = entry(None);
        queue
            .kv_put(&run_entry_key(&e.run_id), &e.to_bytes())
            .await
            .unwrap();
        queue
            .enqueue_with(
                WORKFLOW_QUEUE_NAME,
                Vec::new(),
                EnqueueOptions {
                    headers: [(HEADER_RUN_ID.to_string(), e.run_id.clone())].into(),
                    ..EnqueueOptions::default()
                },
            )
            .await
            .unwrap();

        // Opened after the writes, so the reader's initial view holds
        // them without waiting for a manifest poll.
        let reader = QueueReader::open_with_options(
            object_store,
            "q",
            ReaderOptions {
                mode: ReaderMode::FollowLatest,
                ..ReaderOptions::default()
            },
        )
        .await
        .unwrap();

        let runs = list_runs(&reader).await.unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].run_id, e.run_id);

        let jobs = snapshot_step_jobs(&reader, WORKFLOW_QUEUE_NAME)
            .await
            .unwrap();
        assert!(matches!(
            jobs.get(&e.run_id),
            Some(StepJobState::Waiting(_))
        ));
        assert_eq!(
            derive_display_status(&runs[0], jobs.get(&e.run_id), false),
            RunDisplayStatus::Queued
        );

        reader.close().await.unwrap();
    }

    #[tokio::test]
    async fn cancel_sentinel_round_trip() {
        use taquba::object_store::memory::InMemory;
        let sentinel = CancelSentinel::new(Arc::new(InMemory::new()), &Path::default());
        assert!(!sentinel.is_set("run-1").await);
        assert!(sentinel.requested_at("run-1").await.is_none());
        sentinel.mark("run-1").await.unwrap();
        assert!(sentinel.is_set("run-1").await);
        assert!(sentinel.requested_at("run-1").await.is_some());
        sentinel.clear("run-1").await.unwrap();
        assert!(!sentinel.is_set("run-1").await);
        // Clearing an absent sentinel is not an error.
        sentinel.clear("run-1").await.unwrap();
    }
}
