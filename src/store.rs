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
//! - **At termination**: for runner-issued outcomes (`Succeed` /
//!   `Cancel`) the terminal record joins the terminal step's
//!   settlement transaction via
//!   [`Step::effects`](taquba_workflow::Step::effects); for
//!   terminations that apply no step effects (a dead-lettered step, an
//!   external cancellation) it joins the terminal notification's
//!   settlement, staged by [`TerminalReconciler`].
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
use taquba::{JobRecord, JobStatus, Queue, QueueReader};
use taquba_workflow::{
    HEADER_RUN_ID, HEADER_TERMINAL, RunOutcome, StepError, TerminalEffects, TerminalHook,
    TerminalStatus,
};

use crate::state::{ResearchState, TokenUsage};

/// Queue name the CLI and [`crate::ResearchAgent`] configure on the
/// workflow runtime. Set explicitly so the reader-side queries in this
/// module target the same queue as the runtime.
pub const WORKFLOW_QUEUE_NAME: &str = "research-workflow";

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
    /// Terminal facts, present once the run terminated. Runner-issued
    /// outcomes (`Succeed`, `Cancel`) stage the record in the terminal
    /// step's settlement; outcomes that apply no step effects are
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
    /// Cancellation reason or failure message, when there is one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Wall-clock instant of the terminal outcome.
    pub finished_at: DateTime<Utc>,
    /// Summary statistics, so `status` prints without fetching the
    /// report.
    pub summary: RunSummary,
}

/// Terminal status stored in a [`TerminalRecord`]. Only terminal
/// outcomes are stored; the full display set is [`RunDisplayStatus`],
/// derived at read time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredStatus {
    /// Reached `StepOutcome::Succeed` and produced a report.
    Succeeded,
    /// The run terminated as failed (a dead-lettered step); recorded
    /// by [`TerminalReconciler`].
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
    /// Terminal record says failed and the dead-letter job is no
    /// longer present.
    Failed,
    /// Terminal record says cancelled.
    Cancelled,
    /// A step job for the run is in the dead-letter set, with or
    /// without a `Failed` terminal record.
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
    /// No step job and no terminal record: a dead-lettered run whose
    /// dead job the reaper removed before [`TerminalReconciler`]
    /// processed its notification, store corruption or a version
    /// mismatch. Collectable via the CLI's `gc --status unknown`.
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
/// A `Failed` record whose dead-letter job is still present derives
/// [`RunDisplayStatus::DeadLettered`]; once the reaper removes the
/// job it derives [`RunDisplayStatus::Failed`].
pub fn derive_display_status(
    entry: &RunIndexEntry,
    job: Option<&StepJobState>,
    cancel_requested: bool,
) -> RunDisplayStatus {
    if let Some(terminal) = &entry.terminal {
        return match terminal.status {
            StoredStatus::Succeeded => RunDisplayStatus::Succeeded,
            StoredStatus::Failed => match job {
                Some(StepJobState::Dead(_)) => RunDisplayStatus::DeadLettered,
                _ => RunDisplayStatus::Failed,
            },
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

/// Terminal-hook decorator reconciling the run index with outcomes
/// that applied no step effects. A dead-lettered step (and an external
/// [`WorkflowRuntime::cancel`](taquba_workflow::WorkflowRuntime::cancel))
/// terminates a run without staging its terminal record; this hook
/// stages the missing record on the notification's
/// [`TerminalEffects`], so it commits atomically with the
/// notification's acknowledgement. Entries whose record was staged
/// step-side are left unchanged, and a retried notification stages
/// the record again.
///
/// Wraps the host's own hook: reconciliation runs first and `inner`
/// is invoked only after it succeeds. Every notification is
/// processed, including one for a run another process submitted.
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
            .kv_get(&key)
            .await
            .map_err(|e| StepError::transient(format!("reading run index entry: {e}")))?;
        let Some(bytes) = bytes else {
            // Not a run this index manages, or the entry was already
            // collected.
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
            other => {
                tracing::warn!(
                    run_id = %outcome.run_id,
                    status = %other,
                    "skipping reconciliation of unknown terminal status"
                );
                return Ok(());
            }
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

    /// Best-effort summary for a reconciled record. A failed run's
    /// progress is decoded from its dead-letter job's payload; a
    /// succeeded outcome's from its `RunRecord` result. When neither
    /// source is available the summary holds the wall time alone.
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

    /// Final persisted state of `run_id`'s dead-lettered step, when
    /// its dead-letter job is still present and its payload decodes.
    async fn dead_job_state(&self, run_id: &str) -> Option<ResearchState> {
        let mut cursor: Option<Vec<u8>> = None;
        loop {
            let page = self
                .queue
                .list_jobs(
                    WORKFLOW_QUEUE_NAME,
                    JobStatus::Dead,
                    cursor.as_deref(),
                    SCAN_PAGE,
                )
                .await
                .ok()?;
            for job in page.jobs {
                if job.headers.contains_key(HEADER_TERMINAL) {
                    continue;
                }
                if job.headers.get(HEADER_RUN_ID).map(String::as_str) == Some(run_id) {
                    return ResearchState::from_bytes(&job.payload).ok();
                }
            }
            match page.next_cursor {
                Some(next) => cursor = Some(next),
                None => return None,
            }
        }
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

    // Reconciliation needs every notification, regardless of what
    // `inner` would observe.
    fn observes(&self, _outcome: &RunOutcome) -> bool {
        true
    }
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

    #[test]
    fn failed_record_with_dead_job_derives_dead_lettered() {
        let e = entry(Some(terminal(StoredStatus::Failed, Some("boom"))));
        let dead = StepJobState::Dead(job(&[(HEADER_RUN_ID, "01TESTRUN")]));
        assert_eq!(
            derive_display_status(&e, Some(&dead), false),
            RunDisplayStatus::DeadLettered
        );
    }

    #[test]
    fn failed_record_without_dead_job_derives_failed() {
        let e = entry(Some(terminal(StoredStatus::Failed, Some("boom"))));
        assert_eq!(
            derive_display_status(&e, None, false),
            RunDisplayStatus::Failed
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
        use taquba_workflow::{RunSpec, WorkflowRuntime};

        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let queue = Arc::new(Queue::open(object_store.clone(), "q").await.unwrap());
        let run_id = "01RECONCILE";

        let (tx, rx) = tokio::sync::oneshot::channel();
        let hook = TerminalReconciler::new(
            queue.clone(),
            Signal {
                tx: std::sync::Mutex::new(Some(tx)),
            },
        );
        // The reconciler's dead-job scan targets WORKFLOW_QUEUE_NAME;
        // the runtime must use the same queue name.
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
                run_id: Some(run_id.to_string()),
                input: state.to_bytes(),
                kv_writes: [(run_entry_key(run_id), e.to_bytes())].into(),
                ..Default::default()
            })
            .await
            .unwrap();

        // A permanent step error dead-letters immediately, with no
        // retry backoff.
        let outcome = rx.await.unwrap();
        assert_eq!(outcome.status, TerminalStatus::Failed);
        let _ = shutdown_tx.send(());
        let _ = worker.await;

        // The reconciled record committed with the notification's
        // acknowledgement; the summary comes from the dead job's
        // payload.
        let bytes = queue.kv_get(&run_entry_key(run_id)).await.unwrap().unwrap();
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
