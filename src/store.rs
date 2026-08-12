//! Run-level index of submitted runs, persisted in the same
//! [`ObjectStore`](taquba::object_store::ObjectStore) as the SlateDB
//! queue.
//!
//! The taquba `Queue` keys jobs by job-id (each step is one job), so it
//! can't answer "list all runs" or "find run X" directly. This module
//! maintains a parallel index keyed by run-id to support those queries,
//! and is what the CLI's `list`/`status`/`show`/`cancel`/`init`/`gc`
//! commands operate against.
//!
//! Two objects per run, both under the configured store:
//!
//! - `<store>/runs/<run_id>.json`:
//!   [`RunIndexEntry`](crate::store::RunIndexEntry): query, status, and
//!   the rendered [`crate::Report`] once the run finishes.
//! - `<store>/runs/<run_id>.cancel`: sentinel object written by the
//!   `cancel` command. The runner polls for it concurrently with the
//!   in-flight phase work and fails the run with `Cancelled` once it
//!   appears.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use object_store::path::Path;
use object_store::{ObjectStore, PutPayload};
use serde::{Deserialize, Serialize};
use taquba::object_store;

use crate::report::Report;

/// What we persist per run alongside the SlateDB queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunIndexEntry {
    /// Workflow runtime run identifier.
    pub run_id: String,
    /// Original query passed to the runner.
    pub query: String,
    /// Wall-clock submission time.
    pub submitted_at: DateTime<Utc>,
    /// Lifecycle status of the run.
    pub status: RunIndexStatus,
    /// Rendered report (`Some` once `status == Succeeded`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub report: Option<Report>,
    /// Failure reason (`Some` once `status == Failed`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Last instant the index entry was rewritten.
    pub updated_at: DateTime<Utc>,
}

/// Lifecycle status persisted in [`RunIndexEntry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunIndexStatus {
    /// Submitted and a worker is actively processing it (or at least
    /// claiming the next step).
    Running,
    /// Submitted but no worker is processing it right now. Set when the
    /// CLI is interrupted (Ctrl+C) before the run reaches a terminal
    /// step; `resume <id>` flips it back to [`Running`](Self::Running).
    Paused,
    /// Reached `StepOutcome::Succeed` and produced a report.
    Succeeded,
    /// Reached terminal failure (runner verdict, dead-letter, or step
    /// budget exhaustion). Distinct from
    /// [`Cancelled`](Self::Cancelled): `Failed` is an unintended stop
    /// worth investigating, whereas `Cancelled` reflects deliberate
    /// user intent.
    Failed,
    /// `cancel <id>` was called but the cancellation hasn't yet taken
    /// effect; the runner observes the sentinel at the start of its
    /// next step and transitions the run to
    /// [`Cancelled`](Self::Cancelled). If no worker is running, this
    /// state persists until one starts.
    CancellationRequested,
    /// Cancellation has taken effect: the runner observed the sentinel
    /// and acked the step.
    Cancelled,
}

impl RunIndexStatus {
    /// Stable lowercase identifier used in CLI output, matches the
    /// serde-encoded variant name on disk.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Paused => "paused",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::CancellationRequested => "cancellation_requested",
            Self::Cancelled => "cancelled",
        }
    }
}

impl std::fmt::Display for RunIndexStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Handle to the run index inside the configured object store.
///
/// Cheap to clone (internal `Arc<dyn ObjectStore>`).
#[derive(Clone)]
pub struct RunStore {
    object_store: Arc<dyn ObjectStore>,
    runs_prefix: Path,
}

impl std::fmt::Debug for RunStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunStore")
            .field("runs_prefix", &self.runs_prefix.as_ref())
            .field("object_store", &self.object_store.to_string())
            .finish()
    }
}

impl RunStore {
    /// Build a `RunStore` rooted at `<prefix>/runs/` inside
    /// `object_store`. `prefix` is the key prefix within the store under
    /// which both queue state and run index live; pass
    /// [`Path::default()`] to use the store's bucket root.
    pub fn new(object_store: Arc<dyn ObjectStore>, prefix: &Path) -> Self {
        let runs_prefix = prefix.child("runs");
        Self {
            object_store,
            runs_prefix,
        }
    }

    /// Prefix the run index objects live under, e.g. `foo/runs`.
    pub fn runs_prefix(&self) -> &Path {
        &self.runs_prefix
    }

    /// Object key for `run_id`'s index entry.
    pub fn entry_path(&self, run_id: &str) -> Path {
        self.runs_prefix.child(format!("{run_id}.json"))
    }

    /// Object key for `run_id`'s cancellation sentinel.
    pub fn cancel_path(&self, run_id: &str) -> Path {
        self.runs_prefix.child(format!("{run_id}.cancel"))
    }

    /// Persist `entry`, overwriting any prior version.
    pub async fn put(&self, entry: &RunIndexEntry) -> object_store::Result<()> {
        let path = self.entry_path(&entry.run_id);
        let bytes = serde_json::to_vec_pretty(entry).expect("RunIndexEntry serde");
        self.object_store
            .put(&path, PutPayload::from(bytes))
            .await
            .map(|_| ())
    }

    /// Load a run index entry.
    pub async fn get(&self, run_id: &str) -> object_store::Result<Option<RunIndexEntry>> {
        let path = self.entry_path(run_id);
        match self.object_store.get(&path).await {
            Ok(resp) => {
                let bytes = resp.bytes().await?;
                let entry: RunIndexEntry =
                    serde_json::from_slice(&bytes).map_err(|e| object_store::Error::Generic {
                        store: "RunStore",
                        source: Box::new(e),
                    })?;
                Ok(Some(entry))
            }
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Enumerate every run in the index, sorted by `submitted_at`
    /// descending. Malformed or unreadable entries are logged and
    /// skipped rather than failing the whole listing.
    pub async fn list(&self) -> object_store::Result<Vec<RunIndexEntry>> {
        use futures_util::TryStreamExt;
        let mut out = Vec::new();
        let mut stream = self.object_store.list(Some(&self.runs_prefix));
        while let Some(meta) = stream.try_next().await? {
            // Only `.json` entries are run index files; `.cancel`
            // sentinels share the prefix.
            if !meta.location.as_ref().ends_with(".json") {
                continue;
            }
            match self.object_store.get(&meta.location).await {
                Ok(resp) => match resp.bytes().await {
                    Ok(bytes) => match serde_json::from_slice::<RunIndexEntry>(&bytes) {
                        Ok(rec) => out.push(rec),
                        Err(e) => tracing::warn!(
                            path = %meta.location,
                            error = %e,
                            "skipping malformed run index"
                        ),
                    },
                    Err(e) => tracing::warn!(
                        path = %meta.location,
                        error = %e,
                        "skipping unreadable run index"
                    ),
                },
                Err(e) => tracing::warn!(
                    path = %meta.location,
                    error = %e,
                    "skipping unreadable run index"
                ),
            }
        }
        out.sort_by_key(|e| std::cmp::Reverse(e.submitted_at));
        Ok(out)
    }

    /// Write the cancellation sentinel for `run_id`.
    pub async fn mark_cancelled(&self, run_id: &str) -> object_store::Result<()> {
        self.object_store
            .put(&self.cancel_path(run_id), PutPayload::from_static(b""))
            .await
            .map(|_| ())
    }

    /// Whether the cancellation sentinel exists.
    pub async fn is_cancelled(&self, run_id: &str) -> bool {
        self.object_store
            .head(&self.cancel_path(run_id))
            .await
            .is_ok()
    }
}
