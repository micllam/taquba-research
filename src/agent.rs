//! Convenience wrapper around a Rig client, a search backend, a
//! [`taquba_workflow::WorkflowRuntime`], and a one-shot terminal hook,
//! exposed through a single `run(queue, query)` call.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use rig_core::providers::openai;
use taquba::Queue;
use taquba_workflow::{RunOutcome, RunSpec, TerminalHook, TerminalStatus, WorkflowRuntime};
use tokio::sync::Mutex;
use tokio::sync::oneshot;

use crate::Report;
use crate::runner::{ResearchStepRunner, RunRecord};
use crate::search::SearchBackend;
use crate::state::ResearchConfig;
use crate::store::RunStore;

/// High-level convenience for running a research agent once and getting
/// back a [`Report`] without managing the workflow runtime yourself.
///
/// Build with [`ResearchAgent::builder`].
pub struct ResearchAgent {
    runner: ResearchStepRunner,
    config: ResearchConfig,
    run_store: Option<RunStore>,
}

impl ResearchAgent {
    /// Start configuring a research agent.
    pub fn builder() -> ResearchAgentBuilder {
        ResearchAgentBuilder::default()
    }

    /// Submit `query` to a fresh run, drive the workflow runtime until
    /// the run terminates, and return the rendered [`Report`].
    pub async fn run(&self, queue: Arc<Queue>, query: impl Into<String>) -> Result<Report> {
        let query = query.into();
        let (tx, rx) = oneshot::channel::<RunOutcome>();
        let hook = CaptureOutcome {
            tx: Mutex::new(Some(tx)),
        };

        // The research workflow is strictly sequential: each
        // `StepOutcome::Continue` enqueues the next step only after the
        // current one acks. One worker is enough and avoids unnecessary
        // claim transaction conflicts.
        let runtime = WorkflowRuntime::builder(queue, self.runner.clone(), hook)
            .max_concurrent_steps(1)
            .build();

        let worker_runtime = runtime.clone();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let worker = tokio::spawn(async move {
            worker_runtime
                .run(async move {
                    let _ = shutdown_rx.await;
                })
                .await
        });

        let input = ResearchStepRunner::initial_state(query.clone(), self.config.clone());
        let handle = runtime
            .submit(RunSpec {
                input,
                ..Default::default()
            })
            .await
            .context("submitting research run")?;

        let outcome = rx
            .await
            .map_err(|_| anyhow!("terminal hook dropped before signalling"))?;
        let _ = shutdown_tx.send(());
        let _ = worker.await;

        if let Some(store) = &self.run_store {
            persist_outcome(store, &outcome, &query).await?;
        }

        match outcome.status {
            TerminalStatus::Succeeded => {
                let result = outcome
                    .result
                    .ok_or_else(|| anyhow!("succeeded run has no result"))?;
                let record: RunRecord =
                    serde_json::from_slice(&result).context("decoding terminal RunRecord")?;
                record
                    .report
                    .ok_or_else(|| anyhow!("succeeded RunRecord without report"))
            }
            TerminalStatus::Failed => {
                let _ = handle;
                bail!(
                    "research run failed: {}",
                    outcome.error.unwrap_or_else(|| "(no reason)".to_string())
                );
            }
            TerminalStatus::Cancelled => {
                let _ = handle;
                bail!(
                    "research run cancelled: {}",
                    outcome.error.unwrap_or_else(|| "(no reason)".to_string())
                );
            }
            other => bail!("unknown terminal status: {other}"),
        }
    }
}

async fn persist_outcome(store: &RunStore, outcome: &RunOutcome, query: &str) -> Result<()> {
    use crate::store::{RunIndexEntry, RunIndexStatus};
    use chrono::Utc;

    let (status, report, error) = match outcome.status {
        TerminalStatus::Succeeded => {
            let bytes = outcome.result.clone().unwrap_or_default();
            let record: Result<RunRecord, _> = serde_json::from_slice(&bytes);
            match record {
                Ok(r) => (RunIndexStatus::Succeeded, r.report, None),
                Err(e) => (
                    RunIndexStatus::Succeeded,
                    None,
                    Some(format!("(report decode failed: {e})")),
                ),
            }
        }
        TerminalStatus::Failed => (RunIndexStatus::Failed, None, outcome.error.clone()),
        TerminalStatus::Cancelled => (RunIndexStatus::Cancelled, None, outcome.error.clone()),
        _ => (
            RunIndexStatus::Failed,
            None,
            Some(format!("unknown terminal status: {}", outcome.status)),
        ),
    };

    let now = Utc::now();
    let submitted_at = store
        .get(&outcome.run_id)
        .await
        .ok()
        .flatten()
        .map(|e| e.submitted_at)
        .unwrap_or(now);

    let entry = RunIndexEntry {
        run_id: outcome.run_id.clone(),
        query: query.to_string(),
        submitted_at,
        status,
        report,
        error,
        updated_at: now,
    };
    store
        .put(&entry)
        .await
        .context("persisting run index entry")?;
    Ok(())
}

/// Builder for [`ResearchAgent`]. Required fields:
///
/// - [`Self::openai`]: a Rig OpenAI client.
/// - [`Self::search`]: a [`SearchBackend`] implementation.
///
/// Optional:
///
/// - [`Self::config`]: defaults to [`ResearchConfig::default`].
/// - [`Self::run_store`]: filesystem index for CLI-style `list`/`status`.
#[derive(Default)]
pub struct ResearchAgentBuilder {
    rig: Option<openai::Client>,
    search: Option<Arc<dyn SearchBackend>>,
    config: Option<ResearchConfig>,
    run_store: Option<RunStore>,
}

impl ResearchAgentBuilder {
    /// Set the Rig OpenAI client.
    pub fn openai(mut self, client: openai::Client) -> Self {
        self.rig = Some(client);
        self
    }

    /// Set the search backend.
    pub fn search<B: SearchBackend>(mut self, backend: B) -> Self {
        self.search = Some(Arc::new(backend));
        self
    }

    /// Set the search backend from an already-shared `Arc`.
    pub fn search_arc(mut self, backend: Arc<dyn SearchBackend>) -> Self {
        self.search = Some(backend);
        self
    }

    /// Override the [`ResearchConfig`].
    pub fn config(mut self, config: ResearchConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Attach a [`RunStore`] for cross-process cancellation and CLI
    /// `list`/`status`/`show` visibility.
    pub fn run_store(mut self, store: RunStore) -> Self {
        self.run_store = Some(store);
        self
    }

    /// Finalize the builder.
    pub fn build(self) -> Result<ResearchAgent> {
        let rig = self
            .rig
            .ok_or_else(|| anyhow!("ResearchAgent requires an OpenAI Rig client"))?;
        let search = self
            .search
            .ok_or_else(|| anyhow!("ResearchAgent requires a SearchBackend"))?;
        let config = self.config.unwrap_or_default();
        let mut runner = ResearchStepRunner::new(rig, search);
        if let Some(store) = &self.run_store {
            runner = runner.with_run_store(store.clone());
        }
        Ok(ResearchAgent {
            runner,
            config,
            run_store: self.run_store,
        })
    }
}

/// Terminal hook that forwards the [`RunOutcome`] on a oneshot channel so
/// the caller can `await` the whole run.
struct CaptureOutcome {
    tx: Mutex<Option<oneshot::Sender<RunOutcome>>>,
}

impl TerminalHook for CaptureOutcome {
    async fn on_termination(&self, outcome: &RunOutcome) {
        // Take the sender out of the mutex before sending so the lock
        // guard isn't held across `tx.send`.
        let tx = self.tx.lock().await.take();
        if let Some(tx) = tx {
            let _ = tx.send(outcome.clone());
        }
    }
}
