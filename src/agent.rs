//! Convenience wrapper around a Rig client, a search backend, a
//! [`taquba_workflow::WorkflowRuntime`], and a one-shot terminal hook,
//! exposed through a single `run(queue, query)` call.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use chrono::Utc;
use rig_core::providers::{anthropic, ollama, openai};
use taquba::Queue;
use taquba::object_store::ObjectStore;
use taquba_workflow::{
    RunOutcome, RunSpec, StepError, TerminalEffects, TerminalHook, TerminalStatus, WorkflowRuntime,
};
use tokio::sync::Mutex;
use tokio::sync::oneshot;

use crate::Report;
use crate::fetch_job::spawn_fetch_runner;
use crate::runner::{ProviderClient, ResearchStepRunner, RunRecord};
use crate::search::SearchBackend;
use crate::state::ResearchConfig;
use crate::store::{CancelSentinel, RunIndexEntry, WORKFLOW_QUEUE_NAME, run_entry_key};

/// How long workflow memo blobs are retained after the run reaches a
/// terminal state. Any in-process at-least-once retry happens well
/// before this elapses; the window only needs to outlive the longest
/// realistic run wall-time plus an inspection buffer.
const MEMO_RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// High-level convenience for running a research agent once and getting
/// back a [`Report`] without managing the workflow runtime yourself.
///
/// Build with [`ResearchAgent::builder`].
pub struct ResearchAgent {
    runner: ResearchStepRunner,
    config: ResearchConfig,
}

impl ResearchAgent {
    /// Start configuring a research agent.
    pub fn builder() -> ResearchAgentBuilder {
        ResearchAgentBuilder::default()
    }

    /// Submit `query` to a fresh run, drive the workflow runtime until
    /// the run terminates, and return the rendered [`Report`].
    ///
    /// `object_store` backs the workflow runtime's [memo store] used to
    /// short-circuit at-least-once retries of paid LLM calls. The common
    /// case is to pass the same store the [`Queue`] was opened with.
    ///
    /// [memo store]: taquba_workflow::Memo
    pub async fn run(
        &self,
        queue: Arc<Queue>,
        object_store: Arc<dyn ObjectStore>,
        query: impl Into<String>,
    ) -> Result<Report> {
        let query = query.into();
        // The run id is generated before submit so the index entry's
        // KV key can join the submit transaction and the terminal hook
        // can filter notifications to this run.
        let run_id = ulid::Ulid::new().to_string();
        let (tx, rx) = oneshot::channel::<RunOutcome>();
        let hook = CaptureOutcome {
            run_id: run_id.clone(),
            tx: Mutex::new(Some(tx)),
        };

        // Build the JobRunner the Fetching step submits FetchPage jobs
        // to. It shares the same Queue + ObjectStore as the workflow
        // runtime, distinguished by queue_name.
        let (job_runner, job_handle) = spawn_fetch_runner(&queue, &object_store);
        let runner = self
            .runner
            .clone()
            .with_job_runner(job_runner)
            .with_queue(queue.clone());

        // The research workflow is strictly sequential: each
        // `StepOutcome::Continue` enqueues the next step only after the
        // current one acks. One worker is enough and avoids unnecessary
        // claim transaction conflicts.
        let runtime = WorkflowRuntime::builder(queue, object_store, runner, hook)
            .queue_name(WORKFLOW_QUEUE_NAME)
            .max_concurrent_steps(1)
            .memo_retention(MEMO_RETENTION)
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

        let entry = RunIndexEntry {
            run_id: run_id.clone(),
            query: query.clone(),
            submitted_at: Utc::now(),
            terminal: None,
        };
        let input = ResearchStepRunner::initial_state(query.clone(), self.config.clone());
        runtime
            .submit(RunSpec {
                run_id: Some(run_id.clone()),
                input,
                kv_writes: [(run_entry_key(&run_id), entry.to_bytes())].into(),
                ..Default::default()
            })
            .await
            .context("submitting research run")?;

        let outcome = rx
            .await
            .map_err(|_| anyhow!("terminal hook dropped before signalling"))?;
        let _ = shutdown_tx.send(());
        let _ = worker.await;
        // Stop the JobRunner only after the workflow worker has
        // drained: any still-in-flight FetchPage was awaited by the
        // workflow step that submitted it, so by the time we get here
        // there's nothing left for the job worker to do.
        let _ = job_handle.shutdown().await;

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
                bail!(
                    "research run failed: {}",
                    outcome.error.unwrap_or_else(|| "(no reason)".to_string())
                );
            }
            TerminalStatus::Cancelled => {
                bail!(
                    "research run cancelled: {}",
                    outcome.error.unwrap_or_else(|| "(no reason)".to_string())
                );
            }
            other => bail!("unknown terminal status: {other}"),
        }
    }
}

/// Builder for [`ResearchAgent`]. Required fields:
///
/// - A provider client (call [`Self::openai`], [`Self::anthropic`], or
///   [`Self::ollama`]). Last call wins if more than one is invoked.
/// - [`Self::search`]: a [`SearchBackend`] implementation.
/// - [`Self::config`]: a [`ResearchConfig`] built via
///   [`ResearchConfig::new`] with the model identifier you want.
///
/// Optional:
///
/// - [`Self::cancellation`]: sentinel handle for cross-process
///   cancellation.
#[derive(Default)]
pub struct ResearchAgentBuilder {
    provider: Option<ProviderClient>,
    search: Option<Arc<dyn SearchBackend>>,
    config: Option<ResearchConfig>,
    cancel: Option<CancelSentinel>,
}

impl ResearchAgentBuilder {
    /// Set the Rig OpenAI client.
    pub fn openai(mut self, client: openai::Client) -> Self {
        self.provider = Some(ProviderClient::OpenAi(client));
        self
    }

    /// Set the Rig Anthropic client.
    pub fn anthropic(mut self, client: anthropic::Client) -> Self {
        self.provider = Some(ProviderClient::Anthropic(client));
        self
    }

    /// Set the Rig Ollama client, for local models.
    pub fn ollama(mut self, client: ollama::Client) -> Self {
        self.provider = Some(ProviderClient::Ollama(client));
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

    /// Attach a [`CancelSentinel`] so a `cancel` issued from another
    /// process terminates the run.
    pub fn cancellation(mut self, sentinel: CancelSentinel) -> Self {
        self.cancel = Some(sentinel);
        self
    }

    /// Finalize the builder.
    pub fn build(self) -> Result<ResearchAgent> {
        let provider = self.provider.ok_or_else(|| {
            anyhow!(
                "ResearchAgent requires a provider client (call .openai(), .anthropic(), or .ollama())"
            )
        })?;
        let search = self
            .search
            .ok_or_else(|| anyhow!("ResearchAgent requires a SearchBackend"))?;
        let config = self
            .config
            .ok_or_else(|| anyhow!("ResearchAgent requires a ResearchConfig"))?;
        let mut runner = ResearchStepRunner::from_provider(provider, search);
        if let Some(sentinel) = self.cancel {
            runner = runner.with_cancellation(sentinel);
        }
        Ok(ResearchAgent { runner, config })
    }
}

/// Terminal hook that forwards the [`RunOutcome`] on a oneshot channel so
/// the caller can `await` the whole run.
struct CaptureOutcome {
    /// Run this invocation submitted. Notifications for any other run
    /// are stale; see `on_termination`.
    run_id: String,
    tx: Mutex<Option<oneshot::Sender<RunOutcome>>>,
}

impl TerminalHook for CaptureOutcome {
    async fn on_termination(
        &self,
        outcome: &RunOutcome,
        _effects: &TerminalEffects,
    ) -> std::result::Result<(), StepError> {
        // Terminal notifications are durable jobs: one left unclaimed
        // by a terminated process is delivered to the next worker on
        // the queue. Consuming a foreign notification here would
        // misattribute the outcome, so it is acked and logged; its
        // run's terminal index entry was committed by the step
        // settlement.
        if outcome.run_id != self.run_id {
            tracing::warn!(
                run_id = %outcome.run_id,
                status = %outcome.status,
                "acknowledged stale terminal notification from another run"
            );
            return Ok(());
        }
        // Take the sender out of the mutex before sending so the lock
        // guard is not held across `tx.send`. A redelivered
        // notification finds no sender and is a no-op.
        let tx = self.tx.lock().await.take();
        if let Some(tx) = tx {
            let _ = tx.send(outcome.clone());
        }
        Ok(())
    }
}
