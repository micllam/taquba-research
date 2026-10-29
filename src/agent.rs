//! Convenience wrapper around a Rig client, a search backend, a
//! [`taquba_workflow::WorkflowRuntime`] and a one-shot terminal hook, exposed
//! through a single `run(queue, query)` call.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use rig_core::providers::{anthropic, ollama, openai};
use taquba::Queue;
use taquba::object_store::ObjectStore;
use taquba_workflow::{
    RunId, RunOptions, RunOutcome, RunSpec, StepError, TerminalEffects, TerminalHook,
    TerminalStatus, WorkflowRuntime,
};
use tokio::sync::Mutex;
use tokio::sync::oneshot;

use crate::Report;
use crate::fetch_job::spawn_fetch_runner;
use crate::runner::{ProviderClient, ResearchStepRunner, RunRecord};
use crate::search::SearchBackend;
use crate::state::ResearchConfig;
use crate::store::{
    QUERY_HEADER, WORKFLOW_CONTROL_QUEUE_NAME, WORKFLOW_MEMO_PREFIX, WORKFLOW_QUEUE_NAME,
    count_waiting_step_jobs,
};

/// How long workflow memo blobs are retained after the run reaches a terminal
/// state. The window exceeds the longest realistic run wall-time plus an
/// inspection buffer, so every in-process at-least-once retry occurs within it.
const MEMO_RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// High-level convenience that runs a research agent once and returns a
/// [`Report`], with the workflow runtime managed internally.
///
/// Build with [`ResearchAgent::builder`].
pub struct ResearchAgent {
    runner: ResearchStepRunner,
    config: ResearchConfig,
    inbox: bool,
}

impl ResearchAgent {
    /// Start configuring a research agent.
    pub fn builder() -> ResearchAgentBuilder {
        ResearchAgentBuilder::default()
    }

    /// Submit `query` to a fresh run, run the workflow runtime until the run
    /// terminates, and return the rendered [`Report`].
    ///
    /// `object_store` backs the workflow runtime's [memo store] used to
    /// short-circuit at-least-once retries of paid LLM calls. The common case
    /// is to pass the same store the [`Queue`] was opened with. The finished
    /// report is also written to `reports/<run_id>.md` in `object_store` before
    /// the final step settles.
    ///
    /// [memo store]: taquba_workflow::Memo
    pub async fn run(
        &self,
        queue: Arc<Queue>,
        object_store: Arc<dyn ObjectStore>,
        query: impl Into<String>,
    ) -> Result<Report> {
        let query = query.into();
        // The run id is generated before submit so the terminal hook can filter
        // notifications to this run.
        let run_id = RunId::new(ulid::Ulid::new().to_string()).expect("a ULID is a valid run id");
        let (tx, rx) = oneshot::channel::<RunOutcome>();
        let hook = CaptureOutcome {
            run_id: run_id.clone(),
            tx: Mutex::new(Some(tx)),
        };

        // The worker drains the shared workflow queue, so other queued runs'
        // pending steps execute in this call too.
        let waiting = count_waiting_step_jobs(queue.view())
            .await
            .context("counting queued runs")?;
        if waiting > 0 {
            tracing::warn!(
                waiting,
                "queued runs exist in this store; this worker will execute their pending steps"
            );
        }

        // Build the JobRunner the Fetching step submits FetchPage jobs to. It
        // shares the same Queue + ObjectStore as the workflow runtime,
        // distinguished by queue_name.
        let (job_runner, job_handle) = spawn_fetch_runner(&queue, &object_store);
        let runner = self
            .runner
            .clone()
            .with_job_runner(job_runner)
            .with_queue(queue.clone())
            .with_report_store(
                object_store.clone(),
                &taquba::object_store::path::Path::default(),
            );

        // The research workflow is strictly sequential: each
        // `StepOutcome::Continue` enqueues the next step only after the current
        // one acks. One worker is enough and avoids unnecessary claim
        // transaction conflicts.
        let mut builder = WorkflowRuntime::builder(queue.clone(), object_store, runner, hook)
            .queue_name(WORKFLOW_QUEUE_NAME)
            .memo_prefix(WORKFLOW_MEMO_PREFIX)
            .max_concurrent_steps(1)
            .memo_retention(MEMO_RETENTION);
        if self.inbox {
            builder = builder.control_queue(WORKFLOW_CONTROL_QUEUE_NAME);
        }
        let runtime = match builder.build().context("building the workflow runtime") {
            Ok(runtime) => runtime,
            Err(e) => {
                let _ = job_handle.shutdown().await;
                return Err(e);
            }
        };

        let input = ResearchStepRunner::initial_state(query.clone(), self.config.clone());
        // Submit before spawning the worker, so a submit failure has only the
        // fetch runner to shut down.
        if let Err(e) = runtime
            .submit(RunSpec {
                run_id: Some(run_id.clone()),
                input,
                options: RunOptions {
                    headers: HashMap::from([(QUERY_HEADER.to_string(), query.clone())]),
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .context("submitting research run")
        {
            let _ = job_handle.shutdown().await;
            return Err(e);
        }

        let worker_runtime = runtime.clone();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let mut worker = tokio::spawn(async move {
            worker_runtime
                .run(async move {
                    let _ = shutdown_rx.await;
                })
                .await
        });

        // Await the terminal signal and the worker together, so a worker task
        // that fails or panics before the run terminates ends this call with an
        // error and does not leave it waiting on `rx`.
        let outcome = tokio::select! {
            out = rx => {
                let _ = shutdown_tx.send(());
                // The outcome is already committed, so a worker exit error is
                // logged only.
                if let Err(e) = flatten_worker_exit((&mut worker).await) {
                    tracing::warn!(error = %e, "worker exit error after the terminal outcome");
                }
                out.map_err(|_| anyhow!("terminal hook dropped before signalling"))
            }
            joined = &mut worker => {
                // The worker exited before the terminal signal. The run's step
                // job stays in the queue.
                drop(shutdown_tx);
                Err(match flatten_worker_exit(joined) {
                    Ok(()) => anyhow!("worker exited before the run terminated"),
                    Err(e) => e.context("workflow worker failed before the run terminated"),
                })
            }
        };
        // Stop the JobRunner only after the workflow worker exits: the workflow
        // step that submits a FetchPage awaits it, so at this point the job
        // worker does not have a FetchPage in flight.
        let _ = job_handle.shutdown().await;
        let outcome = outcome?;

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
        }
    }
}

/// Builder for [`ResearchAgent`]. Required fields:
///
/// - A provider client (call [`Self::openai`], [`Self::anthropic`] or
///   [`Self::ollama`]). Last call wins if more than one is invoked.
/// - [`Self::search`]: a [`SearchBackend`] implementation.
/// - [`Self::config`]: a [`ResearchConfig`] built via [`ResearchConfig::new`]
///   with the model identifier of the run.
///
/// Optional: [`Self::inbox`], for the cancellation of a run from another
/// process.
#[derive(Default)]
pub struct ResearchAgentBuilder {
    provider: Option<ProviderClient>,
    search: Option<Arc<dyn SearchBackend>>,
    config: Option<ResearchConfig>,
    inbox: bool,
}

impl ResearchAgentBuilder {
    /// Set the Rig OpenAI client.
    pub fn openai(mut self, client: openai::OpenAI) -> Self {
        self.provider = Some(ProviderClient::OpenAi(Box::new(client)));
        self
    }

    /// Set the Rig Anthropic client.
    pub fn anthropic(mut self, client: anthropic::Anthropic) -> Self {
        self.provider = Some(ProviderClient::Anthropic(client));
        self
    }

    /// Set the Rig Ollama client, for local models.
    pub fn ollama(mut self, client: ollama::Ollama) -> Self {
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

    /// Set the control queue of the workflow runtime, so a `cancel` sent from
    /// another process through the inbox terminates the run.
    /// [`ResearchAgent::run`] then requires a queue opened as
    /// [`WORKFLOW_CONTROL_QUEUE_NAME`] describes. A reply reaches a waiting run
    /// without the control queue, through the inbox of [`WORKFLOW_QUEUE_NAME`].
    pub fn inbox(mut self) -> Self {
        self.inbox = true;
        self
    }

    /// Finalise the builder.
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
        Ok(ResearchAgent {
            runner: ResearchStepRunner::from_provider(provider, search),
            config,
            inbox: self.inbox,
        })
    }
}

/// Flatten a worker task's join result: a panic or abort (`JoinError`) and a
/// [`WorkflowRuntime::run`] error are both returned as the error.
fn flatten_worker_exit(
    joined: std::result::Result<taquba_workflow::Result<()>, tokio::task::JoinError>,
) -> Result<()> {
    Ok(joined??)
}

/// Terminal hook that forwards the [`RunOutcome`] on a oneshot channel so the
/// caller can `await` the whole run.
struct CaptureOutcome {
    /// Run this invocation submitted. `on_termination` acknowledges the
    /// notifications of every other run as stale.
    run_id: RunId,
    tx: Mutex<Option<oneshot::Sender<RunOutcome>>>,
}

impl TerminalHook for CaptureOutcome {
    async fn on_termination(
        &self,
        outcome: &RunOutcome,
        _effects: &TerminalEffects,
    ) -> std::result::Result<(), StepError> {
        // Terminal notifications are durable jobs: a notification left
        // unclaimed by a terminated process is delivered to the next worker on
        // the queue. A foreign notification is acked and logged without
        // forwarding its outcome.
        if outcome.run_id != self.run_id {
            tracing::warn!(
                run_id = %outcome.run_id,
                status = %outcome.status,
                "acknowledged stale terminal notification from another run"
            );
            return Ok(());
        }
        // Take the sender out of the mutex before sending so the lock guard is
        // not held across `tx.send`. A redelivered notification does not find a
        // sender and is a no-op.
        let tx = self.tx.lock().await.take();
        if let Some(tx) = tx {
            let _ = tx.send(outcome.clone());
        }
        Ok(())
    }
}
