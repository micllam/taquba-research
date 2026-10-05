//! Durable research agent for Rust, built on [Rig] and [taquba-workflow].
//!
//! `taquba-research` is a multi-step research agent whose runs persist across
//! process crashes. A run consists of a planning step, a fan-out of search and
//! page-fetch steps, per-page summarisation, an investigation of the gaps by a
//! Rig agent with tools, synthesis and a final report-writing step. The agent
//! writes every transition to the object-storage-backed task queue of taquba
//! and memoizes every completed LLM call. An interrupted run resumes from the
//! last completed step and does not pay for any of its completed model calls
//! again.
//!
//! This crate is a reference implementation and a CLI tool. It is a worked
//! example of how Rig (LLM orchestration) and the taquba stack (durable queues,
//! workflows and jobs) combine into a crash-safe agent. It is not intended as a
//! general-purpose library dependency. To make another Rig agent durable, copy
//! the pattern: per-step memoization of LLM calls over
//! [`taquba_workflow::Memo`], and the mapping of Rig errors to a transient or
//! permanent [`StepError`](workflow::StepError). The phase state machine of
//! this crate is specific to research.
//!
//! Both public surfaces run or embed this research agent:
//!
//! - **High-level**: [`ResearchAgent`], a builder that combines Rig, a
//!   [`SearchBackend`](search::SearchBackend) and a [`ResearchConfig`] into a
//!   `run(queue, object_store, query)` helper.
//! - **Low-level**: [`ResearchStepRunner`], a [`taquba_workflow::StepRunner`]
//!   for a caller's own [`taquba_workflow::WorkflowRuntime`].
//!
//! # Providers
//!
//! OpenAI, Anthropic and Ollama (local models) are supported through Rig.
//!
//! - **CLI**: `--provider openai|anthropic|ollama` selects the provider.
//!   Without it, the CLI selects Anthropic when `ANTHROPIC_API_KEY` is set and
//!   `OPENAI_API_KEY` is not, and OpenAI otherwise. The CLI never selects
//!   Ollama automatically, so an Ollama run requires `--provider ollama`.
//!   Ollama does not use an API key and connects to `http://localhost:11434`
//!   unless `OLLAMA_API_BASE_URL` is set.
//! - **Library**: [`ResearchStepRunner::new_openai`],
//!   [`ResearchStepRunner::new_anthropic`] and
//!   [`ResearchStepRunner::new_ollama`] build the runner, and the
//!   `.openai(...)`, `.anthropic(...)` and `.ollama(...)` methods of
//!   [`ResearchAgent::builder`] configure the agent.
//! - **Citations**: Anthropic runs pass the fetched pages as citation-enabled
//!   document blocks during synthesis. When Claude returns citation metadata,
//!   the final report includes the cited source excerpts. OpenAI and Ollama
//!   runs keep the standard numeric list of sources.
//! - **Structured phases**: the `Planning`, `Summarizing` and `Investigating`
//!   phases use structured (`prompt_typed`) completions, and `Investigating`
//!   also uses tool calls. An Ollama model must emit schema-valid JSON and tool
//!   calls reliably, and a model that does not dead-letters those steps.
//!
//! # Quick start
//!
//! ```no_run
//! use std::sync::Arc;
//! use taquba::{Queue, object_store::local::LocalFileSystem};
//! use taquba_research::{ResearchAgent, ResearchConfig, search::Tavily};
//!
//! # async fn run() -> anyhow::Result<()> {
//! let store = Arc::new(LocalFileSystem::new_with_prefix("./store")?);
//! let queue = Arc::new(Queue::open(store.clone(), "taquba-research").await?);
//!
//! let rig = rig_core::providers::openai::OpenAI::from_env()?;
//!
//! // ...or .anthropic(rig_core::providers::anthropic::Anthropic::from_env()?)
//! //       with a matching model id, for example "claude-haiku-4-5".
//! let agent = ResearchAgent::builder()
//!     .openai(rig)
//!     .search(Tavily::from_env()?)
//!     .config(ResearchConfig::new("gpt-5-nano"))
//!     .build()?;
//!
//! // `store` also backs the per-step memo of the workflow, which
//! // short-circuits retried LLM calls.
//! let report = agent
//!     .run(queue, store, "Postgres vs SQLite for read-heavy workloads")
//!     .await?;
//! println!("{}", report.markdown);
//! # Ok(()) }
//! ```
//!
//! # Command-line interface
//!
//! `cargo install taquba-research` installs a single `taquba-research` binary
//! on `$PATH`. The default invocation starts a new run. Ctrl+C stops the
//! process at any time, and the run persists in the configured store.
//!
//! ```text
//! taquba-research "your query"            # start a new run
//! taquba-research resume <RUN_ID>         # resume an interrupted run
//! taquba-research list                    # list past runs
//! taquba-research status <RUN_ID>         # show recorded status
//! taquba-research show <RUN_ID>           # print the rendered report
//! taquba-research show <RUN_ID> --output  # also accepts s3:// / gs:// / az:// / local path
//! taquba-research cancel <RUN_ID>         # cooperatively cancel
//! taquba-research init                    # fail-fast store reachability check
//! taquba-research gc --older-than-days 7  # delete old runs from the index + reports
//! ```
//!
//! `run` and `resume` are foreground processes that stay alive for the duration
//! of the work. They require `TAVILY_API_KEY` and the API key of the chosen
//! `--provider`. The other subcommands (`list`, `status`, `show`, `cancel`,
//! `init` and `gc`) inspect or maintain the shared store, from another shell
//! while a run is in flight or at any later time, and they require neither key.
//! Every subcommand reads object-store credentials (the standard `AWS_*`,
//! `GOOGLE_*` or `AZURE_*` environment variables) when `--store` is a cloud
//! URL.
//!
//! See `taquba-research --help` for the full flag list.
//!
//! # Storage
//!
//! `--store` (or `TAQUBA_RESEARCH_STORE`) sets the location of the SlateDB
//! queue, the [run index](store::RunIndexEntry) and, by default, the rendered
//! report. It accepts either of these forms:
//!
//! - A local path, `~/.taquba-research/` by default.
//! - An object-storage URL, such as `s3://bucket/prefix`, `gs://bucket/prefix`,
//!   `az://container/prefix` or `file:///abs/path`.
//!
//! Cloud URLs require the matching cargo feature:
//!
//! ```bash
//! cargo install taquba-research --features aws    # S3 / MinIO
//! cargo install taquba-research --features gcp    # Google Cloud Storage
//! cargo install taquba-research --features azure  # Azure Blob
//! ```
//!
//! The CLI always saves the report to `<store>/reports/<run_id>.md` in the
//! store of the queue, so an S3-backed deployment keeps the report, the queue
//! and the index in one bucket. `--output` accepts the same path-or-URL form
//! and writes an additional copy there.
//!
//! # Durability model
//!
//! - **A retry does not pay for a model call again.** Each LLM-backed phase
//!   memoizes its output in its per-step [`Memo`](taquba_workflow::Memo). An
//!   at-least-once redelivery short-circuits to the stored value and does not
//!   call or bill the model again.
//! - **A retried investigation replays its agent.** The investigating step
//!   records each completion and tool result of its Rig agent in the per-step
//!   memo, and a redelivery replays the recorded results. A failed call is not
//!   recorded, so a retry calls it again.
//! - **The delivery lease covers every slow call.** Every LLM and search call
//!   runs under a timeout, and the step extends its lease
//!   ([`taquba::LeaseHandle::ensure_at_least`]) by that bound before the call.
//!   The fetching step extends the lease again as each page job completes. A
//!   hung call times out as a transient step error and is retried.
//!
//! Inherited from taquba:
//!
//! - **Single-process, single-writer.** All workers for a queue run in one
//!   binary and share one `Arc<Queue>`.
//! - **At-least-once delivery.** Steps must be idempotent for
//!   `(run_id, step_number)`. The runtime deduplicates by that pair while a
//!   step is pending or scheduled.
//! - **Per-transition durability.** Every state change of a step is a SlateDB
//!   write to the configured object store (local FS, S3, GCS or Azure).
//!
//! ## Fetching is the one fan-out phase
//!
//! Every other phase is one workflow step per unit of work. The fetching phase
//! is a single workflow step that submits one `FetchPage` job per URL to a
//! [`JobRunner`](taquba_workflow::jobs::JobRunner), then awaits the handles
//! with `try_join_all`. The `JobRunner` shares the queue and uses a distinct
//! queue name. The per-URL `idempotency_key` derives from `(run_id, url)`, so
//! on a step retry the idempotent submit of the job runner returns the recorded
//! result, and no URL is fetched twice. The `fetch_page` tool of the
//! investigating step submits the same job for each page it reads.
//!
//! [`spawn_fetch_runner`] builds and spawns this `JobRunner`.
//! `ResearchAgent::run` and the CLI call it internally, and a caller with a
//! custom [`WorkflowRuntime`](taquba_workflow::WorkflowRuntime) will need to
//! call it and attach the runner with [`ResearchStepRunner::with_job_runner`]
//! and [`ResearchStepRunner::with_queue`]. With the queue attached, the
//! fetching step cancels its in-flight `FetchPage` jobs through `Queue::cancel`
//! when the run is cancelled, and the jobs do not run until the per-fetch HTTP
//! timeout.
//!
//! See [taquba-workflow's docs] for the underlying runtime semantics.
//!
//! [Rig]: https://crates.io/crates/rig-core
//! [taquba-workflow]: https://crates.io/crates/taquba-workflow
//! [taquba-workflow's docs]: https://docs.rs/taquba-workflow

#![warn(missing_docs)]
#![forbid(unsafe_code)]

mod agent;
mod fetch_job;
mod investigate;
mod journal;
mod report;
mod runner;
/// Web-search backends of the searching phase. An implementation of the
/// [`SearchBackend`](search::SearchBackend) trait adds another backend.
pub mod search;
mod state;
/// Run-level index of submitted runs, stored in the user KV namespace of the
/// queue, and the cancellation sentinel. The `list`, `status`, `show`, `cancel`
/// and `gc` subcommands of the CLI use it. See [`store::RunIndexEntry`].
pub mod store;

pub use agent::{ResearchAgent, ResearchAgentBuilder};
pub use fetch_job::{FETCH_QUEUE_NAME, spawn_fetch_runner};
pub use report::{Citation, Report, RunStats};
pub use runner::{ResearchStepRunner, RunRecord};
pub use state::{Phase, ResearchConfig, StateSummary, TokenUsage, summarize_state};
pub use store::{CancelSentinel, TerminalReconciler};

/// Re-exports of the workflow runtime types for a custom
/// [`WorkflowRuntime`](taquba_workflow::WorkflowRuntime) around
/// [`ResearchStepRunner`].
pub mod workflow {
    pub use taquba_workflow::{
        Delivery, EffectsHandle, NoopTerminalHook, RunId, RunOptions, RunOutcome, RunSpec,
        RunState, RunStatus, RunTermination, Step, StepError, StepErrorKind, StepOutcome,
        StepRunner, SubmitOutcome, TerminalEffects, TerminalHook, TerminalStatus, WorkflowRuntime,
        WorkflowView,
    };
}

/// Re-exports of the [`taquba_workflow::jobs`] types for the management of the
/// fetch [`JobRunner`](taquba_workflow::jobs::JobRunner) that
/// [`spawn_fetch_runner`] returns.
pub mod jobs {
    pub use taquba_workflow::jobs::{JobRunner, RunnerHandle};
}
