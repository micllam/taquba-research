//! Durable research agent for Rust, built on [Rig] and [taquba-workflow].
//!
//! `taquba-research` is a multi-step research agent whose runs survive
//! process crashes. A run is decomposed into a planning step, a fan-out of
//! search and page-fetch steps, per-page summarization, synthesis, and a
//! final report-writing step. Every transition is persisted via taquba's
//! object-storage-backed task queue, so an interrupted process resumes
//! without re-paying for already-completed steps.
//!
//! Two surfaces are public:
//!
//! - **High-level**: [`ResearchAgent`] - a builder that wires Rig, a
//!   [`SearchBackend`](search::SearchBackend), and a [`ResearchConfig`]
//!   into a `run(queue, object_store, query)` helper.
//! - **Low-level**: [`ResearchStepRunner`] - a [`taquba_workflow::StepRunner`]
//!   you can drop into your own [`taquba_workflow::WorkflowRuntime`].
//!
//! # Providers
//!
//! OpenAI and Anthropic are both supported via Rig.
//!
//! - **CLI**: pass `--provider openai` (default) or
//!   `--provider anthropic`. If unset, the CLI picks based on which
//!   `*_API_KEY` env var is set: `ANTHROPIC_API_KEY` alone selects
//!   Anthropic; otherwise OpenAI.
//! - **Library**: build the runner via
//!   [`ResearchStepRunner::new_openai`] or
//!   [`ResearchStepRunner::new_anthropic`], or use the
//!   [`ResearchAgent::builder`]'s `.openai(...)` / `.anthropic(...)`.
//! - **Citations**: Anthropic runs pass fetched pages as
//!   citation-enabled document blocks during synthesis; when Claude
//!   returns citation metadata, the final report includes the cited
//!   source excerpts. OpenAI runs keep the standard numeric
//!   source-list citations.
//!
//! # Quick start
//!
//! ```no_run
//! use std::sync::Arc;
//! use rig_core::client::ProviderClient;
//! use taquba::{Queue, object_store::local::LocalFileSystem};
//! use taquba_research::{ResearchAgent, ResearchConfig, search::Tavily};
//!
//! # async fn run() -> anyhow::Result<()> {
//! let store = Arc::new(LocalFileSystem::new_with_prefix("./store")?);
//! let queue = Arc::new(Queue::open(store.clone(), "taquba-research").await?);
//!
//! let rig = rig_core::providers::openai::Client::from_env()?;
//!
//! // ...or .anthropic(rig_core::providers::anthropic::Client::from_env()?)
//! //       paired with a matching model id (e.g. "claude-haiku-4-5").
//! let agent = ResearchAgent::builder()
//!     .openai(rig)
//!     .search(Tavily::from_env()?)
//!     .config(ResearchConfig::new("gpt-4o-mini"))
//!     .build()?;
//!
//! // `store` also backs the workflow's per-step memo, which short-
//! // circuits LLM-call retries; sharing one store is the common case.
//! let report = agent
//!     .run(queue, store, "Postgres vs SQLite for read-heavy workloads")
//!     .await?;
//! println!("{}", report.markdown);
//! # Ok(()) }
//! ```
//!
//! # Command-line interface
//!
//! `cargo install taquba-research` puts a single `taquba-research` binary
//! on `$PATH`. The default invocation starts a new run; Ctrl+C any time
//! interrupts cleanly and the run survives in the configured store.
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
//! `run` and `resume` are **foreground processes** that stay alive for
//! the duration of the work and need `TAVILY_API_KEY` plus either
//! `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` depending on the chosen
//! `--provider`. The other subcommands (`list`, `status`,
//! `show`, `cancel`, `init`, `gc`) are inspection and maintenance
//! commands you run from **another shell** while a run is in flight
//! (or any time after); they only touch the shared store and need
//! neither key. Object-store credentials (the standard `AWS_*` /
//! `GOOGLE_*` / `AZURE_*` env vars) are read independently for every
//! subcommand whenever `--store` is a cloud URL.
//!
//! See `taquba-research --help` for the full flag list.
//!
//! # Storage
//!
//! `--store` (or `TAQUBA_RESEARCH_STORE`) controls where the SlateDB
//! queue, the [run index](store::RunIndexEntry), and (by default) the
//! rendered report all live. It accepts either:
//!
//! - a **local path** — default is `~/.taquba-research/`;
//! - an **object-storage URL** — `s3://bucket/prefix`,
//!   `gs://bucket/prefix`, `az://container/prefix`, `file:///abs/path`.
//!
//! Cloud URLs require the matching cargo feature:
//!
//! ```bash
//! cargo install taquba-research --features aws    # S3 / MinIO
//! cargo install taquba-research --features gcp    # Google Cloud Storage
//! cargo install taquba-research --features azure  # Azure Blob
//! ```
//!
//! `--output` accepts the same path-or-URL form. When omitted, the
//! report is persisted as `<store>/reports/<run_id>.md` in the
//! configured store (so an S3-backed deployment keeps the markdown,
//! queue, and index in the same bucket).
//!
//! # Durability model
//!
//! `taquba-research` inherits taquba's invariants:
//!
//! - **Single-process, single-writer.** All workers for one queue live in
//!   one binary and share one `Arc<Queue>`.
//! - **At-least-once delivery.** Steps must be idempotent for
//!   `(run_id, step_number)`; the runtime dedups by that pair while a step
//!   is pending or scheduled.
//! - **Per-transition durability.** Each step's state-change is a SlateDB
//!   write to the configured object store (local FS / S3 / GCS / Azure).
//!
//! ## Fetching is the one fan-out phase
//!
//! Most phases are one workflow step per unit of work. Fetching is the
//! exception: a single workflow step submits one `FetchPage` taquba-job
//! per URL to a [`JobRunner`](taquba_jobs::JobRunner) sharing the
//! queue (under a distinct queue-name), then `try_join_all`s the
//! handles. The per-URL `idempotency_key` derives from
//! `(run_id, url)`, so taquba-jobs's result-aware idempotent submit
//! short-circuits to cached result blobs on step retry; no URL is
//! fetched twice across attempts.
//!
//! [`spawn_fetch_runner`] is the helper that builds and spawns this
//! `JobRunner`; both `ResearchAgent::run` and the CLI construct it
//! internally, but callers driving a custom
//! [`WorkflowRuntime`](taquba_workflow::WorkflowRuntime) need to call
//! it themselves and attach the runner via
//! [`ResearchStepRunner::with_job_runner`] together with
//! [`ResearchStepRunner::with_queue`] (the latter lets the fetching
//! step cancel in-flight `FetchPage` jobs via `Queue::cancel` when
//! the surrounding run is cancelled, instead of letting them run
//! out to the per-fetch HTTP timeout).
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
mod report;
mod runner;
/// Pluggable web-search backends used by the fetch phase. Implement
/// the [`SearchBackend`](search::SearchBackend) trait to drop in any
/// other backend.
pub mod search;
mod state;
/// Run-level index of submitted runs, kept in the same `ObjectStore`
/// as the SlateDB queue. Backs the CLI's `list`/`status`/`show`/
/// `cancel`/`init`/`gc` subcommands. See [`store::RunIndexEntry`].
pub mod store;

pub use agent::{ResearchAgent, ResearchAgentBuilder};
pub use fetch_job::spawn_fetch_runner;
pub use report::{Citation, Report, RunStats};
pub use runner::{ResearchStepRunner, RunRecord};
pub use state::{ResearchConfig, TokenUsage};
pub use store::RunStore;

/// Re-exports of the workflow runtime types most users will need to wire a
/// custom [`WorkflowRuntime`](taquba_workflow::WorkflowRuntime) around
/// [`ResearchStepRunner`].
pub mod workflow {
    pub use taquba_workflow::{
        NoopTerminalHook, RunOutcome, RunSpec, Step, StepError, StepOutcome, StepRunner,
        SubmitOutcome, TerminalHook, TerminalStatus, WorkflowRuntime,
    };
}

/// Re-exports of the [`taquba_jobs`] types callers need to manage the
/// fetch [`JobRunner`](taquba_jobs::JobRunner) returned by
/// [`spawn_fetch_runner`].
pub mod jobs {
    pub use taquba_jobs::{JobRunner, RunnerHandle};
}
