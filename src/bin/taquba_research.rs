//! `taquba-research` CLI entry point.
//!
//! Subcommands:
//!
//! - default (positional `QUERY`): start a new research run.
//! - `resume <RUN_ID>`: resume an interrupted run.
//! - `list`: list past runs in this store.
//! - `status <RUN_ID>`: print the recorded status of a run.
//! - `show <RUN_ID> [--output ...]`: print or write the rendered report.
//! - `cancel <RUN_ID>`: cooperatively cancel an in-flight run.
//! - `init`: verify the configured store is reachable (fail-fast cred /
//!   bucket check before submitting an expensive run).
//! - `gc [--older-than-days N] [--status S]...`: delete recorded runs
//!   and their default-location reports.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use chrono::Utc;
use clap::{Parser, Subcommand, ValueEnum};
use rig_core::client::ProviderClient;
use rig_core::providers::{anthropic, ollama, openai};
use taquba::Queue;
use taquba::object_store::local::LocalFileSystem;
use taquba::object_store::path::Path as ObjectPath;
use taquba::object_store::{ObjectStore, PutPayload, parse_url};
use taquba_research::jobs::RunnerHandle;
use taquba_research::workflow::{
    RunOutcome, RunSpec, TerminalHook, TerminalStatus, WorkflowRuntime,
};
use taquba_research::{
    ResearchConfig, ResearchStepRunner, RunRecord, RunStore,
    search::{SearchBackend, Tavily},
    spawn_fetch_runner,
};
use tokio::sync::{Mutex, oneshot};
use tracing_subscriber::EnvFilter;
use url::Url;

const QUEUE_DB_NAME: &str = "queue";
const REPORTS_PREFIX: &str = "reports";
/// How long workflow memo blobs are retained after the run reaches a
/// terminal state. Matches the value used by the library's
/// `ResearchAgent::run`; keep in sync.
const MEMO_RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// Provider choice surfaced through `--provider`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CliProvider {
    #[value(name = "openai")]
    OpenAi,
    Anthropic,
    Ollama,
}

impl CliProvider {
    /// Default model identifier when the user doesn't pass `--model`.
    fn default_model(self) -> &'static str {
        match self {
            CliProvider::OpenAi => "gpt-4o-mini",
            CliProvider::Anthropic => "claude-haiku-4-5",
            CliProvider::Ollama => ollama::LLAMA3_2,
        }
    }

    /// Resolve `--provider` against env vars. If the user passed
    /// `--provider <p>` explicitly, that wins (the only way to select
    /// `ollama`, since it has no API key to auto-detect). Otherwise: if
    /// `ANTHROPIC_API_KEY` is set and `OPENAI_API_KEY` is not, pick
    /// Anthropic; otherwise default to OpenAI.
    fn resolve(explicit: Option<CliProvider>) -> CliProvider {
        if let Some(p) = explicit {
            return p;
        }
        let has_openai = std::env::var("OPENAI_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false);
        let has_anthropic = std::env::var("ANTHROPIC_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false);
        if has_anthropic && !has_openai {
            CliProvider::Anthropic
        } else {
            CliProvider::OpenAi
        }
    }
}

/// Durable research agent; single-binary CLI.
#[derive(Debug, Parser)]
#[command(
    name = "taquba-research",
    version,
    about = "Durable research agent for Rust",
    long_about = None,
    arg_required_else_help = false,
)]
struct Cli {
    /// Positional query for the default "start a new run" mode. If both
    /// this and a subcommand are given, the subcommand wins.
    query: Option<String>,

    /// Where to write the final report on completion. Accepts a local
    /// path or an object-storage URL (`s3://bucket/key.md`, `gs://...`,
    /// etc.). When omitted, the report is persisted as
    /// `<store>/reports/<run_id>.md` inside the configured store, so an
    /// S3-backed deployment keeps the markdown next to the queue and
    /// index.
    #[arg(long, value_parser = validate_store_arg)]
    output: Option<String>,

    /// Number of sub-questions to decompose the query into.
    #[arg(long, default_value_t = 6)]
    depth: usize,

    /// Cap URLs fetched per run.
    #[arg(long, default_value_t = 30)]
    max_sources: usize,

    /// LLM provider (`openai`, `anthropic`, or `ollama`). If unset, the
    /// CLI picks one based on which `*_API_KEY` env var is set:
    /// `ANTHROPIC_API_KEY` alone selects `anthropic`; otherwise `openai`
    /// is used. `ollama` (local models) is never auto-selected: pass
    /// `--provider ollama` explicitly.
    #[arg(long, value_enum)]
    provider: Option<CliProvider>,

    /// Specific model identifier passed to the provider. If unset,
    /// the CLI picks a provider-appropriate default
    /// (`gpt-4o-mini` for OpenAI, `claude-haiku-4-5` for Anthropic,
    /// `llama3.2` for Ollama).
    #[arg(long)]
    model: Option<String>,

    /// Search backend: only `tavily` is wired in v0.1.
    #[arg(long, default_value = "tavily")]
    search: String,

    /// Store location. Accepts either a local path or an object-storage
    /// URL (`s3://bucket/prefix`, `gs://...`, `az://...`, `file:///...`).
    /// Cloud URLs require the matching cargo feature
    /// (`--features aws`/`gcp`/`azure`). Defaults to `~/.taquba-research/`.
    #[arg(long, value_parser = validate_store_arg)]
    store: Option<String>,

    /// Suppress per-step output.
    #[arg(short, long)]
    quiet: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Resume an interrupted run.
    Resume {
        /// Run identifier to resume.
        run_id: String,
    },
    /// List past runs.
    List,
    /// Show the recorded status of a run.
    Status {
        /// Run identifier.
        run_id: String,
    },
    /// Print the rendered report for a finished run, or write it to a path / URL.
    Show {
        /// Run identifier.
        run_id: String,
        /// Optional destination. Same syntax as `run --output`. When
        /// omitted, the report is printed to stdout.
        #[arg(long, value_parser = validate_store_arg)]
        output: Option<String>,
    },
    /// Cooperatively cancel an in-flight run.
    Cancel {
        /// Run identifier.
        run_id: String,
    },
    /// Verify the configured store is reachable. Useful before running
    /// an expensive query to catch typos, missing creds, or unreachable
    /// buckets.
    Init,
    /// Delete recorded runs (and their default-location reports) by
    /// age and/or terminal status. Use `--dry-run` to preview.
    Gc {
        /// Delete only runs whose `submitted_at` is at least this many
        /// days in the past.
        #[arg(long)]
        older_than_days: Option<i64>,
        /// Restrict deletion to specific statuses (repeatable).
        /// Allowed: `running`, `paused`, `succeeded`, `failed`,
        /// `cancellation_requested`, `cancelled`. When unset, only
        /// terminal runs (`succeeded` / `failed` / `cancelled`) are
        /// eligible; active runs are protected.
        #[arg(long = "status", value_parser = parse_gc_status)]
        statuses: Vec<taquba_research::store::RunIndexStatus>,
        /// List candidates without deleting anything.
        #[arg(long)]
        dry_run: bool,
    },
}

fn parse_gc_status(s: &str) -> std::result::Result<taquba_research::store::RunIndexStatus, String> {
    use taquba_research::store::RunIndexStatus;
    match s.to_ascii_lowercase().as_str() {
        "running" => Ok(RunIndexStatus::Running),
        "paused" => Ok(RunIndexStatus::Paused),
        "succeeded" => Ok(RunIndexStatus::Succeeded),
        "failed" => Ok(RunIndexStatus::Failed),
        "cancellation_requested" | "cancellation-requested" => {
            Ok(RunIndexStatus::CancellationRequested)
        }
        "cancelled" | "canceled" => Ok(RunIndexStatus::Cancelled),
        other => Err(format!(
            "unknown status `{other}`; expected one of: \
             running, paused, succeeded, failed, cancellation_requested, cancelled"
        )),
    }
}

/// Resolved store handle: the shared `ObjectStore` and the key prefix
/// within it under which queue / runs / reports all live.
#[derive(Clone)]
struct StoreCtx {
    object_store: Arc<dyn ObjectStore>,
    prefix: ObjectPath,
    /// User-visible source string for the store (the raw path or URL),
    /// used in CLI messages so the user knows which bucket they're
    /// looking at without us having to reconstruct a URL from
    /// `Arc<dyn ObjectStore>`.
    source: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.quiet);

    let store_ctx = resolve_store(cli.store.as_deref())
        .await
        .context("opening store")?;
    let run_store = RunStore::new(store_ctx.object_store.clone(), &store_ctx.prefix);

    match &cli.command {
        Some(Command::Resume { run_id }) => {
            cmd_resume(&cli, &store_ctx, &run_store, run_id.clone()).await
        }
        Some(Command::List) => cmd_list(&store_ctx, &run_store).await,
        Some(Command::Status { run_id }) => cmd_status(&run_store, run_id.clone()).await,
        Some(Command::Show { run_id, output }) => {
            cmd_show(&store_ctx, &run_store, run_id.clone(), output.as_deref()).await
        }
        Some(Command::Cancel { run_id }) => cmd_cancel(&run_store, run_id.clone()).await,
        Some(Command::Init) => cmd_init(&store_ctx).await,
        Some(Command::Gc {
            older_than_days,
            statuses,
            dry_run,
        }) => {
            cmd_gc(
                &store_ctx,
                &run_store,
                *older_than_days,
                statuses.clone(),
                *dry_run,
            )
            .await
        }
        None => {
            let query = cli
                .query
                .clone()
                .ok_or_else(|| anyhow!("missing QUERY; pass a query string or use a subcommand"))?;
            cmd_run(&cli, &store_ctx, &run_store, query).await
        }
    }
}

fn init_tracing(quiet: bool) {
    let default = if quiet { "warn" } else { "info" };
    let filter = EnvFilter::try_from_env("RUST_LOG").unwrap_or_else(|_| EnvFilter::new(default));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .without_time()
        .with_writer(std::io::stderr)
        .try_init();
}

/// Resolve `--store` (or `TAQUBA_RESEARCH_STORE`, or the default) into a
/// concrete [`StoreCtx`]. Accepts:
///
/// - object-storage URLs: `s3://bucket/prefix`, `gs://bucket/prefix`,
///   `az://container/prefix`, `file:///abs/path` (cloud schemes require
///   the matching cargo feature);
/// - bare paths: treated as a local directory, created if missing.
async fn resolve_store(flag: Option<&str>) -> Result<StoreCtx> {
    let raw = match flag {
        Some(s) => s.to_string(),
        None => std::env::var("TAQUBA_RESEARCH_STORE").unwrap_or_else(|_| {
            let home = dirs::home_dir()
                .map(|h| h.join(".taquba-research"))
                .unwrap_or_else(|| PathBuf::from(".taquba-research"));
            home.to_string_lossy().into_owned()
        }),
    };

    // URL form? `scheme://...` (any of the cloud schemes), `file://`, or
    // an opaque scheme parse_url knows about.
    if looks_like_url(&raw) {
        let url = Url::parse(&raw).with_context(|| format!("parsing --store as URL: {raw}"))?;
        let (store, path) =
            parse_url(&url).with_context(|| format!("opening object store from URL `{raw}`"))?;
        return Ok(StoreCtx {
            object_store: Arc::from(store),
            prefix: path,
            source: raw,
        });
    }

    // Bare path: ensure the directory exists, then wrap in LocalFileSystem
    // with that directory as the LocalFileSystem prefix. The key prefix
    // inside the store is the empty path; SlateDB / RunStore / reports
    // all live directly under it.
    let dir = PathBuf::from(&raw);
    tokio::fs::create_dir_all(&dir)
        .await
        .with_context(|| format!("creating store dir {}", dir.display()))?;
    let local = LocalFileSystem::new_with_prefix(&dir)
        .with_context(|| format!("opening LocalFileSystem at {}", dir.display()))?;
    Ok(StoreCtx {
        object_store: Arc::new(local),
        prefix: ObjectPath::default(),
        source: raw,
    })
}

fn looks_like_url(s: &str) -> bool {
    // A scheme has to start with a letter and contain `://`.
    s.contains("://") && s.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
}

/// Clap value parser for `--store` / `--output`. Bare paths pass through
/// untouched; URL forms are checked for parseability and a known scheme
/// so typos like `S3://...` or `aws://...` are caught at arg-parse time
/// rather than mid-flow. The real connectivity check (creds, bucket
/// exists) is what `taquba-research init` is for.
fn validate_store_arg(s: &str) -> std::result::Result<String, String> {
    if !looks_like_url(s) {
        return Ok(s.to_string());
    }
    let url = Url::parse(s).map_err(|e| format!("invalid URL `{s}`: {e}"))?;
    match url.scheme() {
        "s3" | "gs" | "az" | "abfs" | "abfss" | "file" | "memory" => Ok(s.to_string()),
        other => Err(format!(
            "unsupported scheme `{other}` in `{s}`; \
             expected one of: s3, gs, az, abfs, abfss, file"
        )),
    }
}

async fn open_queue(ctx: &StoreCtx) -> Result<Arc<Queue>> {
    let queue_path = if ctx.prefix.as_ref().is_empty() {
        QUEUE_DB_NAME.to_string()
    } else {
        format!("{}/{}", ctx.prefix.as_ref(), QUEUE_DB_NAME)
    };
    let queue = Queue::open(ctx.object_store.clone(), &queue_path)
        .await
        .context("opening taquba queue")?;
    Ok(Arc::new(queue))
}

/// Resolve where the final report should land. Returns `None` for the
/// default (write to the unified store under `<store>/reports/<id>.md`).
/// Returns `Some(Local)` for a local filesystem path and
/// `Some(Remote)` for a URL pointing into an object store.
enum OutputTarget {
    /// Write into the run's unified `StoreCtx` at `reports/<run_id>.md`.
    DefaultInStore,
    /// Write to a local filesystem path with `tokio::fs::write`.
    Local(PathBuf),
    /// Write to an object-store URL.
    Remote {
        object_store: Arc<dyn ObjectStore>,
        path: ObjectPath,
    },
}

fn resolve_output(raw: Option<&str>) -> Result<OutputTarget> {
    let Some(raw) = raw else {
        return Ok(OutputTarget::DefaultInStore);
    };
    if !looks_like_url(raw) {
        return Ok(OutputTarget::Local(PathBuf::from(raw)));
    }
    let url = Url::parse(raw).with_context(|| format!("parsing --output as URL: {raw}"))?;
    let (store, path) =
        parse_url(&url).with_context(|| format!("opening output store from URL `{raw}`"))?;
    Ok(OutputTarget::Remote {
        object_store: Arc::from(store),
        path,
    })
}

async fn write_report(
    target: &OutputTarget,
    fallback: &StoreCtx,
    run_id: &str,
    markdown: &str,
) -> Result<String> {
    match target {
        OutputTarget::DefaultInStore => {
            let key = build_default_report_path(&fallback.prefix, run_id);
            fallback
                .object_store
                .put(&key, PutPayload::from(markdown.as_bytes().to_vec()))
                .await
                .with_context(|| format!("writing report to store at {key}"))?;
            Ok(format!("{key} (in configured store)"))
        }
        OutputTarget::Local(path) => {
            if let Some(parent) = path.parent()
                && !parent.as_os_str().is_empty()
            {
                tokio::fs::create_dir_all(parent).await.ok();
            }
            tokio::fs::write(path, markdown)
                .await
                .with_context(|| format!("writing report to {}", path.display()))?;
            Ok(path.display().to_string())
        }
        OutputTarget::Remote { object_store, path } => {
            object_store
                .put(path, PutPayload::from(markdown.as_bytes().to_vec()))
                .await
                .with_context(|| format!("writing report to {path}"))?;
            Ok(path.to_string())
        }
    }
}

fn build_default_report_path(prefix: &ObjectPath, run_id: &str) -> ObjectPath {
    prefix.child(REPORTS_PREFIX).child(format!("{run_id}.md"))
}

fn build_runner(cli: &Cli, run_store: &RunStore) -> Result<ResearchStepRunner> {
    if cli.search != "tavily" {
        bail!(
            "search backend `{}` is not wired in v0.1 (only `tavily` is); see --help",
            cli.search
        );
    }
    let tavily = Tavily::from_env().context("TAVILY_API_KEY not set or empty")?;
    let search: Arc<dyn SearchBackend> = Arc::new(tavily);
    let runner = match CliProvider::resolve(cli.provider) {
        CliProvider::OpenAi => {
            let client = openai::Client::from_env().context("OPENAI_API_KEY missing or invalid")?;
            ResearchStepRunner::new_openai(client, search)
        }
        CliProvider::Anthropic => {
            let client =
                anthropic::Client::from_env().context("ANTHROPIC_API_KEY missing or invalid")?;
            ResearchStepRunner::new_anthropic(client, search)
        }
        CliProvider::Ollama => {
            // No API key; defaults to localhost:11434 (OLLAMA_API_BASE_URL).
            let client = ollama::Client::from_env().context("failed to build Ollama client")?;
            ResearchStepRunner::new_ollama(client, search)
        }
    };
    Ok(runner.with_run_store(run_store.clone()))
}

fn build_config(cli: &Cli) -> ResearchConfig {
    let provider = CliProvider::resolve(cli.provider);
    let model = cli
        .model
        .clone()
        .unwrap_or_else(|| provider.default_model().to_string());
    ResearchConfig {
        depth: cli.depth,
        max_sources: cli.max_sources,
        ..ResearchConfig::new(model)
    }
}

/// Build a [`WorkflowRuntime`] with our standard config (one claimer,
/// `CaptureHook` for the terminal channel) and spawn its worker task.
/// The worker exits on either Ctrl+C or a terminal-hook signal sent via
/// [`WorkerHandles::shutdown_tx`]; the Ctrl+C branch prints an
/// immediate "Interrupting…" acknowledgement so the user doesn't
/// experience a silent ~10-30s drain.
fn spawn_runtime(
    queue: Arc<Queue>,
    object_store: Arc<dyn ObjectStore>,
    runner: ResearchStepRunner,
) -> Result<(
    WorkflowRuntime<ResearchStepRunner, CaptureHook>,
    WorkerHandles,
)> {
    let (tx, rx) = oneshot::channel::<RunOutcome>();
    let hook = CaptureHook {
        tx: Mutex::new(Some(tx)),
    };
    // Stand up the JobRunner the Fetching step submits FetchPage
    // jobs to.
    let (job_runner, job_handle) = spawn_fetch_runner(&queue, &object_store)?;
    let runner = runner.with_job_runner(job_runner).with_queue(queue.clone());

    // Sequential workflow: one claimer is enough. See agent.rs for
    // context.
    let runtime = WorkflowRuntime::builder(queue, object_store, runner, hook)
        .max_concurrent_steps(1)
        .memo_retention(MEMO_RETENTION)
        .build();

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let worker_runtime = runtime.clone();
    let worker = tokio::spawn(async move {
        worker_runtime
            .run(async move {
                // Exit on EITHER Ctrl+C OR a terminal-hook signal from
                // `finalize`. Without the second branch the worker
                // would never return after a successful run, leaving
                // the CLI hung until the user hits Ctrl+C.
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {
                        eprintln!();
                        eprintln!("Interrupting; waiting for current step to finish…");
                    }
                    _ = shutdown_rx => {}
                }
            })
            .await
    });

    Ok((
        runtime,
        WorkerHandles {
            rx,
            worker,
            shutdown_tx,
            job_handle,
        },
    ))
}

async fn cmd_run(
    cli: &Cli,
    store_ctx: &StoreCtx,
    run_store: &RunStore,
    query: String,
) -> Result<()> {
    let runner = build_runner(cli, run_store)?;
    let config = build_config(cli);
    let queue = open_queue(store_ctx).await?;

    let (runtime, handles) = spawn_runtime(queue, store_ctx.object_store.clone(), runner)?;

    let input = ResearchStepRunner::initial_state(query.clone(), config);
    let submit_outcome = runtime
        .submit(RunSpec {
            input,
            ..Default::default()
        })
        .await
        .context("submitting run")?;

    // Persist the "running" index entry up front so `status` works
    // immediately and a process crash leaves a visible breadcrumb.
    let now = Utc::now();
    let entry = taquba_research::store::RunIndexEntry {
        run_id: submit_outcome.run_id.clone(),
        query: query.clone(),
        submitted_at: now,
        status: taquba_research::store::RunIndexStatus::Running,
        report: None,
        error: None,
        updated_at: now,
    };
    run_store
        .put(&entry)
        .await
        .context("writing initial run index entry")?;

    println!(
        "Run {} started. (Ctrl+C to interrupt; resume with `taquba-research resume {}`)",
        submit_outcome.run_id, submit_outcome.run_id
    );

    finalize(
        cli,
        store_ctx,
        run_store,
        handles,
        &submit_outcome.run_id,
        &query,
    )
    .await
}

async fn cmd_resume(
    cli: &Cli,
    store_ctx: &StoreCtx,
    run_store: &RunStore,
    run_id: String,
) -> Result<()> {
    use taquba_research::store::RunIndexStatus;
    let mut existing = run_store
        .get(&run_id)
        .await
        .context("reading run index")?
        .ok_or_else(|| anyhow!("no run index entry for {run_id}"))?;
    match existing.status {
        RunIndexStatus::Running | RunIndexStatus::Paused => {}
        RunIndexStatus::CancellationRequested => {
            // The cancel sentinel is present and will fail the run on
            // its next step. Resuming actuates that: the worker pops
            // a step, sees the sentinel, and marks the run Failed.
            // Surface this so the user isn't surprised.
            eprintln!(
                "Note: run {run_id} has a cancellation requested. Resuming will actuate \
                 the cancellation (the next step will mark the run as cancelled)."
            );
        }
        RunIndexStatus::Succeeded | RunIndexStatus::Failed | RunIndexStatus::Cancelled => {
            bail!(
                "run {run_id} is already terminal ({})",
                existing.status.as_str()
            );
        }
    }
    if matches!(existing.status, RunIndexStatus::Paused) {
        existing.status = RunIndexStatus::Running;
        existing.updated_at = Utc::now();
        run_store
            .put(&existing)
            .await
            .context("updating run index entry to running")?;
    }

    let runner = build_runner(cli, run_store)?;
    let queue = open_queue(store_ctx).await?;

    // We discard the runtime here: cmd_resume doesn't submit new work,
    // it just starts a worker to process the existing pending step.
    let (_runtime, handles) = spawn_runtime(queue, store_ctx.object_store.clone(), runner)?;

    println!("Resuming {run_id}…");

    // The run's step jobs already exist in the queue: we just need to
    // start a worker. We do NOT re-submit. The terminal hook will fire
    // when the existing run reaches a terminal step.
    finalize(cli, store_ctx, run_store, handles, &run_id, &existing.query).await
}

/// Handles tied to a running `WorkflowRuntime` worker task and its
/// paired fetch-job `JobRunner`.
struct WorkerHandles {
    /// Terminal-hook signal.
    rx: oneshot::Receiver<RunOutcome>,
    /// Spawned workflow worker task.
    worker: tokio::task::JoinHandle<taquba_workflow::Result<()>>,
    /// Sends a clean-shutdown signal so the workflow worker stops
    /// polling once the run is done (without waiting for Ctrl+C).
    shutdown_tx: oneshot::Sender<()>,
    /// Handle to the fetch JobRunner's worker task. Shutdown only
    /// after the workflow worker has drained (any in-flight
    /// FetchPage was awaited by the workflow step that submitted it),
    /// so there's no work left at that point.
    job_handle: RunnerHandle,
}

async fn finalize(
    cli: &Cli,
    store_ctx: &StoreCtx,
    run_store: &RunStore,
    handles: WorkerHandles,
    run_id: &str,
    query: &str,
) -> Result<()> {
    // Either the terminal hook fires (run reached a terminal step) or
    // the worker exited (Ctrl+C). Whichever happens first determines the
    // exit message.
    let WorkerHandles {
        mut rx,
        mut worker,
        shutdown_tx,
        job_handle,
    } = handles;
    let result = tokio::select! {
        out = &mut rx => {
            // Run reached a terminal step. Tell the worker to stop so it
            // doesn't keep polling the queue, then await its clean exit
            // before writing the report.
            let _ = shutdown_tx.send(());
            let _ = (&mut worker).await;
            handle_terminal(cli, store_ctx, run_store, run_id, query, out.ok()).await
        }
        joined = &mut worker => {
            // Worker exited without a terminal signal -> interrupted.
            // Drop shutdown_tx; the receiver is already gone.
            drop(shutdown_tx);
            let _ = joined; // ignore join error; we're already exiting
            mark_paused(run_store, run_id).await;
            print_interrupted(run_id);
            Ok(())
        }
    };
    // Workflow worker has stopped (terminal or Ctrl+C). Drain the
    // fetch JobRunner so its task exits before this fn returns.
    let _ = job_handle.shutdown().await;
    result
}

async fn handle_terminal(
    cli: &Cli,
    store_ctx: &StoreCtx,
    run_store: &RunStore,
    run_id: &str,
    query: &str,
    outcome: Option<RunOutcome>,
) -> Result<()> {
    let outcome = outcome.ok_or_else(|| anyhow!("terminal hook produced no outcome"))?;

    let entry = build_terminal_entry(&outcome, run_id, query, run_store).await;
    run_store
        .put(&entry)
        .await
        .context("persisting terminal index entry")?;

    match outcome.status {
        TerminalStatus::Succeeded => {
            let result_bytes = outcome.result.unwrap_or_default();
            let record: RunRecord = serde_json::from_slice(&result_bytes)
                .context("decoding RunRecord from terminal hook")?;
            let report = record
                .report
                .ok_or_else(|| anyhow!("succeeded run has no report"))?;

            let target = resolve_output(cli.output.as_deref())?;
            let where_ = write_report(&target, store_ctx, run_id, &report.markdown).await?;

            println!("✓ Report saved to {where_}");
            println!(
                "  {} steps · {}s",
                report.stats.steps_completed,
                report.stats.wall_time.as_secs(),
            );
        }
        TerminalStatus::Failed => {
            eprintln!(
                "✗ Run {run_id} failed: {}",
                outcome.error.unwrap_or_else(|| "(no reason)".to_string())
            );
            std::process::exit(1);
        }
        TerminalStatus::Cancelled => {
            // User-initiated stop. Not an error; exit 0.
            let reason = outcome.error.as_deref().unwrap_or("(no reason supplied)");
            eprintln!("⊘ Run {run_id} cancelled: {reason}");
        }
        other => {
            eprintln!("✗ Run {run_id} reached unknown terminal status: {other}");
            std::process::exit(1);
        }
    }
    Ok(())
}

async fn build_terminal_entry(
    outcome: &RunOutcome,
    run_id: &str,
    query: &str,
    run_store: &RunStore,
) -> taquba_research::store::RunIndexEntry {
    use taquba_research::store::{RunIndexEntry, RunIndexStatus};
    let now = Utc::now();
    let submitted_at = run_store
        .get(run_id)
        .await
        .ok()
        .flatten()
        .map(|e| e.submitted_at)
        .unwrap_or(now);

    let (status, report, error) = match outcome.status {
        TerminalStatus::Succeeded => {
            let bytes = outcome.result.clone().unwrap_or_default();
            match serde_json::from_slice::<RunRecord>(&bytes) {
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
        other => (
            RunIndexStatus::Failed,
            None,
            Some(format!("unknown terminal status: {other}")),
        ),
    };

    RunIndexEntry {
        run_id: run_id.to_string(),
        query: query.to_string(),
        submitted_at,
        status,
        report,
        error,
        updated_at: now,
    }
}

/// Best-effort: flip the run's index entry from Running to Paused so
/// `list`/`status` reflect the actual state. Errors are swallowed because
/// we're already on the way out; printing a usable resume command is
/// more important than reporting a stale-index write failure.
async fn mark_paused(run_store: &RunStore, run_id: &str) {
    use taquba_research::store::RunIndexStatus;
    if let Ok(Some(mut entry)) = run_store.get(run_id).await
        && matches!(entry.status, RunIndexStatus::Running)
    {
        entry.status = RunIndexStatus::Paused;
        entry.updated_at = Utc::now();
        let _ = run_store.put(&entry).await;
    }
}

fn print_interrupted(run_id: &str) {
    eprintln!();
    eprintln!("Interrupted. Run {run_id} paused. Resume with:\n  taquba-research resume {run_id}",);
}

async fn cmd_list(store_ctx: &StoreCtx, run_store: &RunStore) -> Result<()> {
    let runs = run_store.list().await.context("listing runs")?;
    println!("Store: {}", store_ctx.source);
    if runs.is_empty() {
        println!("No runs yet.");
        return Ok(());
    }
    println!(
        "{:<28} {:<22} {:<25} QUERY",
        "RUN_ID", "STATUS", "SUBMITTED"
    );
    for r in runs {
        let q = if r.query.len() > 50 {
            format!("{}…", &r.query[..49])
        } else {
            r.query
        };
        println!(
            "{:<28} {:<22} {:<25} {}",
            r.run_id,
            r.status.as_str(),
            r.submitted_at.to_rfc3339(),
            q
        );
    }
    Ok(())
}

async fn cmd_status(run_store: &RunStore, run_id: String) -> Result<()> {
    let entry = run_store
        .get(&run_id)
        .await?
        .ok_or_else(|| anyhow!("no run index entry for {run_id}"))?;
    println!("run_id:       {}", entry.run_id);
    println!("query:        {}", entry.query);
    println!("status:       {}", entry.status.as_str());
    println!("submitted_at: {}", entry.submitted_at.to_rfc3339());
    println!("updated_at:   {}", entry.updated_at.to_rfc3339());
    if let Some(err) = entry.error {
        println!("error:        {err}");
    }
    if let Some(report) = entry.report {
        println!(
            "stats:        {} steps · {}s",
            report.stats.steps_completed,
            report.stats.wall_time.as_secs()
        );
    }
    Ok(())
}

async fn cmd_show(
    store_ctx: &StoreCtx,
    run_store: &RunStore,
    run_id: String,
    output: Option<&str>,
) -> Result<()> {
    let entry = run_store
        .get(&run_id)
        .await?
        .ok_or_else(|| anyhow!("no run index entry for {run_id}"))?;
    let report = entry
        .report
        .ok_or_else(|| anyhow!("run {run_id} has no rendered report"))?;

    match output {
        None => {
            print!("{}", report.markdown);
        }
        Some(raw) => {
            // `show --output` always writes to a concrete destination —
            // no implicit `<store>/reports/...` fallback, since the user
            // is asking for a copy of a known-finished report.
            let target = resolve_output(Some(raw))?;
            let where_ = write_report(&target, store_ctx, &run_id, &report.markdown).await?;
            println!("✓ Report written to {where_}");
        }
    }
    Ok(())
}

async fn cmd_cancel(run_store: &RunStore, run_id: String) -> Result<()> {
    use taquba_research::store::RunIndexStatus;
    let mut entry = run_store
        .get(&run_id)
        .await?
        .ok_or_else(|| anyhow!("no run index entry for {run_id}"))?;
    if !matches!(
        entry.status,
        RunIndexStatus::Running | RunIndexStatus::Paused
    ) {
        bail!(
            "run {run_id} is not cancellable (status: {})",
            entry.status.as_str()
        );
    }
    run_store
        .mark_cancelled(&run_id)
        .await
        .context("writing cancel sentinel")?;
    entry.status = RunIndexStatus::CancellationRequested;
    entry.updated_at = Utc::now();
    run_store
        .put(&entry)
        .await
        .context("updating run index entry to cancellation_requested")?;
    println!(
        "Cancellation requested for {run_id}. The next step will mark the run as `cancelled`. \
         If no worker is currently running, the cancellation takes effect on the next `resume`."
    );
    Ok(())
}

async fn cmd_init(store_ctx: &StoreCtx) -> Result<()> {
    use futures_util::TryStreamExt;
    println!("Probing store: {}", store_ctx.source);

    // Pulling one entry off a `list` is the cheapest cross-backend
    // sanity check: it round-trips creds + bucket existence without
    // mutating anything, and an empty index returns `None` cleanly.
    let mut stream = store_ctx
        .object_store
        .list(Some(&store_ctx.prefix.child("runs")));
    let has_any = stream
        .try_next()
        .await
        .context("listing store to verify connectivity")?
        .is_some();
    if has_any {
        println!("✓ Store reachable; existing runs found in index.");
    } else {
        println!("✓ Store reachable; no runs in index yet.");
    }
    Ok(())
}

async fn cmd_gc(
    store_ctx: &StoreCtx,
    run_store: &RunStore,
    older_than_days: Option<i64>,
    statuses: Vec<taquba_research::store::RunIndexStatus>,
    dry_run: bool,
) -> Result<()> {
    use taquba_research::store::RunIndexStatus;

    let cutoff = older_than_days.map(|d| Utc::now() - chrono::Duration::days(d));
    let runs = run_store.list().await.context("listing runs for gc")?;

    let mut candidates: Vec<_> = runs
        .into_iter()
        .filter(|r| {
            // Always protect non-terminal runs unless the user explicitly
            // opted in by passing `--status <that_state>`. Deleting an
            // active run's index entry mid-flight would orphan its queue
            // state without any way to discover it later.
            let is_active = matches!(
                r.status,
                RunIndexStatus::Running
                    | RunIndexStatus::Paused
                    | RunIndexStatus::CancellationRequested
            );
            if is_active && !statuses.contains(&r.status) {
                return false;
            }
            cutoff.is_none_or(|c| r.submitted_at < c)
                && (statuses.is_empty() || statuses.contains(&r.status))
        })
        .collect();
    candidates.sort_by(|a, b| a.submitted_at.cmp(&b.submitted_at));

    if candidates.is_empty() {
        println!("No runs match the gc filter.");
        return Ok(());
    }

    println!("Store: {}", store_ctx.source);
    println!(
        "{} candidate{}:",
        candidates.len(),
        if candidates.len() == 1 { "" } else { "s" }
    );
    for r in &candidates {
        println!(
            "  {}  {:<22}  {}  {}",
            r.run_id,
            r.status.as_str(),
            r.submitted_at.to_rfc3339(),
            if r.query.len() > 60 {
                format!("{}…", &r.query[..59])
            } else {
                r.query.clone()
            }
        );
    }

    if dry_run {
        println!("(dry-run: no objects deleted)");
        return Ok(());
    }

    let mut deleted = 0usize;
    let mut errors = 0usize;
    for r in &candidates {
        let mut paths = vec![
            run_store.entry_path(&r.run_id),
            run_store.cancel_path(&r.run_id),
        ];
        // Default-location report. Custom `--output` destinations are
        // not tracked, so they're left alone.
        paths.push(build_default_report_path(&store_ctx.prefix, &r.run_id));

        let mut row_failed = false;
        for p in paths {
            match store_ctx.object_store.delete(&p).await {
                Ok(_) => {}
                Err(taquba::object_store::Error::NotFound { .. }) => {
                    // Either the report was redirected via --output or
                    // the cancel sentinel never existed. Not an error.
                }
                Err(e) => {
                    tracing::warn!(path = %p, error = %e, "gc delete failed");
                    row_failed = true;
                }
            }
        }
        if row_failed {
            errors += 1;
        } else {
            deleted += 1;
        }
    }
    println!("Deleted {deleted} run(s); {errors} error(s).");
    Ok(())
}

struct CaptureHook {
    tx: Mutex<Option<oneshot::Sender<RunOutcome>>>,
}

impl TerminalHook for CaptureHook {
    async fn on_termination(&self, outcome: &RunOutcome) {
        // Take the sender out of the mutex before sending so the lock
        // guard isn't held across `tx.send`. See agent.rs for context.
        let tx = self.tx.lock().await.take();
        if let Some(tx) = tx {
            let _ = tx.send(outcome.clone());
        }
    }
}
