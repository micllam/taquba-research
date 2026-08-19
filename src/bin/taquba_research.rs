//! `taquba-research` CLI entry point.
//!
//! Subcommands:
//!
//! - default (positional `QUERY`): start a new research run.
//! - `resume <RUN_ID>`: resume an interrupted run.
//! - `list`: list runs, with statuses derived from live queue state.
//! - `status <RUN_ID>`: print a run's derived status and progress.
//! - `show <RUN_ID> [--output ...]`: print or write the stored report.
//! - `cancel <RUN_ID>`: cooperatively cancel an in-flight run.
//! - `init`: verify the configured store is reachable (fail-fast cred /
//!   bucket check before submitting an expensive run).
//! - `gc [--older-than-days N] [--status S]... [--force]`: delete
//!   terminal runs' index entries, sentinels and default-location
//!   reports.
//!
//! `list`, `status`, `show` and `cancel` read through a
//! [`QueueReader`] and work from a second process against a live
//! store. `gc` needs the exclusive writer; it refuses while claimed
//! jobs are visible unless `--force` is passed.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use chrono::Utc;
use clap::{Parser, Subcommand, ValueEnum};
use rig_core::client::ProviderClient;
use rig_core::providers::{anthropic, ollama, openai};
use taquba::object_store::local::LocalFileSystem;
use taquba::object_store::path::Path as ObjectPath;
use taquba::object_store::{ObjectStore, ObjectStoreExt, PutPayload, parse_url};
use taquba::{JobStatus, OpenOptions, Queue, QueueConfig, QueueReader, ReaderMode, ReaderOptions};
use taquba_research::jobs::RunnerHandle;
use taquba_research::store::{
    self, RunDisplayStatus, RunIndexEntry, StepJobState, StoredStatus, WORKFLOW_QUEUE_NAME,
};
use taquba_research::workflow::{
    RunOutcome, RunSpec, StepError, TerminalEffects, TerminalHook, TerminalStatus, WorkflowRuntime,
};
use taquba_research::{
    CancelSentinel, FETCH_QUEUE_NAME, ResearchConfig, ResearchStepRunner, RunRecord,
    search::{SearchBackend, Tavily},
    spawn_fetch_runner, summarize_state,
};
use tokio::sync::{Mutex, oneshot};
use tracing_subscriber::EnvFilter;
use url::Url;

const QUEUE_DB_NAME: &str = "queue";
/// How long workflow memo blobs are retained after the run reaches a
/// terminal state. Matches the value used by the library's
/// `ResearchAgent::run`; keep in sync.
const MEMO_RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);
/// Lease duration for every queue in the store (workflow steps and
/// fetch jobs). Set explicitly because it is the bound on detecting a
/// dead or hung delivery; slow calls extend the lease through
/// `LeaseHandle` to cover their own timeout, so it stays short.
const LEASE_DURATION: Duration = Duration::from_secs(30);

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
            CliProvider::OpenAi => openai::completion::GPT_5_NANO,
            CliProvider::Anthropic => anthropic::completion::CLAUDE_HAIKU_4_5,
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

    /// Write an additional copy of the final report on completion.
    /// Accepts a local path or an object-storage URL
    /// (`s3://bucket/key.md`, `gs://...`, etc.). The report is always
    /// persisted as `<store>/reports/<run_id>.md` inside the
    /// configured store, so an S3-backed deployment keeps the markdown
    /// next to the queue and index.
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
    /// (`gpt-5-nano` for OpenAI, `claude-haiku-4-5` for Anthropic,
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
    /// Delete terminal runs' index entries, cancellation sentinels
    /// and default-location reports. Use `--dry-run` to preview.
    Gc {
        /// Delete only runs whose `submitted_at` is at least this many
        /// days in the past.
        #[arg(long)]
        older_than_days: Option<i64>,
        /// Restrict deletion to specific terminal statuses
        /// (repeatable). Allowed: `succeeded`, `failed`, `cancelled`.
        /// Runs without a terminal record are never deleted: their
        /// state is held in the queue.
        #[arg(long = "status", value_parser = parse_gc_status)]
        statuses: Vec<StoredStatus>,
        /// Proceed even when claimed jobs are visible in the store.
        /// Opening the writer fences a live worker and requeues its
        /// claimed jobs; pass this only when no worker is running.
        #[arg(long)]
        force: bool,
        /// List candidates without deleting anything.
        #[arg(long)]
        dry_run: bool,
    },
}

fn parse_gc_status(s: &str) -> std::result::Result<StoredStatus, String> {
    match s.to_ascii_lowercase().as_str() {
        "succeeded" => Ok(StoredStatus::Succeeded),
        "failed" => Ok(StoredStatus::Failed),
        "cancelled" | "canceled" => Ok(StoredStatus::Cancelled),
        other => Err(format!(
            "unknown status `{other}`; expected one of: succeeded, failed, cancelled"
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
    let sentinel = CancelSentinel::new(store_ctx.object_store.clone(), &store_ctx.prefix);

    match &cli.command {
        Some(Command::Resume { run_id }) => {
            cmd_resume(&cli, &store_ctx, &sentinel, run_id.clone()).await
        }
        Some(Command::List) => cmd_list(&store_ctx, &sentinel).await,
        Some(Command::Status { run_id }) => cmd_status(&store_ctx, &sentinel, run_id.clone()).await,
        Some(Command::Show { run_id, output }) => {
            cmd_show(&store_ctx, run_id.clone(), output.as_deref()).await
        }
        Some(Command::Cancel { run_id }) => cmd_cancel(&store_ctx, &sentinel, run_id.clone()).await,
        Some(Command::Init) => cmd_init(&store_ctx).await,
        Some(Command::Gc {
            older_than_days,
            statuses,
            force,
            dry_run,
        }) => {
            cmd_gc(
                &store_ctx,
                &sentinel,
                *older_than_days,
                statuses.clone(),
                *force,
                *dry_run,
            )
            .await
        }
        None => {
            let query = cli
                .query
                .clone()
                .ok_or_else(|| anyhow!("missing QUERY; pass a query string or use a subcommand"))?;
            cmd_run(&cli, &store_ctx, &sentinel, query).await
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

fn queue_path(ctx: &StoreCtx) -> String {
    if ctx.prefix.as_ref().is_empty() {
        QUEUE_DB_NAME.to_string()
    } else {
        format!("{}/{}", ctx.prefix.as_ref(), QUEUE_DB_NAME)
    }
}

async fn open_queue(ctx: &StoreCtx) -> Result<Arc<Queue>> {
    let opts = OpenOptions {
        default_queue_config: QueueConfig {
            lease_duration: LEASE_DURATION,
            ..QueueConfig::default()
        },
        ..OpenOptions::default()
    };
    let queue = Queue::open_with_options(ctx.object_store.clone(), &queue_path(ctx), opts)
        .await
        .context("opening taquba queue")?;
    Ok(Arc::new(queue))
}

/// Whether a queue has ever been created in this store. A
/// [`QueueReader`] cannot open a store without a manifest, so the
/// inspection commands map that case to "no runs".
async fn queue_exists(ctx: &StoreCtx) -> Result<bool> {
    use futures_util::TryStreamExt;
    let prefix = ctx.prefix.clone().join(QUEUE_DB_NAME);
    let mut stream = ctx.object_store.list(Some(&prefix));
    Ok(stream
        .try_next()
        .await
        .context("probing queue store")?
        .is_some())
}

/// Open a read-only view of the queue store. `FollowLatest` performs
/// no object-store writes, so read-only credentials suffice.
async fn open_reader(ctx: &StoreCtx) -> Result<QueueReader> {
    QueueReader::open_with_options(
        ctx.object_store.clone(),
        &queue_path(ctx),
        ReaderOptions {
            mode: ReaderMode::FollowLatest,
            ..ReaderOptions::default()
        },
    )
    .await
    .context("opening queue reader")
}

/// Run `op` against a read-only view of the queue store, retrying
/// once on failure. A `FollowLatest` read can fail on an object
/// collected under its view; the retry opens a fresh reader whose
/// view postdates the collection.
async fn with_reader<T>(ctx: &StoreCtx, op: impl AsyncFn(&QueueReader) -> Result<T>) -> Result<T> {
    let reader = open_reader(ctx).await?;
    let first = op(&reader).await;
    let _ = reader.close().await;
    match first {
        Ok(v) => Ok(v),
        Err(first) => {
            tracing::debug!(error = %first, "read failed; retrying with a fresh reader");
            let reader = open_reader(ctx).await?;
            let second = op(&reader).await;
            let _ = reader.close().await;
            second
        }
    }
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
            let key = store::report_path(&fallback.prefix, run_id);
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

fn build_runner(cli: &Cli, sentinel: &CancelSentinel) -> Result<ResearchStepRunner> {
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
    Ok(runner.with_cancellation(sentinel.clone()))
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
/// immediate "Interrupting..." acknowledgement so the user doesn't
/// experience a silent ~10-30s drain.
fn spawn_runtime(
    store_ctx: &StoreCtx,
    queue: Arc<Queue>,
    runner: ResearchStepRunner,
    run_id: &str,
) -> Result<(
    WorkflowRuntime<ResearchStepRunner, CaptureHook>,
    WorkerHandles,
)> {
    let (tx, rx) = oneshot::channel::<RunOutcome>();
    let hook = CaptureHook {
        run_id: run_id.to_string(),
        tx: Mutex::new(Some(tx)),
    };
    // Build the JobRunner the Fetching step submits FetchPage jobs to.
    let (job_runner, job_handle) = spawn_fetch_runner(&queue, &store_ctx.object_store);
    let runner = runner
        .with_job_runner(job_runner)
        .with_queue(queue.clone())
        .with_report_store(store_ctx.object_store.clone(), &store_ctx.prefix);

    // Sequential workflow: one claimer is enough. See agent.rs for
    // context.
    let runtime = WorkflowRuntime::builder(queue, store_ctx.object_store.clone(), runner, hook)
        .queue_name(WORKFLOW_QUEUE_NAME)
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
                        eprintln!("Interrupting; waiting for current step to finish...");
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
    sentinel: &CancelSentinel,
    query: String,
) -> Result<()> {
    let runner = build_runner(cli, sentinel)?;
    let config = build_config(cli);
    let queue = open_queue(store_ctx).await?;

    // The run id is generated before submit so the index entry's KV
    // key can join the submit transaction: the run and its entry
    // commit together.
    let run_id = ulid::Ulid::new().to_string();
    let (runtime, handles) = spawn_runtime(store_ctx, queue, runner, &run_id)?;

    let entry = RunIndexEntry {
        run_id: run_id.clone(),
        query: query.clone(),
        submitted_at: Utc::now(),
        terminal: None,
    };
    let input = ResearchStepRunner::initial_state(query.clone(), config);
    runtime
        .submit(RunSpec {
            run_id: Some(run_id.clone()),
            input,
            kv_writes: [(store::run_entry_key(&run_id), entry.to_bytes())].into(),
            ..Default::default()
        })
        .await
        .context("submitting run")?;

    println!(
        "Run {run_id} started. (Ctrl+C to interrupt; resume with `taquba-research resume {run_id}`)"
    );

    finalize(cli, store_ctx, handles, &run_id).await
}

async fn cmd_resume(
    cli: &Cli,
    store_ctx: &StoreCtx,
    sentinel: &CancelSentinel,
    run_id: String,
) -> Result<()> {
    // Guard against resuming a finished, dead-lettered or unknown run
    // before the (exclusive) writer open. A reader-side check
    // suffices: the worker acts on the same entry and step job.
    {
        if !queue_exists(store_ctx).await? {
            bail!("no run index entry for {run_id} (store contains no runs)");
        }
        let (entry, job) = with_reader(store_ctx, async |reader| {
            let entry = store::get_run(reader, &run_id).await?;
            let mut jobs = store::snapshot_step_jobs(reader, WORKFLOW_QUEUE_NAME).await?;
            Ok((entry, jobs.remove(&run_id)))
        })
        .await?;
        let entry = entry.ok_or_else(|| anyhow!("no run index entry for {run_id}"))?;
        if let Some(terminal) = &entry.terminal {
            bail!("run {run_id} is already terminal ({})", terminal.status);
        }
        match job {
            None => bail!("run {run_id} has no step job to resume"),
            // A dead-letter job is never claimed, so a worker would
            // wait on it indefinitely.
            Some(StepJobState::Dead(job)) => bail!(
                "run {run_id} is dead-lettered after {} attempts and cannot be resumed; \
                 inspect it with `status {run_id}`",
                job.attempts
            ),
            Some(_) => {}
        }
    }
    if sentinel.is_set(&run_id).await {
        eprintln!(
            "Note: run {run_id} has a cancellation requested. Resuming applies it: \
             the next step will mark the run as cancelled."
        );
    }

    let runner = build_runner(cli, sentinel)?;
    let queue = open_queue(store_ctx).await?;

    // The runtime is discarded: cmd_resume submits no new work; it
    // starts a worker to process the existing pending step.
    let (_runtime, handles) = spawn_runtime(store_ctx, queue, runner, &run_id)?;

    println!("Resuming {run_id}...");

    // The run's step jobs already exist in the queue; the terminal
    // hook fires when the run reaches a terminal step.
    finalize(cli, store_ctx, handles, &run_id).await
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
    handles: WorkerHandles,
    run_id: &str,
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
            handle_terminal(cli, store_ctx, run_id, out.ok()).await
        }
        joined = &mut worker => {
            // Worker exited without a terminal signal: interrupted.
            // The pending step job makes the run resumable; no index
            // write is needed. Drop shutdown_tx; the receiver is
            // dropped.
            drop(shutdown_tx);
            let _ = joined;
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
    run_id: &str,
    outcome: Option<RunOutcome>,
) -> Result<()> {
    let outcome = outcome.ok_or_else(|| anyhow!("terminal hook produced no outcome"))?;

    match outcome.status {
        TerminalStatus::Succeeded => {
            let result_bytes = outcome.result.unwrap_or_default();
            let record: RunRecord = serde_json::from_slice(&result_bytes)
                .context("decoding RunRecord from terminal hook")?;
            let report = record
                .report
                .ok_or_else(|| anyhow!("succeeded run has no report"))?;

            // The Writing step already wrote the canonical copy under
            // `reports/`; `--output` adds a copy elsewhere.
            let stored = format!(
                "{} (in configured store)",
                store::report_path(&store_ctx.prefix, run_id)
            );
            let where_ = match resolve_output(cli.output.as_deref())? {
                OutputTarget::DefaultInStore => stored,
                target => {
                    let copy = write_report(&target, store_ctx, run_id, &report.markdown).await?;
                    format!("{copy}; also {stored}")
                }
            };

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

fn print_interrupted(run_id: &str) {
    eprintln!();
    eprintln!(
        "Interrupted. Run {run_id} remains queued. Resume with:\n  taquba-research resume {run_id}",
    );
}

/// Truncate `s` to at most `max` characters, replacing the removed
/// tail with `...`. The cut is on a character boundary.
fn ellipsize(s: &str, max: usize) -> String {
    if s.char_indices().nth(max).is_none() {
        return s.to_string();
    }
    let cut = s
        .char_indices()
        .nth(max.saturating_sub(3))
        .map_or(0, |(i, _)| i);
    format!("{}...", &s[..cut])
}

/// One `list` row: the stored entry plus its derived status.
struct RunRow {
    entry: RunIndexEntry,
    status: RunDisplayStatus,
}

/// Read every entry, the step-job snapshot and the sentinels needed
/// to derive display statuses. Rows are returned newest first.
async fn gather_rows(store_ctx: &StoreCtx, sentinel: &CancelSentinel) -> Result<Vec<RunRow>> {
    let (entries, jobs) = with_reader(store_ctx, async |reader| {
        let entries = store::list_runs(reader).await?;
        let jobs = store::snapshot_step_jobs(reader, WORKFLOW_QUEUE_NAME).await?;
        Ok((entries, jobs))
    })
    .await?;

    let mut rows = Vec::with_capacity(entries.len());
    for entry in entries.into_iter().rev() {
        // The sentinel is checked only for runs without a terminal
        // record.
        let cancel_requested = entry.terminal.is_none() && sentinel.is_set(&entry.run_id).await;
        let status =
            store::derive_display_status(&entry, jobs.get(&entry.run_id), cancel_requested);
        rows.push(RunRow { entry, status });
    }
    Ok(rows)
}

async fn cmd_list(store_ctx: &StoreCtx, sentinel: &CancelSentinel) -> Result<()> {
    if !queue_exists(store_ctx).await? {
        println!("Store: {}", store_ctx.source);
        println!("No runs yet.");
        return Ok(());
    }
    let rows = gather_rows(store_ctx, sentinel)
        .await
        .context("listing runs")?;
    println!("Store: {}", store_ctx.source);
    if rows.is_empty() {
        println!("No runs yet.");
        return Ok(());
    }
    println!(
        "{:<28} {:<24} {:<25} QUERY",
        "RUN_ID", "STATUS", "SUBMITTED"
    );
    for row in rows {
        let q = ellipsize(&row.entry.query, 50);
        println!(
            "{:<28} {:<24} {:<25} {}",
            row.entry.run_id,
            row.status.as_str(),
            row.entry.submitted_at.to_rfc3339(),
            q
        );
    }
    Ok(())
}

async fn cmd_status(store_ctx: &StoreCtx, sentinel: &CancelSentinel, run_id: String) -> Result<()> {
    if !queue_exists(store_ctx).await? {
        bail!("no run index entry for {run_id} (store contains no runs)");
    }
    let (entry, job, history) = with_reader(store_ctx, async |reader| {
        let entry = store::get_run(reader, &run_id).await?;
        let mut jobs = store::snapshot_step_jobs(reader, WORKFLOW_QUEUE_NAME).await?;
        let job = jobs.remove(&run_id);
        // The attempt history is printed for dead-lettered runs only.
        let history = match &job {
            Some(StepJobState::Dead(dead)) => reader.attempt_history(&dead.id).await?,
            _ => Vec::new(),
        };
        Ok((entry, job, history))
    })
    .await?;
    let entry = entry.ok_or_else(|| anyhow!("no run index entry for {run_id}"))?;

    let cancel_requested_at = if entry.terminal.is_none() {
        sentinel.requested_at(&run_id).await
    } else {
        None
    };
    let status = store::derive_display_status(&entry, job.as_ref(), cancel_requested_at.is_some());

    println!("run_id:       {}", entry.run_id);
    println!("query:        {}", entry.query);
    println!("status:       {}", status.as_str());
    println!("submitted_at: {}", entry.submitted_at.to_rfc3339());
    if let Some(at) = cancel_requested_at {
        println!("cancel_requested_at: {}", at.to_rfc3339());
    }
    if let Some(t) = &entry.terminal {
        println!("finished_at:  {}", t.finished_at.to_rfc3339());
        if let Some(err) = &t.error {
            println!("error:        {err}");
        }
        println!(
            "stats:        {} steps · {}s · {} tokens",
            t.summary.steps_completed, t.summary.wall_time_secs, t.summary.token_usage.total_tokens,
        );
    }
    if let Some(state) = &job {
        let job = state.job();
        if let Some(progress) = summarize_state(&job.payload) {
            println!(
                "progress:     phase {} · {} steps completed",
                progress.phase, progress.steps_completed
            );
        }
        println!("attempts:     {}/{}", job.attempts, job.max_attempts);
        if let Some(err) = &job.last_error {
            println!("last_error:   {err}");
        }
        if matches!(state, StepJobState::Dead(_)) {
            for attempt in &history {
                if let Some(err) = &attempt.error {
                    println!("attempt {:>2}:   {err}", attempt.attempt);
                }
            }
        }
    }
    Ok(())
}

async fn cmd_show(store_ctx: &StoreCtx, run_id: String, output: Option<&str>) -> Result<()> {
    // The canonical report blob is written by the terminal path in
    // every case; its absence means the run is unknown, unfinished or
    // did not succeed. The index entry is consulted only for a more
    // specific error message.
    let key = store::report_path(&store_ctx.prefix, &run_id);
    // The body read shares the get's error handling: either call can
    // return NotFound depending on the backend.
    let read = match store_ctx.object_store.get(&key).await {
        Ok(resp) => resp.bytes().await,
        Err(e) => Err(e),
    };
    let markdown = match read {
        Ok(bytes) => {
            String::from_utf8(bytes.to_vec()).context("decoding stored report as UTF-8")?
        }
        Err(taquba::object_store::Error::NotFound { .. }) => {
            if !queue_exists(store_ctx).await? {
                bail!("run {run_id} has no stored report (store contains no runs)");
            }
            let entry = with_reader(store_ctx, async |reader| {
                store::get_run(reader, &run_id).await
            })
            .await?;
            match entry.and_then(|e| e.terminal) {
                None => bail!("run {run_id} has no stored report (not finished or unknown)"),
                Some(t) => bail!(
                    "run {run_id} has no stored report (terminal status: {})",
                    t.status
                ),
            }
        }
        Err(e) => return Err(e).context("reading stored report"),
    };

    match output {
        None => {
            print!("{markdown}");
        }
        Some(raw) => {
            // `show --output` always writes to a concrete destination;
            // the store already holds the canonical copy.
            let target = resolve_output(Some(raw))?;
            let where_ = write_report(&target, store_ctx, &run_id, &markdown).await?;
            println!("✓ Report written to {where_}");
        }
    }
    Ok(())
}

async fn cmd_cancel(store_ctx: &StoreCtx, sentinel: &CancelSentinel, run_id: String) -> Result<()> {
    if !queue_exists(store_ctx).await? {
        bail!("no run index entry for {run_id} (store contains no runs)");
    }
    let (entry, job) = with_reader(store_ctx, async |reader| {
        let entry = store::get_run(reader, &run_id).await?;
        let mut jobs = store::snapshot_step_jobs(reader, WORKFLOW_QUEUE_NAME).await?;
        Ok((entry, jobs.remove(&run_id)))
    })
    .await?;
    let entry = entry.ok_or_else(|| anyhow!("no run index entry for {run_id}"))?;
    if let Some(t) = &entry.terminal {
        bail!("run {run_id} is not cancellable (status: {})", t.status);
    }
    match &job {
        // No step will ever run for a dead-letter job, so a sentinel
        // would never take effect.
        Some(StepJobState::Dead(_)) => bail!(
            "run {run_id} is not cancellable: it is dead-lettered and no further step will run"
        ),
        None => bail!("run {run_id} is not cancellable: it has no step job (status: unknown)"),
        Some(_) => {}
    }
    sentinel
        .mark(&run_id)
        .await
        .context("writing cancel sentinel")?;
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
    // check: it round-trips credentials and bucket existence without
    // mutating anything, and an empty store returns `None` cleanly.
    let mut stream = store_ctx.object_store.list(Some(&store_ctx.prefix));
    let has_any = stream
        .try_next()
        .await
        .context("listing store to verify connectivity")?
        .is_some();
    if has_any {
        println!("✓ Store reachable.");
    } else {
        println!("✓ Store reachable; empty.");
    }
    Ok(())
}

async fn cmd_gc(
    store_ctx: &StoreCtx,
    sentinel: &CancelSentinel,
    older_than_days: Option<i64>,
    statuses: Vec<StoredStatus>,
    force: bool,
    dry_run: bool,
) -> Result<()> {
    if !queue_exists(store_ctx).await? {
        println!("No runs match the gc filter.");
        return Ok(());
    }
    // The claimed-job count is used by the guard before the writer
    // open below; a dry run never opens the writer.
    let (claimed, entries) = with_reader(store_ctx, async |reader| {
        let mut claimed = 0usize;
        for queue in [WORKFLOW_QUEUE_NAME, FETCH_QUEUE_NAME] {
            claimed += reader
                .list_jobs(queue, JobStatus::Claimed, None, 1)
                .await?
                .jobs
                .len();
        }
        let entries = store::list_runs(reader).await?;
        Ok((claimed, entries))
    })
    .await?;

    let cutoff = older_than_days.map(|d| Utc::now() - chrono::Duration::days(d));
    let mut candidates: Vec<_> = entries
        .into_iter()
        .filter(|e| match &e.terminal {
            // Runs without a terminal record are still represented in
            // the queue (in flight, interrupted or dead-lettered);
            // deleting their entries would orphan that state.
            None => false,
            Some(t) => {
                (statuses.is_empty() || statuses.contains(&t.status))
                    && cutoff.is_none_or(|c| e.submitted_at < c)
            }
        })
        .collect();
    candidates.sort_by_key(|e| e.submitted_at);

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
    for e in &candidates {
        let status = e
            .terminal
            .as_ref()
            .map(|t| t.status.as_str())
            .unwrap_or("unknown");
        println!(
            "  {}  {:<12}  {}  {}",
            e.run_id,
            status,
            e.submitted_at.to_rfc3339(),
            ellipsize(&e.query, 60)
        );
    }

    if dry_run {
        println!("(dry-run: no entries deleted)");
        return Ok(());
    }

    // Reader-side guard before taking the exclusive writer: opening
    // the writer fences a live worker and requeues its claimed jobs.
    // A claimed job visible through the reader is treated as a live
    // worker. The check is best-effort in both directions: a reader
    // cannot distinguish a live claim from an abandoned one, and
    // reader lag can hide a recent claim.
    if claimed > 0 && !force {
        bail!(
            "claimed jobs are visible in this store; a worker may be live. \
             Re-run with --force only when no `run`/`resume` process is active."
        );
    }

    // KV deletes need the writer; the guard above ran first.
    let queue = open_queue(store_ctx).await?;
    let mut deleted = 0usize;
    let mut errors = 0usize;
    for e in &candidates {
        let mut row_failed = false;
        if let Err(err) = queue.kv_delete(&store::run_entry_key(&e.run_id)).await {
            tracing::warn!(run_id = %e.run_id, error = %err, "gc entry delete failed");
            row_failed = true;
        }
        if let Err(err) = sentinel.clear(&e.run_id).await {
            tracing::warn!(run_id = %e.run_id, error = %err, "gc sentinel delete failed");
            row_failed = true;
        }
        // Default-location report. Custom `--output` destinations are
        // not tracked and are not deleted.
        let report = store::report_path(&store_ctx.prefix, &e.run_id);
        match store_ctx.object_store.delete(&report).await {
            Ok(_) | Err(taquba::object_store::Error::NotFound { .. }) => {}
            Err(err) => {
                tracing::warn!(path = %report, error = %err, "gc report delete failed");
                row_failed = true;
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
    /// Run this invocation submitted or resumed. Notifications for any
    /// other run are stale; see `on_termination`.
    run_id: String,
    tx: Mutex<Option<oneshot::Sender<RunOutcome>>>,
}

impl TerminalHook for CaptureHook {
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

#[cfg(test)]
mod tests {
    use super::ellipsize;

    #[test]
    fn ellipsize_keeps_strings_within_the_limit() {
        assert_eq!(ellipsize("short", 50), "short");
        let exact = "a".repeat(50);
        assert_eq!(ellipsize(&exact, 50), exact);
    }

    #[test]
    fn ellipsize_truncates_on_char_boundaries() {
        let ascii = "a".repeat(60);
        assert_eq!(ellipsize(&ascii, 50), format!("{}...", "a".repeat(47)));
        let multibyte = "é".repeat(30);
        assert_eq!(ellipsize(&multibyte, 25), format!("{}...", "é".repeat(22)));
    }
}
