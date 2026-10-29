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
//! - `reply <RUN_ID> <TEXT>`: reply to the question of a waiting run.
//! - `init`: verify the configured store is reachable (a check of the
//!   credentials and the bucket before a run submits model calls).
//! - `gc [--older-than-days N] [--dry-run]`: delete the reports at the default
//!   location.
//!
//! `list`, `status`, `show`, `cancel` and `reply` read through a
//! [`QueueReader`] and work from a second process against a live store.
//! `cancel` and `reply` write an entry to the inbox of the queue, which the
//! writer applies. `resume` needs the exclusive writer and refuses while
//! claimed jobs are visible unless `--force` is passed.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use chrono::Utc;
use clap::{Parser, Subcommand, ValueEnum};
use rig_core::providers::{anthropic, ollama, openai};
use taquba::object_store::local::LocalFileSystem;
use taquba::object_store::path::Path as ObjectPath;
use taquba::object_store::prefix::PrefixStore;
use taquba::object_store::{ObjectMeta, ObjectStore, ObjectStoreExt, PutPayload, parse_url_opts};
use taquba::{
    DEFAULT_INBOX_RECORD_RETENTION, DEFAULT_INBOX_TOUCH_POLL_INTERVAL, EnqueueEntry, EntryKind,
    EntryRead, InboxEntry, InboxOptions, InboxOutcome, InboxSender, JobRecord, JobStatus,
    OpenOptions, Queue, QueueConfig, QueueReader, ReaderMode, ReaderOptions, SignalEntry,
    SignalOutcome, default_inbox_path,
};
use taquba_research::jobs::RunnerHandle;
use taquba_research::store::{
    self, QUERY_HEADER, RunDisplayStatus, WORKFLOW_CONTROL_QUEUE_NAME, WORKFLOW_MEMO_PREFIX,
    WORKFLOW_QUEUE_NAME,
};
use taquba_research::workflow::{
    ListedRun, RunId, RunOptions, RunOutcome, RunSpec, RunState, RunStatus, StepError,
    StepErrorKind, TerminalEffects, TerminalHook, TerminalStatus, WorkflowRuntime, WorkflowView,
};
use taquba_research::{
    FETCH_QUEUE_NAME, Record, ResearchConfig, ResearchStepRunner, RunRecord,
    search::{SearchBackend, Tavily},
    spawn_fetch_runner, summarize_state,
};
use taquba_workflow::{CancelCommand, CommandState};
use tokio::sync::{Mutex, oneshot};
use tracing_subscriber::EnvFilter;
use url::Url;

const QUEUE_DB_NAME: &str = "queue";
/// Interval of the liveness beats of the writer, by which `reply` and `cancel`
/// judge whether a worker applies their inbox entry.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);
/// How long `reply` and `cancel` wait for a live writer to apply their entry:
/// the touch poll of the writer and two manifest polls of the reader.
const INBOX_WAIT: Duration = Duration::from_secs(DEFAULT_INBOX_TOUCH_POLL_INTERVAL.as_secs() + 4);
/// How long workflow memo blobs are retained after the run reaches a terminal
/// state. It must equal the value of the library's `ResearchAgent::run`.
const MEMO_RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);
/// Lease duration for every queue in the store (workflow steps and fetch jobs).
/// It bounds the detection of a dead or hung delivery. A slow call extends the
/// lease through `LeaseHandle` by its own timeout.
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
    /// Default model identifier when the user does not pass `--model`.
    fn default_model(self) -> &'static str {
        match self {
            CliProvider::OpenAi => openai::completion::GPT_5_NANO,
            CliProvider::Anthropic => anthropic::completion::CLAUDE_HAIKU_4_5,
            CliProvider::Ollama => ollama::LLAMA3_2,
        }
    }

    /// Resolve `--provider` against env vars. If the user passed
    /// `--provider <p>` explicitly, that wins (the only way to select `ollama`,
    /// because it does not have an API key to auto-detect). Otherwise, if
    /// `ANTHROPIC_API_KEY` is set and `OPENAI_API_KEY` is not, choose
    /// Anthropic. In every other case, choose OpenAI.
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

/// Durable research agent, as a single-binary CLI.
#[derive(Debug, Parser)]
#[command(
    name = "taquba-research",
    version,
    about = "Durable research agent for Rust",
    long_about = None,
    arg_required_else_help = false,
)]
struct Cli {
    /// Positional query for the default "start a new run" mode. If both this
    /// and a subcommand are given, the subcommand wins.
    query: Option<String>,

    /// Write an additional copy of the final report on completion. Accepts a
    /// local path or an object-storage URL (`s3://bucket/key.md`, `gs://...`
    /// and the other store schemes). The report is always persisted as
    /// `<store>/reports/<run_id>.md` inside the configured store, so an
    /// S3-backed deployment keeps the markdown next to the queue and index.
    #[arg(long, value_parser = validate_store_arg)]
    output: Option<String>,

    /// Number of sub-questions to decompose the query into.
    #[arg(long, default_value_t = 6)]
    depth: usize,

    /// Cap URLs fetched per run.
    #[arg(long, default_value_t = 30)]
    max_sources: usize,

    /// Model-call budget of the investigating step, which searches and fetches
    /// pages for the gaps of the summaries. With 0 the run skips the step.
    #[arg(long, default_value_t = 8)]
    investigation_turns: usize,

    /// Let the investigating step ask a single question per run, and wait up to
    /// this many seconds for `reply`. After the wait the step proceeds on an
    /// assumption. Without the flag, the step does not ask.
    #[arg(long, value_name = "SECONDS")]
    clarification_wait: Option<u64>,

    /// LLM provider (`openai`, `anthropic` or `ollama`). If unset, the CLI
    /// chooses one based on which `*_API_KEY` env var is set:
    /// `ANTHROPIC_API_KEY` alone selects `anthropic`, and every other case
    /// selects `openai`. `ollama` (local models) is never auto-selected: pass
    /// `--provider ollama` explicitly.
    #[arg(long, value_enum)]
    provider: Option<CliProvider>,

    /// Specific model identifier passed to the provider. If unset, the CLI
    /// chooses a provider-appropriate default (`gpt-5-nano` for OpenAI,
    /// `claude-haiku-4-5` for Anthropic, `llama3.2` for Ollama).
    #[arg(long)]
    model: Option<String>,

    /// Search backend. The CLI supports only `tavily`.
    #[arg(long, default_value = "tavily")]
    search: String,

    /// Store location. Accepts either a local path or an object-storage URL
    /// (`s3://bucket/prefix`, `gs://...`, `az://...`, `file:///...`). Cloud
    /// URLs require the cargo feature of their provider (`aws`, `gcp` or
    /// `azure`). Defaults to `~/.taquba-research/`.
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
        /// Proceed even when claimed jobs are visible in the store. Opening the
        /// writer fences a live worker and requeues its claimed jobs. Pass this
        /// only when no worker is running.
        #[arg(long)]
        force: bool,
    },
    /// List past runs.
    List,
    /// Show the recorded status of a run.
    Status {
        /// Run identifier.
        run_id: String,
    },
    /// Print the report of a finished run, or write it to a path or URL.
    Show {
        /// Run identifier.
        run_id: String,
        /// Optional destination. Same syntax as `run --output`. When omitted,
        /// the report is printed to stdout.
        #[arg(long, value_parser = validate_store_arg)]
        output: Option<String>,
    },
    /// Reply to the question of a run that waits for one.
    Reply {
        /// Run identifier.
        run_id: String,
        /// The reply.
        text: String,
    },
    /// Cooperatively cancel an in-flight run.
    Cancel {
        /// Run identifier.
        run_id: String,
    },
    /// Verify the configured store is reachable. Run it before a query to catch
    /// typos, missing credentials or unreachable buckets.
    Init,
    /// Delete the reports at the default location. Use `--dry-run` to preview.
    Gc {
        /// Delete only the reports written at least this many days ago.
        #[arg(long)]
        older_than_days: Option<i64>,
        /// List candidates without deleting anything.
        #[arg(long)]
        dry_run: bool,
    },
}

/// Resolved store handle: the shared `ObjectStore`, scoped to the path of
/// `--store`, at whose root the queue, the memos, the runs and the reports all
/// live.
#[derive(Clone)]
struct StoreCtx {
    object_store: Arc<dyn ObjectStore>,
    /// User-visible source string for the store (the raw path or URL), which
    /// CLI messages print to identify the store.
    source: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.quiet);

    let store_ctx = resolve_store(cli.store.as_deref())
        .await
        .context("opening store")?;

    match &cli.command {
        Some(Command::Resume { run_id, force }) => {
            let runner = build_runner(&cli)?;
            cmd_resume(
                &cli,
                &store_ctx,
                runner,
                run_id.clone(),
                *force,
                interrupt(),
            )
            .await
        }
        Some(Command::List) => cmd_list(&store_ctx).await,
        Some(Command::Status { run_id }) => cmd_status(&store_ctx, run_id.clone()).await,
        Some(Command::Show { run_id, output }) => {
            cmd_show(&store_ctx, run_id.clone(), output.as_deref()).await
        }
        Some(Command::Reply { run_id, text }) => {
            let outcome = cmd_reply(&store_ctx, run_id.clone(), text).await?;
            print_reply(run_id, &outcome);
            Ok(())
        }
        Some(Command::Cancel { run_id }) => cmd_cancel(&store_ctx, run_id.clone()).await,
        Some(Command::Init) => cmd_init(&store_ctx).await,
        Some(Command::Gc {
            older_than_days,
            dry_run,
        }) => cmd_gc(&store_ctx, *older_than_days, *dry_run).await,
        None => {
            let query = cli
                .query
                .clone()
                .ok_or_else(|| anyhow!("missing QUERY; pass a query string or use a subcommand"))?;
            let runner = build_runner(&cli)?;
            cmd_run(&cli, &store_ctx, runner, query, interrupt()).await
        }
    }
}

/// The interrupt of the terminal, after which the worker stops at the end of
/// its current step.
async fn interrupt() {
    let _ = tokio::signal::ctrl_c().await;
    eprintln!();
    eprintln!("Interrupting; waiting for current step to finish...");
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
/// - Object-storage URLs: `s3://bucket/prefix`, `gs://bucket/prefix`,
///   `az://container/prefix`, `file:///abs/path` (cloud schemes require the
///   matching cargo feature).
/// - Bare paths: treated as a local directory, created if missing.
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

    // The URL form: `scheme://...` (any of the cloud schemes), `file://` or an
    // opaque scheme that parse_url knows.
    if looks_like_url(&raw) {
        let url = Url::parse(&raw).with_context(|| format!("parsing --store as URL: {raw}"))?;
        let (store, path) = parse_url_opts(&url, store_options(std::env::vars()))
            .with_context(|| format!("opening object store from URL `{raw}`"))?;
        return Ok(StoreCtx {
            object_store: Arc::new(PrefixStore::new(store, path)),
            source: raw,
        });
    }

    // Bare path: create the directory if missing, then wrap it in
    // LocalFileSystem with that directory as the LocalFileSystem prefix.
    let dir = PathBuf::from(&raw);
    tokio::fs::create_dir_all(&dir)
        .await
        .with_context(|| format!("creating store dir {}", dir.display()))?;
    let local = LocalFileSystem::new_with_prefix(&dir)
        .with_context(|| format!("opening LocalFileSystem at {}", dir.display()))?;
    Ok(StoreCtx {
        object_store: Arc::new(local),
        source: raw,
    })
}

/// The object store options among the environment variables `vars`: the
/// variables of the three providers, with the name in lower case as the option
/// key. The prefixes exclude a variable of another program whose lower-case
/// name is an option key, such as `TOKEN` or `ENDPOINT`.
fn store_options(
    vars: impl Iterator<Item = (String, String)>,
) -> impl Iterator<Item = (String, String)> {
    vars.filter_map(|(name, value)| {
        let key = name.to_ascii_lowercase();
        ["aws_", "google_", "azure_"]
            .iter()
            .any(|prefix| key.starts_with(prefix))
            .then_some((key, value))
    })
}

fn looks_like_url(s: &str) -> bool {
    // A scheme starts with a letter, and the URL contains `://`.
    s.contains("://") && s.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
}

/// Clap value parser for `--store` / `--output`. Bare paths pass through
/// untouched. A URL form must parse and have a known scheme, so the parser
/// rejects a typo such as `S3://...` or `aws://...` at argument parsing.
/// `taquba-research init` checks the connectivity (the credentials and the
/// existence of the bucket).
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
    let opts = OpenOptions::default()
        .default_queue_config(QueueConfig::default().lease_duration(LEASE_DURATION))
        .queue_config(
            WORKFLOW_CONTROL_QUEUE_NAME,
            QueueConfig::default().keep_done_jobs(DEFAULT_INBOX_RECORD_RETENTION),
        )
        .inbox(InboxOptions::new([
            WORKFLOW_QUEUE_NAME,
            WORKFLOW_CONTROL_QUEUE_NAME,
        ]))
        .liveness_heartbeat(HEARTBEAT_INTERVAL);
    let queue = Queue::open_with_options(ctx.object_store.clone(), QUEUE_DB_NAME, opts)
        .await
        .context("opening taquba queue")?;
    Ok(Arc::new(queue))
}

/// Whether a queue has ever been created in this store. A [`QueueReader`]
/// cannot open a store without a manifest, so the inspection commands map that
/// case to "no runs".
async fn queue_exists(ctx: &StoreCtx) -> Result<bool> {
    use futures_util::TryStreamExt;
    let mut stream = ctx
        .object_store
        .list(Some(&ObjectPath::from(QUEUE_DB_NAME)));
    Ok(stream
        .try_next()
        .await
        .context("probing queue store")?
        .is_some())
}

/// Open a read-only view of the queue store. `FollowLatest` does not perform
/// object-store writes, so read-only credentials suffice.
async fn open_reader(ctx: &StoreCtx) -> Result<QueueReader> {
    QueueReader::open_with_options(
        ctx.object_store.clone(),
        QUEUE_DB_NAME,
        ReaderOptions::default().mode(ReaderMode::FollowLatest),
    )
    .await
    .context("opening queue reader")
}

/// Run `op` against a read-only view of the queue store, retrying once on
/// failure. A `FollowLatest` read can fail on an object collected under its
/// view. The retry opens a fresh reader whose view postdates the collection.
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

/// The workflow queries over `reader` and the runtime's memo store.
fn workflow_view(ctx: &StoreCtx, reader: &QueueReader) -> WorkflowView {
    store::workflow_view(reader, ctx.object_store.clone())
}

/// The status of `run_id` through `view`, or `None` for a run that the workflow
/// store does not contain.
async fn view_status(view: &WorkflowView, run_id: &RunId) -> Result<Option<RunStatus>> {
    view.status(run_id)
        .await
        .with_context(|| format!("reading the workflow status of {run_id}"))
}

/// The step job of `status` with its payload, while the queue retains the job.
async fn current_step(
    reader: &QueueReader,
    status: Option<&RunStatus>,
) -> Result<Option<JobRecord>> {
    match status {
        Some(status) => Ok(reader.view().get_job(&status.step.job_id).await?),
        None => Ok(None),
    }
}

/// The sender of inbox entries to the queues of the store.
fn inbox_sender(ctx: &StoreCtx) -> InboxSender {
    InboxSender::new(ctx.object_store.clone(), default_inbox_path(QUEUE_DB_NAME))
}

/// The first value of `read` through a reader of the store, polled while the
/// writer is live and for at most [`INBOX_WAIT`], or `None` without a live
/// writer or after the wait.
async fn await_writer<T>(
    ctx: &StoreCtx,
    read: impl AsyncFn(&QueueReader) -> Result<Option<T>>,
) -> Result<Option<T>> {
    let reader = QueueReader::open_with_options(
        ctx.object_store.clone(),
        QUEUE_DB_NAME,
        ReaderOptions::default()
            .mode(ReaderMode::FollowLatest)
            .manifest_poll_interval(Duration::from_secs(1)),
    )
    .await
    .context("opening queue reader")?;
    let value = poll_reader(&reader, read).await;
    let _ = reader.close().await;
    value
}

/// The first value of `read` through `reader`, polled while the writer is live
/// and for at most [`INBOX_WAIT`].
async fn poll_reader<T>(
    reader: &QueueReader,
    read: impl AsyncFn(&QueueReader) -> Result<Option<T>>,
) -> Result<Option<T>> {
    if !writer_is_live(reader).await? {
        return Ok(None);
    }
    let deadline = Instant::now() + INBOX_WAIT;
    loop {
        if let Some(value) = read(reader).await? {
            return Ok(Some(value));
        }
        if Instant::now() >= deadline {
            return Ok(None);
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

/// The time of the earliest cancel command of each run that awaits a writer, in
/// milliseconds from the Unix epoch, by run id. A command awaits a writer as an
/// inbox entry, and then as a job of the control queue until the runtime
/// applies it. An entry whose record committed before the deletion of its
/// object is listed as well.
async fn pending_cancellations(
    ctx: &StoreCtx,
    reader: &QueueReader,
) -> Result<HashMap<String, u64>> {
    use futures_util::TryStreamExt;
    const PAGE: usize = 256;
    let mut sent = HashMap::new();
    let mut add = |payload: &[u8], at_ms: u64| {
        if let Ok(taquba_workflow::Command::Cancel(cancel)) = rmp_serde::from_slice(payload) {
            sent.entry(cancel.run_id.to_string())
                .and_modify(|at: &mut u64| *at = (*at).min(at_ms))
                .or_insert(at_ms);
        }
    };
    let sender = inbox_sender(ctx);
    let entries = sender
        .pending(WORKFLOW_CONTROL_QUEUE_NAME)
        .await
        .context("listing the pending commands")?;
    for pending in entries {
        if *pending.kind() != EntryKind::Enqueue {
            continue;
        }
        if let EntryRead::Entry(InboxEntry::Enqueue(entry)) = sender.read(&pending).await? {
            add(&entry.payload, pending.sent_at_ms());
        }
    }
    for status in [JobStatus::Pending, JobStatus::Scheduled, JobStatus::Claimed] {
        let view = reader.view();
        let mut jobs = std::pin::pin!(view.jobs(WORKFLOW_CONTROL_QUEUE_NAME, status, PAGE));
        while let Some(job) = jobs.try_next().await? {
            add(&job.payload, job.enqueued_at);
        }
    }
    Ok(sent)
}

/// Whether the latest liveness beat of the writer is recent and not the beat of
/// a clean close.
async fn writer_is_live(reader: &QueueReader) -> Result<bool> {
    let Some(beat) = reader.writer_heartbeat().await? else {
        return Ok(false);
    };
    let now_ms = Utc::now().timestamp_millis().max(0) as u64;
    let age = Duration::from_millis(now_ms.saturating_sub(beat.at_ms));
    Ok(!beat.closed && age < 2 * beat.interval)
}

/// Number of claimed jobs visible through `reader`, across both queues. Guards
/// an exclusive writer open: opening the writer fences a live worker and
/// requeues its claimed jobs. Best-effort in both directions: a reader cannot
/// distinguish a live claim from an abandoned one, and reader lag can hide a
/// recent claim.
async fn count_claimed_jobs(reader: &QueueReader) -> Result<usize> {
    let mut claimed = 0usize;
    for queue in [WORKFLOW_QUEUE_NAME, FETCH_QUEUE_NAME] {
        claimed += reader
            .view()
            .list_jobs(queue, JobStatus::Claimed, None, 1)
            .await?
            .jobs
            .len();
    }
    Ok(claimed)
}

/// Destination of the final report, resolved from `--output`.
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
    let (store, path) = parse_url_opts(&url, store_options(std::env::vars()))
        .with_context(|| format!("opening output store from URL `{raw}`"))?;
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
            let key = store::report_path(&ObjectPath::default(), run_id);
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

fn build_runner(cli: &Cli) -> Result<ResearchStepRunner> {
    if cli.search != "tavily" {
        bail!(
            "unsupported search backend `{}`: the CLI supports only `tavily`",
            cli.search
        );
    }
    let tavily = Tavily::from_env().context("TAVILY_API_KEY not set or empty")?;
    let search: Arc<dyn SearchBackend> = Arc::new(tavily);
    Ok(match CliProvider::resolve(cli.provider) {
        CliProvider::OpenAi => {
            let client = openai::OpenAI::from_env().context("OPENAI_API_KEY missing or invalid")?;
            ResearchStepRunner::new_openai(client, search)
        }
        CliProvider::Anthropic => {
            let client =
                anthropic::Anthropic::from_env().context("ANTHROPIC_API_KEY missing or invalid")?;
            ResearchStepRunner::new_anthropic(client, search)
        }
        CliProvider::Ollama => {
            // Without an API key. The default is localhost:11434
            // (OLLAMA_API_BASE_URL).
            let client = ollama::Ollama::from_env().context("failed to build Ollama client")?;
            ResearchStepRunner::new_ollama(client, search)
        }
    })
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
        investigation_turns: cli.investigation_turns,
        clarification_wait_secs: cli.clarification_wait,
        ..ResearchConfig::new(model)
    }
}

/// Build a [`WorkflowRuntime`] with one claimer and a `CaptureHook` for the
/// terminal channel, and spawn its worker task. The worker exits on Ctrl+C or
/// on a terminal-hook signal sent through [`WorkerHandles::shutdown_tx`]. The
/// Ctrl+C branch prints an "Interrupting..." acknowledgement before the drain
/// of the current step.
async fn spawn_runtime(
    store_ctx: &StoreCtx,
    queue: Arc<Queue>,
    runner: ResearchStepRunner,
    run_id: &RunId,
    interrupt: impl Future<Output = ()> + Send + 'static,
) -> Result<(
    WorkflowRuntime<ResearchStepRunner, CaptureHook>,
    WorkerHandles,
)> {
    let (tx, rx) = oneshot::channel::<RunOutcome>();
    let hook = CaptureHook {
        run_id: run_id.clone(),
        tx: Mutex::new(Some(tx)),
    };
    // Build the JobRunner the Fetching step submits FetchPage jobs to.
    let (job_runner, job_handle) = spawn_fetch_runner(&queue, &store_ctx.object_store);
    let runner = runner
        .with_job_runner(job_runner)
        .with_queue(queue.clone())
        .with_report_store(store_ctx.object_store.clone(), &ObjectPath::default());

    // Sequential workflow: one claimer is enough. See agent.rs for context.
    let runtime =
        WorkflowRuntime::builder(queue.clone(), store_ctx.object_store.clone(), runner, hook)
            .queue_name(WORKFLOW_QUEUE_NAME)
            .memo_prefix(WORKFLOW_MEMO_PREFIX)
            .max_concurrent_steps(1)
            .memo_retention(MEMO_RETENTION)
            .control_queue(WORKFLOW_CONTROL_QUEUE_NAME)
            .build();
    let runtime = match runtime {
        Ok(runtime) => runtime,
        Err(e) => {
            let _ = job_handle.shutdown().await;
            return Err(e).context("building the workflow runtime");
        }
    };

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let worker_runtime = runtime.clone();
    let worker = tokio::spawn(async move {
        worker_runtime
            .run(async move {
                // Exit on the interrupt or on the shutdown signal that
                // `finalize` sends after a terminal outcome.
                tokio::select! {
                    () = interrupt => {}
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
    runner: ResearchStepRunner,
    query: String,
    interrupt: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    let config = build_config(cli);
    let queue = open_queue(store_ctx).await?;

    let waiting = store::count_waiting_step_jobs(queue.view())
        .await
        .context("counting queued runs")?;
    if waiting > 0 {
        eprintln!(
            "Note: {waiting} queued run(s) already exist in this store; this worker will \
             execute their pending steps too, under this invocation's provider settings. \
             Inspect them with `list`."
        );
    }

    // The run id is generated before submit so the terminal hook can filter
    // notifications to this run.
    let run_id = RunId::new(ulid::Ulid::new().to_string()).expect("a ULID is a valid run id");
    let (runtime, handles) =
        spawn_runtime(store_ctx, queue.clone(), runner, &run_id, interrupt).await?;

    let input = ResearchStepRunner::initial_state(query.clone(), config);
    runtime
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
        .context("submitting run")?;

    println!(
        "Run {run_id} started. (Ctrl+C to interrupt; resume with `taquba-research resume {run_id}`)"
    );

    let result = finalize(cli, store_ctx, handles, &run_id).await;
    drop(runtime);
    result.and(close_queue(queue).await)
}

async fn cmd_resume(
    cli: &Cli,
    store_ctx: &StoreCtx,
    runner: ResearchStepRunner,
    run_id: String,
    force: bool,
    interrupt: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    let run_id: RunId = run_id
        .parse()
        .with_context(|| format!("invalid run id `{run_id}`"))?;
    // Guard against resuming a finished, dead-lettered or unknown run before
    // the (exclusive) writer open. A reader-side check suffices: the worker
    // acts on the same step job.
    {
        if !queue_exists(store_ctx).await? {
            bail!("no run {run_id} (store contains no runs)");
        }
        let (claimed, waiting, status, cancel_requested) = with_reader(store_ctx, async |reader| {
            let claimed = count_claimed_jobs(reader).await?;
            let waiting = store::count_waiting_step_jobs(reader.view()).await?;
            let status = view_status(&workflow_view(store_ctx, reader), &run_id).await?;
            let cancel_requested = pending_cancellations(store_ctx, reader)
                .await?
                .contains_key(run_id.as_str());
            Ok((claimed, waiting, status, cancel_requested))
        })
        .await?;
        let state = status.map(|s| s.state);
        match state {
            None => bail!("no run {run_id} (unknown or expired)"),
            Some(RunState::Terminated(termination)) => {
                bail!("run {run_id} is already terminal ({})", termination.status)
            }
            Some(_) => {}
        }
        // The waiting count includes this run's pending or scheduled step.
        let waiting_others = match state {
            Some(RunState::Pending | RunState::Scheduled { .. } | RunState::Waiting { .. }) => {
                waiting.saturating_sub(1)
            }
            _ => waiting,
        };
        // Guard before taking the exclusive writer, as in `cmd_gc`. The caveats
        // are on `count_claimed_jobs`.
        if claimed > 0 && !force {
            bail!(
                "claimed jobs are visible in this store; a worker may be live. \
                 Re-run with --force only when no `run`/`resume` process is active."
            );
        }
        if waiting_others > 0 {
            eprintln!(
                "Note: {waiting_others} other queued run(s) exist in this store; this worker \
                 will execute their pending steps too, under this invocation's provider \
                 settings."
            );
        }
        if cancel_requested {
            eprintln!(
                "Note: run {run_id} has a cancellation requested. Resuming applies it, and the \
                 run ends as cancelled unless it terminates first."
            );
        }
    }
    let queue = open_queue(store_ctx).await?;

    // cmd_resume does not submit new work. It starts a worker to process the
    // existing pending step.
    let (runtime, handles) =
        spawn_runtime(store_ctx, queue.clone(), runner, &run_id, interrupt).await?;

    println!("Resuming {run_id}...");

    // The run's step jobs already exist in the queue. The terminal hook fires
    // when the run reaches a terminal step.
    let result = finalize(cli, store_ctx, handles, &run_id).await;
    drop(runtime);
    result.and(close_queue(queue).await)
}

/// Close the writer `queue` after its worker and fetch runner stop, so its last
/// liveness beat marks a clean close.
async fn close_queue(queue: Arc<Queue>) -> Result<()> {
    let queue = Arc::into_inner(queue).ok_or_else(|| anyhow!("the queue is still shared"))?;
    queue.close().await.context("closing taquba queue")
}

/// Handles tied to a running `WorkflowRuntime` worker task and its paired
/// fetch-job `JobRunner`.
struct WorkerHandles {
    /// Terminal-hook signal.
    rx: oneshot::Receiver<RunOutcome>,
    /// Spawned workflow worker task.
    worker: tokio::task::JoinHandle<taquba_workflow::Result<()>>,
    /// Sends a clean-shutdown signal so the workflow worker stops polling once
    /// the run is done (without waiting for Ctrl+C).
    shutdown_tx: oneshot::Sender<()>,
    /// Handle to the fetch JobRunner's worker task. It must shut down only
    /// after the workflow worker drains. The workflow step that submits a
    /// FetchPage awaits it, so no fetch work remains then.
    job_handle: RunnerHandle,
}

/// Flatten a worker task's join result: a panic or abort (`JoinError`) and a
/// [`WorkflowRuntime::run`] error are both returned as the error.
fn flatten_worker_exit(
    joined: std::result::Result<taquba_workflow::Result<()>, tokio::task::JoinError>,
) -> Result<()> {
    Ok(joined??)
}

async fn finalize(
    cli: &Cli,
    store_ctx: &StoreCtx,
    handles: WorkerHandles,
    run_id: &str,
) -> Result<()> {
    // Either the terminal hook fires (run reached a terminal step) or the
    // worker exited (Ctrl+C or a worker error). Whichever happens first
    // determines the exit message.
    let WorkerHandles {
        mut rx,
        mut worker,
        shutdown_tx,
        job_handle,
    } = handles;
    let result = tokio::select! {
        out = &mut rx => {
            // Run reached a terminal step. Tell the worker to stop polling the
            // queue, then await its clean exit before writing the report. The
            // outcome is already committed, so an exit error is logged only.
            let _ = shutdown_tx.send(());
            if let Err(e) = flatten_worker_exit((&mut worker).await) {
                tracing::warn!(error = %e, "worker exit error after the terminal outcome");
            }
            handle_terminal(cli, store_ctx, run_id, out.ok()).await
        }
        joined = &mut worker => {
            // Worker exited without a terminal signal. The pending step job
            // keeps the run resumable, without an index write. Drop
            // shutdown_tx, whose receiver is dropped.
            drop(shutdown_tx);
            match flatten_worker_exit(joined) {
                // Clean exit: the worker's Ctrl+C branch resolved.
                Ok(()) => {
                    print_interrupted(run_id);
                    Ok(())
                }
                Err(e) => Err(e.context(format!(
                    "workflow worker failed; run {run_id} remains resumable"
                ))),
            }
        }
    };
    // The workflow worker stopped (terminal or Ctrl+C). Drain the fetch
    // JobRunner so its task exits before this fn returns.
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
            // `reports/`, and `--output` adds a copy elsewhere.
            let stored = format!(
                "{} (in configured store)",
                store::report_path(&ObjectPath::default(), run_id)
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
            // User-initiated stop, which exits 0.
            let reason = outcome.error.as_deref().unwrap_or("(no reason supplied)");
            eprintln!("⊘ Run {run_id} cancelled: {reason}");
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

/// Truncate `s` to at most `max` characters, replacing the removed tail with
/// `...`. The cut is on a character boundary.
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

/// The `status` lines of a transcript: each model reply, followed by its tool
/// calls, with `(no result)` on a call without a result.
fn transcript_lines(transcript: &[Record]) -> Vec<String> {
    let answered: HashSet<&str> = transcript
        .iter()
        .flat_map(|record| match record {
            Record::Results { results } => results.iter().map(|r| r.id.as_str()).collect(),
            _ => Vec::new(),
        })
        .collect();
    let mut lines = Vec::new();
    for record in transcript {
        let Record::Assistant { calls, .. } = record else {
            continue;
        };
        lines.push(format!("{:>2}. model", lines.len() + 1));
        for call in calls {
            let pending = if answered.contains(call.id.as_str()) {
                ""
            } else {
                " (no result)"
            };
            let arguments = ellipsize(&call.arguments.to_string(), 60);
            lines.push(format!(
                "{:>2}. {} {arguments}{pending}",
                lines.len() + 1,
                call.name
            ));
        }
    }
    lines
}

/// The query of `run`, or an empty string for a run submitted without
/// [`QUERY_HEADER`].
fn query_of(run: &ListedRun) -> &str {
    run.headers.get(QUERY_HEADER).map_or("", String::as_str)
}

/// An RFC 3339 rendering of `ms`, milliseconds from the Unix epoch.
fn rfc3339(ms: u64) -> String {
    chrono::DateTime::from_timestamp_millis(ms as i64)
        .map(|at| at.to_rfc3339())
        .unwrap_or_default()
}

/// One `list` row: the listed run plus its derived status.
struct RunRow {
    run: ListedRun,
    status: RunDisplayStatus,
}

/// Read every listed run and the pending cancellations, and derive the display
/// statuses. Rows are returned newest first.
async fn gather_rows(store_ctx: &StoreCtx) -> Result<Vec<RunRow>> {
    let active = |run: &ListedRun| !matches!(run.status.state, RunState::Terminated(_));
    let (runs, cancellations) = with_reader(store_ctx, async |reader| {
        let runs = store::list_runs(&workflow_view(store_ctx, reader)).await?;
        let cancellations = if runs.iter().any(active) {
            pending_cancellations(store_ctx, reader).await?
        } else {
            HashMap::new()
        };
        Ok((runs, cancellations))
    })
    .await?;
    Ok(runs
        .into_iter()
        .map(|run| {
            let cancel_requested = cancellations.contains_key(run.status.run_id.as_str());
            let status = store::derive_display_status(&run.status.state, cancel_requested);
            RunRow { run, status }
        })
        .collect())
}

async fn cmd_list(store_ctx: &StoreCtx) -> Result<()> {
    if !queue_exists(store_ctx).await? {
        println!("Store: {}", store_ctx.source);
        println!("No runs yet.");
        return Ok(());
    }
    let rows = gather_rows(store_ctx).await.context("listing runs")?;
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
        println!(
            "{:<28} {:<24} {:<25} {}",
            row.run.status.run_id,
            row.status.as_str(),
            rfc3339(row.run.submitted_at_ms),
            ellipsize(query_of(&row.run), 50)
        );
    }
    Ok(())
}

async fn cmd_status(store_ctx: &StoreCtx, run_id: String) -> Result<()> {
    if !queue_exists(store_ctx).await? {
        bail!("no run {run_id} (store contains no runs)");
    }
    let parsed: RunId = run_id
        .parse()
        .with_context(|| format!("invalid run id `{run_id}`"))?;
    let (run, progress, job, history, pending_cancel) = with_reader(store_ctx, async |reader| {
        let view = workflow_view(store_ctx, reader);
        let run = view.run(&parsed).await?;
        let pending_cancel = match run.as_ref().map(|run| &run.status.state) {
            None | Some(RunState::Cancelling { .. } | RunState::Terminated(_)) => None,
            Some(_) => pending_cancellations(store_ctx, reader)
                .await?
                .remove(&run_id),
        };
        let progress = store::run_progress(&view, &parsed).await?;
        // The step job is read for its attempts, its last error and its
        // payload.
        let job = current_step(reader, run.as_ref().map(|run| &run.status)).await?;
        // The attempt history is printed for dead-lettered steps only.
        let history = match &job {
            Some(job) if job.status == JobStatus::Dead => {
                reader.view().attempt_history(&job.id).await?
            }
            _ => Vec::new(),
        };
        Ok((run, progress, job, history, pending_cancel))
    })
    .await?;
    let run = run.ok_or_else(|| anyhow!("no run {run_id} (unknown or expired)"))?;

    let state = &run.status.state;
    let cancel_requested_at_ms = match state {
        RunState::Cancelling {
            requested_at_ms, ..
        } => Some(*requested_at_ms),
        _ => pending_cancel,
    };
    let display = store::derive_display_status(state, cancel_requested_at_ms.is_some());

    println!("run_id:       {run_id}");
    println!("query:        {}", query_of(&run));
    println!("status:       {}", display.as_str());
    println!("submitted_at: {}", rfc3339(run.submitted_at_ms));
    if let Some(ms) = cancel_requested_at_ms {
        println!("cancel_requested_at: {}", rfc3339(ms));
    }
    if let RunState::Waiting { timeout_at_ms } = state {
        println!("wait_ends_at: {}", rfc3339(*timeout_at_ms));
    }
    if let RunState::Terminated(termination) = state {
        println!("finished_at:  {}", rfc3339(termination.terminated_at_ms));
        if let Some(err) = &termination.error {
            println!("error:        {err}");
        }
        if let Some(kind) = termination.error_kind {
            let kind = match kind {
                StepErrorKind::Transient => "transient",
                StepErrorKind::Permanent => "permanent",
            };
            println!("error_kind:   {kind}");
        }
        let progress = progress.unwrap_or_default();
        let wall_time_secs = termination
            .terminated_at_ms
            .saturating_sub(run.submitted_at_ms)
            / 1000;
        println!(
            "stats:        {} steps · {wall_time_secs}s · {} tokens",
            progress.steps_completed, progress.token_usage.total_tokens,
        );
    }
    println!("step:         {}", run.status.step.number);
    let progress = job.as_ref().and_then(|job| summarize_state(&job.payload));
    if let Some(progress) = &progress {
        println!("phase:        {}", progress.phase);
    }
    if let Some(question) = progress.as_ref().and_then(|p| p.pending_question.as_ref()) {
        println!("question:     {question}");
        println!("              reply with: taquba-research reply {run_id} \"<text>\"");
    }
    let transcript = progress.map(|p| p.transcript).unwrap_or_default();
    if !transcript.is_empty() {
        println!("transcript:");
    }
    for line in transcript_lines(&transcript) {
        println!("  {line}");
    }
    if let Some(job) = &job {
        println!("attempts:     {}/{}", job.attempts, job.max_attempts);
        if let Some(err) = &job.last_error {
            println!("last_error:   {err}");
        }
        for attempt in &history {
            if let Some(err) = &attempt.error {
                println!("attempt {:>2}:   {err}", attempt.attempt);
            }
        }
    }
    Ok(())
}

async fn cmd_show(store_ctx: &StoreCtx, run_id: String, output: Option<&str>) -> Result<()> {
    // The canonical report blob is written by the terminal path in every case,
    // and its absence means the run is unknown, unfinished or did not succeed.
    // The view is consulted only for a more specific error message.
    let key = store::report_path(&ObjectPath::default(), &run_id);
    // The body read shares the get's error handling: either call can return
    // NotFound depending on the backend.
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
            let parsed: RunId = run_id
                .parse()
                .with_context(|| format!("invalid run id `{run_id}`"))?;
            let status = with_reader(store_ctx, async |reader| {
                view_status(&workflow_view(store_ctx, reader), &parsed).await
            })
            .await?;
            match status.map(|s| s.state) {
                None => bail!("run {run_id} has no stored report (unknown or expired)"),
                Some(RunState::Terminated(termination)) => bail!(
                    "run {run_id} has no stored report (terminal status: {})",
                    termination.status
                ),
                Some(_) => bail!("run {run_id} has no stored report (not finished)"),
            }
        }
        Err(e) => return Err(e).context("reading stored report"),
    };

    match output {
        None => {
            print!("{markdown}");
        }
        Some(raw) => {
            // `show --output` always writes to a concrete destination. The
            // store already contains the canonical copy.
            let target = resolve_output(Some(raw))?;
            let where_ = write_report(&target, store_ctx, &run_id, &markdown).await?;
            println!("✓ Report written to {where_}");
        }
    }
    Ok(())
}

/// The outcome of `reply` for the question of a waiting run.
#[derive(Debug, PartialEq, Eq)]
enum ReplyOutcome {
    /// A live writer delivered the reply to the waiting step.
    Delivered { question: String },
    /// The reply awaits a writer, without one live within [`INBOX_WAIT`].
    Sent { question: String },
}

fn print_reply(run_id: &str, outcome: &ReplyOutcome) {
    match outcome {
        ReplyOutcome::Delivered { question } => {
            println!("Reply delivered to {run_id} for \"{question}\".");
        }
        ReplyOutcome::Sent { question } => println!(
            "Reply sent to {run_id} for \"{question}\". It takes effect when a `run` or \
             `resume` process applies it, unless the wait ends first."
        ),
    }
}

async fn cmd_reply(store_ctx: &StoreCtx, run_id: String, text: &str) -> Result<ReplyOutcome> {
    if text.trim().is_empty() {
        bail!("a reply must contain text");
    }
    if !queue_exists(store_ctx).await? {
        bail!("no run {run_id} (store contains no runs)");
    }
    let parsed: RunId = run_id
        .parse()
        .with_context(|| format!("invalid run id `{run_id}`"))?;
    let (status, job) = with_reader(store_ctx, async |reader| {
        let status = view_status(&workflow_view(store_ctx, reader), &parsed).await?;
        let job = current_step(reader, status.as_ref()).await?;
        Ok((status, job))
    })
    .await?;
    let not_waiting = || anyhow!("run {run_id} is not waiting for a reply");
    let status = status
        .filter(|status| matches!(status.state, RunState::Waiting { .. }))
        .ok_or_else(not_waiting)?;
    let question = job
        .and_then(|job| summarize_state(&job.payload))
        .and_then(|progress| progress.pending_question)
        .ok_or_else(not_waiting)?;
    let entry = SignalEntry {
        job_id: status.step.job_id,
        payload: text.as_bytes().to_vec(),
    };
    let id = inbox_sender(store_ctx)
        .send(WORKFLOW_QUEUE_NAME, entry)
        .await
        .context("writing the inbox entry")?;
    let outcome = await_writer(store_ctx, async |reader| {
        let record = reader.view().inbox_record(WORKFLOW_QUEUE_NAME, &id).await?;
        Ok(record.map(|record| record.outcome))
    })
    .await?;
    match outcome {
        Some(InboxOutcome::Signal(SignalOutcome::Delivered)) => {
            Ok(ReplyOutcome::Delivered { question })
        }
        Some(InboxOutcome::Signal(SignalOutcome::NotWaiting)) => bail!(
            "run {run_id} stopped waiting before the reply arrived; the run proceeds on an \
             assumption"
        ),
        Some(InboxOutcome::Rejected(reason)) => bail!("reply rejected: {reason}"),
        Some(other) => bail!("unexpected outcome of the reply: {other:?}"),
        None => Ok(ReplyOutcome::Sent { question }),
    }
}

async fn cmd_cancel(store_ctx: &StoreCtx, run_id: String) -> Result<()> {
    if !queue_exists(store_ctx).await? {
        bail!("no run {run_id} (store contains no runs)");
    }
    let parsed: RunId = run_id
        .parse()
        .with_context(|| format!("invalid run id `{run_id}`"))?;
    let status = with_reader(store_ctx, async |reader| {
        view_status(&workflow_view(store_ctx, reader), &parsed).await
    })
    .await?;
    match status.map(|s| s.state) {
        Some(RunState::Terminated(termination)) => bail!(
            "run {run_id} is not cancellable (status: {})",
            termination.status
        ),
        None => bail!("no run {run_id} (unknown or expired)"),
        Some(_) => {}
    }
    let command = taquba_workflow::Command::Cancel(CancelCommand {
        reason: Some("Cancelled by user".to_string()),
        ..CancelCommand::new(parsed.clone())
    });
    let id = inbox_sender(store_ctx)
        .send(WORKFLOW_CONTROL_QUEUE_NAME, EnqueueEntry::from(command))
        .await
        .context("writing the inbox entry")?;
    // The state of the run after the runtime applies the command.
    let settled = await_writer(store_ctx, async |reader| {
        let view = workflow_view(store_ctx, reader);
        let command = match view.command(WORKFLOW_CONTROL_QUEUE_NAME, &id).await? {
            None | Some(CommandState::Queued { .. }) => return Ok(None),
            Some(command) => command,
        };
        let state = view_status(&view, &parsed)
            .await?
            .map(|status| status.state);
        Ok(Some((command, state)))
    })
    .await?;
    match settled {
        Some((CommandState::Failed(reason), _)) => bail!("cancellation rejected: {reason}"),
        Some((CommandState::Rejected(reason), _)) => bail!("cancellation rejected: {reason}"),
        Some((_, Some(RunState::Terminated(termination))))
            if termination.status == TerminalStatus::Cancelled =>
        {
            println!("Run {run_id} cancelled.");
        }
        Some((_, Some(RunState::Terminated(_)) | None)) => {
            bail!("run {run_id} terminated before the cancellation arrived")
        }
        Some(_) => println!(
            "Cancellation of {run_id} reached its running step. The run ends as `cancelled` \
             once the step returns."
        ),
        None => println!(
            "Cancellation sent for {run_id}. It takes effect when a `run` or `resume` process \
             applies it."
        ),
    }
    Ok(())
}

async fn cmd_init(store_ctx: &StoreCtx) -> Result<()> {
    use futures_util::TryStreamExt;
    println!("Probing store: {}", store_ctx.source);

    // The first entry of a `list` checks the credentials and the existence of
    // the bucket on every backend without a write, and an empty store returns
    // `None`.
    let mut stream = store_ctx.object_store.list(None);
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

async fn cmd_gc(store_ctx: &StoreCtx, older_than_days: Option<i64>, dry_run: bool) -> Result<()> {
    use futures_util::TryStreamExt;
    let cutoff = older_than_days.map(|d| Utc::now() - chrono::Duration::days(d));
    // Custom `--output` destinations are not tracked and are not deleted.
    let prefix = ObjectPath::from(store::REPORTS_PREFIX);
    let mut candidates: Vec<ObjectMeta> = store_ctx
        .object_store
        .list(Some(&prefix))
        .try_filter(|meta| {
            std::future::ready(
                meta.location.extension() == Some("md")
                    && cutoff.is_none_or(|c| meta.last_modified < c),
            )
        })
        .try_collect()
        .await
        .context("listing the stored reports")?;
    candidates.sort_by_key(|meta| meta.last_modified);

    if candidates.is_empty() {
        println!("No reports match the gc filter.");
        return Ok(());
    }

    println!("Store: {}", store_ctx.source);
    println!(
        "{} candidate{}:",
        candidates.len(),
        if candidates.len() == 1 { "" } else { "s" }
    );
    for meta in &candidates {
        let run_id = meta
            .location
            .filename()
            .and_then(|name| name.strip_suffix(".md"))
            .unwrap_or_default();
        println!("  {run_id}  {}", meta.last_modified.to_rfc3339());
    }

    if dry_run {
        println!("(dry-run: no reports deleted)");
        return Ok(());
    }

    let mut deleted = 0usize;
    let mut errors = 0usize;
    for meta in &candidates {
        match store_ctx.object_store.delete(&meta.location).await {
            Ok(_) | Err(taquba::object_store::Error::NotFound { .. }) => deleted += 1,
            Err(err) => {
                tracing::warn!(path = %meta.location, error = %err, "gc report delete failed");
                errors += 1;
            }
        }
    }
    println!("Deleted {deleted} report(s); {errors} error(s).");
    Ok(())
}

struct CaptureHook {
    /// Run this invocation submitted or resumed. Notifications for any other
    /// run are stale (see `on_termination`).
    run_id: RunId,
    tx: Mutex<Option<oneshot::Sender<RunOutcome>>>,
}

impl TerminalHook for CaptureHook {
    async fn on_termination(
        &self,
        outcome: &RunOutcome,
        _effects: &TerminalEffects,
    ) -> std::result::Result<(), StepError> {
        // Terminal notifications are durable jobs: one left unclaimed by a
        // terminated process is delivered to the next worker on the queue. The
        // hook acks and logs a foreign notification without consuming its
        // outcome.
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

#[cfg(test)]
mod tests;
