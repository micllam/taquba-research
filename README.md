# taquba-research

Durable research agent for Rust, built on
[Rig](https://crates.io/crates/rig-core) and the
[taquba](https://crates.io/crates/taquba) stack.

Give it a question and it plans, searches the web, fetches and reads
pages, then synthesizes a cited report. The whole multi-step run
**survives process crashes**. Every transition is persisted to object
storage and every completed LLM call is memoized, so a run that dies
after twenty paid model calls resumes and re-pays for **none** of them.
Ctrl+C it, lose the machine, redeploy mid-run: it continues from the last
completed step.

This is a **reference implementation and CLI tool**: a worked example of
how Rig (LLM orchestration) and taquba (durable, object-storage-backed
queues, workflows, and jobs) combine into a crash-safe agent. Install it
to use it; read it to learn the durable-agent pattern. It is not meant as
a general-purpose library dependency (see
[Two public surfaces](#two-public-surfaces)).

## Install

```bash
cargo install taquba-research
```

OpenAI, Anthropic, and Ollama (local models) are supported via Rig. The
CLI picks based on which `*_API_KEY` env var is set (`ANTHROPIC_API_KEY`
alone selects Anthropic; otherwise OpenAI); pass
`--provider openai|anthropic|ollama` to force a choice. Ollama is never
auto-selected; request it explicitly; it connects to
`http://localhost:11434` by default (override `OLLAMA_API_BASE_URL`) and
needs no API key. The library exposes `ResearchStepRunner::new_openai`,
`ResearchStepRunner::new_anthropic`, and `ResearchStepRunner::new_ollama`
plus matching `.openai(...)` / `.anthropic(...)` / `.ollama(...)` builder
methods on `ResearchAgent`.

Anthropic runs pass fetched pages as citation-enabled document blocks
during synthesis; when Claude returns citation metadata, the final
report includes the cited source excerpts. OpenAI and Ollama runs keep
the standard numeric source-list citations. The Planning and Summarizing
phases use structured completions, so an Ollama model must reliably emit
schema-valid JSON; small local models that don't will dead-letter those
steps.

## Run

```bash
export OPENAI_API_KEY=...      # or: export ANTHROPIC_API_KEY=...
export TAVILY_API_KEY=...
taquba-research "your research question"
```

The CLI prints the run id at submission. Hit Ctrl+C any time; the run
state persists in `~/.taquba-research/queue/`. Resume with:

```bash
taquba-research resume <RUN_ID>
```

`run` and `resume` are foreground commands that stay alive for the
duration of the work and need `TAVILY_API_KEY` plus either
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY` depending on the chosen
`--provider`. Everything below is inspection or maintenance; run them from
another shell while a run is in flight (or any time after); they only
touch the shared store and need neither key. Object-store credentials
(the standard `AWS_*` / `GOOGLE_*` / `AZURE_*` env vars) are read
independently for every subcommand whenever `--store` is a cloud URL.

Other subcommands:

- `list`, `status <id>`, `show <id>`, `cancel <id>`: inspect and
  manage recorded runs. `show <id> --output <path-or-url>` writes
  the report somewhere instead of stdout.
- `init`: verify the configured store is reachable (creds, bucket
  exists). Recommended before submitting an expensive run against a
  fresh cloud bucket.
- `gc --older-than-days N [--status S]... [--dry-run]`: clean up
  recorded runs and their default-location reports. Active runs
  (`running`, `paused`, `cancellation_requested`) are protected
  unless explicitly opted in via `--status <state>`.

See `taquba-research --help` for the full flag list.

## Storage

`--store` (or `TAQUBA_RESEARCH_STORE`) controls where the SlateDB queue,
the run index, and (by default) the rendered report all live. It
accepts either a local path or an object-storage URL:

```bash
# Local (default at ~/.taquba-research/)
taquba-research "..."

# Cloud (requires the matching cargo feature)
taquba-research --store s3://my-bucket/research "..."
taquba-research --store gs://my-bucket/research "..."
taquba-research --store az://my-container/research "..."
```

```bash
cargo install taquba-research --features aws    # S3 / MinIO
cargo install taquba-research --features gcp    # Google Cloud Storage
cargo install taquba-research --features azure  # Azure Blob
```

`--output` accepts the same path-or-URL form. When omitted, the report
is saved to `<store>/reports/<run_id>.md` in the same store as the
queue, so an S3-backed deployment keeps everything in one bucket.

## Two public surfaces

Both exist so you can run or embed *this research agent*. To add
durability to *your own* Rig agent, copy the pattern instead:
per-step memoization of LLM calls over `taquba_workflow::Memo`, plus the
Rig-error to transient/permanent `StepError` mapping. Those two pieces
carry over to any agent; the phase state machine here is research-specific.

- **High-level**: `ResearchAgent`, a builder that wires Rig, a
  `SearchBackend`, and a `ResearchConfig` into a
  `run(queue, object_store, query)` helper. This is what the embed example
  below uses, and what the CLI drives.
- **Low-level**: `ResearchStepRunner`, a `taquba_workflow::StepRunner`
  you can drop into your own `taquba_workflow::WorkflowRuntime` to compose
  research steps with other workflow steps, share a worker pool, or own
  the terminal hook yourself.

## Embed in your Rig app

```rust
use std::sync::Arc;
use rig_core::client::ProviderClient;
use taquba::{Queue, object_store::local::LocalFileSystem};
use taquba_research::{ResearchAgent, ResearchConfig, search::Tavily};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let store = Arc::new(LocalFileSystem::new_with_prefix("./store")?);
    let queue = Arc::new(Queue::open(store.clone(), "research").await?);

    let agent = ResearchAgent::builder()
        .openai(rig_core::providers::openai::Client::from_env()?)
        // ...or .anthropic(rig_core::providers::anthropic::Client::from_env()?)
        //       and a matching model id, e.g. "claude-haiku-4-5"
        .search(Tavily::from_env()?)
        .config(ResearchConfig::new("gpt-4o-mini"))
        .build()?;

    // `store` also backs the workflow's per-step memo, which short-
    // circuits LLM-call retries; sharing one store is the common case.
    let report = agent
        .run(queue, store, "Postgres vs SQLite for read-heavy workloads")
        .await?;
    println!("{}", report.markdown);
    Ok(())
}
```

## Durability

- **No re-paying on retry.** Each LLM-backed phase memoizes its output in
  its per-step `Memo`; an at-least-once redelivery short-circuits to the
  cached value instead of calling (and billing) the model again.

Inherited from taquba:

- **Single-process, single-writer.** All workers for one queue share
  one process.
- **At-least-once delivery.** Steps are idempotent for
  `(run_id, step_number)`.
- **Per-transition durability.** Every step's state change is a
  SlateDB write.

### Fetching is the one fan-out phase

Most phases are one workflow step per unit of work. Fetching is the
exception: a single workflow step submits one `FetchPage` taquba-job
per URL to a `JobRunner` sharing the queue (under a distinct
queue-name), then `try_join_all`s the handles. The per-URL
`idempotency_key` derives from `(run_id, url)`, so taquba-jobs's
result-aware idempotent submit short-circuits to cached result blobs
on step retry; no URL is fetched twice across attempts.

`spawn_fetch_runner` is the helper that builds and spawns this
`JobRunner`; both `ResearchAgent::run` and the CLI construct it
internally, but callers driving a custom `WorkflowRuntime` need to
call it themselves and attach the runner via
`ResearchStepRunner::with_job_runner` together with
`ResearchStepRunner::with_queue` (the latter lets the fetching step
cancel in-flight `FetchPage` jobs via `Queue::cancel` when the
surrounding run is cancelled, instead of letting them run out to
the per-fetch HTTP timeout).

## License

Dual-licensed under either

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))
