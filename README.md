# taquba-research

Durable research agent for Rust. Built on [Rig](https://crates.io/crates/rig-core)
and [taquba-workflow](https://crates.io/crates/taquba-workflow).

Multi-step agent runs survive process crashes; resume from where you
stopped without re-paying for completed steps.

## Install

```bash
cargo install taquba-research
```

## Run

```bash
export OPENAI_API_KEY=...
export TAVILY_API_KEY=...
taquba-research "your research question"
```

The CLI prints the run id at submission. Hit Ctrl+C any time; the run
state persists in `~/.taquba-research/queue/`. Resume with:

```bash
taquba-research resume <RUN_ID>
```

`run` and `resume` are foreground commands that stay alive for the
duration of the work and need `OPENAI_API_KEY` and `TAVILY_API_KEY`
set. Everything below is inspection or maintenance; run them from
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

- **High-level**: `ResearchAgent`: a builder that wires Rig, a
  `SearchBackend`, and a `ResearchConfig` into a `run(queue, query)`
  helper. This is what the embed example below uses, and what the CLI
  drives.
- **Low-level**: `ResearchStepRunner`: a `taquba_workflow::StepRunner`
  you can drop into your own `taquba_workflow::WorkflowRuntime` if you
  need to compose research steps with other workflow steps, share a
  worker pool, or own the terminal hook yourself.

## Embed in your Rig app

```rust
use std::sync::Arc;
use rig::client::ProviderClient;
use taquba::{Queue, object_store::local::LocalFileSystem};
use taquba_research::{ResearchAgent, ResearchConfig, search::Tavily};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let store = Arc::new(LocalFileSystem::new_with_prefix("./store")?);
    let queue = Arc::new(Queue::open(store, "research").await?);

    let agent = ResearchAgent::builder()
        .openai(rig::providers::openai::Client::from_env()?)
        .search(Tavily::from_env()?)
        .config(ResearchConfig::default())
        .build()?;

    let report = agent.run(queue, "Postgres vs SQLite for read-heavy workloads").await?;
    println!("{}", report.markdown);
    Ok(())
}
```

## Durability invariants (inherited from taquba)

- **Single-process, single-writer.** All workers for one queue share
  one process.
- **At-least-once delivery.** Steps are idempotent for
  `(run_id, step_number)`.
- **Per-transition durability.** Every step's state change is a
  SlateDB write.

## License

Dual-licensed under either

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))
