# taquba-research

Durable research agent for Rust, built on
[Rig](https://crates.io/crates/rig-core) and the
[taquba](https://crates.io/crates/taquba) stack.

Given a question, the agent plans, searches the web, fetches and reads pages,
then synthesises a cited report. The multi-step run persists across process
crashes: the agent writes every transition to object storage and memoizes every
completed LLM call. A run that stops after twenty paid model calls resumes from
the last completed step and does not pay for any of those calls again.

This crate is a reference implementation and a CLI tool. It is a worked example
of how Rig (LLM orchestration) and taquba (durable queues, workflows and jobs on
object storage) combine into a crash-safe agent. It is not intended as a
general-purpose library dependency (see
[Two public surfaces](#two-public-surfaces)).

## Install

```bash
cargo install taquba-research
```

OpenAI, Anthropic and Ollama (local models) are supported through Rig. Without
`--provider`, the CLI selects Anthropic when `ANTHROPIC_API_KEY` is set and
`OPENAI_API_KEY` is not, and OpenAI otherwise. Pass
`--provider openai|anthropic|ollama` to override the selection.

The CLI never selects Ollama automatically, so an Ollama run requires
`--provider ollama`. Ollama does not use an API key and connects to
`http://localhost:11434` unless `OLLAMA_API_BASE_URL` is set. The library
exposes `ResearchStepRunner::new_openai`, `ResearchStepRunner::new_anthropic`
and `ResearchStepRunner::new_ollama`, and the matching `.openai(...)`,
`.anthropic(...)` and `.ollama(...)` builder methods on `ResearchAgent`.

Anthropic runs pass the fetched pages as citation-enabled document blocks during
synthesis. When Claude returns citation metadata, the final report includes the
cited source excerpts. OpenAI and Ollama runs keep the standard numeric list of
sources. The `Planning` and `Summarizing` phases use structured completions, so
an Ollama model must emit schema-valid JSON reliably, and a model that does not
dead-letters those steps.

## Run

```bash
export OPENAI_API_KEY=...      # or: export ANTHROPIC_API_KEY=...
export TAVILY_API_KEY=...
taquba-research "your research question"
```

The CLI prints the run id at submission. Ctrl+C stops the process at any time,
and the run state persists in `~/.taquba-research/queue/`. Resume with:

```bash
taquba-research resume <RUN_ID>
```

`run` and `resume` are foreground commands that stay alive for the duration of
the work. They require `TAVILY_API_KEY` and the API key of the chosen
`--provider`. The other subcommands inspect or maintain the shared store, from
another shell while a run is in flight or at any later time, and they require
neither key. Every subcommand reads object-store credentials (the standard
`AWS_*`, `GOOGLE_*` or `AZURE_*` environment variables) when `--store` is a
cloud URL.

Other subcommands:

- `list`, `status <id>`, `show <id>`, `cancel <id>`: inspect and manage the
  recorded runs. `show <id> --output <path-or-url>` writes the report to that
  location in place of stdout.
- `init`: check that the configured store is reachable, with valid credentials
  and an existing bucket. Run it before the first submission against a fresh
  cloud bucket.
- `gc --older-than-days N [--status S]... [--dry-run]`: delete recorded runs and
  their reports at the default location. `--status` accepts `succeeded`,
  `failed`, `cancelled` and `unknown`. Only `unknown` deletes a run without a
  terminal record, and it selects the entries whose run the workflow store no
  longer contains.

See `taquba-research --help` for the full flag list.

## Storage

`--store` (or `TAQUBA_RESEARCH_STORE`) sets the location of the SlateDB queue,
the run index and, by default, the rendered report. It accepts either a local
path or an object-storage URL:

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

The CLI always saves the report to `<store>/reports/<run_id>.md` in the store of
the queue, so an S3-backed deployment keeps every file in one bucket. `--output`
accepts the same path-or-URL form and writes an additional copy there.

## Two public surfaces

Both surfaces run or embed this research agent. To make another Rig agent
durable, copy the pattern: per-step memoization of LLM calls over
`taquba_workflow::Memo`, and the mapping of Rig errors to a transient or
permanent `StepError`. Those two parts apply to any agent, and the phase state
machine of this crate is specific to research.

- **High-level**: `ResearchAgent`, a builder that combines Rig, a
  `SearchBackend` and a `ResearchConfig` into a
  `run(queue, object_store, query)` helper. The CLI and the embedding example in
  [Embed in a Rig app](#embed-in-a-rig-app) use it.
- **Low-level**: `ResearchStepRunner`, a `taquba_workflow::StepRunner` for a
  caller's own `taquba_workflow::WorkflowRuntime`. The runtime can combine
  research steps with other workflow steps, share a worker pool or install its
  own terminal hook.

## Embed in a Rig app

```rust
use std::sync::Arc;
use taquba::{Queue, object_store::local::LocalFileSystem};
use taquba_research::{ResearchAgent, ResearchConfig, search::Tavily};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let store = Arc::new(LocalFileSystem::new_with_prefix("./store")?);
    let queue = Arc::new(Queue::open(store.clone(), "research").await?);

    let agent = ResearchAgent::builder()
        .openai(rig_core::providers::openai::OpenAI::from_env()?)
        // ...or .anthropic(rig_core::providers::anthropic::Anthropic::from_env()?)
        //       with a matching model id, for example "claude-haiku-4-5".
        .search(Tavily::from_env()?)
        .config(ResearchConfig::new("gpt-5-nano"))
        .build()?;

    // `store` also backs the per-step memo of the workflow, which
    // short-circuits retried LLM calls.
    let report = agent
        .run(queue, store, "Postgres vs SQLite for read-heavy workloads")
        .await?;
    println!("{}", report.markdown);
    Ok(())
}
```

## Durability

- **A retry does not pay for a model call again.** Each LLM-backed phase
  memoizes its output in its per-step `Memo`. An at-least-once redelivery
  short-circuits to the stored value and does not call or bill the model again.
- **The delivery lease covers every slow call.** Every LLM and search call runs
  under a timeout, and the step extends its lease
  (`LeaseHandle::ensure_at_least`) by that bound before the call. The fetching
  step extends the lease again as each page job completes. A hung call times out
  as a transient step error and is retried.

Inherited from taquba:

- **Single-process, single-writer.** All workers for a queue share one process.
- **At-least-once delivery.** Steps must be idempotent for
  `(run_id, step_number)`.
- **Per-transition durability.** Every state change of a step is a SlateDB
  write.

### Fetching is the one fan-out phase

Every other phase is one workflow step per unit of work. The fetching phase is a
single workflow step that submits one `FetchPage` job per URL to a `JobRunner`
(from the `jobs` module of taquba-workflow), then awaits the handles with
`try_join_all`. The `JobRunner` shares the queue and uses a distinct queue name.
The per-URL `idempotency_key` derives from `(run_id, url)`, so on a step retry
the idempotent submit of the job runner returns the recorded result, and no URL
is fetched twice.

`spawn_fetch_runner` builds and spawns this `JobRunner`. `ResearchAgent::run`
and the CLI call it internally, and a caller with a custom `WorkflowRuntime`
will need to call it and attach the runner with
`ResearchStepRunner::with_job_runner` and `ResearchStepRunner::with_queue`. With
the queue attached, the fetching step cancels its in-flight `FetchPage` jobs
through `Queue::cancel` when the run is cancelled, and the jobs do not run until
the per-fetch HTTP timeout.

## License

Dual-licensed under either

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))
