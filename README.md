# taquba-research

taquba-research runs [Rig](https://crates.io/crates/rig-core) agents as durable
workflows on the [taquba](https://crates.io/crates/taquba) stack. Its host, the
code that moves an agent run between steps, gives each workflow step one model
call and the tool calls of its reply. The crate includes a research pipeline
with a command-line interface as a worked example of the host.

The step memo records the model reply and every tool result of a step, so a step
that runs again after a crash reuses them without calling the model or a tool
again. When the agent asks the user a question, the step ends and the run waits
without holding a worker. The reply then resumes the run in a later step.

The research pipeline starts from a question and ends with a cited report. On
the way, it plans, searches the web, fetches and summarises pages, investigates
the gaps with an agent on the host and synthesises the findings. A run that
stops after twenty paid model calls resumes from its last completed step and
does not pay for those calls again.

The host is internal to the crate, and the public API runs or embeds the
research pipeline. A project with its own durable Rig agent can adopt the same
form:

- A hand-run `AgentRun` of rig-agent as the state of each step.
- The step memo for each model reply and each tool result.
- A signal wait when the agent needs a human in the loop.

## Install

```bash
cargo install taquba-research
```

The CLI calls OpenAI, Anthropic or a local Ollama model through Rig. It chooses
Anthropic when `ANTHROPIC_API_KEY` is set and `OPENAI_API_KEY` is not, and
OpenAI in every other case. `--provider openai|anthropic|ollama` overrides that
choice.

To use a local Ollama model, pass `--provider ollama`. The CLI then connects to
`http://localhost:11434` without an API key, or to the address in
`OLLAMA_API_BASE_URL`.

The providers differ in their citations. On Anthropic, synthesis passes the
fetched pages as citation-enabled document blocks, and the report quotes the
excerpts that Claude cites. OpenAI and Ollama runs keep a numbered list of
sources.

The planning, summarising and investigating phases request structured output,
and the investigating phase also calls tools. An Ollama model must produce
schema-valid JSON and tool calls reliably, because a malformed reply
dead-letters its step.

## Run

A run requires the key of its provider and a Tavily key for the web search:

```bash
export OPENAI_API_KEY=...      # or: export ANTHROPIC_API_KEY=...
export TAVILY_API_KEY=...
taquba-research "your research question"
```

The CLI prints the run id when it submits the run. The process can stop at any
moment, through Ctrl+C or a crash, and the run state stays in
`~/.taquba-research/queue/`. `resume` continues the run from its last completed
step:

```bash
taquba-research resume <RUN_ID>
```

The command that starts a run and `resume` both stay in the foreground until the
work ends, and both require the two keys. The other subcommands inspect or
maintain the store, from another shell during a run or at any later time, and
they require neither key. When `--store` is a cloud URL, every subcommand reads
the standard `AWS_*`, `GOOGLE_*` or `AZURE_*` credentials.

The other subcommands:

- `list`, `status <id>`, `show <id>`, `cancel <id>`: inspect and manage the
  runs. A run stays listed until 7 days after it terminates, and its report
  stays until `gc` deletes it. `show <id> --output <path-or-url>` writes the
  report to that location in place of stdout.
- `reply <id> "<text>"`: reply to the question of a run that waits for one.
- `init`: check that the configured store is reachable, with valid credentials
  and an existing bucket. Run it before the first submission against a fresh
  cloud bucket.
- `gc [--older-than-days N] [--dry-run]`: delete the reports at the default
  location, or only those older than `N` days.

With `--clarification-wait <SECONDS>`, the investigating agent can ask the user
one question per run. `status` then shows the question, and `reply` delivers the
reply of the user within the wait. When the wait ends first, the agent proceeds
on an assumption and states it in its findings.

`reply` and `cancel` take effect within 2 seconds while a process runs the work.
Without such a process, they take effect at the next `resume`.

`taquba-research --help` lists every flag.

## Storage

By default, the queue and the reports are stored in `~/.taquba-research/`.
`--store`, or the `TAQUBA_RESEARCH_STORE` variable, moves them to another local
path or to object storage, such as `s3://bucket/prefix`, `gs://bucket/prefix`,
`az://container/prefix` or `file:///abs/path`:

```bash
# Local (default at ~/.taquba-research/)
taquba-research "..."

# Cloud (requires the matching cargo feature)
taquba-research --store s3://my-bucket/research "..."
taquba-research --store gs://my-bucket/research "..."
taquba-research --store az://my-container/research "..."
```

Each cloud scheme requires the cargo feature of its provider:

```bash
cargo install taquba-research --features aws    # S3 / MinIO
cargo install taquba-research --features gcp    # Google Cloud Storage
cargo install taquba-research --features azure  # Azure Blob
```

The CLI saves each report to `<store>/reports/<run_id>.md`, so an S3 deployment
keeps the queue and the reports in one bucket. `--output` writes an additional
copy to any path or URL of the same form.

## Two public surfaces

Both surfaces run the research pipeline, whose phases are specific to research.

`ResearchAgent` is the high-level surface. Its builder combines a Rig provider
client, a `SearchBackend` and a `ResearchConfig` into a
`run(queue, object_store, query)` helper, which the CLI and the example that
follows use.

`ResearchStepRunner` is the low-level surface, a `taquba_workflow::StepRunner`
for a caller's own `taquba_workflow::WorkflowRuntime`. That runtime can combine
research steps with other workflow steps, share a worker pool or install its own
terminal hook. It must also attach the fetch runner that `spawn_fetch_runner`
builds.

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

A run persists across a crash at any step, and a retry repeats only the work
that did not complete. taquba writes every state change of a step to the object
store and delivers each step at least once. The crate adds these guarantees:

- **A retry does not pay for a model call again.** Each step memoizes the reply
  of each completed model call, and a redelivered step reads the stored reply. A
  failed call stays out of the memo, so a retry calls the model again.
- **A page is fetched once per run.** Each fetch is a job that uses the run and
  the URL as its key, so a retried step receives the recorded page.
- **A wait for the user ends the step.** The run waits for a reply without a
  worker or a lease, and the step after the wait reads the reply from its
  signal.
- **Rig classifies each model failure.** A failure that Rig reports as retryable
  is a transient step error, which taquba retries up to `max_attempts`. Any
  other model failure dead-letters the run at once.
- **A hung call fails the step.** Every model and search call runs under a
  timeout within the lease of its step, and a timeout is a transient step error.

The store of taquba allows one writer at a time, so all workers of a queue must
run in one process and share one `Arc<Queue>`.

## License

Dual-licensed under either

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))
