# AGENTS.md

This file provides guidance to agents when working with code in this repository.

Single-crate repo (`[lib]` + `[[bin]]`) built on the [`taquba`](https://crates.io/crates/taquba) workspace's `taquba`, `taquba-workflow`, and `taquba-jobs` crates. Audience: Rust developers using [Rig](https://crates.io/crates/rig-core).

## Build / test

Tests live inline in `mod tests`; there is no `tests/` directory.

The crate's `aws` / `gcp` / `azure` features mirror taquba's same-named flags and are mutually exclusive (taquba/SlateDB constraint). Pick one for cloud builds; the default is no cloud (local FS only).

Canonical local check (run before pushing):

```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
cargo clippy --features aws --all-targets -- -D warnings   # if touching cloud paths
```

## Architectural invariants (inherited from `taquba`)

These constrain almost every design decision; violating them breaks correctness.

- **Single-process, single-writer.** SlateDB allows only one writer per store, so all producers and workers for a given `Queue` must live in the same process and share one `Arc<Queue>`. Do not propose multi-node worker fleets.
- **At-least-once delivery.** `StepRunner::run_step` must be idempotent for `(run_id, step_number)`. Taquba can re-deliver a step if its lease expires before ack.
- **Durability is per-transition.** The bytes returned from `StepOutcome::Continue` are the entire durable state between steps. Design the `ResearchState` shape with serde stability in mind.
- **Pre-1.0:** taquba's minor bumps may break source compat *and* on-disk layout. Pin specific minor versions in `Cargo.toml`; bump deliberately.

## Storage model

A single `Arc<dyn ObjectStore>` backs the queue (`<store>/queue/`), run index (`<store>/runs/`), reports (`<store>/reports/`), workflow memo blobs (`<store>/workflow-memo/<run>/<step>/<key-hash>`), fetch-job result blobs (`<store>/research-fetch-jobs-results/<job-id>`), and cancellation sentinels (`<store>/runs/<id>.cancel`). When adding a new persisted artifact, derive its path from the same store so cloud users don't end up with split state across two backends.

Memo and fetch-job result blobs auto-sweep 7 days after their owning run/job terminates (set via `WorkflowRuntimeBuilder::memo_retention` and `JobRunnerBuilder::result_retention`); run-index entries, reports, and cancellation sentinels are user-managed via the CLI's `gc` subcommand.

## Fetching is the one fan-out phase

`Phase::Fetching` is a single workflow step that submits one `FetchPage` taquba-job per URL to a `JobRunner` sharing the queue (under the `research-fetch-jobs` queue-name), then `try_join_all`s the handles. The `FetchPage` job's `idempotency_key` is derived from `(run_id, url)`, so taquba-jobs's result-aware idempotent submit short-circuits to the cached result blob on step retry; no URL is fetched twice. Per-URL handler failures classify transient (5xx, 429, transport) or permanent (other 4xx, non-text, empty); after exhaustion they surface as `JoinError::Job` and the surrounding step skips that URL (best-effort), while `JoinError::Infra` propagates as a transient step error.

The JobRunner is constructed via `spawn_fetch_runner(&queue, &object_store)` in both `agent.rs::run` and the bin's `spawn_runtime`; it must be shut down only **after** the workflow worker has drained, since the workflow step submitting a job awaits its handle before returning.

## Adding a new phase

Round-trip serde for `ResearchState` must keep working when extending the state machine — `ResearchState::round_trip_serde` should cover any new fields. The recipe: (1) extend the `Phase` enum, (2) add the queue field to `ResearchState`, (3) wire a `run_phase_X` method, (4) update the match in `run_step`.

## Error mapping

The runner converts errors into `taquba-workflow` outcomes; the distinction matters:

- `StepOutcome::Cancel { reason }` — runner-issued user cancellation (cross-process sentinel-detected). The job is acked and the run terminates as `Cancelled`; **not** dead-lettered.
- `Err(StepError::transient)` — retryable infrastructure error (5xx, 429, transport). Taquba backs off and retries up to `max_attempts`.
- `Err(StepError::permanent)` — non-retryable infrastructure error (401/403, other 4xx, malformed state, schema-mismatch on typed prompts). Step is dead-lettered immediately.

The mapping helpers in `runner.rs` — `classify_rig_err`, `classify_structured_err`, `classify_http`, the `is_transient_status` policy, and `impl From<SearchError> for StepError` — define the policy together. Mirror their structure when adding new failure modes.

## Reserved header prefix

`taquba-workflow` reserves `workflow.*` on step-job headers. Use a different namespace (e.g. `research.*`) for custom headers; submission with reserved keys fails at `WorkflowRuntime::submit`.

## LLM providers

OpenAI and Anthropic dispatch through the internal `ProviderClient` enum in `runner.rs`; both are compiled in unconditionally (Rig doesn't feature-gate providers, so neither do we). The CLI's `--provider` flag auto-detects from env keys when unset (`ANTHROPIC_API_KEY` alone → Anthropic; otherwise OpenAI). Default model identifiers live on `CliProvider::default_model`; keep them in sync with the constants exported from `rig_core::providers::<p>::completion`.

## Don't re-add without discussion

- **USD cost accounting** — would require maintaining provider pricing tables. Token counts are surfaced via `RunStats::token_usage`; leave $ to downstream tooling.
- **Brave / Serper search backends** — first-party impls planned for v0.2; the `SearchBackend` trait is public so downstreams can implement them today.
- **Per-URL Fetching workflow steps** — the old shape (one URL per workflow step, popping `fetch_queue`) was replaced by the fan-out-to-jobs step.

## Docstring style

Keep docstrings about the code, not the conversation. State what a type or function is and any non-obvious behaviour or invariant; omit rationale that only makes sense in context of the change that introduced it (specific call sites, design history, debate that landed here). Where the *why* matters and is non-obvious, prefer a short note in this file (AGENTS.md) over a docstring that will drift as the code evolves.

## Content parity

Substantive content in `lib.rs`'s top-level `//!` docstring should mirror `README.md`: anything new (sections, design notes, semantics callouts) lands in both. Format may differ — `lib.rs` uses intra-doc `[Foo]` links and `# `-hidden rustdoc lines inside doctests; `README.md` uses URL links and full `#[tokio::main]` blocks so code is copy-pasteable.

## CHANGELOG hygiene

Any user-visible change — new or renamed public API, breaking signature change, behaviour change, dependency bump with downstream consequences — earns a one-bullet entry under `## [Unreleased]` in `CHANGELOG.md`, in the same commit as the code change. Use the Keep-a-Changelog section names already in use (`Added`, `Changed`, `Removed`, `Fixed`); prefix breaking changes with `**Breaking:**` so the next release-notes pass can spot them. Internal refactors and test-only changes don't need entries.
