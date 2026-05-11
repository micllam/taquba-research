# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Single-crate repo (`[lib]` + `[[bin]]`) built on the [`taquba`](https://crates.io/crates/taquba) workspace's `taquba` and `taquba-workflow` crates. Audience: Rust developers using [Rig](https://crates.io/crates/rig-core).

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

A single `Arc<dyn ObjectStore>` backs the queue (`<store>/queue/`), run index (`<store>/runs/`), reports (`<store>/reports/`), and cancellation sentinels (`<store>/runs/<id>.cancel`). When adding a new persisted artifact, derive its path from the same store so cloud users don't end up with split state across two backends.

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

## Don't re-add without discussion

- Token / USD cost accounting — deferred until Rig surfaces per-call usage metadata. Char-count estimates dressed as USD mislead more than help.
- Brave / Serper search backends — first-party impls planned for v0.2; the `SearchBackend` trait is public so downstreams can implement them today.
- Multi-provider models — OpenAI only via Rig for v0.1.

## Content parity

Substantive content in `lib.rs`'s top-level `//!` docstring should mirror `README.md`: anything new (sections, design notes, semantics callouts) lands in both. Format may differ — `lib.rs` uses intra-doc `[Foo]` links and `# `-hidden rustdoc lines inside doctests; `README.md` uses URL links and full `#[tokio::main]` blocks so code is copy-pasteable.
