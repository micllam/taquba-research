# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Step-error classification reads the provider HTTP status via rig
  0.41's `PromptError::provider_response_status`, so a provider error
  that preserves a status without an HTTP-level error (an error
  envelope on a 2xx response, or a `ProviderResponse` with a 4xx
  status) is classified by that status instead of defaulting to
  transient.
- Steps and fetch jobs extend their delivery lease to cover each slow
  call (`LeaseHandle::ensure_at_least`, new in taquba 0.11): LLM
  completions run under a new 10-minute timeout, search-backend calls
  under a new 60-second timeout, the fetch fan-out join re-extends
  before each page-job await and the `FetchPage` handler extends to
  cover its 20-second HTTP timeout. Previously a step slower than the
  30-second lease was silently re-queued and executed twice; a hung
  call now times out as a transient step error.
- The CLI opens its queue with an explicit `lease_duration` of 30
  seconds, the taquba default, chosen deliberately as the
  hang-detection bound now that slow calls extend it.
- **Breaking:** bumped the taquba stack: `taquba` 0.8 -> 0.11,
  `taquba-workflow` 0.6 -> 0.9, `taquba-jobs` 0.4 -> 0.7. The
  re-exported `workflow` types follow taquba-workflow 0.9's breaking
  changes: `Step` gained the `lease` and `signal` fields (tests
  constructing one add `lease: taquba::LeaseHandle::detached()` and
  `signal: None`), and `StepOutcome::Continue` gained a `when` field
  (use `StepOutcome::continue_now`).
- **Breaking (on-disk):** taquba 0.10 moved the queue's internal keys
  to a binary encoding, so stores written by earlier releases are
  unreadable. Start from a fresh `--store`; there is no migration.
- **Breaking:** `spawn_fetch_runner` returns the runner and handle
  directly; `JobRunnerBuilder::build` is infallible in taquba-jobs 0.7.
- **Breaking:** building requires rustc 1.88 (rig 0.41 uses
  edition-2024 let-chains). The crate declares no `rust-version`, so an
  older toolchain fails with E0658 errors from inside rig's derive
  macros.
- Bumped `rig-core` to 0.41 and added `rig-agent` 0.41, which now
  contains the classic agent runtime (prompting traits, `PromptError`,
  `StructuredOutputError`).
- A run interrupted by a process kill re-delivers its in-flight step
  immediately on `resume`: taquba 0.11 requeues claimed jobs found at
  queue open. Previously the interrupted step became claimable only
  after its lease expired.
- Step payloads above 256 KiB (the fetched corpus from Fetching onward,
  at default config) are written once to a payload object under the
  `queue-payloads` sibling prefix in the same store; previously every
  transition rewrote them inline.
- Ollama now honours `max_tokens_per_call`; rig 0.41 sends it as
  `options.num_predict`. Previously it was silently ignored for Ollama.

## [0.4.0] - 2026-06-17

### Added
- Anthropic synthesis requests now pass fetched pages as citation-enabled
  document blocks, and rendered reports include the source excerpts
  Claude cited when citation metadata is returned.
- Ollama provider support for local models: `ResearchStepRunner::new_ollama`,
  the `ResearchAgent` builder's `.ollama(...)`, and `--provider ollama` on
  the CLI (connects to `http://localhost:11434` by default, override with
  `OLLAMA_API_BASE_URL`, no API key required). Document-citation synthesis
  remains Anthropic-only; Ollama runs use the plain numeric source list.

### Changed
- Bumped `rig-core` to 0.38.
- Bumped the taquba stack: `taquba` 0.7 -> 0.8, `taquba-workflow` 0.5 -> 0.6,
  `taquba-jobs` 0.3 -> 0.4.
- **Breaking (on-disk):** taquba-workflow 0.6 and taquba-jobs 0.4 changed
  the terminal-marker filename format used by the memo- and result-retention
  sweepers. When upgrading a store that previously ran with retention
  enabled (which it always is here), clear the stale markers out-of-band so the
  new sweeper recognises them: delete the `workflow-memo/terminals/` and
  `research-fetch-jobs-results/terminals/` prefixes under your `--store`.
  Markers left behind are inert but never swept, so their blobs are
  retained indefinitely.
- **Breaking (on-disk):** the durable `ResearchState.synthesis` field
  changed type, so a run interrupted at or after the synthesizing phase
  under 0.3.0 cannot be resumed under this release. Drain or discard
  in-flight runs before upgrading; completed runs, reports, and the
  `list`/`show`/`status` subcommands are unaffected.

## [0.3.0] - 2026-05-29

### Added
- Memoization of every LLM call via
  [`taquba_workflow::Memo`](https://docs.rs/taquba-workflow). The
  planning, summarizing, synthesizing, and writing phases each cache
  their structured or text response under a
  `(run_id, step_number)`-scoped key, so an at-least-once retry of any
  of these steps short-circuits to the prior attempt's cached output
  instead of re-paying for the same prompt.
- Parallel fetching as a single fan-out workflow step. `Phase::Fetching`
  now submits one `FetchPage` taquba-job per URL to a `JobRunner`
  sharing the queue (under the `research-fetch-jobs` queue-name) and
  `try_join_all`s the handles. The job's `idempotency_key` derives from
  `(run_id, url)`, so taquba-jobs's result-aware idempotent submit
  short-circuits to cached result blobs on step retry; no URL is
  fetched twice across attempts. Per-URL handler failures classify
  transient (5xx, 429, transport) for queue-level retries or permanent
  (other 4xx, non-text, empty) for immediate dead-letter; on
  exhaustion the surrounding step skips the URL, preserving the prior
  best-effort semantic. Infrastructure errors from the JobRunner
  propagate as transient `StepError`.
- Public `spawn_fetch_runner(queue, object_store)` helper that builds
  and spawns the fetch `JobRunner` (registers `FetchPage`, attaches an
  `Arc<reqwest::Client>` on its state) and returns it alongside a
  `RunnerHandle` for graceful shutdown. Used internally by both
  `ResearchAgent::run` and the CLI's `spawn_runtime`.
- Public `ResearchStepRunner::with_job_runner(...)` builder method to
  attach the runner returned by `spawn_fetch_runner`. Required for any
  caller driving a custom `WorkflowRuntime` that advances into
  `Phase::Fetching`.
- New public re-export module `taquba_research::jobs` exposing
  `JobRunner` and `RunnerHandle`.
- Public `ResearchStepRunner::with_queue(Arc<Queue>)` builder method
  to attach the underlying queue. Required by the fetching phase so
  it can call `Queue::cancel(job_id)` on in-flight FetchPage jobs
  when the surrounding run is cancelled.
- Cancellation now cascades from the workflow run to its in-flight
  FetchPage jobs. When the cancel sentinel fires mid-fetch,
  `run_fetching`'s drop guard cancels every still-pending job via
  `Queue::cancel`, and `FetchPage::run` races the HTTP fetch against
  `JobContext::cancel_token`, so the actual reqwest call aborts
  instead of running out the per-fetch HTTP timeout. New
  `FetchError::Cancelled` variant; classified permanent.

### Changed
- **Breaking:** `ResearchAgent::run` now takes an additional
  `object_store: Arc<dyn ObjectStore>` argument; the new signature is
  `run(queue, object_store, query)`. The store backs the workflow
  runtime's per-step memo store; the common case is to pass the same
  store the `Queue` was opened with.
- Workflow memo blobs and fetch-job result blobs now auto-sweep 7
  days after their owning run/job reaches a terminal state.
  `ResearchAgent::run` and the CLI's `spawn_runtime` set
  `WorkflowRuntimeBuilder::memo_retention`, and `spawn_fetch_runner`
  sets `JobRunnerBuilder::result_retention`.
- Bumped `taquba` to 0.7 and `taquba-workflow` to 0.5; added
  `taquba-jobs` 0.3 as a new dependency.

## [0.2.0] - 2026-05-21

### Added
- Re-export `taquba_workflow::SubmitOutcome` from the `workflow`
  module, alongside the other types users need to call
  `WorkflowRuntime::submit` directly.
- Anthropic LLM provider support via Rig. The runner now dispatches
  per-provider via an internal enum;
  `ResearchStepRunner::new_anthropic` constructs a runner against an
  `anthropic::Client`.
- `ResearchConfig::new(model)` constructor: takes the
  provider-specific model identifier explicitly and applies the
  standard defaults to every other field.
- CLI `--provider openai|anthropic` flag and matching
  `ResearchAgentBuilder::anthropic(...)` method. Provider selection
  through the high-level builder and the CLI is now end-to-end. When
  `--provider` is omitted, the CLI auto-detects from env vars:
  `ANTHROPIC_API_KEY` alone selects Anthropic; otherwise OpenAI. The
  `--model` flag becomes optional and defaults to a
  provider-appropriate identifier when unset.
- Per-run token usage tracking. Every LLM call's `Usage` (input,
  output, total, cached-input, cache-creation, reasoning tokens) is
  logged at info level and accumulated into
  `ResearchState::token_usage` (persisted as part of the durable
  state so resumes don't lose the running tally) and the final
  `RunStats::token_usage`. The rendered report's stats line now
  includes a `tokens: <in> in / <out> out / <total>` suffix when the
  provider reported usage. New public type
  [`TokenUsage`](struct.TokenUsage.html).

### Removed
- **Breaking:** `ResearchStepRunner::new` is renamed to
  `new_openai` so the OpenAI and Anthropic constructors are
  symmetric without any default.
- **Breaking:** `impl Default for ResearchConfig` removed. The
  previous default returned `"gpt-4o-mini"`.
  `ResearchAgent`'s builder now requires a `config(...)` call as a
  consequence.

### Changed
- Bumped `rig-core` to 0.37. The crate's lib name changed from
  `rig` to `rig_core`.
- Bumped `taquba` to 0.6 and `taquba-workflow` to 0.4.
- The cancellation sentinel is now polled concurrently with the
  step's phase work instead of only at the start of each step. A
  long-running LLM or HTTP call is dropped within ~1 second of the
  CLI's `cancel` command landing, rather than blocking until the
  call completes.

## [0.1.0] - 2026-05-13

Initial release.
