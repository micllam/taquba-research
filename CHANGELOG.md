# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
