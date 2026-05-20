# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Re-export `taquba_workflow::SubmitOutcome` from the `workflow`
  module, alongside the other types users need to call
  `WorkflowRuntime::submit` directly.

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
