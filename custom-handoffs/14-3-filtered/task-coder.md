# Story 14.3a — Coder filtered smoke dataset repin

Read custom-handoffs/14-3-filtered/requirements.md and architecture.md in full, plus current r12 PASS/GREEN handoffs.

Implement only architect scope:
- Change `_PINNED_SMOKE_PATHS['data']` to `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke`.
- Change only `ds4-segmented-smoke` catalog `--data` to `dataset_root / 'mlx-4096-smoke'`; preserve default smoke-train/full-train/continue-train entries.
- Add tracked synthetic tests for filtered constant, original-path rejection, segmented catalog filtered path, and default catalog unchanged. No real asset reads in tests.
- Update docs/architecture.md and custom-handoffs/14-3-filtered/requirements.md provenance as architect specifies; update docs/backlog.md Story 14.3a status to implementation/ready for review.
- Keep all parameters, lock, preflight, timeout, reports, provider, vendor, Path A, ADRs and default behavior unchanged. Do not alter original dataset.

TDD red→green. Stage touched files, including new tests/docs/handoffs, but do not commit/push. Run focused smoke tests, finetune regression, source/provider protected suites with exact fork, py_compile, Path A exact digest and functional four-file hash. Do not run real assets, Phase 2, smoke, training, inference, install, network. Write custom-handoffs/14-3-filtered/coder-notes.md and marker only complete; STOP on scope drift.