# Story 14.3b — Coder timeout repin

Read custom-handoffs/14-3-timeout/requirements.md and architecture.md in full, plus current filtered repin PASS/GREEN handoffs.

Implement only architect scope: change `SMOKE_TIMEOUT_SECONDS = 600` to `1200` in scripts/ds4_segmented_smoke.py; add tracked synthetic timeout tests for 1200 constant/message and unchanged lock timeout 60/abort code 2/watchdog derivation; update docs/architecture.md, docs/backlog.md status, and timeout requirements. No scripts/finetune_ds4.py change. Preserve all other paths/parameters/lock/preflight/report/no-retry/fallback behavior.

TDD red→green. Stage touched files, register revision `14-3b-coder-r1` with deterministic four-file functional hash, run exact-fork focused/provider/source suites, py_compile, Path A/pins. No real assets, smoke, training, inference, install/network, commit/push. Write custom-handoffs/14-3-timeout/coder-notes.md and marker.