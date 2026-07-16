# Story 14.3b — Coder r2 reviewer closure

Read custom-handoffs/standby/review.md timeout r1 final and coder-notes.md.

Close exactly:
1. Update docs/backlog.md Story 14.3b status from BA REQUIREMENTS to implementation/gates complete without changing acceptance criteria.
2. Strengthen `test_backup_watchdog_deadline_uses_1200` so a mutation that changes `_backup_watchdog` deadline behavior fails; use a direct controlled monkeypatch/observed deadline or source-behavior oracle, no real timeout.
3. Stage all touched files and register a new deterministic revision `14-3b-coder-r2` with the exact current four-file cached-diff hash. Update coder notes with the exact hash. Do not alter production logic beyond necessary test; preserve only SMOKE_TIMEOUT_SECONDS=1200 production change.

Run exact-fork suites, timeout tests, py_compile, Path A/pins. No real assets/smoke/training/inference/commit/push. Write coder-notes-r2 and marker.