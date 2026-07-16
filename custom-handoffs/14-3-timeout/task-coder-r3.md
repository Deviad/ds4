# Story 14.3b — Coder r3 final reviewer closure

Read custom-handoffs/standby/review.md r2 final and coder-notes-r2.

Close exactly:
1. Strengthen watchdog test with exact AST/source-expression oracle requiring deadline assignment expression `time.monotonic() + SMOKE_TIMEOUT_SECONDS + 5`; a mutation to +6 must fail. No waiting/assets.
2. Update docs/backlog.md Story 14.3b status to implementation complete with Reviewer/Test Manager pending/blocked, not Reviewer PASS. Preserve acceptance criteria.
3. Stage all touched files, compute the exact current four-file cached-diff hash using `git diff --cached --binary -- scripts/ds4_segmented_smoke.py tests/test_ds4_segmented_smoke.py docs/architecture.md docs/backlog.md | shasum -a 256`, update coder notes and registry to exactly that hash. No self-reference.

Run timeout tests/exact-fork full suite, py_compile, Path A/pins. No real assets/smoke/training/inference/commit/push. Write coder-notes-r3 and marker.