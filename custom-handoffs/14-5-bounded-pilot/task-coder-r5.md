# Story 14.5 — Coder r5 lock/watchdog terminal closure

Read latest r4 blocked review. Direct TDD only; no nested dispatch/delegate/real access/commit/push.

1. Partial acquire cleanup: if owner write/fsync then unlink fails, preserve recoverable ownership state or atomically rename/quarantine lock; never leave blocking lock with cleared ownership. Test actual fsync+unlink failure.
2. Release unlink failure: do not clear ownership globals while lock survives; preserve retryable owned state and report failure. Test actual `_release_ft_lock` with unlink injection and exactly-once call accounting.
3. Enclose Event construction and every fallible operation after `signal.alarm()` in watchdog rollback; any setup failure must call alarm(0). Test Event and thread/start failures.
4. Replace shared early failure substitutions with true injections at actual start-save, provider, callback, timeout delivery, checkpoint, each evidence boundary, actual release unlink, Phase A partial failure, complete Phase B success/failure, final aggregation. Drive through run_phase and assert ordering/evidence/exactly-once release.
5. Comprehensive fs trap around every terminal test: os.open and Path open/read/write/unlink/replace/rename/resolve destinations must all be contained by resolved tmp_path (except explicit read-only tracked source fixtures). Cover Phase A/B success/failure and mutation matrix.
6. Keep one-attempt preservation and tombstone behavior. Qualify docs until gates pass. Force-track r5 notes/tasks; no markers staged.
7. Focused/exact/protected tests; write coder-notes-r5.md and marker.