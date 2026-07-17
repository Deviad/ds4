# Story 14.5 — Coder r6 final production-path closure

Read latest r5 blocked review. Direct TDD only; no delegate/real access/commit/push.

1. `run_phase()` terminal cleanup must detect retained partial ownership even when `_acquire_ft_lock()` raises before `lock_acquired=True`. Release or atomically quarantine it before return; no blocking lock survives. Record release attempt/lifecycle truthfully. End-to-end inject owner fsync + cleanup unlink failure and assert no blocking lock, globals clean or durable quarantine, exactly one run_phase cleanup attempt.
2. Phase B dependency must require marker report_path resolves exactly canonical Phase A report path, marker SHA matches those exact bytes, and loaded report object equals parsed hashed payload. Add path substitution and content substitution mutations.
3. Add Phase A partial-progress failure after one completed update, ordered event trace across training failure→watchdog cancel→exactly-once release→failure report→fail marker. Assert release_attempts==1 when acquired.
4. Apply resolved tmp containment guard to every terminal family: evidence boundaries, one-attempt, partial locks, Phase A/B matrices.
5. Track r6 notes/tasks; no markers staged; docs remain pending gates.
6. Focused 69+ tests, canonical real-trainer 2, exact suite, protected/diff. Write coder-notes-r6.md and marker.