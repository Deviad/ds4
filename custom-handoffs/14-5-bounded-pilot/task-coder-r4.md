# Story 14.5 — Coder r4 terminal safety closure

Read latest r3 blocked review. Fix exactly; direct TDD, no delegate/real access/commit/push.

1. `_acquire_ft_lock(workspace=None)` resolves current `PILOT_WORKSPACE` at call time, or run_phase passes explicit current workspace. Add traps around open/unlink proving every synthetic terminal test path remains under tmp_path; no `/Volumes/Data NVME` access.
2. Make success evidence fail-atomic. Any report/phase-marker/final-marker write failure must leave zero OK markers and surviving fail markers bound to final fail reports. Never overwrite a report after an OK marker survives. Add injection at every Phase A/B report and marker boundary and rollback cleanup tests.
3. Complete terminal mutation matrix: parser/config, preflight, start-save, provider, callback, checkpoint validation, report writes, each marker write, timeout, watchdog cancellation, exactly-once lock release, Phase B success/failure, final aggregation. Assert ordering and no contradictory evidence.
4. Add full byte-exact rendered catalog command equality and default-command byte guards.
5. Qualify docs terminal-evidence claim until gates pass. Track r4 notes/tests; no markers staged.
6. Run focused/exact suites, path trap, py_compile/diff/protected gates. Write coder-notes-r4.md and marker.