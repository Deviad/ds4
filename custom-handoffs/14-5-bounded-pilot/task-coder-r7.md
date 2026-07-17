# Story 14.5 — Coder r7 final TOCTOU closure

Read latest r6 blocked review. Direct TDD only.

1. Phase B canonical report: read bytes exactly once; compute SHA-256 from those bytes; strict-parse the same bytes with duplicate-key rejection; compare parsed object to supplied report. No second path read. Add mutation replacing file at old hash/read boundary; must fail/be impossible.
2. Apply install_tmp_fs_guard to every terminal test invoking run_phase, success/failure evidence, acquire/release lock, watchdog cancellation; AST/meta-test asserts no uncovered terminal tests unless explicitly read/write-free.
3. Correct coder evidence to canonical documented result; run exact command expected 317 passed, 3 skipped, 1 warning, 2 subtests plus focused 71 and direct trainer 2.
4. Track r7 notes/tests, no markers staged, protected gates. Write coder-notes-r7.md and marker. No delegate/real assets/commit/push.