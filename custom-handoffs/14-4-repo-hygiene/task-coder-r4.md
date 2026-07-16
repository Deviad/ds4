# Story 14.4 — Coder r4 fresh-clone test fix

Read architecture-r2.md and Reviewer r2 BLOCKED report. Apply exact scoped fix:
1. RED evidence: with `agent-output/cmux-14-2/` absent, confirm `test_active_memory_matrix` fails before fix if still reproducible from saved reviewer evidence; do not revert staged cleanup just to recreate.
2. In `tests/test_ds4_segmented_loss_and_grad.py`, replace direct log `write_text` path with `_log_path`, `_log_path.parent.mkdir(parents=True, exist_ok=True)`, then `_log_path.write_text`, exactly as architecture-r2.
3. Compute new file SHA-256 and update only `TestProtected.test_provider_test` expected hash in `tests/test_ds4_segmented_smoke.py`.
4. Force-add architecture-r2, all current r2/r3/r4 task/notes/review/test handoffs. Create and force-add placeholder paths `custom-handoffs/14-4-repo-hygiene/review-final.md` and `test-report-final.md` containing `PENDING FINAL GATE`; final agents will overwrite these same tracked paths.
5. Rebuild complete sorted commit/deletion manifests LAST; include all staged paths and both manifest files; exact counts.
6. Verify zero untracked/unstaged, no staged binaries, cached diff check, active-memory test with parent absent, exact 246-test suite, Path A 365/digest, provider source unchanged, source sentinel unchanged, new provider-test hash cascade exact, vendor remote fetch. Run synthetic staged clone exact suite; expected 246 passed.
7. Update coder-notes-r4.md. No production edits, no inner/outer commit, no push, no smoke/training/inference. Marker when complete.