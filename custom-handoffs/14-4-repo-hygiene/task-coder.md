# Story 14.4 — Coder implementation: reproducible repo + stale cleanup

Read requirements.md and architecture.md completely. Execute serially. User authorized commits/deletions; no push.

Hard boundaries:
- Preserve all 365 tracked files under `agent-output/cmux-13-3b/` byte-for-byte and preserve digest `7241924d...`. Do not remove/modify Path A evidence.
- Preserve Story 14.3 smoke log/report and successful adapter files on NVME; do not rerun smoke/training/inference.
- Do not delete ambiguous `context.md` or `adapter-converter-implementation-gpt55.md`; quarantine via gitignore.
- Do not touch unrelated user-owned files unless classified stale by requirements/architecture and recorded in deletion manifest.

Implementation:
1. Commit inner `vendor/mlx-lm` staged trainer seam + tests with a clear Story 14.1/14.2a message. Capture new inner SHA. Update outer gitlink and every active pin/reference/test/ADR expectation from `15b522f...` to new SHA, including protected source-sentinel cascade. Do not push.
2. Track every legitimate untracked implementation dependency: all required `python-envs/mlx/src` package modules; canonical `python-envs/torch` definition/package; MLX lockfile if valid; four production scripts; every source/fixture/helper test participating in baseline; ADRs 0001-0023 + README; architecture dossier/MTP policy; project `.pi/agents` scaffolding excluding personal pipeline.conf. Inspect `python-envs/legacy-trans` and quarantine/delete if obsolete.
3. Stage legitimate tracked modifications `.gitignore`, `AGENTS.md`, `docs/technical-spec.md`, `agent-output/cmux-13-1/coder-notes.md` only if verified intentional.
4. Remove tracked `.cmux-status/coder.done` from index and delete runtime markers. Add precise gitignore rules for `.cmux-status/`, runtime epochs/PID/RC/cache/log files, `agent-output/`, `custom-handoffs/`, personal pipeline.conf, compiled test binaries, and quarantined scratch names. Existing tracked Path A evidence remains tracked despite ignore.
5. Delete physically all demonstrably stale untracked agent-output/custom-handoffs/runtime files and obsolete root review/plan scratch artifacts, except current Story 14.4 requirements/architecture/task/coder notes/manifests which must be force-added. Preserve ambiguous quarantined files above.
6. Inspect `tests/ds4_lora_test`, `tests/test_q4k_dot`, and `ds4_agent_test`; delete if compiled/generated, otherwise track only if source and required.
7. Create force-tracked `agent-output/cmux-14-4/commit-manifest.txt` and `deletion-manifest.txt` with exact paths/rules/counts. Write coder-notes.md.
8. Ensure no untracked imported source or verdict-contributing test remains. Run tracked-file checks, py_compile, exact-fork Epic 14 suite, broad relevant pytest suite, git diff --check, Path A digest, vendor inner clean + outer gitlink equals inner HEAD, and simulated fresh-clone/submodule/import check. No real model/assets.
9. Stage outer changes but DO NOT outer-commit; parent commits only after Reviewer PASS and Tester GREEN. Inner vendor commit is authorized/required.

If any file cannot be safely classified, leave it quarantined/ignored and report it rather than delete. Marker only when complete.