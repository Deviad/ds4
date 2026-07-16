# Story 14.4 — Coder r2 reviewer/tester fixes

Read custom-handoffs/standby/review.md and test-report.md fully. Close all local blockers. User's original request explicitly authorizes deletion of files classified stale; no push.

Exact fixes:
1. Remove `tests/ds4_lora_test` from index and physically delete it. Physically delete ignored generated binaries `tests/test_q4k_dot` and root `ds4_agent_test`; add `/ds4_agent_test` ignore rule.
2. Track required canonical files: `python-envs/torch/pyproject.toml`, `python-envs/torch/src/ds4_ft_torch/__init__.py`, `progress.md`, `training-next-status.md`, `training-backend-bakeoff-gpt55.md`.
3. Adjudication: track `python-envs/mlx/uv.lock` as reproducibility input. Delete obsolete unreferenced `python-envs/legacy-trans/pyproject.toml` and empty parents if possible; record.
4. Delete the 111 root-level historical review/plan/scout/status-support Markdown scratch files classified by Reviewer B3, excluding the five required tracked docs above and excluding ambiguous `context.md` + `adapter-converter-implementation-gpt55.md`. Those two remain physically present and narrowly ignored/quarantined.
5. Fix trailing whitespace in ADR 0018/0019; `git diff --cached --check` must return 0.
6. Force-add current Story 14.4 chain of custody despite ignore: requirements.md, architecture.md, task files, coder notes, Reviewer review copied or preserved at a canonical 14-4 path, Tester test-report.md, and final gate task files. Do not retain transient dispatch epochs/pipeline-private state.
7. Update Story 14.4 backlog to truthful implementation state: local cleanup staged, inner commit 80fab4e, outer commit pending gates, remote reachability pending operator authorization; exact counts; Path A 174 stale-looking tracked evidence files preserved under permanent STOP.
8. Correct coder notes: one inner commit occurred; no outer commit/no push. Rebuild sorted complete commit-manifest including itself/deletion manifest and exact modifications/additions/deletions; reconcile 21 vs 22.
9. Re-scan staged additions for NUL/binary/generated artifacts; none allowed except intentional textual fixtures.
10. Verify all imported local modules and verdict tests tracked. Run py_compile, exact 246-test suite, Path A count/digest, smoke evidence unchanged, vendor inner clean/gitlink match, cached diff check.
11. For fresh-clone check, use a temporary synthetic outer commit/ref containing current index, without moving production HEAD, to distinguish outer-index omission from vendor remote reachability. Record that standard remote submodule fetch remains BLOCKED until user authorizes publishing inner 80fab4e. Do not push.

Stage all outer changes; no outer commit. Marker when complete.