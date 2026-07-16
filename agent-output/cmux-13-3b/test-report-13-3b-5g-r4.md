# Story 13.3b-5g r4 — Tester independent validation

## Verdict
GREEN.

## Fresh execution
- mutation-sensitive mask oracle: rc=0, `1 passed, 1 warning in 2.38s`
- focused diagnostic suite: rc=0, `14 passed, 1 warning in 18.90s`
- order A compile alone: rc=0, `1 passed, 1 warning in 8.44s`
- order B diagnostic → compile: rc=0, `14 passed, 1 warning in 19.07s` then `1 passed, 1 warning in 8.40s`
- order C compile → diagnostic: rc=0, `1 passed, 1 warning in 8.37s` then `14 passed, 1 warning in 18.78s`
- tracked baseline in `python-envs/mlx/.venv`: rc=0, `tracked_test_files=45`, `490 passed, 15 skipped, 2 warnings, 86 subtests passed in 74.33s`
- `make`: rc=0, `Nothing to be done for 'all'.`
- `git diff --check`: rc=0
- `git diff --cached --check`: rc=0

## Canonical venv adjudication
- canonical project venv result: `490 passed, 15 skipped, 86 subtests passed`
- older `/Volumes/Data NVME/mlx-ft/ds4/.venv` result: `466 passed, 38 skipped, 79 subtests passed`
- delta: canonical env runs 24 more tests and skips 23 fewer
- cause: environment/package collection difference, not hidden test omission
- baseline command expanded the exact current `git ls-files 'tests/test_*.py'` set; no manual test omission

## Tracking and hashes
- helper worktree/index match: `git diff -- tests/helpers/routed_fp4_multilayer_peak_probe.py` empty
- helper still staged add from prior state: `A  tests/helpers/routed_fp4_multilayer_peak_probe.py`
- prior r3 report untouched during validation: `?? agent-output/cmux-13-3b/test-report-13-3b-5g-r3.md`
- index/worktree hashes recorded for helper, focused test, fixture, and prior report artifact
- protected hashes clean; no production drift
