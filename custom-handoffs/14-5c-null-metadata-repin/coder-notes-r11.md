# Story 14.5c — Coder r11 evidence closure

## Scope

Closed standby review r10 findings with direct tracked tests. No real model/data/provider/training/inference/cleanup execution. No nested agents, cmux, commit, or push. No `.cmux-status` marker created.

## Changes

- `tests/test_ds4_segmented_pilot.py`
  - Added loaded-MLX negative matrix through `mx.load()` and strict model schema loading for missing/extra/duplicate names, dtype, shape, layout, and payload mutations; exact names/dtypes/shapes/values remain required.
  - Added independent wrong-JSON-type rejection for every A3/B3/final marker field, including all Phase-A resume bindings.
  - Added caller-level `run_phase()` publication seam injection for both Phase-A writes and all four Phase-B/final writes; each case asserts exact phase/attempt failure, bound fail-marker report path/hash, and no surviving OK markers.
  - Added coordinated attempt-2 mutation instrumentation proving substituted target path/hash reaches the lower-level verifier before rejection.
  - Added weakened-verifier mutant proving the coordinated substitution is accepted without the immutable target guard and rejected with it.
- `scripts/ds4_segmented_pilot.py`
  - Added an independent immutable attempt-2 verifier target tuple and guard against coordinated path/hash/target substitution.
  - Added optional immutable target binding to the lower-level historical snapshot helper for mutation-sensitive seam tests.
- Updated canonical `docs/architecture.md`, `docs/backlog.md`, and `docs/technical-spec.md` from r10 staging wording to r11 coder evidence wording without authorizing A3/B3.

## Test-first evidence

- RED: coordinated attempt-2 substitution test initially did not raise; the weakened verifier accepted coherent path/manifest/hash substitution.
- GREEN after minimal production guard: focused r11 tests pass.

## Verification

Command:

```bash
PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" /Volumes/Data\ NVME/mlx-ft/ds4/.venv/bin/python -m pytest -q tests/test_ds4_segmented_pilot.py -k 'r11_'
```

Result: `9 passed, 616 deselected, 1 warning`.

- Vendor-first canonical six-file suite: `871 passed, 3 skipped, 1 warning, 2 subtests passed in 121.10s`.
- `python3 -m py_compile tests/test_ds4_segmented_pilot.py scripts/ds4_segmented_pilot.py scripts/finetune_ds4.py`: PASS.
- `git diff --check`: PASS.
- Verdict-contributing test and production files are tracked after staging; no test baseline file remains untracked.

## Boundary

This is coder evidence only. Reviewer PASS and Test Manager GREEN must run on this unchanged staged tree before any fresh phase authorization. No A3/B3 invocation is authorized.
