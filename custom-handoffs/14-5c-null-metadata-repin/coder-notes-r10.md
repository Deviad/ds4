# Story 14.5c — Coder r10 mutation-sensitivity closure

## Scope

- Removed `_pilot_attempt3_command` zero/minted authorization fallback; explicit A3 authorization is required and B3 requires separate retained A3 authorization.
- Generic catalog authorization exposes A3 only. Phase-specific A3/B3 values remain distinct.
- Added stale central-module fixture protection without changing production namespace ownership.
- Added RED-first mutation coverage for final authorization fields, identity, attempt-1/attempt-2 history roots, admission-lineage roots, and recomputed phase/final contracts and hashes.
- Added attempt-2 outer pre-log bytes, manifest semantic bytes, safetensors missing/extra/duplicate/schema/layout/overlap/out-of-bounds matrix, every invalid metadata JSON type at every A3/B3 checkpoint, marker valid-type wrong-value/resume matrix, and caller-level attempt-3 write-seam rollback coverage.
- Updated canonical architecture, backlog, and technical specification to r10. No real A3/B3 access, model/data/training/inference, cleanup, commit, push, delegate, cmux operation, or staged `.cmux-status` marker.

## Verification

- Vendor-first canonical six-file suite: `862 passed, 3 skipped, 1 warning, 2 subtests passed`.
- `PYTHONPATH=$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD uv run --with pytest pytest -q` canonical six-file command: PASS.
- `python3 -m py_compile tests/test_ds4_segmented_pilot.py scripts/ds4_segmented_pilot.py scripts/finetune_ds4.py`: PASS.
- `git diff --check`: PASS.
- All canonical test files and production files cited above are tracked.
- No real model/data/training/inference executed.

## Boundary

Synthetic fixtures only. A3/B3 remains blocked pending independent Reviewer PASS, Test Manager GREEN, and fresh explicit phase authorization. No commit or push performed.
