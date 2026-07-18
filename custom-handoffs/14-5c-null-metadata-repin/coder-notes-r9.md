# Story 14.5c — Coder r9 exhaustive-oracle + staging closure

## Scope

- Added tracked synthetic mutation coverage for the byte-authentic lower-level attempt-2 verifier: all semantic/nested manifest fields, every pre-log and manifest descriptor field, all ten runtime descriptor path/size/hash rows, all ten runtime target-byte rows, all eight absence facts, duplicate-key JSON, and coordinated manifest/path/hash/target substitution.
- Added final trusted-root single and combination substitutions for Phase A3/B3 authorization, identity, both histories, and Phase A admission lineage.
- Added absent/null/object metadata coverage at every A3/B3 safetensors artifact position and invalid-metadata publication guards.
- Added complete Phase A3/B3/final marker missing/extra/type/value/resume matrices and synthetic A3/B3 success-publication write-seam rollback checks.
- Updated canonical architecture, backlog, and technical-spec status to R9 without authorizing real execution.

## Verification

- `PYTHONPATH=/Users/spotted/projects/ds4-finetuning uv run --with pytest pytest -q tests/test_ds4_segmented_pilot.py -k 'r9_'`: 121 passed.
- `PYTHONPATH=/Users/spotted/projects/ds4-finetuning uv run --with pytest pytest -q tests/test_ds4_segmented_pilot.py`: 513 passed.
- `python3 -m py_compile tests/test_ds4_segmented_pilot.py scripts/ds4_segmented_pilot.py`: PASS.
- `git diff --check`: PASS.
- Canonical six-file suite: 757 passed, 3 skipped, 2 subtests passed, 2 pre-existing environment failures in `tests/test_ds4_segmented_loss_and_grad.py` because installed `mlx_lm.tuner.trainer` lacks `TrainUI`; no r9 test failed.
- All six canonical test files are tracked.

## Boundary

- Synthetic temporary fixtures only. No real model, data, adapter, training, inference, CUDA, distributed execution, cleanup, commit, push, delegate, cmux operation, or staged `.cmux-status` marker.
