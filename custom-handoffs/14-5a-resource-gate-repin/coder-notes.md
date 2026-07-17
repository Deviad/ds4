# Story 14.5a — Coder notes

## Scope

Synthetic-only resource observer repair and immutable attempt-2 namespace. No real model, dataset, adapter, training, inference, cleanup, commit, or push.

## TDD evidence

- Added tracked RED tests first for allowed psutil races, strict resource boundaries, current-PID exclusion, unknown observer errors, exact attempt-2 paths, collision immutability, and non-default catalog logging.
- Implemented `_resource_gate()` as a per-process fail-closed scan.
- Added structured `ResourceObserverError` evidence with stage, exception type, and PID.
- Added exact attempt-1 historical five-file size/SHA manifest verification.
- Added centralized attempt-2 namespace/specs, non-mutating launch checks, A2/B2 dependency separation, active log FD attestation, and no-write collision handling.
- Added non-default attempt-2 catalog wrappers using launch-check, `noclobber` FD creation, `--log-fd`, and `tee /dev/fd/3`.
- Preserved Story 14.5 attempt-1 constants/catalog command bytes and default backend selection.
- Updated canonical architecture, technical specification, and backlog status.

## Validation

- `python-envs/mlx/.venv/bin/python -m pytest -q tests/test_ds4_segmented_pilot.py -k 'resource_gate or attempt2'`: 8 passed.
- `python-envs/mlx/.venv/bin/python -m pytest -q tests/test_ds4_segmented_pilot.py`: 77 passed.
- `python-envs/mlx/.venv/bin/python -m pytest -q tests/test_finetune_ds4.py`: 51 passed.
- `python3 -m py_compile scripts/ds4_segmented_pilot.py scripts/finetune_ds4.py tests/test_ds4_segmented_pilot.py`: passed.
- `git diff --check`: passed.
- No real `/Volumes/Data NVME/...` asset access or training/inference execution performed.

## Gate status

Coder implementation complete. Independent Reviewer PASS and Test Manager GREEN remain pending. Fresh explicit operator authorization remains required before real Phase A2; separate authorization remains required before Phase B2.
