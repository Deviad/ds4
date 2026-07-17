# Story 14.5a — Coder r5

## Scope closed

Implemented direct pre-log trusted-identity admission and strict canonical evidence validation.

### Pre-log B2 identity

- `--launch-check-only` now materializes the exact attempt-2 pinned argument set and validates it through `validate_pins()`.
- It performs read-only `_runtime_preflight()` before `check_attempt2_launch()` and before any B2 log/output/evidence mutation.
- Current immutable identity covers exact interpreter/runtime, MLX-LM/vendor/source/provider/smoke identity, model/data/config/provenance manifests and hashes, provenance split hashes, LoRA pins, and current git HEAD.
- B2 dependency validation receives that trusted current identity and canonical report validation compares report-owned immutable identity against it.
- Added executable CLI-path filesystem-trap coverage: coordinated current `git_head` substitution fails before B2 log/output/final-report creation.

### Strict provider/update schema

- Added finite numeric validation excluding `bool` for loss, elapsed, learning rate, throughput, wall time, and validation values.
- Added strict positive ordinal pairing for provider call, local/global step, optimizer update, and checkpoint records.
- Added gradient path syntax, positive integer shape, and dtype allowlist validation, including duplicate map/schema self-pairing checks.
- Added mutation coverage for coordinated ordinal swaps, bool/nonnumeric values, malformed paths/shapes/dtypes.

### Central catalog namespace

- Catalog attempt-2 wrappers now load `attempt2_namespace()` and `attempt2_phase_specs()` from `scripts/ds4_segmented_pilot.py` rather than reconstructing output/log/resume/config/checkpoint paths.
- Wrapper binds and records the canonical namespace paths before invoking the training command while preserving exclusive FD-attested `noclobber` logging.
- Added mutation coverage proving every namespace binding reaches catalog output.

### Attempt-1 oracle/tracking

- Added tracked stable attempt-1 effective-pin and contract digest/byte oracle for both phases.
- `coder-notes-r3.md`, `task-coder-r3.md`, `coder-notes-r4.md`, `task-coder-r4.md`, `task-coder-r5.md`, and this handoff are force-tracked in the slice staging set.

## TDD evidence

Red first: new provider/schema, pre-log identity, catalog binding, and attempt-1 oracle tests failed against r4.

Green:

```text
406 passed, 3 skipped, 1 warning, 2 subtests passed in 23.14s
```

Canonical command:

```bash
PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" PYTHONDONTWRITEBYTECODE=1 python-envs/mlx/.venv/bin/python -m pytest -q tests/test_ds4_segmented_pilot.py tests/test_ds4_segmented_smoke.py tests/test_finetune_ds4.py tests/test_mlx_lm_source.py tests/test_ds4_segmented_loss_and_grad.py tests/test_ds4_gguf_base_smoke.py
```

Additional checks:

- `python3 -m py_compile scripts/ds4_segmented_pilot.py scripts/finetune_ds4.py tests/test_ds4_segmented_pilot.py`: pass.
- Generated Phase B catalog wrapper `bash -n`: pass.
- No real model, dataset, training, inference, cleanup, commit, or push performed.

## Gate status

Coder implementation complete. Independent Reviewer and Test Manager must adjudicate this revision before any authorization decision.
