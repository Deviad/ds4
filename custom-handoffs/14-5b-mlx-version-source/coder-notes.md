# Story 14.5b — Coder notes

## Implementation

- Added module-scope `import importlib.metadata` in `scripts/ds4_segmented_pilot.py`.
- `_runtime_preflight()` now requires `importlib.metadata.version("mlx")` to return exact string `0.31.2`.
- `mlx` and `mlx_lm` imports remain mandatory; `mlx.__version__` is not read.
- Added explicit fail-closed diagnostics for missing distribution metadata, metadata exceptions, malformed values, and version mismatch. Accepted metadata value populates immutable `mlx_version`.
- Added tracked production-boundary tests for absent/`None`/spoofed module attributes, mismatch/suffix/missing/error/malformed metadata, mandatory module import, direct launch-check no-write/no-call behavior, and the generated attempt-2 wrapper boundary.
- `docs/technical-spec.md` already contains the Story 14.5b canonical contract from the handoff; no additional documentation mutation was needed.

## TDD evidence

- RED: new mutation-sensitive tests failed against the prior `mlx.__version__` implementation.
- GREEN: targeted matrix passed: `26 passed`.
- Canonical regression passed: `534 passed, 3 skipped, 2 subtests passed`.
- `py_compile`, `git diff --check`, tracked-test check, and protected-byte hash checks passed.
- Protected hashes remained unchanged for `scripts/finetune_ds4.py`, `scripts/ds4_segmented_smoke.py`, and `python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py`.

No real assets, provider calls, training, inference, namespace mutation, cleanup, commit, or push performed.
