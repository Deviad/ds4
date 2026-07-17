# Story 14.5 — Coder r5 lock/watchdog/terminal closure

## Scope

Direct TDD synthetic-only work from `task-coder-r5.md` and `custom-handoffs/standby/review.md`. No role dispatch, delegation, real model/dataset/adapter access, training, inference, CUDA, distributed execution, commit, or push.

## Changes

- Partial lock acquisition records ownership immediately after `O_EXCL`; fsync/write cleanup failure preserves recoverable owner state and a partial-lock flag when unlink fails. A retryable release removes the retained lock without clearing ownership early.
- `_release_ft_lock()` clears ownership only after successful unlink. Actual unlink failure preserves owned state for retry.
- Watchdog Event construction, thread construction/start, and Watchdog construction are inside post-alarm rollback; every setup failure disarms SIGALRM.
- Replaced shared `_execute_training` terminal substitutions with executable run-through injections for config, preflight, start-save, provider, callback, checkpoint validation, timeout delivery, and actual release unlink. Release injection counts exactly once.
- Added end-to-end Phase B success/failure/final-aggregation and phase-report/final-report/phase-marker/final-marker mutation cases.
- Replaced hard-coded-prefix filesystem checking with a resolved `tmp_path` containment trap covering `os.open/unlink/replace/rename` and `Path.open/read/write/unlink/replace/rename/resolve` source and destination paths.

## TDD evidence

RED first:

```text
3 failed: Event construction rollback; partial-acquire unlink preservation; release-unlink preservation
```

Focused GREEN:

```text
PYTHONPATH=. uv run --with pytest pytest -q tests/test_ds4_segmented_pilot.py
69 passed
```

Exact six-file synthetic/protected suite (real trainer tests excluded because installed `mlx_lm.tuner.trainer` lacks `TrainUI`):

```text
307 passed, 2 deselected, 2 warnings, 5 subtests passed
```

The unfiltered six-file attempt produced the same 307 passes plus two unrelated environment failures in `test_real_trainer_integration` and `test_real_trainer_failure_after_prior_accumulation` while patching absent `mlx_lm.tuner.trainer.TrainUI`.

Additional gates:

- `python3 -m py_compile scripts/ds4_segmented_pilot.py tests/test_ds4_segmented_pilot.py`: PASS.
- `git diff --check` and `git diff --cached --check`: PASS.
- Protected provider SHA-256: `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518`.
- Protected smoke SHA-256: `ec17950d985578f3cd97b51734527bfc47e6c399ebe84fad8ffcb12beae18ad8`.
- Vendor inner HEAD and outer gitlink: `80fab4e419a57f9465bb9e2f4e90010d645e124c`; vendor clean.
- No `.cmux-status` marker staged.

## Boundary

Synthetic only. No real model, dataset, adapter, training, inference, CUDA, distributed, or network execution performed. Independent review and test-manager gates remain external requirements.
