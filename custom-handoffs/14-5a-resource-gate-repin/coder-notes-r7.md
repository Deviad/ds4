# Story 14.5a — Coder r7

## Scope closed

- Rebuilt `_pilot_attempt2_command()` as one complete inner shell script followed by exactly one `shlex.quote()` around the `bash -lc` script argument.
- Kept all central attempt-2 namespace, launch identity, phase, FD-attested `noclobber`, and `PIPESTATUS` bindings intact.
- Added temp-only executable A2/B2 wrapper tests with paths containing spaces. Tests assert exactly three top-level argv entries, launch-check then training order, FD-backed log output, and training exit-status propagation without false success.
- Added a valid synthetic B2 canonical report/publication case and fail-closed mutation coverage for every B2 namespace binding.
- Expanded A2/B2 artifact, dependency, phase-spec, canonical, and catalog checkpoint/config binding matrices.

## TDD evidence

Red first: the new executable wrapper tests failed against r6 because the emitted outer single-quoted `bash -lc` command was split by nested `shlex.quote()` values and parsed into more than three argv entries.

Green:

```text
224 passed, 1 warning in tests/test_ds4_segmented_pilot.py
470 passed, 3 skipped, 1 warning, 2 subtests passed in canonical six-file suite
```

Canonical command:

```bash
PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" PYTHONDONTWRITEBYTECODE=1 python-envs/mlx/.venv/bin/python -m pytest -q tests/test_ds4_segmented_pilot.py tests/test_ds4_segmented_smoke.py tests/test_finetune_ds4.py tests/test_mlx_lm_source.py tests/test_ds4_segmented_loss_and_grad.py tests/test_ds4_gguf_base_smoke.py
```

Additional checks:

- `py_compile` passed for `scripts/finetune_ds4.py`, `scripts/ds4_segmented_pilot.py`, and `tests/test_ds4_segmented_pilot.py`.
- Generated A2 and B2 catalog wrappers each parsed as exactly `bash`, `-lc`, one script and passed `bash -n`.
- `git diff --check` and `git diff --cached --check` passed.
- Protected runtime, Metal, segmented provider, and segmented smoke files matched `HEAD` byte-for-byte.
- No real model, dataset, training, inference, cleanup, commit, push, or marker staging performed.
