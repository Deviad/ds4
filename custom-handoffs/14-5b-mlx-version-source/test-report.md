# Story 14.5b — Test Manager report

Status: GREEN
Same-revision binding: staged diff hash `f8fd02c895e651e20b13a28b9115f97c3db2d9358329845710d505882c13b79c`.

## Commands run

1. `PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" PYTHONDONTWRITEBYTECODE=1 python-envs/mlx/.venv/bin/python -m pytest -q tests/test_ds4_segmented_pilot.py -k 'mlx_distribution_metadata or launch_check or attempt2_catalog'`
   - Result: `37 passed, 251 deselected, 1 warning in 15.09s`

2. `PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" PYTHONDONTWRITEBYTECODE=1 python-envs/mlx/.venv/bin/python -m pytest -q tests/test_ds4_segmented_pilot.py tests/test_ds4_segmented_smoke.py tests/test_finetune_ds4.py tests/test_mlx_lm_source.py tests/test_ds4_segmented_loss_and_grad.py tests/test_ds4_gguf_base_smoke.py`
   - Result: `534 passed, 3 skipped, 1 warning, 2 subtests passed in 102.65s`

3. `PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" PYTHONDONTWRITEBYTECODE=1 python-envs/mlx/.venv/bin/python -m py_compile scripts/ds4_segmented_pilot.py tests/test_ds4_segmented_pilot.py scripts/finetune_ds4.py scripts/ds4_segmented_smoke.py python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py`
   - Result: no output

4. `git diff --check`
   - Result: no output

5. `git ls-files --error-unmatch -- tests/test_ds4_segmented_pilot.py tests/test_ds4_segmented_smoke.py tests/test_finetune_ds4.py tests/test_mlx_lm_source.py tests/test_ds4_segmented_loss_and_grad.py tests/test_ds4_gguf_base_smoke.py`
   - Result: all six files tracked

6. SHA-256 checks for protected bytes
   - `scripts/finetune_ds4.py`: `de303d1166f2ecd2428dc15f63c8be8ed22e9f30274008afd35be458ef4b188b`
   - `scripts/ds4_segmented_smoke.py`: `ec17950d985578f3cd97b51734527bfc47e6c399ebe84fad8ffcb12beae18ad8`
   - `python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py`: `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518`

## Verdict

GREEN.

No real namespace paths touched. No production wrapper mutation observed. Protected hashes unchanged.
