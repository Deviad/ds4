# Story 14.5c Test Manager r7

## Verdict
BLOCKED.

## Direct validation
- Focused r7 slice:
  - `PYTHONPATH=$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD python-envs/mlx/.venv/bin/python -m pytest -q tests/test_ds4_segmented_pilot.py -k 'r7_'`
  - `12 passed, 378 deselected, 1 warning`
- Pilot regression:
  - `PYTHONPATH=$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD python-envs/mlx/.venv/bin/python -m pytest -q tests/test_ds4_segmented_pilot.py`
  - `390 passed, 1 warning`
- Structural checks:
  - `PYTHONPATH=$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD python-envs/mlx/.venv/bin/python -m compileall scripts/ds4_segmented_pilot.py scripts/finetune_ds4.py tests/test_ds4_segmented_pilot.py`
  - passed
  - `git diff --check`
  - passed
  - `git diff --cached --check`
  - passed
  - `git ls-files custom-handoffs/14-5c-null-metadata-repin/coder-notes-r2.md custom-handoffs/14-5c-null-metadata-repin/coder-notes-r3.md custom-handoffs/14-5c-null-metadata-repin/coder-notes-r5.md custom-handoffs/14-5c-null-metadata-repin/coder-notes-r6.md custom-handoffs/14-5c-null-metadata-repin/coder-notes-r7.md custom-handoffs/14-5c-null-metadata-repin/coder-notes.md docs/architecture.md docs/backlog.md docs/technical-spec.md scripts/ds4_segmented_pilot.py scripts/finetune_ds4.py tests/test_ds4_segmented_pilot.py`
  - all tracked
  - `git diff --cached --name-only | grep -E '(^|/)\\.cmux-status/'`
  - no staged markers

## Broader repo sweep
- `PYTHONPATH=$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD python-envs/mlx/.venv/bin/python -m pytest -q tests`
- `6 failed, 1217 passed, 18 skipped, 96 subtests passed in 302.58s`
- blockers:
  - `tests/helpers/routed_fp4_multilayer_peak_probe.py:1287` hash mismatch for `docs/technical-spec.md`
  - `tests/test_deepseek_v4_real_config_reference_forward.py:423` ADR drift for `docs/adr/0019-fusion-primary-adapter-serving.md`

## Conclusion
- r7 slice gates green.
- repo-wide regression not green.
- exact canonical end-state still blocked by protected-doc drift / hash mismatch.
