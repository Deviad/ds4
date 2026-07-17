# Story 14.5a — Test Manager r3b report

## Verdict
GREEN for declared synthetic-only gate evidence.

## Verified
- `custom-handoffs/14-5a-resource-gate-repin/task-tester-r3b.md` read in full.
- `custom-handoffs/14-5a-resource-gate-repin/task-tester-r3.md`, `coder-notes-r3.md`, `test-report-r3.md` read.
- Declared interpreter used explicitly:
  - `python-envs/mlx/.venv/bin/python -m pytest -q tests/test_ds4_segmented_pilot.py`
  - `python-envs/mlx/.venv/bin/python -m pytest -q tests/test_ds4_segmented_pilot.py tests/test_ds4_segmented_smoke.py tests/test_finetune_ds4.py tests/test_mlx_lm_source.py tests/test_ds4_segmented_loss_and_grad.py tests/test_ds4_gguf_base_smoke.py`
- Tracked verdict-contributing tests:
  - `tests/test_ds4_segmented_pilot.py`
  - `tests/test_ds4_segmented_smoke.py`
  - `tests/test_finetune_ds4.py`
  - `tests/test_mlx_lm_source.py`
  - `tests/test_ds4_segmented_loss_and_grad.py`
  - `tests/test_ds4_gguf_base_smoke.py`
- `git diff --check`: PASS.
- Direct protected-source hash check:
  - `ds4.c` MATCH
  - `ds4_cli.c` MATCH
  - `ds4_server.c` MATCH
  - `ds4_metal.m` MATCH
  - `metal/moe.metal` MATCH
  - `scripts/ds4_segmented_pilot.py` differs from `HEAD` as expected for this slice.

## Test results
- Full focused pilot suite: `141 passed, 1 warning`
- Exact six-file synthetic suite: `387 passed, 3 skipped, 1 warning, 2 subtests passed in 22.34s`

## Acceptance
- Synthetic-only A2 admission closure evidence present.
- All required six verdict tests tracked.
- No real model, dataset, adapter, training, inference, cleanup, commit, or push performed.
