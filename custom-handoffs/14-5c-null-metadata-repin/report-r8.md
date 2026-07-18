GREEN

Exact canonical six-file suite:
- tests/test_ds4_segmented_pilot.py
- tests/test_ds4_segmented_smoke.py
- tests/test_finetune_ds4.py
- tests/test_mlx_lm_source.py
- tests/test_ds4_segmented_loss_and_grad.py
- tests/test_ds4_gguf_base_smoke.py

Results:
- pytest: 638 passed, 3 skipped, 2 subtests passed
- py_compile: PASS
- git diff --check: PASS
- git ls-files tracked check for all six files: PASS

Scope:
- focused tests only
- no unrelated full tests
- no markers
- no real model/data/training/inference
