# Story 14.3 Test Manager r10 exact-fork recheck

Rerun final r10 validation after protected index restoration. Functional hash: `d9b4f9e60ae4158423f41dd09de62611355427525c022973faefef077ae2e584`.

Before tests, assert Path A tracked-path digest exactly `7241924d6f9be2eb0df974fdcb1717728c71b6ee77d41dea3fe8a8af55ebb2af`, 365 rows, `.pyc` exact; vendor gitlink/inner HEAD `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`; source/provider/ADR hashes and `mlx-lm==0.31.3`.

Run protected suites with exact fork routing, not site-packages:
`PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" PYTHONDONTWRITEBYTECODE=1 python-envs/mlx/.venv/bin/python -m pytest -q tests/test_ds4_segmented_smoke.py tests/test_finetune_ds4.py tests/test_mlx_lm_source.py tests/test_ds4_segmented_loss_and_grad.py`
Also run focused smoke 88+, finetune 52, py_compile, functional hash and registry checks. No real assets/Phase 2/commit/push. Write only test-report.md and marker. GREEN only if reproducible.