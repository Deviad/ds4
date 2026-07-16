GREEN — final independent test report for Story 13.3b-5f

Scope
- No production edits.
- No model/shards.
- Verified staged docs match worktree and carry the r3 contract.

Docs / staging
- `cmp -s docs/backlog.md <(git show :docs/backlog.md)` -> yes.
- `cmp -s docs/technical-spec.md <(git show :docs/technical-spec.md)` -> yes.
- `cmp -s docs/adr/0028-opaque-packed-fp4-metal-training-primitive.md <(git show :docs/adr/0028-opaque-packed-fp4-metal-training-primitive.md)` -> yes.
- Staged docs present: `docs/backlog.md`, `docs/technical-spec.md`, `docs/adr/0028-opaque-packed-fp4-metal-training-primitive.md`.
- r3 contract hits confirmed in worktree/staged docs: combined `2e-6 + 1e-6*abs(ref)` forward bound, `simd_sum`, `0.0080s`, `0.0140s`, `1.25h`, `1.35h`.

Focused tests
- Command: `python-envs/mlx/.venv/bin/python -m pytest -q tests/test_deepseek_v4_nn_routed_fp4_metal.py tests/test_deepseek_v4_nn_sparse_routed_backward.py tests/test_deepseek_v4_nn_fp4_parity.py`
- Result: `49 passed, 1 warning in 26.41s`.

Tracked-only baseline
- Command: `python-envs/mlx/.venv/bin/python -m pytest -q $(git ls-files 'tests/test*.py')`
- Result: `476 passed, 15 skipped, 86 subtests passed in 52.48s`.
- Tracked verdict/helper files confirmed with `git ls-files`:
  - `tests/helpers/routed_fp4_bench.py`
  - `tests/helpers/routed_fp4_memory_probe.py`
  - `tests/test_deepseek_v4_nn_fp4_parity.py`
  - `tests/test_deepseek_v4_nn_routed_fp4_metal.py`
  - `tests/test_deepseek_v4_nn_sparse_routed_backward.py`

Parity / proxy metrics
- Primitive forward: max_abs `4.547473508864641e-13`, max_rel `2.7300740157443215e-07`, nrmse `1.2193259611494375e-07`.
- Primitive dx: max_abs `4.547473508864641e-13`, max_rel `3.410605131648481e-07`, nrmse `1.2766804593054315e-07`.
- Primitive a: max_abs `3.410605131648481e-13`, max_rel `2.4413850496785017e-07`, nrmse `8.176585443129183e-08`.
- Sparse forward: max_abs `1.0728836059570312e-06`, max_rel `5.77370030896418e-07`, nrmse `2.583439879954046e-07`.
- Proxy logits: max_abs `4.76837158203125e-07`, max_rel `9.383676342622493e-07`, nrmse `2.9635254251590007e-07`.
- Proxy top-1: identical; reference margin `0.000669487752020359`.
- Proxy gradients: input `max_abs 7.450580596923828e-09`, score `max_abs 2.384185791015625e-07`.

Memory / formula
- Fixed E=2: peak delta `79680160`, x finite `True`, score finite `True`.
- Fixed E=4: peak delta `83227408`, x finite `True`, score finite `True`.
- Fixed E=8: peak delta `90309428`, x finite `True`, score finite `True`.
- Spread: `10629268` bytes.
- Real-dim: peak delta `208720484`, x finite `True`, score finite `True`, direct finite `True`, shapes `[8,4096]/[8,4096]/[8]`.
- Formula: `assignment_y=402653184`, `assignment_i=201326592`, `a_partial=25165824`, `forward_total=1145044992`, `backward_total=1308622848`, `envelope=1577058304`.

Performance
- Command: `python-envs/mlx/.venv/bin/python tests/helpers/routed_fp4_bench.py --repeats 25`
- R=1: forward `0.001054s/0.001733s`, input-VJP `0.001693s/0.001966s`.
- R=8: forward `0.001049s/0.001350s`, input-VJP `0.001638s/0.001908s`.
- R=32: forward `0.002603s/0.003263s`, input-VJP `0.004441s/0.005142s`.
- R=96: forward `0.006544s/0.006951s`, input-VJP `0.011557s/0.012858s`.
- Extrapolation `256x43x20`: p50 `0.707h`, p95 `0.786h`; warmup `3`, repeats `25`.

Build / diff / hashes
- `make`: `Nothing to be done for all` (`MAKE_STATUS=0`).
- `git diff --check`: clean.
- `git diff --cached --check`: clean.
- Hashes: `ds4.c`, `ds4_metal.m`, `ds4_cuda.cu`, `ds4_distributed.c`, `ds4_ssd.c`, `deepseek_v4.py`, ADR 0024 all matched pinned SHA-256 values.

Verdict
- GREEN.