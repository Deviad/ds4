# Coder Notes — Story 13.3b-2b CSA attention wiring

## Scope done

- Added additive helpers in `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`:
  - `_csa_compressor_real_mlx`
  - `_csa_block_bias_mlx`
  - `_csa_attention_real_mlx`
- No dispatch branch added in `_attention_mlx`.
- `_attention_real_mlx` HCA body untouched; CSA ships as sibling helper for 13.3b-3 wiring.
- Added tests:
  - `tests/test_csa_compressor_real_mlx_parity.py`
  - `tests/test_csa_attention_real_mlx_parity.py`
- Advanced `deepseek_v4.py` sha16 pins:
  - `812df0f7a34c0f07` → `dc5aaaab9bb079d2`
  - pin sites updated in `tests/test_numpy_real_forward_reference_composition.py`, `tests/test_deepseek_v4_real_config_reference_forward.py`, `python-envs/mlx/src/ds4_ft_mlx/real_forward_intermediate_dump.py`

## TDD red-first

Initial new-test run before implementation:

```text
pytest -q tests/test_csa_compressor_real_mlx_parity.py tests/test_csa_attention_real_mlx_parity.py
2 failed
ImportError: cannot import name '_csa_compressor_real_mlx'
ImportError: cannot import name '_csa_attention_real_mlx'
```

## Rope + APE handling

- Confirmed FROZEN `_csa_windowed_compressor_mlx` bakes wrong real-path behavior:
  - expects transposed APE shape `(2*out_dim, rate)`
  - uses `_rope_full_tables_mlx` + `_apply_rope_full_mlx`
  - full-channel plain-theta rope
- Implemented option α instead:
  - inline Ca/Cb overlap pooling
  - APE consumed token-major `(rate=4, 2*head_dim=1024)` with no transpose
  - compress-YaRN tail RoPE via `_compress_rope_yarn_tail_tables_mlx` + `_apply_rope_tail_mlx`
  - positions `arange(n_win) * 4`

## Parity deltas

```text
compressor_max_abs 5.9604644775390625e-06
attention_max_abs 6.48200511932373e-07
attention_mean_abs 4.686803833919839e-08
attention_compressor_max_abs 6.556510925292969e-06
block_bias_neginf_equal True
block_bias_finite_max_abs 0.0
```

## Validation

```text
pytest -q tests/test_csa_compressor_real_mlx_parity.py tests/test_csa_attention_real_mlx_parity.py
2 passed
```

Core regression:

```text
PYTHONPATH=.:python-envs/mlx/src pytest -q \
  tests/test_compress_rope_oracle.py \
  tests/test_hca_compressor_mlx_parity.py \
  tests/test_attention_real_mlx_hca_parity.py \
  tests/test_13_3b_1_dispatch_additive.py \
  tests/test_indexer_scorer_mlx_parity.py \
  tests/test_indexer_mlx_parity.py \
  tests/test_csa_compressor_real_mlx_parity.py \
  tests/test_csa_attention_real_mlx_parity.py \
  tests/test_deepseek_v4_nn_*.py \
  tests/test_deepseek_v4_fp4_dequant_*.py
29 passed, 1 skipped
```

Tiny CSA regression selection:

```text
15 passed, 1 subtests passed
```

Full suite:

```text
PYTHONPATH=.:python-envs/mlx/src python -m pytest tests/ --no-header -q
565 passed, 13 skipped, 1 failed
```

Full-suite failure:

```text
tests/test_deepseek_v4_mlx_port.py::DeepSeekV4MlxPortTests::test_finetune_readiness_report_records_b0b_a_without_unblocking_b0
AssertionError: 9 != 8
```

Pre-existing stale assertion evidence:

```text
scripts/finetune_ds4.py B0_REAL_MODE_PROOF_SPECS ids = 9
untracked tests/test_deepseek_v4_mlx_port.py line 1104 still asserts proofs_total == 8
```

No CSA/HCA/indexer/nn/FP4/tiny regression introduced.

## Frozen-body proof

Compared current `deepseek_v4.py` to `HEAD:a2c20b0` by source hash + AST for protected symbols.

All OK:

```text
_csa_windowed_compressor_mlx
_attention_real_mlx
_hca_compressor_mlx
_indexer_mlx
_indexer_scorer_mlx
_compress_rope_yarn_tail_tables_mlx
_apply_rope_tail_mlx
_apply_rope_full_mlx
_rope_full_tables_mlx
_attention_mlx
_csa_attention_mlx
_csa_compressor_mlx
_csa_indexer_mlx
_csa_config_error
_require_csa_config
_causal_sliding_mask_mlx
_hyperconnection_mlx
_hyperhead_mlx
sanitize_weights
Model
```

Additional proof:

```text
new_symbols ['_csa_compressor_real_mlx', '_csa_block_bias_mlx', '_csa_attention_real_mlx']
attention_mlx_has_csa_real_dispatch False
file_sha16 dc5aaaab9bb079d2
```

`git diff --check` clean.

## Blast-radius surprise

- Full suite's one RED is not the expected live-logit Test #8 carve-out in this checkout.
- It is stale proof-count assertion `8` vs builder `9` in untracked `tests/test_deepseek_v4_mlx_port.py`.
- Left untouched as unrelated/out-of-scope.
