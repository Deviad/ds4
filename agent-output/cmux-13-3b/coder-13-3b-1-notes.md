# Coder notes — Story 13.3b-1 HCA real-dim MLX port + RoPE oracle

## Scope delivered

- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`
  - Added `_compress_rope_yarn_tail_tables_mlx` for trailing `qk_rope_head_dim=64` compress RoPE with YaRN (`factor=16`, `original_max_position_embeddings=65536`, `theta=160000`) at explicit positions.
  - Added `_hca_compressor_mlx` for stateless HCA (`rate=128`, no Ca/Cb overlap, `ape` shape `(128,512)` no transpose), returning `compressed_kv [B,1,n_win,512]` and causal `block_bias [B,1,S,n_win]`.
  - Added `_attention_real_mlx` HCA branch: shared cr=0 projections/norms/sinks/grouped output, compress-RoPE on q/kv, HCA compressed-KV append, block-bias mask append, sink softmax, rope undo with `-sin`.
  - Added `_attention_mlx` dispatch branch: tiny `_csa_config_error(args) is None` routes unchanged to `_csa_attention_mlx`; real dims route to `_attention_real_mlx`.
- New tests:
  - `tests/test_compress_rope_oracle.py` (AC0)
  - `tests/test_hca_compressor_mlx_parity.py` (AC1)
  - `tests/test_attention_real_mlx_hca_parity.py` (AC2)
  - `tests/test_13_3b_1_dispatch_additive.py` (AC3)
- Updated stale whole-file vendor hash sentinels to the sanctioned ADR 0026 additive baseline `764162ca41531e52`:
  - `tests/test_deepseek_v4_real_config_reference_forward.py`
  - `tests/test_numpy_real_forward_reference_composition.py`
  - `python-envs/mlx/src/ds4_ft_mlx/real_forward_intermediate_dump.py`

## TDD sequence

- AC0 RED: `ImportError: _compress_rope_yarn_tail_tables_mlx`.
- AC0 GREEN: `tests/test_compress_rope_oracle.py` passed.
- AC1 RED: `ImportError: _hca_compressor_mlx`.
- AC1 GREEN: `tests/test_hca_compressor_mlx_parity.py` passed.
- AC2 RED: `ImportError: _attention_real_mlx`.
- AC2 GREEN: `tests/test_attention_real_mlx_hca_parity.py` passed.
- AC3 RED: real-dim `_attention_mlx` still raised tiny CSA `NotImplementedError`.
- AC3 GREEN: `tests/test_13_3b_1_dispatch_additive.py` passed.

## Parity deltas

- AC0: `cos_max_abs=7.15256e-07`, `sin_max_abs=1.86265e-06`, `rotated_max_abs=0.0078125` (within `1e-3` table / `1e-2` rotated gates).
- AC1: `compressed_kv_max_abs=3.91901e-06`; `block_bias` `-inf` masks equal.
- AC2: `output_max_abs=6.25849e-07`.

## Validation

- New AC suite: `4 passed`.
- FP4 13.2: `9 passed, 1 skipped`.
- 13.3a nn: `12 passed`.
- Tiny CSA 11.14/11.15 subset: `7 passed`.
- Full suite: `1 failed, 561 passed, 13 skipped`; sole failure is pre-existing `test_finetune_readiness_report_records_b0b_a_without_unblocking_b0` (`proofs_total 9 != 8`).
- `git diff --check`: clean.

## Frozen-body proof

Compared working tree against `HEAD` via AST source segments:

- Unchanged: `_csa_config_error`, `_require_csa_config`, `_csa_attention_mlx`, `_csa_compressor_mlx`, `_csa_indexer_mlx`, `_csa_windowed_compressor_mlx`, `_hyperconnection_mlx`, `_hyperhead_mlx`, `sanitize_weights`, `_apply_rope_full_mlx`, `_rope_tail_tables_mlx`, `_apply_rope_tail_mlx`.
- `_attention_mlx` docstring unchanged; cr=0 remainder after the sanctioned dispatch branch unchanged (`sha16=7c9960107ef0919d`).
- New symbols present: `_compress_rope_yarn_tail_tables_mlx`, `_hca_compressor_mlx`, `_attention_real_mlx`.

## Blast-radius notes

- No nn port edits.
- No convert/remap edits.
- No CSA/Indexer real-path implementation.
- No C++.
- Extra stale-hash sentinel updates were required for full-suite compatibility after ADR 0026 sanctioned an additive `deepseek_v4.py` baseline change.
