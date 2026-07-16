# Review — Story 13.3b-1 HCA real MLX path + RoPE oracle

Verdict: **PASS**

Commit: `1f5d8e4`  
Parent: `702199f`

## Axis 1 — Scope additive / blast-radius

**GREEN**

`git show --name-status 1f5d8e4` changed:

- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`
- 4 NEW tests:
  - `tests/test_compress_rope_oracle.py`
  - `tests/test_hca_compressor_mlx_parity.py`
  - `tests/test_attention_real_mlx_hca_parity.py`
  - `tests/test_13_3b_1_dispatch_additive.py`
- 3 mandatory sha-pin sentinel one-line updates:
  - `tests/test_deepseek_v4_real_config_reference_forward.py`
  - `tests/test_numpy_real_forward_reference_composition.py`
  - `python-envs/mlx/src/ds4_ft_mlx/real_forward_intermediate_dump.py`
- handoff note: `agent-output/cmux-13-3b/coder-13-3b-1-notes.md`

No `deepseek_v4_nn.py`. No convert/remap script. No CSA/Indexer real-path helper added. New top-level defs in `deepseek_v4.py` exactly:

- `_compress_rope_yarn_tail_tables_mlx`
- `_hca_compressor_mlx`
- `_attention_real_mlx`

Existing `_attention_mlx` change is the one sanctioned dispatch branch only.

## Axis 2 — FROZEN bodies byte-intact

**GREEN**

Independent AST/source extraction from `702199f` and `1f5d8e4`; not `git diff`-based.

Protected vendor symbols identical:

- `_csa_config_error` `99704f941c45548b`
- `_require_csa_config` `efa91e952d065be4`
- `_csa_attention_mlx` `7650504c8b941988`
- `_csa_compressor_mlx` `a1fa4b102f9886f0`
- `_csa_indexer_mlx` `23dd586f4e757db3`
- `_csa_windowed_compressor_mlx` `bd693675dfcc5742`
- `_hyperconnection_mlx` `bd2318f5bb7d1b8e`
- `_hyperhead_mlx` `23f668299c02c5af`
- `sanitize_weights` `d851df75062f5e6b`
- parity `Model` `2323d29529cb1e84`
- `_dequantize_i8_block_scale_mlx` `844271cbe8c95d56`
- `_dequantize_fp4_block_scale_mlx` `3d650549b0c83341`
- `_SUPPORTED_EXPERT_DTYPES` `943705066669d6e1`
- `_apply_rope_full_mlx` `3916b9a8312051b0`
- `_rope_tail_tables_mlx` `79572e90303e4369`
- `_apply_rope_tail_mlx` `0824ad7ed1981c50`

ADR-0017 external forbidden symbols identical:

- `decode_f8_e8m0_scales` `fa46bce4318dc549`
- `dequantize_f8_e4m3fn_with_e8m0_scales` `8237f9a694a2bead`
- `_apply_i8_block_scales` `ff511a61723d9c69`
- `dequantize_i8_block_scale` `2c923f4908b7934b`
- `dequantize_expert_packed` `7a85b33cf2280f75`
- `f8_e8m0_to_bf16` `d07bc51f9cde725d`

`_attention_mlx` cr=0 remainder after sanctioned dispatcher block is byte-identical. Coder-claimed hash verified using AST statement slice `node.body[2:]`: `7c9960107ef0919d` parent=head.

## Axis 3 — AC0 RoPE oracle

**GREEN**

`tests/test_compress_rope_oracle.py` uses torch `DeepseekV4RotaryEmbedding` + torch `apply_rotary_pos_emb` as oracle, with same synthesized config and same base tensor as MLX. Not a second MLX oracle.

Confirmed:

- trailing `qk_rope_head_dim=64`, not full `head_dim=512`
- asserts cos shape `(1, n_win, 32)`
- YaRN: `factor=16`, `original_max_position_embeddings=65536`, `theta=160000`, beta `32/1`
- positions: `arange(n_win) * rate`
- tolerances: cos/sin `atol=1e-3`, rotated `atol=1e-2`, `rtol=0.0`
- anti-tautology: independent torch tensors in `torch.testing.assert_close`; breaking scope/yarn/positions fails

Independent deltas:

- `cos_max_abs=7.1525574e-07`
- `sin_max_abs=1.8626451e-06`
- `rotated_max_abs=0.0078125`

## Axis 4 — AC1 HCA compressor

**GREEN**

`tests/test_hca_compressor_mlx_parity.py` uses torch `DeepseekV4HCACompressor.forward(... past_key_values=None ...)` as reference.

Confirmed `_hca_compressor_mlx:591-654`:

- `rate=128`
- `out_dim=head_dim=512`
- `compressor_ape` shape `(128, 512)`; no transpose
- no Ca/Cb overlap
- single non-overlap window pooling
- returns `compressed_kv [B,1,n_win,head_dim]`
- returns `block_bias [B,1,S,n_win]`
- causal rule: entry `i` visible iff `i < ((position_ids + 1) // rate)`, else `-inf`

Real header shape checks passed for layer 3 compressor weights.

Independent delta:

- `compressed_kv_max_abs=3.9190054e-06`
- block-bias `-inf` mask equal

## Axis 5 — AC2 HCA attention

**GREEN**

`tests/test_attention_real_mlx_hca_parity.py` uses torch `DeepseekV4Attention.forward` for HCA layer as reference.

Confirmed `_attention_real_mlx:658-749` follows Architect §3.4:

1. shared cr=0 projections/norms + compress YaRN tail RoPE
2. HCA branch via `_hca_compressor_mlx`
3. `kv_full = cat([kv, compressed_kv], axis=2)`
4. `mask = cat([sliding_mask, block_bias], axis=-1)`
5. multi-head scores + per-head sink + softmax + attend, same cr=0 math pattern
6. rope-undo with same cos and `-sin`; grouped `o_a` / `o_b` output tail

Q3 confirmed: `wq_a/wq_b/wkv/wo_a/wo_b`, `q_norm`, `kv_norm`, and `sinks` are shared cr=0 weights; only `compressor_wkv/wgate/ape/norm` are HCA-specific.

Q4 confirmed: rope-undo and grouped-o are in HCA path.

Independent delta:

- `output_max_abs=6.2584877e-07`

## Axis 6 — AC3 additive dispatch

**GREEN**

`tests/test_13_3b_1_dispatch_additive.py` confirms:

- tiny CSA: `_csa_config_error(args) is None`
- tiny direct `_csa_attention_mlx(...)` equals routed `_attention_mlx(...)` by exact `.tolist()` comparison
- real HCA: `_csa_config_error(real_args) is not None`
- real HCA dispatch reaches `_attention_real_mlx` monkeypatch, no old `NotImplementedError`

Dispatch branch at `_attention_mlx:852-855`:

```python
if args.compression_ratio != 0:
    if _csa_config_error(args) is None:
        return _csa_attention_mlx(args, x, weights, index_topk=index_topk)
    return _attention_real_mlx(args, x, weights, index_topk=index_topk)
```

No existing tiny test input changes branch.

## Axis 7 — sha-pin cascade watch-item

**GREEN**

Whole-file vendor hash changed legitimately because ADR 0026 sanctioned additive `deepseek_v4.py` baseline changed.

Recomputed:

- parent vendor sha16: `a7d75d4c8f5ae73e`
- head vendor sha16: `764162ca41531e52`

Exactly 3 source pin sites advanced to `764162ca41531e52`:

1. `tests/test_deepseek_v4_real_config_reference_forward.py`
2. `tests/test_numpy_real_forward_reference_composition.py`
3. `python-envs/mlx/src/ds4_ft_mlx/real_forward_intermediate_dump.py`

Each changed exactly one line. No old `a7d75d4c8f5ae73e` remains in non-pycache source scan. No phantom sentinel update: all three pin full-file `deepseek_v4.py` hash and all three would be stale after sanctioned additions + dispatch branch.

Adjudication: cascade legitimate, not phantom.

## Axis 8 — Regression / no scope explosion

**GREEN**

Reviewer reruns:

- 4 new AC tests: `4 passed`
- 13.2 FP4: `9 passed, 1 skipped`
- 13.3a nn strict subset: `12 passed`
- tiny CSA grep subset: `49 passed, 526 deselected, 8 subtests passed`
- rope regression excluding new oracle: `10 passed, 565 deselected, 5 subtests passed`
- full suite: `1 failed, 561 passed, 13 skipped, 96 subtests passed`

Sole full-suite RED:

- `tests/test_deepseek_v4_mlx_port.py::DeepSeekV4MlxPortTests::test_finetune_readiness_report_records_b0b_a_without_unblocking_b0`
- failure: `proofs_total 9 != 8`
- matches task carve-out; no introduced RED beyond this.

`git diff --check 1f5d8e4^ 1f5d8e4 -- .`: clean.

No nn port edit. No convert edit. No CSA/Indexer real-path implementation.

## Final

All 8 axes GREEN. No remediation needed.
