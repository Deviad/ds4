# Review — Story 13.3b-2b (CSA attention wiring)

Verdict: **PASS**

Commit reviewed: `aa93c5d`
Parent: `a2c20b0`
Status: `{"status":"ok","role":"Reviewer"}`

## 4-axis audit summary

- Scope/blast-radius: **GREEN** — additive helper slice only; no nn port, no convert/remap, no `_attention_mlx` dispatch, no GroupedLinear.
- Correctness/parity: **GREEN** — CSA compressor, block-bias, and CSA attention parity independently reproduced.
- Regression: **GREEN** — targeted gates pass; full suite has the known stale proof-count failure only.
- Boundary/maintainability: **GREEN** — FROZEN bodies byte-identical; CSA helper standalone for 13.3b-3 wiring.

## Axis 1 — Scope / additive-only blast radius

**GREEN**

Evidence:

- `git show --stat aa93c5d` reports 7 files:
  - `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`
  - 2 new AC tests:
    - `tests/test_csa_compressor_real_mlx_parity.py`
    - `tests/test_csa_attention_real_mlx_parity.py`
  - 3 required sha/capture sentinel updates:
    - `tests/test_deepseek_v4_real_config_reference_forward.py`
    - `tests/test_numpy_real_forward_reference_composition.py`
    - `python-envs/mlx/src/ds4_ft_mlx/real_forward_intermediate_dump.py`
  - handoff note: `agent-output/cmux-13-3b/coder-13-3b-2b-notes.md`
- Strict “3 files only” wording conflicts with Axis 7 sha-pin cascade requirement. Adjudication: extra hunks are hash/capture sentinels + handoff artifact, not semantic scope expansion.
- `deepseek_v4.py`: `186 insertions(+), 0 deletions(-)`.
- Only one `deepseek_v4.py` hunk: adds `_csa_compressor_real_mlx`, `_csa_block_bias_mlx`, `_csa_attention_real_mlx` after `_hca_compressor_mlx`.
- AST top-level diff: new symbols exactly `['_csa_attention_real_mlx', '_csa_block_bias_mlx', '_csa_compressor_real_mlx']`; removed `[]`; changed existing defs `[]`.
- Forbidden scope scan clean: no `deepseek_v4_nn.py`, no convert/remap script, no GroupedLinear / `_grouped_linear_mlx`, no `_attention_mlx` dispatch edit.

## Axis 2 — FROZEN bodies byte-intact

**GREEN**

Independent AST/source-hash proof vs `a2c20b0`:

- `_attention_real_mlx`: source `fe11f06bcbbde0fd` → `fe11f06bcbbde0fd`; AST `29410aefa740e99f` → `29410aefa740e99f`.
- `_csa_windowed_compressor_mlx`: `bd693675dfcc5742` unchanged; AST unchanged.
- `_hca_compressor_mlx`: `5da7138a26c4e9e0` unchanged; AST unchanged.
- `_indexer_mlx`: `a370d543b86d4cbd` unchanged; AST unchanged.
- `_indexer_scorer_mlx`: `27429b628bf61e51` unchanged; AST unchanged.
- `_compress_rope_yarn_tail_tables_mlx`: `4aed889cadabba74` unchanged; AST unchanged.
- `_apply_rope_tail_mlx`: `0824ad7ed1981c50` unchanged; AST unchanged.
- `_apply_rope_full_mlx`: `3916b9a8312051b0` unchanged; AST unchanged.
- `_rope_full_tables_mlx`: `fbc5959b50bcbca4` unchanged; AST unchanged.
- `_attention_mlx`: `59ba0b5364b468a6` unchanged; AST unchanged.
- `_csa_attention_mlx`: `7650504c8b941988` unchanged; AST unchanged.
- `_csa_compressor_mlx`: `a1fa4b102f9886f0` unchanged; AST unchanged.
- `_csa_indexer_mlx`: `23dd586f4e757db3` unchanged; AST unchanged.
- `_csa_config_error`: `99704f941c45548b` unchanged; AST unchanged.
- `_require_csa_config`: `efa91e952d065be4` unchanged; AST unchanged.
- `_causal_sliding_mask_mlx`: `1e142c334a0b9300` unchanged; AST unchanged.
- `_hyperconnection_mlx`: `bd2318f5bb7d1b8e` unchanged; AST unchanged.
- `_hyperhead_mlx`: `23f668299c02c5af` unchanged; AST unchanged.
- `sanitize_weights`: `d851df75062f5e6b` unchanged; AST unchanged.
- `Model`: `2323d29529cb1e84` unchanged; AST unchanged.
- ADR-0017 forbidden symbols also byte/AST unchanged: `decode_f8_e8m0_scales`, `dequantize_f8_e4m3fn_with_e8m0_scales`, `_apply_i8_block_scales`, `dequantize_i8_block_scale`, `dequantize_expert_packed`, `_SUPPORTED_EXPERT_DTYPES`, `_dequantize_i8_block_scale_mlx`, `_moe_mlx`, `f8_e8m0_to_bf16`.

Critical watch-item 1: HCA `_attention_real_mlx` body byte-identical. No HCA branch edit.

## Axis 3 — RoPE option α + APE no-transpose

**GREEN**

Confirmed `_csa_compressor_real_mlx:819-884`:

- Does not AST-call `_csa_windowed_compressor_mlx`; mention is comment only.
- Validates `compressor_ape` shape `(rate, 2*out_dim)` = `(4, 1024)`.
- Consumes APE token-major with no transpose:
  - `gate.reshape((batch, n_windows, rate, 2 * out_dim)) + weights["compressor_ape"].reshape((1, 1, rate, 2 * out_dim))`
  - no `.T`, `transpose`, or `swapaxes` on `compressor_ape`.
- Reimplements Ca/Cb overlap inline:
  - `cb_kv = kv[..., out_dim:]`
  - `zero_kv` + `masked_gate` for window-0 Ca
  - previous-window Ca via `kv[:, :-1, :, :out_dim]` and `gate[:, :-1, :, :out_dim]`
  - `new_kv = concatenate([ca_kv, cb_kv], axis=2)`
  - fp32 softmax over overlap axis, RMSNorm over `out_dim=512`.
- RoPE uses reused 13.3b-1 helpers:
  - `positions = mx.arange(n_windows) * rate`
  - `_compress_rope_yarn_tail_tables_mlx(... args.qk_rope_head_dim, args.compress_rope_theta, factor=16, original_max=65536 ...)`
  - `_apply_rope_tail_mlx(... qk_rope_head_dim=args.qk_rope_head_dim)`
- Returns `mx.expand_dims(compressed, 1)` → `[B,1,n_win,512]`.
- FROZEN `_csa_windowed_compressor_mlx:347-416` byte-identical; no rope-suppress mode edit.

Watch-item 2 resolved.

## Axis 4 — AC1 CSA compressor parity + anti-circularity

**GREEN**

Confirmed `tests/test_csa_compressor_real_mlx_parity.py`:

- Torch oracle is installed `DeepseekV4CSACompressor.forward` (`623-702` in local source), invoked directly.
- SAME synthesized weights copied into torch module and MLX helper.
- Real layer-2 header shape cross-checks present:
  - `compressor.wkv.weight` `(1024, 4096)`
  - `compressor.wgate.weight` `(1024, 4096)`
  - `compressor.ape` `(4, 1024)`
  - `compressor.norm.weight` `(512,)`
- Fixture uses single kv head, `out_dim=head_dim=512`, `cr=4`, token-major APE, Ca/Cb overlap, compress-YaRN tail.
- Independent recomputation:
  - shape `(1, 1, 64, 512)`
  - `compressor_max_abs = 5.9604644775390625e-06` vs torch, within `atol=1e-2`.

## Axis 5 — AC2 CSA attention parity + block_bias exactness

**GREEN**

Confirmed `tests/test_csa_attention_real_mlx_parity.py`:

- Torch oracle is installed `DeepseekV4Attention.forward` (`801-873` in local source), invoked directly for a `compressed_sparse_attention` layer.
- SAME synthesized weights copied into torch attention module and MLX helper.
- Test separately calls torch `module.compressor(...)` for expected `compressed_kv` and `block_bias`.
- Block-bias assertions cover:
  - `torch.isneginf(got_bias_t)` equals `torch.isneginf(expected_bias)` exactly.
  - full value equality after `nan_to_num(..., neginf=-1e9)` with `atol=0`, which pins valid picks to exact `0.0` and sentinel/unpicked to `-inf`.
- Independent recomputation:
  - `attention_compressor_max_abs = 6.556510925292969e-06`
  - `block_bias_neginf_equal = True`
  - `block_bias_finite_max_abs = 0.0`
  - attention shape `(1,129,4096)`
  - `attention_max_abs = 6.48200511932373e-07`
  - `attention_mean_abs = 4.686803833919839e-08`

Confirmed `_csa_attention_real_mlx:907-1002`:

- q/kv projections + compress-YaRN tail RoPE reused.
- `compressed_kv = _csa_compressor_real_mlx(...)`.
- `top_k_indices = _indexer_mlx(...)`.
- `block_bias = _csa_block_bias_mlx(...)`.
- KV append: `kv_full = mx.concatenate([kv_by_head, compressed_kv], axis=2)`.
- Mask append: `mx.concatenate([sliding_mask + zeros(...), block_bias], axis=-1)`.
- Multi-head score/softmax/sink/attend path mirrors HCA/cr=0 pattern.
- Rope undo via `_apply_rope_tail_mlx(attended, cos, -sin, ...)`.
- Grouped output path loops `o_groups` and applies `o_a_proj.weight` chunks then `o_b_proj.weight`.

Watch-item 3 resolved.

## Axis 6 — Standalone helper boundary + HCA unchanged

**GREEN**

- New helpers are standalone: `_csa_compressor_real_mlx`, `_csa_block_bias_mlx`, `_csa_attention_real_mlx`.
- `_attention_mlx` source/AST unchanged vs `a2c20b0`; no cr=4 dispatch branch.
- `_attention_real_mlx` HCA source/AST unchanged vs `a2c20b0`.
- `_csa_attention_real_mlx` production call sites: definition only.
- Non-production call sites: only `tests/test_csa_attention_real_mlx_parity.py` imports/calls it.
- 13.3b-3 still owns consumer wiring.

Watch-item 1 resolved.

## Axis 7 — sha-pin cascade slice invariant

**GREEN**

Whole-tree scan scope: `tests/`, `scripts/finetune_ds4.py`, `python-envs/mlx/src/`, excluding `.venv`, `node_modules`, `__pycache__`.

Computed vendor hashes:

- parent `a2c20b0`: `812df0f7a34c0f07`
- head `aa93c5d`: `dc5aaaab9bb079d2`
- current working file: `dc5aaaab9bb079d2`

Vendor `deepseek_v4.py` content-hash pin sites:

1. `tests/test_numpy_real_forward_reference_composition.py:397`
2. `tests/test_deepseek_v4_real_config_reference_forward.py:383`
3. `python-envs/mlx/src/ds4_ft_mlx/real_forward_intermediate_dump.py:61`

Adjudication:

- Exactly 3 content-hash pins advanced to `dc5aaaab9bb079d2`.
- Old hash `812df0f7a34c0f07` occurrences in non-pycache scan: `0`.
- No stale old hash.
- No phantom content-hash update: all 3 pin the full-file `deepseek_v4.py` hash and must change after additive helper insertion.
- `real_forward_intermediate_dump.py` also advances `VENDOR_CAPTURE_LINE` to `deepseek_v4.py:2518 return h`, matching current `return h` line after helper insertion.

Watch-item 4 resolved.

## Axis 8 — Regression + no scope explosion

**GREEN**

Reviewer reruns:

- Targeted 13.3b-1 + 13.3b-2a + 13.3b-2b + 13.3a nn + 13.2 FP4 command:
  - `29 passed, 1 skipped`.
- Tiny CSA focused regression:
  - `7 passed, 53 deselected`.
- Full suite:
  - `1 failed, 565 passed, 13 skipped, 96 subtests passed`.

Sole full-suite failure:

- `tests/test_deepseek_v4_mlx_port.py::DeepSeekV4MlxPortTests::test_finetune_readiness_report_records_b0b_a_without_unblocking_b0`
- `AssertionError: 9 != 8` at `tests/test_deepseek_v4_mlx_port.py:1104`.
- Matches existing stale proof-count / Test #8 carve-out. No CSA/HCA/indexer/nn/FP4 introduced red.

Other checks:

- `git diff --check aa93c5d~1..aa93c5d` clean.
- No nn port edit.
- No convert/remap edit.
- No dispatch branch.
- No GroupedLinear.

## Final verdict

All 8 axes **GREEN**.

`{"status":"ok","role":"Reviewer"}`
