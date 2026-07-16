# Review — Story 13.3b-2a (Indexer real MLX port)

Verdict: **PASS**

Commit reviewed: `a2c20b0`
Parent: `1f5d8e4`
Status: `{"status":"ok","role":"Reviewer"}`

## Axis 1 — Scope / additive-only blast radius

**GREEN**

Evidence:

- `git show --stat a2c20b0` reports 7 files:
  - `deepseek_v4.py`
  - 2 new tests: `tests/test_indexer_mlx_parity.py`, `tests/test_indexer_scorer_mlx_parity.py`
  - 3 required sha-pin sites: `tests/test_deepseek_v4_real_config_reference_forward.py`, `tests/test_numpy_real_forward_reference_composition.py`, `python-envs/mlx/src/ds4_ft_mlx/real_forward_intermediate_dump.py`
  - handoff note: `agent-output/cmux-13-3b/coder-13-3b-2a-notes.md`
- Strict “3 files only” wording conflicts with Axis 7 / BA §C sha-pin cascade requirement. Adjudication: extra hunks are sanctioned hash-only pins + agent-output evidence, not semantic scope expansion.
- `deepseek_v4.py`: `161 insertions(+), 0 deletions(-)`.
- Single hunk appended `_indexer_scorer_mlx` + `_indexer_mlx` after `_compress_rope_yarn_tail_tables_mlx`.
- No `deepseek_v4_nn.py` edit.
- No convert/remap script edit.
- No CSA attention wiring / consumer branch added in this slice.
- `_attention_mlx` source hash unchanged vs `1f5d8e4`.

## Axis 2 — FROZEN bodies byte-intact

**GREEN**

Independent AST/source-hash proof vs `1f5d8e4`:

- `deepseek_v4.py` protected symbols checked: 22; changed: 0.
- ADR-0017 external forbidden symbols checked: 6; changed: 0.
- Combined checked: 28; changed: 0.

Key samples:

- `_csa_windowed_compressor_mlx`: source `bd693675dfcc5742` → `bd693675dfcc5742`; AST `5110ed9e6dc469fc` → `5110ed9e6dc469fc`.
- `_compress_rope_yarn_tail_tables_mlx`: source `4aed889cadabba74` → `4aed889cadabba74`; AST unchanged.
- `_attention_real_mlx`: source `fe11f06bcbbde0fd` → `fe11f06bcbbde0fd`; AST unchanged.
- `_attention_mlx`: source `59ba0b5364b468a6` → `59ba0b5364b468a6`; AST unchanged.
- `Model`: `2323d29529cb1e84` → `2323d29529cb1e84`.
- `sanitize_weights`: `d851df75062f5e6b` → `d851df75062f5e6b`.
- ADR-0017 samples: `_dequantize_i8_block_scale_mlx`, `_moe_mlx`, `decode_f8_e8m0_scales`, `f8_e8m0_to_bf16` all byte-identical.

No FROZEN body edit. No ADR-0017 violation.

## Axis 3 — Rope landmine

**GREEN**

Confirmed `_indexer_mlx:620-749`:

- Does **not** call `_csa_windowed_compressor_mlx` (AST call count `0`; only comment mention).
- Reimplements Ca/Cb overlap pooling inline:
  - `cb_kv = kv[..., out_dim:]`
  - prior-window leading half via `ca_kv = concatenate([zero_kv, kv[:, :-1, :, :out_dim]], axis=1)`
  - `new_kv = concatenate([ca_kv, cb_kv], axis=2)`
  - same `new_gate` / fp32 softmax / RMSNorm contract as torch lines 535-549.
- Applies compress-YaRN tail RoPE through reused 13.3b-1 helpers:
  - `_compress_rope_yarn_tail_tables_mlx` call count `2`
  - `_apply_rope_tail_mlx` call count `2`
  - compressed positions: `mx.arange(n_windows) * rate`
  - q positions: `position_ids` broadcast per `index_n_heads`
  - trailing `qk_rope_head_dim=64` of `index_head_dim=128`.
- `_csa_windowed_compressor_mlx:347-416` byte-identical; no rope-suppress FROZEN edit.

Watch-item 1 adjudication: resolved by additive inline pooling + reused rope helper. No S-frozen-body breach.

## Axis 4 — AC1 IndexerScorer parity / anti-circularity

**GREEN**

Confirmed `tests/test_indexer_scorer_mlx_parity.py`:

- Torch oracle: `DeepseekV4IndexerScorer.forward:455`.
- Same synthesized weights copied into torch module and MLX helper.
- `weights_proj` synthesized shape `(64, 4096)` asserted.
- `_indexer_scorer_mlx:591-617` math:
  - fp32 `q` + `compressed_kv`
  - ReLU dot scores scaled by `index_head_dim ** -0.5`
  - weights `(hidden @ weights_proj.T) * index_n_heads ** -0.5`
  - sum over index heads → `[B,S,T]`
  - returns `mx.float32`.

AC1 test run: `1 passed` inside targeted 2a/3b-1 run.

## Axis 5 — AC2 Indexer parity / set-based assertion

**GREEN**

Confirmed `tests/test_indexer_mlx_parity.py`:

- Torch oracle: `DeepseekV4Indexer.forward:511` plus torch score extraction mirroring lines 567-586.
- Same synthesized weights/config loaded into torch and MLX.
- Real header shape cross-checks for layer-2 indexer tensors.
- Assertions:
  - `-1` sentinel mask exact: `torch.equal(got_t.eq(-1), expected.eq(-1))`
  - full score tensor close: `atol=1e-3`
  - non-sentinel valid picks compared as `set(got_valid) == set(expected_valid)` per query
  - length exact
  - no exact tie-order assertion.

Confirmed `_indexer_mlx` top-k recipe:

- `causal_threshold = ((position_ids + 1) // rate).astype(mx.int32)`
- future scores → `-inf`
- `top_k = min(int(index_topk), compressed_len)`
- `top_k_indices = mx.argsort(-masked_scores, axis=-1)[..., :top_k]`
- sentinel via `mx.where(invalid, -1, top_k_indices)`.

Watch-item 2 adjudication: set-based, not tie-order flaky.

## Axis 6 — Standalone helper boundary

**GREEN**

- `_indexer_mlx` exact production call sites: definition only.
- `_indexer_mlx` called only by `tests/test_indexer_mlx_parity.py`.
- `_indexer_scorer_mlx` called by `_indexer_mlx` and tests only.
- `_attention_mlx` source/AST unchanged vs `1f5d8e4`.
- `_attention_real_mlx` source/AST unchanged vs `1f5d8e4`.
- HCA path unchanged; still 13.3b-1 body.
- No CSA attention consumer wiring in this slice.

## Axis 7 — sha-pin cascade

**GREEN**

Whole-tree scan scope: `tests/`, `scripts/finetune_ds4.py`, `python-envs/mlx/src/` excluding `.venv`, `node_modules`, `__pycache__`.

Computed vendor hashes:

- parent `1f5d8e4`: `764162ca41531e52`
- head `a2c20b0`: `812df0f7a34c0f07`

Vendor `deepseek_v4.py` pin sites:

1. `tests/test_deepseek_v4_real_config_reference_forward.py:383`
2. `tests/test_numpy_real_forward_reference_composition.py:397`
3. `python-envs/mlx/src/ds4_ft_mlx/real_forward_intermediate_dump.py:61`

Adjudication:

- Exactly 3 content-hash pin sites advanced.
- All 3 now use `812df0f7a34c0f07`.
- Old `764162ca41531e52` occurrences in non-pycache source scan: `0`.
- No phantom update: each changed line pins full-file `deepseek_v4.py` hash and would be stale after additive helpers.

Watch-item 3 adjudication: legitimate cascade, not phantom.

## Axis 8 — Regression / no scope explosion

**GREEN**

Reviewer reruns:

- 13.3b-2a + 13.3b-1 targeted: `6 passed`.
- 13.3a nn strict subset: `12 passed`.
- 13.2 FP4 selected regression: `9 passed, 1 skipped`.
- Tiny CSA grep regression: `50 passed, 527 deselected, 8 subtests passed`.
- Full suite: `1 failed, 563 passed, 13 skipped, 96 subtests passed`.

Sole full-suite failure:

- `tests/test_deepseek_v4_mlx_port.py::DeepSeekV4MlxPortTests::test_finetune_readiness_report_records_b0b_a_without_unblocking_b0`
- `AssertionError: 9 != 8`
- matches pre-existing Test #8 carve-out / proof-count drift.

Other checks:

- `git diff --check` clean.
- No nn port edit.
- No convert edit.
- No CSA attention wiring.
- No new dispatch branch.

## Final verdict

All 8 axes: **GREEN**

`{"status":"ok","role":"Reviewer"}`
