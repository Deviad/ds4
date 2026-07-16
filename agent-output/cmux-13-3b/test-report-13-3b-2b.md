# Test Report — Story 13.3b-2b (Test Manager)

**Coder commit:** `aa93c5d` (parent `a2c20b0`) — Story 13.3b-2b: add CSA real attention helper
**Test Manager model:** neuralwatt/qwen3.6-35b · surface:85, fresh-context
**Date:** 2026-06-26

---

## 1. Git confirm

```
aa93c5d Story 13.3b-2b: add CSA real attention helper
a2c20b0 Story 13.3b-2a: add real indexer MLX helpers
```

✅ **aa93c5d confirmed HEAD.**

## 2. Full test suite

```
1 failed, 565 passed, 13 skipped, 2 warnings, 96 subtests passed in 120.17s
```

- **RED:** Test #8 — `test_finetune_readiness_report_records_b0b_a_without_unblocking_b0`
- **PASS:** 565
- **SKIP:** 13
- **Introduced RED:** 0

✅ **Full suite verdict: EXPECTED** (1 known RED / 565 PASS / 13 SKIP / 0 introduced RED).

## 3. NEW tests (AC1 + AC2)

```
tests/test_csa_compressor_real_mlx_parity.py::test_csa_compressor_real_mlx_matches_torch_stateless_forward_real_dims PASSED
tests/test_csa_attention_real_mlx_parity.py::test_csa_attention_real_mlx_matches_torch_attention_forward_real_dims PASSED
```

Both GREEN. 2.30s total.

**Parity deltas (from test expectations):**
- AC1 (compressor): `torch.testing.assert_close(..., atol=1e-2, rtol=0.0)` — PASS. Max delta ≈5.96e-6 (Coder claim).
- AC2 (attention + block_bias): `torch.testing.assert_close(got_bias_t, ..., atol=0.0, rtol=0.0)` EXACT `-inf` mask + finite delta 0.0 — PASS. Attention `atol=1e-2` — PASS. Max delta ≈6.48e-7 (Coder claim).
- Block bias `-inf` mask: `torch.isneginf` EXACT boolean equality. `torch.nan_to_num(expected_bias)` delta EXACT 0.0.

✅ **AC1 + AC2 GREEN with parity deltas within tolerances. Block bias -inf mask EXACT.**

## 4. WATCH-ITEM 1: HCA `_attention_real_mlx` byte-identical

```
git show --stat: 1 file changed, 186 insertions(+), 0 deletions(-)
git show grep -E '^-[^-]': (empty — zero removal lines)
grep deepseek_v4.py def line numbers:
  819  _csa_compressor_real_mlx(    ← NEW
  887  _csa_block_bias_mlx(...)     ← NEW
  907  _csa_attention_real_mlx(...) ← NEW
 1005  _attention_real_mlx(         ← HCA, untouched
```

- (a) `_attention_real_mlx` HCA branch: **byte-identical** — zero `-` diff lines. Coder chose NEW `_csa_attention_real_mlx` helper; HCA branch not edited.
- (b) 3 NEW helpers added additively: `_csa_compressor_real_mlx:819`, `_csa_block_bias_mlx:887`, `_csa_attention_real_mlx:907`.

✅ **HCA `_attention_real_mlx` byte-identical. 3 new helpers additive.**

## 5. WATCH-ITEM 2: Rope option α + ape no-transpose

Read `_csa_compressor_real_mlx:819` body:

(a) **APE token-major, NO transpose:**
- `rate = int(args.compression_ratio)` → rate=4.
- `out_dim = int(args.head_dim)` → 512; `2*out_dim = 1024`.
- Weight shape validation: `(rate, 2*out_dim)` = `(4, 1024)` — token-major.
- APE applied directly: `gate.reshape(...) + weights["compressor_ape"].reshape((1, 1, rate, 2*out_dim))` — **NO `.T` transpose**. The `ape[:out_dim,:].T` lives only in the bypassed FROZEN `_csa_windowed_compressor_mlx:347`.

(b) **compress-YaRN tail rope via REUSED helpers:**
- Line ~884: `_compress_rope_yarn_tail_tables_mlx` at `:545` — REUSED from 13.3b-1.
- Line ~889: `_apply_rope_tail_mlx` at `:266` — REUSED from 13.3b-1.
- `positions = mx.arange(n_windows) * rate` — window indices.
- `qk_rope_head_dim=args.qk_rope_head_dim` — trailing slice only.

(c) **FROZEN `_csa_windowed_compressor_mlx:347` byte-identical:**
- Comment at L828: "Do not call FROZEN _csa_windowed_compressor_mlx here."
- git diff: zero `-` lines in entire file → no body edits anywhere.

✅ **Rope option α: APE token-major `(4, 1024)` NO transpose. Reused 13.3b-1 rope helpers. FROZEN bypassed.**

## 6. WATCH-ITEM 3: block_bias `-inf` mask EXACT

Read `_csa_block_bias_mlx:887` body:

(a) **`-1` sentinel → `-inf` mask:**
- `valid = top_k_indices >= 0` — identifies valid vs sentinel positions.
- `block_bias = mx.full(..., -float("inf"))` — initialized to `-inf`.
- `scatter_indices = where(valid, top_k_indices, sentinel).astype(int32)` — non-valid positions mapped to `compressed_len` sentinel.
- `mx.put_along_axis(block_bias, scatter_indices, zeros, axis=-1)` — only valid indices get scatter-zero; `-inf` preserved at sentinel positions.
- AC2 test: `torch.testing.assert_close(torch.isneginf(got_bias_t), torch.isneginf(expected_bias))` — EXACT boolean equality.

(b) **valid pick → 0 (finite_max_abs == 0.0):**
- AC2 test: `torch.testing.assert_close(torch.nan_to_num(got_bias_t, neginf=-1e9), torch.nan_to_num(expected_bias, neginf=-1e9), atol=0.0, rtol=0.0)` — finite deltas exactly 0.0.

(c) **scatter correctness:**
- `scatter_indices = mx.expand_dims(safe_indices, 1)` — `[batch, 1, seq, k]` broadcast to `[batch, 1, seq, compressed_len+1]`.
- `axis=-1` — scatter along bias column dimension.
- `[..., :compressed_len]` — trim trailing sentinel column.

✅ **Block bias -inf mask: EXACT boolean equality with torch. Finite deltas 0.0. Scatter correct.**

## 7. FROZEN byte-intact

```
git show --stat: 186 insertions, 0 deletions
git show grep -E '^-[^-]': (empty)
```

Zero `-` diff lines = NO body edits to any existing function. Confirmed FROZEN bodies all byte-identical:
- `_attention_real_mlx` (HCA) ✅
- `_csa_windowed_compressor_mlx:347` ✅
- `_hca_compressor_mlx:752` ✅
- `_indexer_mlx:620` ✅
- `_indexer_scorer_mlx:591` ✅
- `_compress_rope_yarn_tail_tables_mlx:545` ✅
- `_apply_rope_tail_mlx:266` ✅
- `_rope_*_tables_mlx:210/227/300` ✅
- `_attention_mlx cr=0:1190` ✅
- `_csa_attention_mlx:515` ✅
- `_csa_compressor_mlx:419` ✅
- `_csa_indexer_mlx:435` ✅
- `_csa_config_error:317` ✅
- `_causal_sliding_mask_mlx:1100` ✅
- `_hyperconnection_mlx:1108` ✅
- `_hyperhead_mlx:1158` ✅
- `sanitize_weights:2160` ✅
- parity `Model` ✅
- 9 ADR-0017 forbidden ✅

✅ **FROZEN byte-intact. Pure additions only.**

## 8. WATCH-ITEM 4: Sha-pin cascade rigorous count

Full-tree grep (non-pycache, non-.venv, non-binary):

| # | File | Line | Description |
|---|------|------|-------------|
| 1 | `tests/test_deepseek_v4_real_config_reference_forward.py` | 374 | `EXPECTED_SHAS = {` |
| 2 | same | 404 | `def _sha16(path: Path) -> str:` |
| 3 | same | 423 | `assert _sha16(p) == sha` (ADR drift) |
| 4 | same | 427 | `assert _sha16(p) == EXPECTED_SHAS[...]` (vendor deepseek_v4.py FROZEN) |
| 5 | same | 437 | `assert _sha16(p) == EXPECTED_SHAS[rel]` (frozen test drift) |
| 6 | same | 442 | `assert _sha16(p) == EXPECTED_SHAS[rel]` (C-engine drift) |
| 7 | `tests/test_numpy_real_forward_reference_composition.py` | 392 | `EXPECTED_SHAS = {` |
| 8 | same | 403 | `def _sha16(path: Path) -> str:` |
| 9 | same | 422 | `assert _sha16(p) == sha` (ADR drift) |
| 10 | same | 426 | `assert _sha16(p) == EXPECTED_SHAS[...]` (vendor deepseek_v4.py) |
| 11 | same | 435 | `assert _sha16(p) == EXPECTED_SHAS[...]` (C-engine drift) |
| — | `tests/test_real_forward_intermediate_dump.py` | 78 | sha256[:16] comment (mirrors 11.54 EXPECTED_SHAS) |

- **FROZEN test assertions: 11 sites** (6 in real_config_reference + 5 in numpy_real_forward).
- `finetune_ds4.py:1740/3829` — `run_tiny_hash_moe_fixture` imports (not sha-pin assertions).
- `test_finetune_ds4.py:513` — `test_deepseek_v4_mapping_marker_is_bound_to_index_hash` — test NAME contains "hash" but validates marker binding, not sha comparison.
- **No stale old hash.** FROZEN tests pass (`test_frozen_tests_unchanged` PASS, `test_vendor_deepseek_v4_frozen` PASS, `test_11_15h_stop_audit_frozen` PASS, `test_attention_spec_frozen` PASS).

✅ **Sha-pin cascade: 11 definitive non-pycache sha-pin assertion sites, all GREEN. No phantom, no stale.**

## 9. Anti-circularity

- `test_csa_compressor_real_mlx_parity.py`: torch `DeepseekV4CSACompressor` (from `transformers`) with synthesized weights (`manual_seed(133220)`). **No second MLX primitive.**
- `test_csa_attention_real_mlx_parity.py`: torch `DeepseekV4Attention` (same source) with synthesized weights (`manual_seed(133221)`). **No second MLX primitive.**
- ADR 0007 §4: Both tests use torch ref from SAME synthesized weights. No MLX-on-MLX oracle.

✅ **Anti-circularity: HONORED.**

## 10. Regression tests

```
tests/test_13_3b_1_dispatch_additive.py: 1 passed (13.3b-1 AC0-AC3)
tests/test_deepseek_v4_forward_parity.py: passed (full forward parity)
tests/test_deepseek_v4_nn_forward.py: passed (nn forward)
tests/test_deepseek_v4_nn_backward.py: passed (nn backward)
tests/test_deepseek_v4_fp4_dequant_mlx.py: passed (FP4 dequant)
```

✅ **All regression tests GREEN.**

## 11. Clean diff

```
git diff --check: exit 0 (no whitespace errors)
```

✅ **Clean.**

## Final verdict

| Check | Result |
|-------|--------|
| AC1 GREEN (compressor parity) | ✅ PASS |
| AC2 GREEN (attention parity + block_bias -inf mask EXACT) | ✅ PASS |
| Full suite 1 RED / 565 PASS / 13 SKIP | ✅ PASS |
| HCA `_attention_real_mlx` byte-identical | ✅ PASS |
| Rope α + ape no-transpose | ✅ PASS |
| FROZEN byte-intact | ✅ PASS |
| Sha-pin cascade (11 sites, advanced) | ✅ PASS |
| Anti-circularity (torch oracle) | ✅ PASS |
| Regressions GREEN | ✅ PASS |
| git diff --check clean | ✅ PASS |

**PASS.** All checks green. 0 introduced RED. FROZEN bodies byte-intact. HCA branch untouched.
