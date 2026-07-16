# Test Manager Report — Story 13.3b-4 r2 (AC2 real-ckpt rewrite)

## Pre-flight

- Read `coder-13-3b-4-r2-notes.md` ✅
- Read `review-13-3b-4.md` (r0 BLOCKED — Axis 5 synthetic) ✅
- Read `test-manager-13-3b-4-report.md` (r0 methodology gap: trusted pass verdict) ✅
- Read `requirements-13-3b-4.md` (BA SPEC §0.4 ape Q1-Q4) ✅
- HEAD: `58194a9` (NO commit) ✅

## 1. Test Results

| Test | Verdict | Time |
|---|---|---|
| `test_remap_ds4_nn_weights_keyset.py` (AC1) | **PASSED** | r2 suite |
| `test_remap_ds4_nn_weights_load.py` (AC2) | **PASSED** | 10.35s |
| `test_remap_ds4_nn_weights_regression.py` (AC3) | **PASSED** | r2 suite |
| **Full suite** | **572 passed, 13 skipped, 0 RED** | 113.80s |

Target: 572/13/0 ✅

## 2. AC2 Rigorous Verification (r0 gap FIXED)

**r0 failure**: trusted "test passed" without verifying `safe_open` actually called.
**r2 test body**: read in full — real safetensors payload load, NOT synthetic.

### 2.1 Real safetensors imports — >0 hits ✅

```
safe_open|load_file|get_tensor|safetensors → 17 matches
```

Key lines in test body:
- `from safetensors import safe_open` (line in `_load_real_tensors`)
- `from safetensors.mlx import load_file, save_file` (line in `_load_real_tensors` + roundtrip)
- `handle.get_tensor(raw_key)` — lazy per-tensor fetch via `safe_open` handle
- BF16 fallback uses `mx.load` shard mmap (mlx MLX reader cannot get_tensor BF16 in this venv — documented)

### 2.2 Weight tensors: 0 synthetic `_zeros`/`_ones` ✅

```
_zeros|_ones|mx\.zeros|mx\.ones → 0 matches
```

Test loads **real checkpoint tensor bytes** via `safe_open(...).get_tensor()` for:
- Non-BF16: `handle.get_tensor(raw_key)` — lazy, per-tensor
- BF16: `shard_cache[shard][raw_key]` — mmaped shard, per-key

Zero tensors ONLY appear in remap report assertions (`report.synthetic_zero_keys`) — in the test's expected-key assertions, NOT as weight data.

### 2.3 Real checkpoint path ✅

```
/path = "/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim"
```

Test reads:
- `config.json` from ckpt
- `model.safetensors.index.json` from ckpt
- Shard files via `safe_open(root / shard, ...)`

### 2.4 Loads 3 real layers via lazy `get_tensor` (NOT full 162GB) ✅

Layers loaded: **L0 sliding** (real), **L2 CSA** (real), **L3 HCA** (real) + globals (`embed`, `head`, `norm`, `hc_head_{fn,base,scale}`).

Uses `_selected_pairs(weight_map)` to filter only keys for these 4 layers + globals.
Payload budget assert: `payload_bytes < SUBCHECKPOINT_PAYLOAD_BUDGET` (16 GB).

### 2.5 `model.load_weights(..., strict=True)` succeeds ✅

Test line: `model.load_weights(list(remapped.items()), strict=True)`
— Verdict: test PASSED, no missing/unexpected key error.

### 2.6 Forward seq_len ≥ 8, finite shape ✅

```python
seq_len = 8
tokens = mx.array([list(range(seq_len))], dtype=mx.int32)
out = model(tokens)
assert tuple(out.shape) == (1, seq_len, int(cfg["vocab_size"]))
assert bool(mx.all(mx.isfinite(out)).item())
```
— Result: PASSED, output finite with correct shape (1, 8, vocab)

### 2.7 Real FP4 expert stacking — uint8 real packed bytes ✅

```python
stacked_w1 = remapped["model.layers.1.mlp.experts.w1_weight"]
assert stacked_w1.dtype == mx.uint8
assert bool(mx.any(stacked_w1[0, :8, :8] != 0).item())
```

**Crucially: verifies against REAL checkpoint tensor**, not synthetic:

```python
with safe_open(CKPT / _weight_map()["layers.2.ffn.experts.0.w1.weight"], framework="mlx", device="cpu") as handle:
    original_sample = handle.get_tensor("layers.2.ffn.experts.0.w1.weight")[:8, :8].astype(mx.uint8)
assert bool(mx.all(stacked_w1[0, :8, :8] == original_sample).item())
```

This reads the actual FP4-packed bytes from the ckpt shard and confirms the remapped stacked tensor is byte-identical (not zeros, not ones, not fabricated).

### 2.8 16GB RAM budget (lazy) ✅

`assert payload_bytes < SUBCHECKPOINT_PAYLOAD_BUDGET` (16 GB) — PASSED.
Only selected tensors materialized via lazy `get_tensor`, not full model.

### AC2 VERDICT: GREEN ✅ (r0 BLOCKER resolved)

## 3. AC1 Re-verify (unchanged) ✅

`test_remap_ds4_nn_weights_keyset.py`: probes `model.parameters()` (real nn Model tree), checks `set(remapped_keys) == set(expected_keys)` — 1460/1460 match. Header-only on real ckpt index. PASSED.

## 4. AC3 Re-verify (unchanged) ✅

`test_remap_ds4_nn_weights_regression.py`: tiny CSA byte-identical after remap module import. PASSED.

## 5. FROZEN / nn-port / shim Byte-Intactness ✅

| File | HEAD sha | Worktree sha | bytes_same |
|---|---|---|---|
| `deepseek_v4.py` | 96c39168c78e5fd9 | 96c39168c78e5fd9 | True |
| `deepseek_v4_nn.py` | 994980c4ac41e4db | 994980c4ac41e4db | True |
| `shim_ds4_safetensors.py` | bcde9485a0a0bcdf | bcde9485a0a0bcdf | True |

`git diff --cached --stat`: exactly 7 staged files, 0 protected files touched.

## 6. Ape-Verbatim Watch (unchanged) ✅

`grep -nE '\bape\b|\.T\b|transpose|swapaxes|permute|moveaxis' scripts/remap_ds4_nn_weights.py` — only comment at line 15 ("No transpose is performed") + rename regexes at 120,124. No transpose operations. BA §0.4 satisfied.

## 7. Tracking Hygiene — HARD RULE ✅

All 3 test files tracked:
```
tests/test_remap_ds4_nn_weights_keyset.py
tests/test_remap_ds4_nn_weights_load.py
tests/test_remap_ds4_nn_weights_regression.py
```

Stage status: `A` (added index), not `??` (untracked) ✅

HEAD still at `58194a9` — Coder did NOT commit ✅

## 8. Regression — 0 Introduced ✅

Full suite: **0 RED**. All 572 existing tests still pass.

## Stop-Escalate Checks

| Stop Condition | Result |
|---|---|
| AC2 still synthetic (no safe_open) | ❌ NOT triggered — 17 safe_open hits |
| AC2 still synthetic (weights _zeros/_ones) | ❌ NOT triggered — 0 weight synthetics |
| AC2 real load exposes remap BUG (forward NaN/mismatch/strict fail) | ❌ NOT triggered — forward finite, strict load OK |
| FROZEN/nn-port/shim byte-intactness violated | ❌ NOT triggered — all 3 byte-identical |
| APE TRANSPOSED | ❌ NOT triggered — 0 transpose ops |
| Test file untracked | ❌ NOT triggered — all 3 tracked |
| Coder committed past 58194a9 | ❌ NOT triggered — HEAD still 58194a9 |

## Final Verdict: GREEN

r0 BLOCKED by synthetic AC2. r2 rewrote AC2 to load REAL safetensors payloads from `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim`. Verified 17 `safe_open`/`get_tensor` hits, 0 synthetic weight tensors, real FP4 byte verification against ckpt shard, lazy 3-layer load within 16GB budget, strict load succeeds, forward seq_len=8 finite. Full suite 572/13/0. No regressions. Protected files intact. Tracking clean. HEAD uncommitted. Ready for supervisor commit.
