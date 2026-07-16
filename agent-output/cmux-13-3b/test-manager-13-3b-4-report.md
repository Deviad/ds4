# Test Manager Report — Story 13.3b-4

## Pre-flight
- Read: `coder-13-3b-4-notes.md` ✅
- Read: `requirements-13-3b-4.md` (BA SPEC + §0.4 ape + Q1-Q4) ✅
- Read: `task-coder-13-3b-4.md` (brief) ✅
- Read: AGENTS.md tracking hygiene HARD RULE ✅
- HEAD: `58194a9` (NO commit) ✅

## 1. Test Results — 3 NEW tests + full suite

| Test | Verdict | Notes |
|---|---|---|
| `test_remap_ds4_nn_weights_keyset.py` (AC1) | PASSED | Header-only, probes real ckpt index + real nn Model, 1:1 key+shape match |
| `test_remap_ds4_nn_weights_load.py` (AC2) | PASSED | Sub-checkpoint: 3 layers (sliding+CSA+HCA), strict load, forward@seq129, finite |
| `test_remap_ds4_nn_weights_regression.py` (AC3) | PASSED | Tiny CSA path still byte-identical after remap module import |
| **Full suite** | **572 passed, 13 skipped, 0 RED** | Target was 569+3=572. ✅ |

## 2. Acceptance Criteria Verification

### AC1 — Keyset match (PASSED)
- Expected key-set probed from actual `Model(ModelArgs.from_dict(cfg))` on real ckpt config (43 layers, num_hash_layers=3). NOT hand-written list.
- `remap_header_specs_from_checkpoint()` over real shimmed ckpt index → `set(specs) == set(expected)` EXACT.
- Shape mismatch dict: empty.
- BA §0.4 ape: CSA ape `(4,1024)` ✅, indexer ape `(4,256)` ✅, HCA ape `(128,512)` ✅ — all `transform=="identity"` — no transpose.
- Synthetic zero keys (hash e_score_correction_bias ×3): present ✅.
- `payload_bytes_read == 0`: header-only, zero tensor bytes read ✅.

### AC2 — Sub-checkpoint strict load (PASSED)
- Sub-checkpoint: 1 sliding + 1 CSA + 1 HCA layer, ckpt-native raw keys.
- `model.load_weights(list(remapped.items()), strict=True)` → no missing/unexpected key error.
- Forward at seq 129 → finite output with correct shape `(1, 129, vocab)`.
- Expert stacking verified: `w1_weight.shape == (4, 4, 16)` dtype `uint8`.
- Sidecar drop: no `.scale` keys in any `.self_attn.` path.
- **No full 162GB run** — BA Q2: shape-only + sub-checkpoint load is acceptable.

### AC3 — Tiny CSA regression (PASSED)
- Reuses `_tiny_csa_args/_tiny_weights` from `test_13_3b_1_dispatch_additive.py`.
- After importing remap module: `_attention_mlx` routed == `_csa_attention_mlx` direct.
- Ape transform: CSA → identity, indexer → identity.
- **Regression 0 introduced.** ✅

## 3. FROZEN / nn-port / Shim Byte-Intactness (PASSED)

| File | Staged diff? | Lines | SHA pin |
|---|---|---|---|
| `deepseek_v4.py` (FROZEN) | NO diff | 2577 | ✓ |
| `deepseek_v4_nn.py` (nn-port/ADR0025) | NO diff | 576 | ✓ |
| `shim_ds4_safetensors.py` (shim/BA Q1) | NO diff | 446 | ✓ |

`git diff --cached --stat` confirms ONLY 6 staged deliverables:
```
.cmux-status/coder.done                          |   1 +
agent-output/cmux-13-3b/coder-13-3b-4-notes.md   |  69 +++
scripts/remap_ds4_nn_weights.py                  | 465 +++++++++++
tests/test_remap_ds4_nn_weights_keyset.py        |  76 +++
tests/test_remap_ds4_nn_weights_load.py          | 198 ++++++
tests/test_remap_ds4_nn_weights_regression.py    |  33 ++
6 files changed, 842 insertions(+)
```

No edits to any FROZEN/nn-port/shim file. sha-pin cascade SLICE-INVARIANT: no-op. ✅

## 4. Ape-Verbatim Watch-Item (PASSED)

`grep -n 'transpose\|swapaxes\|\.T' scripts/remap_ds4_nn_weights.py`:
- **1 match** — line 15 comment only: *"copied verbatim in checkpoint token-major layout. No transpose is performed."*
- **0 transpose operations** on ape in production code.
- Ape rename patterns (lines 120, 124): simple rename, no reshape/transpose.
- All ape transforms confirmed `"identity"` in tests.
- **BA §0.4 satisfied: Coder copies ape verbatim, no .T.** ✅

## 5. Tracking Hygiene — HARD RULE (PASSED)

`git ls-files -- tests/test_remap_ds4_nn_weights_keyset.py tests/test_remap_ds4_nn_weights_load.py tests/test_remap_ds4_nn_weights_regression.py`:
```
tests/test_remap_ds4_nn_weights_keyset.py
tests/test_remap_ds4_nn_weights_load.py
tests/test_remap_ds4_nn_weights_regression.py
```
All 3 test files ARE tracked ✅

`git status --short` shows all 6 slice files as `A` (added to index):
```
A  .cmux-status/coder.done
A  agent-output/cmux-13-3b/coder-13-3b-4-notes.md
A  scripts/remap_ds4_nn_weights.py
A  tests/test_remap_ds4_nn_weights_keyset.py
A  tests/test_remap_ds4_nn_weights_load.py
A  tests/test_remap_ds4_nn_weights_regression.py
```
No untracked/unstaged test file left as baseline dependency. ✅

## 6. Regression — 0 introduced (PASSED)
Full suite: **0 RED**. All existing tests still pass. ✅

## Verdict: GREEN

All 6 deliverables staged, all acceptance criteria met, FROZEN/nn-port/shim byte-intact, ape verbatim confirmed, tracking hygiene clean, no regressions. Ready for supervisor commit.
