# Story 13.3b-4 — Test Manager r2 (validate Coder r2 staged work)

## Slice context
- **Story**: 13.3b-4 — real ckpt convert / key remap + load. Convert side ONLY.
- **Coder r2 done**: HEAD still `58194a9` (NO commit). AC2 test rewritten to load REAL safetensors payloads. r0 was BLOCKED by Reviewer for synthetic `_zeros/_ones`; r2 fixed it (17 `safe_open`/`get_tensor` hits, 0 `_zeros/_ones`, 1 `/Volumes` reference).
- **Your r0 methodology gap**: you said AC2 PASSED but didn't check `safe_open` was actually called — Reviewer caught it. **r2 must explicitly verify real payload load, not trust "load strict + forward finite".**

## What Coder r2 staged (validate via `git diff --cached`)
```
scripts/remap_ds4_nn_weights.py              (465L, unchanged from r0)
tests/test_remap_ds4_nn_weights_keyset.py    (76L, AC1, unchanged)
tests/test_remap_ds4_nn_weights_load.py     (268L, AC2 — REWRITTEN r2: real payload)
tests/test_remap_ds4_nn_weights_regression.py (33L, AC3, unchanged)
agent-output/cmux-13-3b/coder-13-3b-4-notes.md (r0 notes)
agent-output/cmux-13-3b/coder-13-3b-4-r2-notes.md (r2 notes)
.cmux-status/coder.done
```

## Your job — independent validation (r0 methodology gap FIXED)

### 1. Run the slice's 3 NEW tests + full suite
```bash
unset SSLKEYLOGFILE
source python-envs/mlx/.venv/bin/activate
python3 -m pytest tests/test_remap_ds4_nn_weights_keyset.py tests/test_remap_ds4_nn_weights_load.py tests/test_remap_ds4_nn_weights_regression.py -v
python3 -m pytest tests/ --no-header -q  # full suite — expect 572 / 13 / 0 RED
```

### 2. AC2 verification (r0 FAILED this — be rigorous)
**MUST explicitly check the AC2 test source, not just "it passes":**
- `grep -c -E 'safe_open|load_file|get_tensor|safetensors' tests/test_remap_ds4_nn_weights_load.py` → must be > 0 (real payload load).
- `grep -c -E '_zeros|_ones|mx\.zeros|mx\.ones' tests/test_remap_ds4_nn_weights_load.py` → must be 0 for WEIGHT tensors (input_ids as integers OK; if any weight is `_zeros`/`_ones` → BLOCKED).
- `grep 'Data NVME/mlx-ft/ds4/hf-f8shim' tests/test_remap_ds4_nn_weights_load.py` → must reference the real ckpt path.
- Confirm the test loads 3 real layers (1 sliding + 1 CSA + 1 HCA) via lazy `get_tensor` (NOT full 162GB materialization).
- `model.load_weights(..., strict=True)` succeeds.
- Forward at seq_len ≥ 8 → finite output, correct shape `(1, seq_len, vocab_size)`.
- Real FP4 expert stacking verified (uint8 dtype, real packed bytes NOT zeros, shape `(256, ...)`).
- Stays within 16GB RAM (lazy `get_tensor`).

**DO NOT just trust "test passed" — read the test body.** r0 failure was trusting the pass verdict.

### 3. AC1 + AC3 (unchanged from r0, re-verify)
- **AC1** (`test_remap_ds4_nn_weights_keyset.py`): expected key-set probed from `model.parameters()` actually probed (anti-circular). Header-only on real ckpt. Should still pass.
- **AC3** (`test_remap_ds4_nn_weights_regression.py`): tiny CSA byte-identical. Should still pass.

### 4. FROZEN / nn-port / shim byte-intactness
- `git diff --cached --stat` — confirm ONLY the 7 staged deliverables. NO edits to `deepseek_v4.py`, `deepseek_v4_nn.py`, `shim_ds4_safetensors.py`.
- Source-hash at HEAD `58194a9` byte-identical.
- sha-pin cascade SLICE-INVARIANT: 13.3b-4 should NOT touch `deepseek_v4.py` or nn-port, so no-op. Verify no phantom/stale pin sites.

### 5. ape-verbatim watch-item (BA §0.4)
- `grep -n -E 'transpose|swapaxes|\.T' scripts/remap_ds4_nn_weights.py` — should be ABSENT on ape tensors (r0 was clean; r2 shouldn't regress).

### 6. Tracking-hygiene HARD RULE (AGENTS.md `58194a9`)
- `git ls-files` for `tests/test_remap_ds4_nn_weights_*.py` — all tracked (staged `A`, NOT `??`).
- Coder did NOT run `git commit` (HEAD still `58194a9`).

### 7. Regression: 0 introduced RED
Full suite 572 / 13 / 0 RED.

## Deliverables
1. `agent-output/cmux-13-3b/test-manager-13-3b-4-r2-report.md` — full validation report (test runs + AC1-AC3 with explicit `safe_open` check for AC2 + byte-intactness + ape-verbatim + tracking + regression).
2. `.cmux-status/test-manager.done` marker.
3. In-pane JSON: `{"status":"ok","role":"Test Manager"}` in surface:85 ONLY.

## STOP-ESCALATE
Write `agent-output/cmux-13-3b/test-manager-13-3b-4-r2-stop.md` if:
- AC2 still synthetic (no `safe_open`, or weights still `_zeros/_ones`).
- AC2 real load EXPOSES a remap BUG (forward NaN, shape mismatch, strict load fails) — genuine remap correctness issue.
- FROZEN/nn-port/shim byte-intactness violated.
- ape TRANSPOSED (BA §0.4 violated).
- Test file untracked/unstaged (HARD RULE violation).
- Coder committed anyway (HEAD advanced past `58194a9`).

## Pre-flight read
1. `agent-output/cmux-13-3b/coder-13-3b-4-r2-notes.md` (Coder r2 summary — what was wrong in r0, the real-payload approach, any remap bugs found).
2. `agent-output/cmux-13-3b/review-13-3b-4.md` (Reviewer r0 BLOCKED verdict — your r2 must satisfy Axis 5).
3. `agent-output/cmux-13-3b/test-manager-13-3b-4-report.md` (your r0 — note the methodology gap, don't repeat it).
4. `agent-output/cmux-13-3b/requirements-13-3b-4.md` (BA SPEC).
5. `AGENTS.md` "Tracking hygiene for test files participating in the baseline (HARD RULE)" section.

## Build
`source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first). The sub-checkpoint build takes a few minutes (real I/O on 46 shards for 3 layers). Caveman ultra default; byte-exact exempt. BEGIN NOW. Echo `{"status":"ok","role":"Test Manager"}` in this pane only.
