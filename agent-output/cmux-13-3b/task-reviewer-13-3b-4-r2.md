# Story 13.3b-4 — Reviewer r2 (8-axis review of Coder r2 staged work)

## Slice context
- **Story**: 13.3b-4 — real ckpt convert / key remap + load. Convert side ONLY.
- **Coder r2 done**: HEAD still `58194a9` (NO commit). AC2 test rewritten to load REAL safetensors payloads (17 `safe_open`/`get_tensor` hits, 0 `_zeros/_ones`, 1 `/Volumes` reference, test grew 198→268L).
- **Your r0 verdict**: BLOCKED Axis 5/AC2 — CORRECT (synthetic `_zeros/_ones`, no `safe_open`). r2 rewrote AC2. Re-review.
- **Commit gating**: Supervisor commits only AFTER BOTH you (Reviewer) AND Test Manager return GREEN on r2.

## What Coder r2 staged (review via `git diff --cached`)
```
scripts/remap_ds4_nn_weights.py              (465L, unchanged from r0)
tests/test_remap_ds4_nn_weights_keyset.py    (76L, AC1, unchanged)
tests/test_remap_ds4_nn_weights_load.py     (268L, AC2 — REWRITTEN r2: real payload)
tests/test_remap_ds4_nn_weights_regression.py (33L, AC3, unchanged)
agent-output/cmux-13-3b/coder-13-3b-4-notes.md
agent-output/cmux-13-3b/coder-13-3b-4-r2-notes.md (r2 notes)
.cmux-status/coder.done
```

## 8-axis review (r2 focus: Axis 5/AC2 was BLOCKED — verify the fix)

### Axis 1 — Scope / blast radius
- ONLY convert-side files staged. NO edits to `deepseek_v4.py`, `deepseek_v4_nn.py`, `shim_ds4_safetensors.py`.
- `git diff --cached --stat` confirms 7-file scope.

### Axis 2 — FROZEN / nn-port / shim byte-intactness
- Source-hash + AST at HEAD `58194a9` byte-identical for all protected bodies (`_csa_*`, `_hca_*`, `_indexer_*`, `_attention_mlx`, `sanitize_weights`, parity `Model`, dequant, 9 forbidden, `_apply_rope_*`, `_rope_*_tables`, `_compress_rope_yarn_tail_tables_mlx`, all 13.3b-1/2a/2b helpers).
- nn-port `deepseek_v4_nn.py` byte-identical (ADR 0025).
- `shim_ds4_safetensors.py` byte-identical (BA Q1).

### Axis 3 — Correctness / architecture
- `scripts/remap_ds4_nn_weights.py` unchanged from r0 (you r0-approved the remap mechanics: 1460 expected keys == 1460 remapped, 0 missing/extra). r2 should NOT have touched the script. Verify via `git diff --cached` — script should be identical to r0 staging.

### Axis 4 — ape-verbatim (BA §0.4)
- r0 was GREEN (no `.T`/transpose/swapaxes on ape). r2 shouldn't regress. Re-grep `scripts/remap_ds4_nn_weights.py`.

### Axis 5 — AC2 legitimacy (r0 BLOCKED — CRITICAL re-review)
**r0 finding**: AC2 used synthetic `_tiny_mixed_config()` + `_zeros/_ones` — no `safe_open`, no real payload, tautology-adjacent.

**r2 verification — must confirm ALL of:**
- `grep -c -E 'safe_open|load_file|get_tensor|safetensors' tests/test_remap_ds4_nn_weights_load.py` → > 0 (real payload).
- `grep -c -E '_zeros|_ones|mx\.zeros|mx\.ones' tests/test_remap_ds4_nn_weights_load.py` → 0 for WEIGHT tensors (input_ids as integers OK).
- `grep 'Data NVME/mlx-ft/ds4/hf-f8shim' tests/test_remap_ds4_nn_weights_load.py` → references real ckpt.
- Test loads 3 real layers (1 sliding L0 + 1 CSA L2 + 1 HCA L3) via lazy `get_tensor` (NOT full 162GB).
- `model.load_weights(..., strict=True)` succeeds.
- Forward at seq_len ≥ 8 → finite output, correct shape `(1, seq_len, vocab_size)`.
- Real FP4 expert stacking verified (uint8 dtype, real packed bytes NOT zeros, shape `(256, ...)`).
- Stays within 16GB RAM (lazy `get_tensor`).

**If all present → Axis 5 GREEN. If still synthetic or partial → BLOCKED again (Coder r3 or STOP).**

### Axis 6 — Tracking-hygiene HARD RULE (AGENTS.md `58194a9`)
- `git ls-files` for `tests/test_remap_ds4_nn_weights_*.py` — all tracked (staged `A`, NOT `??`).
- `git status --short` shows `A` (added).
- Coder did NOT run `git commit` (HEAD still `58194a9`).

### Axis 7 — sha-pin cascade SLICE-INVARIANT
13.3b-4 should NOT touch `deepseek_v4.py` or nn-port, so no-op. Verify no phantom/stale pin sites.

### Axis 8 — Regression / no scope explosion
- Full suite: 572 / 13 / 0 RED expected.
- No new dependencies, no C++, no scope creep.

## Deliverables
1. `agent-output/cmux-13-3b/review-13-3b-4-r2.md` — 8-axis verdict with evidence per axis (Axis 5 is the focus — show the `safe_open` grep results).
2. `.cmux-status/reviewer.done` marker.
3. In-pane JSON: `{"status":"ok","role":"Reviewer"}` in surface:84 ONLY.

## STOP-ESCALATE
Write `agent-output/cmux-13-3b/reviewer-13-3b-4-r2-blocked.md` if:
- Axis 5/AC2 still synthetic (no `safe_open`, weights still `_zeros/_ones`).
- AC2 real load exposes a remap BUG (forward NaN, shape mismatch, strict load fails) — genuine remap correctness issue, Coder r3 or STOP.
- Any other axis RED (FROZEN edited, ape transposed, tracking violation, Coder committed anyway).

## Pre-flight read
1. `agent-output/cmux-13-3b/coder-13-3b-4-r2-notes.md` (Coder r2 summary).
2. `agent-output/cmux-13-3b/review-13-3b-4.md` (your r0 BLOCKED verdict).
3. `agent-output/cmux-13-3b/requirements-13-3b-4.md` (BA SPEC + §0.4).
4. `agent-output/cmux-13-3b/task-coder-13-3b-4-r2.md` (the r2 brief).
5. `AGENTS.md` "Tracking hygiene for test files participating in the baseline (HARD RULE)" section.

## Build
`source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first). Caveman ultra default; byte-exact exempt. BEGIN NOW. Echo `{"status":"ok","role":"Reviewer"}` in this pane only.
