# Story 13.3b-4 — Reviewer (8-axis review of Coder 13.3b-4 staged work)

## Slice context
- **Story**: 13.3b-4 — real ckpt convert / key remap + load. Convert side ONLY.
- **Coder done**: HEAD still `58194a9` (NO commit per commit-gating policy). All 6 deliverables `git add`-staged. You review the STAGED work via `git diff --cached`. Coder MUST NOT have committed.
- **Commit gating**: Supervisor commits only AFTER BOTH you (Reviewer) AND Test Manager return GREEN. Your job = 8-axis review of staged diff, write `review-13-3b-4.md`, emit verdict.

## What Coder staged (review via `git diff --cached`)
```
scripts/remap_ds4_nn_weights.py              (465 lines, NEW — the remap script)
tests/test_remap_ds4_nn_weights_keyset.py    (76L, NEW — AC1)
tests/test_remap_ds4_nn_weights_load.py     (198L, NEW — AC2)
tests/test_remap_ds4_nn_weights_regression.py (33L, NEW — AC3)
agent-output/cmux-13-3b/coder-13-3b-4-notes.md (69L)
.cmux-status/coder.done
```

## 8-axis review

### Axis 1 — Scope / blast radius
- ONLY convert-side files staged. NO edits to `deepseek_v4.py` (FROZEN bodies), `deepseek_v4_nn.py` (nn-port, ADR 0025), `shim_ds4_safetensors.py` (BA Q1 chose NEW script).
- `git diff --cached --stat` confirms the 6-file scope. Flag any unexpected file.

### Axis 2 — FROZEN / nn-port / shim byte-intactness
- Source-hash + AST extraction: FROZEN bodies (`_csa_*`, `_hca_*`, `_indexer_*`, `_attention_mlx`, `sanitize_weights`, parity `Model`, dequant, 9 forbidden OUR-Python symbols, `_apply_rope_*`, `_rope_*_tables`, `_compress_rope_yarn_tail_tables_mlx`, all 13.3b-1/2a/2b helper bodies) byte-identical at HEAD `58194a9`.
- nn-port `deepseek_v4_nn.py` byte-identical (ADR 0025 — 13.3b-4 is convert-side ONLY).
- `shim_ds4_safetensors.py` byte-identical (BA Q1 — NEW script, not shim extension).

### Axis 3 — Correctness / architecture
- `scripts/remap_ds4_nn_weights.py` implements Architect §5.1 rename table correctly:
  - `embed.weight→model.embed_tokens.weight`, `head.weight→lm_head.weight`
  - `layers.N.attn.*→model.layers.N.self_attn.*` (wq_a/wq_b/wkv/wo_a/wo_b/q_norm/kv_norm/sinks/compressor.*/indexer.*)
  - `layers.N.ffn.*→model.layers.N.mlp.*` (gate/shared_experts w1=gate/w3=up/w2=down/per-expert→stacked)
  - `hc_attn_*/hc_ffn_*→13.3a leaf names`, `mtp.*→DROP`
- BF16 `.scale` sidecars dropped (attn core + indexer.wq_b); routed experts STAY FP4 (stacked uint8 weight + BF16 scale, `_dequantize_fp4` at forward ADR 0024).
- `model.` prefix applied.
- per-expert→stacked: shape `(n_routed, ...)` correct.

### Axis 4 — ape-verbatim (BA §0.4 — CRITICAL)
BA caught Architect §5.2 prose STALE. The wired real helpers consume ape token-major (NO transpose):
- `_csa_compressor_real_mlx:819` → `ape = (rate, 2*out_dim)` token-major
- `_indexer_mlx:653` → `indexer ape = (rate, 2*out_dim)` token-major
- `_hca_compressor_mlx:773` → `ape = (rate, out_dim)` token-major

ckpt headers IDENTICAL to nn leaves. **Coder MUST copy ape verbatim (no `.T`).** Verify in `scripts/remap_ds4_nn_weights.py`: grep for `.T`, `transpose`, `swapaxes` on ape tensors — should be ABSENT, or only on non-ape tensors with clear justification. If ape is TRANSPOSED → BLOCKED, Coder r2.

### Axis 5 — AC1/AC2/AC3 test legitimacy (NOT tautology)
- **AC1** (`test_remap_ds4_nn_weights_keyset.py`): expected key-set comes from `model.parameters()` actually PROBED on the nn module (anti-circular), NOT a hand-written list. If the expected set is hand-written/hard-coded, the test could pass on a wrong remap by accident → flag as tautology.
- **AC2** (`test_remap_ds4_nn_weights_load.py`): real sub-checkpoint load + forward finite (not just "load returns True"). Shape-only on full ckpt is OK (BA Q2).
- **AC3** (`test_remap_ds4_nn_weights_regression.py`): tiny CSA 11.14/11.15 byte-identical pre/post remap (remap is real-ckpt-only, must not touch tiny path).

### Axis 6 — Tracking-hygiene HARD RULE (AGENTS.md `58194a9`)
This rule was added after 13.3b-3's chain-of-custody break (Coder modified untracked `test_deepseek_v4_mlx_port.py` and left it uncommitted). For 13.3b-4:
- Run `git ls-files` for `tests/test_remap_ds4_nn_weights_*.py` — all MUST be tracked (staged = will-be-tracked). If untracked/unstaged → BLOCKED (HARD RULE violation).
- Confirm `git status --short` shows them as `A` (added), NOT `??` (untracked).
- Confirm Coder did NOT run `git commit` (HEAD still `58194a9`).

### Axis 7 — sha-pin cascade SLICE-INVARIANT
13.3b-4 should NOT touch `deepseek_v4.py` or nn-port, so the sha-pin cascade should be a NO-OP. Verify no phantom/stale pin sites, no unexpected sha16 changes in `finetune_ds4.py` or test files.

### Axis 8 — Regression / no scope explosion
- Full suite: 569+3 NEW / 13 skip / 0 RED expected.
- No new dependencies, no C++, no scope creep into smoke-train (13.3b-5) or other slices.

## Deliverables
1. `agent-output/cmux-13-3b/review-13-3b-4.md` — 8-axis verdict with evidence per axis.
2. `.cmux-status/reviewer.done` marker.
3. In-pane JSON: `{"status":"ok","role":"Reviewer"}` in surface:84 ONLY.

## STOP-ESCALATE
Write `agent-output/cmux-13-3b/reviewer-13-3b-4-blocked.md` if:
- Any axis is RED/BLOCKED (ape transposed, FROZEN edited, AC tautology, tracking violation, etc.).
- Coder committed anyway (HEAD advanced past `58194a9`) — commit-gating violation.

## Pre-flight read
1. `agent-output/cmux-13-3b/coder-13-3b-4-notes.md` (Coder's own summary).
2. `agent-output/cmux-13-3b/requirements-13-3b-4.md` (BA SPEC + §0.4 ape correction + Q1-Q4).
3. `agent-output/cmux-13-3b/task-coder-13-3b-4.md` (the brief Coder worked from).
4. `agent-output/cmux-13-3b/architecture.md` §5.1-§5.4 + §0 nn tree (NOTE §5.2 ape-transpose prose STALE — follow wired helpers + nn tree).
5. `AGENTS.md` "Tracking hygiene for test files participating in the baseline (HARD RULE)" section (committed `58194a9`).
6. `docs/adr/0024-*.md` (FP4 dequant — routed experts STAY FP4) + `docs/adr/0025-*.md` (nn file ownership — DO NOT edit nn-port).

## Build
`source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first). Use `ctx_execute` for AST/source-hash extraction. Caveman ultra default; byte-exact exempt. BEGIN NOW. Echo `{"status":"ok","role":"Reviewer"}` in this pane only.
