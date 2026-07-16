# Story 13.3b-2a — Reviewer Task Brief (8-axis review + rope-landmine + sha-pin watch-items)

## Role
Reviewer (openai-codex/gpt-5.5 · xhigh, surface:84, fresh-context via /new, system-prompt `replace`). **Must NOT edit production code.**

## Context
- Coder 13.3b-2a complete: Indexer real-dim MLX port. Commit `a2c20b0` (parent `1f5d8e4`).
- BA verdict: AC2 = GO (top-k solvable set-based). Architect design: §3.2/§3.3 of `agent-output/cmux-13-3b/architecture.md` + ADR 0026.
- Coder claims: 2 new helpers (`_indexer_scorer_mlx:591`, `_indexer_mlx:620`); 2 new tests GREEN; rope landmine resolved (reimplemented Ca/Cb overlap inline + reused 13.3b-1 rope helper; FROZEN `_csa_windowed_compressor_mlx:347` byte-identical); full suite 1/563/13 (Test #8 carve-out), 0 introduced RED; FROZEN bodies byte-intact via AST; sha-pin cascade advanced.
- **Watch-items (supervisor-flagged)**:
  1. **Rope-landmine resolution**: confirm `_indexer_mlx` reimplements Ca/Cb overlap pooling inline (additive new code in `_indexer_mlx` body, NOT a FROZEN body edit) + applies compress-yarn-tail via REUSED 13.3b-1 `_compress_rope_yarn_tail_tables_mlx:545` + `_apply_rope_tail_mlx`. FROZEN `_csa_windowed_compressor_mlx:347` body MUST be byte-identical (the whole point of the landmine — Coder routed AROUND, not through).
  2. **AC2 set-based assertion**: confirm `test_indexer_mlx_parity.py` asserts SET of non-sentinel picks + scores, NOT exact index order at ties (BA Q6/Q4 — MLX argsort tie order ≠ torch). If the test asserts exact tie order → BLOCKED (flaky/incorrect test).
  3. **sha-pin cascade SLICE-INVARIANT**: deepseek_v4.py edited additively → advance exactly the right count of sha16 pin sites. (Test Manager 13.3b-1 under-counted "2 sites" vs your "3 sites" — for 13.3b-2a, do the rigorous whole-tree scan you did in 13.3b-1, don't trust grep subset.)

## Your job — 8-axis review (fresh context, adversarial)

### Axis 1 — Scope: additive ONLY, blast-radius honored
- `git show a2c20b0 --stat` — confirm ONLY `deepseek_v4.py` + 2 NEW test files changed. NO nn port edit (13.3b-3), NO convert (13.3b-4), NO CSA attention wiring (13.3b-2b — confirm NO dispatch branch added, NO consumer of `_indexer_mlx` yet), NO other production files.
- `git show a2c20b0 -- deepseek_v4.py` — confirm ONLY additions (2 new helpers in 13.3b module section). NO body edits. NO dispatch branch.

### Axis 2 — FROZEN bodies byte-intact (independent AST/source-hash proof, NOT git diff)
FROZEN must-not-edit-body list: `_csa_windowed_compressor_mlx:347` (CRITICAL — the reused indexer compressor; Coder claims byte-identical, verify), `_csa_compressor_mlx`, `_csa_indexer_mlx:435` (tiny), `_csa_attention_mlx`, `_csa_config_error`, `_require_csa_config`, `_attention_mlx`, `_hyperconnection_mlx`, `_hyperhead_mlx`, `_hca_compressor_mlx`, `_attention_real_mlx`, `_compress_rope_yarn_tail_tables_mlx`, `sanitize_weights`, parity `Model`, FP4/i8 dequant, `_apply_rope_full_mlx`, `_apply_rope_tail_mlx`, `_rope_full_tables_mlx`, `_rope_tail_tables_mlx`, 9 ADR-0017 forbidden.
- Independently extract each symbol's source from HEAD~1 (`1f5d8e4`) and HEAD (`a2c20b0`), compute sha16, confirm identical. Use `ctx_execute`. CRITICAL: `_csa_windowed_compressor_mlx:347` — if its body changed (Coder edited it to add a rope-suppress mode instead of routing around), that's an S-frozen-body violation → BLOCKED.

### Axis 3 — Rope-landmine resolution (watch-item 1, CRITICAL)
- Read `_indexer_mlx:620` body. Confirm:
  - (a) Ca/Cb overlap pooling reimplemented INLINE in `_indexer_mlx` (additive new code): `new_kv[:,:,:rate] = chunk_kv[...,:head_dim]` (Ca=leading 128 of 256), `new_kv[:,:,rate:] = chunk_kv[...,head_dim:]` (Cb=trailing 128) per BA Q1. NOT a call to FROZEN `_csa_windowed_compressor_mlx` (or if it calls it, only in a no-rope mode — but Coder claims no suppress mode exists, so should be inline reimpl).
  - (b) RoPE applied via REUSED 13.3b-1 `_compress_rope_yarn_tail_tables_mlx:545` + `_apply_rope_tail_mlx`, TRAILING `qk_rope_head_dim=64` of `index_head_dim=128`. Positions: `arange(n_win)*rate` for compressed_kv, `position_ids` for q.
  - (c) FROZEN `_csa_windowed_compressor_mlx:347` body byte-identical (cross-check Axis 2).
- If Coder instead edited FROZEN `_csa_windowed_compressor_mlx` to add a rope-suppress mode → BLOCKED (S-frozen-body violation, ADR 0026 additive-only).

### Axis 4 — AC1 IndexerScorer parity + anti-circularity
- `tests/test_indexer_scorer_mlx_parity.py` — confirm torch `DeepseekV4IndexerScorer.forward:455` is the reference, from SAME synthesized weights (ADR 0007 §4).
- Confirm `_indexer_scorer_mlx:591`: `scores = relu(q @ ckv.T) * index_head_dim**-0.5` `[B,S,H,T]`; `weights = (hidden @ weights_proj.T) * index_n_heads**-0.5` `[B,S,H]`; `out = sum(scores * weights[...,None], axis=2)` `[B,S,T]`; fp32 accum; return fp32. BA Q3.
- Confirm `weights_proj` shape correct (verify against real ckpt key `attn.indexer.weights_proj.weight` if cross-checked, or synthesized at correct shape).

### Axis 5 — AC2 Indexer parity: set-based assertion (watch-item 2)
- `tests/test_indexer_mlx_parity.py` — confirm torch `DeepseekV4Indexer.forward:511` (esp 567-586) is the reference.
- Confirm the AC asserts: (a) `-1` sentinel mask EXACT (positions of sentinels match torch); (b) SET of non-sentinel valid picks == torch set per query; (c) scores at valid picks == torch within tol; (d) **tie order NOT asserted** (BA Q6/Q4). If the test asserts exact index order at ties → BLOCKED (flaky, MLX argsort ≠ torch topk tie order).
- Confirm `_indexer_mlx:620`: `causal_threshold=(position_ids+1)//rate`; future mask→`-inf`; `top_k=min(index_topk=512,T)`; `top_k_indices=mx.argsort(-scores)[...,:top_k]` DESC; `-1` sentinel `where(invalid, -1, top_k_indices)`. BA Q4.

### Axis 6 — Standalone helper boundary
- Confirm `_indexer_scorer_mlx` + `_indexer_mlx` are STANDALONE — NO dispatch branch added in `_attention_mlx`, NO consumer wired into `_attention_real_mlx` (13.3b-2b owns). HCA path UNCHANGED (the 13.3b-1 HCA branch has no indexer; confirm `_attention_real_mlx` HCA branch body byte-identical to 13.3b-1).
- Confirm `_indexer_mlx` NOT called anywhere in production code except the test (grep `git show a2c20b0` for call sites).

### Axis 7 — sha-pin cascade SLICE-INVARIANT (watch-item 3)
- Do the rigorous WHOLE-TREE scan (don't trust grep subset, unlike Test Manager 13.3b-1):
  - `grep -rn -E '0x[0-9a-f]{16}|sha16|deepseek_v4.*hash|hash.*deepseek_v4' tests/ scripts/finetune_ds4.py python-envs/mlx/src/ 2>/dev/null | grep -iv 'node_modules\|\.venv'`
  - Enumerate ALL pin sites that reference `deepseek_v4.py` content hash.
  - Confirm: (a) exactly the right count advanced (Coder claims "right count" — compute the actual count); (b) new sha16 values are post-`a2c20b0` hashes; (c) no phantom (no pin updated that didn't need to change); (d) no stale old hash remains in non-pycache source.
- Adjudicate: legitimate cascade (additive 2 helpers shifted deepseek_v4.py content hash → cascade required) vs phantom.

### Axis 8 — regression + no scope explosion
- 13.3b-1 (AC0-AC3) + 13.3a nn (12 incl 13.3a-3 backward AC) + 13.2 FP4 (9) + tiny CSA (11.14/11.15) all GREEN.
- Full suite: 1 RED (Test #8 carve-out, pre-existing) / 563 PASS / 13 SKIP / 0 introduced RED.
- No nn port edit, no convert, no CSA attention wiring, no dispatch branch.

## Verdict
For each axis: GREEN / BLOCKED / NEEDS-INFO. If all 8 GREEN → `{"status":"ok","role":"Reviewer"}`. If any BLOCKED → `{"status":"blocked","axis":"<N>","reason":"<...>","role":"Reviewer"}` + `agent-output/cmux-13-3b/review-13-3b-2a.md` with block + remediation (Architect § clarification OR Coder r2).

## Deliverables
1. `agent-output/cmux-13-3b/review-13-3b-2a.md` — 8-axis verdict with watch-item adjudications.
2. `.cmux-status/reviewer.done` (`{"status":"ok","role":"Reviewer"}` or blocked JSON).
3. In-pane JSON in surface:84 ONLY.

## Pre-flight read
- `agent-output/cmux-13-3b/task-coder-13-3b-2a.md` (Coder's spec, esp. the rope-landmine front-load).
- `agent-output/cmux-13-3b/requirements-13-3b-2a.md` (BA §0 GO + §A Q1-Q7 + §B AC1-AC2).
- `agent-output/cmux-13-3b/architecture.md` §2.1/§3.2/§3.3/§8.
- `docs/adr/0026-csa-hca-indexer-mlx-real-port.md`.
- `git show a2c20b0` (the commit).
- FROZEN `deepseek_v4.py` (the 2 new helpers + check `_csa_windowed_compressor_mlx:347` body).
- The 2 new test files.

## Style
Caveman ultra default; byte-exact exempt. Use `ctx_execute`/`ctx_batch_execute` for AST/hash + git diff. Cite axis + AC + Q numbers. MUST NOT edit production code. BEGIN NOW.
