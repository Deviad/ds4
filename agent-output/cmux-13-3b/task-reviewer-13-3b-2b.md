# Story 13.3b-2b — Reviewer Task Brief (8-axis review + rope + HCA-byte-identical + sha-pin watch-items)

## Role
Reviewer (openai-codex/gpt-5.5 · xhigh, surface:84, fresh-context via /new, system-prompt `replace`). **Must NOT edit production code.**

## Context
- Coder 13.3b-2b complete: CSA attention wiring. Commit `aa93c5d` (parent `a2c20b0`).
- BA verdict: AC2 = GO (option α proven 13.3b-2a; block_bias scatter primitives present; KV-append/multi-head/grouped-o REUSE HCA). BA §0.2 INVERTED the original brief's ape-transpose premise: NO transpose at helper input (option α bypasses FROZEN helper → ape token-major `(rate=4, 2*head_dim=1024)` NO transpose; the `ape[:out_dim,:].T` lives ONLY inside the bypassed FROZEN helper).
- Architect design: §3.4 (unified `_attention_real_mlx` CSA+HCA) + §3.5 (RoPE parity) + ADR 0026.
- Coder claims: 3 new helpers `_csa_compressor_real_mlx:819` + `_csa_block_bias_mlx` + `_csa_attention_real_mlx:907`; chose NEW `_csa_attention_real_mlx` (NOT editing HCA `_attention_real_mlx` — keeps HCA byte-identical); rope option α; parity: compressor 5.96e-6, attention 6.48e-7, block_bias -inf mask EXACT + finite 0.0; full suite 1/565/13; FROZEN bodies byte-intact via AST; sha-pin cascade advanced.
- **Watch-items (supervisor-flagged)**:
  1. **HCA `_attention_real_mlx` byte-identical** (CRITICAL — Coder chose NEW `_csa_attention_real_mlx` helper, so the 13.3b-1 HCA `_attention_real_mlx` body MUST be byte-identical; the CSA path is NOT wired in via the dispatch yet — 13.3b-3 owns). Confirm NO HCA branch edit.
  2. **Rope option α + ape no-transpose** (BA §0.2 — the brief premise was INVERTED): confirm `_csa_compressor_real_mlx:819` consumes ape token-major `(rate=4, 2*head_dim=1024)` NO transpose + applies compress-yarn-tail TRAILING-64 via REUSED `_compress_rope_yarn_tail_tables_mlx:545` + `_apply_rope_tail_mlx:266`; FROZEN `_csa_windowed_compressor_mlx:347` body byte-identical (bypassed, NOT edited to add rope-suppress mode).
  3. **block_bias -inf mask EXACT + scatter correctness**: confirm `_csa_block_bias_mlx` produces `-inf` at sentinel positions (EXACT boolean equality with torch) + 0 at valid picks (finite_max_abs == 0.0); scatter from `_indexer_mlx` top-k correct.
  4. **sha-pin cascade SLICE-INVARIANT**: deepseek_v4.py edited additively → advance exactly the right count. Use rigorous whole-tree scan (Test Manager 13.3b-1/2a under-counted).

## Your job — 8-axis review (fresh context, adversarial)

### Axis 1 — Scope: additive ONLY, blast-radius honored
- `git show aa93c5d --stat` — confirm ONLY `deepseek_v4.py` + 2 NEW test files. NO nn port (13.3b-3), NO convert (13.3b-4), NO dispatch branch in `_attention_mlx` (13.3b-3 owns — CONFIRM no consumer of `_csa_attention_real_mlx` yet), NO GroupedLinear (13.3b-3).
- `git show aa93c5d -- deepseek_v4.py` — confirm ONLY additions (3 new helpers). NO body edits.

### Axis 2 — FROZEN bodies byte-intact (independent AST/source-hash, NOT git diff)
FROZEN must-not-edit-body list (BA §C + Architect §8): `_attention_real_mlx` HCA branch (CRITICAL — Coder chose NEW helper so HCA MUST be byte-identical; verify source/AST HEAD~1 `a2c20b0` vs HEAD `aa93c5d` sha16 identical), `_csa_windowed_compressor_mlx:347` (bypassed, byte-identical), `_hca_compressor_mlx:752`, `_indexer_mlx:620`, `_indexer_scorer_mlx:591`, `_compress_rope_yarn_tail_tables_mlx:545`, `_apply_rope_tail_mlx:266`, `_apply_rope_full_mlx:311`, `_rope_full_tables_mlx:300`, `_attention_mlx` cr=0 body, `_csa_attention_mlx:515`, `_csa_compressor_mlx:419`, `_csa_indexer_mlx:435`, `_csa_config_error:317`, `_require_csa_config:335`, `_causal_sliding_mask_mlx:914`, `_hyperconnection_mlx`, `_hyperhead_mlx`, `sanitize_weights`, parity `Model:2059`, 9 ADR-0017 forbidden.
- Independently extract each symbol's source from HEAD~1 + HEAD, compute sha16, confirm identical. CRITICAL: HCA `_attention_real_mlx` — if its body changed (Coder edited HCA to add a CSA branch instead of a NEW helper), that's an S-frozen-body violation → BLOCKED.

### Axis 3 — Rope option α + ape no-transpose (watch-item 2, CRITICAL — BA §0.2)
- Read `_csa_compressor_real_mlx:819` body. Confirm:
  - (a) Ca/Cb overlap pooling reimplemented INLINE (additive new code): `chunk_gate` from `new_kv` reshaped `[B, n_win, rate=4, 2*head_dim=1024]`; `chunk_gate += position_bias` (ape `(rate=4, 2*head_dim=1024)` token-major, **NO transpose** §0.2). Pool to `compressed_kv [B,1,n_win,head_dim=512]` via `expand_dims(compressed, 1)` mirror HCA `_hca_compressor_mlx:806`.
  - (b) RoPE applied via REUSED 13.3b-1 `_compress_rope_yarn_tail_tables_mlx:545` + `_apply_rope_tail_mlx:266`, TRAILING `qk_rope_head_dim=64` of `head_dim=512`. Positions `arange(n_win)*rate`.
  - (c) FROZEN `_csa_windowed_compressor_mlx:347` body byte-identical (cross-check Axis 2) — NOT edited to add a rope-suppress mode.
- If Coder edited FROZEN `_csa_windowed_compressor_mlx` to add rope-suppress mode → BLOCKED (S-frozen-body, ADR 0026 additive-only).

### Axis 4 — AC1 CSA compressor parity + anti-circularity
- `tests/test_csa_compressor_real_mlx_parity.py` — confirm torch `CSACompressor.forward:589-754` is the reference, from SAME synthesized weights (ADR 0007 §4).
- Confirm `_csa_compressor_real_mlx:819`: SINGLE-head (NOT multi-head — BA §0 correction), out_dim=head_dim=512, cr=4, Ca/Cb overlap, correct rope. Parity delta 5.96e-6 within tol.

### Axis 5 — AC2 CSA attention parity + block_bias -inf EXACT (watch-item 3)
- `tests/test_csa_attention_real_mlx_parity.py` — confirm torch `Attention.forward` CSA layer (modeling:801-875) is the reference.
- Confirm the AC asserts: (a) `-1` sentinel positions → `-inf` block_bias mask EXACT (boolean equality with torch); (b) valid pick positions → 0 (finite_max_abs == 0.0); (c) CSA attention output `[B,S,hidden]` matches torch within tol (6.48e-7).
- Confirm `_csa_attention_real_mlx:907`: compressor + block_bias from `_indexer_mlx` + KV-append (`cat([kv,compressed_kv],axis=2)`) + mask cat + multi-head (REUSE cr=0) + grouped-o (REUSE HCA step 6 pattern).

### Axis 6 — Standalone helper boundary + HCA byte-identical (watch-item 1)
- Confirm `_csa_compressor_real_mlx` + `_csa_block_bias_mlx` + `_csa_attention_real_mlx` are STANDALONE — NO dispatch branch in `_attention_mlx`, NO consumer wired (13.3b-3 owns dispatch + integration).
- **HCA path UNCHANGED**: `_attention_real_mlx` HCA branch body byte-identical to 13.3b-1 (cross-check Axis 2).
- Confirm `_csa_attention_real_mlx` NOT called anywhere in production except test (grep call sites).

### Axis 7 — sha-pin cascade SLICE-INVARIANT (watch-item 4)
- Rigorous WHOLE-TREE scan (don't trust grep subset, unlike Test Manager 13.3b-1/2a):
  - `grep -rn -E '0x[0-9a-f]{16}|sha16|deepseek_v4.*hash|hash.*deepseek_v4' tests/ scripts/finetune_ds4.py python-envs/mlx/src/ 2>/dev/null | grep -iv 'node_modules\|\.venv'`
  - Enumerate ALL pin sites referencing `deepseek_v4.py` content hash.
  - Confirm: (a) exactly the right count advanced; (b) new sha16 post-`aa93c5d` hashes; (c) no phantom; (d) no stale old hash in non-pycache.
- Adjudicate: legitimate cascade (additive 3 helpers shifted deepseek_v4.py content hash) vs phantom.

### Axis 8 — regression + no scope explosion
- 13.3b-1 (AC0-AC3) + 13.3b-2a (AC1-AC2) + 13.3a nn (12 incl backward AC) + 13.2 FP4 (9) + tiny CSA (11.14/11.15) all GREEN.
- Full suite: 1 RED (Test #8 carve-out, pre-existing) / 565 PASS / 13 SKIP / 0 introduced RED.
- No nn port edit, no convert, no dispatch branch, no GroupedLinear.

## Verdict
For each axis: GREEN / BLOCKED / NEEDS-INFO. If all 8 GREEN → `{"status":"ok","role":"Reviewer"}`. If any BLOCKED → `{"status":"blocked","axis":"<N>","reason":"<...>","role":"Reviewer"}` + `agent-output/cmux-13-3b/review-13-3b-2b.md` with block + remediation (Architect § clarification OR Coder r2).

## Deliverables
1. `agent-output/cmux-13-3b/review-13-3b-2b.md` — 8-axis verdict with watch-item adjudications.
2. `.cmux-status/reviewer.done` (`{"status":"ok","role":"Reviewer"}` or blocked JSON).
3. In-pane JSON in surface:84 ONLY.

## Pre-flight read
- `agent-output/cmux-13-3b/task-coder-13-3b-2b.md` (Coder spec, esp. BA corrections).
- `agent-output/cmux-13-3b/requirements-13-3b-2b.md` (§0/§0.1/§0.2/§A/§B/§C/§D).
- `agent-output/cmux-13-3b/architecture.md` §2.1/§3.4/§3.5/§8.
- `docs/adr/0026-csa-hca-indexer-mlx-real-port.md`.
- `git show aa93c5d` (the commit).
- FROZEN `deepseek_v4.py` (the 3 new helpers + check `_attention_real_mlx` HCA branch byte-identical + `_csa_windowed_compressor_mlx:347` byte-identical).
- The 2 new test files.

## Style
Caveman ultra default; byte-exact exempt. Use `ctx_execute`/`ctx_batch_execute` for AST/hash + git diff. Cite axis + AC + Q numbers. MUST NOT edit production code. BEGIN NOW.
