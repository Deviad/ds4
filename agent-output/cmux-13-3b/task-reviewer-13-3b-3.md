# Story 13.3b-3 — Reviewer Task Brief (8-axis review + dispatch additive + GroupedLinear-absent + §4.1 nn wiring + sha-pin)

## Role
Reviewer (openai-codex/gpt-5.5 · xhigh, surface:84, fresh-context via /new, system-prompt `replace`). **Must NOT edit production code.**

## Context
- Coder 13.3b-3 complete: INTEGRATION milestone. Commit `d1f1488` (parent `aa93c5d`).
- BA verdict (§0): GO. Discriminator clean (`_csa_config_error` already distinguishes tiny from real via heads/o_groups guard). GroupedLinear NOT needed (Q2: inlined in real branches already). AttentionNN guard lift = full §4.1 nn wiring (Q3). Integration AC via route-equivalence (Q4).
- Architect §4.1 (AttentionNN submodule layout) + §4.2 (dispatch additive) + §4.3 (GroupedLinear not needed) + ADR 0026 §Decision-1 (sanctioned dispatch edit site).
- Coder claims: dispatch branch `if cr==4: return _csa_attention_real_mlx(...)` inserted; tiny + cr=128 byte-identical; cr=0 body byte-identical; AttentionNN guard lifted + `AttentionCompressorNN`/`AttentionIndexerNN` submodules per §4.1; `__call__` delegates to `_attention_mlx`; NO `_grouped_linear_mlx`; full suite 569 passed/13 skipped/0 RED (Test #8 now passes); FROZEN byte-intact; sha-pin cascade advanced.
- **Watch-items (supervisor-flagged)**:
  1. **Dispatch additive** (CRITICAL — tiny + cr=128 + cr=0 byte-identical; only the new cr==4 branch added). ADR 0026 §Decision-1 sanctioned site.
  2. **GroupedLinear NOT introduced** (Q2 — confirm `_grouped_linear_mlx` ABSENT; extracting would edit FROZEN cr=0 body).
  3. **AttentionNN §4.1 wiring** (guard lifted + submodules + `__call__` delegation + cr=0 sliding path UNCHANGED).
  4. **sha-pin cascade** (rigorous whole-tree scan — Test Manager 13.3b-1/2a under-counted).

## Your job — 8-axis review (fresh context, adversarial)

### Axis 1 — Scope: additive ONLY, blast-radius honored
- `git show d1f1488 --stat` — confirm ONLY `deepseek_v4.py` (additive dispatch) + `deepseek_v4_nn.py` (guard lift + §4.1 wiring) + extended/NEW test files. NO convert (13.3b-4), NO new real-branch helpers (13.3b-1/2a/2b shipped; 13.3b-3 only WIRES).
- `git show d1f1488 -- deepseek_v4.py` — confirm ONLY the new `if cr==4: return _csa_attention_real_mlx(...)` branch added.

### Axis 2 — FROZEN bodies byte-intact (independent AST/source-hash, NOT git diff)
FROZEN must-not-edit-body list: cr=0 body `_attention_mlx:1203+` (CRITICAL — the dispatch edit is at `:1199-1204`, the cr=0 body must be byte-identical), `_attention_real_mlx:1005` (HCA, 13.3b-1), `_csa_attention_real_mlx:907` (CSA, 13.3b-2b), `_csa_attention_mlx:515` (tiny), `_csa_config_error:317`, `_indexer_mlx:620`, `_indexer_scorer_mlx:591`, `_csa_compressor_real_mlx:819`, `_csa_block_bias_mlx:887`, `_hca_compressor_mlx`, `_compress_rope_yarn_tail_tables_mlx:545`, `_apply_rope_*`, `_rope_*_tables_mlx`, `_hyperconnection_mlx`, `_hyperhead_mlx`, `_causal_sliding_mask_mlx`, `sanitize_weights`, parity `Model`, FP4/i8 dequant, 9 ADR-0017 forbidden.
- Independently extract each symbol's source from HEAD~1 `aa93c5d` + HEAD `d1f1488`, compute sha16, confirm identical. CRITICAL: cr=0 body — if its body changed, that's an S-frozen-body violation → BLOCKED.

### Axis 3 — Dispatch additive (watch-item 1, CRITICAL — ADR 0026 §Decision-1)
- Read `_attention_mlx:1199-1206`. Confirm the EXACT shape:
  ```python
  if args.compression_ratio != 0:
      if _csa_config_error(args) is None:          # tiny — UNCHANGED byte-identical
          return _csa_attention_mlx(args, x, weights, index_topk=index_topk)
      if args.compression_ratio == 4:              # NEW real cr=4 — CSA
          return _csa_attention_real_mlx(args, x, weights, index_topk=index_topk)
      return _attention_real_mlx(args, x, weights, index_topk=index_topk)   # cr=128 HCA byte-identical
  ```
- Tiny path byte-identical, cr=128 HCA fallback byte-identical, cr=0 body byte-identical (cross-check Axis 2).
- If Coder edited the tiny path or cr=128 fallback or cr=0 body → BLOCKED (S-frozen-body, ADR 0026 additive-only).

### Axis 4 — GroupedLinear NOT introduced (watch-item 2, Q2)
- `grep -n '_grouped_linear_mlx' deepseek_v4.py` — confirm ABSENT.
- Confirm grouped-o loop still inline at `_attention_mlx:1247-1256` (FROZEN cr=0 body, byte-identical) AND inlined in `_attention_real_mlx` + `_csa_attention_real_mlx` (reused, not extracted).
- If `_grouped_linear_mlx` introduced → flag (would require FROZEN cr=0 body edit → BLOCKED).

### Axis 5 — AttentionNN §4.1 wiring (watch-item 3, Q3) + AC4
- Read `deepseek_v4_nn.py` `AttentionNN:135+`. Confirm:
  - (a) guard `:153-154` (`compression_ratio != 0 → NotImplementedError`) LIFTED (no longer raises).
  - (b) NEW `AttentionCompressorNN` + `AttentionIndexerNN` submodules per §4.1: CSA leaves (compressor.{wkv,wgate,ape,norm.weight} + indexer.compressor.{...} + indexer.weights_proj + indexer.wq_b); HCA leaves (compressor only).
  - (c) `__call__`: if sliding → existing `_attention_mlx` cr=0 UNCHANGED; else build real weights dict via `linear_weight` helper (QuantizedLinear/LoRA-aware) + `position_ids = mx.arange(S)` + delegate to `_attention_mlx` (dispatch routes cr=4-real→CSA, cr=128→HCA).
  - (d) cr=0 sliding path UNCHANGED (cross-check existing 13.3a AttentionNN tests still GREEN).
- AC4: `test_13_3b_3_nn_mixed_layers_forward.py` — `AttentionNN.__call__` over real-dim CSA + HCA + sliding mixed `layer_types` matches torch within tol. Anti-circularity: torch ref from SAME synthesized weights (ADR 0007 §4).

### Axis 6 — AC1-AC3 (dispatch routing + tiny byte-identity + parity-through-dispatch)
- AC1: `test_13_3b_1_dispatch_additive.py` (extended) — real cr=4 → `_csa_attention_real_mlx` (spy); real cr=128 → `_attention_real_mlx`; cr=4-tiny → `_csa_attention_mlx` byte-identical; cr=0 → main body.
- AC2: tiny byte-identity — 11.14/11.15 tiny CSA byte-identical before/after the dispatch change.
- AC3: `test_csa_attention_real_mlx_parity.py` + `test_attention_real_mlx_hca_parity.py` (extended) — through-`_attention_mlx`-dispatch output identical to direct helper (dispatch is pure router); transitive parity to torch via direct-helper GREEN.

### Axis 7 — sha-pin cascade SLICE-INVARIANT (watch-item 4)
- Rigorous WHOLE-TREE scan: `grep -rn -E '0x[0-9a-f]{16}|sha16|deepseek_v4.*hash|hash.*deepseek_v4' tests/ scripts/finetune_ds4.py python-envs/mlx/src/ 2>/dev/null | grep -iv 'node_modules\|\.venv'`.
- Enumerate ALL pin sites referencing `deepseek_v4.py` content hash.
- Confirm: (a) exactly the right count advanced; (b) new sha16 post-`d1f1488` hashes; (c) no phantom; (d) no stale old hash in non-pycache.
- Adjudicate: legitimate cascade (additive dispatch branch shifted deepseek_v4.py content hash) vs phantom.

### Axis 8 — regression + no scope explosion
- 13.3b-1 (AC0-AC3) + 13.3b-2a (AC1-AC2) + 13.3b-2b (AC1-AC2) + 13.3a nn (12 incl backward AC) + 13.2 FP4 (9) + tiny CSA (11.14/11.15) all GREEN.
- Full suite: 569 passed / 13 skipped / 0 introduced RED (Test #8 now passes — Coder updated stale proof-count expectation; verify this is legitimate, not a tautology/relaxation).
- No convert, no new real-branch helpers, no GroupedLinear.

## Verdict
For each axis: GREEN / BLOCKED / NEEDS-INFO. If all 8 GREEN → `{"status":"ok","role":"Reviewer"}`. If any BLOCKED → `{"status":"blocked","axis":"<N>","reason":"<...>","role":"Reviewer"}` + `agent-output/cmux-13-3b/review-13-3b-3.md` with block + remediation.

**SPECIAL SCRUTINY on Test #8**: Coder claims it now passes (stale proof-count expectation updated). Reviewer must verify this is a LEGITIMATE update (the new dispatch genuinely adds a proof → expectation correctly bumped) NOT a tautology/relaxation that hides a regression. If the update relaxes a real assertion to make a regression pass → BLOCKED.

## Deliverables
1. `agent-output/cmux-13-3b/review-13-3b-3.md` — 8-axis verdict with watch-item adjudications + Test #8 scrutiny.
2. `.cmux-status/reviewer.done` (`{"status":"ok","role":"Reviewer"}` or blocked JSON).
3. In-pane JSON in surface:84 ONLY.

## Pre-flight read
- `agent-output/cmux-13-3b/task-coder-13-3b-3.md` (Coder spec, esp. BA Q1-Q4).
- `agent-output/cmux-13-3b/requirements-13-3b-3.md` (§0/§A Q1-Q5/§B/§C/§D).
- `agent-output/cmux-13-3b/architecture.md` §4.1/§4.2/§4.3/§3.4/§8.
- `docs/adr/0026-csa-hca-indexer-mlx-real-port.md` (§Decision-1 sanctioned dispatch edit site).
- `docs/adr/0025-*.md` (nn file ownership).
- `git show d1f1488` (the commit).
- FROZEN `deepseek_v4.py` (the dispatch branch + check cr=0 body byte-identical).
- `deepseek_v4_nn.py` (AttentionNN guard lift + §4.1 submodules).
- The new + extended test files.

## Style
Caveman ultra default; byte-exact exempt. Use `ctx_execute`/`ctx_batch_execute` for AST/hash + git diff. Cite axis + AC + Q numbers. MUST NOT edit production code. BEGIN NOW.
