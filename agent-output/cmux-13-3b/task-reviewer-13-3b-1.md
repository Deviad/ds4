# Story 13.3b-1 — Reviewer Task Brief (8-axis review + sha-pin cascade watch-item)

## Role
Reviewer (openai-codex/gpt-5.5 · xhigh, surface:84, fresh-context via /new, system-prompt `replace`). **Must NOT edit production code.**

## Context
- Coder 13.3b-1 complete: HCA real-dim MLX port + RoPE oracle. Commit `1f5d8e4` (parent `702199f`).
- BA verdict: AC0 = GO (compress-rope parity achievable additively). Architect design: §3.1/§3.4/§3.5/§4.2 of `agent-output/cmux-13-3b/architecture.md` + ADR 0026.
- Coder claims: 4 new helpers (`_compress_rope_yarn_tail_tables_mlx:545`, `_hca_compressor_mlx:591`, `_attention_real_mlx:658`, additive dispatch at `:853`); 4 new tests GREEN; parity deltas tight; FROZEN bodies byte-intact (AST); full suite 1/561/13 (Test #8 carve-out), 0 introduced RED; sha-pin cascade advanced.
- **Watch-item (supervisor-flagged)**: Coder updated stale-hash sentinels as part of the 3-site sha-pin cascade SLICE-INVARIANT (AGENTS.md: any edit to `deepseek_v4.py` MUST advance 3 sha16 pin sites). Reviewer MUST adjudicate: (a) exactly 3 sites advanced (not 2, not 4)? (b) the new sha16 values correct (post-edit hash of the relevant frozen regions)? (c) no phantom sentinel updates (values that didn't need to change)?

## Your job — 8-axis review (fresh context, adversarial)

### Axis 1 — Scope: additive ONLY, blast-radius honored
- `git show 1f5d8e4 --stat` — confirm ONLY `python-envs/mlx/.../deepseek_v4.py` + 4 NEW test files changed. NO nn port edit (13.3b-3 owns), NO convert (13.3b-4 owns), NO CSA/Indexer helpers (13.3b-2 owns), NO other production files.
- `git show 1f5d8e4 -- python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` — confirm ONLY additions (new module section + ONE dispatch branch). NO body edits.

### Axis 2 — FROZEN bodies byte-intact (independent AST/source-hash proof, NOT git diff)
The FROZEN must-not-edit-body list (BA §C + Architect §8): `_csa_config_error`, `_require_csa_config`, `_csa_attention_mlx`, `_csa_compressor_mlx`, `_csa_indexer_mlx`, `_csa_windowed_compressor_mlx`, `_hyperconnection_mlx`, `_hyperhead_mlx`, `_attention_mlx` cr=0 body, `sanitize_weights`, parity `Model`, FP4/i8 dequant, `_apply_rope_full_mlx`, `_rope_tail_tables_mlx`, `_apply_rope_tail_mlx`, 9 ADR-0017 forbidden symbols.
- Independently extract each symbol's source from HEAD~1 (`702199f`) and HEAD (`1f5d8e4`), compute sha16, confirm identical. Use `ctx_execute` to keep bytes out of context. Coder claims `_attention_mlx` cr=0 body `sha16=7c9960107ef0919d` — verify.

### Axis 3 — AC0 RoPE oracle: parity claim legitimate + anti-circularity
- `tests/test_compress_rope_oracle.py` — read it. Confirm the torch reference computes expected cos/sin (and rotated q/kv) from the SAME synthesized weights/config as the MLX path (ADR 0007 §4 anti-circularity — NO second MLX primitive as oracle).
- Confirm channel scope TRAILING `qk_rope_head_dim=64` (not full head_dim=512) — BA §0 Q1 + the test asserts cos shape `(1, n_win, 32)` → rope_dim=64.
- Confirm yarn (factor=16, orig_max=65536, theta=160000) applied — not plain theta.
- Confirm positions = `arange(n_win)*rate` for compress layers — not all positions.
- Confirm tolerance `atol=1e-3` (cos/sin) / `atol=1e-2` (rotated) — BA Q1.
- Confirm the asserted parity deltas (cos 7.2e-7, sin 1.9e-6, rotated 7.8e-3) are within tol AND the test would FAIL if parity broke (not a tautology).

### Axis 4 — AC1 HCA compressor: parity vs torch HCACompressor.forward
- `tests/test_hca_compressor_mlx_parity.py` — confirm torch `DeepseekV4HCACompressor.forward` stateless branch (past_key_values=None) is the reference.
- Confirm `_hca_compressor_mlx` (`deepseek_v4.py:591`): rate=128, out_dim=head_dim=512, NO Ca/Cb overlap, ape `(128, 512)` NO TRANSPOSE (BA Q2 resolved), single window pooling, returns `(compressed_kv [B,1,n_win,head_dim], block_bias [B,1,S,n_win]` causal entry i visible iff `i < (position_ids+1)//rate` else -inf).
- Confirm parity delta `compressed_kv 3.9e-6` within tol.

### Axis 5 — AC2 HCA attention: parity vs torch Attention.forward + reuse map honored
- `tests/test_attention_real_mlx_hca_parity.py` — confirm torch `DeepseekV4Attention.forward` for an HCA layer is the reference.
- Confirm `_attention_real_mlx` (`deepseek_v4.py:658`) steps 1-6 per Architect §3.4: step1 proj+norm+RoPE=compress-yarn-tail REUSE cr=0 math; step2 HCA branch; step3 kv_full cat; step4 mask cat; step5 multi-head scores+sink+softmax+attend REUSE cr=0; step6 rope-undo (-sin same cos/sin) + grouped-o REUSE cr=0 tail.
- Confirm Q3: projections SHARED with cr=0 (wq_a/wq_b/wkv/wo_a/wo_b + q_norm/kv_norm + attn_sink) — only compressor (wkv/wgate/ape/norm) is HCA-specific.
- Confirm Q4: rope-undo + grouped-o IN the HCA path (not skipped).
- Confirm parity delta `output 6.3e-7` within tol.

### Axis 6 — AC3 dispatch additive: tiny byte-identical + real routes NEW
- `tests/test_13_3b_1_dispatch_additive.py` — confirm (a) tiny compress input: `_csa_config_error(args) is None` AND return value byte-identical to pre-edit (assert via running both the tiny path directly AND through `_attention_mlx`); (b) real-dim compress input: routes to `_attention_real_mlx` (no NotImplementedError).
- Confirm the ONE dispatch branch at `:853`: `if _csa_config_error(args) is None: return _csa_attention_mlx(...)` (tiny byte-identical) `else: return _attention_real_mlx(...)` (NEW). NO existing test input changes branch.

### Axis 7 — sha-pin cascade SLICE-INVARIANT (watch-item)
- AGENTS.md: any edit to `deepseek_v4.py` MUST advance 3 sha16 pin sites. Coder claimed "3 sites" advanced + "stale-hash sentinel updates required for full-suite compatibility after ADR 0026 sanctioned an additive deepseek_v4.py baseline change."
- Find the sha16 pin sites (grep `sha16` or `0x[0-9a-f]{16}` in `tests/` + `scripts/finetune_ds4.py`). Confirm:
  - (a) EXACTLY 3 sites advanced (not 2, not 4, not phantom).
  - (b) The new sha16 values are the post-edit hashes of the correct frozen regions (recompute the hash of the actual edited file region and compare).
  - (c) No phantom sentinel updates (values that didn't need to change).
- Adjudicate: is the cascade legitimate (the additive dispatch branch DID shift the sha16 of the `_attention_mlx` symbol, hence the pin cascade is required), or phantom (pins updated for no real reason)?

### Axis 8 — regression + no scope explosion
- 13.2 FP4 (9) + 13.3a nn (12 incl 13.3a-3 backward AC) + tiny CSA (11.14/11.15) all GREEN.
- Full suite: 1 RED (Test #8 carve-out, pre-existing) / 561 PASS / 13 SKIP / 0 introduced RED.
- No nn port edit, no convert, no CSA/Indexer helpers.

## Verdict
For each axis: GREEN / BLOCKED / NEEDS-INFO. If all 8 GREEN → `{"status":"ok","role":"Reviewer"}`. If any BLOCKED → `{"status":"blocked","axis":"<N>","reason":"<...>","role":"Reviewer"}` + `agent-output/cmux-13-3b/review-13-3b-1.md` with the block + recommended remediation (Architect § clarification OR Coder r2).

## Deliverables
1. `agent-output/cmux-13-3b/review-13-3b-1.md` — 8-axis verdict with watch-item adjudication.
2. `.cmux-status/reviewer.done` (`{"status":"ok","role":"Reviewer"}` or blocked JSON).
3. In-pane JSON in surface:84 ONLY.

## Pre-flight read
- `agent-output/cmux-13-3b/task-coder-13-3b-1.md` (Coder's spec).
- `agent-output/cmux-13-3b/requirements-13-3b-1.md` (BA LOCKED Q-list + AC0-AC3).
- `agent-output/cmux-13-3b/architecture.md` §2/§3.1/§3.4/§3.5/§4.2/§8.
- `docs/adr/0026-csa-hca-indexer-mlx-real-port.md`.
- `git show 1f5d8e4` (the commit).
- FROZEN `deepseek_v4.py` (the 4 new helpers + dispatch branch).
- The 4 new test files.

## Style
Caveman ultra default; byte-exact exempt (paths/SHAs/line numbers/torch:line/FROZEN:line/parity deltas/axis numbers verbatim). Use `ctx_execute`/`ctx_batch_execute` for AST/hash computation + git diff inspection. Cite axis numbers + AC numbers. MUST NOT edit production code. BEGIN NOW.
