# Story 13.3b-1 — Coder Task Brief (TDD red→green; AC0 RoPE oracle GATES AC1/AC2)

## Role
Coder (openai-codex/gpt-5.5 · xhigh, surface:83, fresh-context via /new). Implement 13.3b-1: HCA real-dim MLX port + RoPE oracle. TDD red-first; AC0 GREEN must precede AC1/AC2.

## Slice context
- **Story**: 13.3b-1 (FIRST sub-slice of 13.3b). Epic go/no-go CLEARED by BA (AC0 verdict = GO; compress-rope parity achievable with FROZEN primitives additively).
- **Predecessor**: 13.3a-3 complete (HEAD `702199f`). 13.3b Architect r0 + BA r0 complete.
- **Estimated**: 5 dev-days (oracle 1.5 + HCA compressor 1.5 + HCA attn 2). AC0 GREEN-first gates AC1/AC2.

## Authoritative contracts — READ IN FULL BEFORE CODING (in order)
1. `agent-output/cmux-13-3b/requirements-13-3b-1.md` — **§0 EPIC GO/NO-GO = GO** (the RoPE recon verdict) + §A Q1-Q7 LOCKED + §B AC0-AC3 + §C blast-radius + §D STOP-rules. This is your SPEC.
2. `agent-output/cmux-13-3b/architecture.md` §0-§5 — esp. §2 core finding (FROZEN tiny `_csa_attention_mlx:515` ≠ real arch; real = multi-head MLA sliding + compressed-KV append + block_bias), §2.1 reuse map, §3.1 (`_hca_compressor_mlx` design), §3.4 (`_attention_real_mlx` unified steps 1-6), §3.5 (RoPE parity risk), §4.2 (additive dispatch), §8 (FROZEN must-not-edit-body list).
3. `agent-output/cmux-13-3b/subslice-breakdown.md` 13.3b-1 row — scope/AC/STOP/dev-days.
4. `docs/adr/0026-csa-hca-indexer-mlx-real-port.md` — the additive FROZEN expansion contract.
5. `agent-output/cmux-13-3a/architecture.md` + ADR 0025 — the nn port being extended (13.3b-1 does NOT edit nn port; 13.3b-3 does).

## Scope — EXACTLY what you implement (BA §C blast-radius confirmed)
### File 1: FROZEN `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` — ADDITIVE ONLY
- **NEW module section** (append near the existing CSA section, AFTER the FROZEN tiny helpers, before parity `Model`): additive real-dim helpers. Name it clearly (e.g. `# --- Real-dim CSA/HCA port (Story 13.3b; additive only; tiny path byte-identical) ---`).
- **NEW helpers**:
  1. `_compress_rope_yarn_tail_tables_mlx(head_dim, qk_rope_head_dim, rope_theta, yarn_factor, yarn_orig_max, seq_len, *, positions)` → `(cos, sin)` for the TRAILING `qk_rope_head_dim=64` slice at the given positions (block-start positions `i*rate` for compress layers), with yarn scaling (factor=16, orig_max=65536, theta=160000). BA §0 Q1: pin against torch `DeepseekV4RotaryEmbedding.forward:153` + `apply_rotary_pos_emb:344`. Pure-math only.
  2. `_hca_compressor_mlx(args, x, weights, *, position_ids)` → `tuple[compressed_kv, block_bias]` per Architect §3.1 + BA §A Q2. rate=128, out_dim=head_dim=512, NO Ca/Cb overlap, single window pooling. `ape` shape `(128, 512)` added DIRECTLY (NO transpose — BA Q2 resolved). Returns `compressed_kv [B,1,n_win,head_dim]` + `block_bias [B,1,S,n_win]` (causal: entry `i` visible to query `t` iff `i < (position_ids+1)//rate`, else `-inf`).
  3. `_attention_real_mlx(args, x, weights, *, index_topk=None)` — the HCA branch of the unified real path per Architect §3.4 steps 1-6 + BA Q3/Q4:
     - Step 1: proj+norm+RoPE (REUSE cr=0 math from `_attention_mlx:632` body; RoPE = compress-yarn-tail for HCA layers).
     - Step 2 (HCA branch): `compressed_kv, block_bias = _hca_compressor_mlx(...)`.
     - Step 3: `kv_full = cat([kv, compressed_kv], axis=2)` `[B,1,S+n_win,head_dim]`.
     - Step 4: `mask = cat([sliding_causal_mask, block_bias], axis=-1)` `[B,1,S,S+n_win]`.
     - Step 5: multi-head scores `q_by_head @ kv_full.T * scale + mask` + per-head sink logit + softmax + drop sink + attend. REUSE cr=0 math.
     - Step 6: rope-undo (`-sin` compress-yarn-tail on output rope slice; SAME cos/sin as step 1, negated sin) + grouped-o (o_a block-diagonal + o_b). REUSE cr=0 tail (`_attention_mlx:632` body ~700-755).
- **ONE additive dispatch branch** in `_attention_mlx` (current `:641`, the `if args.compression_ratio != 0:` line). Per Architect §4.2 + BA Q5:
  ```python
  if args.compression_ratio != 0:
      if _csa_config_error(args) is None:          # tiny proven subset → UNCHANGED
          return _csa_attention_mlx(args, x, weights, index_topk=index_topk)
      return _attention_real_mlx(args, x, weights, index_topk=index_topk)   # NEW
  ```
  Tiny path byte-identical: existing tiny inputs satisfy `_csa_config_error is None` → identical return.

### Files 2-5: NEW tests (TDD RED-first)
2. `tests/test_compress_rope_oracle.py` (AC0 — LOAD-BEARING, write+make GREEN FIRST): for ONE real-dim compress layer (synthesize, BA Q6), assert MLX compress-rope cos/sin match torch reference within `atol=1e-3`; assert rotated q/kv match within `atol=1e-2`. Pin channel scope (TRAILING `qk_rope_head_dim=64`, verify cos shape `(1, n_win, 32)` → rope_dim=64) + yarn (factor=16) + positions (`arange(n_win)*rate`) in test docstring. Anti-circularity (BA Q6 + ADR 0007 §4): torch reference computes expected from the SAME synthesized weights/config; NO second MLX primitive as oracle.
3. `tests/test_hca_compressor_mlx_parity.py` (AC1): `_hca_compressor_mlx` output `(compressed_kv, block_bias)` matches torch `DeepseekV4HCACompressor.forward` (stateless branch, `past_key_values=None`) within tol at real dims (head_dim=512, rate=128, hidden=4096, q_lora_rank=1024, o_groups=8, num_heads=64, qk_rope_head_dim=64). Q2 ape-no-transpose pinned.
4. `tests/test_attention_real_mlx_hca_parity.py` (AC2): `_attention_real_mlx` HCA branch output matches torch `DeepseekV4Attention.forward` for an HCA layer within tol. Q3 projection-reuse (shared `wq_a/wq_b/wkv/wo_a/wo_b` + norms + `attn_sink`) + Q4 rope-undo/grouped-o confirmed.
5. `tests/test_13_3b_1_dispatch_additive.py` (AC3): (a) tiny compress input → `_csa_config_error(args) is None` + return value byte-identical to pre-edit (assert via running both the tiny path directly and through `_attention_mlx`); (b) real-dim compress input → routes to `_attention_real_mlx` (no `NotImplementedError`). Q5.

## §A — LOCKED spec (BA `requirements-13-3b-1.md` §A — follow EXACTLY)
- **Q1 oracle**: pin channel scope TRAILING-64, yarn YES (factor=16, orig_max=65536), positions `arange(n_win)*rate`, tol `atol=1e-3` cos/sin / `atol=1e-2` rotated.
- **Q2 ape**: HCA NO transpose; synthesize `compressor_ape` as `(128, 512)` directly.
- **Q3 reuse**: HCA projections SHARED with cr=0 (same names); only compressor (`wkv/wgate/ape/norm`) is HCA-specific. RoPE for HCA = compress-yarn-tail.
- **Q4 rope-undo + grouped-o**: YES in HCA path; identical to cr=0 tail using same compress-yarn-tail cos/sin (negated sin).
- **Q5 dispatch**: `if _csa_config_error(args) is None: return _csa_attention_mlx(...); else: return _attention_real_mlx(...)`. No existing test changes branch.
- **Q6 fixtures**: synthesize at real dims; ONE real HCA layer cross-check via safetensors header (shape/name assert only, no tensor load). Anti-circularity: torch reference computes expected from SAME synthesized weights.
- **Q7 STOP conditions**: see STOP-ESCALATE below.

## §B — Acceptance criteria (TDD red-first; AC0 GREEN gates AC1/AC2)
- **AC0** (LOAD-BEARING, write+GREEN FIRST): `test_compress_rope_oracle.py` GREEN. If you CANNOT make AC0 GREEN additively → STOP-ESCALATE immediately (do NOT proceed to AC1/AC2). The BA verdict was GO based on recon; your job is to PROVE it in code.
- **AC1**: `test_hca_compressor_mlx_parity.py` GREEN.
- **AC2**: `test_attention_real_mlx_hca_parity.py` GREEN.
- **AC3**: `test_13_3b_1_dispatch_additive.py` GREEN — tiny byte-identical + real-dim routes NEW.
- **regression**: 13.2 FP4 (9 tests) + 13.3a nn (12 incl 13.3a-3 backward AC) + tiny CSA fixtures (11.14/11.15) all GREEN. Full suite: 1 RED (pre-existing Test #8 `test_finetune_readiness_report_records_b0b_a_without_unblocking_b0`) / 557+4new PASS / 13 SKIP / 0 introduced RED.
- **byte-intactness**: FROZEN bodies (full list BA §C + Architect §8) byte-intact via source hash + AST (NOT `git diff` — AGENTS.md notes many files untracked-baseline). `git diff --check` for whitespace.

## STOP-ESCALATE (BA §A Q7 + Architect §9 — binding)
Emit `{"status":"error","role":"Coder",...}` + write `agent-output/cmux-13-3b/coder-13-3b-1-stop.md` if:
- **(S-rope, AC0)**: torch compress rope needs a math primitive FROZEN cannot supply additively (kernel, unsupported concat, non-derivable yarn). BA recon said NO; STOP only if code proves the recon wrong.
- **(S-frozen-body)**: any FROZEN symbol body (`_attention_mlx:632` cr=0 math, `_csa_*`, `_hyperconnection_mlx`, `_hyperhead_mlx`, parity `Model:1685`, `sanitize_weights`, `_apply_rope_full_mlx`, `_rope_tail_tables_mlx`, `_apply_rope_tail_mlx`, 9 ADR-0017 forbidden symbols, FP4/i8 dequant) would need EDIT (not additive new symbol) → STOP. The contract is ADDITIVE ONLY.
- **(S-tiny-regression)**: any tiny CSA fixture (11.14/11.15) goes RED → STOP (tiny path must stay byte-identical per ADR 0026).
- **(S-parity)**: HCA compressor or HCA attention cannot match torch reference within tol at real dims after best-effort → STOP with the failing assertion + the torch-vs-MLX delta.
- **(S-backward)**: 13.3a-3 backward AC (`test_deepseek_v4_nn_backward.py`) goes RED → STOP (additive dispatch must not break 13.3a nn port).

## Pre-flight read list (READ in this order)
1. `agent-output/cmux-13-3b/requirements-13-3b-1.md` (SPEC — §0 verdict, §A Q1-Q7, §B AC0-AC3, §C blast-radius, §D STOP).
2. `agent-output/cmux-13-3b/architecture.md` (§2 core finding, §2.1 reuse map, §3.1 HCA compressor, §3.4 unified real attention, §3.5 RoPE risk, §4.2 dispatch, §8 frozen list).
3. `agent-output/cmux-13-3b/subslice-breakdown.md` 13.3b-1 row.
4. `docs/adr/0026-csa-hca-indexer-mlx-real-port.md`.
5. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` — FROZEN. Read: `_attention_mlx:632` (cr=0 body to REUSE, the dispatch `:641` to edit additively), `_csa_attention_mlx:515`, `_csa_config_error:317`, `_require_csa_config:335`, `_csa_windowed_compressor_mlx:347`, `_apply_rope_full_mlx`, `_rope_tail_tables_mlx`, `_apply_rope_tail_mlx`, `_rms_norm_mlx:133`, `_linear_mlx:137`, the FROZEN import list at top, parity `Model:1685`, `sanitize_weights:1600`.
6. `python-envs/mlx/.venv/lib/python3.13/site-packages/transformers/models/deepseek_v4/modeling_deepseek_v4.py` — torch reference. Read lines 75-170 (`DeepseekV4RotaryEmbedding` + `apply_rotary_pos_emb`), 362-445 (`HCACompressor.forward`), 755-875 (`Attention.forward`), 876-973 (HyperConnection/HyperHead — confirm parity, not port). Use `ctx_execute_file`/`ctx_execute` to probe (keep 1526 lines out of context).
7. `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json` — real config (compress_rope_theta=160000, rope_scaling.factor=16, qk_rope_head_dim=64, partial_rotary_factor=0.125, head_dim=512, hidden=4096, num_heads=64, o_groups=8, q_lora_rank=1024, hc_mult=4, compress_ratios array, num_hidden_layers=43).
8. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py` — the nn port (13.3b-1 does NOT edit; 13.3b-3 owns AttentionNN guard lift). Read for context only.

## Build/test workflow
1. `source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first).
2. Write AC0 test RED first. Run it RED. Then implement `_compress_rope_yarn_tail_tables_mlx`. Make AC0 GREEN.
3. AC0 GREEN → write AC1 RED → implement `_hca_compressor_mlx` → GREEN.
4. AC1 GREEN → write AC2 RED → implement `_attention_real_mlx` HCA branch → GREEN.
5. AC2 GREEN → write AC3 RED → add the ONE additive dispatch branch → GREEN.
6. Run full suite: `python3 -m pytest tests/ --no-header -q | tail -5`. Confirm 557+4new/1/13, 0 introduced RED.
7. FROZEN byte-intactness: source-hash + AST check on `deepseek_v4.py` for the frozen-body symbol set (BA §C). `git diff --check` for whitespace.

## Deliverables
1. FROZEN `deepseek_v4.py` additive edits (NEW module section + ONE dispatch branch; tiny path byte-identical).
2. 4 NEW test files (AC0-AC3).
3. `agent-output/cmux-13-3b/coder-13-3b-1-notes.md` — what you did, AC0-AC3 results, parity deltas, any blast-radius surprises (e.g. if you had to touch a FROZEN body — STOP first).
4. `.cmux-status/coder.done` (`{"status":"ok","role":"Coder"}`).
5. In-pane JSON: `{"status":"ok","role":"Coder"}` in surface:83 ONLY.
6. git commit (you own the commit; message per AGENTS.md convention — Story 13.3b-1 + ADR 0026 reference + AC results).

## Style
Caveman ultra default; byte-exact exempt (code/paths/line numbers/torch:line/FROZEN:line/ADR cites/SHAs verbatim). Use `ctx_execute`/`ctx_batch_execute` for torch/FROZEN file probes (keep 1526-line torch file out of context). Cite §-numbers + Q-numbers + AC numbers.

MUST NOT: edit nn port (13.3b-3), write convert (13.3b-4), write CSA/Indexer helpers (13.3b-2), edit FROZEN bodies (additive ONLY), introduce C++. BEGIN NOW. AC0 GREEN-FIRST GATES EVERYTHING.
