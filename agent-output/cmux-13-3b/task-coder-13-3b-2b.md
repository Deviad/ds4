# Story 13.3b-2b — Coder Task Brief (TDD red→green; CSA attention wiring)

## Role
Coder (openai-codex/gpt-5.5 · xhigh, surface:83, fresh-context via /new). Implement 13.3b-2b: CSA attention wiring (CSA compressor real keys + block_bias from `_indexer_mlx` + KV-append + multi-head). TDD red-first.

## Slice context
- **Story**: 13.3b-2b (SECOND half of SPLIT 13.3b-2). CSA attention branch as a HELPER (additive). 13.3b-3 owns the dispatch wiring (cr=4→CSA, cr=128→HCA) + GroupedLinear + AttentionNN guard lift + integration. 13.3b-2b produces the CSA branch, NOT the dispatch.
- **Predecessor**: 13.3b-2a complete (HEAD `a2c20b0`). `_indexer_mlx:620` + `_indexer_scorer_mlx:591` GREEN, CONSUMED here for block_bias. `_compress_rope_yarn_tail_tables_mlx:545` + `_apply_rope_tail_mlx:266` (13.3b-1) available + proven.
- **Estimated**: 4 dev-days.

## ⚠️ BA CORRECTIONS (vs the BA brief — read carefully, these CHANGED)
1. **CSA compressor = SINGLE-head, out_dim=head_dim=512** (NOT multi-head). "Multi-head" is the ATTENTION q-side (`num_attention_heads=64` heads broadcasting against single `num_key_value_heads=1` kv head), NOT the compressor. Identical single-kv-head pattern to HCA `_hca_compressor_mlx:806` (`expand_dims(compressed, 1)`).
2. **ape-transpose: NO transpose at helper input**. The `ape[:out_dim,:].T` the original BA brief premise referred to lives ONLY inside the FROZEN MLX helper `_csa_windowed_compressor_mlx:387-388` (and only because that helper STORES ape convert-transposed `(2*out_dim, rate)` and transposes it back internally). **Option α does NOT call that FROZEN helper** → consumes ape token-major `(rate=4, 2*head_dim=1024)` with NO transpose. BOTH HCA (13.3b-1) AND CSA (here) AND indexer (13.3b-2a) consume ape token-major NO-transpose under option α. The "HCA=NO / CSA=YES transpose mirror" is VOID — both NO. (The convert-side transpose architecture.md §5.2 is a 13.3b-4 concern ONLY for feeding the FROZEN helper, which option α bypasses.)
3. **Rope landmine = option α** (reimplement Ca/Cb overlap inline + compress-yarn-tail via REUSED 13.3b-1 helpers) — CONFIRMED, same resolution as 13.3b-2a. NO new kernel.
4. **block_bias build = scatter from `_indexer_mlx` top-k** (modeling:693-702). MLX has scatter primitives.
5. **KV-append + multi-head + grouped-o = REUSE HCA `_attention_real_mlx:870-910` byte-pattern**.

## Authoritative contracts — READ IN FULL BEFORE CODING (in order)
1. `agent-output/cmux-13-3b/requirements-13-3b-2b.md` — **§0 GO verdict** + §0.1 rope-landmine re-adjudication (option α LOCKED) + §0.2 ape-transpose re-adjudication (NO transpose) + §A Q1-Q7 LOCKED + §B AC1-AC2 + §C blast-radius + §D STOP. **This is your SPEC.**
2. `agent-output/cmux-13-3b/architecture.md` §0-§5 — esp. §3.4 (unified `_attention_real_mlx` CSA+HCA), §3.5 (RoPE parity), §2.1 reuse map (NOTE §2.1↔§3.5 tension — BA §0.1 resolved: option α), §8 frozen list.
3. `agent-output/cmux-13-3b/subslice-breakdown.md` 13.3b-2b row + 13.3b-3 fence.
4. `agent-output/cmux-13-3b/requirements-13-3b-2a.md` (rope-landmine resolution proven — mirror for CSA compressor).
5. `agent-output/cmux-13-3b/requirements-13-3b-1.md` (HCA pattern — the CSA branch mirrors `_attention_real_mlx` HCA step 3/5/6).
6. `docs/adr/0026-csa-hca-indexer-mlx-real-port.md`.

## Scope — EXACTLY what you implement (BA §C blast-radius confirmed)
### File 1: FROZEN `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` — ADDITIVE ONLY
- **NEW CSA branch** — ADD a CSA branch to `_attention_real_mlx:819` (next to the 13.3b-1 HCA branch; HCA branch body MUST stay byte-identical) OR a NEW `_csa_attention_real_mlx` helper (BA pin which is cleaner per §3.4). The CSA branch:
  1. **CSA compressor** (option α, reimplement inline per BA Q1): port torch `CSACompressor.forward:589-754`. SINGLE-head, out_dim=head_dim=512, cr=4. Ca/Cb overlap pooling: `chunk_gate` from `new_kv` reshaped `[B, n_win, rate=4, 2*head_dim=1024]`; `chunk_gate += position_bias` (ape `(rate=4, 2*head_dim=1024)` token-major, **NO transpose** §0.2). Pool to `compressed_kv [B,1,n_win,head_dim=512]` via `expand_dims(compressed, 1)` (mirror HCA `_hca_compressor_mlx:806`).
  2. **RoPE** (landmine §0.1): compress-yarn-tail TRAILING-64 of head_dim=512, positions `arange(n_win)*rate` (modeling:676-680), via REUSED `_compress_rope_yarn_tail_tables_mlx:545` + `_apply_rope_tail_mlx:266` (`head_dim=512`, `qk_rope_head_dim=64`). Mirror shipped `_hca_compressor_mlx:795-805` rope block.
  3. **block_bias build** (Q3): scatter from `_indexer_mlx` top-k indices + `-1` sentinels → `block_bias [B,1,S,n_win]` per modeling:693-702. Valid pick (non-sentinel) → 0; else → `-inf`. MLX scatter primitives. BA Q3 pins exact dtype/fill/scatter.
  4. **KV-append** (Q4): `kv_full = cat([kv, compressed_kv], axis=2)` — REUSE 13.3b-1 HCA step 3 pattern (modeling). BA confirm shape broadcast.
  5. **mask cat** (Q4): `mask = cat([sliding_causal_mask, block_bias], axis=-1)` — REUSE 13.3b-1 HCA step 4 pattern.
  6. **Multi-head attention** (Q5): scores + sink + softmax + attend — REUSE cr=0 math (same as HCA step 5). rope-undo + grouped-o REUSE HCA step 6 (`_attention_real_mlx:870-910` byte-pattern). Q5: CSA uses grouped-o (o_groups=8) LIKE HCA — confirm via torch; if REUSE, mirror HCA step 6.
- **NO dispatch branch** in `_attention_mlx` (13.3b-3 owns). **NO FROZEN body edits** (additive ONLY; the HCA branch in `_attention_real_mlx:819` MUST stay byte-identical; CSA branch ADDED additively).

### Files 2-3: NEW tests (TDD RED-first)
2. `tests/test_csa_compressor_real_mlx_parity.py` (AC1): CSA compressor (inline Ca/Cb + correct compress-yarn-tail rope) output matches torch `CSACompressor.forward:589-754` within tol at real dims. Q1 + rope + §0.2 ape no-transpose.
3. `tests/test_csa_attention_real_mlx_parity.py` (AC2): CSA attention branch (compressor + block_bias from `_indexer_mlx` + KV-append + multi-head + grouped-o) output matches torch `Attention.forward` CSA layer (modeling:801-875) within tol at real dims. Q3 + Q4 + Q5 + Q6 anti-circ.

## §A — LOCKED spec (BA `requirements-13-3b-2b.md` §A — follow EXACTLY)
- **Q1 CSA compressor**: SINGLE-head, out_dim=head_dim=512, cr=4, Ca/Cb overlap, real ckpt keys `attn.compressor.*` (shape/name assert on layer 2, no tensor load). Rope option α.
- **Q2 ape**: token-major `(rate=4, 2*head_dim=1024)` NO transpose (§0.2 — BA INVERTED the brief premise).
- **Q3 block_bias**: scatter from `_indexer_mlx` top-k + sentinels; `[B,1,S,n_win]`; 0 valid / -inf else; dtype/fill BA pins.
- **Q4 KV-append + mask cat**: `kv_full=cat([kv,compressed_kv],axis=2)`; `mask=cat([sliding_causal,block_bias],axis=-1)`; REUSE HCA step 3/4.
- **Q5 multi-head + grouped-o**: REUSE cr=0 math + HCA step 5/6 (`_attention_real_mlx:870-910`).
- **Q6 fixtures + anti-circularity**: synthesize real dims; ONE CSA layer 2 header cross-check; torch ref from same synthesized weights (ADR 0007 §4).
- **Q7 STOP**: see STOP-ESCALATE below.

## §B — Acceptance criteria (TDD red-first)
- **AC1**: `test_csa_compressor_real_mlx_parity.py` GREEN.
- **AC2**: `test_csa_attention_real_mlx_parity.py` GREEN.
- **regression**: 13.3b-1 (AC0-AC3) + 13.3b-2a (AC1-AC2 incl `_indexer_mlx` reuse) + 13.3a nn (12 incl backward AC) + 13.2 FP4 (9) + tiny CSA (11.14/11.15) all GREEN. Full suite: 1 RED (Test #8 carve-out, pre-existing) / 563+2new PASS / 13 SKIP / 0 introduced RED.
- **byte-intactness**: FROZEN bodies byte-intact via source hash + AST (`_attention_real_mlx:819` HCA branch byte-identical; CSA ADDED additively). `git diff --check` clean. sha-pin cascade SLICE-INVARIANT: advance exactly the right count of sha16 pin sites (Reviewer verifies via rigorous whole-tree scan).

## STOP-ESCALATE (BA §A Q7 + §D — binding)
Emit `{"status":"error","role":"Coder",...}` + write `agent-output/cmux-13-3b/coder-13-3b-2b-stop.md` if:
- **(S-rope-csa)**: CSA compressor rope needs a contract FROZEN can't supply additively (BA recon: option α proven 13.3b-2a; STOP only if code proves wrong).
- **(S-frozen-body)**: any FROZEN symbol body (`_csa_windowed_compressor_mlx:347` byte-identical even if bypassed; `_attention_real_mlx:819` HCA branch byte-identical — CSA ADDED additively NOT HCA edited; `_hca_compressor_mlx:752`, `_indexer_mlx:620`, `_indexer_scorer_mlx:591`, `_compress_rope_yarn_tail_tables_mlx:545`, `_apply_rope_tail_mlx:266`, `_apply_rope_full_mlx:311`, `_rope_full_tables_mlx:300`, `_attention_mlx` cr=0 body, `_csa_attention_mlx:515`, `_csa_compressor_mlx:419`, `_csa_indexer_mlx:435`, `_csa_config_error:317`, `_require_csa_config:335`, `_causal_sliding_mask_mlx:914`, `_hyperconnection_mlx`, `_hyperhead_mlx`, parity `Model:2059`, `sanitize_weights`, 9 ADR-0017 forbidden) would need EDIT → STOP. ADDITIVE ONLY.
- **(S-tiny-regression)**: tiny CSA fixtures (11.14/11.15) go RED → STOP.
- **(S-parity-csa)**: CSA attention output cannot match torch within tol after best-effort → STOP with failing assertion + torch-vs-MLX delta.
- **(S-backward)**: 13.3a-3 backward AC goes RED → STOP.

## Pre-flight read list (READ in this order)
1. `agent-output/cmux-13-3b/requirements-13-3b-2b.md` (SPEC — §0/§0.1/§0.2/§A/§B/§C/§D).
2. `agent-output/cmux-13-3b/architecture.md` (§2.1, §3.4, §3.5, §8).
3. `agent-output/cmux-13-3b/subslice-breakdown.md` 13.3b-2b row.
4. `agent-output/cmux-13-3b/requirements-13-3b-2a.md` (rope-landmine resolution proven).
5. `agent-output/cmux-13-3b/requirements-13-3b-1.md` (HCA pattern).
6. `docs/adr/0026-csa-hca-indexer-mlx-real-port.md`.
7. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` — FROZEN. Read: `_attention_real_mlx:819` (13.3b-1 HCA branch — CSA branch ADDED here or beside), `_csa_windowed_compressor_mlx:347` (the bypassed helper — confirm baked-in rope + byte-identical), `_indexer_mlx:620` + `_indexer_scorer_mlx:591` (13.3b-2a — consumed for block_bias), `_compress_rope_yarn_tail_tables_mlx:545` + `_apply_rope_tail_mlx:266` (13.3b-1 reused), `_hca_compressor_mlx:752` + the shipped rope block `:795-805` (HCA pattern to MIRROR for CSA compressor), `_causal_sliding_mask_mlx:914`, `_apply_rope_full_mlx:311`/`_rope_full_tables_mlx:300` (WRONG baked-in rope to avoid), `_csa_config_error:317`, FROZEN imports.
8. `python-envs/mlx/.venv/lib/python3.13/site-packages/transformers/models/deepseek_v4/modeling_deepseek_v4.py` — torch ref. Read: `CSACompressor:589-754` (forward + rope + ape usage), `Attention.forward:801-875` (CSA branch + block_bias build :693-702 + KV-append + multi-head). Use `ctx_execute_file` (keep 1526 lines out).
9. `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json` (compress_ratios 21×cr=4, num_heads=64, o_groups=8, head_dim=512, qk_rope_head_dim=64, index_topk=512, index_n_heads=64, index_head_dim=128, hidden=4096, q_lora_rank=1024, hc_mult=4).
10. The 13.3b-1/2a test files (mirror anti-circularity pattern).

## Build/test workflow
1. `source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first).
2. **FIRST**: probe FROZEN `_csa_windowed_compressor_mlx:347` via `ctx_execute_file` — confirm baked-in full-channel rope + identify the exact lines to AVOID (mirror 13.3b-2a probe). Confirms option α (inline reimpl).
3. Write AC1 RED → implement CSA compressor (inline Ca/Cb + correct rope) → GREEN.
4. Write AC2 RED → implement CSA attention branch (compressor + block_bias from `_indexer_mlx` + KV-append + mask-cat + multi-head + grouped-o) → GREEN.
5. Full suite: `python3 -m pytest tests/ --no-header -q | tail -5`. Confirm 563+2new/1/13, 0 introduced RED.
6. FROZEN byte-intactness: source-hash + AST on the frozen-body symbol set (CRITICAL: `_attention_real_mlx:819` HCA branch byte-identical). `git diff --check`.
7. sha-pin cascade: advance exactly the right count of sha16 pin sites.

## Deliverables
1. FROZEN `deepseek_v4.py` additive edits (CSA branch; NO dispatch branch; NO FROZEN body edits).
2. 2 NEW test files (AC1-AC2).
3. `agent-output/cmux-13-3b/coder-13-3b-2b-notes.md` — what you did, AC1-AC2 results, parity deltas, rope-landmine resolution (option α confirmed), ape no-transpose handling, blast-radius surprises.
4. `.cmux-status/coder.done` (`{"status":"ok","role":"Coder"}`).
5. In-pane JSON: `{"status":"ok","role":"Coder"}` in surface:83 ONLY.
6. git commit (Story 13.3b-2b + ADR 0026 + AC results).

## Style
Caveman ultra default; byte-exact exempt. Use `ctx_execute`/`ctx_batch_execute` for torch/FROZEN probes. Cite §/Q/AC numbers.

MUST NOT: wire dispatch (13.3b-3), edit nn port (13.3b-3), write convert (13.3b-4), GroupedLinear (13.3b-3), edit FROZEN bodies (additive ONLY; HCA branch byte-identical), introduce C++. BEGIN NOW. The rope landmine + ape-no-transpose are the #1 risks — resolve via the probe in step 2 first.
