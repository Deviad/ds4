# Story 13.3b-2a — Coder Task Brief (TDD red→green; Indexer real path + rope landmine)

## Role
Coder (openai-codex/gpt-5.5 · xhigh, surface:83, fresh-context via /new). Implement 13.3b-2a: Indexer real-dim MLX port (`_indexer_scorer_mlx` + `_indexer_mlx`). TDD red-first.

## Slice context
- **Story**: 13.3b-2a (FIRST half of SPLIT 13.3b-2). Indexer only; CSA attention wiring is 13.3b-2b (LATER).
- **Predecessor**: 13.3b-1 complete (HEAD `1f5d8e4`). HCA + RoPE oracle + `_compress_rope_yarn_tail_tables_mlx:545` helper GREEN.
- **Estimated**: 4 dev-days. AC2 top-k is the load-bearing piece.

## ⚠️ ROPE LANDMINE (BA §0/Q2 caught — front-loaded, do NOT miss)
The indexer q-rope + compressed_kv-rope use **compress-yarn over TRAILING `qk_rope_head_dim=64`** of `index_head_dim=128` (NOT full 128). torch `apply_rotary_pos_emb:357` rotates trailing 64.

- ✅ **REUSE** 13.3b-1 `_compress_rope_yarn_tail_tables_mlx:545` + `_apply_rope_tail_mlx` to apply compress-yarn-tail AFTER the FROZEN-windowed-pooling step.
- ❌ **DO NOT** rely on FROZEN `_csa_windowed_compressor_mlx:347`'s BAKED-IN full-channel plain-theta rope (`_rope_full_tables_mlx` + `_apply_rope_full_mlx`) — it MISMATCHES torch real indexer compress-yarn-trailing-64.
- ❌ **DO NOT** rely on FROZEN tiny `_csa_indexer_mlx:436` full-channel rope — also wrong contract.
- ❌ **DO NOT EDIT** FROZEN bodies to fix the rope mismatch — that's an **S-frozen-body STOP** (forbidden, ADR 0026 additive-only). Route AROUND by applying the correct rope via the REUSED 13.3b-1 helper in the NEW `_indexer_mlx`.
- **Implication**: when you call FROZEN `_csa_windowed_compressor_mlx:347` for the indexer compressor pooling, you may need to suppress/skip its baked-in rope (e.g. call it in a mode that doesn't apply rope, OR re-pool without rope then apply the correct compress-yarn-tail rope yourself). BA §0/Q2 says the NEW `_indexer_mlx` applies compress-yarn-tail via REUSED 13.3b-1 helper. If FROZEN `_csa_windowed_compressor_mlx` cannot be called without its baked-in rope (no rope-suppress mode), you must REIMPLEMENT the Ca/Cb overlap pooling inline in `_indexer_mlx` (additive new code, NOT a FROZEN body edit) and apply the correct rope. BA Q1 documents the exact Ca/Cb overlap recipe (`new_kv[:,:,:rate] = chunk_kv[...,:head_dim]` Ca=leading 128; `new_kv[:,:,rate:] = chunk_kv[...,head_dim:]` Cb=trailing 128).

## Authoritative contracts — READ IN FULL BEFORE CODING (in order)
1. `agent-output/cmux-13-3b/requirements-13-3b-2a.md` — **§0 GO verdict** (top-k solvable, set-based AC) + §A Q1-Q7 LOCKED + §B AC1-AC2 + §C blast-radius + §D STOP. This is your SPEC.
2. `agent-output/cmux-13-3b/architecture.md` §0-§5 — esp. §2.1 reuse map, §3.2 (`_indexer_scorer_mlx`), §3.3 (`_indexer_mlx`), §8 (FROZEN must-not-edit-body list).
3. `agent-output/cmux-13-3b/subslice-breakdown.md` 13.3b-2a row + 13.3b-2 SPLIT note.
4. `agent-output/cmux-13-3b/requirements-13-3b-1.md` — the 13.3b-1 BA pattern + tolerance conventions (mirror: 1e-3 cos/sin, 1e-2 rotated → 1e-3 scores, exact for -1 sentinel mask, set-based for indices).
5. `docs/adr/0026-csa-hca-indexer-mlx-real-port.md`.

## Scope — EXACTLY what you implement (BA §C blast-radius confirmed)
### File 1: FROZEN `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` — ADDITIVE ONLY
- **NEW helpers** in the 13.3b module section (beside `_hca_compressor_mlx:591`/`_attention_real_mlx:658`):
  1. `_indexer_scorer_mlx(q, compressed_kv, hidden, *, index_n_heads, index_head_dim) -> scores` per Architect §3.2 + BA Q3:
     - `scores = relu(q @ compressed_kv.T) * index_head_dim**-0.5` `[B,S,H,T]` (H=index_n_heads=64, T=n_compressed_windows)
     - `weights = (hidden @ weights_proj.T) * index_n_heads**-0.5` `[B,S,H]`
     - `out = sum(scores * weights[...,None], axis=2)` `[B,S,T]`
     - fp32 accum (cast inputs to float32, return fp32 — downstream topk wants fp32).
  2. `_indexer_mlx(args, hidden, q_residual, weights, *, position_ids, index_topk) -> top_k_indices` per Architect §3.3 + BA Q1/Q2/Q4:
     - **indexer compressor**: Ca/Cb OVERLAP windowed pooling at `index_head_dim=128`, rate=4. BA Q1: `new_kv[:,:,:rate] = chunk_kv[...,:head_dim]` (Ca=leading 128 of 256), `new_kv[:,:,rate:] = chunk_kv[...,head_dim:]` (Cb=trailing 128). REUSE FROZEN `_csa_windowed_compressor_mlx:347` IF it can be called without its baked-in rope; OTHERWISE reimplement the Ca/Cb overlap pooling inline (additive new code, NOT a FROZEN body edit). Produces `compressed_kv [B,1,n_win,128]`.
     - **RoPE** (LANDMINE, see above): apply compress-yarn-tail over TRAILING `qk_rope_head_dim=64` of `index_head_dim=128` via REUSED 13.3b-1 `_compress_rope_yarn_tail_tables_mlx:545` + `_apply_rope_tail_mlx`. For compressed_kv: positions = `arange(n_win)*rate`. For q: positions = `position_ids`.
     - `q = (q_residual @ wq_b.T).reshape(B,S,index_n_heads=64,index_head_dim=128)`; apply compress-yarn-tail rope.
     - `scores = _indexer_scorer_mlx(q, compressed_kv, hidden, ...)` `[B,S,T]`.
     - **top-k + sentinel + future-mask** (BA Q4): `causal_threshold = (position_ids+1)//rate`; future mask scores→`-inf`; `top_k = min(index_topk=512, T)`; `top_k_indices = mx.argsort(-scores, axis=-1)[...,:top_k]` DESC (or argpartition + sort); `-1` sentinel: `invalid = top_k_indices >= causal_threshold`; `return mx.where(invalid, -1, top_k_indices)`. Shape `[B,S,top_k]`.
- **NO dispatch branch** (no consumer yet — 13.3b-2b wires CSA attention). **NO FROZEN body edits** (additive ONLY).

### Files 2-3: NEW tests (TDD RED-first)
2. `tests/test_indexer_scorer_mlx_parity.py` (AC1): `_indexer_scorer_mlx` output `[B,S,T]` matches torch `DeepseekV4IndexerScorer.forward:455` within tol at real dims. BA Q3 fp32 accum + weights_proj shape pinned.
3. `tests/test_indexer_mlx_parity.py` (AC2): `_indexer_mlx` top-k indices match torch `DeepseekV4Indexer.forward` indices (incl `-1` sentinels for future picks) on real-dim fixture. **AC asserts SET of non-sentinel picks + their scores == torch set+scores within tol; NOT exact index order at ties** (BA Q6 tie-break). Q1 compressor reuse + Q2 rope landmine + Q4 sentinel recipe pinned.

## §A — LOCKED spec (BA `requirements-13-3b-2a.md` §A — follow EXACTLY)
- **Q1 indexer compressor**: Ca/Cb overlap, rate=4, out_dim=index_head_dim=128. Real ckpt keys `attn.indexer.compressor.*`. ONE layer-2 safetensors header cross-check (shape/name assert only, no tensor load).
- **Q2 rope**: compress-yarn-tail TRAILING-64 of 128; REUSE `_compress_rope_yarn_tail_tables_mlx:545`; positions `arange(n_win)*rate` for compressed_kv, `position_ids` for q. (LANDMINE above.)
- **Q3 scorer**: fp32 accum, `weights_proj` shape from ckpt, return fp32.
- **Q4 top-k recipe**: causal_threshold, future→-inf, top_k=min(index_topk,T), argsort desc, `-1` sentinel via `where(invalid, -1, ...)`.
- **Q5 standalone**: no wiring, no dispatch branch, HCA path UNCHANGED.
- **Q6 fixtures + anti-circularity**: synthesize real dims; torch ref from same synthesized weights (ADR 0007 §4). Tie-break set-based.
- **Q7 STOP conditions**: see STOP-ESCALATE below.

## §B — Acceptance criteria (TDD red-first)
- **AC1**: `test_indexer_scorer_mlx_parity.py` GREEN.
- **AC2**: `test_indexer_mlx_parity.py` GREEN — set-of-valid-picks + scores, NOT tie-order.
- **regression**: 13.3b-1 (AC0-AC3 incl `_compress_rope_yarn_tail_tables_mlx` reuse) + 13.3a nn (12 incl backward AC) + 13.2 FP4 (9) + tiny CSA (11.14/11.15) all GREEN. Full suite: 1 RED (Test #8 carve-out, pre-existing) / 561+2new PASS / 13 SKIP / 0 introduced RED.
- **byte-intactness**: FROZEN bodies byte-intact via source hash + AST. `git diff --check` clean. sha-pin cascade SLICE-INVARIANT: advance exactly the right count of sha16 pin sites (deepseek_v4.py edited additively).

## STOP-ESCALATE (BA §A Q7 + Architect §9 — binding)
Emit `{"status":"error","role":"Coder",...}` + write `agent-output/cmux-13-3b/coder-13-3b-2a-stop.md` if:
- **(S-rope-indexer)**: indexer q-rope/ckv-rope needs a contract FROZEN can't supply additively (BA recon said NO — REUSE 13.3b-1 helper; STOP only if code proves wrong).
- **(S-frozen-body)**: any FROZEN symbol body (`_csa_windowed_compressor_mlx:347` even though reused — the body must stay byte-identical; if you can't call it without baked-in rope, REIMPLEMENT inline in `_indexer_mlx`, do NOT edit the FROZEN body; `_attention_mlx`, `_csa_*`, `_hyperconnection_mlx`, parity `Model`, `sanitize_weights`, `_compress_rope_yarn_tail_tables_mlx:545`, `_apply_rope_*`, `_rope_*_tables_mlx`, 9 ADR-0017 forbidden) would need EDIT → STOP. ADDITIVE ONLY.
- **(S-tiny-regression)**: tiny CSA fixtures (11.14/11.15) go RED → STOP.
- **(S-parity-topk)**: `_indexer_mlx` top-k indices (set-based AC) cannot match torch within tol after best-effort → STOP with failing assertion + torch-vs-MLX delta.
- **(S-backward)**: 13.3a-3 backward AC goes RED → STOP.

## Pre-flight read list (READ in this order)
1. `agent-output/cmux-13-3b/requirements-13-3b-2a.md` (SPEC — §0 verdict, §A Q1-Q7, §B AC1-AC2, §C blast-radius, §D STOP).
2. `agent-output/cmux-13-3b/architecture.md` (§2.1 reuse map, §3.2 scorer, §3.3 indexer, §8 frozen list).
3. `agent-output/cmux-13-3b/subslice-breakdown.md` 13.3b-2a row.
4. `agent-output/cmux-13-3b/requirements-13-3b-1.md` (tolerance conventions to mirror).
5. `docs/adr/0026-csa-hca-indexer-mlx-real-port.md`.
6. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` — FROZEN. Read: the 13.3b-1 new section (`_compress_rope_yarn_tail_tables_mlx:545`, `_hca_compressor_mlx:591`, `_attention_real_mlx:658`), `_csa_windowed_compressor_mlx:347` (the indexer compressor — CHECK if it has a rope-suppress mode or baked-in rope), `_csa_indexer_mlx:436` (tiny — understand contract, its full-channel rope is WRONG for real), `_apply_rope_tail_mlx`, `_rope_tail_tables_mlx`, `_apply_rope_full_mlx`, `_rope_full_tables_mlx`, `_csa_config_error:317`, FROZEN imports.
7. `python-envs/mlx/.venv/lib/python3.13/site-packages/transformers/models/deepseek_v4/modeling_deepseek_v4.py` — torch ref. Lines 446-588 (`IndexerScorer.forward:455` + `Indexer.forward:511`, esp 567-586 top-k+sentinel). Use `ctx_execute_file` (keep 1526 lines out).
8. `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json` (index_n_heads=64, index_head_dim=128, index_topk=512, rate=4 via compress_ratios, qk_rope_head_dim=64).
9. The 13.3b-1 test files (`tests/test_compress_rope_oracle.py`, `tests/test_attention_real_mlx_hca_parity.py`) — mirror anti-circularity pattern.

## Build/test workflow
1. `source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first).
2. **FIRST**: probe FROZEN `_csa_windowed_compressor_mlx:347` — does it have a rope-suppress mode or baked-in full-channel rope? (via `ctx_execute_file` grep). This determines Q1 strategy (reuse vs reimplement-inline). If baked-in rope + no suppress mode → reimplement Ca/Cb overlap pooling inline in `_indexer_mlx`.
3. Write AC1 RED → implement `_indexer_scorer_mlx` → GREEN.
4. Write AC2 RED → implement `_indexer_mlx` (compressor + rope + scorer + top-k+sentinel) → GREEN.
5. Full suite: `python3 -m pytest tests/ --no-header -q | tail -5`. Confirm 561+2new/1/13, 0 introduced RED.
6. FROZEN byte-intactness: source-hash + AST on the frozen-body symbol set. `git diff --check`.
7. sha-pin cascade: advance exactly the right count of sha16 pin sites.

## Deliverables
1. FROZEN `deepseek_v4.py` additive edits (2 NEW helpers; NO body edits; NO dispatch branch).
2. 2 NEW test files (AC1-AC2).
3. `agent-output/cmux-13-3b/coder-13-3b-2a-notes.md` — what you did, AC1-AC2 results, parity deltas, the rope-landmine resolution (did you reuse FROZEN windowed w/ suppress, or reimplement Ca/Cb inline?), blast-radius surprises.
4. `.cmux-status/coder.done` (`{"status":"ok","role":"Coder"}`).
5. In-pane JSON: `{"status":"ok","role":"Coder"}` in surface:83 ONLY.
6. git commit (Story 13.3b-2a + ADR 0026 + AC results).

## Style
Caveman ultra default; byte-exact exempt. Use `ctx_execute`/`ctx_batch_execute` for torch/FROZEN probes. Cite §/Q/AC numbers.

MUST NOT: wire CSA attention (13.3b-2b), edit nn port (13.3b-3), write convert (13.3b-4), edit FROZEN bodies (additive ONLY), introduce C++. BEGIN NOW. The rope landmine is the #1 risk — resolve it FIRST via the probe in step 2 before writing `_indexer_mlx`.
