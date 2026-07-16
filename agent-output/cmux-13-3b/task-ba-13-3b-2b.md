# Story 13.3b-2b — BA Task Brief (THIN: CSA attention wiring — compressor + block_bias + KV-append + multi-head)

## Role
BA (anthropic/claude-opus-4-8 · high, surface:81, fresh-context via /new). THIN — Architect r0 designed §3.4 (`_attention_real_mlx` unified CSA/HCA branch) + §3.5 (RoPE parity) + §2.1 reuse map in `agent-output/cmux-13-3b/architecture.md`. Translate to LOCKED Q-list + acceptance tests + blast-radius + STOP-rules. Do NOT re-design.

## Slice context
- **Story**: 13.3b-2b (SECOND half of SPLIT 13.3b-2). CSA attention wiring. Consumes `_indexer_mlx` (shipped 13.3b-2a, HEAD `a2c20b0`) to build `block_bias`; uses CSA compressor real keys (cr=4, multi-head, overlap windows); appends compressed_kv to KV axis; multi-head attention with REUSED cr=0 math.
- **Predecessor**: 13.3b-2a complete (`a2c20b0`). `_indexer_mlx` (standalone) + `_indexer_scorer_mlx` GREEN. `_compress_rope_yarn_tail_tables_mlx:545` + `_apply_rope_tail_mlx` (13.3b-1) available + proven.
- **Estimated**: 4 dev-days.
- **13.3b-3 owns**: the dispatch branch in `_attention_mlx` (cr=4→CSA, cr=128→HCA) + GroupedLinear (`_grouped_linear_mlx` o_groups=8) + AttentionNN guard lift + integration. 13.3b-2b produces the CSA attention branch as a HELPER (additive), NOT the dispatch. 13.3b-3 wires it.

## ⚠️ ROPE LANDMINE (carryover from 13.3b-2a — MUST re-adjudicate for CSA compressor)
BA 13.3b-2a discovered: FROZEN `_csa_windowed_compressor_mlx:347` has a BAKED-IN full-channel plain-theta rope (`_rope_full_tables_mlx` + `_apply_rope_full_mlx`) that MISMATCHES torch real compress-rope (yarn, factor=16, theta=160000, trailing `qk_rope_head_dim=64`). The 13.3b-2a indexer resolved this by reimplementing the Ca/Cb overlap pooling inline + applying compress-yarn-tail via REUSED 13.3b-1 helpers.

**BA 13.3b-2b MUST pin**: does the CSA attention compressor (cr=4, the 21 real CSA layers, NOT the indexer) ALSO need compress-yarn-tail rope? Per Architect §3.5 (RoPE parity risk — torch compress rope uses theta=160000+yarn for ALL compress layers CSA+HCA), YES. So the CSA compressor at cr=4 has the SAME baked-in-rope problem as the indexer. BA must decide:
- **Option α**: reimplement the CSA compressor Ca/Cb overlap pooling inline in the new CSA attention helper (additive, NOT a FROZEN body edit), apply compress-yarn-tail via REUSED 13.3b-1 helpers — SAME pattern as 13.3b-2a indexer. (Architect §2.1 said "CSA compressor = FROZEN `_csa_windowed_compressor_mlx:347`" — but that was BEFORE BA 13.3b-2a discovered the baked-in-rope mismatch. BA must flag this Architect §2.1↔§3.5 tension and resolve: option α (reimplement inline, correct rope) is consistent with §3.5 parity.)
- **Option β**: reuse FROZEN `_csa_windowed_compressor_mlx:347` accepting wrong rope — REJECT (parity broken).

LOCK expected: Option α. BA confirms via torch `CSACompressor.forward` + `Attention.forward` (torch lines per critical context).

## 13.3b-2b scope (Architect r0 §3.4 — translate, don't re-derive)
Per architecture.md §3.4 + subslice-breakdown.md 13.3b-2b row:
1. **CSA branch in `_attention_real_mlx` OR a new `_csa_attention_real_mlx` helper** (Architect §3.4 says "unified `_attention_real_mlx`" — so likely a CSA BRANCH added to `_attention_real_mlx` next to the 13.3b-1 HCA branch). BA pin the exact shape: does Coder ADD a CSA branch to `_attention_real_mlx` (additive `if cr==4: ...` after the existing `if cr==128: HCA` branch), OR a NEW `_csa_attention_real_mlx` helper that 13.3b-3 dispatches to? Either is additive; BA pin which is cleaner + matches §3.4.
2. **CSA compressor**: Ca/Cb overlap pooling at cr=4, out_dim=head_dim=512, multi-head (vs HCA cr=128 single-head). BA Q1 pin the exact multi-head shape + Ca/Cb overlap layout (Ca=leading, Cb=trailing, 2*head_dim split). Real ckpt keys `attn.compressor.*` — shape/name assert via safetensors header on ONE real CSA layer (e.g. layer 2). NO tensor load.
3. **CSA ape TRANSPOSE**: torch `CSACompressor` uses `ape[:out_dim,:].T` — DIFFERENT from HCA's no-transpose (13.3b-1 Q2). BA Q2 pin the exact transpose semantics + the `ape` ckpt key + shape.
4. **block_bias build**: from `_indexer_mlx` top-k indices + `-1` sentinels → `block_bias [B,1,S,n_win]` causal. BA Q3 pin the exact gather/scatter: index `i` at query position `p` visible iff `i ∈ top_k_indices[p] AND i < (p+1)//rate`; sentinel `-1` → masked. Architect §3.4 step 2 (HCA branch) referenced `_hca_compressor_mlx` returning `block_bias`; for CSA the `block_bias` comes from the INDEXER output, not a compressor. BA pin the exact construction.
5. **KV append**: `kv_full = cat([kv, compressed_kv], axis=2)` — REUSE 13.3b-1 HCA step 3 pattern. BA Q4 confirm axis + ordering.
6. **Multi-head attention**: scores + sink + softmax + attend — REUSE FROZEN cr=0 math (same as 13.3b-1 HCA step 5). rope-undo + grouped-o in path (13.3b-1 HCA step 6 pattern; but CSA may differ on grouped-o — BA Q5 pin if CSA uses grouped-o like HCA or different o projection).

### blast-radius (Architect §4.2 + §8 — confirm + refine for CSA)
| # | File | Change |
|---|---|---|
| 1 | `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` | ADDITIVE ONLY: CSA branch in `_attention_real_mlx` (or NEW `_csa_attention_real_mlx` helper) + CSA compressor inline reimpl (if option α) + block_bias build. NO dispatch branch in `_attention_mlx` (13.3b-3 owns). NO FROZEN body edits. |
| 2 | NEW test files: CSA compressor parity test + CSA attention parity test. | NEW |

**NO nn port edit** (13.3b-3). **NO dispatch wiring** (13.3b-3). **NO convert** (13.3b-4). **NO GroupedLinear** (13.3b-3).

## §A — LOCKED Q-list (Coder MUST follow)

### Q1 — CSA compressor: multi-head shape + Ca/Cb overlap + real ckpt keys (§3.4, rope landmine)
LOCKED expected: option α (reimplement inline). BA pin:
- CSA compressor out_dim = head_dim = 512 (vs HCA out_dim=head_dim=512 single-head — wait, is CSA multi-head compressor or single-head? BA pin from torch `CSACompressor.forward` + real config `num_heads=64`, `o_groups=8`). The CSA compressor produces `compressed_kv [B, num_kv_heads=1, n_win, head_dim=512]`? Or multi-head `[B, H, n_win, head_dim]`? BA pin precisely.
- Ca/Cb overlap layout at cr=4: Ca=leading head_dim of 2*head_dim chunk, Cb=trailing. Rate=4 (vs HCA rate=128).
- Real ckpt keys `attn.compressor.*` — shape/name assert on layer 2. NO tensor load.
- **Rope (landmine)**: compress-yarn-tail TRAILING-64 of head_dim=512, via REUSED 13.3b-1 `_compress_rope_yarn_tail_tables_mlx:545` + `_apply_rope_tail_mlx`, AFTER the inline Ca/Cb pooling. Positions = `arange(n_win)*rate` for ckv. BA confirm by reading torch `CSACompressor.forward` rope application.

### Q2 — CSA ape transpose (DIFFERENT from HCA — the 13.3b-1 Q2 landmine's MIRROR)
LOCKED: torch CSA uses `ape[:out_dim,:].T` (transpose). HCA used `ape` directly (NO transpose). BA pin:
- The exact torch line (e.g. `CSACompressor:649` mirror — find the real CSA line).
- The `ape` ckpt key + shape `(head_dim, head_dim)` or `(out_dim, head_dim)`? BA pin from real ckpt.
- The transpose: `ape[:out_dim, :].T` → result shape. BA pin precisely so Coder doesn't confuse with HCA.

### Q3 — block_bias build from indexer top-k (§3.4 — the CSA-specific piece)
LOCKED: `_indexer_mlx` (13.3b-2a) returns `top_k_indices [B,S,top_k]` with `-1` sentinels. The CSA attention branch builds `block_bias [B,1,S,n_win]` (or shape BA pins) such that:
- For each query position `p`, the valid compressed windows are the non-sentinel entries in `top_k_indices[p]` (set-based, since 13.3b-2a AC2 is set-based).
- The `block_bias` at `[B,1,p,i]` = 0 if `i` is a valid pick, `-inf` (or large negative) otherwise.
- Combined with the sliding causal mask (cr=0 path), final mask = `cat([sliding_causal_mask, block_bias], axis=-1)`.
- BA pin the exact dtype (fp32? bf16?), the fill values (0 / -inf), and the scatter recipe. Also pin whether the block_bias is ADDED to scores or concatenated to the mask (Architect §3.4 step 4 HCA pattern: mask cat).

### Q4 — KV append (§3.4 step 3 — reuse 13.3b-1 HCA pattern)
LOCKED: `kv_full = cat([kv, compressed_kv], axis=2)` — axis=2 (the sequence/window axis). BA confirm ordering (kv first, compressed_kv appended) + that compressed_kv shape `[B,1,n_win,head_dim]` broadcasts against `kv [B,1,S,head_dim]` on the head axis or needs a reshape. BA pin.

### Q5 — multi-head attention + grouped-o in CSA path (§3.4 step 5-6 — reuse vs diff from HCA)
LOCKED: scores + sink + softmax + attend REUSE cr=0 math (same as 13.3b-1 HCA step 5). rope-undo + grouped-o: BA pin if CSA uses grouped-o (o_groups=8) LIKE HCA, or a DIFFERENT o projection. Read torch `Attention.forward` CSA branch. If grouped-o — REUSE 13.3b-1 HCA step 6. If different — pin the difference.

### Q6 — standalone + fixtures + anti-circularity (pattern from 13.3b-1/2a)
LOCKED: synthesize at real dims (`num_heads=64`, `head_dim=512`, `o_groups=8`, `q_lora_rank=1024`, `qk_rope_head_dim=64`, `cr=4`, `index_topk=512`, `index_n_heads=64`, `index_head_dim=128`, `hidden=4096`, `hc_mult=4`). ONE real CSA layer (layer 2) cross-check via safetensors header (shape/name assert, no tensor load). Anti-circularity: torch reference computes expected from the SAME synthesized weights/config (ADR 0007 §4, proven 13.3b-1/2a).

### Q7 — STOP conditions (Architect §9 + ADR 0026 + 13.3b-2a pattern)
LOCKED — emit `coder-stop.md` + `{"status":"error",...}` if:
- **(S-rope-csa)**: CSA compressor rope needs a contract FROZEN can't supply additively (BA recon: likely option α same as 13.3b-2a; STOP only if code proves wrong).
- **(S-frozen-body)**: any FROZEN symbol body (`_csa_windowed_compressor_mlx:347` even reused — byte-identical; `_attention_mlx` cr=0 body; `_attention_real_mlx` HCA branch from 13.3b-1 — the HCA branch body MUST stay byte-identical, CSA branch ADDED additively; `_csa_*`, `_hyperconnection_mlx`, `_hyperhead_mlx`, `_hca_compressor_mlx`, `_indexer_mlx`, `_indexer_scorer_mlx` (13.3b-2a — byte-identical), parity `Model`, `sanitize_weights`, `_compress_rope_yarn_tail_tables_mlx`, `_apply_rope_*`, 9 ADR-0017 forbidden) would need EDIT → STOP. ADDITIVE ONLY.
- **(S-tiny-regression)**: tiny CSA fixtures (11.14/11.15) go RED → STOP.
- **(S-parity-csa)**: CSA attention output cannot match torch within tol after best-effort → STOP with failing assertion + torch-vs-MLX delta.
- **(S-backward)**: 13.3a-3 backward AC goes RED → STOP.

## §B — acceptance criteria → trackable tests (TDD red-first)
- **AC1** — `test_csa_compressor_real_mlx_parity.py` (NEW): CSA compressor (inline Ca/Cb + correct compress-yarn-tail rope) output matches torch `CSACompressor.forward` within tol at real dims. Q1 + Q2 + rope.
- **AC2** — `test_csa_attention_real_mlx_parity.py` (NEW): CSA attention branch (compressor + block_bias from `_indexer_mlx` + KV-append + multi-head) output matches torch `Attention.forward` CSA layer within tol at real dims. Q3 + Q4 + Q5 + Q6 anti-circ.
- **regression**: 13.3b-1 (AC0-AC3) + 13.3b-2a (AC1-AC2 incl `_indexer_mlx` reuse) + 13.3a nn (12 incl backward AC) + 13.2 FP4 (9) + tiny CSA (11.14/11.15) all GREEN. Full suite: 1 RED (Test #8) / 563+2new PASS / 13 SKIP / 0 introduced RED.
- **byte-intactness**: FROZEN bodies byte-intact via source hash + AST. `git diff --check` clean. sha-pin cascade SLICE-INVARIANT: deepseek_v4.py edited additively → advance exactly the right count of sha16 pin sites (Reviewer verifies count — use the rigorous whole-tree scan, NOT Test Manager 13.3b-1/2a's grep-subset).

## Deliverables
1. `agent-output/cmux-13-3b/requirements-13-3b-2b.md` — §A Q1-Q7 LOCKED + §B AC1-AC2 + blast-radius + STOP-rules + rope-landmine re-adjudication (option α confirmed). Do NOT design (Architect owns §3.4).
2. `.cmux-status/ba.done` (`{"status":"ok","role":"BA"}`).
3. In-pane JSON: `{"status":"ok","role":"BA"}` in surface:81 ONLY.
4. `docs/backlog.md` 13.3b-2b row update.

**STOP-ESCALATE** if AC2 (CSA attention parity) reveals an unsolvable FROZEN-primitive gap (e.g. CSA compressor + block_bias cannot match torch). Write `ba-stop.md` + `{"status":"error",...}`.

## Pre-flight read list (READ in this order)
1. `agent-output/cmux-13-3b/architecture.md` §0-§5 (esp. §2.1 reuse map — note the §2.1↔§3.5 tension on CSA compressor reuse; §3.4 unified `_attention_real_mlx` CSA+HCA; §3.5 RoPE parity; §8 frozen list).
2. `agent-output/cmux-13-3b/subslice-breakdown.md` 13.3b-2b row + 13.3b-2 SPLIT note + 13.3b-3 fence.
3. `agent-output/cmux-13-3b/requirements-13-3b-2a.md` (BA pattern + Q1-Q7 structure + rope-landmine resolution proven — mirror for CSA).
4. `agent-output/cmux-13-3b/requirements-13-3b-1.md` (Q2 ape-transpose landmine for HCA = NO; CSA mirror = YES transpose).
5. `docs/adr/0026-csa-hca-indexer-mlx-real-port.md`.
6. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` — FROZEN. Read: `_attention_real_mlx:658` (13.3b-1 HCA branch — the CSA branch will be ADDED here or beside it), `_csa_windowed_compressor_mlx:347` (the reused-or-bypassed CSA compressor), `_indexer_mlx:620` + `_indexer_scorer_mlx:591` (13.3b-2a — consumed for block_bias), `_compress_rope_yarn_tail_tables_mlx:545` + `_apply_rope_tail_mlx` (13.3b-1 reused), `_hca_compressor_mlx:591` (HCA pattern to mirror for CSA compressor), `_apply_rope_full_mlx`/`_rope_full_tables_mlx` (the WRONG baked-in rope to avoid), `_csa_config_error:317`, FROZEN imports.
7. `python-envs/mlx/.venv/lib/python3.13/site-packages/transformers/models/deepseek_v4/modeling_deepseek_v4.py` — torch ref. Read: `CSACompressor:589-754` (esp. forward + the `ape[:out_dim,:].T` transpose + rope application), `Attention.forward:801-875` (CSA branch — the block_bias build from indexer + KV-append + multi-head). Use `ctx_execute_file` (keep 1526 lines out).
8. `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json` — real config (compress_ratios array: 21×cr=4 CSA layers; num_heads=64, o_groups=8, head_dim=512, qk_rope_head_dim=64, index_topk=512, index_n_heads=64, index_head_dim=128, hidden=4096, q_lora_rank=1024, hc_mult=4).
9. The 13.3b-1/2a test files — mirror the anti-circularity pattern (torch ref from same synthesized weights).

## Style
Caveman ultra default; byte-exact exempt (code/paths/line numbers/torch:line/FROZEN:line/ADR cites verbatim). Cite §/Q/AC numbers. Use `ctx_execute`/`ctx_batch_execute` for torch/FROZEN/ckpt-header probes. BEGIN NOW.
