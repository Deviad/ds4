# Story 13.3b-2a — BA Task Brief (THIN: Indexer real path — top-k + -1 sentinel + future-mask)

## Role
BA (anthropic/claude-opus-4-8 · high, surface:81, fresh-context via /new). THIN — Architect r0 already designed §3.2 (`_indexer_scorer_mlx`) + §3.3 (`_indexer_mlx`) in `agent-output/cmux-13-3b/architecture.md`. Translate to LOCKED Q-list + acceptance tests + blast-radius + STOP-rules. Do NOT re-design.

## Slice context
- **Story**: 13.3b-2a (FIRST half of the SPLIT 13.3b-2 per Architect subslice-breakdown STOP-rule: 13.3b-2 was 8d → split 4+4). 13.3b-2a = Indexer only (`_indexer_scorer_mlx` + `_indexer_mlx`, top-k + sentinel + future-mask parity vs torch). 13.3b-2b = CSA attention wiring (compressor real keys + block_bias build + multi-head) — LATER slice.
- **Predecessor**: 13.3b-1 complete (HEAD `1f5d8e4`). HCA path + RoPE oracle + yarn-tail compress-rope helper (`_compress_rope_yarn_tail_tables_mlx:545`) all GREEN. The indexer q-rope will likely REUSE this helper — BA to confirm.
- **Estimated**: 4 dev-days (per subslice-breakdown 13.3b-2a).

## 13.3b-2a scope (Architect r0 §3.2 + §3.3 — translate, don't re-derive)
Per subslice-breakdown.md 13.3b-2a row:
1. **`_indexer_scorer_mlx` (§3.2)**: port torch `DeepseekV4IndexerScorer.forward:455`.
   - `scores = relu(q @ compressed_kv.T) * index_head_dim**-0.5` → `[B,S,H,T]`
   - `weights = (hidden @ weights_proj.T) * index_n_heads**-0.5` → `[B,S,H]`
   - return `sum(scores * weights[...,None], axis=2)` → `[B,S,T]`
   - fp32 accum (torch `.float()`)
2. **`_indexer_mlx` (§3.3)**: port torch `DeepseekV4Indexer.forward:511`.
   - indexer compressor = FROZEN `_csa_windowed_compressor_mlx:347` (Ca/Cb overlap, out_dim=index_head_dim=128, rate=4). REUSE.
   - `q = (q_residual @ wq_b.T).reshape(B,S,index_n_heads,index_head_dim)`; full RoPE at position_ids, compress rope.
   - `scores = _indexer_scorer_mlx(...)` → `[B,S,T]`
   - causal_threshold = `(position_ids+1)//rate`; future_mask scores→`-inf`; `top_k = min(index_topk, T)`
   - `top_k_indices = topk(scores, k).indices`; mark picks `>= causal_threshold` with `-1` sentinel. Port torch:572-585.
   - MLX has no native masked `topk`-with-sentinel: emulate via `mx.argpartition`/`mx.argsort` desc + gather + where. Coder owns exact MLX op choice; AC = parity vs torch indices.
   - Returns `top_k_indices` (shape `[B,S,top_k]` with `-1` sentinels for future picks).

### blast-radius (Architect §4.2 + subslice-breakdown — confirm)
| # | File | Change |
|---|---|---|
| 1 | `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` | ADDITIVE ONLY: NEW helpers `_indexer_scorer_mlx` + `_indexer_mlx` in the 13.3b module section (beside `_hca_compressor_mlx`/`_attention_real_mlx`). NO dispatch branch yet (the CSA attention that CONSUMES `_indexer_mlx` is 13.3b-2b). NO FROZEN body edits. |
| 2 | NEW test files: IndexerScorer parity test + Indexer parity test (top-k indices incl `-1` sentinels). | NEW |

**NO nn port edit** (13.3b-3 owns AttentionNN guard lift). **NO CSA attention wiring** (13.3b-2b). **NO convert** (13.3b-4). Helper-level + functional tests only.

## §A — LOCKED Q-list (resolutions Coder MUST follow)

### Q1 — indexer compressor reuse: exactly which FROZEN helper + what dims/keys? (§3.3)
LOCKED: FROZEN `_csa_windowed_compressor_mlx:347` (Ca/Cb overlap, rate=4) REUSED with `out_dim=index_head_dim=128`. BA must pin:
- Does the indexer compressor use the SAME Ca/Cb overlap semantics as the CSA compressor (torch `DeepseekV4Indexer.forward:511` calls its own compressor)? Verify the overlap layout (2*index_head_dim split into Ca/Cb) matches FROZEN windowed helper.
- Real ckpt keys for the indexer compressor: `attn.indexer.compressor.*` — confirm key names + shapes (BA cross-check via safetensors header on ONE real CSA layer, e.g. layer 2). NO tensor load, shape/name assert only.
- The indexer compressor produces `compressed_kv` at `index_head_dim=128` → `[B,1,n_win,128]`. Confirm this is the `compressed_kv` fed to `_indexer_scorer_mlx`.

### Q2 — indexer q computation: wq_b projection + reshape + RoPE — full or trailing? yarn? (§3.3)
LOCKED: `q = (q_residual @ wq_b.T).reshape(B,S,index_n_heads=64,index_head_dim=128)`. RoPE applied. BA MUST pin:
- **RoPE scope**: does indexer q rope cover FULL `index_head_dim=128` or only a TRAILING slice (analogous to qk_rope_head_dim=64 for the main attention)? Read torch `DeepseekV4Indexer.forward:511` + the RoPE application. There may be a separate `index_qk_rope_head_dim` config field — check real config.json.
- **RoPE type**: compress-rope (yarn, factor=16, theta=160000) — SAME as 13.3b-1 HCA? Or a DIFFERENT rope contract for the indexer? If same → REUSE `_compress_rope_yarn_tail_tables_mlx:545` (13.3b-1 helper). If different (e.g. full-channel not trailing, or different theta) → a new variant helper may be needed (FLAG for Coder, evaluate STOP).
- **Positions**: `position_ids` (all query positions) — NOT block-start positions (those were for the compressed_kv). Confirm from torch.

### Q3 — indexer scorer math: exact fp32 accumulation + weight broadcast (§3.2)
LOCKED per Architect §3.2:
- `scores = relu(q @ compressed_kv.T) * index_head_dim**-0.5` `[B,S,H,T]` (H=index_n_heads=64, T=n_compressed_windows)
- `weights = (hidden @ weights_proj.T) * index_n_heads**-0.5` `[B,S,H]`
- `out = sum(scores * weights[...,None], axis=2)` `[B,S,T]`
- fp32 accum (torch `.float()` cast before matmul/sum). BA LOCK the MLX fp32 strategy: cast inputs to float32 for the matmuls, cast output back to BF16 (or keep fp32 for downstream topk — downstream is topk which wants fp32 anyway).
- Verify `weights_proj` shape `(index_n_heads, hidden)` → wait, torch `weights_proj` is `[index_n_heads, hidden]` or `[hidden, index_n_heads]`? BA pin from real ckpt keys (`attn.indexer.weights_proj.weight`).

### Q4 — top-k + -1 sentinel + future-mask: exact recipe (§3.3, the hard part)
LOCKED per Architect §3.3 + torch `DeepseekV4Indexer.forward:572-585`:
- `causal_threshold = (position_ids + 1) // rate` (rate=4 for CSA indexer). A compressed window at index `i` is "future" relative to query at `position p` iff `i >= (p+1)//rate`.
- `future_mask`: scores at future windows → `-inf` (so topk won't pick them).
- `top_k = min(index_topk, T)` where T = n_compressed_windows. Real config `index_topk=512`.
- `top_k_indices = topk(scores_masked, k).indices` (descending). MLX has NO native masked-topk-with-sentinel → Coder emulates via `mx.argpartition`/`mx.argsort` (desc) + gather + `mx.where` for the sentinel.
- **`-1` sentinel**: after topk, any selected index `>= causal_threshold` is marked `-1` (per torch:572-585). These are "no valid pick" sentinels the downstream CSA attention clamps (drops). BA confirm: is the sentinel applied to the INDEX value (so output has -1 where future), or is it a separate mask? Pin precisely from torch.
- **Tie-breaking**: torch topk ties — does MLX argpartition produce the same tie order? BA FLAG: tie-break parity may be a flake source; Coder should use a deterministic order (e.g. argsort desc with stable index order) andtolerate tie-order differences via a test that only asserts the SET of non-sentinel picks + their scores, not exact index order when scores tie. Propose: AC asserts `set(valid_indices) == torch_set` + `scores[valid] == torch_scores[valid]` within tol, NOT exact index order at ties.

### Q5 — standalone helper: does 13.3b-2a wire `_indexer_mlx` into anything? (§ scope boundary)
LOCKED: 13.3b-2a produces STANDALONE helpers `_indexer_scorer_mlx` + `_indexer_mlx` + functional tests against torch `IndexerScorer`/`Indexer`. It does NOT wire into `_attention_real_mlx` (that's 13.3b-2b CSA attention). NO dispatch branch added (no consumer yet). BA confirm: the 13.3b-1 HCA path is UNCHANGED (HCA has no indexer). The CSA branch of `_attention_real_mlx` is 13.3b-2b.

### Q6 — test fixtures + anti-circularity (§ anti-circularity + BA Q6 pattern from 13.3b-1)
LOCKED: synthesize at real dims (`index_n_heads=64`, `index_head_dim=128`, `index_topk=512`, `rate=4`, `hidden=4096`, `q_lora_rank=1024`, `o_groups=8`, `num_heads=64`, `qk_rope_head_dim=64`). ONE real CSA layer (layer 2) cross-check via safetensors header (shape/name assert only, no tensor load). Anti-circularity: torch reference computes expected from the SAME synthesized weights/config; NO second MLX primitive as oracle (ADR 0007 §4, proven legit in 13.3b-1).

### Q7 — STOP conditions (Architect subslice-breakdown + ADR 0026)
LOCKED — emit `coder-stop.md` + `{"status":"error",...}` if:
- **(S-rope-indexer)**: indexer q-rope needs a contract FROZEN can't supply additively (e.g. full-channel yarn where 13.3b-1 helper is trailing-only, AND a full-channel variant can't be derived additively) → STOP-ESCALATE. (BA recon: likely REUSE 13.3b-1 trailing helper, but pin via torch:511.)
- **(S-frozen-body)**: any FROZEN symbol body (`_csa_windowed_compressor_mlx:347` the reused compressor, `_attention_mlx`, `_csa_*`, `_hyperconnection_mlx`, parity `Model`, `sanitize_weights`, `_compress_rope_yarn_tail_tables_mlx:545` from 13.3b-1, `_apply_rope_*`, 9 ADR-0017 forbidden) would need EDIT (not additive new symbol) → STOP. ADDITIVE ONLY.
- **(S-tiny-regression)**: tiny CSA fixtures (11.14/11.15) go RED → STOP.
- **(S-parity-topk)**: `_indexer_mlx` top-k indices (incl `-1` sentinels) cannot match torch reference within the AC (set-of-valid-picks + scores, NOT tie-order) after best-effort → STOP with failing assertion + torch-vs-MLX delta.
- **(S-backward)**: 13.3a-3 backward AC goes RED → STOP (additive helpers must not break 13.3a nn port; the 13.3b-1 HCA path must stay GREEN).

## §B — Acceptance criteria → trackable tests (TDD red-first)
- **AC1** — `test_indexer_scorer_mlx_parity.py` (NEW): `_indexer_scorer_mlx` output `[B,S,T]` matches torch `DeepseekV4IndexerScorer.forward:455` within tol at real dims. Q3 fp32 accum + weights_proj shape pinned.
- **AC2** — `test_indexer_mlx_parity.py` (NEW): `_indexer_mlx` top-k indices match torch `DeepseekV4Indexer.forward` indices (incl `-1` sentinels for future picks) on real-dim fixture. AC asserts `set(valid_indices) == torch_set` + `scores[valid] == torch_scores[valid]` within tol; NOT exact index order at ties (Q4 tie-break). Q1 compressor reuse + Q2 rope + Q4 sentinel recipe pinned.
- **regression**: 13.3b-1 (AC0-AC3, including `_compress_rope_yarn_tail_tables_mlx` reuse if Q2 confirms) + 13.3a nn (12 incl backward AC) + 13.2 FP4 (9) + tiny CSA (11.14/11.15) all GREEN. Full suite: 1 RED (Test #8) / 561+2new PASS / 13 SKIP / 0 introduced RED.
- **byte-intactness**: FROZEN bodies byte-intact via source hash + AST. `git diff --check` clean. sha-pin cascade SLICE-INVARIANT: if `deepseek_v4.py` edited (additive new helpers), advance exactly the right number of sha16 pin sites (per AGENTS.md — verify count with Reviewer).

## Deliverables
1. `agent-output/cmux-13-3b/requirements-13-3b-2a.md` — §A Q1-Q7 LOCKED + §B AC1-AC2 + blast-radius + STOP-rules. Do NOT design (Architect r0 owns §3.2/§3.3).
2. `.cmux-status/ba.done` (`{"status":"ok","role":"BA"}`).
3. In-pane JSON: `{"status":"ok","role":"BA"}` in surface:81 ONLY.
4. `docs/backlog.md` 13.3b-2a row update (scope LOCKED + AC + dev-days).

**STOP-ESCALATE** if AC2 (top-k parity) reveals an unsolvable FROZEN-primitive gap (e.g. MLX argpartition can't match torch topk semantics even with set-based AC). Write `ba-stop.md` + `{"status":"error","error":"AC2 top-k: <reason>","role":"BA"}`.

## Pre-flight read list (READ in this order)
1. `agent-output/cmux-13-3b/architecture.md` §0-§5 (esp. §3.2 `_indexer_scorer_mlx`, §3.3 `_indexer_mlx`, §2.1 reuse map, §8 frozen list). NOTE: §3.3 says "MLX has no native masked topk-with-sentinel: emulate via argpartition/argsort".
2. `agent-output/cmux-13-3b/subslice-breakdown.md` 13.3b-2a row + 13.3b-2 SPLIT note.
3. `agent-output/cmux-13-3b/requirements-13-3b-1.md` — the 13.3b-1 BA pattern (Q1-Q7 + AC0-AC3); mirror the structure + tolerance conventions (atol=1e-3 cos/sin, 1e-2 rotated; propose similar for indexer: 1e-3 scores, exact for -1 sentinel mask, set-based for indices).
4. `docs/adr/0026-csa-hca-indexer-mlx-real-port.md`.
5. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` — FROZEN. Read: the 13.3b-1 new section (`_compress_rope_yarn_tail_tables_mlx:545`, `_hca_compressor_mlx:591`, `_attention_real_mlx:658`), `_csa_windowed_compressor_mlx:347` (the reused indexer compressor), `_csa_indexer_mlx` (the TINY indexer — understand the tiny contract before the real port), `_csa_config_error:317`, FROZEN import list.
6. `python-envs/mlx/.venv/lib/python3.13/site-packages/transformers/models/deepseek_v4/modeling_deepseek_v4.py` — torch reference. Read lines 446-588 (`DeepseekV4IndexerScorer.forward:455` + `DeepseekV4Indexer.forward:511`, esp. 572-585 the top-k + sentinel recipe). Use `ctx_execute_file`/`ctx_execute` (keep 1526 lines out of context).
7. `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json` — real config (index_n_heads, index_head_dim, index_topk=512, rate=4 via compress_ratios, qk_rope_head_dim=64; ALSO check for `index_qk_rope_head_dim` or similar indexer rope field).
8. The 13.3b-1 test files (`tests/test_compress_rope_oracle.py`, `tests/test_attention_real_mlx_hca_parity.py`) — mirror the torch-reference-from-same-synthesized-weights anti-circularity pattern.

## Style
Caveman ultra default; byte-exact exempt (code/paths/line numbers/torch:line/FROZEN:line/ADR cites verbatim). Cite §-numbers + Q-numbers + AC numbers. Use `ctx_execute`/`ctx_batch_execute` for torch/FROZEN/ckpt-header probes. BEGIN NOW.
