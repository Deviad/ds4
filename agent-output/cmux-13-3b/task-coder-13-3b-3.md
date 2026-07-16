# Story 13.3b-3 — Coder Task Brief (TDD red→green; INTEGRATION: dispatch + AttentionNN guard lift)

## Role
Coder (openai-codex/gpt-5.5 · xhigh, surface:83, fresh-context via /new). Implement 13.3b-3: the INTEGRATION milestone — dispatch discriminator + AttentionNN guard lift (full §4.1 nn wiring). TDD red-first.

## Slice context
- **Story**: 13.3b-3 (INTEGRATION). After this, all 41/43 real layers run at real dims through `_attention_mlx` dispatch (3 cr=0 already worked pre-13.3b).
- **Predecessor**: 13.3b-2b complete (HEAD `aa93c5d`). 3 new CSA helpers shipped STANDALONE (no consumer wired yet): `_csa_compressor_real_mlx:819`, `_csa_block_bias_mlx:887`, `_csa_attention_real_mlx:907`. 13.3b-1 HCA `_attention_real_mlx:1005` + `_indexer_mlx:620` (13.3b-2a) all available.
- **Estimated**: 4 dev-days (BA: dispatch 0.5 + nn §4.1 wiring 2.5 + integration tests 1).
- **13.3b-4 owns**: real ckpt convert (key remap + load_weights + sanitize). 13.3b-3 does NOT touch convert.

## BA resolutions (CRITICAL — read)

### §0 GO verdict — **GO** (no `ba-stop.md`)
- **S-discriminator does NOT fire**: a CLEAN discriminator EXISTS today; no config flag needed.
- **S-frozen-body does NOT fire**: dispatch edit site is the 13.3b-1-added dispatch branch (ADR 0026 §Decision-1 SANCTIONED additive site, NOT a FROZEN body); cr=0 body + all helper bodies + tiny path stay byte-identical.

### Q1 — Dispatch discriminator — RESOLVED, PINNED (CRITICAL)
`_csa_config_error:317` ALREADY distinguishes tiny from real via guards at `:320` (heads/o_groups gate: `if args.num_attention_heads != 1 or args.o_groups != 1: return "...proven only for single-head o_groups=1 tiny fixtures"`).
- Tiny config (`tests/test_13_3b_1_dispatch_additive.py` `_tiny_csa_args`): `hidden_size=4`, `num_attention_heads=1`, `o_groups=1`, `head_dim=4`, `compression_ratio=4`. `_csa_config_error` returns **None** (valid tiny).
- Real config (`config.json`): `hidden_size=4096`, `num_attention_heads=64`, `o_groups=8`, `head_dim=512`, `num_key_value_heads=1`. Real cr=4 → `_csa_config_error` returns **NON-None** (heads≠1).
- **THE BUG**: real cr=4 layers currently route to `_attention_real_mlx` (HCA) at `:1202` — WRONG (HCA helper, not CSA). They do NOT silently hit the tiny path; they mis-route to HCA.

### Q1 EXACT additive dispatch shape (BA pinned — ONE inserted branch, tiny + cr=128 returns byte-identical)
At `_attention_mlx:1199-1202`, insert a NEW real-cr=4 branch BEFORE the HCA fallback:
```python
if args.compression_ratio != 0:
    if _csa_config_error(args) is None:                                       # :1200 tiny — UNCHANGED
        return _csa_attention_mlx(args, x, weights, index_topk=index_topk)   # :1201 byte-identical
    if args.compression_ratio == 4:                                           # NEW real cr=4 — CSA
        return _csa_attention_real_mlx(args, x, weights, index_topk=index_topk)
    return _attention_real_mlx(args, x, weights, index_topk=index_topk)       # :1202 cr=128 HCA byte-identical
```
`:1200`-`:1201` (tiny) byte-identical; current `:1202` (HCA) byte-identical. Real cr=4 now routes to `_csa_attention_real_mlx` instead of the HCA fallback. **Editing the 13.3b-1 dispatch branch IS allowed** — ADR 0026 §Decision-1 names the `_attention_mlx` dispatch branch as THE sanctioned additive edit site (NOT FROZEN body).

(BA Q1 also offers: if you prefer a `cr == 128` explicit HCA branch instead of `cr == 4` CSA branch, that's equivalent — pick whichever is cleaner; pin your choice in coder-notes.)

### Q2 — GroupedLinear (`_grouped_linear_mlx`) — RESOLVED: **NOT NEEDED (reused inline)**
Recon confirms: `for group in range(args.o_groups)` is ALREADY inline at `_attention_mlx:1247-1256` (FROZEN cr=0 body) AND inlined in `_attention_real_mlx` (13.3b-1) + `_csa_attention_real_mlx` (13.3b-2b). Both real branches' AC2 parity-vs-torch GREEN through the inline grouped-o (vs torch `DeepseekV4GroupedLinear.forward:328`). Extracting into a shared `_grouped_linear_mlx` would EDIT the FROZEN cr=0 inline loop → REJECTED (S-frozen-body). **13.3b-3 = dispatch + guard lift ONLY. NO new `_grouped_linear_mlx` helper.**

### Q3 — AttentionNN guard lift (`deepseek_v4_nn.py:153`) — PINNED (Architect §4.1 owns submodule layout)
**HONEST CONTRACT (BA Q3)**: removing the guard ALONE is INSUFFICIENT. `AttentionNN.__call__:203` already delegates `return _attention_mlx(self.config, x, weights)`. BUT the real cr≠0 dispatch (`_csa_attention_real_mlx` needs `compressor_*`, `indexer_wq_a`, `indexer_wq_b`; `_attention_real_mlx` needs `compressor_*`) would `KeyError` on the missing keys. So the guard lift = full §4.1 nn wiring:
1. **LIFT guard** `:153-154` (`compression_ratio != 0 → NotImplementedError`). Replace with: construct compressor/indexer submodules when `layer_types[layer_idx] != "sliding_attention"`.
2. **ADD submodules (CSA)** per Architect §4.1: `compressor` (wkv/wgate `nn.Linear`, `ape`+`norm.weight` params), `indexer.compressor` (same), `indexer.weights_proj` `nn.Linear`, `indexer.wq_b` `nn.Linear`.
3. **ADD submodules (HCA)** per §4.1: `compressor` only (wkv/wgate/ape/norm).
4. **`__call__`**: if sliding → existing `_attention_mlx` cr=0 (UNCHANGED); else build real weights dict (via `linear_weight(...)` helper — QuantizedLinear/LoRA-aware, reuse for compressor leaves) + `position_ids = mx.arange(S)` + delegate to `_attention_mlx` (which 13.3b-3 dispatch routes cr=4-real→CSA, cr=128→HCA).
5. Keep `linear_weight(...)` helper.

**Architect §4.1 owns the exact submodule/leaf layout** — read it and follow. BA pins ONLY: guard removed, cr≠0 no longer raises, `__call__` delegates to `_attention_mlx` with the real submodule leaves built into the weights dict.

### Q4 — Integration AC — PINNED
- **Dispatch routing** (extend `tests/test_13_3b_1_dispatch_additive.py`): add legs — real cr=4 routes to `_csa_attention_real_mlx` (monkeypatch spy / sentinel); real cr=128 routes to `_attention_real_mlx`; cr=4-tiny routes to `_csa_attention_mlx` byte-identical (existing leg — KEEP).
- **Parity-through-dispatch** (extend `tests/test_csa_attention_real_mlx_parity.py` + `tests/test_attention_real_mlx_hca_parity.py`): call through `_attention_mlx(...)` (dispatch) instead of the direct helper; assert identical to the direct `_csa_attention_real_mlx(...)` / `_attention_real_mlx(...)` output. Dispatch is a pure router ⇒ identical. Direct-helper-vs-torch parity ALREADY GREEN (13.3b-1 AC2 / 13.3b-2b AC2) ⇒ transitively real cr=4 + cr=128 match torch THROUGH the dispatch.
- **nn end-to-end** (NEW `tests/test_13_3b_3_nn_mixed_layers_forward.py`): `AttentionNN.__call__` over real-dim CSA + HCA + sliding layers (mixed `layer_types`) matches torch within tol. Anti-circularity: torch ref from SAME synthesized weights (ADR 0007 §4).
- **Tiny byte-identity regression**: 11.14/11.15 tiny CSA fixtures produce byte-identical output before/after the dispatch change (extend or add explicit test if absent).

## Scope — EXACTLY what you implement (BA §C blast-radius)

### File 1: FROZEN `deepseek_v4.py` — ADDITIVE ONLY (the §4.2 sanctioned dispatch edit site)
- Insert real-cr=4 branch in `_attention_mlx:1199-1202` per Q1 EXACT shape. Tiny path byte-identical. cr=128 HCA fallback byte-identical.
- **NO FROZEN body edits** to: cr=0 body (`_attention_mlx:1203+`), `_attention_real_mlx:1005` (HCA, 13.3b-1), `_csa_attention_real_mlx:907` (CSA, 13.3b-2b), `_csa_attention_mlx:515` (tiny), `_csa_config_error:317`, `_indexer_mlx:620`, `_indexer_scorer_mlx:591`, `_csa_compressor_real_mlx:819`, `_csa_block_bias_mlx:887`, `_hca_compressor_mlx`, `_compress_rope_yarn_tail_tables_mlx:545`, `_apply_rope_*`, `_rope_*_tables_mlx`, `_hyperconnection_mlx`, `_hyperhead_mlx`, `_causal_sliding_mask_mlx`, `sanitize_weights`, parity `Model`, FP4/i8 dequant, 9 ADR-0017 forbidden.
- **NO `_grouped_linear_mlx`** (Q2: not needed).

### File 2: `deepseek_v4_nn.py` — guard lift + §4.1 nn wiring (SANCTIONED nn-file edit, ADR 0025 owns nn file)
- `AttentionNN:135` — lift guard `:153-154`, add compressor/indexer submodules per Architect §4.1, `__call__` builds real weights dict + delegates to `_attention_mlx`.
- Keep `linear_weight(...)` helper. Keep cr=0 sliding path UNCHANGED.

### File 3+: NEW test files
- Extend `tests/test_13_3b_1_dispatch_additive.py` (dispatch routing legs).
- Extend `tests/test_csa_attention_real_mlx_parity.py` + `tests/test_attention_real_mlx_hca_parity.py` (through-dispatch parity).
- NEW `tests/test_13_3b_3_nn_mixed_layers_forward.py` (nn end-to-end mixed layers, anti-circ torch ref).
- NEW or extended tiny-byte-identity regression test.

## §B — Acceptance criteria (TDD red-first)
- **AC1** — dispatch routing: real cr=4 → `_csa_attention_real_mlx`; real cr=128 → `_attention_real_mlx`; cr=4-tiny → `_csa_attention_mlx` byte-identical; cr=0 → main body.
- **AC2** — tiny byte-identity regression: 11.14/11.15 tiny CSA byte-identical before/after.
- **AC3** — parity-through-dispatch: real cr=4 (CSA) + cr=128 (HCA) match torch within tol through `_attention_mlx` dispatch (transitive via direct-helper GREEN).
- **AC4** — AttentionNN guard lift: cr≠0 forward through `AttentionNN.__call__` no longer raises; delegates to `_attention_mlx`; nn end-to-end mixed-layers matches torch.
- **regression**: 13.3b-1 (AC0-AC3) + 13.3b-2a (AC1-AC2) + 13.3b-2b (AC1-AC2) + 13.3a nn (12 incl backward AC) + 13.2 FP4 (9) + tiny CSA (11.14/11.15) all GREEN. Full suite: 1 RED (Test #8 carve-out, pre-existing) / 565+new PASS / 13 SKIP / 0 introduced RED.
- **byte-intactness**: FROZEN bodies byte-intact via source hash + AST. tiny path byte-identical. `git diff --check` clean. sha-pin cascade SLICE-INVARIANT (deepseek_v4.py edited additively → advance right count; deepseek_v4_nn.py edits → advance if pinned).

## STOP-ESCALATE (BA §A Q5 — binding)
Emit `{"status":"error","role":"Coder",...}` + write `agent-output/cmux-13-3b/coder-13-3b-3-stop.md` if:
- **(S-discriminator)**: the clean `_csa_config_error`-based discriminator fails in code (BA recon said clean; STOP only if code proves wrong).
- **(S-frozen-body)**: dispatch fix requires editing the FROZEN cr=0 body or any FROZEN helper body (ADDITIVE ONLY; the 13.3b-1 dispatch branch `:1199-1202` IS the sanctioned additive site per ADR 0026 §Decision-1).
- **(S-tiny-regression)**: tiny CSA fixtures (11.14/11.15) go RED.
- **(S-integration-parity)**: real cr=4 or cr=128 cannot match torch within tol through the dispatch.
- **(S-nn-wiring)**: §4.1 submodule layout cannot supply the keys the real dispatch needs (would require FROZEN helper signature change).

## Pre-flight read list (READ in this order)
1. `agent-output/cmux-13-3b/requirements-13-3b-3.md` (SPEC — §0/§A Q1-Q5/§B AC1-AC4/§C blast-radius/§D STOP).
2. `agent-output/cmux-13-3b/architecture.md` §4.1 (AttentionNN submodule layout — CRITICAL), §4.2 (dispatch additive), §4.3 (GroupedLinear — confirm NOT needed), §3.4 (unified real path), §8 (frozen list). NOTE §4.2 shows dispatch as `cr≠0 → _attention_real_mlx` — BA Q1 REFINES: real cr=4 needs `_csa_attention_real_mlx`, insert that branch.
3. `agent-output/cmux-13-3b/subslice-breakdown.md` 13.3b-3 row.
4. `agent-output/cmux-13-3b/requirements-13-3b-1.md` (the 13.3b-1 dispatch contract + tiny-byte-identical via `_csa_config_error is None`).
5. `agent-output/cmux-13-3b/requirements-13-3b-2a.md` + `requirements-13-3b-2b.md` (predecessor scope).
6. `docs/adr/0026-csa-hca-indexer-mlx-real-port.md` (§Decision-1 sanctioned dispatch edit site; ADDITIVE ONLY; tiny byte-identical).
7. `docs/adr/0025-*.md` (nn file ownership).
8. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` — FROZEN. Read: `_attention_mlx:1190-1257` (dispatch `:1199-1202` + cr=0 body + grouped-o loop), `_csa_config_error:317` (6 guards — discriminator), `_attention_real_mlx:1005` (HCA), `_csa_attention_real_mlx:907` (CSA), `_csa_attention_mlx:515` (tiny), `_indexer_mlx:620`.
9. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py` — `AttentionNN:135-203` (guard `:153` + `__call__:203` delegation + `linear_weight` helper + existing submodule pattern from 13.3a).
10. `tests/test_13_3b_1_dispatch_additive.py` (the `_tiny_csa_args` fixture + dispatch routing patterns to extend).
11. `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json` (real config).
12. The 13.3b-1/2a/2b test files (mirror anti-circularity pattern).

## Build/test workflow
1. `source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first).
2. Write AC1 RED (dispatch routing legs) → insert real-cr=4 branch per Q1 EXACT shape → GREEN.
3. Write AC2 RED (tiny byte-identity) → confirm tiny still byte-identical → GREEN (should pass immediately if additive clean; if RED, you broke tiny).
4. Write AC3 RED (parity-through-dispatch) → confirm dispatch is pure router → GREEN.
5. Write AC4 RED (nn end-to-end mixed layers) → implement §4.1 AttentionNN wiring (guard lift + submodules + `__call__` delegation) → GREEN.
6. Full suite: `python3 -m pytest tests/ --no-header -q | tail -5`. Confirm 565+new/1/13, 0 introduced RED.
7. FROZEN byte-intactness: source-hash + AST on the frozen-body symbol set (CRITICAL: cr=0 body `_attention_mlx:1203+` byte-identical; tiny path byte-identical). `git diff --check`.
8. sha-pin cascade: advance right count.

## Deliverables
1. FROZEN `deepseek_v4.py` additive dispatch edit (real-cr=4 branch).
2. `deepseek_v4_nn.py` AttentionNN guard lift + §4.1 submodule wiring.
3. Extended + NEW test files (AC1-AC4).
4. `agent-output/cmux-13-3b/coder-13-3b-3-notes.md` — what you did, AC1-AC4 results, parity deltas, dispatch shape chosen (cr==4 CSA branch vs cr==128 HCA branch), §4.1 submodule layout, blast-radius surprises.
5. `.cmux-status/coder.done` (`{"status":"ok","role":"Coder"}`).
6. In-pane JSON: `{"status":"ok","role":"Coder"}` in surface:83 ONLY.
7. git commit (Story 13.3b-3 + ADR 0026/0025 + AC results).

## Style
Caveman ultra default; byte-exact exempt. Use `ctx_execute`/`ctx_batch_execute` for FROZEN/torch probes. Cite §/Q/AC numbers.

MUST NOT: write convert (13.3b-4), edit FROZEN bodies (additive ONLY; cr=0 body byte-identical; tiny path byte-identical), introduce `_grouped_linear_mlx` (Q2: not needed), introduce C++. BEGIN NOW. The §4.1 AttentionNN wiring is the bulk of the work (BA est. 2.5 of 4 dev-days) — read §4.1 + the existing 13.3a AttentionNN pattern carefully first.
