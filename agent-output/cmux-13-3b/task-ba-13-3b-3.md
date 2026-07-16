# Story 13.3b-3 — BA Task Brief (THIN: integration — dispatch + GroupedLinear + AttentionNN guard lift)

## Role
BA (anthropic/claude-opus-4-8 · high, surface:81, fresh-context via /new). THIN — Architect r0 designed §3.4 unified dispatch + §4.2 dispatch branch + §8 in `agent-output/cmux-13-3b/architecture.md`. Translate to LOCKED Q-list + acceptance tests + blast-radius + STOP-rules. Do NOT re-design.

## Slice context
- **Story**: 13.3b-3 (INTEGRATION milestone). After this, all 41/43 real layers run at real dims (3 cr=0 already worked pre-13.3b). Consumes 13.3b-1 (`_attention_real_mlx` HCA) + 13.3b-2a (`_indexer_mlx` for CSA block_bias) + 13.3b-2b (`_csa_attention_real_mlx` CSA).
- **Predecessor**: 13.3b-2b complete (HEAD `aa93c5d`). 3 new helpers shipped: `_csa_compressor_real_mlx:819`, `_csa_block_bias_mlx:887`, `_csa_attention_real_mlx:907` (all STANDALONE, no consumer wired yet).
- **Estimated**: 4 dev-days.
- **13.3b-4 owns**: real ckpt convert (key remap + load_weights + sanitize). 13.3b-3 does NOT touch convert.

## ⚠️ CRITICAL pre-flight finding (supervisor recon — BA MUST re-verify + pin)
The dispatch at `_attention_mlx:1199-1202` is **already wired** (13.3b-1 added it):
```python
1199:    if args.compression_ratio != 0:
1200:        if _csa_config_error(args) is None:           # cr=4 CSA config valid →
1201:            return _csa_attention_mlx(args, x, weights, index_topk=index_topk)  # TINY fixture CSA
1202:        return _attention_real_mlx(args, x, weights, index_topk=index_topk)     # HCA real (cr=128 or cr≠4)
```

**BUG (the integration gap 13.3b-3 must fix)**: a real CSA layer (cr=4, real dims hidden=4096/num_heads=64/head_dim=512) passes `_csa_config_error is None` (cr=4 valid CSA config) → routes to **`_csa_attention_mlx` (TINY fixture path)**, NOT to the new real `_csa_attention_real_mlx`. So real cr=4 layers are BROKEN at the dispatch today.

**13.3b-3 must add a real-dims discriminator** so:
- cr=4 + real dims → `_csa_attention_real_mlx` (the NEW real CSA helper, 13.3b-2b)
- cr=4 + tiny fixture → `_csa_attention_mlx` (tiny, FROZEN byte-identical — tiny path byte-identical is a hard contract per ADR 0026)
- cr=128 (HCA) → `_attention_real_mlx` (HCA, 13.3b-1)
- cr=0 → main cr=0 body (unchanged)

**BA Q1 pin the EXACT discriminator**: how to tell cr=4-real from cr=4-tiny at dispatch time. Options candidates:
- (a) dim-size check: e.g. `args.hidden_size >= 4096` or `args.num_attention_heads == 64` or `args.head_dim == 512` (real) vs tiny dims (hidden=64? 128?).
- (b) explicit flag in args/config (e.g. a `real_dims: bool` or `model_type` field).
- (c) `_csa_config_error` augmented to return "tiny-only" vs None for real (but that FROZEN body edit — REJECT, additive-only).

BA MUST probe the tiny fixture config (Story 11.14 test fixtures) vs real config (`/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json`) to pin the exact discriminator. Report the tiny config's `hidden_size`, `num_attention_heads`, `head_dim` and the real config's, and pin the discriminator predicate.

**ALSO**: the 13.3b-1 dispatch line 1200 was justified as "tiny path byte-identical via `_csa_config_error is None` guard" (ADR 0026). 13.3b-3 MUST preserve tiny byte-identity: when discriminator says tiny, route to `_csa_attention_mlx` UNCHANGED. The new real branch is ADDITIVE (an `elif` / additional check before line 1201, OR a refactor that preserves the tiny path exactly for tiny dims).

## ⚠️ GroupedLinear: already inline at `_attention_mlx:1239-1257`
Supervisor recon: the grouped-o loop is ALREADY inline in `_attention_mlx:1247-1256` (the cr=0 body):
```python
1239:    heads_per_group = args.num_attention_heads // args.o_groups
...
1247:    for group in range(args.o_groups):
...
1255:        low_rank_chunks.append(_linear_mlx(flat_group, weights["o_a_proj.weight"][row_start:row_end, :]))
1256:    low_rank = mx.concatenate(low_rank_chunks, axis=-1)
1257:    return _linear_mlx(low_rank, weights["o_b_proj.weight"])
```
The HCA `_attention_real_mlx` (13.3b-1) and CSA `_csa_attention_real_mlx` (13.3b-2b) both REUSE this pattern (per their notes). **BA Q2 pin**: does 13.3b-3 need to EXTRACT this loop into a NEW `_grouped_linear_mlx` helper (additive, FROZEN cr=0 body stays byte-identical — the loop in `_attention_mlx:1247-1256` is part of the cr=0 body which is FROZEN; extracting would EDIT the FROZEN body → REJECT)? OR do the real branches already inline-reuse the math (no new helper needed)? Architect §4.2 said "ONE dispatch branch ADDITIVE ONLY" — if grouped-o is already inlined in both real branches, NO new GroupedLinear helper is needed. BA pin: is `_grouped_linear_mlx` a NEW helper, or NOT NEEDED (reused inline)? STOP if Architect §4.2 said new helper but recon proves not needed → Architect § clarification.

## Scope — what 13.3b-3 implements (Architect §3.4 + §4.2 + §8 — translate, don't re-derive)
1. **Dispatch discriminator + branch** in `_attention_mlx:1199-1202` (ADDITIVE ONLY; tiny path byte-identical preserved). BA Q1 pins discriminator. Real cr=4 → `_csa_attention_real_mlx`; real cr=128 → `_attention_real_mlx`; tiny cr=4 → `_csa_attention_mlx` (UNCHANGED); cr=0 → main body (UNCHANGED).
2. **GroupedLinear** (`_grouped_linear_mlx` o_groups=8) — IF NEEDED (BA Q2). If grouped-o already inlined in real branches, no new helper. BA pin.
3. **AttentionNN guard lift** at `deepseek_v4_nn.py:153`: `if config.compression_ratio != 0: raise NotImplementedError("13.3a-1 AttentionNN supports only compression_ratio=0")` → LIFT to delegate cr≠0 to `_attention_mlx` (which now dispatches correctly). BA Q3 pin the exact guard semantics — what replaces the `NotImplementedError`? Likely: remove the guard, OR replace with `pass`/delegation. Confirm `AttentionNN.forward` already calls `_attention_mlx` at `:203` (per critical context) — so lifting the guard = remove the `if config.compression_ratio != 0: raise NotImplementedError` block. BA pin.
4. **Integration test**: full cr-dispatch — a real-config forward through one cr=4 CSA layer + one cr=128 HCA layer + one cr=0 layer, all at real dims, matches torch within tol. BA Q4 pin the AC.

### blast-radius (Architect §4.2 + §8 — confirm + refine)
| # | File | Change |
|---|---|---|
| 1 | `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` | ADDITIVE ONLY: dispatch discriminator + real cr=4 branch in `_attention_mlx:1199-1202`. Tiny path byte-identical. NO FROZEN body edits (the cr=0 body `_attention_mlx:1203+` stays byte-identical; `_attention_real_mlx`, `_csa_attention_real_mlx`, `_csa_attention_mlx`, `_csa_config_error`, `_indexer_mlx`, all FROZEN). + optional `_grouped_linear_mlx` (BA Q2 — if NEW). |
| 2 | `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py` | Guard lift at `:153`. BA Q3 pin exact edit. |
| 3 | NEW test files: integration dispatch test + tiny-byte-identical regression test. | NEW |

**NO convert** (13.3b-4). **NO FROZEN body edits**. **NO new real-branch helpers** (13.3b-1/2a/2b shipped them; 13.3b-3 only WIRES).

## §A — LOCKED Q-list

### Q1 — Dispatch discriminator (cr=4-real vs cr=4-tiny) — CRITICAL
LOCKED — BA pin exact predicate. Probe BOTH:
- tiny fixture config (Story 11.14 test fixtures — find in `tests/test_deepseek_v4_mlx_port.py` or wherever 11.14 lives)
- real config `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json`
Report tiny `hidden_size`/`num_attention_heads`/`head_dim`/`o_groups`/`compression_ratio` vs real. Pin the discriminator (e.g. `args.hidden_size >= 4096 and args.num_attention_heads == 64` → real; else tiny). Pin exact dispatch branch shape in `_attention_mlx:1199-1202`.

### Q2 — GroupedLinear (`_grouped_linear_mlx`) — NEW helper OR not needed?
LOCKED — BA pin. Recon: grouped-o loop ALREADY inline at `_attention_mlx:1247-1256` (FROZEN cr=0 body) AND inlined in `_attention_real_mlx` (13.3b-1) + `_csa_attention_real_mlx` (13.3b-2b). If both real branches already inline the grouped-o math, NO new `_grouped_linear_mlx` needed — 13.3b-3 = dispatch + guard lift only. If they DON'T (e.g. a shared helper would DRY them), then NEW `_grouped_linear_mlx` additive — but ONLY IF it doesn't require editing FROZEN bodies (the cr=0 inline loop is FROZEN). STOP + Architect § clarification if Architect §4.2 said new helper but recon proves the dispatch-only approach is right.

### Q3 — AttentionNN guard lift (deepseek_v4_nn.py:153)
LOCKED — BA pin exact edit. Read `deepseek_v4_nn.py:135-203` (AttentionNN class + forward). The guard `if config.compression_ratio != 0: raise NotImplementedError(...)` at `:153` — lift it. Pin: remove guard entirely? Replace with comment? Does `AttentionNN.forward` already delegate to `_attention_mlx` at `:203` (per critical context)? If yes, removing the guard lets cr≠0 flow to `_attention_mlx` (which 13.3b-3 dispatches correctly). BA pin.

### Q4 — Integration AC: full cr-dispatch at real dims
LOCKED — BA pin. AC3: a real-config forward through one cr=4 CSA layer + one cr=128 HCA layer + one cr=0 layer, all at real dims, matches torch within tol. OR (cleaner) extend the existing real-config parity tests (13.3b-1 HCA, 13.3b-2b CSA) to go through `_attention_mlx` dispatch (not direct helper call). BA pin exact AC shape. ALSO: tiny-byte-identical regression AC (the 11.14/11.15 tiny CSA fixtures MUST stay byte-identical through the dispatch — add an explicit test if not present).

### Q5 — STOP conditions
LOCKED — emit `ba-stop.md` + `{"status":"error",...}` if:
- **(S-discriminator)**: no clean discriminator between cr=4-real and cr=4-tiny (would need a config flag not present).
- **(S-frozen-body)**: dispatch fix requires editing the FROZEN cr=0 body or `_attention_real_mlx` HCA branch (ADDITIVE ONLY; the cr=0 body `_attention_mlx:1203+` and the 13.3b-1 dispatch lines 1199-1202 — wait, are those FROZEN now? 13.3b-1 added them; 13.3b-3 edits 1199-1202. BA clarify: is editing the 13.3b-1 dispatch branch allowed, or must 13.3b-3 add a NEW branch alongside?).
- **(S-tiny-regression)**: tiny CSA fixtures go RED.
- **(S-integration-parity)**: real cr=4 or cr=128 cannot match torch within tol through the dispatch.

## §B — Acceptance criteria (TDD red-first)
- **AC1** — dispatch discriminator test: cr=4-real routes to `_csa_attention_real_mlx`; cr=4-tiny routes to `_csa_attention_mlx`; cr=128 routes to `_attention_real_mlx`; cr=0 routes to main body. (Test via dispatch spy / output match.)
- **AC2** — tiny-byte-identical regression: 11.14/11.15 tiny CSA fixtures produce byte-identical output before/after the dispatch change.
- **AC3** — integration parity: real-config cr=4 (CSA) + cr=128 (HCA) + cr=0 layers match torch within tol through `_attention_mlx` dispatch.
- **AC4** — AttentionNN guard lift: cr≠0 forward through `AttentionNN.forward` no longer raises; delegates to `_attention_mlx`.
- **regression**: 13.3b-1 (AC0-AC3) + 13.3b-2a (AC1-AC2) + 13.3b-2b (AC1-AC2) + 13.3a nn (12 incl backward AC) + 13.2 FP4 (9) + tiny CSA (11.14/11.15) all GREEN. Full suite: 1 RED (Test #8) / 565+new PASS / 13 SKIP / 0 introduced RED.
- **byte-intactness**: FROZEN bodies byte-intact via source hash + AST. tiny path byte-identical. `git diff --check` clean. sha-pin cascade SLICE-INVARIANT.

## Deliverables
1. `agent-output/cmux-13-3b/requirements-13-3b-3.md` — §0 GO verdict + §A Q1-Q5 LOCKED + §B AC1-AC4 + blast-radius + STOP-rules + discriminator predicate pinned. Do NOT design (Architect owns §3.4).
2. `.cmux-status/ba.done` (`{"status":"ok","role":"BA"}`).
3. In-pane JSON: `{"status":"ok","role":"BA"}` in surface:81 ONLY.
4. `docs/backlog.md` 13.3b-3 row update.

**STOP-ESCALATE** if no clean discriminator (S-discriminator) or FROZEN-body conflict (S-frozen-body). Write `ba-stop.md` + `{"status":"error",...}`.

## Pre-flight read list (READ in this order)
1. `agent-output/cmux-13-3b/architecture.md` §3.4 (unified dispatch) + §4.2 (ONE dispatch branch) + §8 (frozen list). NOTE §4.2 "ONE dispatch branch" was written BEFORE 13.3b-1 added the existing dispatch at `:1199-1202` — BA reconcile.
2. `agent-output/cmux-13-3b/subslice-breakdown.md` 13.3b-3 row.
3. `agent-output/cmux-13-3b/requirements-13-3b-1.md` (the 13.3b-1 dispatch contract).
4. `agent-output/cmux-13-3b/requirements-13-3b-2a.md` + `requirements-13-3b-2b.md` (predecessor scope).
5. `docs/adr/0026-csa-hca-indexer-mlx-real-port.md` (ADDITIVE ONLY, tiny byte-identical).
6. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` — FROZEN. Read: `_attention_mlx:1190-1257` (dispatch + cr=0 body + grouped-o loop), `_csa_config_error:317` (6 guards — to understand the discriminator), `_attention_real_mlx:1005` (HCA, 13.3b-1), `_csa_attention_real_mlx:907` (CSA, 13.3b-2b), `_csa_attention_mlx:515` (tiny), `_indexer_mlx:620`.
7. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py` — `AttentionNN:135-203` (the guard at `:153` + forward delegation).
8. The tiny fixture config (find in `tests/` for 11.14/11.15).
9. `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json` (real config).
10. torch `modeling_deepseek_v4.py` Attention.forward (only if integration AC needs torch ref).

## Style
Caveman ultra default; byte-exact exempt. Cite §/Q/AC numbers. Use `ctx_execute`/`ctx_execute_file` for FROZEN/ckpt probes. BEGIN NOW.
