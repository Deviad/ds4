# Requirements — Story 13.3b-3 (INTEGRATION: dispatch discriminator + GroupedLinear confirm + AttentionNN guard lift)

BA. THIN. Architect r0 owns design (`architecture.md` §3.4 / §4.1 / §4.2 / §4.3 / §8 + ADR 0026).
This doc = LOCKED Q-list + AC + blast-radius + STOP-rules. NO design. Caveman-ultra; code/paths/line-numbers byte-exact.

FROZEN file: `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`.
nn file (SANCTIONED, ADR 0025): `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py`.
real cfg: `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json`. Predecessor HEAD `aa93c5d`.

---

## §0 GO/NO-GO VERDICT — **GO** (no `ba-stop.md`)

S-discriminator does NOT fire: a CLEAN discriminator EXISTS today; no config flag needed.
S-frozen-body does NOT fire: dispatch edit site is the 13.3b-1-added dispatch branch (ADR 0026
SANCTIONED additive site, NOT a FROZEN body); cr=0 body + all helper bodies + tiny path stay byte-identical.

### §0.1 BRIEF CORRECTION (load-bearing — supervisor recon re-verified, empirically PINNED)

Brief premise: "real cr=4 passes `_csa_config_error is None` → routes to TINY `_csa_attention_mlx`."
**FALSE. INVERTED.** `_csa_config_error:320` ALREADY gates
`if args.num_attention_heads != 1 or args.o_groups != 1: return "...proven only for single-head o_groups=1 tiny fixtures"`.
Real CSA (cr=4, `num_attention_heads=64`, `o_groups=8`) ⇒ `_csa_config_error` returns a NON-None string.

Empirically verified (`.venv` + `real_hca_mlx_args` + `dataclasses.replace(compression_ratio=4)`):
```
realCSA(cr=4, heads=64, o_groups=8): _csa_config_error isNone=False
  snippet='MLX compressors/indexers CSA compressed attention is proven only for s...'
```
⇒ TODAY real cr=4 routes: `_attention_mlx:1199` (cr≠0) → `:1200` `is None`? **NO** → `:1202`
`_attention_real_mlx` → `:1014` `compression_ratio != 128` → `:1018` `raise NotImplementedError("...HCA compression_ratio=128")`.

**The REAL integration gap (what 13.3b-3 fixes):** real cr=4 layers RAISE at dispatch (routed to HCA-only
`_attention_real_mlx`), they do NOT silently hit the tiny path. The fix is NOT a "tiny-vs-real" discriminator
on cr=4 (that already exists via `_csa_config_error`). The fix is splitting the NON-None (real) branch by
`compression_ratio`: cr=4 → NEW `_csa_attention_real_mlx` (13.3b-2b); cr=128 → `_attention_real_mlx` (13.3b-1, existing).

---

## §A — LOCKED Q-list

### Q1 — Dispatch discriminator (cr=4-real vs cr=4-tiny vs cr=128) — RESOLVED, PINNED — CRITICAL

**Discriminator already exists in two layers; NO config flag, NO `_csa_config_error` edit.**

| input | `_csa_config_error(args) is None` | route (after 13.3b-3) | today |
|---|---|---|---|
| cr=4 + tiny (heads=1, o_groups=1, hidden==head_dim, hc_mult=1, q_lora==hidden) | **True** | `_csa_attention_mlx` (tiny, UNCHANGED byte-identical) | same |
| cr=4 + real (heads=64, o_groups=8, head_dim=512≠hidden=4096) | **False** | **NEW** `_csa_attention_real_mlx` | RAISES (HCA-only) |
| cr=128 (HCA real) | **False** | `_attention_real_mlx` (existing :1202) | same |
| cr=0 | n/a (branch not entered) | main body `:1203+` (UNCHANGED) | same |

Tiny config pins (from `tests/test_13_3b_1_dispatch_additive.py` `_tiny_csa_args`): `hidden_size=4`,
`num_attention_heads=1`, `o_groups=1`, `head_dim=4`, `compression_ratio=4`. Real config pins (`config.json`):
`hidden_size=4096`, `num_attention_heads=64`, `o_groups=8`, `head_dim=512`, `num_key_value_heads=1`.
The tiny/real split is fully carried by the EXISTING `_csa_config_error` guards (`:320` heads/o_groups gate
is decisive). 13.3b-3 adds ONLY a cr-value split inside the already-real (`is None`==False) leg.

**EXACT additive dispatch shape (ONE inserted branch, tiny + cr=128 returns byte-identical):**
```python
    if args.compression_ratio != 0:
        if _csa_config_error(args) is None:                                       # :1200 tiny — UNCHANGED
            return _csa_attention_mlx(args, x, weights, index_topk=index_topk)     # :1201 byte-identical
        if args.compression_ratio == 4:                                           # NEW real-CSA split
            return _csa_attention_real_mlx(args, x, weights, index_topk=index_topk)# NEW
        return _attention_real_mlx(args, x, weights, index_topk=index_topk)        # :1202 cr=128 HCA byte-identical
```
ADDITIVE: inserts ONE `if args.compression_ratio == 4:` + its return BEFORE current `:1202`. Lines
`:1200`-`:1201` (tiny) byte-identical; current `:1202` (HCA) byte-identical. Real cr=4 now routes to
`_csa_attention_real_mlx` instead of raising. **Editing the 13.3b-1 dispatch branch IS allowed** — ADR 0026
§Decision-1 names the `_attention_mlx` dispatch branch as THE sanctioned additive edit site (NOT FROZEN body).

### Q2 — GroupedLinear (`_grouped_linear_mlx`) — RESOLVED: **NOT NEEDED (reused inline)**

Recon (grep `for group in range(args.o_groups)` + `o_a_proj.weight`/`o_b_proj.weight`):
- grouped-o loop ALREADY inline in `_csa_attention_real_mlx:992-1002` (13.3b-2b) ✓
- grouped-o loop ALREADY inline in `_attention_real_mlx` HCA tail (13.3b-1) ✓
- grouped-o loop is the FROZEN cr=0 body `_attention_mlx:1247-1256` ✓
- `def _grouped_linear_mlx` — **ABSENT** (does not exist).

Architect §4.3: "If identical → REUSE, no new helper." 13.3b-1/2b already inlined the IDENTICAL block-diagonal
o_groups=8 math (proven by their AC2 parity-vs-torch GREEN). ⇒ **NO new `_grouped_linear_mlx` helper.**
GroupedLinear parity is ALREADY confirmed GREEN through 13.3b-1 AC2 + 13.3b-2b AC2 (both pass through the inline
grouped-o vs torch `DeepseekV4GroupedLinear.forward:328`). Extracting into a shared helper would EDIT the FROZEN
cr=0 inline loop (`:1247-1256`) → REJECTED (S-frozen-body). **13.3b-3 = dispatch + guard lift ONLY.**

No Architect-clarification STOP: §4.2 ("ONE dispatch branch additive") + §4.3 ("if identical REUSE") are
CONSISTENT with dispatch-only. Recon does not contradict §4.

### Q3 — AttentionNN guard lift (`deepseek_v4_nn.py:153`) — PINNED (Architect §4.1 owns design)

Current `:153-154`:
```python
        if config.compression_ratio != 0:
            raise NotImplementedError("13.3a-1 AttentionNN supports only compression_ratio=0")
```
`AttentionNN.__call__:203` already delegates `return _attention_mlx(self.config, x, weights)`. BUT `__call__`
builds a cr=0-ONLY weights dict (`:193-202`: `q_a_proj/q_norm/q_b_proj/kv_proj/kv_norm/o_a_proj/o_b_proj/sinks`).

**PIN (honest contract):** removing the guard ALONE is INSUFFICIENT — the real cr≠0 dispatch (`_csa_attention_real_mlx`
needs `compressor_wkv/compressor_wgate/compressor_ape/compressor_norm` + `indexer_compressor_*`/`indexer_proj`/
`indexer_wq_b`; `_attention_real_mlx` needs `compressor_*`) would `KeyError` on the missing keys. So the guard lift =
full §4.1 nn wiring:
1. REMOVE the `:153-154` `if config.compression_ratio != 0: raise NotImplementedError` block.
2. ADD compressor submodules (when layer is CSA/HCA per `layer_types[layer_idx] != "sliding_attention"`): CSA →
   `compressor.{wkv,wgate}` `nn.Linear` + `compressor.{ape,norm.weight}` params + `indexer.compressor.{wkv,wgate}`
   `nn.Linear` + `indexer.compressor.{ape,norm.weight}` + `indexer.weights_proj` `nn.Linear` + `indexer.wq_b`
   `nn.Linear`; HCA → `compressor.{wkv,wgate}` + `compressor.{ape,norm.weight}` ONLY.
3. `__call__` real branch: build the extended weights dict (keys exactly as the real helpers read — `compressor_wkv`,
   `compressor_wgate`, `compressor_ape`, `compressor_norm`, and for CSA `indexer_compressor_wkv/wgate/ape/norm`,
   `indexer_proj`, `indexer_wq_b`) and delegate to `_attention_mlx` (which now dispatches cr=4-real→CSA, cr=128→HCA);
   pass `index_topk` for CSA. sliding (cr=0) branch UNCHANGED (existing `:193-203`).

Architect §4.1 owns the exact submodule/leaf layout; BA pins ONLY: guard removed, cr≠0 no longer raises, `__call__`
builds the real weights dict + delegates to `_attention_mlx`. SANCTIONED nn-file edit (ADR 0025 owns nn file).

### Q4 — Integration AC: dispatch routing + parity-through-dispatch + tiny byte-identity — PINNED

Pin (cleaner = route-equivalence, NOT a 4th torch run):
- **Dispatch routing** (extend `test_13_3b_1_dispatch_additive.py`): add legs — real cr=4 routes to
  `_csa_attention_real_mlx` (monkeypatch spy / sentinel); real cr=128 routes to `_attention_real_mlx`; cr=4-tiny
  routes to `_csa_attention_mlx` byte-identical (existing leg — KEEP).
- **Parity-through-dispatch** (extend `test_csa_attention_real_mlx_parity.py` + `test_attention_real_mlx_hca_parity.py`):
  assert `_attention_mlx(args, x, weights, index_topk=...)` output == the direct `_csa_attention_real_mlx`/
  `_attention_real_mlx(...)` output (dispatch is a pure router ⇒ identical). The direct-helper-vs-torch parity is
  ALREADY GREEN (13.3b-1 AC2 / 13.3b-2b AC2) ⇒ transitively real cr=4 + cr=128 match torch THROUGH the dispatch.
- **nn end-to-end** (NEW `test_13_3b_3_nn_mixed_layers_forward.py`): `AttentionNN.__call__` over real-dim CSA + HCA +
  sliding configs no longer raises; produces finite `mx.array` matching the corresponding functional helper output.

### Q5 — STOP conditions — RESOLVED (none fire)

- **S-discriminator**: NO STOP. Clean discriminator exists (§0.1/Q1): `_csa_config_error` (tiny/real) + `compression_ratio==4`
  (CSA/HCA). No config flag, no `_csa_config_error` edit.
- **S-frozen-body**: NO STOP. Dispatch edit is the ADR-0026 sanctioned additive site (13.3b-1 dispatch branch), NOT a
  FROZEN body. cr=0 body `:1203+` byte-identical (sha16 `5b0849ad8172420e` pre-edit, MUST hold post-edit). Tiny return
  `:1200-1201` byte-identical. NO new helper extracts the FROZEN grouped-o loop (Q2). cr=128 return byte-identical.
- **clarification on "edit 13.3b-1 dispatch":** ALLOWED. 13.3b-3 inserts a NEW sibling branch (`if args.compression_ratio
  == 4:`) before current `:1202`; it does NOT alter the tiny return nor the cr=128 return text.
- Emit `ba-stop.md` + `{"status":"error",...}` ONLY if (during build) the discriminator turns out to need a config flag
  absent in args (S-discriminator) OR the dispatch fix forces a FROZEN-body edit (S-frozen-body) OR a tiny CSA fixture
  goes RED (S-tiny-regression) OR real cr=4/cr=128 cannot match torch through dispatch (S-integration-parity).

---

## §B — Acceptance criteria (TDD red-first)

- **AC1** — dispatch discriminator: cr=4-real → `_csa_attention_real_mlx`; cr=128 → `_attention_real_mlx`; cr=4-tiny →
  `_csa_attention_mlx`; cr=0 → main body. (`test_13_3b_1_dispatch_additive.py` extended; sentinel/spy.)
- **AC2** — tiny byte-identical regression: 11.14/11.15 tiny CSA fixtures + the existing dispatch test's tiny leg produce
  byte-identical output before/after the dispatch change (`routed.tolist() == direct.tolist()`).
- **AC3** — integration parity-through-dispatch: real cr=4 (CSA) + cr=128 (HCA) match torch within tol THROUGH
  `_attention_mlx` dispatch (route-equivalence to direct helper, whose torch parity is already GREEN).
- **AC4** — AttentionNN guard lift: cr≠0 `AttentionNN.__call__` no longer raises; builds real weights dict + delegates to
  `_attention_mlx`; output finite + matches functional helper over CSA/HCA/sliding configs.
- **regression** — 13.3b-1 (AC0-AC3) + 13.3b-2a (AC1-AC2) + 13.3b-2b (AC1-AC2) + 13.3a nn (12, incl backward AC) + 13.2
  FP4 (9) + tiny CSA (11.14/11.15) all GREEN. Full suite: 1 pre-existing RED (Test #8) / 565+new PASS / 13 SKIP /
  **0 introduced RED**.
- **byte-intactness** — FROZEN bodies byte-intact (cr=0 body `:1203+` sha16 `5b0849ad8172420e`; `_attention_real_mlx`,
  `_csa_attention_real_mlx`, `_csa_compressor_real_mlx`, `_csa_block_bias_mlx`, `_csa_attention_mlx`, `_indexer_mlx`,
  `_indexer_scorer_mlx`, `_csa_config_error`, `_csa_windowed_compressor_mlx`, `_hyperconnection_mlx`, `_hyperhead_mlx`,
  `sanitize_weights`, parity `Model:1685` all FROZEN). Tiny dispatch return byte-identical. `git diff --check` clean.
  AST/grep proof (untracked-file caveat). **sha-pin cascade: SLICE-VARIANT** — `deepseek_v4.py` content changes
  (dispatch +2 lines) ⇒ advance EXACTLY **2** sha16 pin sites currently `dc5aaaab9bb079d2`:
  `tests/test_numpy_real_forward_reference_composition.py:397` + `tests/test_deepseek_v4_real_config_reference_forward.py:383`.
  Reviewer verifies count==2; do NOT under/over-advance.

AC convention (every slice): (a) parity vs torch; (b) tiny CSA GREEN; (c) backward AC (13.3a-3) GREEN; (d) `git diff
--check` clean + FROZEN bodies byte-intact (AST/grep, not zero-line-diff).

---

## §C — Blast-radius

| # | File | Change | Sanction |
|---|---|---|---|
| 1 | `deepseek_v4.py` | ADDITIVE: insert `if args.compression_ratio == 4: return _csa_attention_real_mlx(...)` BEFORE `_attention_mlx:1202`. Tiny return `:1200-1201` byte-identical; cr=128 return `:1202` byte-identical; cr=0 body `:1203+` byte-identical. **NO new helper** (Q2). NO FROZEN body edits. | ADR 0026 additive |
| 2 | `deepseek_v4_nn.py` | `AttentionNN` guard lift `:153-154` + compressor/indexer submodules + real `__call__` branch (§4.1). | ADR 0025 nn-file |
| 3 | `tests/test_13_3b_1_dispatch_additive.py` | EXTEND: cr=4-real + cr=128 routing legs (AC1). | — |
| 4 | `tests/test_csa_attention_real_mlx_parity.py` + `tests/test_attention_real_mlx_hca_parity.py` | EXTEND: route-equivalence through `_attention_mlx` (AC3). | — |
| 5 | `tests/test_13_3b_3_nn_mixed_layers_forward.py` | NEW (AC4). | — |
| 6 | `tests/test_numpy_real_forward_reference_composition.py:397` + `tests/test_deepseek_v4_real_config_reference_forward.py:383` | sha-pin advance `dc5aaaab9bb079d2` → new sha (2 sites). | byte-intact cascade |

**OUT of scope (fences):** NO convert/remap (13.3b-4). NO smoke-train (13.3b-5). NO new real-branch helpers
(13.3b-1/2a/2b shipped them; 13.3b-3 only WIRES). NO `_grouped_linear_mlx` (Q2 reused inline). NO FROZEN body edits.

FROZEN — MUST NOT EDIT BODY (architecture.md §8 + ADR 0026): `_csa_config_error`, `_require_csa_config`,
`_csa_attention_mlx`, `_csa_compressor_mlx`, `_csa_indexer_mlx`, `_csa_windowed_compressor_mlx`,
`_csa_compressor_real_mlx`, `_csa_block_bias_mlx`, `_csa_attention_real_mlx`, `_indexer_mlx`, `_indexer_scorer_mlx`,
`_hca_compressor_mlx`, `_attention_real_mlx`, `_hyperconnection_mlx`, `_hyperhead_mlx`, `_attention_mlx` cr=0 body
(`:1203+`), `sanitize_weights`, parity `Model:1685`, FP4/i8 dequant, 9 ADR-0017 forbidden symbols, rope tail
primitives, `_compress_rope_yarn_tail_tables_mlx`.

---

## §D — Dev-days / sequence
4 dev-days (dispatch branch 0.5 + guard lift/nn wiring §4.1 2.5 + integration tests 1). Serial; file-mutating coder
one slice, Reviewer + Test Manager parallel after Coder GREEN. INTEGRATION milestone: after GREEN, 41/43 real layers
run at real dims (3 cr=0 already worked pre-13.3b). 13.3b-4 (convert) NEXT.

## Deliverables (this BA pass)
1. THIS doc.
2. `.cmux-status/ba.done` (`{"status":"ok","role":"BA"}`).
3. `docs/backlog.md` 13.3b-3 row.
4. In-pane JSON `{"status":"ok","role":"BA"}` surface:81 only.

GO verdict (§0). No `ba-stop.md`. Discriminator clean (§0.1/Q1); GroupedLinear reused inline (Q2); guard lift = full
§4.1 nn wiring (Q3); integration via route-equivalence (Q4). Story 13.3b-3 proceeds.
