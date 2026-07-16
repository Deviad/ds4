# Story 13.3b-1 — BA Task Brief (THIN: HCA real path + RoPE oracle — Epic go/no-go)

## Role
BA (anthropic/claude-opus-4-8 · high, surface:81, fresh-context via /new). THIN — Architect r0 already designed §3.1 (HCA compressor) + §3.4 (unified `_attention_real_mlx`) + §3.5 (RoPE oracle) + §4.2 (additive dispatch). Translate to LOCKED Q-list + acceptance tests + blast-radius + STOP-rules. Do NOT re-design.

## Slice context
- **Story**: 13.3b-1 (FIRST sub-slice of 13.3b per Architect subslice-breakdown.md). **Epic go/no-go**: the RoPE oracle (AC#0) determines whether the compress-rope contract is parity-achievable with FROZEN primitives additively. If it STOPs, the full-port path (Option B) needs re-scope.
- **Predecessor**: 13.3a-3 complete (HEAD `702199f`). 13.3b Architect r0 complete (`agent-output/cmux-13-3b/architecture.md` + ADR 0026 + subslice-breakdown.md).
- **Authoritative contracts** (READ IN FULL):
  - `agent-output/cmux-13-3b/architecture.md` §0-§5 (esp. §2 core finding, §2.1 reuse map, §3.1 HCA compressor design, §3.4 unified real attention, §3.5 RoPE oracle risk, §4.2 additive dispatch).
  - `agent-output/cmux-13-3b/subslice-breakdown.md` (13.3b-1 row: scope/AC/STOP/dev-days).
  - `docs/adr/0026-csa-hca-indexer-mlx-real-port.md` (FROZEN additive expansion contract).
  - `agent-output/cmux-13-3a/architecture.md` + ADR 0025 (the nn port being extended).

## 13.3b-1 scope (Architect r0 — translate, don't re-derive)
Per subslice-breakdown.md 13.3b-1 row (5 dev-days):
1. **RoPE oracle (AC#0 — FIRST, load-bearing go/no-go)**: build a torch-vs-MLX rope oracle for ONE compress layer. Pin: (a) does torch compress rope cover full head_dim or only trailing `qk_rope_head_dim`? (b) does compress rope apply yarn scaling (factor=16)? Lock the rope contract before writing the real path.
2. **`_hca_compressor_mlx` (§3.1)**: port torch `DeepseekV4HCACompressor.forward:391` stateless branch (rate=128, out_dim=head_dim=512, NO Ca/Cb overlap, single window pooling + position_bias ape + norm + compress rope). Returns `(compressed_kv [B,1,n_win,head_dim], block_bias [B,1,S,n_win])`.
3. **`_attention_real_mlx` HCA branch (§3.4)**: the HCA case of the unified real multi-head compressed path. Steps 1-6 from §3.4: proj+norm+RoPE → HCA branch (compressed_kv + block_bias via `_hca_compressor_mlx`) → KV append → mask append → multi-head scores+sink+softmax+attend → rope-undo → grouped-o. Reuse cr=0 `_attention_mlx:632` body math.

### blast-radius (Architect §4.2 + subslice-breakdown — confirm exact set)
| # | File | Change |
|---|---|---|
| 1 | `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` | ADDITIVE ONLY: NEW helpers `_hca_compressor_mlx` + `_attention_real_mlx` (HCA branch) in a NEW module section; ONE additive dispatch branch in `_attention_mlx:640` (`if _csa_config_error(args) is None: return _csa_attention_mlx(...); else: return _attention_real_mlx(...)`). Tiny path byte-identical (existing tiny inputs still satisfy `_csa_config_error is None`). NO FROZEN body edits. |
| 2 | NEW test files: RoPE oracle test + HCA compressor parity test + HCA attention parity test. | NEW |

**NO nn port edit yet** (13.3b-3 owns AttentionNN guard lift). **NO convert** (13.3b-4). **NO CSA/Indexer** (13.3b-2). Helper-level + functional tests only.

## §A — LOCKED Q-list (resolutions Coder MUST follow)

### Q1 — RoPE oracle: what exactly must the torch-vs-MLX oracle pin? (AC#0 — load-bearing)
LOCKED: the oracle must pin, for ONE real compress layer (a CSA or HCA layer from the real ckpt, or a synthesized real-dim fixture with `rope_layer_type="compress"`):
1. **Channel scope**: does torch compress rope rotate the FULL `head_dim` (FROZEN `_apply_rope_full_mlx`) or only the TRAILING `qk_rope_head_dim=64` slice (FROZEN `_rope_tail_tables_mlx`)? Probe torch `apply_rotary_pos_emb:344` + `DeepseekV4RotaryEmbedding.forward:153` for the compress branch.
2. **Yarn scaling**: does compress rope apply yarn (factor=16, original_max=65536, theta=compress_rope_theta=160000)? If yes, the cos/sin tables differ from FROZEN tiny (theta=10000, no yarn).
3. **Positions**: which positions get rope in the compress path — block-start positions `i*rate` (rate=128 for HCA)? All compressed-block starts?
4. **Parity tolerance**: torch BF16 vs MLX BF16 — what tol? (1e-3? 1e-4? — BA LOCK based on existing 13.2 FP4 parity 1e-5 being too tight for cross-framework BF16; propose 1e-3 for rotary cos/sin, 1e-2 for downstream attention output.)

The oracle is a PYTHON test that constructs the torch reference + MLX implementation for ONE compress layer and asserts cos/sin (and/or rotated q/kv) match within tol. **STOP-ESCALATE (AC#0)** if torch compress rope needs a math primitive FROZEN cannot supply additively (e.g. a custom kernel for yarn-scaled rotary at compress positions, or a concat pattern FROZEN `_apply_rope_full_mlx` doesn't support).

### Q2 — HCA compressor: what's the exact stateless contract? (§3.1)
LOCKED per Architect §3.1: `_hca_compressor_mlx(args, x, weights, *, position_ids) -> tuple[compressed_kv, block_bias]`:
- `rate=128`, `out_dim=head_dim=512`.
- `kv = x @ wkv.T` `[B,S,head_dim]`; `gate = x @ wgate.T` `[B,S,head_dim]`.
- `usable = (S//rate)*rate`; reshape `[B, n_win, rate, head_dim]`; `+ position_bias` (ape `(rate, head_dim)`, NO transpose — verify vs torch HCACompressor:415 which uses `ape[:out_dim,:].T` for CSA but HCA may differ; BA to lock from torch:394-444).
- `compressed = norm( sum(kv * softmax(gate, axis=2, fp32), axis=2) )` → `[B, n_win, head_dim]`.
- RoPE at positions `i*rate` with compress rope (per Q1 oracle).
- `compressed_kv = compressed[:, None]` `[B,1,n_win,head_dim]`.
- `block_bias`: causal — entry `i` visible to query `t` iff `i < (position_ids+1)//rate`; else `-inf`. Shape `[B,1,S,n_win]`. Port torch:454-461.

**Q2 sub-question (LOCKED)**: does HCA ape need the `ape[:out_dim,:].T` transpose (CSA-style) or NOT (Architect §3.1 says "NO transpose")? BA must resolve by reading torch `HCACompressor.forward:394-444` precisely — this is a shape-mismatch landmine.

### Q3 — `_attention_real_mlx` HCA branch: what exactly is reused from cr=0 vs new? (§3.4)
LOCKED per Architect §3.4: steps 1 (proj+norm+RoPE), 5 (multi-head scores+sink+softmax+attend), 6 (rope-undo + grouped-o) REUSE cr=0 `_attention_mlx:632` body math. NEW: step 2 HCA branch (`compressed_kv, block_bias = _hca_compressor_mlx`), step 3 (`kv_full = cat([kv, compressed_kv], axis=2)`), step 4 (`mask = cat([sliding_causal_mask, block_bias], axis=-1)`).

**Q3 sub-question (LOCKED)**: does the HCA branch use the SAME q/kv projections as cr=0 (FROZEN `q_a_proj`/`q_b_proj`/`kv_proj` + norms + sinks), or do compress layers have DIFFERENT projection weights/t-sink handling (e.g. `attn_sink` dims differ for compress vs sliding)? BA must pin from torch `DeepseekV4Attention.forward:801-875` + the real ckpt keys (layer 2+ has `attn.wq_a/wq_b/wkv/wo_a/wo_b` — same names as sliding, so projections likely shared; confirm).

### Q4 — RoPE-undo + grouped-o: are these in the HCA path or only sliding? (§3.4 step 6)
LOCKED: Architect §3.4 step 6 says "rope-undo + grouped-o, REUSE cr=0 tail (632 body 700-755)". So HCA path MUST include rope-undo (`-sin` on output rope slice) + grouped-o (o_a block-diagonal + o_b), identical to cr=0. BA confirm torch `DeepseekV4Attention.forward` does the same for compress layers (the grouped-o is shared, not sliding-only).

### Q5 — additive dispatch: how exactly does the tiny path stay byte-identical? (§4.2, ADR 0026)
LOCKED per Architect §4.2: the ONE FROZEN edit is `_attention_mlx:640`:
```python
if args.compression_ratio != 0:
    if _csa_config_error(args) is None:          # tiny proven subset → UNCHANGED
        return _csa_attention_mlx(args, x, weights, index_topk=index_topk)
    return _attention_real_mlx(args, x, weights, index_topk=index_topk)   # NEW
```
BA confirm: every existing tiny test input satisfies `_csa_config_error(args) is None` (so it still routes to `_csa_attention_mlx`, byte-identical). NO existing test changes branch.

### Q6 — test fixtures: real-dim compress-layer fixture — where do the weights come from? (anti-circularity)
LOCKED: the HCA parity test needs real-dim weights (wkv, wgate, ape, norm, + attn projections + sinks). Options: (a) synthesize random weights at real dims (head_dim=512, rate=128, hidden=4096, q_lora_rank=1024, o_groups=8, num_heads=64); (b) load ONE real HCA layer from the ckpt (layer 3, say). BA LOCK: synthesize is cheaper + parity-exact (no ckpt-shard read in unit test); but cross-check ONE assertion against a real-ckpt layer load to confirm dims/names match. Anti-circularity: the torch reference computes expected values from the SAME synthesized weights (no second MLX primitive as oracle).

### Q7 — STOP conditions (Architect subslice-breakdown + ADR 0026)
LOCKED — emit `coder-stop.md` + `{"status":"error",...}` if:
- **(S-rope, AC#0)** torch compress rope needs a math primitive FROZEN cannot supply additively (kernel, concat pattern, yarn-scaled rotary helper not derivable from FROZEN) → STOP-ESCALATE (Epic-level re-scope).
- **(S-frozen-body)** any FROZEN symbol body (`_attention_mlx:632` cr=0 math, `_csa_*`, `_hyperconnection_mlx`, parity `Model`, `sanitize_weights`, `_apply_rope_full_mlx`, `_rope_tail_tables_mlx`) would need EDIT (not additive new symbol) → STOP.
- **(S-tiny-regression)** any tiny CSA fixture (11.14/11.15) goes RED → STOP (the tiny path must stay byte-identical per ADR 0026).
- **(S-parity)** HCA compressor or HCA attention cannot match torch reference within tol at real dims after best-effort → STOP with the failing assertion.
- **(S-backward)** 13.3a-3 backward AC goes RED (the additive dispatch must not break the 13.3a nn port) → STOP.

## §B — Acceptance criteria → trackable tests (TDD red-first)
- **AC0 (LOAD-BEARING go/no-go)** — `test_compress_rope_oracle.py` (NEW): for ONE real-dim compress layer, assert MLX compress-rope cos/sin (and/or rotated q/kv) match torch reference within tol (Q1 Q4 tolerance). Pin channel scope (full vs trailing) + yarn + positions in test docstring. If this test cannot be made GREEN additively → STOP-ESCALATE (do NOT proceed to AC1/AC2).
- **AC1** — `test_hca_compressor_mlx_parity.py` (NEW): `_hca_compressor_mlx` output `(compressed_kv, block_bias)` matches torch `DeepseekV4HCACompressor.forward` (stateless branch) within tol at real dims (head_dim=512, rate=128, hidden=4096). Q2 ape-transpose resolution pinned.
- **AC2** — `test_attention_real_mlx_hca_parity.py` (NEW): `_attention_real_mlx` HCA branch output matches torch `DeepseekV4Attention.forward` for an HCA layer within tol. Q3 projection reuse + Q4 rope-undo/grouped-o confirmed.
- **AC3** — `test_13_3b_1_dispatch_additive.py` (NEW): tiny compress input still routes `_csa_attention_mlx` (assert `_csa_config_error is None` + return value byte-identical to pre-edit); real-dim compress input now routes `_attention_real_mlx` (no longer `NotImplementedError`). Q5.
- **regression**: 13.2 FP4 (9) + 13.3a nn (12 incl 13.3a-3 backward AC) + tiny CSA fixtures (11.14/11.15) all GREEN. Full suite 1 RED (Test #8) / 557+4new PASS / 13 SKIP / 0 introduced RED.

## Deliverables
1. `agent-output/cmux-13-3b/requirements-13-3b-1.md` — §A Q1-Q7 LOCKED + §B AC0-AC3 + blast-radius + STOP-rules. Do NOT design (Architect r0 owns §3.1/§3.4/§3.5/§4.2).
2. `.cmux-status/ba.done` (`{"status":"ok","role":"BA"}`).
3. In-pane JSON: `{"status":"ok","role":"BA"}` in surface:81 ONLY.
4. `docs/backlog.md` 13.3b-1 row update (scope LOCKED + AC0 go/no-go flag).

**STOP-ESCALATE** if AC0 (RoPE oracle) reveals an unsolvable FROZEN-primitive gap (the Epic-level go/no-go). Write `ba-stop.md` + `{"status":"error","error":"AC0 rope oracle: <reason>","role":"BA"}`.

## Pre-flight read list
1. `agent-output/cmux-13-3b/architecture.md` §0-§5 (esp. §3.1, §3.4, §3.5, §4.2).
2. `agent-output/cmux-13-3b/subslice-breakdown.md` 13.3b-1 row.
3. `docs/adr/0026-csa-hca-indexer-mlx-real-port.md`.
4. `python-envs/mlx/.venv/lib/python3.13/site-packages/transformers/models/deepseek_v4/modeling_deepseek_v4.py` lines 75-170 (RotaryEmbedding), 362-445 (HCACompressor), 755-875 (Attention.forward) — for Q1/Q2/Q3/Q4.
5. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` — `_attention_mlx:632`, `_apply_rope_full_mlx`, `_rope_tail_tables_mlx`, `_csa_config_error:317`, `_csa_attention_mlx:515`, the cr=0 body 700-755.
6. `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json` (compress_rope_theta=160000, rope yarn factor=16).

## Style
Caveman ultra default; byte-exact exempt. Cite §-numbers + Q-numbers + torch:line + FROZEN:line. Use `ctx_execute`/`ctx_batch_execute` for torch/FROZEN probes. BEGIN NOW.
