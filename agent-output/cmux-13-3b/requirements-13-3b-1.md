# Requirements — Story 13.3b-1 (HCA real path + RoPE oracle; Epic go/no-go)

BA. THIN. Architect r0 owns design (`architecture.md` §3.1/§3.4/§3.5/§4.2 + ADR 0026).
This doc = LOCKED Q-list + acceptance tests + blast-radius + STOP-rules. NO design.
Caveman-ultra. Code/paths/line-numbers/numbers byte-exact.

FROZEN file: `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`.
torch ref: `python-envs/mlx/.venv/lib/python3.13/site-packages/transformers/models/deepseek_v4/modeling_deepseek_v4.py`.
torch cfg: `.../deepseek_v4/configuration_deepseek_v4.py`.
real cfg: `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json`.

---

## §0 EPIC GO/NO-GO VERDICT (AC0 oracle, resolved by BA recon) — **GO**

AC0 is load-bearing: does compress-rope contract reach parity with FROZEN primitives
ADDITIVELY. BA ran torch RotaryEmbedding on real cfg. Findings:

| dim | value | source |
|---|---|---|
| compress rope_type | **yarn** | cfg.rope_parameters['compress'].rope_type (DeepseekV4Config.__post_init__ configuration_deepseek_v4.py:294-321) |
| compress theta | **160000** | compress_rope_theta (config.json) |
| yarn factor | **16** | rope_scaling.factor (config.json) |
| beta_fast / beta_slow | **32 / 1** | rope_scaling |
| original_max_position_embeddings | **65536** | rope_scaling |
| attention_factor | **1.0** (NO mscale on cos/sin) | configuration_deepseek_v4.py:319 `compress.setdefault("attention_factor", 1.0)`; verified `compress_attention_scaling == 1.0` |
| **channel scope** | **TRAILING qk_rope_head_dim=64**, NOT full head_dim=512 | partial_rotary_factor=0.125; cfg.qk_rope_head_dim=64; `apply_rotary_pos_emb:344` rotates trailing `rope_dim=cos.shape[-1]*1` = 64; verified compress cos shape `(1,4,32)` → rope_dim doubled = 64 |
| interleaved convention | pairwise `rotate_half = stack((-x2,x1)).flatten` | modeling:336-343 |

**Verdict GO.** Reasoning:
1. **Channel scope TRAILING-64** ⇒ FROZEN tail-apply primitives are the right shape:
   `_apply_rope_tail_mlx` (deepseek_v4.py:269) + `_broadcast_rope_tail_table_mlx`
   (deepseek_v4.py:238) + `_apply_rope_mlx` (deepseek_v4.py:260, interleaved pairwise,
   matches torch rotate_half). REUSE for APPLY — parity-exact, additive.
2. **Yarn TABLE gap is pure MLX math, NOT a kernel.** FROZEN `_rope_tail_tables_mlx`
   (deepseek_v4.py:227) builds PLAIN inv_freq `1/(theta^(dim_idx/qk_rope_head_dim))`,
   NO yarn. Yarn changes inv_freq (BA measured max rel diff **0.9375** vs plain theta=160000).
   Yarn = ramp interpolation between low/high-freq inv_freq + attention_factor=1.0
   (no cos/sin mscale). Expressible in plain `mx.arange`/`mx.log`/`mx.clip`/`mx.where`
   arithmetic ⇒ a NEW additive helper (`_rope_compress_tail_tables_mlx` or Coder's
   chosen name) supplies the yarn cos/sin tables; APPLY reuses FROZEN tail apply.
   **No FROZEN-primitive gap. No custom kernel. No concat pattern FROZEN cannot do.**
3. ⇒ S-rope STOP condition does NOT fire. Epic 13.3b proceeds.

> BA did NOT design the yarn helper (Architect §3.5 owns the design call; this doc
> only LOCKS that an additive pure-math helper is sufficient → no STOP). AC0 test
> still REQUIRED (red-first) to PROVE the cos/sin parity in code before AC1/AC2.

---

## §A — LOCKED Q-list (Coder MUST follow)

### Q1 — RoPE oracle: what must it pin? (AC0, load-bearing) — LOCKED
Oracle = NEW python test for ONE compress layer (real-dim HCA fixture). MUST pin + assert:
1. **Channel scope = TRAILING 64** (resolved §0): assert MLX rotates only last
   `qk_rope_head_dim=64` of head_dim=512, leading 448 untouched. NOT `_apply_rope_full_mlx`.
2. **Yarn = YES** (resolved §0): assert MLX compress inv_freq == torch yarn inv_freq
   (theta=160000, factor=16, beta_fast=32, beta_slow=1, original_max=65536,
   attention_factor=1.0). Compare against torch
   `DeepseekV4RotaryEmbedding(cfg).compress_inv_freq` (modeling:153-184) — NOT plain.
3. **Positions** = block-start `i*rate` (rate=128) per window i; HCA compressor RoPEs
   compressed entries at `positions = arange(n_win)*compress_rate + 0` (first_window_position=0,
   stateless) — modeling:675-678. Oracle asserts cos/sin at those positions.
4. **Tolerance** (LOCKED): cross-framework BF16. cos/sin parity **atol=1e-3**;
   rotated-q/kv (downstream) **atol=1e-2**. (13.2 FP4 1e-5 is intra-MLX, too tight here.)
   Oracle MAY compute tables in fp32 both sides for the cos/sin assert (tighter, 1e-5
   acceptable in fp32) AND a bf16 end-to-end rotate assert at 1e-2.

Oracle asserts MLX compress-rope cos/sin (and rotated q/kv) == torch within tol.
**STOP-ESCALATE (S-rope)** ONLY if a math primitive FROZEN cannot supply additively is
discovered (kernel / unsupported concat). BA recon says this will NOT happen (§0); Coder
confirms in code.

### Q2 — HCA compressor stateless contract (§3.1) — LOCKED
`_hca_compressor_mlx(args, x, weights, *, position_ids) -> (compressed_kv, block_bias)`.
Port torch `DeepseekV4HCACompressor.forward` (modeling:655-712), `past_key_values=None` branch:
- `rate = 128` (compress_rates["heavily_compressed_attention"]); `out_dim = head_dim = 512`.
- `kv = x @ wkv.T` `[B,S,512]`; `gate = x @ wgate.T` `[B,S,512]` (torch kv_proj/gate_proj, bias=False).
- `usable = (S//rate)*rate`; `first_window_position = 0`; drop remainder (modeling:671-674).
- reshape `[B, n_win, rate, 512]`; `chunk_gate + position_bias` (modeling:677).
- `compressed = norm( sum( kv * softmax(gate, axis=2, dtype=fp32→cast back), axis=2 ) )` `[B,n_win,512]` (modeling:678-680).
- RoPE compress-yarn-tail at `positions = arange(n_win)*rate` (modeling:681-684), via
  Q1 yarn-tail table helper + FROZEN tail apply.
- `compressed_kv = compressed[:, None]` `[B,1,n_win,512]` (modeling:687).
- `block_bias` `[B,1,S,n_win]`: entry `i` visible to query `t` iff
  `i < (position_ids+1)//rate` else `-inf` (modeling:701-711).
  Short-circuit: torch returns `(compressed_kv, None)` when `S==1 or n_win==0` (modeling:691-693);
  MLX MUST mirror (None block_bias for those degenerate cases — but parity fixture uses S>rate).

**Q2 ape-transpose landmine — RESOLVED: HCA = NO transpose.**
torch `self.position_bias = nn.Parameter(torch.empty(self.compress_rate, self.head_dim))`
= shape **(rate=128, head_dim=512)** (modeling:649), added DIRECTLY to
`chunk_gate.view(B, n_win, rate, head_dim)` (modeling:677). NO `.T`, NO `[:out_dim,:]`
slice. Architect §3.1 "NO transpose" CONFIRMED. (CSA `ape[:out_dim,:].T` is a DIFFERENT
codepath — 13.3b-2, out of scope here.) MLX `weights["compressor_ape"]` must already be
`(rate, head_dim)` at the fixture (synthesize directly so; convert-side transpose is 13.3b-4 CSA-only).

### Q3 — `_attention_real_mlx` HCA branch: reuse vs new (§3.4) — LOCKED
Steps reuse cr=0 `_attention_mlx:632` body math; NEW = compressor + KV-append + mask-append.
Port torch `DeepseekV4Attention.forward` (modeling:801-875):
- **Step 1 (REUSE cr=0 math)**: `q_residual = q_a_norm(q_a(x))`; `q = q_b_norm(q_b(q_residual)).view[B,heads,S,512]`;
  `q = rope(q)`; `kv = kv_norm(kv_proj(x)).view[B,1,S,512]`; `kv = rope(kv)` (modeling:823-831).
  **RoPE here = compress-yarn-tail** (Q1) for HCA layers (rope_layer_type="compress", modeling:760).
- **Step 2 (NEW, HCA branch)**: `compressed_kv, block_bias = _hca_compressor_mlx(args, x, weights, position_ids=...)` (modeling:834-838).
- **Step 3 (NEW)**: `kv_full = concat([kv, compressed_kv], axis=2)` `[B,1,S+n_win,512]` (modeling:838).
- **Step 4 (NEW)**: `mask = concat([sliding_causal_mask, block_bias], axis=-1)` `[B,1,S,S+n_win]` (modeling:847-852).
- **Step 5 (REUSE cr=0)**: multi-head `scores = q @ kv_full.T * scale + mask`; append per-head
  sink logit (`attn_sink`); subtract rowmax; softmax; drop sink col; attend (modeling:716-742 eager + 854-863).
- **Step 6 (REUSE cr=0)**: rope-undo (`-sin` compress-yarn-tail on output rope slice) +
  grouped o_a (block-diagonal, o_groups=8) + o_b (modeling:866-873).

**Q3 sub: projections SHARED with cr=0 — CONFIRMED.** torch HCA layer uses the SAME
`q_a_proj/q_a_norm/q_b_proj/q_b_norm/kv_proj/kv_norm/o_a_proj/o_b_proj/sinks` as sliding
(modeling:786-799, built unconditionally; compressor is the ONLY addition, modeling:799-801).
Real ckpt HCA layer keys = `attn.wq_a/wq_b/wkv/wo_a/wo_b/q_norm/kv_norm/attn_sink` — SAME
names as sliding (architecture.md §1.3). ⇒ HCA branch REUSES cr=0 projection/sink machinery;
compressor uses its OWN `compressor.wkv/wgate/ape/norm` weights only. `attn_sink` dims
identical (`[64]`, per-head) for compress + sliding.

**Q3 sub: HCA compressor uses hidden_states ONLY, NOT q_residual.** torch HCACompressor.forward
signature takes `q_residual` but does NOT use it (modeling:662-712; only Indexer uses q_residual).
⇒ `_hca_compressor_mlx` takes `x` (hidden), NOT q_residual. (CSA indexer needs q_residual — 13.3b-2.)

### Q4 — rope-undo + grouped-o in HCA path? — LOCKED: YES, shared.
torch applies rope-undo + grouped-o for ALL layer types AFTER the compressor branch
(modeling:866-873) — single shared tail, NOT sliding-only. ⇒ HCA `_attention_real_mlx`
MUST include rope-undo (`apply_rotary_pos_emb(out, cos, -sin)`, compress-yarn-tail cos/sin)
+ grouped o_a + o_b, identical to cr=0 tail (`_attention_mlx:632` body, region ~700-755).
NOTE: rope-undo uses the SAME compress-yarn-tail cos/sin as step 1 (negated sin).

### Q5 — additive dispatch byte-identical (§4.2, ADR 0026) — LOCKED
ONE FROZEN edit, `_attention_mlx` (current `deepseek_v4.py:641` `if args.compression_ratio != 0:`):
```python
if args.compression_ratio != 0:
    if _csa_config_error(args) is None:          # tiny proven subset → UNCHANGED
        return _csa_attention_mlx(args, x, weights, index_topk=index_topk)
    return _attention_real_mlx(args, x, weights, index_topk=index_topk)   # NEW
```
**Byte-identical proof (CONFIRMED):** every tiny fixture (11.14/11.15) sets
`compression_ratio==4`, `num_attention_heads==1`, `o_groups==1`, `num_key_value_heads==1`,
`hc_mult==1`, `hidden_size==head_dim`, `q_lora_rank==hidden_size` ⇒ `_csa_config_error`
(deepseek_v4.py:317-334) returns None ⇒ routes `_csa_attention_mlx` UNCHANGED. Real HCA
dims (heads=64, o_groups=8, head_dim=512≠hidden=4096, …) ⇒ `_csa_config_error` returns a
string ⇒ NEW path. NO existing test input changes branch. AC3 asserts both legs.

### Q6 — test fixtures: synthesize, anti-circular (§ anti-circularity) — LOCKED
- **Synthesize** random weights at real HCA dims: `head_dim=512, rate=128, hidden=4096,
  q_lora_rank=1024, o_lora_rank=1024, o_groups=8, num_heads=64, num_kv_heads=1,
  qk_rope_head_dim=64, rms_norm_eps=1e-6`. Cheaper + parity-exact, no 162GB shard read.
- **Anti-circularity**: torch reference computes expected from the SAME synthesized
  weights (load into `DeepseekV4HCACompressor` / `DeepseekV4Attention`). MLX helper reads
  the SAME arrays. NO second MLX primitive used as oracle. torch is the ONLY oracle.
- **One cross-check**: assert synthesized weight key-names + shapes match ONE real HCA
  ckpt layer header (e.g. layer 3: `compressor.wkv [512,4096]`, `compressor.wgate [512,4096]`,
  `compressor.ape [128,512]`, `compressor.norm [512]` — architecture.md §1.3 HCA block) via
  safetensors header read (no tensor load). Shape/name assert only.
- ape synthesized DIRECTLY as `(rate=128, head_dim=512)` (Q2, NO transpose).

### Q7 — STOP conditions (architecture.md §9 + ADR 0026) — LOCKED
Emit `coder-stop.md` + `{"status":"error",...}` if:
- **S-rope (AC0)**: torch compress rope needs a primitive FROZEN cannot supply additively
  (kernel / unsupported concat / non-derivable yarn). BA recon says NO (§0); STOP only if
  code disproves. Epic-level go/no-go.
- **S-frozen-body**: any FROZEN body would need EDIT not additive-new — `_attention_mlx:632`
  cr=0 math, `_csa_attention_mlx:515`, `_csa_windowed_compressor_mlx:347`, `_csa_config_error:317`,
  `_require_csa_config:335`, `_hyperconnection_mlx`, `_hyperhead_mlx`, parity `Model` (deepseek_v4.py:1685),
  `sanitize_weights`, `_apply_rope_full_mlx:311`, `_rope_tail_tables_mlx:227`, `_apply_rope_tail_mlx:269`.
- **S-tiny-regression**: any tiny CSA fixture (11.14/11.15) RED.
- **S-parity**: `_hca_compressor_mlx` or HCA `_attention_real_mlx` cannot match torch within
  tol at real dims after best-effort → STOP w/ failing assertion.
- **S-backward**: 13.3a-3 backward AC RED (additive dispatch must not break nn port).

---

## §B — Acceptance criteria → trackable tests (TDD red-first)

- **AC0 (LOAD-BEARING go/no-go)** — `tests/test_compress_rope_oracle.py` (NEW): ONE real-dim
  compress layer. Assert MLX compress-rope cos/sin == torch `DeepseekV4RotaryEmbedding`
  `compress_inv_freq`/cos/sin (yarn theta=160000, factor=16, attention_factor=1.0) within Q1 tol
  (fp32 cos/sin 1e-5, bf16 rotated 1e-2). Pin in docstring: channel-scope=TRAILING-64,
  yarn=YES, positions=`i*rate`. If un-GREEN additively → STOP-ESCALATE, do NOT touch AC1/AC2.
- **AC1** — `tests/test_hca_compressor_mlx_parity.py` (NEW): `_hca_compressor_mlx` output
  `(compressed_kv, block_bias)` matches torch `DeepseekV4HCACompressor.forward`
  (`past_key_values=None`) within tol at real dims (head_dim=512, rate=128, hidden=4096).
  Q2 NO-transpose pinned.
- **AC2** — `tests/test_attention_real_mlx_hca_parity.py` (NEW): HCA `_attention_real_mlx`
  matches torch `DeepseekV4Attention.forward` for an HCA layer within tol. Q3 shared-projections
  + Q4 rope-undo/grouped-o confirmed.
- **AC3** — `tests/test_13_3b_1_dispatch_additive.py` (NEW): tiny compress input still routes
  `_csa_attention_mlx` (assert `_csa_config_error(args) is None` + return byte-identical to
  pre-edit); real-dim HCA input routes `_attention_real_mlx` (no `NotImplementedError`). Q5.
- **regression**: 13.2 FP4 (9) + 13.3a nn (12, incl 13.3a-3 backward AC) + tiny CSA (11.14/11.15)
  all GREEN. Full suite: 1 RED (pre-existing Test #8) / **557+4new** PASS / 13 SKIP / **0 introduced RED**.

AC convention (every slice, architecture.md subslice-breakdown): (a) parity vs torch ref;
(b) tiny CSA fixtures GREEN; (c) backward AC (13.3a-3) GREEN; (d) `git diff --check` clean +
FROZEN bodies byte-intact (AST/grep proof, not zero-line-diff — AGENTS.md untracked-file caveat).

---

## §C — Blast-radius (Architect §4.2/§8 + subslice-breakdown — confirmed set)

| # | File | Change |
|---|---|---|
| 1 | `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` | ADDITIVE ONLY: NEW helpers `_hca_compressor_mlx` + `_attention_real_mlx` (HCA branch) + NEW yarn-tail compress-rope table helper, in a NEW module section; ONE additive dispatch branch in `_attention_mlx` (current `:641`). Tiny path byte-identical. NO FROZEN body edits. |
| 2 | `tests/test_compress_rope_oracle.py` | NEW (AC0) |
| 3 | `tests/test_hca_compressor_mlx_parity.py` | NEW (AC1) |
| 4 | `tests/test_attention_real_mlx_hca_parity.py` | NEW (AC2) |
| 5 | `tests/test_13_3b_1_dispatch_additive.py` | NEW (AC3) |

NO nn-file edit (`deepseek_v4_nn.py` AttentionNN guard lift = 13.3b-3). NO convert (13.3b-4).
NO CSA/Indexer helpers (13.3b-2). Helper-level + functional tests only.

FROZEN — MUST NOT EDIT BODY (architecture.md §8): `_csa_config_error`, `_require_csa_config`,
`_csa_attention_mlx`, `_csa_compressor_mlx`, `_csa_indexer_mlx`, `_csa_windowed_compressor_mlx`,
`_hyperconnection_mlx`, `_hyperhead_mlx`, `_attention_mlx` cr=0 body, `sanitize_weights`,
parity `Model:1685`, FP4/i8 dequant, 9 ADR-0017 forbidden symbols, `_apply_rope_full_mlx`,
`_rope_tail_tables_mlx`, `_apply_rope_tail_mlx`.

---

## §D — Dev-days / sequence
5 dev-days (oracle 1.5 + HCA compressor 1.5 + HCA attn 2). FIRST Epic story; go/no-go.
Serial; file-mutating coder one slice, reviewed before 13.3b-2. AC0 GREEN-first gates AC1/AC2.

## Deliverables (this BA pass)
1. THIS doc.
2. `.cmux-status/ba.done` (`{"status":"ok","role":"BA"}`).
3. `docs/backlog.md` 13.3b-1 row (scope LOCKED + AC0 GO flag).
4. In-pane JSON `{"status":"ok","role":"BA"}` surface:81 only.

AC0 verdict = **GO** (§0). No `ba-stop.md`. Epic 13.3b proceeds.
