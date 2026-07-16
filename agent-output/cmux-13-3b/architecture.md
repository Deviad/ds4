# Story 13.3b — Architecture: real-dim CSA + HCA + Indexer MLX port

Architect r0. Extends `agent-output/cmux-13-3a/architecture.md` (§0–§15) + ADR 0025.
Option B picked by operator: full CSA+HCA+Indexer real-dim port. Additive FROZEN
contract expansion (ADR 0026). Caveman-ultra prose; code/paths/line-numbers byte-exact.

---

## §0 Status recap (honored, NOT re-arbitrated)

- ADR 0017 boundary X / 9 forbidden symbols: CONSUMED unchanged.
- ADR 0024 FP4 on-the-fly dequant: CONSUMED unchanged.
- ADR 0025 nn.Module sibling port (`deepseek_v4_nn.py`, `model_type="deepseek_v4_nn"`):
  this story EXTENDS it. Parity fixture `Model` (`deepseek_v4.py:1685`) + 27 parity
  tests STAY GREEN. Tiny CSA fixtures (Story 11.14/11.15) STAY GREEN.
- 13.3a-3 baseline HEAD `702199f`: backward AC GREEN, `model_type` wired,
  sanitize/load self-consistent. Build on it.

This story makes the real 43-layer Flash ckpt forward + smoke-train.

---

## §1 Ground-truth recon (verified from ckpt, NOT brief)

Ckpt `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/` (162GB, 46 shards, 69187 keys).
Index `model.safetensors.index.json`.

### §1.1 Layer classification — by WEIGHT PRESENCE (authoritative)

Shimmed `config.json` `layer_types` is FLAT all-`sliding_attention` (stale/wrong) —
do NOT trust it. Classify by which attn sub-weights each layer carries:

| kind | rule | layers | count |
|---|---|---|---|
| sliding | no `attn.compressor.*`, no `attn.indexer.*` | 0, 1 | 2 |
| CSA | has `attn.indexer.*` (⇒ also `attn.compressor.*`) | 2,4,6,…,40,42 (even≥2 + 42) | 21 |
| HCA | has `attn.compressor.*`, NO `attn.indexer.*` | 3,5,7,…,41 (odd≥3) | 20 |

> **Brief correction (cite evidence):** brief recon said cr=0={0,1,42}, cr=4=21,
> cr=128=20. Ckpt shows **layer 42 carries `layers.42.attn.indexer.*` ⇒ CSA**, not
> sliding. Corrected: sliding={0,1} (2), CSA=21, HCA=20. `compress_rate_csa=4`,
> `compress_rate_hca=128`. `mlp_layer_types`: hash_moe={0,1,2}, moe={3..42} (3 hash).

### §1.2 Real config fields (config.json, verified)

```
hidden_size=4096  head_dim=512  num_attention_heads=64  num_key_value_heads=1
o_groups=8  o_lora_rank=1024  q_lora_rank=1024  qk_rope_head_dim=64
index_head_dim=128  index_n_heads=64  index_topk=512
hc_mult=4  hc_sinkhorn_iters=20  num_hidden_layers=43  num_hash_layers=3
sliding_window=128  rope_theta=10000  compress_rope_theta=160000
rope_scaling={type:yarn, factor:16, beta_fast:32, beta_slow:1, original_max_position_embeddings:65536}
n_routed_experts=256  num_experts_per_tok=6  n_shared_experts=1  moe_intermediate_size=2048
expert_dtype=fp4
```

### §1.3 Real ckpt key names + shapes (verified safetensors headers)

Ckpt uses DeepSeek-native names — these ALREADY match FROZEN `_csa_*_mlx` semantics,
NOT torch HF names. This DE-RISKS the port (no algorithm re-derivation, only rename+transpose).

CSA layer (e.g. layer 2):
```
attn.compressor.wkv.weight        [1024, 4096]   = (2*head_dim=1024, hidden)
attn.compressor.wgate.weight      [1024, 4096]   = (2*head_dim, hidden)
attn.compressor.ape               [4, 1024]  F32 = (compress_rate=4, 2*head_dim)   ← (rate,2*out)
attn.compressor.norm.weight       [512]          = (head_dim)
attn.indexer.compressor.wkv.weight   [256, 4096] = (2*index_head_dim=256, hidden)
attn.indexer.compressor.wgate.weight [256, 4096]
attn.indexer.compressor.ape          [4, 256] F32 = (rate, 2*index_head_dim)
attn.indexer.compressor.norm.weight  [128]        = (index_head_dim)
attn.indexer.weights_proj.weight     [64, 4096]   = (index_n_heads=64, hidden)   ← FROZEN "indexer_proj"
attn.indexer.wq_b.{weight,scale}     [8192, 1024] = (index_n_heads*index_head_dim=8192, q_lora_rank)  ← FROZEN "indexer_wq_b"
attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}     core MLA (BF16, scale sidecar)
attn.{q_norm,kv_norm}.weight   attn.attn_sink [64] F32
attn_norm.weight  ffn_norm.weight  hc_attn_{fn[24,16384],base[24],scale[3]}  hc_ffn_{...}
```

HCA layer (e.g. layer 3): SIMPLER — single compressor, NO Ca/Cb overlap, NO indexer.
```
attn.compressor.wkv.weight    [512, 4096]   = (head_dim=512, hidden)    ← NOT 2*head_dim
attn.compressor.wgate.weight  [512, 4096]   = (head_dim, hidden)
attn.compressor.ape           [128, 512] F32 = (compress_rate=128, head_dim) ← (rate,out), no overlap
attn.compressor.norm.weight   [512]
(NO attn.indexer.*)  + same core MLA + hc params
```

sliding layer (0,1): core MLA only (matches FROZEN cr=0 `_attention_mlx` path).

Global: `embed.weight`, `head.weight`, `hc_head_{fn[4,16384],base,scale}`, `mtp.*` (stripped).

---

## §2 The CORE architectural finding (drives whole design)

**FROZEN tiny `_csa_attention_mlx` (`deepseek_v4.py:515`) is NOT the real architecture.**
It collapses CSA to single-head compressed-only attention returning `[B,S,head_dim]`.

The REAL torch architecture (`modeling_deepseek_v4.py` `DeepseekV4Attention.forward:805`)
is: **standard multi-head MLA sliding attention, with extra compressed-KV entries
APPENDED to the KV axis and a `block_bias` appended to the attention mask.**

```
torch Attention.forward (805-870):
  q  = rope(q_b(q_a_norm(q_a(h))))              # [B, heads, S, head_dim]
  kv = rope(kv_norm(kv(h)))                     # [B, 1, S, head_dim]  (single KV head)
  if compressor:                               # CSA or HCA
      compressed_kv, block_bias = compressor(h, q_residual, position_ids, ...)
      kv = cat([kv, compressed_kv], dim=2)     # [B, 1, S+T, head_dim]
      mask = cat([sliding_mask, block_bias])   # [B, 1, S, S+T]
  attn = eager_attention(q, kv, kv, mask, sinks)  # multi-head + per-head sink
  out  = rope_undo(attn, -sin)                  # conjugate rope on output rope slice
  out  = o_b(GroupedLinear_o_a(grouped(out)))   # grouped low-rank output
```

⟹ The real cr≠0 path = **the existing multi-head cr=0 `_attention_mlx` path (632) +
compressed-KV-append + block_bias**. The cr=0 path is already real-dim-capable
(num_kv_heads==1, heads%o_groups==0, per-head sinks, grouped o_a/o_b). The DELTA
is: produce compressed_kv + block_bias, concat, extend mask. Everything else reuses
the proven cr=0 machinery.

⟹ Dispatch is ADDITIVE: tiny subset (passes `_csa_config_error`) keeps tiny path;
real dims route to NEW helpers. No FROZEN symbol changes meaning.

### §2.1 Reuse map (what FROZEN already gives us — dim-generic)

| need | reuse | dim-generic? |
|---|---|---|
| CSA compressor (Ca/Cb overlap, rate=4) | FROZEN `_csa_windowed_compressor_mlx:347` | YES — out_dim/rate parametric |
| CSA indexer compressor (overlap, index_head_dim) | FROZEN `_csa_windowed_compressor_mlx:347` (out_dim=index_head_dim) | YES |
| multi-head sliding MLA + sinks + grouped-o | FROZEN `_attention_mlx:632` cr=0 body | YES (real dims) |
| HC stream / HyperHead (hc_mult=4) | FROZEN `_hyperconnection_mlx:550` / `_hyperhead_mlx:600` | YES (validate hc_mult) |
| FP4 experts dequant | ADR 0024 `_dequantize_fp4_block_scale_mlx` | YES (proven 13.3a-2 AC5) |

| need | NEW helper (additive) | why new |
|---|---|---|
| HCA compressor (single window, rate=128, NO overlap) | `_hca_compressor_mlx` | torch HCACompressor:362 has no Ca/Cb; FROZEN windowed is Ca/Cb-only |
| Indexer top-k → block_bias (real path) | `_indexer_mlx` + `_indexer_scorer_mlx` | FROZEN `_csa_indexer_mlx:435` returns valid_mask for tiny single-head attn; real path needs `[B,1,S,T]` block_bias + `-1` sentinel + clamp (torch Indexer:462) |
| real multi-head compressed attention | `_attention_real_mlx` (cr≠0, multi-head, KV-append) | FROZEN `_csa_attention_mlx:515` is single-head collapse |
| grouped output block-diagonal | reuse cr=0 o_a/o_b OR `_grouped_linear_mlx` | confirm cr=0 o_a == torch GroupedLinear:303 bmm at o_groups=8 |

---

## §3 MLX functional helper design (NEW, additive, dict-based stateless)

All NEW helpers mirror FROZEN pattern: `(args, x, weights: dict[str, mx.array], *, ...)`,
stateless per-forward (training full-seq, `past_key_values=None` torch branch only),
return `mx.array`. NO new ModelArgs state (Option A: per-forward kwargs). Live BESIDE
FROZEN helpers in a NEW module section, IMPORT FROZEN primitives, never edit FROZEN bodies.

### §3.1 `_hca_compressor_mlx(args, x, weights, *, position_ids) -> tuple[compressed_kv, block_bias]`

Port of torch `DeepseekV4HCACompressor.forward:391` (stateless branch only).
- `rate = compress_rate_hca = 128`; `out_dim = head_dim = 512`.
- `kv = x @ wkv.T` `[B,S,head_dim]`; `gate = x @ wgate.T` `[B,S,head_dim]`.
- `usable = (S//rate)*rate`; reshape `[B, n_win, rate, head_dim]`; `+ position_bias` (ape `(rate,head_dim)`, NO transpose).
- `compressed = norm( sum(kv * softmax(gate, axis=2, fp32), axis=2) )` → `[B, n_win, head_dim]`.
- RoPE at positions `i*rate` with compress rope (theta=compress_rope_theta + yarn): see §3.5.
- `compressed_kv = compressed[:, None]` `[B,1,n_win,head_dim]`.
- `block_bias`: causal — entry `i` visible to query `t` iff `i < (position_ids+1)//rate`;
  else `-inf`. Shape `[B,1,S,n_win]`. Port torch:454-461.
- NO indexer, NO Ca/Cb overlap. Single window pooling.

### §3.2 `_indexer_scorer_mlx(q, compressed_kv, hidden, *, index_n_heads, index_head_dim) -> scores`

Port torch `DeepseekV4IndexerScorer.forward:455`.
- `scores = relu(q @ compressed_kv.T) * index_head_dim**-0.5`   `[B,S,H,T]`.
- `weights = (hidden @ weights_proj.T) * index_n_heads**-0.5`   `[B,S,H]`.
- return `sum(scores * weights[...,None], axis=2)`   `[B,S,T]`.
- fp32 accum (torch `.float()`).

### §3.3 `_indexer_mlx(args, hidden, q_residual, weights, *, position_ids, index_topk) -> top_k_indices`

Port torch `DeepseekV4Indexer.forward:511`.
- indexer compressor = FROZEN `_csa_windowed_compressor_mlx` (Ca/Cb overlap, out_dim=index_head_dim, rate=4). REUSE.
- `q = (q_residual @ wq_b.T).reshape(B,S,index_n_heads,index_head_dim)`; full RoPE at position_ids, compress rope.
- `scores = _indexer_scorer_mlx(...)`   `[B,S,T]`.
- causal_threshold = `(position_ids+1)//rate`; future_mask scores→`-inf`; `top_k = min(index_topk, T)`;
  `top_k_indices = topk(scores, k).indices`; mark picks `>= causal_threshold` with `-1` sentinel. torch:572-585.
- MLX has no native masked `topk`-with-sentinel: emulate via `mx.argpartition`/`mx.argsort` desc + gather + where. Coder owns exact MLX op choice; AC = parity vs torch indices.

### §3.4 `_csa_attention_real_mlx` / `_hca_attention_real_mlx` (or unified `_attention_real_mlx`)

Unified real multi-head compressed path. Steps (mirror cr=0 `_attention_mlx:632` body, ADD compressed-KV):
1. q/kv projections + norms + RoPE (compress rope for CSA/HCA layers). REUSE cr=0 math.
2. branch on `compress_rate`:
   - HCA (rate=128): `compressed_kv, block_bias = _hca_compressor_mlx(...)`.
   - CSA (rate=4): `compressed_kv = _csa_compressor_mlx_realkeys(...)` (FROZEN windowed, real keys);
     `top_k_indices = _indexer_mlx(...)`; build `block_bias [B,1,S,T]` from top-k (`-inf` except selected, clamp `-1`→drop). Port torch:739-754.
3. `kv_full = cat([kv, compressed_kv], axis=2)`   `[B,1,S+T,head_dim]`.
4. `mask = cat([sliding_causal_mask, block_bias], axis=-1)`   `[B,1,S,S+T]`.
5. multi-head scores `q_by_head @ kv_full.T * scale + mask`; append per-head sink logit; softmax; drop sink; attend.
6. rope-undo (`-sin`) on output rope slice; grouped o_a (block-diagonal) ; o_b. REUSE cr=0 tail (632 body 700-755).

> Stateless: torch `past_key_values=None` branch ⇒ `first_window_position=0`,
> no overlap-cache carry, drop remainder tokens. Matches FROZEN windowed compressor
> cache-less contract (`deepseek_v4.py:347` docstring).

### §3.5 RoPE parity risk (FLAG — Coder must resolve in 13.3b-1 first)

torch compress layers use `rope_layer_type="compress"` ⇒ `compress_rope_theta=160000`
+ yarn scaling. FROZEN cr=0 path uses `_rope_tail_tables_mlx(..., rope_theta)` (partial
tail, theta=10000, NO yarn). FROZEN windowed compressor uses `_apply_rope_full_mlx`
(FULL channel). torch `apply_rotary_pos_emb:344` applies to TRAILING `rope_dim` slice.

**Unknowns the Coder MUST pin against torch BEFORE writing the real path:**
1. Does torch compress rope cover full head_dim or only trailing `qk_rope_head_dim`?
   (Determines whether FROZEN `_apply_rope_full_mlx` reuse is parity-exact or needs a
   partial-tail compress variant.)
2. Does compress rope apply yarn scaling (factor=16) — if so a yarn-scaled cos/sin
   table helper is needed (FROZEN tiny had no yarn).

→ 13.3b-1 AC#0: build a torch-vs-MLX rope oracle for ONE compress layer; lock the
rope contract. **STOP-ESCALATE if torch compress rope needs a math primitive FROZEN
cannot supply additively** (e.g. a kernel) — that flips this from port to new-kernel work.

---

## §4 nn.Module wiring (`deepseek_v4_nn.py`, SANCTIONED edits)

Mirror 13.3a pattern: params as `nn.Linear`/`mx.array` leaves on submodules; `__call__`
builds a `weights` dict via `linear_weight(...)` then delegates to the functional helper.

### §4.1 `AttentionNN` (`deepseek_v4_nn.py:135`) — guard lift + compressor submodules

- LIFT guard `:153-154` (`compression_ratio != 0 → NotImplementedError`). Replace with:
  construct compressor/indexer submodules when `layer_types[layer_idx] != "sliding_attention"`.
- ADD submodules (CSA): `compressor` (wkv/wgate `nn.Linear`, `ape`+`norm.weight` params),
  `indexer.compressor` (same), `indexer.weights_proj` `nn.Linear`, `indexer.wq_b` `nn.Linear`.
- ADD submodules (HCA): `compressor` only (wkv/wgate/ape/norm).
- `__call__`: if sliding → existing `_attention_mlx` cr=0 (UNCHANGED); else build real
  weights dict + `position_ids = mx.arange(S)` + delegate to `_attention_real_mlx`.
- Keep `linear_weight(...)` helper (QuantizedLinear/LoRA-aware) — reuse for compressor leaves.

### §4.2 dispatch in `_attention_mlx` (`deepseek_v4.py:640`) — ADDITIVE

FROZEN file IS edited here but ADDITIVELY (new branch + new symbols only; tiny path byte-identical):
```python
if args.compression_ratio != 0:
    if _csa_config_error(args) is None:          # tiny proven subset → UNCHANGED
        return _csa_attention_mlx(args, x, weights, index_topk=index_topk)
    return _attention_real_mlx(args, x, weights, index_topk=index_topk)   # NEW real path
```
> This is the ONLY FROZEN-file edit, and it is purely additive: existing tiny inputs
> still satisfy `_csa_config_error(args) is None` → identical return. New real dims
> (which previously raised `NotImplementedError` via `_require_csa_config`) now route
> to the new symbol. No existing test input changes branch. See ADR 0026 §Decision.

### §4.3 `_grouped_linear_mlx` / GroupedLinear parity

torch `DeepseekV4GroupedLinear.forward:328` = block-diagonal bmm per o_group.
Confirm FROZEN cr=0 o_a application (632 body, `expected_o_a_shape=(o_groups*o_lora_rank,
heads_per_group*head_dim)`) is mathematically identical to torch bmm at o_groups=8.
If identical → REUSE, no new helper. If not → add `_grouped_linear_mlx` (additive).
13.3b-3 AC decides.

---

## §5 Real ckpt convert / key remap (13.3b-4 — convert side, NOT FROZEN)

NEW script `scripts/remap_ds4_nn_weights.py` (or extend shim). Ckpt-native → nn-port keys.

### §5.1 Rename table (ckpt → nn-port)

```
embed.weight                          → model.embed_tokens.weight
head.weight                           → lm_head.weight            (verify nn output head name)
hc_head_{fn,base,scale}               → model.<hyperhead leaf>     (13.3a leaf names)
layers.N.attn_norm.weight             → model.layers.N.input_layernorm.weight
layers.N.ffn_norm.weight              → model.layers.N.post_attention_layernorm.weight
layers.N.attn.wq_a.{weight,scale}     → model.layers.N.self_attn.q_a_proj.weight  (drop scale if BF16)
layers.N.attn.wq_b.*                  → ...self_attn.q_b_proj.weight
layers.N.attn.wkv.*                   → ...self_attn.kv_proj.weight
layers.N.attn.wo_a.*                  → ...self_attn.o_a_proj.weight
layers.N.attn.wo_b.*                  → ...self_attn.o_b_proj.weight
layers.N.attn.q_norm.weight           → ...self_attn.q_norm.weight
layers.N.attn.kv_norm.weight          → ...self_attn.kv_norm.weight
layers.N.attn.attn_sink               → ...self_attn.sinks
layers.N.attn.compressor.wkv.weight   → ...self_attn.compressor.wkv.weight
layers.N.attn.compressor.wgate.weight → ...self_attn.compressor.wgate.weight
layers.N.attn.compressor.ape          → ...self_attn.compressor.ape    (CSA/indexer: TRANSPOSE; HCA: keep)
layers.N.attn.compressor.norm.weight  → ...self_attn.compressor.norm.weight
layers.N.attn.indexer.compressor.*    → ...self_attn.indexer.compressor.*   (CSA only; ape TRANSPOSE)
layers.N.attn.indexer.weights_proj.weight → ...self_attn.indexer.weights_proj.weight
layers.N.attn.indexer.wq_b.*          → ...self_attn.indexer.wq_b.weight
layers.N.hc_attn_{fn,base,scale}      → model.layers.N.<hc attn leaf>   (13.3a)
layers.N.hc_ffn_{fn,base,scale}       → model.layers.N.<hc ffn leaf>
layers.N.ffn.gate.weight              → model.layers.N.mlp.gate.weight
layers.N.ffn.gate.tid2eid             → model.layers.N.mlp.gate.tid2eid   (hash layers 0,1,2)
layers.N.ffn.gate.bias                → model.layers.N.mlp.gate.bias      (moe layers)
layers.N.ffn.shared_experts.w1.*      → model.layers.N.mlp.shared_experts.gate_proj.weight
layers.N.ffn.shared_experts.w3.*      → ...shared_experts.up_proj.weight
layers.N.ffn.shared_experts.w2.*      → ...shared_experts.down_proj.weight  (w1=gate,w3=up,w2=down)
layers.N.ffn.experts.M.w{1,2,3}.{weight,scale}  → STACK over M → model.layers.N.mlp.experts.w{1,2,3}_{weight,scale}  (n_routed,…)
mtp.*                                  → DROP (sanitize strips)
```

### §5.2 ape transpose

ckpt CSA/indexer `ape` is `(rate, 2*out_dim)`; FROZEN `_csa_windowed_compressor_mlx:369`
expects `(2*out_dim, rate)`. Remap transposes CSA + indexer ape. HCA new helper uses
`(rate, head_dim)` directly (torch convention) → NO transpose.

### §5.3 dtypes

attn core (wq_a/wq_b/wkv/wo_a/wo_b) + indexer.wq_b are BF16 in ckpt with a redundant
`.scale` sidecar (shim already unpacked FP8 E4M3 → BF16). Remap DROPS those `.scale`.
Routed experts STAY FP4 (uint8 weight + BF16 scale) → stacked, `_dequantize_fp4` at forward (ADR 0024).
ape/sinks/hc params are F32 → keep.

### §5.4 load

nn `Model.sanitize` (`deepseek_v4_nn.py:431`) delegates FROZEN `sanitize_weights:1600`
(mtp strip, pass-through). After remap, keys match nn param tree → `model.load_weights(list(...))`.
Convert via `mlx_lm.convert` resolves nn module by `model_type="deepseek_v4_nn"` (13.3a-3 wiring).

---

## §6 Sub-slice sequence (see subslice-breakdown.md for AC/STOP/dev-days)

```
13.3b-1  HCA real path (compressor + multi-head compressed attn, no indexer) + RoPE oracle  [first; pins rope contract]
13.3b-2  CSA + Indexer real path (reuse FROZEN windowed compressor; new indexer top-k→block_bias)  [STOP-split if >7d → 2a indexer / 2b csa attn]
13.3b-3  GroupedLinear parity confirm + _attention_mlx cr≠0 dispatch + AttentionNN full guard lift + integration
13.3b-4  real ckpt remap script (rename + ape transpose + per-expert→stacked + FP8-scale drop + model. prefix) + load
13.3b-5  smoke-train 43-layer real (mlx_lm.lora --train --iters 20)
```

Serial. File-mutating coder slices one at a time, reviewed before next (AGENTS.md).
HCA first: simpler (no indexer/overlap), pins the rope contract cheaply, de-risks CSA.

---

## §7 FROZEN contract expansion (ADR 0026 summary)

- ONLY FROZEN-file edit: additive dispatch branch in `_attention_mlx:640` (§4.2) +
  NEW symbols (`_hca_compressor_mlx`, `_indexer_mlx`, `_indexer_scorer_mlx`,
  `_attention_real_mlx`, optional `_grouped_linear_mlx`) appended.
- Tiny guards (`_csa_config_error:317`, `_require_csa_config:335`) UNCHANGED and STILL
  GATE the tiny `_csa_attention_mlx` path. Tiny fixtures (11.14/11.15) hit identical code.
- NO FROZEN symbol changes meaning. Additive only.
- `AttentionNN:153` guard LIFTED (SANCTIONED nn-file edit, not FROZEN — ADR 0025 owns nn file).

---

## §8 Blast-radius matrix

SANCTIONED edits:
- `deepseek_v4.py`: ADD new helpers + ADD dispatch branch `:640` (additive). NO body edits to FROZEN helpers.
- `deepseek_v4_nn.py`: `AttentionNN` guard lift + compressor/indexer submodules + `__call__` real branch.
- NEW `scripts/remap_ds4_nn_weights.py` (13.3b-4).
- NEW tests under `tests/`.

FROZEN — MUST NOT touch (carry ADR 0017/0024/0025 + §11 13.3a):
- `_csa_config_error`, `_require_csa_config`, `_csa_attention_mlx`, `_csa_compressor_mlx`,
  `_csa_indexer_mlx`, `_csa_windowed_compressor_mlx` BODIES, `_hyperconnection_mlx`,
  `_hyperhead_mlx`, `_attention_mlx` cr=0 BODY (700-755), `sanitize_weights`,
  parity `Model:1685`, FP4/i8 dequant primitives, 9 ADR-0017 forbidden symbols.

---

## §9 STOP-ESCALATE rules (Architect-prescribed; Coder enforces)

1. torch compress rope needs a math/kernel primitive FROZEN cannot supply additively (§3.5) → STOP.
2. Any tiny CSA parity test (11.14/11.15) flips RED → STOP (contract violated, not additive).
3. Backward AC (13.3a-3) flips RED → STOP.
4. Any single sub-slice estimate >7 dev-days even after the prescribed split → STOP, re-scope to operator.
5. A real-path requirement forces a SEMANTIC change to a FROZEN symbol (not additive) → STOP.
6. **Total Epic envelope >3 weeks** (it IS — see §10) → already PRESCRIBED Epic re-scope; operator confirms (Option B implies sanction).

---

## §10 Epic-scope prescription (MANDATORY — STOP-rule per task brief)

Summing subslice-breakdown.md estimates: **~19–23 dev-days ≈ 4.0–4.6 calendar weeks**.
This EXCEEDS the 3-week single-story ceiling. Per task-brief STOP-ESCALATE + AGENTS.md:

**PRESCRIPTION:** treat 13.3b as **Epic 13.3b**, with 13.3b-1…13.3b-5 promoted to
independent gated STORIES (each its own BA→Architect→Coder→Reviewer+Tester cycle,
own `.done` markers, own handoff docs). Operator already picked Option B (multi-week)
→ Epic envelope is sanctioned; this prescription FORMALIZES it rather than running
five slices under one story. Gate between every story; do NOT start story N+1 until
story N reviewer+tester GREEN. 13.3b-1 (HCA + rope oracle) is the first story and the
go/no-go for the whole Epic (its rope-oracle result confirms FROZEN reuse is viable).

This is the architect-mandated re-scope; not a silent multi-week run.
