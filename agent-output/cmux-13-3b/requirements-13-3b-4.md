# Requirements — Story 13.3b-4 (real ckpt remap script + load)

BA r0. Convert side ONLY. NO FROZEN edit. NO nn-port code edit. Predecessor 13.3b-3
HEAD `d1f1488` (41/43 real layers run real-dim through `_attention_mlx` dispatch; nn
port `deepseek_v4_nn.py` wired). 13.3b-5 owns smoke-train. Architect §5 owns design;
BA confirms scope + Q1-Q4 + AC. Ground truth = nn `model.parameters()` tree (probed),
NOT §5.1 prose.

---

## §0 Verdict — GO (no STOP). One §5.2 spec-staleness flagged (Coder-resolvable).

- Q3 STOP-rule does NOT fire: every nn param target has a ckpt source; no orphan tensor.
- Q4 nn tree probed; ckpt→nn cross-check clean.
- ONE divergence between §5.1/§5.2 prose and the WIRED real path (ape orientation,
  §0.4 below). It is spec-staleness, NOT an unknown-arch STOP: the real dispatch
  helpers (13.3b-2b) already define the exact orientation, it matches ckpt-native AND
  nn-leaf shape, and the GREEN 13.3b-3 test pins it. Coder follows the WIRED path, not
  §5.2. Flagged for Architect doc-fix, not escalated.

### §0.1 Scope (Q1 resolved)
- **NEW `scripts/remap_ds4_nn_weights.py`** (RECOMMENDED over extending shim).
  Rationale: `scripts/shim_ds4_safetensors.py` is a byte-level FP8→BF16 shard
  rewriter (`rewrite_shard:269`, `convert_payload:253`, `copy_sidecars:346` already
  writes `model_type="deepseek_v4_nn":361`). Its job = dtype unpack + sidecar copy,
  NOT key-space renaming. Key remap is a different concern (rename + transpose +
  per-expert stack + scale-drop + `model.` prefix) operating on the ALREADY-shimmed
  dir. Mixing them couples two pipelines and bloats a 446-line single-purpose file.
  A NEW script keeps each surface narrow (AGENTS.md "keep public APIs narrow";
  ADR 0025 nn-file ownership unaffected — convert-only). Extending the shim is
  ACCEPTABLE per Architect §5 but NOT cleaner — BA picks NEW script.
- NO edit to `deepseek_v4.py` (FROZEN) or `deepseek_v4_nn.py` (nn port). Convert side
  only. NEW tests under `tests/`.

### §0.2 Validation budget (Q2 resolved — mirror 13.3a-3 Q9)
- 16GB RAM constraint: grading does **NOT** require a full 162GB / 46-shard / 69187-key
  materialized run. Shape-only + sub-checkpoint validation is ACCEPTABLE.
- Mechanism: read safetensors **headers** (dtype+shape, zero tensor bytes — proven in
  this BA's probes) + load ONE real CSA layer (e.g. 2) + ONE real HCA layer (e.g. 3) +
  ONE sliding layer (0 or 1) + globals, into the nn module, strict. Forward those on a
  tiny seq. Full-ckpt key-set equality is checked at TEMPLATE level over the index
  (cheap) plus per-instance on the loaded sub-checkpoint.

### §0.3 STOP-rule confirmation (Q3)
- "ckpt tensor with no nn param target → STOP" did **NOT** materialize. Cross-check
  (§3) shows every consumed ckpt template maps to an nn leaf. The only un-consumed
  ckpt templates are DELIBERATE drops: `mtp.*` (sanitize strips, §5.1/§5.4) and the
  FP8 `.scale` sidecars on BF16 attn-core + `indexer.wq_b` (§5.3 drop). No unknown
  arch piece. Exact shapes pinned in §2.

### §0.4 ape orientation — §5.2 prose is STALE (FLAG for Architect, Coder follows wired path)
- §5.2 says: "ckpt CSA/indexer `ape` is `(rate, 2*out_dim)`; nn port expects
  `(2*out_dim, rate)` → remap TRANSPOSES CSA + indexer ape; HCA NO transpose." That
  premise cites FROZEN `_csa_windowed_compressor_mlx:368` (`expected = (2*out_dim, rate)`).
- **But that FROZEN helper is the LEGACY/tiny path, NOT the wired real path.** The real
  dispatch (13.3b-2b, what `_attention_mlx:1203` routes 41 real layers to) uses:
  - `_csa_compressor_real_mlx:819` → `expected ape = (rate, 2*out_dim)` token-major, NO transpose.
  - `_indexer_mlx:653` → `expected indexer ape = (rate, 2*out_dim)`, NO transpose.
  - `_hca_compressor_mlx:773` → `expected ape = (rate, out_dim)`, NO transpose.
- nn leaves (probed): `compressor.ape (4,1024)`, `indexer.compressor.ape (4,256)`,
  HCA `compressor.ape (128,512)` — ALL `(rate, ...)` token-major.
- ckpt headers (probed): CSA `compressor.ape [4,1024]`, CSA `indexer.compressor.ape [4,256]`,
  HCA `compressor.ape [128,512]` — IDENTICAL to nn leaves.
- GREEN 13.3b-3 test (`test_13_3b_3_nn_mixed_layers_forward.py:105,109,120`) feeds
  `_matrix(4, 2*head_dim)` / `_matrix(4, 2*index_head_dim)` / `_matrix(128, head_dim)`
  — token-major, NO transpose — and passes through the wired forward.
- **CONCLUSION: ape needs ZERO transpose for ALL three (CSA / indexer / HCA).** ckpt
  layout == nn-leaf layout == wired-real-path expectation. §5.2's transpose is a relic
  of the superseded FROZEN tiny helper. Coder MUST copy ape verbatim (no `.T`).
  Architect: please correct architecture.md §5.2 (drop the CSA/indexer transpose).

---

## §1 nn param tree — Q4 GROUND TRUTH (probed, this is the remap TARGET)

`Model(ModelArgs.from_dict(real_config | model_type="deepseek_v4_nn"))`,
`tree_flatten(model.parameters())`: **1460 leaves, 44 unique templates** (43 layers).
Remapped ckpt key set MUST equal this set EXACTLY (AC1).

### §1.1 Global (5)
```
model.embed_tokens.weight        ← embed.weight
lm_head.weight                   ← head.weight
model.norm.weight                ← norm.weight
model.hc_head.{fn,base,scale}    ← hc_head_{fn,base,scale}
```

### §1.2 Per-layer always-present (per N in 0..42)
```
model.layers.N.input_layernorm.weight          ← layers.N.attn_norm.weight
model.layers.N.post_attention_layernorm.weight ← layers.N.ffn_norm.weight
model.layers.N.attn_hc.{fn,base,scale}         ← layers.N.hc_attn_{fn,base,scale}
model.layers.N.ffn_hc.{fn,base,scale}          ← layers.N.hc_ffn_{fn,base,scale}
model.layers.N.self_attn.q_a_proj.weight       ← layers.N.attn.wq_a.weight   (drop wq_a.scale)
model.layers.N.self_attn.q_b_proj.weight       ← layers.N.attn.wq_b.weight   (drop wq_b.scale)
model.layers.N.self_attn.kv_proj.weight        ← layers.N.attn.wkv.weight    (drop wkv.scale)
model.layers.N.self_attn.o_a_proj.weight       ← layers.N.attn.wo_a.weight   (drop wo_a.scale)
model.layers.N.self_attn.o_b_proj.weight       ← layers.N.attn.wo_b.weight   (drop wo_b.scale)
model.layers.N.self_attn.q_norm.weight         ← layers.N.attn.q_norm.weight
model.layers.N.self_attn.kv_norm.weight        ← layers.N.attn.kv_norm.weight
model.layers.N.self_attn.sinks                 ← layers.N.attn.attn_sink
model.layers.N.mlp.gate_weight                 ← layers.N.ffn.gate.weight
model.layers.N.mlp.e_score_correction_bias     ← layers.N.ffn.gate.bias      (moe layers 3..42; absent layers 0,1,2 → zeros init OK, see §1.5)
model.layers.N.mlp.shared_experts.gate_proj.weight ← layers.N.ffn.shared_experts.w1.weight  (BF16, drop .scale)
model.layers.N.mlp.shared_experts.up_proj.weight   ← layers.N.ffn.shared_experts.w3.weight  (w3=up)
model.layers.N.mlp.shared_experts.down_proj.weight ← layers.N.ffn.shared_experts.w2.weight  (w2=down)
model.layers.N.mlp.experts.w1_weight  ← STACK_M layers.N.ffn.experts.M.w1.weight  (256, I8/uint8)
model.layers.N.mlp.experts.w1_scale   ← STACK_M layers.N.ffn.experts.M.w1.scale   (256, BF16)
model.layers.N.mlp.experts.w2_weight  ← STACK_M layers.N.ffn.experts.M.w2.weight
model.layers.N.mlp.experts.w2_scale   ← STACK_M layers.N.ffn.experts.M.w2.scale
model.layers.N.mlp.experts.w3_weight  ← STACK_M layers.N.ffn.experts.M.w3.weight
model.layers.N.mlp.experts.w3_scale   ← STACK_M layers.N.ffn.experts.M.w3.scale
```

### §1.3 Hash layers only (N in 0,1,2 — `num_hash_layers=3`)
```
model.layers.N.mlp.tid2eid   ← layers.N.ffn.gate.tid2eid    [3 instances]
```
(`gate.bias` is ABSENT in ckpt for hash layers — ckpt has it only for 40 moe layers.
nn leaf `e_score_correction_bias` exists for ALL 43. See §1.5 routing decision.)

### §1.4 CSA layers (compressor + indexer; 21 layers: 2,4,…,40,42)
```
model.layers.N.self_attn.compressor.wkv.weight          ← layers.N.attn.compressor.wkv.weight        [2*head_dim,hidden]
model.layers.N.self_attn.compressor.wgate.weight        ← layers.N.attn.compressor.wgate.weight
model.layers.N.self_attn.compressor.ape                 ← layers.N.attn.compressor.ape   NO TRANSPOSE (rate,2*out)=(4,1024)
model.layers.N.self_attn.compressor.norm.weight         ← layers.N.attn.compressor.norm.weight
model.layers.N.self_attn.indexer.compressor.wkv.weight  ← layers.N.attn.indexer.compressor.wkv.weight
model.layers.N.self_attn.indexer.compressor.wgate.weight← layers.N.attn.indexer.compressor.wgate.weight
model.layers.N.self_attn.indexer.compressor.ape         ← layers.N.attn.indexer.compressor.ape  NO TRANSPOSE (4,256)
model.layers.N.self_attn.indexer.compressor.norm.weight ← layers.N.attn.indexer.compressor.norm.weight
model.layers.N.self_attn.indexer.weights_proj.weight    ← layers.N.attn.indexer.weights_proj.weight   [64,4096]
model.layers.N.self_attn.indexer.wq_b.weight            ← layers.N.attn.indexer.wq_b.weight   (drop indexer.wq_b.scale)
```

### §1.5 HCA layers (compressor, NO indexer; 20 layers: 3,5,…,41)
```
model.layers.N.self_attn.compressor.wkv.weight    ← layers.N.attn.compressor.wkv.weight    [head_dim,hidden] (NOT 2*)
model.layers.N.self_attn.compressor.wgate.weight  ← layers.N.attn.compressor.wgate.weight
model.layers.N.self_attn.compressor.ape           ← layers.N.attn.compressor.ape  NO TRANSPOSE (rate,out)=(128,512)
model.layers.N.self_attn.compressor.norm.weight   ← layers.N.attn.compressor.norm.weight
```
Sliding layers 0,1: core MLA only (no compressor / no indexer leaves).

### §1.6 nn leaves with NO direct ckpt source (init-default, NOT remapped)
- `model.layers.N.mlp.e_score_correction_bias` for hash layers 0,1,2 — ckpt has NO
  `ffn.gate.bias` there (only 40 moe layers). nn `__init__` zero-inits it. STRICT load
  (AC2) must still pass: confirm `model.load_weights(..., strict=True)` tolerates leaves
  left at init when not in the remap dict, OR remap MUST supply a zeros tensor for those
  3 leaves. **Coder decision (TDD): pick whichever keeps strict load GREEN; default =
  remap supplies zeros for the 3 hash `e_score_correction_bias` so the key set is a
  proper superset-equality.** This is the ONLY nn leaf without a 1:1 ckpt tensor; it is
  NOT a STOP (known arch — DeepSeek hash-routing has no learned bias) — flag noted.

---

## §2 ckpt → nn shape/dtype pins (header-probed, §5.3)

| ckpt | dtype | shape | nn target dtype | transform |
|---|---|---|---|---|
| `attn.wq_a/wq_b/wkv/wo_a/wo_b.weight` | BF16 | e.g. wq_a `[1024,4096]` | BF16 | rename; **DROP `.scale`** |
| `attn.indexer.wq_b.weight` | BF16 | `[8192,1024]` | BF16/F32* | rename; DROP `.scale` |
| `attn.compressor.ape` (CSA) | F32 | `[4,1024]` | F32 | rename; **NO transpose** |
| `attn.indexer.compressor.ape` | F32 | `[4,256]` | F32 | rename; **NO transpose** |
| `attn.compressor.ape` (HCA) | F32 | `[128,512]` | F32 | rename; NO transpose |
| `attn.attn_sink` | F32 | `[64]` | F32 | rename |
| `hc_attn_fn` | F32 | `[24,16384]` | F32 | rename |
| `ffn.experts.M.w1.weight` | I8(uint8) | `[2048,2048]` | uint8 | STACK 256 → `[256,2048,2048]` |
| `ffn.experts.M.w1.scale` | BF16 | `[2048,128]` | BF16 | STACK 256 → `[256,2048,128]` |
| `ffn.shared_experts.w1.weight` | BF16 | `[2048,4096]` | F32* | rename w1→gate_proj; DROP `.scale` |

\* nn `__init__` builds some leaves as F32 (e.g. `indexer.wq_b`, `shared_experts.*`,
ape via `_normal`). `mlx_lm.convert` + `Model.cast_predicate` handle dtype casts on
load; remap should preserve ckpt dtype and let load/cast resolve. Coder verifies
strict-load dtype acceptance in AC2 (TDD). Routed experts STAY FP4 (uint8 weight +
BF16 scale), `_dequantize_fp4` at forward (ADR 0024) — remap does NOT dequant.

---

## §3 ckpt→nn cross-check result (Q3/Q4 — clean)

- ckpt index: 69187 keys, 98 templates. Drop `mtp.*` (46 templates) → 52 non-mtp templates.
- nn tree: 1460 leaves, 44 templates.
- Every nn template has a §1 ckpt source. Un-consumed ckpt templates = ONLY the
  intended drops:
  - `mtp.*` — sanitize strip (§5.4).
  - `attn.{wq_a,wq_b,wkv,wo_a,wo_b}.scale`, `attn.indexer.wq_b.scale` — FP8 sidecar drop (§5.3).
  - `ffn.experts.M.w{1,2,3}.scale` — NOT dropped, FOLDED into stacked `w{1,2,3}_scale` (§1.2).
  - `ffn.shared_experts.w{1,2,3}.scale` — BF16 sidecar drop (§5.3).
- Only nn leaf w/o 1:1 ckpt tensor: hash `e_score_correction_bias` ×3 (§1.6, init/zeros).
- **No orphan ckpt tensor pointing at an unknown arch piece → Q3 STOP-rule does NOT fire.**

---

## §4 Acceptance Criteria (TDD red-first; Coder)

**AC1 — key-set exactness.** Run remap over the real ckpt index (header/template level,
no full materialization). Assert `set(remapped_keys) == set(nn model.parameters() keys)`
EXACTLY — zero missing, zero extra. (Template-level over all 43 layers + per-instance on
the sub-checkpoint of AC3.)

**AC2 — strict load.** `model.load_weights(list(remapped.items()), strict=True)` (after
`Model.sanitize` mtp strip) succeeds with no missing/unexpected key error on the AC3
sub-checkpoint. Dtype accepted (or cast-resolved) for every leaf.

**AC3 — one-of-each real layer load + finite forward.** Build the nn module, load a
sub-checkpoint = {global, sliding layer (0 or 1), one real HCA (3), one real CSA (2)},
run a tiny-seq forward through `_attention_mlx` dispatch; output is finite (no NaN/Inf),
shape `[1, seq, hidden]`. Exercises CSA + HCA + sliding real-dim paths AND FP4 expert
dequant (ADR 0024) on the loaded layers.

**AC4 — ape orientation.** Assert remapped `compressor.ape` / `indexer.compressor.ape` /
HCA `compressor.ape` shapes EQUAL the ckpt header shapes (NO transpose): `(4,1024)`,
`(4,256)`, `(128,512)`. (Guards against re-introducing §5.2's stale transpose.)

**AC5 — per-expert→stacked.** Assert stacked `experts.w1_weight` shape ==
`(n_routed_experts=256, moe_intermediate_size=2048, hidden_packed=2048)`, stack order
== ascending expert id M=0..255, uint8 preserved (no dequant). Same for w2/w3 + scales.

**AC6 — scale-drop.** Assert NO `.scale` key for attn-core (`q_a_proj`/…/`o_b_proj`),
`indexer.wq_b`, `shared_experts.*` survives into the remapped set (FP8 sidecars dropped,
§5.3). Routed-expert `*_scale` DO survive (FP4 needs them).

**AC7 — tiny byte-identity regression.** 13.3b-1/-2a/-2b/-3 + 13.3a nn (incl backward) +
13.2 FP4 + tiny CSA fixtures STAY GREEN; remap script is NEW + additive, touches NO
FROZEN/nn file → `routed.tolist() == direct.tolist()` style regression on the tiny
fixtures unchanged. No introduced RED.

---

## §5 STOP-ESCALATE rules (Coder enforces; write `ba-13-3b-4-stop.md` analog as `coder` stop)

- S-orphan: a ckpt tensor (non-mtp, non-scale-drop) maps to NO nn leaf, OR an nn leaf
  (other than the 3 hash `e_score_correction_bias`, §1.6) has NO ckpt source AND no
  init-default that keeps strict load GREEN → STOP, escalate Architect § clarification.
- S-shape: a remapped tensor's shape ≠ the nn leaf's shape after the §1/§2 transform
  (and it is NOT the known ape/stack case) → STOP (unknown layout).
- S-frozen: remap requires editing `deepseek_v4.py` or `deepseek_v4_nn.py` → STOP
  (convert-side-only contract; ADR 0025/0026).
- S-strict: strict load cannot be made GREEN without an nn-port `__init__`/`sanitize`
  change → STOP (that is nn-port territory, not convert).

NONE fired at BA recon. ape §5.2-staleness is a doc-fix flag, NOT a STOP.

---

## §6 Architect doc-fix flag (NON-blocking)

architecture.md **§5.2** instructs transposing CSA + indexer ape to `(2*out_dim, rate)`.
The WIRED real path (`_csa_compressor_real_mlx:819`, `_indexer_mlx:653`,
`_hca_compressor_mlx:773` — all 13.3b-2b) expects token-major `(rate, …)`, matching
ckpt-native + nn-leaf layout, pinned GREEN by `test_13_3b_3_nn_mixed_layers_forward.py`.
The transpose is a relic of the superseded FROZEN tiny helper `_csa_windowed_compressor_mlx:368`.
**Request: correct §5.2 to "ape: NO transpose (all three) — ckpt layout == nn leaf".**
This is informational; Coder already directed (§0.4, AC4) to follow the wired path.
