# ADR 0022 — Key-remap adapter + fp4 routed-expert packing contract + numpy<->gold single-dequant parity

- **Status:** SUPERSEDED 2026-06-22 (Head-of-AI due-diligence pivot)
- **Date:** 2026-06-25 (originally authored); SUPERSEDED 2026-06-22
- **Superseded by:** `docs/adr/0022-strategic-pivot-local-mlx-qlora-parity-marker-retired.md`
  (new ADR 0022 — strategic pivot to local MLX QLoRA Path A; forward-parity marker
  retired) + `docs/adr/0023-post-fuse-generation-coherence-cross-check.md` (new
  safety net replaces the gold-forward cross-check that this ADR's numpy MoE
  re-impl arc was scaffolding)
- **Reason for supersession:** 6 consecutive STOPs (11.15h, -r2, -r3, -r4, -r5,
  -r6) established that a non-circular gold forward reference is HARDWARE-
  INFEASIBLE on the M3 Ultra (512GB unified RAM < ~570GB BF16 expansion needed
  for HF Transformers 5.x). The numpy MoE re-impl arc this ADR was sanctioning
  is structurally circular under ADR 0007 §4, hence cannot serve as the
  parity gold reference it was scoped to become. The strategic pivot (new ADR
  0022) replaces the gold-forward gate with a post-fuse generation coherence
  cross-check (ADR 0023).
- **Status (preserved for provenance):** ACCEPTED 2026-06-25
- **Date (original):** 2026-06-25
- **Supersedes / amends:** none (lifts the r4 deferral of "ADR 0022 implementation cannot
  close"; ADR 0022 now closes per recon findings 2+3; CONSUMES ADR 0017's
  `dequantize_i8_block_scale` primitive UNCHANGED — does NOT amend ADR 0017)
- **Author:** Architect (Story 11.15h-r5 Option 2A slice)
- **Related:** ADR 0007 (expert block geometry, shape authoritative), ADR 0017 (Metal
  carry-forward + `dequantize_i8_block_scale` primitive), ADR 0021 (loader paired-tensor
  synthesis I8_E8M0)

## Context

Story 11.15h-r4 STOP+ESCALATED at STOP-rule (xv) on two blockers:
1. "Missing keys" — shimmed ckpt keys (`embed.weight`, `layers.N.*`, `head.weight`,
   `hc_head_*`) are unprefixed, while HF canonical `DeepseekV4ForCausalLM` expects
   `model.`-prefixed (`model.embed_tokens.weight`, `model.layers.N.*`, `lm_head.weight`,
   `model.hc_head_*`).
2. "SwiGLU dims do not close" — routed expert `w1.out=2048` ≠ `w2.in=1024`.

The r4 Architect deferred ADR 0022 ("key-remap adapter + fp4 contract") on the
rationale "implementation cannot close." Story 11.15h-r5 recon
(`agent-output/cmux-11-15h-r5/ckpt-provenance-recon.md`, 2026-06-19, read-only
safetensors header + `model.safetensors.index.json` analysis, 69,187-key vocab
enumeration) FALSIFIES both r4 blockers as measurement artifacts:

- **Blocker #2 FALSIFIED**: the dim mismatch is an fp4-PACKING ARTIFACT. Routed
  experts are I8 containers holding 2 fp4 nibbles/byte along the in-features dim
  (`config.expert_dtype="fp4"`). SwiGLU closes exactly:
  `hidden(4096) → w1,w3 → intermediate(2048) → silu(w1·x)⊙(w3·x) → w2 → hidden(4096)`,
  matching `hidden_size=4096` / `moe_intermediate_size=2048`. The "1024" =
  `2048 logical in-features ÷ 2 nibbles-per-byte = 1024 packed bytes`. Shared
  experts are unpacked BF16 (`w1=[2048,4096]`, `w2=[4096,2048]`) — already logical,
  confirming the 4096/2048 chain.
- **Blocker #1 FALSIFIED**: "missing keys" is a naming-convention mismatch, not
  missing data. Every key `_load_real_weights` reported missing has a 1:1
  counterpart in the shimmed ckpt under a different naming convention
  (`ffn.*↔mlp.*`, `attn_norm↔input_layernorm`, `attn.kv_norm↔kv_norm`,
  `attn.wkv↔kv_proj`, `hc_attn_*↔attn_hc.*`, `hc_ffn_*↔ffn_hc.*`). Systematic
  export-convention difference, fully resolvable by a deterministic key-string
  adapter. NO weight data missing.

The shimmed ckpt at `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/` IS genuine DeepSeek V4
(config self-declares `model_type=deepseek_v4`,
`architectures=["DeepseekV4ForCausalLM"]`, `transformers_version=4.57.1`; 69,187-key
vocab carries every V4-distinctive structure: DSA sparse-attention indexer on even
layers 2..42, `tid2eid` hash routing on layers 0..2 = `num_hash_layers=3`, mHC
Sinkhorn `hc_*` columns, full MTP head `mtp.0.*`, aux-free `noaux_tc` bias ×40,
sqrtsoftplus scoring, 256 routed + 1 shared MoE). `tid2eid` is the V4 hash-layer
token→expert routing table, NOT a V2 MTP leftover.

Since the r4 deferral rationale is falsified, ADR 0022 proceeds. Parent arbitration
(2026-06-25) LOCKED Decision 1 = Option 2A: build a deterministic key-remap adapter
on the existing shimmed ckpt. Option 1A (download HF canonical) unnecessary — recon
proves the shimmed ckpt IS canonical V4, only naming + fp4 layout differ.
Option 3 (re-shim) rejected — no re-quantization needed.

## Decision

Adopt **Option 2A**: a pure key-string translation adapter as the gold-load path
for the shimmed ckpt. The adapter sits ABOVE `DeepseekV4ForCausalLM.from_pretrained`,
reads the 46 on-disk safetensors shards into an in-memory `state_dict`,
applies the R1..R28 rename table to the KEYS ONLY (no tensor bytes, no dtype
recast, no layout reshape, no re-quantization), and passes the remapped dict to
`from_pretrained(..., state_dict=remapped)` via the HF-sanctioned `state_dict=`
kwarg. The on-disk ckpt + all 46 shards stay byte-identical; the adapter is a
load-time shim, NOT an on-disk transformation.

ADR 0022 sanctions THREE components:

### (a) Key-remap adapter — pure key-string translation at load time

**Module boundary**: pure function `remap_shimmed_ckpt_keys(raw_keys) -> dict`.
Lives under `python-envs/mlx/src/ds4_ft_mlx/` (NOT under `scripts/` — `scripts/*`
is FROZEN per AGENTS.md). Recommended filename
`shimmed_ckpt_key_remap.py` (grep-ability).

**Adapter invariants (NON-NEGOTIABLE)**:
- **Pure key-string translation**: touches key strings ONLY. No tensor bytes
  mutated, no dtype recast, no layout reshape, no re-quantization.
- **Deterministic**: same input keys → same output map, every call. No state,
  no RNG, no time-dependence.
- **Total**: returns a mapping for EVERY input key; no key dropped.
- **Identity-on-already-canonical**: a key that already matches the canonical
  vocabulary passes through unchanged (R28 universal prefix rule is idempotent:
  only prepends `model.` if absent).
- **Coverage-exhaustive**: `set(remapped_keys) ⊆ set(vendor_canonical_keys)`
  with residue `set(remapped_keys) − set(vendor_canonical_keys) = ∅` modulo
  documented safe defaults (`e_score_correction_bias`=zeros optional; `mtp.*`
  pass-through). Residue → STOP-rule (xviii).
- **Reversible documentation**: the inverse map is recorded in §"R1..R28 rename
  table" below so the numpy reference (`numpy_real_forward_reference.py`, FROZEN)
  and gold `forward_capture` (vendor `_real_forward`) consume IDENTICAL state
  dicts after remap — both dequantize through the SAME ADR 0017 primitive on the
  SAME logical shapes (component (c)).

**R1..R28 rename table** (Architect derives LIVE from ckpt 69,187-key vocab ×
HF canonical attribute tree; Coder validates exhaustively against the full vocab
programmatically; the recon's Evidence 3 sample is the seed set; R21-R25 are
extended deterministically to the full compressor/indexer subtree):

| # | Pattern (regex on raw ckpt key) | Replacement (vendor-canonical) | Scope |
|---|---|---|---|
| R1 | `^embed\.weight$` | `model.embed_tokens.weight` | Top-level embedding |
| R2 | `^head\.weight$` | `lm_head.weight` | LM head (tied via `_tied_weights_keys`) |
| R3 | `^norm\.weight$` | `model.norm.weight` | Final RMSNorm |
| R4 | `^hc_head_(base\|fn\|scale)$` | `model.hc_head.hc_\1` | HyperHead 3 keys |
| R5 | `^layers\.(\d+)\.attn_norm\.weight$` | `model.layers.\1.input_layernorm.weight` | Pre-attn norm |
| R6 | `^layers\.(\d+)\.ffn_norm\.weight$` | `model.layers.\1.post_attention_layernorm.weight` | Post-attn norm |
| R7 | `^layers\.(\d+)\.attn\.kv_norm\.weight$` | `model.layers.\1.self_attn.kv_norm.weight` | KV norm |
| R8 | `^layers\.(\d+)\.attn\.wkv\.(weight\|scale)$` | `model.layers.\1.self_attn.kv_proj.\2` | KV projection |
| R9 | `^layers\.(\d+)\.attn\.wq_(a\|b)\.(weight\|scale)$` | `model.layers.\1.self_attn.q_\2_proj.\3` | q_a/q_b projections |
| R10 | `^layers\.(\d+)\.attn\.wo_(a\|b)\.(weight\|scale)$` | `model.layers.\1.self_attn.o_\2_proj.\3` | o_a/o_b projections |
| R11 | `^layers\.(\d+)\.attn\.q_norm\.weight$` | `model.layers.\1.self_attn.q_a_norm.weight` | q_norm → q_a_norm |
| R12 | `^layers\.(\d+)\.attn\.attn_sink$` | `model.layers.\1.self_attn.sinks` | attn sink term |
| R13 | `^layers\.(\d+)\.hc_attn_(base\|fn\|scale)$` | `model.layers.\1.attn_hc.\2` | Hadamard attn column |
| R14 | `^layers\.(\d+)\.hc_ffn_(base\|fn\|scale)$` | `model.layers.\1.ffn_hc.\2` | Hadamard ffn column |
| R15 | `^layers\.(\d+)\.ffn\.shared_experts\.w1\.(weight\|scale)$` | `model.layers.\1.mlp.shared_experts.gate_proj.\2` | shared w1 → gate_proj |
| R16 | `^layers\.(\d+)\.ffn\.shared_experts\.w3\.(weight\|scale)$` | `model.layers.\1.mlp.shared_experts.up_proj.\2` | shared w3 → up_proj |
| R17 | `^layers\.(\d+)\.ffn\.shared_experts\.w2\.(weight\|scale)$` | `model.layers.\1.mlp.shared_experts.down_proj.\2` | shared w2 → down_proj |
| R18 | `^layers\.(\d+)\.ffn\.gate\.weight$` | `model.layers.\1.mlp.gate.weight` | router weight (hash/topk) |
| R19 | `^layers\.(\d+)\.ffn\.gate\.bias$` | `model.layers.\1.mlp.gate.bias` | aux-free `noaux_tc` bias (×40) |
| R20 | `^layers\.(\d+)\.ffn\.gate\.tid2eid$` | `model.layers.\1.mlp.gate.tid2eid` | hash router table (×3, layers 0..2) |
| R21 | `^layers\.(\d+)\.ffn\.experts\.(\d+)\.w1\.(weight\|scale)$` | `model.layers.\1.mlp.experts.\2.w1.\3` | per-expert w1 (vendor per-expert layout) |
| R22 | `^layers\.(\d+)\.ffn\.experts\.(\d+)\.w2\.(weight\|scale)$` | `model.layers.\1.mlp.experts.\2.w2.\3` | per-expert w2 |
| R23 | `^layers\.(\d+)\.ffn\.experts\.(\d+)\.w3\.(weight\|scale)$` | `model.layers.\1.mlp.experts.\2.w3.\3` | per-expert w3 |
| R24 | `^layers\.(\d+)\.attn\.compressor\.(.*)$` | `model.layers.\1.self_attn.compressor.\2` | HCA compressor subtree (kv_proj, gate_proj, position_bias, kv_norm.weight) — ALL 43 layers have compressor |
| R25 | `^layers\.(\d+)\.attn\.indexer\.(.*)$` | `model.layers.\1.self_attn.indexer.\2` | CSA indexer subtree (kv_proj, gate_proj, weights_proj, q_b_proj, compressor.*) — 21 even-indexed layers 2..42 |
| R26 | `^mtp\..*$` | (pass-through UNCHANGED) | MTP NextN; `_keys_to_ignore_on_load_unexpected=[r"(^\\|\\.)mtp\\..*"]` strips at load; forward() does not consume MTP |
| R27 | `^layers\.(\d+)\.attn\.(.*)$` | `model.layers.\1.self_attn.\2` | CATCH-ALL any `attn.*` not matched by R7-R12, R24, R25 → `self_attn` rename + prefix |
| R28 | (universal prefix rule) | if key does NOT start with `model.`, prepend `model.` | Final safety net; idempotent (no double-prefix) |

**mtp.* decision (Architect OVERRIDE of BA Q5 "ignore")**: BA verdict
"ignore mtp.*" is semantically correct but operationally ambiguous. HF
`_keys_to_ignore_on_load_unexpected=[r"(^|\.)mtp\..*"]` already strips any
`mtp.*` key from `unexpected_keys` at load time. The deterministic, total rule
is R26 PASS-THROUGH UNCHANGED (no `model.` prefix add); the HF regex strips
them silently. Zero state_dict divergence. If r6+ invokes `model.generate`
(speculative MTP) the rule extends to `mtp.0.*`→`model.mtp.0.*` — r6+ scope,
NOT r5.

**Layout caveat (recorded as known-open, NOT a blocker for r5 gold-load)**: HF
canonical `DeepseekV4Experts` (modeling L52935) uses concatenated
`gate_up_proj`/`down_proj` 3D `[num_experts, 2*inter, in]` layout; the shimmed
ckpt stores per-expert separate `w1`/`w2`/`w3` tensors. Reconciling the two is a
LAYOUT RESHAPE, NOT a key-string remap — explicitly OUT of r5 NARROW scope (no
tensor byte mutation). r5 routes around this by loading the ckpt into the
VENDOR `_real_forward` path (the mlx_lm port's per-expert w1/w2/w3 layout
consumer) which accepts R21-R23 per-expert keys directly. The HF-canonical-load
path (concat reshape) is r6+ multi-slice work, deferred.

### (b) fp4 routed-expert packing contract — exact logical shapes (LOCKED)

Recon Evidence 2 (ground truth; LIVE-measured `scale shape` and `weight shape`
from safetensors headers; logical shapes derived from `config.expert_dtype="fp4"`
+ `hidden_size=4096` + `moe_intermediate_size=2048`):

| Tensor | Stored dtype | Stored shape | Scale dtype | Scale shape | Logical shape | Packing |
|---|---|---|---|---|---|---|
| routed `experts.N.w1.weight` | I8 (fp4 container) | `[2048, 2048]` | BF16 | `[2048, 128]` | `[out=2048, in=4096]` | 2 fp4 nibbles/byte along in-dim (in_bytes=2048) |
| routed `experts.N.w3.weight` | I8 (fp4 container) | `[2048, 2048]` | BF16 | `[2048, 128]` | `[out=2048, in=4096]` | same as w1 |
| routed `experts.N.w2.weight` | I8 (fp4 container) | `[4096, 1024]` | BF16 | `[4096, 64]`  | `[out=4096, in=2048]` | 2 fp4 nibbles/byte along in-dim (in_bytes=1024) |
| shared `shared_experts.w1.weight` | BF16 | `[2048, 4096]` | (placeholder unused) | n/a | `[out=2048, in=4096]` (already logical) | unpacked BF16 |
| shared `shared_experts.w2.weight` | BF16 | `[4096, 2048]` | (none) | n/a | `[out=4096, in=2048]` (already logical) | unpacked BF16 |
| shared `shared_experts.w3.weight` | BF16 | `[2048, 4096]` | (placeholder unused) | n/a | `[out=2048, in=4096]` (already logical) | unpacked BF16 |

**Two orthogonal block-scale numbers (BOTH true; recorded to kill the
supervisor-alternate-vs-recon ambiguity)**:
- **Group count per out-row** (supervisor phrasing in requirements.md US-3-fp4
  "128 logical / 64 packed"): w1/w3 have 128 scales per out-row
  (`scale shape [2048, 128]`); w2 has 64 scales per out-row
  (`scale shape [4096, 64]`).
- **Per-scale block size** (recon phrasing "32 logical / 16 packed"):
  per-scale block = `in_bytes / scales_per_out_row` = `2048/128 = 16` bytes for
  w1/w3 (→ 32 logical fp4 features per scale), `1024/64 = 16` bytes for w2
  (→ 32 logical fp4 features per scale).

Both numbers describe DIFFERENT axes. ADR 0022 records both explicitly so Coder
and the numpy reference cannot disagree on which is authoritative. The
NON-NEGOTIABLE invariants: (i) I8 container with 2 fp4 nibbles/byte along the
in-dim; (ii) SwiGLU closes to `[out=2048, in=4096]`→`[out=4096, in=2048]`;
(iii) both numpy + gold dequant through the SAME ADR 0017 single-dequant
primitive on the SAME logical shapes.

**SwiGLU closure (LOCKED, supersedes r4 STOP (xv))**:
`hidden(4096) → w1(in=4096)·x → [2048]` ∥ `w3(in=4096)·x → [2048]`;
`silu(w1·x) ⊙ (w3·x) → [2048]`; `w2(in=2048)·[2048] → [4096] = hidden`. Closes.
The "1024" in raw I8 shape `[4096, 1024]` = `2048 logical ÷ 2 nibbles/byte =
1024 packed bytes`. r4 blocker #2 falsified.

### (c) numpy-side single-dequant per ADR 0017 — numpy<->gold bit-identical parity

The numpy reference dequantizes routed experts via the SAME ADR 0017
`dequantize_i8_block_scale` primitive the vendor gold path uses, on the SAME
post-remap logical shapes; ONE dequant call per expert tensor (no
double-dequant); FP4 unpacking stays in the dequant primitive, NOT in the
adapter.

**Parity path (Architect spec, AC11 acceptance gates)**:

```
Gold path (vendor _real_forward):              Numpy path (numpy_real_forward_reference.py):
  adapter-remapped state_dict                     adapter-remapped state_dict (SAME dict)
        │                                                │
        ▼                                                ▼
  vendor reads experts.N.w1.{weight,scale}      numpy reads experts.N.w1.{weight,scale}  (SAME keys post-remap)
        │                                                │
        ▼                                                ▼
  _dequantize_i8_block_scale_mlx               dequantize_i8_block_scale
  (MLX-side proven primitive, ADR 0017)        (numpy-side proven primitive, ADR 0017)
        │                                                │
        ▼                                                ▼
  weight I8 [out=2048, in_bytes=2048]           weight I8 [out=2048, in_bytes=2048]
  → unpack 2 nibbles/byte axis=1               → unpack 2 nibbles/byte axis=1
  → reshape [out=2048, in=4096] fp4            → reshape [out=2048, in=4096] fp4
  → cast fp4→BF16 via scale lookup             → cast fp4→BF16 via scale lookup
        │                                                │
        ▼                                                ▼
  scale BF16 [out=2048, 128] (w1/w3)            scale BF16 [out=2048, 128] (w1/w3)
  broadcast over 32 logical features per      broadcast over 32 logical features per
  scale (block_size=32 logical, 16 packed)     scale (block_size=32 logical, 16 packed)
        │                                                │
        ▼                                                ▼
  dequantized BF16 [out=2048, in=4096]          dequantized BF16 [out=2048, in=4096]
  ==== BIT-IDENTICAL (single dequant primitive, same inputs) ====
```

**AC11 acceptance gates (exact logical shapes, BOTH sides)**:

```python
assert dequant_w1.dtype == torch.bfloat16
assert tuple(dequant_w1.shape) == (2048, 4096)            # out=2048, in=4096
assert tuple(dequant_w3.shape) == (2048, 4096)            # out=2048, in=4096
assert tuple(dequant_w2.shape) == (4096, 2048)            # out=4096, in=2048
# SwiGLU closure sanity (one expert, one random token):
x = torch.randn(4096, dtype=torch.bfloat16)
silu_w1x = torch.nn.functional.silu(dequant_w1 @ x) * (dequant_w3 @ x)
out = dequant_w2 @ silu_w1x
assert tuple(out.shape) == (4096,)                          # back to hidden
assert torch.isfinite(out).all()
# ONE dequant call per expert tensor (no double-dequant):
#   w1 dequanted ONCE, w3 dequanted ONCE, w2 dequanted ONCE.
#   Adapter does NOT dequant; dequant primitive does NOT remap.
```

## Consequences

**Positive**:
- Unblocks r5 NARROW gold-load (Dec 2(c)): adapter loads shimmed ckpt into vendor
  `_real_forward` WITHOUT downloading HF canonical (Option 1A) or re-shimming
  (Option 3). No re-quantization.
- Locks the fp4 packing contract NOW so r6+ numpy reference bit-matches gold
  `forward_capture` from day one — eliminates the r4 dimension-closure ambiguity
  at the source.
- ADR 0017 stays UNCHANGED — ADR 0022 CONSUMES its primitive, does NOT amend it.
- On-disk ckpt + all 46 safetensors shards stay byte-identical (adapter is
  load-time, in-memory, key-strings only).
- Vendor `deepseek_v4.py`, `numpy_real_forward_reference.py`,
  `real_forward_intermediate_dump.py`, `deepseek_v4_dequant.py`, `metal/*.metal`,
  `ds4.*`, `scripts/*` ALL stay byte-identical (AC6).

**Negative / known-open**:
- HF canonical `DeepseekV4ForCausalLM.from_pretrained` direct load is NOT
  achievable in r5 with adapter alone — the per-expert w1/w2/w3 → concatenated
  `gate_up_proj`/`down_proj` reshape is a LAYOUT change, not a key-remap, and is
  explicitly OUT of r5 NARROW scope (no tensor byte mutation). r5 routes around
  this via the VENDOR `_real_forward` path (per-expert w1/w2/w3 consumer).
  r6+ multi-slice epic owns the concat-reshape path if direct HF-canonical load
  is needed.
- If a LIVE measurement shows the shimmed ckpt fp4 dequant contract diverges
  from ADR 0017's primitive (e.g. the fused `_dequantize_i8_block_scale_mlx`
  expects a different block layout than the shimmed ckpt ships), STOP-rule (xx)
  fires; ADR 0017 amendment is deferred to r6 BA slice. ADR 0022 records the
  divergence as a known-open item; it does NOT silently amend ADR 0017.

## Invariants (NON-NEGOTIABLE; Reviewer enforces)

1. **Adapter is pure key-string translation.** No tensor bytes mutated, no dtype
   recast, no layout reshape, no re-quantization, no state, no RNG. Deterministic
   + total + identity-on-already-canonical. Coverage-exhaustive
   (`set(remapped) ⊆ set(canonical)`, residue `∅` modulo `e_score_correction_bias`
   zeros + `mtp.*` pass-through).
2. **On-disk ckpt byte-identical.** All 46 `model-NNNNN-of-00046.safetensors`
   shards + `model.safetensors.index.json` NEVER touched. ONLY `config.json`
   `mlp_layer_types` field patched (`["mo"]*43` → `["hash_moe"]*3 + ["moe"]*40`)
   per r5 AC2; NO OTHER config field mutated; NO safetensors re-touched.
3. **FROZEN byte-identical files**: `numpy_real_forward_reference.py`,
   `real_forward_intermediate_dump.py`, `deepseek_v4_dequant.py`, vendor
   `mlx_lm_models/deepseek_v4.py` (HEAD `221bdac`), `metal/*.metal`, `ds4.c`,
   `ds4.h`, `ds4_metal.m`, `ds4_cli.c`, `ds4_server.c`, `scripts/*` (existing).
4. **ADRs 0001-0021 UNCHANGED.** ADR 0022 is ADDED this slice; it CONSUMES
   ADR 0017 unchanged, does NOT amend it.
5. **Vendor `transformers/models/deepseek_v4/*` (HF lib install)**: NOT mutated
   — vendored read-only; adapter feeds `from_pretrained` via the `state_dict=`
   kwarg, no HF source patches.
6. **Markers unchanged**: `.deepseek-v4-forward-parity-ok` STAYS ABSENT (11.15j
   owns parity; r5 proves gold loadable only); `model-4bit` + `convert.shimmed`
   STAY ABSENT; Track-A `.ds4-gguf-generate-ok` PRESENT; `ds4flash.gguf`
   byte-identical (never mutate).
7. **Single-dequant parity**: BOTH numpy (r6+) AND gold (`forward_capture`)
   dequant routed experts via ADR 0017 `dequantize_i8_block_scale` on the SAME
   post-remap logical shapes; ONE dequant call per expert tensor; adapter does
   NOT dequant; dequant primitive does NOT remap. Divergence → STOP-rule (xx).
8. **fp4 packing contract LOCKED**: I8 container, 2 fp4 nibbles/byte along
   in-dim; SwiGLU closure `hidden(4096)→[2048]→[4096]`; block-scale group count
   128/out-row (w1/w3) and 64/out-row (w2); per-scale block 32 logical fp4
   features / 16 packed bytes.

## STOP-rule carry-forward + NEW (xviii)/(xix)/(xx) (Option 2A reframed)

Carry-forward: (i)/(iii)/(iv)/(vi)/(vii)/(viii)/(ix)/(x)/(xi)/(xii)/(xiii)/(xiv)/(xv-r).

- **(xv-r)**: "SwitchGLU architecture inference ambiguous" FALSE-POSITIVE per
  recon finding 2; re-tiered to "(xv-r) FP4 block_size closure must verify on
  shimmed ckpt LIVE before treating closure as resolved" (§b closure math LOCKED;
  closure CLOSES once fp4 unpacked).
- **(xviii) NEW**: Key-remap adapter FAILS — residual unmapped keys after R1-R28
  (Q1 exhaustiveness fails with a key lacking canonical counterpart); OR vendor
  `_real_forward` strict-key STILL fails after adapter; OR `num_hash_layers`
  absent/`!=3`; OR hash_moe layers 0..2 lacking `tid2eid` buffer (3 keys expect
  3) → STOP+ESCALATE. (Core Dec 1 Option 2A hypothesis falsified LIVE.)
- **(xix) NEW**: Vendor gold `_real_forward` on adapter-loaded ckpt CRASHES —
  dtype mismatch, MPS device fallback, fp4 dequant path missing (Q2), mHC
  Sinkhorn numerical instability, hybrid attention HCA/CSA indexer crash,
  per-expert w1/w2/w3 layout mismatch against vendor module expectation,
  `ffn.*→mlp.*` consumer contract divergence → STOP+ESCALATE. (Core Dec 2(c)
  hypothesis falsified LIVE.)
- **(xx) NEW**: Shimmed ckpt fp4 dequant contract violates ADR 0017
  primitive-tier provenance — IF `dequantize_i8_block_scale` does NOT consume
  adapter-routed fp4 weights cleanly (un-reconcilable block_size/layout mismatch;
  OR numpy ↔ gold diverge on logical shapes / dequant output) → STOP+ESCALATE.
  (LIVE-falsification of AC11 contract. ADR 0017 amendment deferred r6.)

LIVE-falsification criterion: if Coder TDD red run hits ANY of (xviii)/(xix)/(xx)
symptoms, emit `{"status":"ok","role":"Coder","stop":true,"reason":"<clause>"}`
+ `agent-output/cmux-11-15h-r5/coder-stop.md`; do NOT attempt fix in-slice.
