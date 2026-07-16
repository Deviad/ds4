# ADR 0007: Expert block geometry is shape-authoritative; declared quantization_config is advisory

Date: 2026-06-18
Status: Accepted

## Context

The real DeepSeek V4 Flash checkpoint
(`models--deepseek-ai--DeepSeek-V4-Flash`, snapshot `553034d…`) carries a single
`quantization_config`:

```json
{"activation_scheme":"dynamic","fmt":"e4m3","quant_method":"fp8",
 "scale_fmt":"ue8m0","weight_block_size":[128,128]}
```

But the two expert families are stored in **different** packings, verified
header-only from the real shards (single-tensor reads, no full-model load):

| Family | weight dtype | scale dtype | example weight shape | example scale shape | shape ratio (per axis) | block geometry |
|---|---|---|---|---|---|---|
| Routed (`layers.N.ffn.experts.E.w{1,2,3}`) | `I8` | `F8_E8M0` | `[2048, 2048]` | `[2048, 128]` | `(1, 16)` | **1-D, block_size 16 on axis 1** |
| Shared (`layers.N.ffn.shared_experts.w{1,2,3}`) | `F8_E4M3` | `F8_E8M0` | `[2048, 4096]` | `[16, 32]` | `(128, 128)` | **2-D, 128×128 sub-block** |

Two durable facts follow:

1. The declared `weight_block_size [128,128]` matches the **shared** F8_E4M3
   family exactly (scale ratio 128 on both axes) and does **not** describe the
   **routed** I8 family (whose scale shape forces block_size 16 on a single
   axis; `[128,128]` would require a `[16,16]` scale, which the shard does not
   contain). The declared string therefore governs the `fmt=e4m3` fp8 path, not
   the I8 routed micro-block scale.
2. The weight↔scale **tensor shapes** uniquely determine block geometry. For the
   routed family there is exactly one shape-consistent mapping
   (`scale[i, j//16]`); no other axis/block assignment fits the bytes.

Earlier docs (the dossier and Story 11.15 plan wording) described routed experts
as "FP4 packed". The checkpoint contains **no FP4** in the expert families; that
wording is fictional and is corrected by this decision and the dossier update in
the same slice (cmux-11-15c).

## Decision

1. **Block geometry is derived from observed weight/scale tensor shapes and is
   authoritative.** `resolve_routed_block_layout(report)` returns the
   shape-inferred routed geometry (`{axis: 1, block_size: 16}` for the real
   checkpoint) whenever the observed scale shape yields a single unambiguous
   block layout. The declared `quantization_config.weight_block_size` is recorded
   as **advisory** and explained, never used to override or veto the observed
   shape.
2. **Fail closed only when shapes are genuinely ambiguous.** If the observed
   weight/scale shapes do not determine a single block_size/axis (rank mismatch,
   more than one exactly-dividing axis with no tiebreak, or no paired shapes),
   geometry resolution raises `ExpertBlockLayoutError` — it never guesses a
   default.
3. **Shared experts use a genuine 2-D sub-block decode.** The shared
   `F8_E4M3 + F8_E8M0` 128×128 layout is decoded by an N-D block-scale primitive
   whose per-axis block sizes come from the shape ratio. The existing per-axis
   broadcast helper (`dequantize_f8_e4m3fn_with_e8m0_scales`,
   `scale_dim ∈ {1, value_dim}`) is preserved byte-identical for its narrower
   contract.
4. **Routed dispatch stays fail-closed until an independent full-path reference
   exists.** Per spec §10.9, `dequantize_expert_packed("i8")` remains
   `NotImplementedError` until a *trusted, independent* full-path reference
   (e.g. the official DeepseekV4 routed-expert dequant, or a DS4-CPU harness)
   confirms scale-application order/orientation for the I8+F8_E8M0 routed layout.
   A torch/numpy reconstruction of our own `int8 * decode_e8m0(scale)` formula is
   **circular** and does not satisfy this gate. Transformers' shipped dequant ops
   cover FP8-e4m3 (`Fp8Dequantize`) and MXFP4 (`Mxfp4Dequantize`) only — neither
   defines the I8+E8M0 routed dequant — so as of this ADR no independent routed
   reference is locally attainable and routed dispatch remains deferred.
   Geometry being shape-unambiguous removes the *geometry* risk but not the
   *consumer-convention* risk that this gate protects.

## Consequences

- Easier: 11.15d (256-expert / top-6 routing) can rely on `resolve_routed_block_layout`
  returning real geometry instead of raising; the shared 2-D decode is no longer
  fail-closed.
- Harder / explicitly out of scope: routed `dequantize_expert_packed("i8")` stays
  raising until an independent routed full-path reference lands; this is a
  deliberate spec-§10.9 gate, not an omission.
- The `.deepseek-v4-forward-parity-ok` marker remains absent; reconciliation and
  per-tensor decode parity are necessary but not sufficient for it.
- The dossier's "FP4 routed expert" language is corrected to the verified
  packing (`I8` routed 1-D bs16 axis1; `F8_E4M3` shared 2-D 128×128) in the same
  slice, per ADR 0005.
