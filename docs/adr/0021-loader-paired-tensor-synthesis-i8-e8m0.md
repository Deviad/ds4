# ADR 0021 — Loader paired-tensor synthesis for routed I8_E8M0 experts

- **Status:** ACCEPTED
- **Date:** 2026-06-22
- **Supersedes / amends:** none (lifts Story 11.48 §0 STOP4(c) prohibition on loader
  paired-tensor synthesis for routed I8_E8M0 experts; complements ADR 0007)
- **Author:** Architect (Story 11.49 slice)

## Context

The DS4 routed-expert family (DeepSeek-V4 Flash checkpoint) stores each expert's
weights as a **paired-tensor** layout, verified header-only against the real
checkpoint shards (no full 76 GB load) and codified in ADR 0007:

| Family | weight dtype | weight shape | scale dtype | scale shape | block geometry |
|---|---|---|---|---|---|
| Routed (`layers.N.ffn.experts.E.w{1,2,3}`) | `I8` | `[out, in]` | `F8_E8M0` | `[out, in/16]` | 1-D, `block_size=16` on axis=1, row-major |

The `metal/moe.metal` production kernel `ds4_e8m0_decode_i8` (11.43-shipped,
FROZEN) and the batched-matmul host binding in `ds4_metal.m` L24913-25816 both
implement exactly this `/16` row-major contract:
`scale = scales[row * (in/16) + col/16]`, weight `int8 * 2^(e-127)`.

Two facts make this contract non-trivial to surface from a GGUF:

1. **`DS4_TENSOR_I8_E8M0 = 64` is outside the ggml_type range.** `ds4.c`
   `gguf_types[]` (L1557-1591) runs `[0..30]` (`[30]={"bf16",1,2}`); there is no
   `f8_e8m0` / `i8_e8m0` entry, and `tensor_type(64)` (L1694) returns `NULL`.
   The reader therefore cannot decode a type-64 tensor directly.
2. **The routed-expert weight loads as raw `i8` (gguf_type 24) and the scale as a
   separate sibling tensor named `<weight>.scale`.** The sibling is plain bytes;
   the loader already parses both as byte blobs. The semantics — this is a paired
   I8+F8_E8M0 expert — are invisible to the reader until a synthesis step links
   them and flips the weight to the virtual type 64.

Story 11.47 shipped the Metal decode kernels + the batched-matmul host dispatch
exercised only by a synthetic smoke that sets type=64 directly (no real GGUF
flow). Story 11.48 Round-2 stopped at the enum mirror + two getter relaxations
(`ds4.c` L1597, `ds4_metal.m` L20016/L25914) and explicitly **deferred** the
loader-side synthesis to Story 11.49 pending this ADR. Until a loader synthesis
emits type=64 from a real (or synthetic-fixture) GGUF, no dispatch path is
exercised against real I8_E8M0 bytes — that is the gap this ADR closes on the
loader side.

### Correction to the BA handoff's block-geometry citation

The Story 11.49 BA requirements (`agent-output/cmux-11-49/requirements.md` §1
carry-forward, §2 D1 item 5, AC4) cite `gguf-tools/deepseek4-quantize.c` L683
`dequant_fp8_weight` as the routed-I8_E8M0 contract, with `block_out=128,
block_in=128, scale_rows=out/128, scale_cols=in/128`. **This is a mis-citation.**
`deepseek4-quantize.c` L683 is the **shared-FP8 family** contract
(`w->dtype=="F8_E4M3"`, `scale->dtype=="F8_E8M0"`, 2-D 128×128 sub-block),
which ADR 0007 attributes to the **shared** experts, not the routed ones. L711
`dequant_fp4_weight` is the FP4-packed path (`in_dim=packed_in*2`, `n_blocks=in/32`),
which ADR 0007 explicitly marks fictional for the real checkpoint ("the checkpoint
contains **no FP4** in the expert families").

The **authoritative** routed-I8_E8M0 geometry is ADR 0007's shape-derived
(`{axis=1, block_size=16}`) contract, confirmed byte-exact by the Metal kernel
`ds4_e8m0_decode_i8` and the `ds4_metal.m` batch path's
`gate_scale_cols = expert_in_dim/16u` (L24946). ADR 0021 codifies **that**
contract (weight `I8 [out,in]` + scale `F8_E8M0 [out,in/16]`, 1-D block_size 16
on axis 1, row-major). The BA's L683/128×128 framing is **not** used in the
FROZEN byte-spec for Story 11.49; `routed_expert_block_bytes(64)` and
`routed_expert_row_bytes(64)` (below) are derived from ADR 0007 + the kernel, not
from L683.

## Decision

### D1 — Post-load name-pattern synthesis pass in `ds4.c`

Add a **post-load** synthesis pass that runs after the routed-expert weight
tensors are bound into `ds4_layer_weights` (and the mtp weights) and **before**
`weights_validate_layout` (because `tensor_expect_routed_expert`, `ds4.c` L3416,
hard-enforces `tensor_is_routed_expert_type(t->type)` and would die on the raw
`i8` type 24). The pass:

1. For each routed-expert weight tensor (`ffn_gate_exps`, `ffn_up_exps`,
   `ffn_down_exps`, plus the mtp block): if the weight's loaded type is
   `DS4_TENSOR_I8` (=24, `"i8"`), derive the sibling scale tensor name by
   replacing the trailing `.weight` suffix on the weight's name with `.scale`
   (mirrors `deepseek4-quantize.c` L1185), and look it up via the existing
   `model_find_tensor(m, name)`.
2. If the sibling `.scale` tensor **exists**: mutate the weight tensor's
   `type` field `24 → 64` (`DS4_TENSOR_I8_E8M0`). Do **not** touch
   `weight->bytes`, `weight->abs_offset`, or any sibling field — both are
   already correct from the GGUF load (the weight bytes are `in*out*n_expert`
   raw int8; the sibling scale tensor keeps its own `abs_offset`/`bytes`,
   available to any dispatcher via `model_find_tensor`).
3. If the weight is type 24 but **no** `.scale` sibling exists:
   `ds4_die("routed I8 expert %.*s missing sibling .scale tensor (ADR 0021)")`
   — fail-closed on malformed GGUFs.
4. If the weight is **not** type 24 (existing Q2_K / Q4_K / IQ2_XXS routed
   GGUFs): no-op. **Existing GGUFs are byte-untouched**; no regression.

### Q3 resolution — scale binding site

**Option (iii)+ — no struct mutation.** The `ds4_tensor` struct (`ds4.c` L1607)
already carries `abs_offset` and `bytes`. The sibling `.scale` tensor is a
first-class member of `m->tensors`; the synthesis pass merely confirms its
existence and flips the weight type. Any downstream consumer (the Story 11.50
single-token encoder, the Story 11.47 batched path, an eventual SSD-streaming
scale reader) resolves the sibling at dispatch time via
`model_find_tensor(m, "<weight_stem>.scale")` and reads its `abs_offset`/`bytes`
directly. **No field is added to `ds4_tensor` or `ds4_layer_weights`; `ds4.h`
is not edited.** This is strictly less invasive than the BA's options (i)/(ii)
and even the BA's recommended (iii) lookup-table variant — there is no new
accessor, no new map, only the existing tensor array + existing
`model_find_tensor`.

### Helper expansion (AC4)

In `ds4.c`:
- `tensor_is_routed_expert_type` (L3227): add `|| type == DS4_TENSOR_I8_E8M0`.
- `routed_expert_block_bytes` (L3233 switch): add
  `case DS4_TENSOR_I8_E8M0: return 1;` (the routed weight is raw int8, 1 byte per
  element; the scale lives in a separate sibling, so the weight "block" is a
  single element — `block_elems=1, block_bytes=1`).
- `routed_expert_row_bytes` (L3243): add an early-return
  `if (t->type == DS4_TENSOR_I8_E8M0) return t->dim[0];` **before** the existing
  QK_K die. Rationale: the weight is unblocked int8, so one contiguous in-dim
  span (dim[0]) is `dim[0]` bytes; the `/16` scale grouping applies to the
  separate sibling tensor, not to the weight stride. QK_K (256) alignment does
  not apply to type 64 (the alignment constraint is `dim[0] % 16 == 0`, enforced
  on the scale sibling's shape, not here).

These return values feed `tensor_expert_bytes` (L5530, CPU dispatch — dies on
real I8_E8M0 per AGENTS.md reference-only policy) and the SSD-streaming /
Metal-host stride computations at `ds4.c` L11989/L15558/L18797/L19716/L19768/
L19841/L25510, all of which compute the weight-per-expert stride. They are
**internally consistent** with the `/16` row-major layout and the batched
Metal path's own inline `expert_in_dim/16u` derivation (L24946).

## Alternatives considered

- **Packing I8 + F8_E8M0 into a single virtual ggml_type with a C block struct**
  (analogous to `block_q4_K`). **Rejected:** would require a packed on-disk
  layout that does not match how `deepseek4-quantize.c` writes the real GGUF
  (weight + scale as **distinct** tensors), and would force a reader refactor
  to decode the packed type. The synthesis approach keeps both tensors as the
  loader already parses them and only links them semantically.
- **Extending `gguf_types[]` with `f8_e8m0`/`i8_e8m0` entries** so the reader
  decodes type 64 natively. **Rejected:** type 64 is deliberately outside ggml_type
  range (it is a DS4-specific paired dispatch type, not a self-contained ggml
  block format); the synthesis pass is the correct layering. (Story 11.48 §0 Q2
  ruling confirmed no reader type-table extension.)
- **Arg-binding lookup table (BA Q3 option (iii) original).** **Superseded by
  (iii)+:** the existing `ds4_tensor` array + `model_find_tensor` already
  provide name→tensor lookup; a separate `tensor_id → scale_buf` map would
  duplicate that indirection. No new map is added.
- **New field on `ds4_tensor` (Q3 option (i)) or `ds4_layer_weights` (option
  (ii)).** **Rejected:** requires a `ds4.h` edit (option (i)) or a sibling
  pointer on `ds4_layer_weights`; both are more invasive than necessary and
  violate the "keep public APIs narrow" rule. The synthesis link is owned by the
  loader; dispatch resolves the sibling on demand.
- **Deferring loader synthesis and shipping ADR 0021 contract-only in 11.49.**
  **Rejected:** an ADR without the loader implementation is paper; the contract
  is validated by the synthesizing loader + synthetic tiny-GGUF TDD fixture
  (Story 11.49 Q5(b)), which proves the loader emits type 64 + binds the sibling
  before any dispatch wiring (Story 11.50) is layered on.
- **Monolithic 11.49 (loader + new fused single-token encoder + selector +
  D2.2/D2.3/D2.4 relax + scale-wrap + D3.2/D3.3 + numeric correctness).**
  **Rejected sub-slicing (Q1 = CONFIRM 3-slice split):** Story 11.48 Coder
  Round-1 proved that bundling dispatch-side I8_E8M0 wiring with other changes
  risks silent-numerics cascades. 11.49 ships loader + this ADR + synthetic TDD
  only; Story 11.50 owns the single-token fused encoder + selector + relax +
  scale-wrap + dispatch asserts; Story 11.51 owns numeric correctness vs the
  11.47 batched-matmul reference.

## Consequences

- **+** The loader now emits `DS4_TENSOR_I8_E8M0 (=64)` for routed-expert weight
  tensors when a `.scale` sibling is present, unblocking (on real or synthetic
  paired-tensor GGUFs) the 11.47-shipped batched-matmul path and the 11.50
  single-token fused encoder (when that lands).
- **+** Additive + fail-closed: existing Q2_K / Q4_K / IQ2_XXS routed-expert
  GGUFs (weight type != 24) are byte-untouched; malformed raw-I8 GGUFs missing
  the `.scale` sibling die with a clear error.
- **+** No struct mutation (`ds4_tensor`, `ds4_layer_weights`, `ds4.h`
  unchanged); the sibling scale tensor stays a first-class tensor resolved by
  name at dispatch time.
- **+** ADR 0007's shape-derived `/16` geometry is the single source of truth;
  the BA's L683/128×128 mis-citation is corrected in the FROZEN byte-spec.
- **− (known utility limitation, supervisor-endorsed)**: as of this slice, no
  `deepseek4-quantize.c` recipe emits a raw routed I8 + sibling `.scale` GGUF —
  the quantizer dequants the HF I8+scale pair to F32 and re-quants into a single
  block-quant ggml_type (Q2_K / Q4_K / IQ2_XXS / etc.), and the `ds4q_type` enum
  has no raw-I8_E8M0 output. Therefore the loader synthesis is **forward-looking
  on real GGUFs**: it fires only on GGUFs produced by a future raw-I8_E8M0
  recipe (separate tooling slice) and on the synthetic tiny-GGUF TDD fixture
  shipped in this slice. On every GGUF the current tooling produces (incl.
  `ds4flash.gguf`) the synthesis is a safe no-op. ADR 0021 codifies the loader
  contract so the future raw-I8_E8M0 producer + the Story 11.50/11.51 dispatch +
  numeric-correctness work have a stable surface to target. This is the
  supervisor-endorsed framing (Story 11.49 brief §0: "ADR 0021 codifies the
  contract; it is no longer STOP4(c) 'synthesizing nothing'"). It is **not** a
  silent-numerics cascade (the loader no-ops safely; no dispatch runs in 11.49).
- **− Story 11.50 dependency:** the single-token fused encoder + selector +
  scale-wrap + relax still required for single-token decode; the loader alone
  does not make single-token I8_E8M0 dispatch reachable (matches the 11.47/11.48
  precedent: batched matmul reachable via 11.47 + this slice; single-token fused
  decode stays fail-closed until 11.50).
- **− SSD-streaming I8_E8M0 scale wiring is not exercised in 11.49.** The stride
  helpers are additive/correct for SSD cache sizing (weight-per-expert stride =
  `out_dim * dim[0]`), but the SSD-streaming routed-expert load path's scale-byte
  reading is a separate concern belonging to Story 11.50 (or a dedicated
  SSD-streaming slice) once a real producer exists. Flagged, not blocking: no
  real SSD I8_E8M0 dispatch runs in 11.49.

## Cross-references

- ADR 0007 — Expert block geometry is shape-authoritative (routed I8
  `[out,in]` + F8_E8M0 `[out,in/16]`, `{axis=1, block_size=16}`) — **the
  authoritative geometry source for this ADR**.
- ADR 0001 — Metal graph is the production path (loader synthesis targets the
  Metal-default mmap path; CPU path dies on real I8_E8M0 per AGENTS.md
  reference-only policy).
- ADR 0002 — Parity-first fail-closed gates (synthesis dies on malformed
  raw-I8-without-scale; existing GGUFs unaffected).
- ADR 0020 — Story 12.3 close-out (no scope overlap; readiness invariants
  frozen).
- `metal/moe.metal` L4494-4648 — `ds4_e8m0_decode_i8` + the I8_E8M0 kernels
  (FROZEN, 11.43 Reviewer PASS) — the decode-math authority.
- `ds4_metal.m` L24913-25816 — 11.47-shipped batched-matmul I8_E8M0 dispatch
  (scale offsets derived positionally `weight_offset + weight_bytes`; the
  synthesis makes `gate_type==I8_E8M0` reachable there).
- `agent-output/cmux-11-49/architecture.md` — Story 11.49 FROZEN byte-spec +
  Q1-Q5 resolution.
- `agent-output/cmux-11-48/architecture.md` §0 + §6.2 — the deferral this slice
  discharges.
