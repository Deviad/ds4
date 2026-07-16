# ADR 0017: B2 derivation boundary X carries forward to the Metal MSL production path (corollary to ADR 0016)

Date: 2026-06-20
Status: Accepted
Corollary to: ADR 0016 ("B2 DS4-CPU-harness independence adjudicated as
independent-possible (with derivation boundary X)")

## Context

ADR 0016 adjudicated a fresh C implementation in `ds4.c`'s **CPU reference
backend** of routed `I8 + F8_E8M0` 1-D `block_size=16 axis=1` dequant as
**independent-possible (with derivation boundary X)** — reasoning that ADR 0007
§4's anti-circular rule reaches *code-transliteration of our imperative Python*,
not the OCP Microscaling Formats (MX) specification's math.

Story 11.43 is the **first code slice on Track-B's unlock path**. The user has
explicitly chosen the **Metal production path** ("Let's proceed then with the
metal path. I was not aware of this!"), which AGENTS.md also designates as the
production path ("the SSD streaming, CUDA, distributed inference, Metal default
inference" production paths; CPU backend CPU-only and reference/debug-only).

The BA (Story 11.43 `requirements.md` §1) resolved the load-bearing scope
question as **Option (a): carry the ADR 0016 verdict forward to the Metal MSL
production path under the SAME derivation boundary X** — a deliberate,
non-circular extension (the derivation boundary is math-level, not
language-level). This ADR records that carry-forward durably, plus the
supervisor-confirmed Q1/Q2 findings that 11.43's design rests on, so the
Reviewer has a discrete Metal-surface artifact to anchor on (not a silent scope
expansion of 0016).

## Decision

**Adjudicate the carry-forward as ACCEPTED: boundary X (ADR 0016 §"Derivation
boundary X", allowed surface S1/S2/S3 + 9 forbidden OUR-Python symbols) extends
verbatim to a Metal MSL kernel in `metal/moe.metal`. The 11.43 Metal routed-I8
dequant kernel qualifies as the FIRST independently-derived (ADR-0007-§4-
satisfying, non-circular) reference precisely because its derivation draws exactly
and only from the carried-forward boundary X.**

The reasoning is the BA's §1 reasoning (recorded, not re-litigated):

1. The three boundary-X authorities are all math/authoring-language-agnostic:
   - **S2 (math authority)** = OCP MX v1.0 spec E8M0 decode `scale = 2^(e−127)`
     (a closed-form external-standard definition; MSL `exp2f((float)e - 127.0f)`
     and C `powf(2.0f, (float)e - 127.0f)` are derived from the same spec).
   - **S3 (consumer convention)** = real safetensors header geometry
     (`I8 [2048,2048]` + `F8_E8M0 [2048,128]` ⟹ `axis=1, block_size=16`), bytes.
   - **S1 (integration convention)** = decoded routed weights feed the MoE grouped
     matmul — identical convention between the CPU `ds4.c` site and the
     `metal/moe.metal` site; only the dispatch site differs.
2. ADR 0016's decisive argument — "the prohibition reaches code-transliteration
   of our imperative Python, not the OCP MX spec's math" — transfers verbatim
   because it is about *what is prohibited*, not *what language the independent
   implementation is written in*. A Metal-MSL kernel derived from the OCP spec
   and never consulting our Python is no more a "transliteration of our Python"
   than a C function derived from the same spec — actually less, because MSL and
   Python share no surface syntax.
3. The 9 forbidden symbols (listed below) are off-limits regardless of target
   language — the prohibition is a derivation-source rule, equally enforceable in
   MSL and C.
4. Re-adjudicating (Option (b), fresh ADR) would re-run identical reasoning
   against identical inputs, producing the identical verdict — busywork/slop under
   AGENTS.md anti-slop.
5. MLX's Metal `quantized.h` affine idiom is **secondary engineering
   corroboration only** (structural pattern: threadgroup tile + scale broadcast +
   simdgroup MMA) — the Reviewer confirms the Coder derives the decode MATH from
   the OCP spec, NOT from MLX's `mxfp4`/`mxfp8` kernel (which would make us
   MLX-dependent).

## The 9 forbidden OUR-Python symbols (off-limits for the Metal kernel's math; unchanged from ADR 0016)

- `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_dequant.py`:
  `decode_f8_e8m0_scales` (L72), `dequantize_f8_e4m3fn_with_e8m0_scales` (L113),
  `_apply_i8_block_scales` (L291), `dequantize_i8_block_scale` (L322),
  `dequantize_expert_packed` (L1262).
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`:
  `_dequantize_i8_block_scale_mlx` (L135), `_moe_mlx` (L650),
  `_SUPPORTED_EXPERT_DTYPES` (L52).
- `scripts/shim_ds4_safetensors.py`: `f8_e8m0_to_bf16` (L183).
- Any torch/numpy reconstruction of `int8 * decode_e8m0(scale)`.

## Q1 finding (supervisor-confirmed, recorded in ADR for durability)

DeepSeek-V4 does **NOT** use the MXINT8 implicit `2^(−6)` fixed-point factor.
Supervisor empirical run: `payload[0]=1` (signed int8 `+1`), `scale_byte=120`,
`shape=(16,)`, `block_size=16`, `scale_axis=0` → decoded `0.0078125 = 1 *
2^(120−127) = 2^(−7)`, NOT `2^(−13)`. DeepSeek-V4's "I8 + F8_E8M0" is a simpler
microscaling variant (signed I8 × E8M0 shared scale, no fixed-point factor), not
strict OCP MXINT8.

→ The 11.43 Metal dequant formula is `result = signed_int8_element * 2^(e − 127)`,
with `e = 0` → `2^(−127)` (subnormal scale, NOT zero), `e = 255` → NaN for the
whole 16-element block, NO infinity encoding — all per OCP MX v1.0 §scale. The
OCP default `k = 32` is NOT used (`block_size = 16` from S3 header geometry).

## Q2 reframe (supervisor-confirmed, recorded in ADR for durability)

The incumbent Python E8M0 i8 dequant path (Story 11.22 — the `B2-a-1/2/3`
synthetic integration proofs + the `topk-moe-i8-block-scale` fixture in source)
is **ALREADY OPEN in source** but **CIRCULAR per ADR 0007 §4** (its docstring
claims "proven via closed-form + shim-pipeline parity" — our own formula + our
own shim pipeline, not an independent reference). The 11.42 supervisor invariant
was incomplete (it tested only the no-metadata path that raises; the
full-metadata path `dequantize_expert_packed("i8", payload, scales=<E8M0 bytes>,
shape=(N,M), block_size=16, scale_axis=1)` does NOT raise — it returns decoded
floats).

**Implications (binding for 11.43):**

1. The 11.43 Metal kernel is **NOT "lifting the `dequantize_expert_packed` gate"**
   — the Python full-metadata path is already open. The Coder builds the FIRST
   independently-derived (OCP-spec, non-circular) dequant, implemented as a Metal
   kernel.
2. The **independence gap is the REAL gate.** The Metal kernel, IF derived from
   the OCP spec per boundary X, IS the first §4-satisfying reference.
3. The **red test compares the Metal kernel against an OCP-spec-derived witness,
   NOT against the Python path.** Comparing Metal-vs-Python would be circular
   (both compute the same formula; matching proves nothing about independence).
   The witness is the load-bearing independent reference.
4. The math agreement (any spec-faithful implementation computes
   `int8 * 2^(e-127)` — because that IS the format's definition) is a sanity
   floor; the **convention agreement** (axis=1, block_size=16, broadcast,
   orientation, special cases) derived from the spec + header geometry is the
   load-bearing independent confirmation that resolves §4's consumer-convention
   risk. This matches ADR 0016 §2's AES/FIPS-197 analogy (a spec-derived
   implementation is independent of derivation source, not math identity).

## PASS-time state-change target (Test Manager owns; 11.43 itself edits NO JSON)

On a passing 11.43 red test (Reviewer PASS + Test Manager real-mode isolation
proof at `≤1e-5`, Metal kernel output vs OCP-spec witness output on the real HF
DeepSeek-V4 Flash checkpoint `553034d…` bytes), the Test Manager applies to the
`b2_routed_dequant_trusted_reference_readiness` readiness block (regenerated by
re-running the checker under `python-envs/mlx/.venv/bin/python`):

1. **Flip `candidate_reference_landscape[5].independence_undetermined`**
   `true → false` (the `ds4_cpu_harness` candidate — independence no longer
   undetermined; adjudicated by a real independently-derived reference on the
   Metal surface).
2. **Update `candidate_reference_landscape[5].verdict`**
   `not_yet_built → built_adjudicated_independent`.
3. **Add `metalpath_routed_dequant_independence_adjudicated: true`** (top-level
   field in the b2 block), recording the Metal MSL surface earned the
   adjudication (distinct from the CPU harness the 11.42 verdict originally
   named).
4. **Accept `real_mode_proofs` entry `B2-a-4`** ("Metal routed I8+F8_E8M0 dequant
   isolation proof (OCP-spec-witness, non-circular)") as the 9th spec in
   `B0_REAL_MODE_PROOF_SPECS` — a DIFFERENT evidence class than the circular
   `B2-a-1/2/3` (the 11.41 `future_proof_criteria` explicitly anticipated:
   *"accept that proof as a real_mode_proofs entry before changing any
   b2_routed_dequant_trusted_reference_readiness verdict field"*). Regenerated
   JSON `real_mode_proofs.proofs_total = 9`.
5. **Reconcile `dequantize_expert_packed_i8_status`** to record the Q2 nuance
   (full-metadata non-raising/circular; no-metadata guard still raising). RECORD
   reconciliation only — NOT a gate lift.

## What STAYS invariant on PASS (B2 is NOT fully satisfied; only the routed-I8 independence sub-gate resolves)

- `full_forward_parity = false`, `marker_earned = false`, `blockers_count = 4`,
  `status = "not-ready"`, `fail_closed = True`, `ready = False`, `schema = 1`.
  The FULL B2 blocker (shared `F8_E4M3 + F8_E8M0` 2-D 128×128 sub-blocker,
  FP4-absent, `convert-shimmed`, real-payload full-forward parity, B1, B3)
  remains unproven.
- `.deepseek-v4-forward-parity-ok` and `model-4bit` stay **ABSENT**;
  `convert-shimmed` still hard-requires the absent marker.
- NO conversion/training/generation/quantization; NO payload decode beyond the
  proof itself (≥1 routed expert's `gate/up/down`); NO full-model forward.
- NO edit to `dequantize_expert_packed`, the readiness JSON by the Coder (the
  Test Manager applies the PASS-time reconciliation); NO edit to `metal/*.metal`
  beyond the new I8+E8M0 kernel section (no perturbation of the existing Q4_K /
  Q2_K / IQ2_XXS / SSD-streaming / distributed / Metal-default inference paths).

## Why a corollary ADR and not an amendment to ADR 0016

A short standalone ADR (vs appending a section to 0016) keeps 0016's
"CPU-harness candidate" verdict intact as the decision that was carried forward,
gives the Reviewer a discrete Metal-surface artifact to anchor on (one decision =
one ADR), and makes the Q1 finding (no `2^(−6)`) + the Q2 reframe durable under
versioning in their own ADR (so a future slice cannot silently collapse the
"Metal-vs-Python is circular" point). The BA chose Option (a); this ADR is that
choice's durable record, not a re-adjudication.

## Consequences

- **Boundary X is now surface-crossing.** A Metal-MSL routed-I8 dequant kernel
  derived within boundary X is adjudicated independent under ADR 0007 §4. The
  Coder (surface:39) implements `metal/moe.metal`'s new I8+E8M0 section under
  the §6 Coder hard constraints; the Reviewer audits boundary-X compliance; the
  Test Manager runs the real-payload isolation proof.
- **The 11.42 `independent-possible` verdict becomes durable** on a passing 11.43
  red test (a real Metal kernel exists, not just a decision). The field flip +
  the new `B2-a-4` entry + the new `metalpath_routed_dequant_independence_adjudicated`
  field are the PASS-time record the Test Manager owns.
- **No gate, marker, counter, or readiness field is lifted by this ADR** — the
  carry-forward is a scoping decision; the independence is EARNED only on a
  passing 11.43 red test + Reviewer + Test Manager PASS.
- **Reviewer-anchored:** the xhigh-reviewer (`openai-codex/gpt-5.5`, fresh
  context, `system-prompt: replace`) audits this ADR on (1) the carry-forward
  being non-circular + reasoned (§Decision), (2) boundary-X precision (no allowed
  element smuggles in our Python formula), (3) the Q1/Q2 findings being
  supervisor-confirmed (not re-derived from forbidden symbols), (4) invariant
  preservation. A Reviewer disagreement that the boundary is the same is the
  load-bearing case: the parent + a fresh Architect re-adjudicate the Metal
  surface specifically before the Coder is dispatched.
