# ADR 0016: B2 DS4-CPU-harness independence adjudicated as independent-possible (with derivation boundary)

Date: 2026-06-20
Status: Accepted

## Context

Story 11.41 / ADR 0015 landed the fail-closed `b2_routed_dequant_trusted_reference_readiness`
landscape block for the routed `I8 + F8_E8M0` 1-D `block_size=16 axis=1` dequant.
That block pinned, as the one remaining *undetermined* future-criterion:
`ds4_cpu_harness.independence_undetermined=true`, with the explicit future test:

> a DS4-CPU harness (or other reference) is independent only if it is
> authored/derived from a source NOT sharing our `int8 * decode_e8m0(scale)`
> formula; a freshly-written re-expression of our formula does NOT pass.

ADR 0007 §4 names **exactly two** candidate independent references for the routed
I8 dequant gate — (a) the official DeepseekV4 routed-expert dequant op, and
(b) a DS4-CPU harness. The 11.41 scout load-bearingly disproved candidate (a)
for this checkpoint: the official `inference/convert.py` assumes
e2m1fn-FP4-per-32 packing and asserts `scale.size(1) == in_dim // fp4_block_size`
with `fp4_block_size=32`, which fails on the real per-16 `[2048,128]` scale
(expects 64, observed 128); `inference/kernel.py` defines no genuine
I8+E8M0 block-16 op; Transformers 5.12.1 `DeepseekV4Experts` does plain `F.linear`
and delegates FP8/int8 to backends that do not define the routed I8+UE8M0
micro-block. Our torch/numpy reconstruction of `int8 * decode_e8m0(scale)` and
the FP8 shim's `f8_e8m0_to_bf16` (which IS our `decode_e8m0` — the shim body
reconstructs `2^(e-127)` as BF16 bits: `e=0→0x0040`, `e=255→0x7FC0`, else
`b<<7`) are circular under ADR 0007 §4.

Candidate (b) — the DS4-CPU harness — is therefore the **last remaining candidate
independent reference** for B2 real-payload dequant. ADR 0015 deliberately left
its independence `undetermined` (not `false`) precisely so Story 11.42 would
resolve the question via a deliberate, reviewed adjudication rather than silently
abandon the track the operator has pursued across 11.40–11.41.

The load-bearing distinction to adjudiate: the E8M0 microscaling format defines
the **math** (an 8-bit unsigned exponent decoded to a scale value; one shared
E8M0 scale per 16 int8 weights on axis 1; `out = int8_weight × scale`). Our Python
path (`decode_f8_e8m0_scales` / `_apply_i8_block_scales` /
`dequantize_i8_block_scale` / `_dequantize_i8_block_scale_mlx` / shim
`f8_e8m0_to_bf16`) is **one imperative implementation** of that math. ADR 0007 §4
/ 11.41 AC6 prohibit re-expressing *our formula*. The open question is whether
that prohibition reaches **the math** (making every correct implementation
necessarily a re-expression of the same closed-form scale decode + block-scale
application → verdict "not-independent / circular") OR reaches only
**code-transliteration** (making a spec-derived C implementation that never
consults our Python independent → verdict "independent-possible (with derivation
boundary X)").

## Decision

**Adjudicate the DS4-CPU harness independence as independent-possible, with derivation
boundary X durably recorded below.** ADR 0007 §4's anti-circular rule reaches
**code-transliteration of our imperative Python**, not the OCP Microscaling
Formats (MX) specification's math. The decisive argument: §4 explicitly names the
DS4-CPU harness as a *candidate independent reference* — that can only be coherent
if the prohibition does not reach the format's math (otherwise the candidate §4
names could never be independent and §4 would contradict itself). ADRs do not
contradict themselves. The format's math (`int8 * 2^(e-127)` for block-16 axis-1)
is the **standard's** definition, published by the OCP MX spec; an implementation
derived from the OCP spec + the checkpoint header shape + the architecture dossier
is independently **sourced** even though the resulting math is necessarily
identical to our Python (because the format defines the math). Independence is a
property of the **derivation source**, not the math identity — exactly as a
standards-derived AES implementation (FIPS-197) is not "a re-expression of some
existing AES implementation" despite identical math.

### Derivation boundary X (hard-gated; a future Coder slice must satisfy exactly this)

A fresh C implementation in `ds4.c` (the CPU reference backend, per AGENTS.md
"keep the CPU backend CPU-only and use it only as reference/debug code") of routed
`I8 + F8_E8M0` 1-D `block_size=16 axis=1` dequant qualifies as independent iff
its derivation draws **exactly and only** from:

- **S2 — E8M0 closed-form decode math (derivation authority):** the OCP
  Microscaling Formats (MX) Specification (v1.0+), which defines E8M0 as an
  8-bit unsigned exponent (bias 127, no mantissa) with `scale = 2^(e - 127)`
  and documented special-case handling for `e = 0` and `e = 255`. The Coder MUST
  cite the spec (section + version) in a code comment beside the implementation.
  **Honest gap recorded:** the BA-proposed S2 candidate
  `agent-output/research-deepseek-mla-dsa.md` does NOT contain the E8M0
  closed-form decode citation — it is about MLA/DSA *attention*, not microscaling
  formats (verified by full read); the DeepSeek MLA/DSA papers are likewise about
  attention. These candidates are NOT S2 derivation surfaces; the OCP MX spec is
  the genuine S2 derivation surface. The `e=0`/`e=255` special-case handling (which
  our Python sets to `2^-127`/`NaN`) is a derivation detail the Coder must pin
  against the spec, NOT against our Python — this is exactly the
  consumer-convention detail §4 protects.
- **S3 — real checkpoint header facts (consumer convention: axis, block_size,
  broadcast direction):** per ADR 0007 §1, header-only from the real
  `models--deepseek-ai--DeepSeek-V4-Flash` snapshot (`553034d…`, original HF F8
  checkpoint, NOT `hf-f8shim`): routed weight `I8` `[2048,2048]` + scale
  `F8_E8M0` `[2048,128]` ⟹ `axis=1, block_size=16`, shape-unambiguous; shared
  `F8_E4M3 + F8_E8M0` 2-D 128×128 (separate sub-blocker); `fp4_absent=true`;
  declared `weight_block_size=[128,128]` advisory for the `fmt=e4m3` shared path.
  The Coder MUST derive `(axis=1, block_size=16)` from the safetensors header
  JSON or the header-only `resolve_routed_block_layout(report)` helper (NOT from
  our Python `_apply_i8_block_scales` / `dequantize_i8_block_scale` /
  `_dequantize_i8_block_scale_mlx`).
- **S1 — architecture dossier + ADRs (integration / output convention):**
  `docs/architecture.md`, `docs/deepseek-v4-architecture-dossier.md`,
  `docs/deepseek-v4-mtp-policy.md`, `docs/adr/` — the dequant output flows into
  the ds4.c CPU reference backend's MoE `_combine`/`_moe_mlx` path, with the
  output dtype/layout matching the weight tensor (row-major, for the routers'
  matmul-with-input). The dossier confirms the real ds4.c frontend's routed
  I8+F8_E8M0 path is distinct from the Python MLX shim path (which consumes
  BF16-shimmed scales), so a fresh CPU-harness dequant is a genuinely parallel
  path.

**and the Coder NEVER consults the forbidden derivation surface (a C
implementation transliterating or reconstructing any of these is circular and
fails the independence test):**

- `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_dequant.py`: `decode_f8_e8m0_scales`
  (L72), `dequantize_f8_e4m3fn_with_e8m0_scales` (L113), `_apply_i8_block_scales`
  (L291), `dequantize_i8_block_scale` (L322), `dequantize_expert_packed` (L1262).
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`:
  `_dequantize_i8_block_scale_mlx` (L135), `_moe_mlx` (L650),
  `_SUPPORTED_EXPERT_DTYPES` (L52).
- `scripts/shim_ds4_safetensors.py`: `f8_e8m0_to_bf16` (L183).
- Any torch/numpy reconstruction of `int8 * decode_e8m0(scale)` from our Python.

### Why "independent-possible" and not "independent-built"

The verdict is **possible** with a hard-gated boundary, not "already built or
already proven". The DS4-CPU harness is not yet built. A downstream Coder slice
earns independence only by (i) demonstrating boundary compliance at build time
(Reviewer-anchored: OCP spec cited in-code, geometry header-derived, the nine
forbidden symbols unconsulted) and (ii) a reviewed real-mode real-payload decode
proof against a closed-form spec-derived gold at `≤1e-5` isolation / `≤1e-3`
independent reference (Test-Manager-anchored). This ADR records the
**adjudication** that such an independently-sourced C implementation is not
circular under ADR 0007 §4; it does not itself lift any gate, write any marker,
bump any counter, decode any payload, or build the harness.

### ADR 0015 negative-branch worry bounded

ADR 0015 recorded "a freshly-written re-expression of the same formula would
also be circular" as part of the landscape disproof. This ADR **bounds** that
worry: it applies only to a re-expression derived from our Python
(transliteration / port). It does not extend to a re-derivation whose source is
the OCP MX spec, because the derivation source is then independent. The
boundary above operationalizes this: a harness built within the boundary is
independently sourced; a harness that copies our Python (even "freshly") is
circular. This preserves ADR 0007 §4's consumer-convention-risk protection
while resolving the `undetermined` field ADR 0015 deliberately left open.

## Consequences

- **Durably resolves** the `ds4_cpu_harness.independence_undetermined=true`
  field pinned by ADR 0015: the DS4-CPU-harness candidate is adjudicated
  **independent-possible (with derivation boundary X)**. A future harness+proof
  slice that satisfies the boundary may be scoped; one that consults the
  forbidden symbols or fails to cite the OCP MX spec remains circular and is
  rejected.
- **Does NOT lift any gate, marker, counter, or readiness field in 11.42.**
  11.42 is a decision slice. The `independence_undetermined` field is flipped
  to `independence_adjudicated=true` only by a *later* harness+proof slice —
  naturally the same slice that lands the independent reference and (on Reviewer
  + Test Manager PASS) bumps `proofs_total` 8→9 and lifts the routed-I8
  `dequantize_expert_packed("i8")` `NotImplementedError`. 11.42 adds no
  readiness block and edits no readiness JSON.
- **Invariants invariant:** `real_mode_proofs.proofs_total=8` (no new entry),
  `full_forward_parity=false`, `marker_earned=false`, `blockers_count=4`,
  `status="not-ready"`, `coverage.fixtures_total=19`, readiness schema stays
  `1`. `.deepseek-v4-forward-parity-ok` and `model-4bit` stay absent;
  `convert-shimmed` still hard-requires the absent marker; `dequantize_expert_packed("i8"|"fp4")`
  still raise `NotImplementedError`; `_validate_real_mode` byte-identical.
- **No production/vendor/spec math edit, no harness code written, no payload bytes
  decoded, no Metal/SSD/distributed inference path perturbed.** The downstream
  harness is added to the ds4.c **CPU reference backend only** (AGENTS.md), not
  the Metal production graph.
- **Reviewer-anchored:** the xhigh-reviewer (`openai-codex/gpt-5.5`, fresh
  context, `system-prompt: replace`) checks the adjudication on (1) reasoned
  coherence against ADR 0007 §4 verbatim, (2) boundary precision / non-circularity
  (no allowed-surface element smuggles in our Python formula; the S2 attainment
  gap honestly recorded), (3) invariant preservation. A Reviewer disagreement on
  the verdict itself (e.g. "the prohibition reaches the math, verdict should be
  negative") is the load-bearing case: the parent + a fresh Architect
  re-adjudicate before any downstream Coder slice is scoped.
- **Out of scope for this ADR / 11.42:** the downstream harness+proof slice's
  exact proof mechanism (closed-form hand-eval vs. second-derivation C cross-check)
  and the shared `F8_E4M3 + F8_E8M0` 2-D 128×128 sub-blocker (separate, also-open
  trusted-reference gap). Both are downstream / out-of-scope; this ADR records
  only the routed-I8 independence adjudication and its boundary.
