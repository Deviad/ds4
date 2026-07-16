# ADR 0015: B2 routed-dequant trusted-reference readiness recorded as fail-closed diagnostic

Date: 2026-06-20
Status: Accepted

## Context

Story 11.40 / ADR 0014 records the real-checkpoint expert payload classification
for B2: routed experts are genuine `I8` weights with `F8_E8M0` scales in a 1-D
`block_size=16 axis=1` layout; shared experts are `F8_E4M3` + `F8_E8M0` 2-D
128×128; FP4 is absent from the real expert families. That block also records the
missing trusted-reference requirement as a string, but it does not enumerate the
candidate trusted-reference landscape.

A read-only/header-only Story 11.41 scout established that no independent routed
I8+F8_E8M0 per-16 full-path dequant reference is locally attainable. The official
HF `inference/convert.py` assumes e2m1fn-FP4-per-32 packing and asserts
`scale.size(1) == in_dim // fp4_block_size` with `fp4_block_size=32`; for the real
routed `w1` scale this expects 64, but the checkpoint has 128, so the assertion
fails on this checkpoint. The official `inference/kernel.py` defines FP8(e4m3)
128-block and FP4(e2m1fn) 32-block paths only, not an I8+E8M0 block-16 op.
Transformers delegates expert FP8/int8 handling to backends that do not define the
routed I8+UE8M0 micro-block. Our torch/numpy reconstruction of
`int8 * decode_e8m0(scale)` and the FP8 shim's `f8_e8m0_to_bf16` path are circular
under ADR 0007 §4. A DS4-CPU harness is not yet built, and a freshly-written
re-expression of the same formula would also be circular.

## Decision

Add an additive `b2_routed_dequant_trusted_reference_readiness` block to the
`deepseek-v4-forward-parity-readiness` JSON report. The block is produced by a
pure fail-closed builder fed by a presence-only introspection probe. The probe may
record whether local candidate source files/symbols are present, but it never
decodes payload bytes, imports torch/triton/mlx, executes a candidate reference,
constructs a model, writes markers, converts, trains, quantizes, or generates.

The candidate verdicts are pinned static ADR/scout facts, not probe-derived proof:
`official_hf_inference_convert_py=assumes_other_packing`,
`official_hf_inference_kernel_py=no_i8_e8m0_block16_op`,
`transformers_deepseek_v4_modeling=delegates_no_i8_backend`,
`our_torch_numpy_int8_decode_e8m0=circular`,
`fp8_shim_f8_e8m0_to_bf16=circular`, and `ds4_cpu_harness=not_yet_built`. The block
machine-readably records the convert.py mismatch and the future independence test
a DS4-CPU harness or other reference must pass.

The diagnostic is fail-closed and no-false-positive: `status="fail-closed"`,
`decision="not-ready"`, `fail_closed=true`, `ready=false`,
`aggregate_decode_reference_available=false`, `decode_trusted_reference_available=false`,
`real_payload_decoded=false`, and `proof_available=false` are pinned regardless of
probe outcome. Recording the landscape is necessary but never sufficient; readiness
can change only after an independent trusted routed I8+F8_E8M0 full-path reference
passes the recorded independence test and a reviewed real-mode real-payload decode
proof is accepted as a `real_mode_proofs` entry.

The top-level readiness schema stays `1` because this is a backward-compatible
additive key. The block is written as ordinary readiness DATA by `_write_json_atomic`,
not by `_write_gate_marker`, and is not loadable as a gate marker.

## Consequences

The readiness report now prevents B2-a-1/B2-a-2/B2-a-3 synthetic I8 partial evidence
from being overclaimed as trusted-reference availability for real routed payload
decode. The official convert.py mismatch and the DS4-CPU-harness independence test
are durable machine-readable facts rather than only handoff narrative.

No proof counter changes: `real_mode_proofs.proofs_total` remains 8, with no new
proof entry. `full_forward_parity=false`, `marker_earned=false`, `blockers_count=4`,
`status="not-ready"`, and `coverage.fixtures_total=19` remain the expected state.
`.deepseek-v4-forward-parity-ok` and `model-4bit` stay absent; `convert-shimmed`
still hard-requires the absent marker; `dequantize_expert_packed("fp4",...)` and
the ungated discriminator form `dequantize_expert_packed("i8",...)` still raise
`NotImplementedError`.

No production/vendor/spec math changes, no gate lift, no new DS4-CPU harness, no
payload decode, and no candidate-reference execution are part of this decision.
