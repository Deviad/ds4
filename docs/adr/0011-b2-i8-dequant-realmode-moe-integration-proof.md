# ADR 0011: B2-a I8 block-scale expert dequant is integration-proven through real-mode MoE

Date: 2026-06-20
Status: Accepted

## Context

B2 is the forward-parity blocker for full MoE parity with packed FP4/I8 expert
dequant and expert kernels. Before Story 11.37, the isolated I8 MoE primitive
(`topk-moe-i8-block-scale`) was proven, but the production real-mode path had not
exercised `_moe_mlx`'s `expert_dtype="i8"` branch through public
`Model.load_weights()` and `Model.__call__`. All existing real-mode proofs used
`expert_dtype="fp4"`, which takes the raw expert-weight branch.

A bounded real-mode seam is available without a gate lift: `_validate_real_mode`
already accepts `i8`, `_load_real_weights` already requires per-expert `.scale`
keys for I8 experts, and the default I8 block-scale dequant path is legal for a
tiny `hidden_size=16` / `moe_intermediate_size=16` config.

## Decision

Add one `real_mode_proofs` entry, `B2-a-1`, that constructs a single-layer
real-mode model with `expert_dtype="i8"`, deterministic signed-int8 routed
experts, and BF16 block scales. The proof drives the public
`load_weights()` -> `__call__` -> `_real_forward` -> `_real_layer_forward` ->
`_moe_mlx` path and gates on all of:

1. `max_abs_error <= REAL_MODE_FORWARD_TOLERANCE` (`1e-5`) for production I8
   output vs the same production real-mode forward fed numerically-identical
   dequantized float experts through the raw branch.
2. `reference_max_abs_error <= 1e-3` vs the trusted pure-Python
   `_integrated_layer_forward` fed the same dequantized float experts.
3. Non-degenerate quantization (`quantization_gap > 1e-5` and non-unit scale).
4. I8 branch evidence: deleting a required `.scale` key makes `load_weights()`
   fail, confirming the I8 loader/dequant branch is selected.

The proof is recorded as partial B2 evidence only. The report-wide tolerance is
not weakened; the `1e-3` reference tolerance is per-proof and matches the existing
I8 block-scale MoE primitive tolerance.

## Consequences

`real_mode_proofs.proofs_total` moves from 5 to 6 when the readiness command runs
under the MLX venv. The top-level readiness state remains fail-closed:
`full_forward_parity=false`, `marker_earned=false`, `blockers_count=4`, and no
`.deepseek-v4-forward-parity-ok` marker or `model-4bit` artifact is written.

B2 is not satisfied. Packed FP4 expert dequant, real checkpoint payload decode,
expert parallel kernels, multi-layer I8 stacking, B1, and B3 remain deferred.
No production/vendor MoE, dequant, forward, gate, marker, conversion, training,
or generation path changes are part of this decision.
