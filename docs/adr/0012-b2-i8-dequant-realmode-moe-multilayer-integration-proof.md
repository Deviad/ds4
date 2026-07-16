# ADR 0012: Multi-layer I8 block-scale expert dequant is integration-proven through real-mode MoE

Date: 2026-06-20
Status: Accepted

## Context

ADR 0011 added `B2-a-1`, a single-layer synthetic I8 block-scale expert-dequant integration proof through the public real-mode `Model.load_weights()` / `__call__` path. Its explicit deferred item was multi-layer I8 stacking (`num_hidden_layers>1`).

The multi-layer seam is reachable without a gate lift: `_validate_real_mode` permits `num_hidden_layers in {1,2,3}` when `hc_mult==1`, `_load_real_weights` already requires per-layer `layers.{i}.*.scale` keys for `expert_dtype="i8"`, and `_real_forward` loops over stacked layers, calling `_real_layer_forward` and `_moe_mlx` for each layer.

## Decision

Add one `real_mode_proofs` entry, `B2-a-2`, immediately after `B2-a-1`. It constructs a tiny two-layer real-mode model with `expert_dtype="i8"`, `hidden_size=16`, `moe_intermediate_size=16`, deterministic signed-int8 routed experts, and BF16 block scales. The proof drives public `load_weights()` -> `__call__` -> `_real_forward` -> stacked `_real_layer_forward` -> `_moe_mlx`, so `_dequantize_i8_block_scale_mlx` is selected on every layer.

The proof reuses `_i8_dequant_proof_status(...)` unchanged for the primary NL=2 gate:

1. production I8 output equals the same real-mode forward fed per-layer dequantized float experts through the raw branch at `REAL_MODE_FORWARD_TOLERANCE` (`1e-5`);
2. production I8 output matches `_integrated_multilayer_forward` fed the same per-layer dequantized float experts at the established I8 per-proof tolerance (`1e-3`);
3. quantization is non-degenerate (`quantization_gap > 1e-5`, non-unit scale);
4. deleting a required per-layer `.scale` key makes `load_weights()` fail.

The runner also records a secondary `num_hidden_layers=3` check under `secondary_multilayer` and fails closed if that check exceeds either tolerance. The recorded config remains NL=2.

## Consequences

`real_mode_proofs.proofs_total` moves from 6 to 7 when the readiness command runs under the MLX venv. The top-level readiness state remains fail-closed: `full_forward_parity=false`, `marker_earned=false`, `blockers_count=4`, and no `.deepseek-v4-forward-parity-ok` marker or `model-4bit` artifact is written.

B2 is not satisfied. Packed FP4 expert dequant, real checkpoint payload decode, expert parallel kernels, `hc_mult>1` multi-layer stacking (B1), and shimmed-checkpoint load/generation (B3) remain deferred. No production/vendor MoE, dequant, forward, gate, marker, conversion, training, quantization, or generation path changes are part of this decision.
