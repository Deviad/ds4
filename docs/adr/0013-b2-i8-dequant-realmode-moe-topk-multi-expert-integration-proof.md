# ADR 0013: Top-k>1 multi-expert I8 dequant is integration-proven through real-mode MoE

Date: 2026-06-20
Status: Accepted

## Context

ADR 0011 added `B2-a-1`, a single-layer synthetic I8 block-scale expert-dequant integration proof. ADR 0012 added `B2-a-2`, the multi-layer counterpart. Both routed exactly one expert per token (`num_experts_per_tok=1`), leaving the production `_moe_mlx` multi-expert combine path — proper-subset `argsort`, multi-term `denom`, per-expert `factor=scores/denom`, and masked unselected experts — unproven with I8-dequantized experts.

The seam is reachable without a gate lift: `_validate_real_mode` permits `n_routed_experts<=4` and `num_experts_per_tok<=n_routed_experts`, `_load_real_weights` already requires I8 `.scale` keys for every routed expert, and `_integrated_layer_forward` already delegates to the top-k-aware pure-Python MoE reference.

## Decision

Add one `real_mode_proofs` entry, `B2-a-3`, immediately after `B2-a-2`. It constructs a tiny one-layer real-mode model with `expert_dtype="i8"`, `hidden_size=16`, `moe_intermediate_size=16`, `n_routed_experts=4`, `num_experts_per_tok=2`, deterministic signed-int8 routed experts, and BF16 block scales. The proof drives public `load_weights()` -> `__call__` -> `_real_forward` -> `_real_layer_forward` -> `_moe_mlx`, so `_dequantize_i8_block_scale_mlx` is selected while the production multi-expert routing combine is load-bearing.

The proof reuses `_i8_dequant_proof_status(...)` unchanged for the primary gates:

1. production I8 output equals the same real-mode forward fed dequantized float experts through the raw branch at `REAL_MODE_FORWARD_TOLERANCE` (`1e-5`);
2. production I8 output matches `_integrated_layer_forward` fed the same dequantized float experts at the established I8 per-proof tolerance (`1e-3`);
3. quantization is non-degenerate (`quantization_gap > 1e-5`, non-unit scale);
4. deleting a required routed-expert `.scale` key makes `load_weights()` fail.

The runner adds thin B2-a-3 guards without changing the status helper: `multi_expert_combine_exercised=true`, `tie_free_margin>=1e-4`, `routing_subset_stable=true`, and a secondary `num_experts_per_tok=3` check under `secondary_topk` that fails closed if either tolerance is exceeded. Routing facts are certified with `_csa_topk_proof_input(seq_len=2, hidden_size=16)` and `tiny_topk_moe_routing` against the proof's router.

## Consequences

`real_mode_proofs.proofs_total` moves from 7 to 8 when the readiness command runs under the MLX venv. The top-level readiness state remains fail-closed: `full_forward_parity=false`, `marker_earned=false`, `blockers_count=4`, and no `.deepseek-v4-forward-parity-ok` marker or `model-4bit` artifact is written.

B2 is not satisfied. Packed FP4 expert dequant, real checkpoint payload decode, expert parallel kernels, `hc_mult>1` multi-layer stacking (B1), and shimmed-checkpoint load/generation (B3) remain deferred. No production/vendor MoE, dequant, forward, gate, marker, conversion, training, quantization, or generation path changes are part of this decision.
