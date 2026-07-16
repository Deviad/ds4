# ADR 0028 — Opaque packed-FP4 Metal training primitive

- **Date:** 2026-07-14 (Story 13.3b-5f)
- **Status:** Accepted for primitive correctness; reduction-order parity amended by Architect r3; real-training memory path STOPPED by Story 13.3b-5g pending multi-layer synthetic proof (2026-07-14)
- **Supersedes / amends:** Supersedes ADR 0025's Story 13.3b-5d amendment only where that amendment authorizes the Python `forward_one` + nested `mx.vjp` implementation or claims that per-expert `mx.eval` barriers bound its lifetime under an outer MLX transform. Does not change ADR 0024 or ADR 0026.
- **Related:** ADR 0024 (binding FP4 packing/scale oracle), ADR 0025 (training sibling), Story 13.3b-5e (graph-lifetime RED/root-cause authority).

## Context

MLX 0.31.2 retains the Python-composed expert dequantization and nested-VJP graphs until the trainer's outer `mx.grad` / `mx.value_and_grad` result is evaluated. Per-expert `mx.eval`, `mx.stop_gradient`, deletion, cache clearing, phase custom functions, and route chunks do not create a one-expert lifetime boundary. At fixed assignments the measured operation peak therefore grows with expert count.

Installed MLX 0.31.2 can instead launch `mx.fast.metal_kernel` nodes and use them inside an `mx.custom_function` VJP. A light probe produced the correct outer `mx.value_and_grad` result, returned zero cotangent for arrays captured by the custom-function closure, and exported each Metal operation as one `CustomKernel` graph node. BF16 inputs, FP32 outputs, static threadgroup storage, padded-grid bounds, multiple outputs, and FP32 atomics were also launch-proven. No model or shard was used.

## Decision

### Training-only boundary and backend isolation

Add the primitive only to the MLX fine-tuning package:

- `python-envs/mlx/src/ds4_ft_mlx/metal/ds4_routed_fp4_train.metal`
- `python-envs/mlx/src/ds4_ft_mlx/routed_fp4_metal.py`
- one call-site replacement in `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py`

The Metal file is package data loaded only by `routed_fp4_metal.py`. It must not be placed under root `metal/`: `Makefile` treats `metal/*.metal` as a production `ds4_metal.o` dependency. No production source loader, Makefile, Objective-C wrapper, C/CUDA/ROCm source, inference backend, model loader, SSD path, or distributed path may reference the training source.

The module fails closed unless all conditions hold: Darwin, arm64, installed `mlx==0.31.2`, `mx.metal.is_available()`, and a GPU default device. There is no CPU implementation and no fallback to the stopped Python graph composition.

### Exact packed-weight and forward contract

For activation width `H`, routed intermediate width `I`, `Hp=ceil(H/32)*32`, and `Ip=ceil(I/32)*32`, one expert has:

```text
w1, w3: uint8 [I, Hp/2]       s1, s3: bfloat16 [I, Hp/32]
w2:     uint8 [H, Ip/2]       s2:     bfloat16 [H, Ip/32]
```

Each byte decodes LSB-first: low nibble first, then high nibble, through the exact ADR 0024 E2M1 table:

```text
0,+0.5,+1,+1.5,+2,+3,+4,+6,-0,-0.5,-1,-1.5,-2,-3,-4,-6
```

For logical input coordinate `k`, decoded weight is multiplied directly by the BF16 scale at `k//32`, converted to FP32. Native E8M0/mxfp4 scale handling is forbidden. Padding coordinates in `Hp-H` or `Ip-I` consume the checkpoint's complete 32-logical-input block but multiply a zero activation.

For logical selected rows `x_e[r]=x_flat[rows_e[r]]`:

```text
u1     = x_e @ W1.T
gate   = min(u1, limit)
u3     = x_e @ W3.T
up     = clamp(u3, -limit, limit)
s      = sigmoid(gate)
hidden = (gate * s) * up
y_e    = hidden @ W2.T
```

All dot products accumulate in FP32. `hidden` and `y_e` are FP32; the public primitive output is `y_e[R_e,H]` FP32, matching the current FP32-dequantized MLX path. Metal uses FP32 and `metal::precise::exp` for sigmoid. Because deterministic SIMD and ordinary-MLX reductions may associate the same FP32 products differently, forward parity against an independent ADR 0024 reference uses the scale-aware elementwise bound `abs(got-ref) <= 2e-6 + 1e-6*abs(ref)` and NRMSE `<=1e-6`. Tests report maximum absolute error and maximum relative error above scale `1e-6`; a raw maximum absolute error above `1e-6` is not independently a failure.

The kernels receive `x_flat` plus detached `rows_e` and gather logical `x_e` internally. They do not materialize an MLX `x_e[R_e,H]` copy. This is an allocation optimization only; row order and mathematical input are unchanged.

### Exact first-order VJP contract

For routed-output cotangent `g_e[r]=g_flat[rows_e[r]]` and normalized expert factor

```text
f_e = routed_scaling_factor * selected_score_e / D'
D'  = duplicate-collapsed selected-score denominator + 1e-20
```

the opaque backward returns only:

```text
dx_e[R_e,H] float32
a_e[R_e]     float32, where a_e = rowwise_dot(g_e, y_e)
```

Define:

```text
gu      = g_e @ W2
sig     = sigmoid(gate)
silu    = gate * sig
dhidden = f_e[:,None] * gu

du3 = dhidden * silu * 1(-limit < u3 < limit)
du1 = dhidden * up * (sig + gate*sig*(1-sig)) * 1(u1 < limit)

dx_e = du1 @ W1 + du3 @ W3
a_e  = rowwise_sum(hidden * gu)
```

The inequalities are strict. MLX 0.31.2 gives zero derivative at `u1==limit` and at `u3==-limit` or `u3==limit`; the Metal masks must do the same. No higher-order-gradient contract is made.

The routed custom VJP computes score cotangents outside the Metal primitive with the exact identity:

```text
dscore_e = routed_scaling_factor*a_e/D' - dot(g, routed)/D'
```

This is algebraically identical to the former Q expression and removes the Q recomputation pass. Duplicate expert ids within a token are collapsed before `D'`; unselected scores receive zero. Packed weights and scales are captured frozen constants, are passed only as Metal kernel inputs, and receive no cotangent.

### Selected kernel decomposition

Use six cached Metal kernels. Every matrix kernel uses `BM=8`, `BN=8`, `BK=32`, `threadgroup=(256,1,1)`, eight SIMDgroups per threadgroup, FP32 lane accumulators, and a 32-logical-input dequant tile in threadgroup memory. A threadgroup owns an `8x8` output tile; one SIMDgroup owns one token row. The first SIMDgroup loads and dequantizes the `8x32` weight tile, all row SIMDgroups reuse it, and `simd_sum` reduces the K lanes. Lane `l` accumulates coordinates `l,l+32,l+64,...` with FP32 `metal::fma` in increasing block order; after all K blocks every lane participates in `simd_sum`, and lane 0 alone writes. `pair_swiglu_forward` and `pair_swiglu_vjp_terms` must use the same lane mapping, FMA order, final reduction, and reduced `u1/u3` values for clamp decisions. `BK=32` aligns exactly with one ADR 0024 scale block. Lane-0 serial tile dots and shape-selected reduction variants are forbidden.

1. `pair_swiglu_forward`: gather `x_e`; jointly project packed `w1/w3`; write `hidden[R,I]`.
2. `down_forward`: project `hidden` through packed `w2`; write `y[R,H]`.
3. `down_input_vjp`: gather unscaled `g_e`; multiply by the logical transpose of packed `w2`; write `gu[R,I]`.
4. `pair_swiglu_vjp_terms`: recompute `u1/u3` once, apply exact clamp masks, write `du1[R,I]`, `du3[R,I]`, and `a_partial[R,ceil(I/8)]`.
5. `pair_input_vjp`: multiply `du1/du3` by logical transposes of packed `w1/w3`; write `dx[R,H]`.
6. `reduce_a`: one 256-thread threadgroup per row reduces `a_partial`; write `a[R]`.

Matrix grids are `(256, ceil(R/8), ceil(N/8))`; `reduce_a` uses `(256,R,1)`. Every kernel guards partial row/output tiles. `R==0` is skipped in Python. `ensure_row_contiguous=False` is mandatory; kernels use generated shape/stride metadata, avoiding hidden copies of expert slices. Outputs are row contiguous by MLX contract.

No selected kernel uses atomics. This avoids non-deterministic accumulation and mixed atomic/non-atomic output constraints. MLX 0.31.2 FP32 atomic support was nevertheless launch-proven and is not a design dependency.

The decomposition is `O(R*H*I)`: forward performs three projection-equivalents; backward performs five. It never recomputes an `H`-wide projection separately for each `dx` coordinate and therefore has no `O(R*H^2*I)` term.

### Python transform and assignment layout

One `mx.custom_function` wraps the complete routed operation with positional primals `(x_flat, scores_flat)`. It captures packed weights/scales, detached route metadata, and configuration. MLX's documented closure rule makes captured arrays constants.

Host code builds deterministic int32 `assignment_rows`, `assignment_eids`, and expert segment offsets from `mx.stop_gradient(indices)`. Expert outputs are concatenated once in expert-segment order. Forward performs one denominator scatter and one routed-output scatter; backward performs one `dx` scatter and one selected-score scatter. Repeated full `[T,H]` or `[T,E]` per-expert accumulators are forbidden.

The outer transform may retain opaque `CustomKernel` nodes and assignment-proportional arrays. It must not see or retain dequantized `[I,H]` / `[H,I]` expert matrices. Shared expert execution remains one ordinary MLX call on the original full input outside the routed custom function.

### Memory contract

Let `A=sum_e R_e<=T*K` after duplicate collapse. Packed payload residency is frozen baseline state and is outside the measured operation delta; no dequantized weight matrix exists.

At `T=4096`, `K=6`, `E=256`, `H=4096`, `I=2048`, `A<=24576`:

```text
A*H*4                 = 384 MiB
A*I*4                 = 192 MiB
A*ceil(I/8)*4          = 24 MiB
T*H*4                  =  64 MiB
T*E*4                  =   4 MiB
```

Conservative simultaneous forward bound:

```text
hidden + y + contribution            192 + 384 + 384 = 960 MiB
x + routed + scores                    64 +  64 +   4 = 132 MiB
forward total                                          = 1092 MiB
```

Conservative simultaneous backward bound:

```text
gu + du1 + du3 + a_partial           192 + 192 + 192 + 24 = 600 MiB
dx_assignment                                               384 MiB
x + routed + g + dx                    64 * 4                = 256 MiB
scores + dscores                         4 * 2                =   8 MiB
backward total                                                = 1248 MiB
```

Factors, denominators, assignment metadata, and `a` are below 1 MiB combined. Static threadgroup scratch is at most a few KiB per resident threadgroup. Allowing 256 MiB for graph metadata, layout materialization discovered by telemetry, and allocator variance gives a documented operation envelope of `1504 MiB`, below the `2 GiB` hard bound with `544 MiB` headroom. There is no `E*H*I` term. Any implicit-contiguity copy or repeated full scatter that invalidates this envelope is a STOP.

### Failure and validation policy

Implementation proceeds RED to GREEN with tracked tests only. Required evidence includes independent packed-forward/VJP/score references, strict order-invariant clamp-boundary cases, duplicate and empty routes, closure/no-frozen-cotangent checks, focused LoRA propagation, a deterministic nonuniform-cotangent/downstream-logit proxy, fresh-process E=2/4/8 operation peaks, a real-dimension no-shard kernel probe, the formula bound above, graph/source opacity checks, and direct production/FROZEN hashes. VJP and score parity remain `atol=rtol=1e-5`; exact boundary masks are not tolerance-relaxed.

No model, shard, dataset run, or 4096 smoke is allowed before independent Reviewer PASS and Test Manager GREEN. ADR 0024 remains byte-unchanged. The later one-smoke policy and 340 GB process gate remain exactly as pinned by Story 13.3b-5f.

## Consequences

- The routed expert forward and first-order input VJP become opaque Metal work while routing and score algebra remain ordinary, inspectable MLX.
- Memory scales with assignments, not dequantized expert count.
- Frozen packed experts remain non-trainable; LoRA gradients continue through activation and router-score paths.
- Training now requires Apple Metal and exactly the validated MLX release; unsupported platforms fail closed without changing CPU or inference behavior.
- Six launches per non-empty expert trade launch overhead for a small exact implementation. The `8x8x32` weight-reuse tile keeps the design credible for the gated 43-layer/20-iteration smoke and avoids the prohibited quadratic recomputation. Before Reviewer round 3, a tracked 25-repeat benchmark at `R=96` must show forward p50 `<=0.0080s`, input-VJP p50 `<=0.0140s`, and `256x43x20` extrapolation `<=1.25h` p50 / `<=1.35h` p95. These are review gates, not authorization for the later 5,000-iteration run.

## Story 13.3b-5g post-smoke memory amendment

The single authorized real 4096-token smoke at commit `5dee4ce` completed finite validation (`18.109817504882812`) and then aborted during evaluation of the first backward graph with Metal command-buffer out-of-memory, exit `134`. Last completed validation telemetry was active `160,085,895,190` bytes and peak `205,523,058,182` bytes. No gradient callback, optimizer update, or adapter payload completed.

The one-operation gates above establish packed-primitive opacity and assignment-proportional local tensors only. They do not establish a 43-layer trainer peak: the real-dimension helpers cover one routed block with two experts, omit ordinary attention/shared-expert backward, and do not exercise the outer checkpointed graph's depth-by-expert launch count. MLX-LM's `grad_checkpoint(model.layers[0])` does cover all 43 current `DecoderLayerNN` instances through their shared class `__call__`; no checkpoint call-site fix is authorized from this RED. Coverage does not itself prove scheduler or command-buffer lifetime bounds.

Real training through this path is STOPPED. The next authorized slice is diagnostic-only: a tracked, no-model/no-shard multi-layer synthetic peak proof with fresh-process depth, expert-count, component-ablation, and single-graph-versus-sequential controls. Any production redesign requires a later Architect re-entry using that report. No second real smoke, shorter fallback, full run, custom-primitive redesign, or layer-serial backward is authorized by this amendment.
