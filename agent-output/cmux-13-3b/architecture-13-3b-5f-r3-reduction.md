# Story 13.3b-5f — tiled reduction/parity/performance adjudication

**Decision: Option 3 — re-pin parity to the accepted SIMD reduction order, with combined absolute/relative and downstream derivative/logit evidence.**

The current lane-0 matrix reductions are not accepted. Restore ADR 0028's SIMD matrix reductions. Keep one semantic path for every shape.

## Evidence reviewed

Read in full:

- `task-architect-13-3b-5f-r3-reduction.md`;
- `coder-13-3b-5f-notes.md`;
- `review-13-3b-5f-r2.md`;
- current `python-envs/mlx/src/ds4_ft_mlx/metal/ds4_routed_fp4_train.metal`;
- `docs/adr/0028-opaque-packed-fp4-metal-training-primitive.md`;
- Story 13.3b-5f BA requirements, original architecture, canonical backlog and technical-spec clauses;
- relevant primitive/full-MoE parity, memory, benchmark, and tracked-baseline logs.

No model, shard, dataset, or full smoke was loaded. One in-memory light probe temporarily constructed uniquely named SIMD kernels from the current source; it did not edit a project file.

## Why strict absolute `1e-6` is not the correctness definition

FP32 addition is non-associative. A serial dot and a deterministic SIMD tree can decode the same FP4 values, multiply the same FP32 operands, and implement the same real-number expression while differing by several FP32 ulps. A scale-independent absolute-only bound mistakes one reduction order for model semantics. It is especially unsuitable when output magnitude changes with activation and projection scale.

Correctness remains strict for:

- ADR 0024 nibble order, E2M1 values, BF16 linear scales, and per-32 scale blocks;
- FP32 products and accumulators;
- projection topology, routing, normalization, duplicate collapse, shared expert, and token order;
- clamp formulas and strict equality masks;
- the same deterministic reduction order in forward and VJP recomputation.

Numerical parity becomes a combined absolute/relative contract against the independent ordinary-MLX reference. This permits reduction-order-equivalent FP32 rounding; it does not permit approximated values, altered clamps, mixed precision, or a shape-selected implementation.

## Light-probe result

Deterministic full sparse-MoE fixture, independent ordinary-MLX reference, nonuniform output cotangent, and deterministic `32 -> 17` downstream linear-logit projection:

| Quantity | lane-0 vs reference | SIMD vs reference |
|---|---:|---:|
| forward max absolute | `1.9073486328125e-06` | `1.1444091796875e-05` |
| forward max relative for scale `>1e-6` | `2.8643003344929933e-07` | `2.3600217971613185e-07` |
| forward NRMSE | `3.335597764342698e-08` | `1.925280576151898e-07` |
| input-gradient max absolute | `5.7220458984375e-06` | `3.814697265625e-06` |
| input-gradient max relative for scale `>1e-6` | `4.4739478437897845e-07` | `2.2238913762496183e-07` |
| input-gradient NRMSE | `1.6941104545410652e-07` | `1.0327389454352521e-07` |
| logit-proxy max absolute | `7.152557373046875e-07` | `2.86102294921875e-06` |
| logit-proxy NRMSE | `6.001391165695709e-08` | `2.652090297376562e-07` |
| logit-proxy top-1 | identical | identical |

SIMD scalar-loss difference was `0.0` in FP32. These results show that the `1.1444091796875e-05` full-MoE absolute maximum is magnitude-dependent reduction rounding, not semantic drift: relative and normalized errors remain approximately `1e-7`, input gradients improve slightly, and the downstream projection decision is unchanged.

This light probe is sufficient to choose the reduction contract. It is not a substitute for Coder's tracked RED/GREEN tests or independent Reviewer/Test Manager reruns.

## Accepted matrix reduction order

K1 through K5 shall use the original ADR 0028 SIMD order:

1. `BM=8`, `BN=8`, `BK=32`, `threadgroup=(256,1,1)`, eight SIMDgroups, and one SIMDgroup per token row remain unchanged.
2. Lane `l` owns reduction coordinates `l, l+32, l+64, ...` in increasing `k0` order.
3. Each lane updates its FP32 accumulator with `metal::fma` for each visited coordinate.
4. After all K blocks, every lane participates in `simd_sum`; lane 0 alone writes the guarded output.
5. K1 `pair_swiglu_forward` and K4 `pair_swiglu_vjp_terms` use byte-equivalent lane mapping, FMA order, and `simd_sum` placement for `u1/u3`. K4 derives clamp masks from those SIMD-reduced values.
6. At `u1==+limit` and `u3==±limit` under that accepted reduction, derivatives remain exactly zero. Boundary fixtures use order-invariant, exactly representable sums so the equality test itself is unambiguous.
7. K6 keeps its existing deterministic two-level `simd_sum` reduction.

Forbidden:

- lane-0 serial loops across the 32-wide tile in any matrix kernel;
- shape-selected serial or alternate reduction branches;
- different K1/K4 reduction or clamp inputs;
- atomics, mixed-precision accumulators, native mxfp4/E8M0 substitution, or a permanent semantic flag.

## Re-pinned tolerances

### Forward

For primitive and whole sparse-MoE forward parity, require every element to satisfy:

```text
abs(got - ref) <= 2e-6 + 1e-6 * abs(ref)
```

Also require:

- finite output, exact shape/dtype/token order;
- NRMSE `<=1e-6`;
- reported maximum absolute error;
- reported maximum relative error for `max(abs(got),abs(ref)) > 1e-6`.

A raw `max_abs >1e-6` is no longer a failure when the combined bound and NRMSE pass. This is approximately a fourfold margin over the observed SIMD maximum relative error, while the absolute floor protects values near zero.

### First-order derivatives and score algebra

Keep BA's existing independent-reference gates unchanged:

```text
atol=1e-5, rtol=1e-5
```

Exact clamp-boundary masks remain exact and are not tolerance-relaxed. Packed weights/scales still receive no cotangent.

### Downstream guard

Add a tracked deterministic no-model probe using a nonuniform cotangent and fixed linear projection of sparse-MoE output to proxy logits. Require:

- input and score gradients pass `atol=rtol=1e-5` against the independent path;
- projected logits pass the forward combined bound and NRMSE `<=1e-6`;
- top-1 identity on a fixture with a documented nonzero reference margin;
- no clamp-decision mismatch between K1 and K4.

## Exact Coder changes

1. Restore SIMD lane accumulation and `simd_sum` in all five matrix kernels. Do not change K6.
2. Keep current tiles, eight-row weight reuse, package boundary, wrapper API, allocation plan, and memory layout.
3. Make K1/K4 reduction-order identity load-bearing. Prefer a shared common-header helper for lane accumulation/final reduction where Metal permits it; otherwise expose normalized source sections and assert identical lane/FMA/reduction skeletons.
4. Replace the stale structural test name/body with per-kernel assertions. Parse each kernel marker independently; require `simd_sum` in K1-K5 and reject lane-0 `for c in [0,32)` dot loops and every shape-selected branch.
5. Replace absolute-only forward assertions with the combined bound plus NRMSE and diagnostic maxima. Do not merely increase a scalar `max_abs` threshold.
6. Add the tracked downstream cotangent/logit-proxy test above.
7. Rerun primitive forward/VJP/score/clamp tests, full sparse-MoE parity and gradients, tracked full baseline, source/hash isolation, memory probes, and the tracked benchmark.

## Performance gate

Measured evidence:

- current lane-0, R=96 primitive extrapolation for `256 x 43 x 20`: `2.232 h` p50 / `2.276 h` p95;
- prior SIMD evidence: approximately `1.017-1.043 h` p50 / `1.058-1.090 h` p95;
- current lane-0 is about `2.14x` the independently reviewed SIMD p50.

Linear primitive-only 5,000-iteration extrapolation:

- lane-0: `558.011 h` p50 / `568.905 h` p95 (`23.250-23.704 d`);
- SIMD: `260.750 h` p50 / `272.500 h` p95 (`10.865-11.354 d`);
- lane-0 adds about `297.261 h` (`12.386 d`) before non-primitive training work.

Lane-0 is feasible for the 20-iteration smoke but fails final-training credibility. It is rejected.

Before Reviewer round 3, the tracked `R={1,8,32,96}` helper shall run at least 25 measured repeats after warm-up. At R=96 require:

```text
forward p50 <= 0.0080 s
input-VJP p50 <= 0.0140 s
256 x 43 x 20 extrapolation <= 1.25 h p50 and <= 1.35 h p95
```

These gates allow normal run variance above both recorded SIMD runs while rejecting the lane-0 design. Passing them authorizes review, not the full 5,000-iteration run. Final training still requires a separately authorized end-to-end pilot and operator runtime budget after the one allowed smoke.

## Memory and backend invariants

The reduction re-pin changes no output or assignment tensor. Existing evidence remains supportive: fixed-E peak spread `37,605,660` bytes and real-dimension operation peak `208,720,484` bytes. Coder and independent roles must rerun the load-bearing helpers after restoring SIMD.

Hard gates remain:

- E spread `<=64 MiB` at fixed assignments;
- real-dimension no-shard operation peak `<2 GiB`;
- formula envelope `<=2 GiB`;
- no `E*H*I` allocation or retained graph;
- no production/FROZEN/inference/SSD/CUDA/ROCm/distributed edit.

## Documentation impact

- ADR 0028 requires amendment because its SIMD decomposition remains binding but its absolute-only `1e-6` forward gate is replaced by the combined reduction-order contract and performance gate.
- `docs/technical-spec.md` requires the same durable parity/order/performance text; amended in this adjudication.
- The historical BA requirements artifact is evidence and must not be rewritten. BA owns a canonical `docs/backlog.md` re-pin: AC 2 and the parity STOP line must adopt the combined forward contract before Coder declares the r3 slice complete. VJP/score tolerances remain unchanged.
- `docs/architecture.md` needs no change: package boundary, kernel topology, and memory architecture remain the accepted ADR 0028 design.

## STOP conditions

STOP and return to Architect if any occurs:

- SIMD forward fails the combined bound or NRMSE gate;
- independent input/score gradients fail `atol=rtol=1e-5`;
- K1 and K4 do not use exactly the same `u1/u3` reduction order or make different clamp decisions;
- an exact order-invariant clamp-boundary fixture changes derivative semantics;
- the downstream proxy changes top-1 without a reference tie or fails its numerical gates;
- R=96 exceeds any performance gate above;
- a shape-selected/lane-0 serial semantic variant is reintroduced;
- E-spread, real-dimension, formula, opacity, tracking, production-isolation, or baseline gates regress;
- canonical backlog remains at absolute-only `1e-6` when Coder requests r3 review;
- Reviewer is not PASS or Test Manager is not GREEN.

No real 4096 smoke is authorized by this adjudication.
