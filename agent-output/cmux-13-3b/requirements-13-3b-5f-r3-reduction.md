# Story 13.3b-5f — BA r3 reduction-order parity/performance re-pin

## User story

As a training engineer (WHO), I want the opaque packed-FP4 Metal primitive pinned to one accepted SIMD reduction order with scale-aware parity and measured performance gates (WHAT), so that reduction-order-equivalent FP32 rounding remains correct without accepting the lane-0 slowdown or downstream semantic drift (WHY).

## Decision source

Canonical backlog re-pin applies `architecture-13-3b-5f-r3-reduction.md` Option 3 and the matching amended ADR 0028 / technical-spec clauses. Historical requirements artifacts remain evidence and are not rewritten.

## Scope

- Update only the current Story 13.3b-5f block in `docs/backlog.md`.
- Re-pin forward parity, accepted SIMD reduction order, downstream no-model evidence, performance gates, and Architect r3 STOP conditions.
- Preserve VJP/score tolerance, exact clamp semantics, memory/backend/scope gates, double-green gate, and prior smoke policy.

## Out of scope

- Production code, Metal kernels, tests, ADRs, technical specification, or architecture edits.
- Model, shard, dataset, training, or real 4096-token smoke execution.
- Authorization of a full 5,000-iteration run.

## Acceptance criteria

1. Primitive and whole sparse-MoE forward parity require every element to satisfy:

   ```text
   abs(got-ref) <= 2e-6 + 1e-6*abs(ref)
   ```

   Outputs must be finite with exact shape, dtype, and token order; NRMSE must be `<=1e-6`; evidence reports maximum absolute error and maximum relative error where `max(abs(got),abs(ref)) > 1e-6`. A raw maximum absolute error above `1e-6` is not independently a failure when the combined element-wise bound and NRMSE pass.

2. Primitive VJP, whole sparse input cotangents, and score/gate cotangents remain pinned to independent-reference `atol=rtol=1e-5`. Exact clamp-boundary masks remain exact and are not tolerance-relaxed. Packed weights/scales receive no cotangent.

3. K1-K5 use the accepted SIMD order: `BM=8`, `BN=8`, `BK=32`, `threadgroup=(256,1,1)`, eight SIMDgroups, one SIMDgroup per token row; lane `l` accumulates `l,l+32,l+64,...` in increasing block order with FP32 `metal::fma`; every lane participates in final `simd_sum`; lane 0 alone writes. K1/K4 use the same lane mapping, FMA order, `simd_sum` placement, and reduced `u1/u3` clamp inputs. K6 remains unchanged.

4. Tracked per-kernel structural evidence requires `simd_sum` in K1-K5 and rejects lane-0 32-wide serial dot loops and every shape-selected reduction branch.

5. A tracked deterministic downstream no-model proxy using a nonuniform cotangent and fixed linear projection requires:
   - input and score gradients pass `atol=rtol=1e-5` against the independent path;
   - projected logits satisfy the forward combined element-wise bound and NRMSE `<=1e-6`;
   - top-1 identity on a fixture with a documented nonzero reference margin;
   - identical K1/K4 clamp decisions.

6. Before Reviewer round 3, a tracked `R={1,8,32,96}` helper runs at least 25 measured repeats after warm-up. At `R=96`:
   - forward p50 `<=0.0080 s`;
   - input-VJP p50 `<=0.0140 s`;
   - `256 x 43 x 20` primitive extrapolation `<=1.25 h` p50 and `<=1.35 h` p95.

7. Performance GREEN authorizes review only. Independent Reviewer PASS and Test Manager GREEN remain mandatory before the separately authorized one-smoke policy. This re-pin authorizes no real smoke and no full 5,000-iteration run.

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

## BA result

Canonical `docs/backlog.md` Story 13.3b-5f now carries the Architect r3 reduction-order parity/performance contract. No code, test, ADR, technical-spec, or architecture file was edited.
