# Story 13.3b-5f — Architect: opaque packed-FP4 Metal/MLX primitive

## Goal
Produce an implementation-ready architecture for the authorized training-only opaque packed-FP4 primitive. Prove feasibility with only tiny/light Metal API probes; no full model/shards.

## Model
Use current user-selected `openai-codex/gpt-5.6-sol` high. Do not switch.

## Read
- `requirements-13-3b-5f-metal-primitive.md`
- `architecture-13-3b-5e-peak-adjudication.md`
- current `deepseek_v4_nn.py`, ADR 0024/0025/0026
- installed MLX 0.31.2 APIs/source/docs for `mx.fast.metal_kernel`, custom functions/VJP, output shapes/dtypes, atomic/threadgroup behavior
- existing project Metal kernel/wrapper conventions, without changing production paths.

## Required investigation/design
1. Verify installed MLX can launch a training-only custom Metal kernel from Python and use its opaque outputs inside outer `mx.grad`/`mx.value_and_grad` without capturing internal dequant operations.
2. Pin exact current FP4 expert math and shapes for `w1`, `w3`, `w2`, packed storage, BF16 scales, clip, sigmoid-SwiGLU, output dtype, and clamp derivative.
3. Select exact kernel decomposition—no menu. Define kernels, inputs, outputs, grid/threadgroup, tiling, accumulation dtype, bounds, row/expert layout, and dynamic `R_e` handling.
4. Keep dequant tile-local. No full FP32 weight matrix MLX output or retained buffer.
5. Define forward outputs and backward outputs `dx_e` and `a_e`; use simplified exact score derivative outside the primitive. Explain how packed weights/scales are closure/frozen inputs and receive no cotangent.
6. Define Python wrapper/custom-function boundary so outer transforms retain only assignment-proportional arrays and opaque kernel nodes.
7. Quantify simultaneous buffers for real `R`, H=4096, I=2048; derive <=2GiB operation bound and fixed-assignment E-independence.
8. Address correctness/performance enough for a 43-layer/20-iteration smoke; correctness first, but reject an obviously infeasible O(H²I) recomputation design.
9. Define failure handling and platform guard for Apple Metal/MLX 0.31.2 without affecting CPU/inference paths.
10. Map exact files/functions and RED→GREEN tests, including independent references, boundary/clamp cases, fresh-process memory telemetry, tracking, and backend hashes.

## Durable docs required now
- Create `docs/adr/0028-opaque-packed-fp4-metal-training-primitive.md` with accepted boundary, exact derivative contract, memory contract, and backend isolation.
- Amend ADR 0025 to explicitly supersede the disproved Python `mx.eval` lifetime claim and point to ADR 0028.
- Update `docs/architecture.md` only if needed for the new training-only component map.
- ADR 0024 unchanged.

## STOP
STOP if installed MLX cannot expose the opaque exact first-order leaf, custom Metal kernels cannot implement required dtypes/atomics/bounds, design needs production/FROZEN changes, or formula cannot credibly meet <2GiB.

## Deliverable
Write `agent-output/cmux-13-3b/architecture-13-3b-5f-metal-primitive.md` with selected design, probe evidence, kernel pseudocode/signatures, wrapper API, derivative algebra, memory/performance bounds, edit map, TDD plan, docs, risks, and STOP conditions.

Create `.cmux-status/architect.done` only on success. End with one unwrapped JSON line.
