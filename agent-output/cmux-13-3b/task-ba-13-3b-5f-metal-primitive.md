# Story 13.3b-5f — BA: opaque packed-FP4 Metal/MLX training primitive

## Authorization
User explicitly authorized proceeding after Architect 13.3b-5e STOPped the current Python custom-VJP Path A.

## Goal
Define trackable requirements for the smallest lower-level Metal/MLX training primitive that reopens local 4096-token QLoRA by eliminating the E×H×I expert dequant/VJP graph retained by MLX 0.31.2 outer transforms.

## Model
Use current user-selected `openai-codex/gpt-5.6-sol` high. Do not switch.

## Read first
1. `agent-output/cmux-13-3b/architecture-13-3b-5e-peak-adjudication.md`
2. `agent-output/cmux-13-3b/review-13-3b-5d.md`
3. `agent-output/cmux-13-3b/coder-13-3b-5d-stop.md`
4. ADR 0024, ADR 0025, ADR 0026
5. Story 13.3b-5 section in `docs/backlog.md`
6. `docs/architecture.md`, `docs/technical-spec.md`

## Requirements to pin
Create one new user story using exact project format.

The primitive must:
- consume packed OCP E2M1 FP4 weights directly, LSB-first;
- apply BF16 linear-domain scales per 32 logical inputs exactly as ADR 0024;
- dequantize tiles internally and never materialize/return full FP32 expert matrices as MLX arrays;
- operate only on unique routed token rows for one selected expert/assignment group;
- provide exact first-order outputs/cotangents required by routed MoE training: expert output `y_e`, expert-input cotangent `dx_e`, and rowwise `a_e=dot(g_e,y_e)` for normalized score gradients;
- use the exact simplified score derivative `dscore_e = rsf*a_e/D' - dot(g,routed)/D'`;
- return no packed-weight or scale cotangents;
- preserve clip, sigmoid-SwiGLU, w1/w3/w2 ordering, routing normalization, duplicate collapse, shared expert, token order, and LoRA gradient semantics;
- be exposed as an opaque custom-VJP leaf so outer MLX transforms retain assignment-proportional tensors, not expert dequant graphs;
- remain training-only and isolated from production inference, SSD, CUDA, distributed, and default Metal paths.

Pin RED/GREEN gates:
- current Python path reproduces E-scaled peak RED;
- forward parity `max_abs <=1e-6` versus independent ADR 0024 reference;
- input and score/gate VJP parity `atol=rtol=1e-5`, including clamp boundaries;
- fixed K/T E=2/4/8 operation-peak spread `<=64 MiB` and no E×H×I slope;
- real H=4096/I=2048 no-shard operation peak `<2 GiB` with finite outputs/cotangents;
- formula-backed E=256/K=6/T=4096 routed primitive bound `<=2 GiB`;
- focused LoRA gradients non-empty/all-finite/at-least-one-nonzero;
- all verdict tests tracked;
- no full model/shard run during implementation/review/tests;
- Reviewer PASS + Tester GREEN before one 4096 smoke;
- 4096 smoke retains 400GB graph limit, 340GB pre-backward STOP, one validation batch, eager wrapper, per-step memory telemetry, no shorter fallback.

## Documentation/architecture boundary
- Revoke/supersede ADR 0025 claim that Python per-expert `mx.eval` guarantees one-expert lifetime.
- Require a new durable ADR for the opaque packed-FP4 training primitive and derivative/backend-isolation contract.
- ADR 0024 semantics unchanged.
- Preserve 13.3b-5b sanctioned FROZEN stop-gradient fix and complete staged chain of custody.

## STOP conditions
Carry Architect 13.3b-5e §10 forward exactly. Add STOP if the primitive requires production inference/FROZEN edits, changes FP4 values, allocates full dense expert matrices, cannot expose exact first-order VJP through installed MLX, or cannot meet measured peak gates.

## Deliverable
Update only relevant canonical Story 13.3b-5 text in `docs/backlog.md` and write `agent-output/cmux-13-3b/requirements-13-3b-5f-metal-primitive.md` with user story, AC, scope, double-green gate, run policy, and STOP conditions.

No code/Metal/test edits. Create `.cmux-status/ba.done` only on success. End with one unwrapped JSON line.
