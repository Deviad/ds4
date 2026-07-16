# Story 13.3b-5e — Architect: E-scaled transient peak adjudication

## Goal
Adjudicate and redesign the remaining sparse routed FP4 VJP memory peak after Coder removed explicit all-expert caches. No full model/shard run and no production edits.

## Read
- `review-13-3b-5d.md`
- `coder-13-3b-5d-stop.md`
- 13.3b-5d requirements/architecture/Coder notes
- current `deepseek_v4_nn.py` and tracked sparse tests.

## Locked evidence
After two-pass recomputation and per-expert eval/stop/clear:

```text
K=2,T=1024,H=1024,I=512, all experts non-empty
E=2 active=60,325,900 peak=584,225,254
E=4 active=60,358,668 peak=779,041,700
E=8 active=60,424,204 peak=1,179,289,384
```

Active retained delta is flat; operation peak grows materially with E. Real-dimension E=2 probe previously reported ~4.08GB operation peak and ~201–268MB active delta. Sparse math tests pass; no real smoke allowed.

## Required investigation — light/no-shard only
1. Reproduce with per-expert/per-phase telemetry: active, cache, peak before/after baseline reset, Q update, direct VJP, accumulator eval, stop-gradient, cache clear.
2. Verify probe legitimacy: baseline materialization, `mx.reset_peak_memory`, compile disabled, fresh process, payload residency, constant total assignments, and peak semantics.
3. Identify exactly what remains live/temporarily concurrent across experts: expert dequant arrays, Q graph, nested VJP tape, packed payload materialization, allocator cache, or custom-function transform capture.
4. Distinguish cumulative allocator high-water behavior from true concurrent graph retention. Do not relax the gate merely because final active memory is flat.
5. Extrapolate a defensible E=256/K=6/T=4096 bound using measured components; determine whether 400GB/340GB gate is credible.
6. Compare minimal corrections: phase-isolated custom functions, host-materialized Q only if semantics/gradient safety permits, expert-level VJP factory, score derivative rearrangement, route chunks, or a custom Metal/MLX primitive. Reject semantic approximations.
7. Reassess whether `mx.eval`/`mx.stop_gradient` inside a custom VJP can guarantee graph release in MLX 0.31.2.

## Decision
Choose exactly one next path:
- bounded implementation correction with RED tests and quantified expected peak, or
- STOP Path A as infeasible with current MLX primitives and specify the minimal lower-level primitive required.

Do not recommend another real smoke until Reviewer/Test Manager double-green.

## Deliverable
Write `agent-output/cmux-13-3b/architecture-13-3b-5e-peak-adjudication.md` with root cause, telemetry, selected design, memory bound, file/test map, TDD acceptance criteria, STOP conditions, and BA/ADR impact.

No edits outside handoff. Create `.cmux-status/architect.done` only on success. End with one unwrapped JSON line.
