# Story 13.3b-5f — BA micro-repin: SIMD reduction-order parity

## Goal
Apply Architect r3 reduction adjudication to canonical Story 13.3b-5f backlog before Coder round 3.

## Model
Use `openai-codex/gpt-5.6-sol` high.

## Read
- `architecture-13-3b-5f-r3-reduction.md`
- current Story 13.3b-5f in `docs/backlog.md`
- amended ADR 0028 and technical spec.

## Required canonical changes
Update only relevant Story 13.3b-5f AC/STOP text:

Forward parity becomes element-wise:
```text
abs(got-ref) <= 2e-6 + 1e-6*abs(ref)
```
plus finite/exact shape,dtype,order; NRMSE <=1e-6; report max absolute and max relative error.

Keep VJP/score `atol=rtol=1e-5`; exact clamp-boundary masks unchanged.

Add downstream no-model proxy gates:
- input/score gradients 1e-5;
- projected logits combined bound + NRMSE <=1e-6;
- top-1 identity with documented nonzero reference margin;
- K1/K4 clamp-decision identity.

Restore/require SIMD accumulation+`simd_sum` in K1-K5; reject lane-0/shape branches.

Performance gate, >=25 measured repeats after warm-up at R=96:
- forward p50 <=0.0080s;
- input-VJP p50 <=0.0140s;
- 20-iteration primitive extrapolation <=1.25h p50 and <=1.35h p95.

Carry Architect STOP conditions exactly. No real smoke authorization.

## Deliverable
Update `docs/backlog.md` only in the current 13.3b-5f block. Write `agent-output/cmux-13-3b/requirements-13-3b-5f-r3-reduction.md`. No code/test/ADR edits. Marker only success; unwrapped JSON.
