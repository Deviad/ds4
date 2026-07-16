# Story 13.3b-5f — Coder round 3: restore SIMD reduction path

## Goal
Implement Architect/BA r3 decision before Reviewer round 3. No full model/shards/smoke.

## Model
Use user-authorized `openai-codex/gpt-5.5` high.

## Read
- `architecture-13-3b-5f-r3-reduction.md`
- `requirements-13-3b-5f-r3-reduction.md`
- amended ADR 0028 / technical spec
- current kernel/tests/benchmarks.

## Required implementation
1. Restore SIMD lane accumulation and `simd_sum` in K1–K5; K6 unchanged.
2. Keep BM8/BN8/BK32 tiles, eight-row reuse, one path for every shape.
3. No lane-0 dot loops or shape-selected semantic branches.
4. K1/K4 must use identical lane/FMA/reduction skeleton and clamp decisions; make this source/test-load-bearing.

## Re-pinned correctness
Forward primitive and whole sparse outputs:
```text
abs(got-ref) <= 2e-6 + 1e-6*abs(ref)
NRMSE <=1e-6
```
Report max abs/max relative. Keep derivative/score `atol=rtol=1e-5`; exact clamp equality masks unchanged.

Add tracked downstream no-model proxy: nonuniform cotangent + fixed projection to proxy logits; x/score gradient parity 1e-5; projected logits combined bound+NRMSE; top-1 identity with nonzero reference margin; no K1/K4 clamp mismatch.

## Performance
Tracked R=1/8/32/96 helper: warm-up then >=25 measured repeats. R=96 must meet:
- forward p50 <=0.0080s;
- input-VJP p50 <=0.0140s;
- 20-iteration primitive extrapolation <=1.25h p50 and <=1.35h p95.

## Preserve all prior GREEN gates
BF16/FP16 cast path, eid guards, legitimate derivative/score/frozen/opacity tests, score-inclusive E-memory spread, real peak <2GiB, 1504MiB formula, tracked-only baseline, complete staged chain/hashes/docs.

STOP on any Architect condition. No tolerance weakening beyond canonical combined contract. No commit.

Update `coder-13-3b-5f-notes.md` with RED/GREEN, parity diagnostics, performance, tests, memory, tracking. Marker only GREEN; error JSON on STOP.
