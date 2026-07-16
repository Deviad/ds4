# Story 13.3b-5f — Architect micro-repin: tiled reduction/parity/performance contract

## Goal
Adjudicate Coder r2's deviation from ADR 0028 before Reviewer round 3. No production edits/full model.

## Evidence
Read updated Coder notes, round-2 review, current package Metal source, ADR 0028, and relevant parity/performance logs.

Coder removed shape-selected serial overwrite branches and kept BM8/BN8/BK32 shared tiles/8-row reuse. To recover strict small-shape parity, all five matrix kernels now perform deterministic lane-0 accumulation over each 32-wide tile; only K6 uses `simd_sum`. ADR 0028 currently requires SIMD reductions for every matrix kernel.

Metrics:
- parity/tests GREEN under current lane-0 tiled reduction;
- E peak spread 37.6MB; real peak 208.7MB;
- R=96 primitive extrapolation 2.232h p50 / 2.276h p95 for 20 iterations;
- earlier SIMD tiled implementation was ~1.04h but produced ~1.14e-5 full-MoE small-fixture drift versus the pinned 1e-6 contract;
- final 5,000-iteration plan would amplify primitive runtime substantially.

## Required decision
Use light probes/analysis only. Choose exactly one:
1. retain ADR 0028 SIMD reductions and specify a numerically justified correction that meets 1e-6 without a semantic variant;
2. amend ADR 0028 to accept deterministic lane-0 per-tile accumulation with an explicit performance/full-training consequence;
3. re-pin parity tolerance/reduction order based on independent absolute+relative error and downstream gradient/logit evidence;
4. STOP and require another kernel reduction design.

Must address:
- correctness meaning of strict absolute 1e-6 versus reduction-order-equivalent FP32 math;
- forward/VJP using exactly the same reduction order and clamp decisions;
- smoke feasibility and 5,000-iteration feasibility;
- no shape-selected semantic variants;
- no E×H×I memory regression;
- whether BA requirements/ADR 0028/technical spec require amendment.

## Deliverable
Write `agent-output/cmux-13-3b/architecture-13-3b-5f-r3-reduction.md` with selected decision, evidence, exact kernel/test changes or accepted current design, tolerances, performance gate, docs impact, and STOP conditions. Amend ADR 0028 only if decision requires it.

No implementation edits. Create `.cmux-status/architect.done` only on success. End with unwrapped JSON.
