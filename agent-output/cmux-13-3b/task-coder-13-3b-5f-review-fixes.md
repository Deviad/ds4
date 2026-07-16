# Story 13.3b-5f — Coder remediation after Reviewer FAIL

## Goal
Address all six blockers in `review-13-3b-5f.md`. No full model/shards/smoke.

## Model
Use user-authorized `openai-codex/gpt-5.5` high.

## TDD and blockers

### R1 — reproducible tracked baseline
Do not add unrelated 17 untracked tests merely to inflate counts. Build the full verdict baseline only from `git ls-files 'tests/test*.py'`; report it explicitly as tracked-only. Every new/modified verdict test and probe helper must be `git add`ed. Never cite the previous 598/18 count as reproducible.

### R2/R5 — implement accepted tiled kernels and performance
Write RED structural/performance tests first. Implement ADR 0028 `BM=8, BN=8, BK=32` design for every matrix kernel:
- 256 threads / eight SIMDgroups;
- threadgroup FP32 8×32 dequant tiles (pair tiles for w1/w3);
- eight output channels and eight routed rows per grid tile;
- weight reuse across eight rows;
- `simd_sum` FP32 reductions;
- exact padding/stride/nibble/BF16-scale semantics.

K4 pointwise and K6 reduction remain per architecture; do not invent unnecessary matrix tiling.

Rerun p50/p95 at R=1/8/32/96 and extrapolate 256 experts ×43 layers ×20 iterations. If tiling does not materially remove the ~15.17-hour primitive-only bottleneck or remains plainly impractical, STOP for Architect rather than claiming GREEN.

### R3 — load-bearing tracked memory proof
Add a tracked file-backed fresh-process probe helper (non-pytest name if appropriate) and tracked tests that independently reproduce:
- fixed K=2,T=1024,H=1024,I=512,E=2/4/8, A fixed/all experts non-empty;
- operation peak deltas and `max-min <=64 MiB`;
- active/cache/peak telemetry at expert boundaries plus complete operation peak;
- H=4096/I=2048 no-shard operation peak delta `<2 GiB`, not final active only;
- exact 1504 MiB assignment-buffer formula, no stale 96MiB full-dequant/1.25GiB assertions.

Subprocess commands must be reproducible from tracked files and work with multiprocessing spawn; never run spawn code from stdin.

### R4 — complete derivative/opacity tests
Add legitimate independent tests for:
- exact `u1==+limit`, `u3==-limit`, `u3==+limit` derivative behavior;
- simplified score identity versus prior exact Q form with duplicate-collapsed denominators;
- DOT/export outer graph has `CustomKernel`, no dequant arithmetic;
- captured packed/scales receive no cotangent / differentiable primals only;
- installed package resource contains Metal source;
- unsupported platform/version fail closed;
- source/API rejects fallback `mx.vjp(forward_one)`, native mxfp4/E8M0, dense dequant symbols/outputs;
- kernel source structurally proves required tiles/threadgroup/simd reductions, backed by parity/performance—not marker names only.

### R6 — wrapper/package validation
Add deterministic guards and tests for score dtype/expert dimension, rows/expert count, expert/payload bounds, finite positive routed scale, unsigned grid limits, duplicate source markers, shapes/dtypes/padding. Align package dependency with exact runtime contract (`mlx==0.31.2`) unless Architect evidence requires a different fail-closed mechanism.

## Preserve
- exact parity gates 1e-6/1e-5;
- package-local training-only source;
- no full dense matrices or E×H×I graph;
- no production/FROZEN/root-Metal/ADR0024 changes;
- complete staged 13.3b-5b + 5f chain;
- protected hashes and old-hash-zero checks.

## Verification
Focused primitive/sparse, tracked memory helpers, tracked-only full regression, `make`, `git diff --check`, tracking, staged scope, protected hash manifest. Record exact metrics.

Update `coder-13-3b-5f-notes.md`. On GREEN create `coder.done` and unwrapped success JSON. If performance/memory/parity remains blocked, write `coder-13-3b-5f-stop.md`, no success marker, error JSON. No commit.
