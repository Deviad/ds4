# Story 13.3b-5g r3 — Tester independent validation

## Verdict
GREEN.

## Fresh execution
- focused diagnostic suite: rc=0, `13 passed, 1 warning in 2.22s`
- compile test alone: rc=0, `1 passed, 1 warning in 0.02s`
- order A compile-alone: rc=0, `1 passed, 1 warning in 0.02s`
- order B diagnostic → compile: rc=0, `15 passed, 1 warning in 18.36s`
- order C compile → diagnostic: rc=0, `15 passed, 1 warning in 18.39s`
- tracked baseline: rc=0, `tracked_test_files=50`, `490 passed, 15 skipped, 2 warnings, 86 subtests passed in 73.30s`
- make: rc=0, `Nothing to be done for 'all'.`
- `git diff --check`: rc=0
- `git diff --cached --check`: rc=0

## Independent report audit
- rows: 64
- evidence entries: 64
- row schema: exact 25 keys
- row outcomes: all `exit_code=0`, `signal=null`, `error=null`
- serial proof: 64 unique nonces, no PID overlap, no interval overlap
- planned/executed identity list hash: match
- per-row evidence identity hash: matches planned identity list by ordinal
- Probe A: 43-layer checkpoint parity, non-empty equal gradient keys, equal gradient shapes, separate loss/gradient finiteness, `atol=rtol=1e-5` allclose, zero max diffs
- Probe B/D: trainable key counts and hashes consistent with exact LoRA topology; B/D input gradients present, exact shape `[1,64,4,128]`, nonzero, finite
- Probe C: exact `6 * depth * nonempty_experts` custom-kernel counts, `E=2` low-E control kept separate from normalized `E=8/32/128/256` axis
- Probe D fixture: 43 layers, validated schema, derived config, source hashes and provenance intact
- protected hashes: all matched
- staged index/worktree hashes: all matched

## Critical child cells
- Probe B `depth=43`: `custom_kernel_nodes=516`, `nonempty_experts=8`, finite loss/gradients
- Probe C one-graph `depth=43`, `experts=256`: `custom_kernel_nodes=66048`, finite loss/gradients
- Probe C sequential `depth=43`, `experts=256`: `custom_kernel_nodes=66048`, finite loss/gradients
- Probe D `full`, `compression_ratio=4`: `custom_kernel_nodes=516`, finite loss/gradients
- Probe D `no_routed`, `compression_ratio=4`: `custom_kernel_nodes=0`, finite loss/gradients
- Probe D `no_shared`, `compression_ratio=4`: `custom_kernel_nodes=516`, finite loss/gradients
- Probe D `no_attention`, `compression_ratio=4`: `custom_kernel_nodes=516`, finite loss/gradients
- Probe D `routed_only`, `compression_ratio=4`: `custom_kernel_nodes=516`, finite loss/gradients

## Tracking and hashes
- tracked helper/test/fixture/report files: yes
- staged files checked for index/worktree equality: 17
- mismatches: 0
- protected hash count: 30
- protected hash mismatches: 0

## Adjudication
- `test-13-3b-5g-r3-tracked-baseline.log` old failure is stale pre-fix evidence
- fresh execution reproduces the green isolation behavior
- authoritative fresh logs: `agent-output/cmux-13-3b/test-13-3b-5g-r3-execute-*.log`
