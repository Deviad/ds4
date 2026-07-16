# Story 13.3b-5f — independent review round 2

## Verdict

**FAIL — no real 4096-token smoke authorized.**

Round-1 memory, tiling, performance, baseline-tracking, hash, and package-version findings materially improved. Blocking contract and test-legitimacy defects remain.

## Findings

### R2-1 — BLOCKER — routed integration rejects BF16/FP16 instead of performing ADR 0028's explicit FP32 promotion

ADR 0028 and accepted architecture require `x32 = x_flat.astype(mx.float32)` before the routed custom function so ordinary MLX autodiff carries the cotangent back to the upstream activation dtype.

Current code does the opposite:

- `routed_fp4_metal.py:325-327` requires `x_flat` already be FP32 and assigns `x32 = x_flat`.
- `packed_fp4_forward_one:197-202` validates a temporary cast, then rejects the original non-FP32 array.
- `deepseek_v4_nn.py:535-546` forwards the model activation without promotion.

Independent reproduction through `SparseMoeBlockNN` with BF16 input:

```text
ValueError x_flat must have dtype mlx.core.float32, got mlx.core.bfloat16
```

Tiny tests use FP32 only, so they do not expose this real trainer-path incompatibility. Restore the architecture's explicit pre-custom-function cast and test BF16/FP16 forward plus first-order cotangent propagation.

### R2-2 — BLOCKER — unconditional shape-based serial overwrite violates ADR 0028's single tiled decomposition

The five matrix kernels now contain genuine `BM=8/BN=8/BK=32` threadgroup tiles, eight SIMDgroups, `simd_sum`, and eight-row weight reuse. Real `H=4096/I=2048` execution uses that design.

But two kernels then unconditionally overwrite tiled results with a second serial implementation:

- `ds4_routed_fp4_train.metal:90-105` when `hidden_size <= 32`.
- `ds4_routed_fp4_train.metal:142-152` when `intermediate_size <= 32`.

This is a permanent shape-selected semantic implementation, not diagnostic instrumentation. ADR 0028 says every matrix kernel uses the selected tiled decomposition and leaves no alternative open. More importantly, small-shape forward uses serial reduction order while `pair_swiglu_vjp_terms` recomputes `u1/u3` with tiled SIMD reduction order. The custom VJP can therefore make clamp decisions from a different reduction than the forward it differentiates.

Adjudication: no full dense matrix, nested VJP, `O(R*H²I)` work, or real-dimension memory regression observed; production `4096/2048` dimensions do not enter this branch. Nevertheless it violates the accepted one-path/no-permanent-variant contract and weakens exact small-shape forward/VJP consistency. Remove it or return to Architect for an explicit ADR amendment and a proof of derivative consistency.

### R2-3 — BLOCKER — new exact derivative/score/frozen/opacity tests are not load-bearing

1. `test_input_vjp_exact_clip_boundaries...` does not test the `u1 == +limit` mask. Its `u1` case sets `w3_code=0x00`, making `u3`, `up`, hidden, `a`, and the unmasked `dgate` all zero. Independent fixture inspection:

   ```text
   u1_unique [1.] u3_unique [0.] up_nonzero 0
   dgate_unmasked_nonzero 0 hidden_nonzero 0 a_ref [0.]
   ```

   The test passes even if the `u1` equality mask is wrong.

2. `test_simplified_score_identity...:267-280` is algebraically tautological. It compares `rsf*a/D-common/D` with `(rsf*a-common)/D`, supplies arbitrary `common` unrelated to scores/outputs, and keeps duplicate `(token, expert)` assignments instead of testing the primitive's duplicate-collapsed plan. It does not independently reproduce the prior Q-form derivative.

3. No transform test requests gradients while proving packed weights/scales have no cotangent. `test_source_and_api...` performs source-string checks only.

4. The DOT test exports `packed_fp4_forward_one`, not `routed_fp4` under the outer `mx.grad`/`mx.value_and_grad` boundary. It therefore does not prove the accepted outer-transform opacity claim.

Whole sparse input/gate parity remains useful and green, but it does not replace these explicitly required independent contract tests.

### R2-4 — MAJOR — expert-id fail-closed guard remains incomplete

`_host_unique_rows_per_expert` indexes `rows_by_expert[eid]` without validating `0 <= eid < n_experts` (`deepseek_v4_nn.py:449-456`). Independent reproduction:

```text
indices=-1  -> [[0], [0]]       # silently aliases last expert
indices=2,E=2 -> IndexError: list index out of range
```

ADR architecture requires deterministic `ValueError` before kernel launch for out-of-range expert ids. Negative hash-table ids currently misroute silently.

### R2-5 — MAJOR — tracked load-bearing probes still omit required assertions

The tracked memory helper measures `mx.grad(...)(x)` only, not a score/gate cotangent, and does not assert finiteness. Its real-dimension case evaluates routed `y/gx`, not direct real-dimension `y_e/dx_e/a_e`, despite claiming those outputs.

Independent score-inclusive probes pass and support the implementation's memory shape:

```text
E=2/4/8 value_and_grad(x,scores) peak deltas:
75,485,868 / 79,033,116 / 86,115,136 bytes
spread: 10,629,268 bytes
all finite; score gradients non-zero

H=4096,I=2048,T=512,E=2 value_and_grad(x,scores):
peak delta: 200,331,888 bytes
all finite; score gradients non-zero
```

Direct `R=8,H=4096,I=2048` `y_e/dx_e/a_e` were finite with expected shapes. These independent results are positive, but the tracked acceptance tests should carry the same load-bearing assertions.

### R2-6 — MAJOR — performance harness remains untracked

Performance was independently reproduced from `agent-output/cmux-13-3b/bench_routed_fp4_tiled.py`, but that Python harness is not returned by `git ls-files`. Fresh-clone reproduction therefore lacks the exact benchmark code.

Independent rerun:

```text
R=96 forward p50/p95: 0.006261s / 0.006789s
R=96 input-VJP p50/p95: 0.010788s / 0.011038s
256x43x20 extrapolation: 1.043h p50 / 1.090h p95
```

Performance is materially improved from round 1 and no longer the former ~15.17h primitive-only blocker. Track the harness or encode its exact method in a tracked test/tool before using it as chain-of-custody evidence.

## Positive audit

- Genuine tiled real-dimension kernels confirmed: `8x32` threadgroup dequant tiles, `8x8` outputs, first-SIMDgroup loads, eight-row reuse, `BK=32`, and SIMD reductions in all five matrix kernels.
- No full FP32 expert matrix, `mx.vjp(forward_one)`, dense fallback, native mxfp4/E8M0 substitution, or quadratic nested VJP found.
- Independent ordinary packed parity:

  ```text
  forward max_abs 4.547473508864641e-13
  dx max_abs      4.547473508864641e-13
  a max_abs       3.410605131648481e-13
  ```

- Focused rerun: `39 passed`.
- Tracked-only baseline independently reproduced from all 45 `git ls-files 'tests/test*.py'` entries: `467 passed, 15 skipped, 86 subtests passed`.
- Coder helper E=2/4/8 operation-peak deltas independently reproduced: `94,049,004 / 99,357,588 / 109,954,088` bytes; spread `15,905,084` bytes.
- Coder helper real-dimension peak delta independently reproduced: `253,814,944` bytes, below 2 GiB.
- Formula reproduced: 384 MiB assignment-H, 192 MiB assignment-I, 24 MiB a-partial, 1504 MiB envelope.
- `make`, staged/unstaged `git diff --check`: clean.
- Protected production hashes, FROZEN SHA `5e11a9c4...`, ADR 0024 SHA `33dc15ce...`, package/root-Metal isolation: pass.
- Staged predecessor chain includes sanctioned FROZEN fix, three hash-pin advances, ADR 0026 amendment, passthrough script, ADR 0028, package primitive, integration, and verdict tests.

## Round-1 closure matrix

| Round-1 item | Round-2 result |
|---|---|
| Tracked-only baseline | PASS |
| True real-dimension 8x8x32 tiling/row reuse | PASS, but small-shape serial variant blocks contract |
| Performance p50/p95 | Numeric PASS; harness untracked |
| Fixed E=2/4/8 operation peaks | PASS independently |
| Real-dimension peak/formula | PASS independently |
| Exact derivative/score tests | FAIL — vacuous/tautological fixtures |
| Frozen cotangent transform test | FAIL — source assertion only |
| Outer opacity test | FAIL — wrong graph boundary |
| Wrapper guards/version pin | PARTIAL — version pin pass; dtype cast and eid guards fail |
| Chain/hashes/docs | Hash/chain PASS; canonical primitive intent retained |

## Required remediation

1. Implement ADR 0028's explicit FP32 promotion before the custom-function boundary; test BF16/FP16 activation and cotangent flow.
2. Remove serial overwrite branches, or obtain Architect/ADR re-pin with exact forward/VJP consistency evidence.
3. Replace vacuous `u1` boundary and tautological score-identity tests with independent, sensitivity-proven fixtures.
4. Add actual outer-transform frozen-cotangent and routed-custom-function DOT opacity tests.
5. Validate every route expert id before indexing; negative and upper-bound cases must raise deterministic `ValueError`.
6. Make tracked memory tests score-inclusive and finite for real `y_e/dx_e/a_e`; track the timing harness.
7. Rerun Reviewer and Test Manager. No smoke before both independently green.
