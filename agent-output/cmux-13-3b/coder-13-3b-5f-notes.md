# Story 13.3b-5f — Coder notes through r2 fixes

## Summary

Addressed `review-13-3b-5f-r2.md` without full model, shards, dataset, or 4096-token smoke. R2 fixes cover BF16/FP16 activation promotion, removal of permanent small-shape serial Metal variants, load-bearing derivative/score/opacity tests, deterministic expert-id guards, score-inclusive memory probes, and a tracked benchmark helper.

## RED authority

Added RED tests first, then verified they failed before implementation:

- `test_forward_one_promotes_half_activation_without_inconsistent_reject` failed with `ValueError: x_flat must be float32 before packed_fp4_forward_one`.
- `test_matrix_kernels_structurally_use_8x8x32_tiles_and_simd_reductions` failed on the `uint(hidden_size) <= 32u` serial branch.
- `test_routed_outer_transform_freezes_payloads_and_returns_x_score_cotangents` and BF16/FP16 `SparseMoeBlockNN` tests failed on `x_flat must have dtype mlx.core.float32`.
- Expert-id guard tests failed because `-1` silently aliased and `E` did not raise deterministic `ValueError` at host materialization.

## Implementation fixes

- `routed_fp4_metal.py` now accepts FP32/FP16/BF16 activations and performs explicit `x32 = x_flat.astype(mx.float32)` before the routed custom-function boundary. The cast stays outside `mx.custom_function`, so ordinary MLX autodiff carries cotangents back to the upstream activation dtype.
- `packed_fp4_forward_one` now uses the same activation-promotion path and no longer validates a temporary cast then rejects the original half input.
- `ds4_routed_fp4_train.metal` no longer contains `hidden_size <= 32` or `intermediate_size <= 32` serial overwrite branches. All five matrix kernels keep the single BM8/BN8/BK32 tiled decomposition. Reductions now use per-row tile scratch and lane-0 deterministic accumulation inside each 32-wide tile, preserving small-shape parity without a shape-selected semantic variant.
- `_host_unique_rows_per_expert` now validates every host-materialized route id and raises `ValueError("route expert id ... out of range ...")` before any expert indexing or kernel launch.
- Added load-bearing tests for exact `u1==+limit`, `u3==+limit`, and `u3==-limit` boundary sensitivity; duplicate-collapsed score derivative equivalence against an independent Q-form; routed outer-transform frozen payloads with x/score gradients; routed DOT opacity; BF16/FP16 `SparseMoeBlockNN` forward plus first-order cotangents; valid-boundary/duplicate route ids; and tracked benchmark helper metadata.
- `tests/helpers/routed_fp4_memory_probe.py` now measures `value_and_grad` over `{x, scores}`, asserts finite/non-zero x and score gradients, reports operation peak, and directly exercises real-dimension `y_e/dx_e/a_e` shapes.
- Added and staged `tests/helpers/routed_fp4_bench.py`, a tracked R=1/8/32/96 p50/p95 harness with reproducible command and `256x43x20` extrapolation.

## Memory evidence

Tracked helper rerun:

- Fixed `T=1024,H=1024,I=512,K=2,A=2048` operation peak deltas:
  - E=2: `52,703,768` bytes.
  - E=4: `83,227,408` bytes.
  - E=8: `90,309,428` bytes.
  - Spread: `37,605,660` bytes, below 64 MiB.
- All fixed-E x and score gradients were finite and non-zero; every expert boundary remained non-empty.
- Real-dimension no-shard `H=4096,I=2048,T=512,K=2,E=2`: operation peak delta `208,720,484` bytes, below 2 GiB; x and score gradients finite/non-zero.
- Direct real-dimension one-expert shapes: `y_e=[8,4096]`, `dx_e=[8,4096]`, `a_e=[8]`, all finite.

## Performance evidence

Tracked helper command:

```bash
python-envs/mlx/.venv/bin/python tests/helpers/routed_fp4_bench.py --repeats 25
```

Result, no model/shard:

- R=1: forward p50 `0.002505s`, p95 `0.004280s`; input-VJP p50 `0.004145s`, p95 `0.004317s`.
- R=8: forward p50 `0.002546s`, p95 `0.002641s`; input-VJP p50 `0.004138s`, p95 `0.004378s`.
- R=32: forward p50 `0.007608s`, p95 `0.007703s`; input-VJP p50 `0.012879s`, p95 `0.013415s`.
- R=96: forward p50 `0.021484s`, p95 `0.022180s`; input-VJP p50 `0.036498s`, p95 `0.037210s`.
- R=96 primitive-only extrapolation for `256 experts × 43 layers × 20 iterations`: p50 `2.232h`, p95 `2.276h`.

## Validation

- R2 RED-focused subset before fixes: `7 failed` as expected.
- R2 focused acceptance subset after fixes: `14 passed`.
- Routed primitive suite: `21 passed`.
- Sparse routed backward suite: `26 passed`.
- Combined routed+sparse focused suites: `47 passed`.
- FP4 parity regression: `1 passed`.
- Tracked-only full Python baseline from `git ls-files 'tests/test*.py'`: `475 passed, 15 skipped, 86 subtests passed` from `45` tracked test files.
- `make`: `Nothing to be done for all`.
- `git diff --check` and `git diff --cached --check`: clean.
- Root production C/Objective-C/Metal files have zero hits for `routed_fp4`, `ds4_routed_fp4_train`, `packed_fp4`, or `mxfp4`.

## Tracking and staging

R2 load-bearing files are tracked/staged, including:

- `python-envs/mlx/src/ds4_ft_mlx/metal/ds4_routed_fp4_train.metal`
- `python-envs/mlx/src/ds4_ft_mlx/routed_fp4_metal.py`
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py`
- `tests/helpers/routed_fp4_bench.py`
- `tests/helpers/routed_fp4_memory_probe.py`
- `tests/test_deepseek_v4_nn_routed_fp4_metal.py`
- `tests/test_deepseek_v4_nn_sparse_routed_backward.py`

No commit performed.

---

# Story 13.3b-5f — Coder r3 SIMD reduction re-pin

## Summary

Restored the one-path ADR 0028 SIMD reduction order for K1-K5. K6 unchanged. No full model, shards, dataset, 4096-token smoke, or commit.

## RED / authority

- R2 lane-0 performance evidence was the RED authority: R=96 forward p50 `0.021484s`, input-VJP p50 `0.036498s`, 20-iteration primitive extrapolation p50 `2.232h` / p95 `2.276h`, exceeding Architect r3 gates.
- Tracked baseline initially failed after SIMD restore at `tests/test_deepseek_v4_nn_fp4_parity.py::test_sparse_moe_fp4_forward_matches_frozen_moe_mlx_with_same_weights` because stale absolute-only whole-MoE tolerance rejected SIMD-equivalent `max_abs=1.1444091796875e-05`.

## Implementation fixes

- `ds4_routed_fp4_train.metal`: K1-K5 now use lane-local FP32 `metal::fma` accumulation over coordinates `lane,lane+32,...`, followed by `simd_sum`; lane 0 alone writes. K6 reduction left unchanged.
- Removed lane-0 32-wide serial dot loops and per-row value scratch tiles from K1-K5; retained BM8/BN8/BK32 tiles, eight SIMDgroups, and one semantic path for every shape.
- K1 and K4 share the same source-load-bearing `u1/u3` lane/FMA/`simd_sum` skeleton before clamp decisions.
- Forward parity tests now use the canonical combined bound `abs(got-ref) <= 2e-6 + 1e-6*abs(ref)` plus NRMSE `<=1e-6`, with max absolute and max relative diagnostics.
- Added tracked downstream no-model proxy coverage: fixed rows-by-expert, nonuniform cotangent, fixed 32→7 projection, projected-logit combined bound + NRMSE, top-1 identity with nonzero reference margin, and x/score gradient parity at `1e-5`.
- `tests/helpers/routed_fp4_bench.py` now performs 3 warm-up calls before 25 measured repeats and records warmup/sample counts.

## Performance evidence

Command:

```bash
python-envs/mlx/.venv/bin/python tests/helpers/routed_fp4_bench.py --repeats 25
```

Result, no model/shard:

- R=1: forward p50 `0.001030s`, p95 `0.001626s`; input-VJP p50 `0.001608s`, p95 `0.002274s`.
- R=8: forward p50 `0.001012s`, p95 `0.001161s`; input-VJP p50 `0.001634s`, p95 `0.002276s`.
- R=32: forward p50 `0.002514s`, p95 `0.002601s`; input-VJP p50 `0.004341s`, p95 `0.004469s`.
- R=96: forward p50 `0.006165s` <= `0.0080s`; input-VJP p50 `0.010857s` <= `0.0140s`.
- R=96 primitive-only extrapolation for `256 experts × 43 layers × 20 iterations`: p50 `0.663961h` <= `1.25h`; p95 `0.674759h` <= `1.35h`.
- Samples: 25 measured repeats after 3 warm-ups.

## Validation

- Focused SIMD/proxy subset: `4 passed`.
- Routed primitive + sparse routed backward suites: `48 passed`.
- FP4 whole-MoE parity regression: `1 passed` after canonical combined contract update.
- Tracked-only full Python baseline from `git ls-files 'tests/test_*.py'`: `476 passed, 15 skipped, 86 subtests passed`.
- `make`: `Nothing to be done for all`.
- `git diff --check`: clean.
- Structural source check: K1-K5 each contain `simd_sum`; no `for (uint c = 0u; c < 32u; ++c)` lane-0 serial dot loop; no hidden/intermediate shape-selected branch.
- Documentation check: `docs/backlog.md`, ADR 0028, and `docs/technical-spec.md` contain the r3 combined parity/SIMD/performance contract.

## Tracking and staging

R3 load-bearing source/test/helper files are tracked and staged after this note update:

- `python-envs/mlx/src/ds4_ft_mlx/metal/ds4_routed_fp4_train.metal`
- `tests/helpers/routed_fp4_bench.py`
- `tests/test_deepseek_v4_nn_fp4_parity.py`
- `tests/test_deepseek_v4_nn_routed_fp4_metal.py`
- `tests/test_deepseek_v4_nn_sparse_routed_backward.py`
- `agent-output/cmux-13-3b/coder-13-3b-5f-notes.md`

R3 log/json artifacts are written under `agent-output/cmux-13-3b/coder-13-3b-5f-r3-*` but left unstaged to keep `git diff --cached --check` clean.

No commit performed.
