# Story 13.3b-5f independent review

## Verdict

**FAIL — no real 4096-token smoke authorized.**

Blocking implementation, evidence, and reproducibility defects remain.

## Findings

### R1 — BLOCKER — claimed `598 passed / 18 skipped` baseline not reproducible from tracked files

Independent `pytest --collect-only -q` found 616 tests from 61 files, matching Coder's 598 passed + 18 skipped total. Only 44 contributing files are tracked; 17 are untracked:

- `tests/test_convert_lora_to_ds4.py`
- `tests/test_deepseek_v4_forward_parity.py`
- `tests/test_deepseek_v4_lora_targets.py`
- `tests/test_deepseek_v4_validate_real_mode_relaxation.py`
- `tests/test_ds4_metal_routed_i8_e8m0_isolation.py`
- `tests/test_finetune_ds4_b1_hc_mult_multilayer_readiness.py`
- `tests/test_finetune_ds4_b2_real_checkpoint_payload_readiness.py`
- `tests/test_finetune_ds4_b2_routed_dequant_trusted_reference_readiness.py`
- `tests/test_finetune_ds4_csa_topk_primitive.py`
- `tests/test_finetune_ds4_fuse_hf_gate.py`
- `tests/test_finetune_ds4_fused_gate_perf.py`
- `tests/test_finetune_ds4_i8_dequant_integration.py`
- `tests/test_finetune_ds4_stateful_decode_readiness.py`
- `tests/test_fuse_lora_hf.py`
- `tests/test_make_synth_lora.py`
- `tests/test_real_forward_intermediate_dump.py`
- `tests/test_shim_ds4_safetensors.py`

Project tracking rule explicitly forbids GREEN/PASS when any file contributing to claimed count is untracked. Coder's “full tracked Python suite” claim false.

### R2 — BLOCKER — Metal implementation violates accepted ADR 0028 kernel decomposition

ADR 0028 requires every matrix kernel to use `BM=8`, `BN=8`, `BK=32`, threadgroup FP32 dequant tiles, eight SIMDgroups, weight reuse across eight token rows, and `simd_sum` reduction.

Current source does not implement that design:

- `python-envs/mlx/src/ds4_ft_mlx/metal/ds4_routed_fp4_train.metal:51-64`: `ds4_fp4_pair_swiglu_forward` activates only lane 0 of each SIMDgroup, loops serially over all `H`, has no threadgroup tile, no `BK=32` tile, and no `simd_sum`.
- `python-envs/mlx/src/ds4_ft_mlx/metal/ds4_routed_fp4_train.metal:76-87`: `ds4_fp4_down_forward` likewise loops serially over all `I`, with no tile or SIMD reduction.
- K3–K5 use SIMD reductions but still dequantize directly from device memory per row; no accepted shared `8x32` dequant tile or eight-row reuse exists.

Kernel-marker test at `tests/test_deepseek_v4_nn_routed_fp4_metal.py:64-84` checks names only, so cannot detect this architectural substitution.

Correctness remains plausible; accepted implementation boundary not met.

### R3 — BLOCKER — required fixed-assignment and real-dimension operation-peak proof absent

Tracked memory tests do not implement AC8–AC10:

- `tests/test_deepseek_v4_nn_sparse_routed_backward.py:755-784` tests only tiny E=4/8 cases, not fresh E=2/4/8 at `T=1024,H=1024,I=512,K=2`, and asserts a loose ratio rather than `max(peak)-min(peak) <= 64 MiB`.
- `tests/test_deepseek_v4_nn_sparse_routed_backward.py:791-838` still pins superseded 13.3b-5d math: one full 96 MiB dequantized expert and a 1.25 GiB budget. ADR 0028 requires no full dequant matrix and the assignment-proportional 1504 MiB envelope.
- `tests/test_deepseek_v4_nn_sparse_routed_backward.py:928-941` records `peak_bytes` but gates only final `active_delta`, expressly forbidden by requirements.
- `coder-13-3b-5f-fixed-e-memory.log` contains only three aggregate active/peak lines and spread. No baseline/cache/active/peak samples at every expert boundary; no independently reproducible command or tracked probe.
- Real-dimension timing log reports a small peak but no operation baseline/delta semantics and no tracked `<2 GiB` peak assertion.

Final active memory and aggregate undocumented logs cannot satisfy load-bearing memory gate.

### R4 — BLOCKER — required derivative/opacity test matrix incomplete

Focused 25 tests pass, but required independent acceptance cases absent:

- no exact `u1 == +limit`, `u3 == -limit`, or `u3 == +limit` derivative fixture;
- no isolated simplified-score identity versus prior Q-form test, including duplicate-collapsed denominator;
- no DOT/export assertion proving outer graph exposes `CustomKernel` nodes and no dequant arithmetic;
- no transform test proving captured packed weights/scales receive no cotangent;
- no installed-package resource test;
- no unsupported platform/version fail-closed test;
- no test rejecting forbidden wrapper fallback or dense-materialization symbols.

`tests/test_deepseek_v4_nn_routed_fp4_metal.py` contains only four tests: source markers, ordinary forward parity, ordinary VJP parity, and one malformed-weight guard. Natural-valued VJP fixture does not hit exact clip boundaries.

### R5 — BLOCKER — measured runtime fails smoke-feasibility gate

Independent no-shard benchmark, `H=4096,I=2048,R=96`, five warm repetitions:

- forward p50 `0.218184s`, p95 `0.221010s`;
- input-VJP p50 `0.029911s`, p95 `0.030524s`;
- finite `y`, `dx`, `a`.

Balanced real routing gives approximately 256 experts × 43 layers × 20 iterations. Primitive-only p50 extrapolation:

```text
(0.218184 + 0.029911) * 256 * 43 * 20 = 54,620.6 seconds = 15.17 hours
```

Excludes attention, shared experts, optimizer, validation, and orchestration. ADR architecture requires p50/p95 extrapolation and return to Architect on plainly impractical runtime. Coder log supplied one sample per R and no extrapolation. Missing tiled weight reuse explains forward bottleneck.

### R6 — MAJOR — wrapper fail-closed validation incomplete

`python-envs/mlx/src/ds4_ft_mlx/routed_fp4_metal.py:292-330` validates only score rank/first dimension before constructing assignments. Missing deterministic guards required by architecture:

- score dtype;
- score expert dimension versus `experts.n_routed_experts`;
- `len(rows_by_expert)` versus expert count;
- each expert id within both score and payload bounds;
- finite routed scaling factor;
- unsigned grid-size limits.

Malformed calls can reach indexing/kernel errors instead of deterministic `ValueError`. Marker parser also accepts duplicate marker sections by splitting on first occurrence.

Package metadata declares `mlx>=0.31.2` while runtime hard-requires exactly `mlx==0.31.2`; a fresh resolver may install an intentionally rejected version.

## Positive audit evidence

- Independent focused run: `25 passed` in `64.49s`.
- Independent `make`: exit 0, nothing to rebuild.
- `git diff --check` and `git diff --cached --check`: clean.
- Both directly cited 5f verdict tests tracked:
  - `tests/test_deepseek_v4_nn_routed_fp4_metal.py`
  - `tests/test_deepseek_v4_nn_sparse_routed_backward.py`
- LUT values, LSB-first nibble extraction, BF16 direct scale multiply, projection ordering, strict clamp masks, FP32 outputs, duplicate collapse, shared-expert placement, and simplified score formula appear mathematically correct in source.
- No full dense `w1/w3/w2` MLX output found; work remains `O(R*H*I)`, not `O(R*H²*I)`.
- Package-local Metal source absent from root `metal/`; production Make/runtime references absent.
- Protected hashes match pinned values:
  - `ds4.c` `a9cb4d37d1b5ce34580aec077857bd47f30217ba4e18a36697d86de8e877f957`
  - `ds4_metal.m` `6624500152a779c1201e55c2c56d74221ee4f4b8982ff47d181e90c8b265a1c6`
  - `ds4_cuda.cu` `5a325e571fe05e929de249492d2bf564c37b759ca87d7a1230d97b735a56d727`
  - `ds4_distributed.c` `ad333f0bb30be211d745e9b0c3a391f0f8e8b368f763d8afe2e591de308afb8c`
  - `ds4_ssd.c` `2b2ba6e2278c24196bde916f7149f9c2a9cbc87460d5ce6c421651899f4a4c78`
  - FROZEN `deepseek_v4.py` `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b`
  - ADR 0024 `33dc15ce2d7ed64cb77509cb95ab48300988f83bec5513b6656a644bcef6c751`
- Root `metal/*.metal` manifest matches Coder record.
- Old hash `96c39168c78e5fd9` has zero tracked hits after excluding Coder's self-referential “zero hits” note.
- ADR 0024 content unchanged; ADR 0025 supersession text and ADR 0028 intent consistent.
- Staged chain contains sanctioned 13.3b-5b FROZEN state, three hash-pin advances, ADR 0026/0028, passthrough script, package primitive, integration, and directly cited tests.

## Acceptance audit

| Gate | Result |
|---|---|
| Packed LUT/nibble/BF16 scale/ordering | PASS by source + focused parity |
| Exact boundary derivatives | FAIL — no exact-boundary test |
| Whole sparse dx/score semantics | PARTIAL PASS — ordinary dense parity passes; isolated Q identity absent |
| No frozen cotangents | NEEDS TEST |
| No dense matrices / no quadratic work | PASS structurally |
| ADR 0028 selected tiling | FAIL |
| E=2/4/8 fresh-process operation peaks | FAIL — non-load-bearing evidence |
| Real-dimension operation peak | FAIL — test gates active, not peak |
| 1504 MiB formula | FAIL — tracked test pins stale 1.25 GiB/dequant formula |
| Timing/smoke feasibility | FAIL — ~15.17h primitive-only extrapolation |
| Package/build/runtime isolation | PASS |
| Protected hashes/FROZEN/ADR 0024 | PASS |
| Tracked full baseline | FAIL — 17 contributing files untracked |
| ADR 0025/0028 consistency | PASS |

## Required remediation

1. Track every file contributing to claimed full-suite count, or rerun and report an explicitly tracked-only baseline.
2. Implement accepted `BM=8/BN=8/BK=32` shared dequant tiles and eight-row weight reuse for all matrix kernels; rerun independent parity and p50/p95 extrapolation.
3. Replace stale memory tests with tracked fresh-process E=2/4/8 operation-peak probes, expert-boundary active/cache/peak telemetry, real-dimension operation-peak assertion, and exact 1504 MiB formula.
4. Add exact clamp-boundary, Q-identity, closure-cotangent, DOT opacity, package resource, platform/version, and fallback-rejection tests.
5. Complete deterministic wrapper guards and align package dependency/version policy.
6. Rerun Reviewer and Test Manager. No smoke before both green.
