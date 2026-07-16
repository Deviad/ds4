# Story 13.3b-5f — independent review round 3

## Verdict

**FAIL — no real 4096-token smoke authorized.**

SIMD implementation and reproduced numerical, memory, performance, tracking, hash, and isolation gates pass. One chain-of-custody blocker remains: accepted r3 canonical documentation exists only as unstaged worktree changes, while the staged chain still carries the superseded absolute-only forward contract and omits the r3 SIMD/performance clauses.

## Finding

### R3-1 — BLOCKER — r3 canonical documentation amendments absent from staged chain

`git status --short` reports:

```text
AM docs/adr/0028-opaque-packed-fp4-metal-training-primitive.md
AM docs/backlog.md
AM docs/technical-spec.md
```

Worktree files correctly contain Architect/BA r3 amendments. Index does not:

- staged ADR 0028 still says `Status: Accepted` and `Forward parity remains gated at maximum absolute error <=1e-6`;
- staged backlog AC 2 still says `max_abs <=1e-6` and lacks AC 12–14 SIMD/downstream/performance gates;
- staged technical specification lacks r3 reduction-order, combined-bound, and benchmark clauses.

Direct worktree/index hashes differ:

```text
docs/adr/0028-opaque-packed-fp4-metal-training-primitive.md
  worktree dce498570176f47d5f3b09a6d844f4a861119b1f795982d4e91dc2a6ee7f698b
  index    0644d7d77b08c7bfccc60653a2d1c1dbc40b8650946b3e59d0fe40f4cde87280
docs/backlog.md
  worktree 7a1807b0e554a66341039b08f44aa1b3a222dcf8cf9e58626f2b22a413ad4eb4
  index    f8cd0088b54b6d8663b03c6ac5031778947a66f578462a0e3f4340166fe28d56
docs/technical-spec.md
  worktree c701b0408e77679e36c55fe32df10ce3934cd85ebcb5f009919a03be9f382013
  index    fd9ecf792226738a11e27ed48530d4b26c6999d2cfbac4ab0b111c424c16ea72
```

This violates requested complete staged chain/docs gate. A commit from current index would pair r3 code/tests with stale canonical contract. Stage these three already-correct worktree amendments, then rerun final staging checks and Reviewer.

## Independent audit

### SIMD structure and clamp identity — PASS

- K1–K5 each contain `k0 += 32u`, lane-local FP32 `metal::fma`, and final `simd_sum`.
- K1/K4 each contain two FMA and two `simd_sum` reductions.
- Extracted K1/K4 `u1/u3` reduction prefixes byte-identical: `1543/1543` characters.
- No lane-0 32-wide serial dot loop.
- No hidden/intermediate shape-selected branch.
- K6 unchanged path retained.
- Exact `u1==+limit`, `u3==+limit`, `u3==-limit` sensitivity tests pass.

### Forward, VJP, score, downstream proxy — PASS

Independent no-model reproductions:

```text
primitive forward:
  max_abs 4.54747350886e-13
  max_rel 2.73007401574e-07
  NRMSE   1.21932596115e-07
  combined element-wise bound PASS

whole sparse forward:
  max_abs 1.14440917969e-05
  max_rel 2.36002179577e-07
  NRMSE   1.92528057615e-07
  combined element-wise bound PASS

primitive VJP:
  dx max_abs 4.54747350886e-13
  a  max_abs 3.41060513165e-13

proxy logits:
  max_abs 4.76837158203e-07
  max_rel 9.38367634262e-07
  NRMSE   2.96352542516e-07
  top-1 identical
  minimum reference margin 0.00066948775202

proxy gradients:
  input max_abs 7.45058059692e-09
  score max_abs 2.38418579102e-07
```

Duplicate-collapsed simplified score identity versus independent prior Q form passes at `1e-5`. Actual routed score-gradient proxy passes. Packed LUT, low-nibble-first decode, BF16 direct per-32 scale multiply, projection ordering, strict clamp masks, and FP32 accumulation/output match ADR 0024/0028.

### BF16/FP16, guards, frozen/opacity — PASS

- BF16 input cotangent remains BF16, finite, nonzero.
- FP16 input cotangent remains FP16, finite, nonzero.
- Negative and upper-bound expert ids raise deterministic `ValueError` before indexing/kernel launch.
- Score shape/dtype, rows/expert count, payload first dimension, finite-positive scaling factor, source-marker uniqueness, exact MLX version, platform/arch, Metal availability, GPU-default, shape/dtype, row uniqueness/order, and uint-grid guards present.
- Routed custom function positional primals limited to activation and scores; packed payloads captured constants.
- DOT evidence contains opaque `CustomKernel` nodes and no dequant/GatherQMM nodes.
- No dense dequant fallback, native mxfp4/E8M0 substitution, nested `mx.vjp(forward_one)`, full expert matrix output, or quadratic recomputation found.

### Memory — PASS

Fresh-process tracked helper rerun:

```text
fixed assignments T=1024 H=1024 I=512 K=2 A=2048
E=2 peak delta 52,703,772 bytes
E=4 peak delta 83,227,408 bytes
E=8 peak delta 90,309,428 bytes
spread          37,605,656 bytes <= 64 MiB
```

All x/score gradients finite and nonzero. Every expert non-empty. Per-expert active/cache/peak telemetry emitted.

```text
real dimension H=4096 I=2048 T=512 K=2 E=2
operation peak delta 208,720,484 bytes < 2 GiB
direct y_e [8,4096], dx_e [8,4096], a_e [8]
all direct outputs finite
```

Formula reproduced:

```text
A*H*4       384 MiB
A*I*4       192 MiB
a_partial    24 MiB
forward    1092 MiB
backward   1248 MiB
envelope   1504 MiB < 2 GiB
```

No `E*H*I` operation allocation/retained dequant graph found.

### Performance — PASS

Tracked helper, 3 warm-ups + 25 measured repeats:

```text
R=96 forward p50   0.006462625 s <= 0.0080 s
R=96 input-VJP p50 0.011129041 s <= 0.0140 s
helper extrapolation p50 0.680603 h <= 1.25 h
helper extrapolation p95 0.706169 h <= 1.35 h
```

Conservative forward+input-VJP extrapolation also passes: `1.075828 h` p50 / `1.123958 h` p95.

### Regression, tracking, hashes, isolation — PASS except staged-doc blocker

```text
focused routed+sparse+FP4: 49 passed
tracked-only baseline:     476 passed, 15 skipped, 86 subtests passed
tracked test files:        44
git diff --check:          clean
git diff --cached --check: clean
make:                      exit 0
```

All five load-bearing tests/helpers return non-empty `git ls-files` results.

Protected hashes match accepted pins:

```text
ds4.c            a9cb4d37d1b5ce34580aec077857bd47f30217ba4e18a36697d86de8e877f957
ds4_metal.m      6624500152a779c1201e55c2c56d74221ee4f4b8982ff47d181e90c8b265a1c6
ds4_cuda.cu      5a325e571fe05e929de249492d2bf564c37b759ca87d7a1230d97b735a56d727
ds4_distributed.c ad333f0bb30be211d745e9b0c3a391f0f8e8b368f763d8afe2e591de308afb8c
ds4_ssd.c        2b2ba6e2278c24196bde916f7149f9c2a9cbc87460d5ce6c421651899f4a4c78
FROZEN deepseek_v4.py 5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b
ADR 0024         33dc15ce2d7ed64cb77509cb95ab48300988f83bec5513b6656a644bcef6c751
```

All 19 root `metal/*.metal` files tracked and unchanged. No Story 13.3b-5f symbols in production C/Objective-C/CUDA/distributed/SSD/root-Metal paths. Package-local training Metal absent from production loader/build references. Staged predecessor candidates include sanctioned FROZEN change, hash-pin note advances, ADR 0026 amendment, passthrough chat-template script, ADR 0028, primitive integration, and verdict tests. Old hash `96c39168c78e5fd9` has zero tracked hits outside historical evidence.

## Required remediation

1. Stage current amendments to `docs/adr/0028-opaque-packed-fp4-metal-training-primitive.md`, `docs/backlog.md`, and `docs/technical-spec.md`.
2. Re-run `git diff --cached --check` and inspect staged versions for combined forward bound, SIMD order, downstream proxy, performance gates, and no-smoke policy.
3. Re-run final Reviewer. No smoke before Reviewer PASS plus Test Manager GREEN.
