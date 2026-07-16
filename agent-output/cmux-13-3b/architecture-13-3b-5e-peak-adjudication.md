# Story 13.3b-5e — E-scaled transient peak adjudication

**Decision: STOP Path A.** The current MLX 0.31.2 Python composition cannot provide the required one-expert lifetime bound inside the trainer's outer gradient transform. `mx.eval`, `mx.stop_gradient`, `del`, and `mx.clear_cache` do not release the transformed custom-VJP graph between experts. No bounded correction using the current `forward_one` + nested `mx.vjp` composition is authorized. No real smoke is authorized.

Path A may reopen only around an opaque lower-level routed FP4 forward/input-VJP primitive that never exposes full dequantized expert matrices as MLX arrays. This is not a semantic approximation: routing, FP4 E2M1/BF16-scale math, SwiGLU/clamp behavior, score derivatives, and token-input derivatives remain exact.

No production file, model shard, or full model was touched for this adjudication.

## 1. Evidence read

Read in full:

- `agent-output/cmux-13-3b/task-architect-13-3b-5e-peak-adjudication.md`
- `agent-output/cmux-13-3b/review-13-3b-5d.md`
- `agent-output/cmux-13-3b/coder-13-3b-5d-stop.md`
- `agent-output/cmux-13-3b/requirements-13-3b-5d-routed-fp4.md`
- `agent-output/cmux-13-3b/architecture-13-3b-5d-routed-fp4-backward.md`
- `agent-output/cmux-13-3b/coder-13-3b-5d-notes.md`
- current `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py`
- tracked `tests/test_deepseek_v4_nn_sparse_routed_backward.py`

Also inspected the exact FROZEN `_dequantize_fp4_block_scale_mlx` function, the accepted ADR 0025 amendment, and the canonical backlog entry to define the lower-level boundary and documentation impact. The sparse verdict test is tracked by `git ls-files`.

## 2. Probe legitimacy

All new measurements used fresh Python processes and the project MLX environment:

```text
/Volumes/Data NVME/mlx-ft/ds4/.venv/bin/python
MLX 0.31.2
compile disabled explicitly with mx.disable_compile()
no model shards
no full model
K=2, T=1024, H=1024, I=512
E=2/4/8
hash routes: [t % E, (t + 1) % E]
```

This route construction gives exactly `T*K=2048` unique assignments in every case and makes every expert non-empty:

```text
E=2:  R_e=1024 each, sum R_e=2048
E=4:  R_e=512  each, sum R_e=2048
E=8:  R_e=256  each, sum R_e=2048
```

Before the measured operation, the probe evaluated `x`, `input_ids`, `tid2eid`, gate weights, all packed FP4 payloads, and all BF16 scales. It then called `mx.clear_cache()`, `gc.collect()`, recorded active/cache baselines, and called `mx.reset_peak_memory()`. Thus payload construction and first materialization are outside the measured peak while payload residency remains in the active baseline.

The independent replay produced:

```text
E=2 peak_delta=561,156,582
E=4 peak_delta=755,973,028
E=8 peak_delta=1,156,220,712
```

The locked values are exactly `23,068,672` bytes higher for every E:

```text
E=2 locked=584,225,254
E=4 locked=779,041,700
E=8 locked=1,179,289,384
```

Therefore the E-dependent component reproduced byte-for-byte:

```text
E=2 -> E=4: 97,408,223 bytes per added expert
E=4 -> E=8: 100,061,921 bytes per added expert
```

The constant offset is fixture-scope residency, not an expert-scaling effect.

`mx.set_cache_limit(0)` produced the same peaks as the default cache policy. This independently rules out free-cache accumulation as the cause of the high-water growth.

## 3. Per-phase telemetry

Telemetry wrapped the current implementation's calls without changing production code. Values below are deltas from the materialized-payload active baseline. `cache=0` after each explicit clear is load-bearing evidence: these buffers are still active, not merely cached.

### 3.1 Detailed E=2 trace

```text
phase / expert                    active delta     cache          peak delta
baseline after reset                       0          25                   0
forward accumulator clear e0      44,167,228           0          59,404,389
forward accumulator clear e1      84,054,104           0         103,485,565
Q update clear e0                167,268,537           0         175,648,953
Q update clear e1                229,462,258           0         237,846,770
score y/dscore eval e0           300,077,355           0         300,077,355
direct VJP graph built e0        291,697,000   8,388,608         300,077,355
direct dx eval e0                384,815,472  12,582,912         397,398,384
dx/dscores accumulator clear e0  393,224,560           0         397,398,384
direct VJP graph built e1        455,455,198   8,388,608         463,839,657
direct dx eval e1                548,573,670  12,582,912         561,156,582
dx/dscores accumulator clear e1  552,780,262           0         561,156,582
top-level grad graph returned     552,780,270           0         561,156,582
after final mx.eval(g)              8,421,384 544,358,886         561,156,582
```

`mx.stop_gradient` leaves active bytes unchanged at each measured barrier. It is documented as an identity that prevents gradient flow; MLX does not document it as a graph-lifetime or buffer-detachment primitive.

### 3.2 Phase-end active deltas across E

```text
E=4
forward clears: 29,507,668; 54,694,000; 79,880,332; 105,066,664
Q clears:      171,497,753;216,899,922;262,302,091;307,704,260
grad clears:   422,412,866;532,906,680;643,400,494;753,894,308
final eval:      8,454,152 active; 745,440,164 cache; peak 755,973,028

E=8
forward clears: 22,239,368;40,075,424;57,911,484;75,747,544;
                93,583,604;111,419,664;129,255,724;147,091,784
Q clears:      205,143,513;242,149,906;279,156,299;316,162,692;
               353,169,085;390,175,478;427,181,871;464,188,264
grad clears:   554,394,598;640,369,756;726,344,914;812,320,072;
               898,295,230;984,270,388;1,070,245,546;1,156,220,704
final eval:      8,519,688 active; 1,147,701,024 cache; peak 1,156,220,712
```

At E=8, active memory rises after every expert despite `mx.eval`, `mx.stop_gradient`, cache clear, and local deletion. Only the final evaluation of the complete top-level gradient releases it. Final flat active memory therefore does not validate the operation peak.

## 4. Root cause and live objects

The root cause is **custom-function transform capture under the outer `mx.grad`/trainer `mx.value_and_grad` transformation**.

Current execution creates four `forward_one` graph instances for every non-empty expert:

1. routed custom-function forward;
2. VJP Q pass;
3. VJP score-output recomputation;
4. nested direct-input `mx.vjp` recomputation.

Each `forward_one` graph contains three FP32 dequant results plus gather, clamp, SwiGLU, matmul, and output nodes. The gradient phase additionally creates the nested VJP tape and cotangent graph. The transformed parent graph retains these expert subgraphs until the complete outer gradient is evaluated.

Exact attribution:

- **Expert dequant arrays:** retained through repeated `forward_one` transformed graphs; fixed `H*I` cost per expert.
- **Q graph:** retained cumulatively; E=8 Q phase grows from 147.1 MB to 464.2 MB active.
- **Nested VJP tape:** largest remaining increment; E=8 gradient phase grows from 464.2 MB to 1,156.2 MB active.
- **Accumulator graphs:** each stopped/evaluated scatter accumulator still belongs to the transformed parent graph; stopping gradient flow does not detach its storage lifetime.
- **Packed payloads/scales:** materialized before the baseline and resident in it; not the measured E-dependent cause. Their dequantized products are the problem.
- **Allocator cache:** not the cause. Cache is zero after each explicit clear, and disabling the cache does not change peak.

Two isolation probes confirm the boundary:

```text
ordinary forward only, peak delta:
E=2 53,035,085; E=4 39,397,461; E=8 32,590,949

one direct expert mx.vjp at a time outside an outer transform, peak delta:
E=2 86,020,145; E=4 53,774,385; E=8 37,651,505
```

Both are bounded by `R_e` and do not grow with E. The same operations become cumulative only when embedded in the custom VJP used by the outer gradient transform.

## 5. Memory bound and gate adjudication

### 5.1 Measured fixed-shape extrapolation

Least-squares fit of the locked fixed-`K*T` peaks is:

```text
P_small(E) = 384,101,412 + 99,303,721.6 * E bytes
```

Residuals are only `+1.52/-2.27/+0.76 MB`. At E=256 this predicts `25,805,854,134` bytes (`24.03 GiB`) even at the small `H=1024,I=512` dimensions.

The real expert's `H*I` is 16 times larger. Anchoring at the measured real-dimension E=2 peak (`4,076,454,246` bytes) and scaling only the measured per-expert coefficient by 16 gives a stress extrapolation:

```text
P_real(256) ~= 407,646,778,712 bytes = 379.65 GiB
```

That is routed-operation peak alone. It already exceeds the 340 GB gate and exceeds the 400,000,000,000-byte scheduler limit.

This is an extrapolation, not a claimed exact allocation trace. The structural floor below reaches the same decision without relying on the full fitted coefficient.

### 5.2 Structural floor

At real dimensions one evaluated expert dequantizes exactly 96 MiB:

```text
(w1 + w3 + w2) = 3 * H * I * 4 = 96 MiB
```

The current transformed path creates four such graph instances per non-empty expert. If those materialized dequant results remain live as the phase telemetry demonstrates, their floor is:

```text
4 * 96 MiB * 256 = 96 GiB
```

Replacing the prior architecture's 1.25 GiB routed allowance inside its 300 GiB whole-process design ceiling gives:

```text
300 - 1.25 + 96 = 394.75 GiB = 423.86 GB decimal
```

This exceeds both the 340 GB stop gate and the 400 GB decimal scheduler limit before additional variance.

Even optimistic Python rearrangements do not restore credible margin:

```text
remove Q pass, three dequant graph instances/expert:
72 GiB floor; design envelope 370.75 GiB = 398.09 GB decimal

hypothetical two-instance expert factory:
48 GiB floor; design envelope 346.75 GiB = 372.32 GB decimal
```

Both exceed the 340 GB gate. The three-instance case leaves less than 2 GB decimal under the scheduler before unmodelled variance. Actual measured expert-factory behavior still grows with E, so the two-instance line is an optimistic floor, not an achievable current design.

**Gate verdict:** the 400 GB/340 GB gates are not credible for the current Python custom-VJP path. Do not relax the gate based on flat final active memory.

## 6. Candidate adjudication

| Candidate | Verdict | Evidence |
|---|---|---|
| More `mx.eval` / `mx.stop_gradient` / `del` / cache clears | Reject | Already present. Active phase-end bytes increase after every expert while cache is zero. MLX 0.31.2 provides no lifetime-detach guarantee. |
| Phase-isolated or expert-level custom functions | Reject | Light expert-factory probe still grew: E=2 `278,470,790`, E=4 `341,975,304`, E=8 `478,421,516` peak bytes, despite a reduced loss without the score path. Transform capture moves boundaries but remains cumulative. |
| Host-materialized Q only | Reject | First-order Q could be treated as a computed cotangent constant, but removing Q does not remove the direct-input nested VJP tape or forward/score expert graphs. Host round-tripping `dx`/activations would violate the host-metadata boundary and add large CPU transfers. |
| Score derivative rearrangement using routed output | Exact but insufficient | `dscore_e = rsf*a_e/D' - dot(g, routed)/D'` eliminates the Q pass exactly because `routed=rsf*N/D'`. It still leaves at least three dequant graph instances per expert and an envelope above the 340 GB gate. Keep this algebra for the lower-level design. |
| Route chunks | Reject | Chunking changes scheduling granularity, not the outer transformed graph lifetime. All chunk subgraphs remain part of one top-level gradient until final evaluation. No E-independent bound results. |
| Existing `mx.gather_mm` / `mx.gather_qmm` | Reject | `gather_mm` requires dense matrices; `gather_qmm(mode="mxfp4")` requires E8M0 scales, while ADR 0024 requires BF16 linear-domain scales. |
| Opaque custom Metal/MLX primitive | **Required to reopen Path A** | Can consume packed FP4/scales directly, keep dequant tiles internal, expose only assignment-proportional outputs/cotangents, and define an exact first-order VJP. |

## 7. Minimal lower-level primitive required

The minimal reopening boundary is a training-only, one-expert-at-a-time opaque primitive with two operations:

```text
routed_fp4_forward_one(
    x_e[R,H], w1/w2/w3 packed uint8, s1/s2/s3 BF16
) -> y_e[R,H]

routed_fp4_input_vjp_one(
    x_e[R,H], g_e[R,H], f_e[R],
    w1/w2/w3 packed uint8, s1/s2/s3 BF16
) -> dx_e[R,H], a_e[R]
```

Required mechanics:

- decode OCP E2M1 nibbles LSB-first;
- multiply BF16 linear-domain scale per 32 logical inputs;
- tile dequantization inside the Metal/MLX primitive; never return or retain full FP32 `w1/w2/w3` arrays;
- reproduce current `clip`, sigmoid-SwiGLU, and matrix products exactly enough for existing `1e-6` forward and `1e-5` VJP contracts, including clamp-boundary behavior;
- `a_e = rowwise_dot(g_e, y_e)` for the exact normalized-score derivative;
- `dx_e` uses `f_e*g_e` and returns no weight/scale cotangents;
- routing metadata remains detached and host-materialized exactly as now;
- use `dscore_e = rsf*a_e/D' - dot(g,routed)/D'` to remove the Q recomputation pass;
- expose the primitive as an opaque custom-VJP leaf so the outer transform can retain at most assignment-proportional `x_e/y_e/dx_e`, not `E*H*I` dequant graphs.

A practical project boundary is a new Metal source under `metal/`, a narrow MLX wrapper under `python-envs/mlx/src/ds4_ft_mlx/`, and one call-site change in the training sibling. If `mx.fast.metal_kernel` cannot provide the necessary opaque transform/lifetime behavior, this requires an upstream/native MLX primitive with a registered VJP. Do not reintroduce C++ into this project merely to bypass that limitation.

Expected real-shape order:

```text
sum_e R_e * H * 4 = T*K*H*4 = 384 MiB
```

The design target is `<=1.5 GiB` routed-operation peak; the hard no-shard acceptance gate remains `<2 GiB`. At a 2 GiB routed allowance, the prior conservative process envelope becomes about `300.75 GiB` (`322.93 GB decimal`), below the 340 GB gate. This is a target requiring measurement, not current GREEN evidence.

## 8. Future file/test map — not authorized by this STOP

Future BA re-pin and architecture approval would permit only:

- `metal/ds4_routed_fp4_train.metal` — new exact packed-FP4 forward/input-VJP kernels.
- `python-envs/mlx/src/ds4_ft_mlx/routed_fp4_metal.py` — narrow kernel wrapper and shape/dtype guards.
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py` — training sibling dispatch only; remove the current nested `forward_one` VJP composition from the sparse routed path.
- `tests/test_deepseek_v4_nn_routed_fp4_metal.py` — primitive parity, transform, and memory tests.
- `tests/test_deepseek_v4_nn_sparse_routed_backward.py` — retain all route/score/shared/LoRA contracts; strengthen the memory verdict around the opaque primitive.
- `docs/adr/0025-real-trainable-nn-module-deepseek-v4-port.md` plus a new durable ADR if the lower-level primitive changes the accepted boundary.
- `docs/backlog.md` and `docs/technical-spec.md` — BA re-pin and revised double-green gate.

Still forbidden: FROZEN `deepseek_v4.py`, ADR 0024 semantics, production inference, SSD streaming, CUDA, distributed inference, default Metal inference, model shards, datasets, site-packages, approximations, token dropping, capacity limits, or changed normalization.

## 9. RED-to-GREEN acceptance criteria for any future reopening

1. **Current-path RED:** fixed `K=2,T=1024,H=1024,I=512`, all experts non-empty, fresh E=2/4/8 processes reproduce material E-scaled operation peak before the old implementation is removed.
2. **Primitive forward parity:** deterministic nonuniform/distinct expert fixtures match the existing ADR 0024 `forward_one` reference with maximum absolute error `<=1e-6`.
3. **Primitive input VJP parity:** ordinary MLX reference and primitive `dx_e` match at `atol=rtol=1e-5`, including positive/negative clip boundaries and non-zero finite leaves.
4. **Whole sparse parity:** input and gate/score gradients match the independent dense reference at `atol=rtol=1e-5`; score-rearrangement parity is tested independently against the Q formula.
5. **Routing invariants:** learned/hash routes, stable ties, duplicate collapse, correction-bias selection-only behavior, token order, empty-expert skip, and one shared-expert call remain unchanged.
6. **No dense dequant exposure:** source/API tests prove the primitive consumes packed weights/scales and never returns full dequantized matrices. No native E8M0 substitution.
7. **Fixed-assignment memory GREEN:** fresh-process E=2/4/8 telemetry records active/cache/peak at every expert boundary. At fixed `T*K`, peak may vary with `max R_e` and small graph metadata but must not have an `E*H*I` slope. Target `max(peak)-min(peak) <=64 MiB` on the small probe.
8. **Real-dimension no-shard GREEN:** light H=4096/I=2048 synthetic probes remain `<2 GiB` operation peak and finite, with a formula-backed E=256/K=6/T=4096 bound `<=2 GiB`. No 256-expert real payload allocation is required.
9. **Regression:** the tracked sparse suite, focused FP4/MoE/remap/LoRA suites, and full tracked non-live regression remain green. Every verdict-participating test passes `git ls-files`.
10. **Scope:** direct checks prove FROZEN `deepseek_v4.py`, ADR 0024, and all production inference backends unchanged.
11. **Independent gates:** Reviewer PASS and Test Manager GREEN are mandatory. Neither may infer memory safety from final active memory; both must inspect operation peak and per-expert active telemetry.
12. **No smoke before double-green:** only after both gates may BA/Architect consider one 4096 smoke under the existing no-concurrent-model and adapter-path controls.

## 10. STOP conditions

STOP any future reopening if:

- the primitive materializes full FP32 expert matrices as MLX arrays;
- operation peak retains a statistically/materially positive `E*H*I` term at fixed total assignments;
- the real-dimension no-shard bound is `>=2 GiB`;
- forward or VJP parity exceeds the pinned tolerances;
- clamp-boundary derivatives differ from the current MLX reference;
- packed nibble order or BF16 linear-scale semantics change;
- host materialization expands beyond detached integer route metadata;
- a FROZEN, inference, SSD, CUDA, distributed, default Metal, shard, dataset, or site-package edit becomes necessary;
- any verdict test is untracked;
- Reviewer or Test Manager is not independently green;
- the 340 GB process gate lacks a measured component budget with at least 10% headroom.

## 11. BA and ADR impact

- **BA:** mark Story 13.3b-5/5d Path A blocked/STOP on AC6. Revoke the current custom-VJP implementation authorization and keep the real 4096 smoke forbidden. A future story must explicitly authorize lower-level primitive research and TDD; it is not a review-fix continuation.
- **ADR 0025:** its 13.3b-5d amendment claims per-expert `mx.eval` barriers bound routed forward/backward lifetime. This adjudication disproves that consequence under MLX 0.31.2 outer transforms. Amend or supersede it before implementation.
- **New ADR:** recommended for the opaque packed-FP4 training primitive because it creates a durable Metal/MLX boundary, exact derivative contract, and backend-isolation rule.
- **ADR 0024:** unchanged. It remains the value/packing/scale oracle.
- **Technical spec:** retain the 400 GB/340 GB command only as a forbidden-until-double-green future run contract; do not present it as currently credible.

{"status":"ok","role":"Architect"}
