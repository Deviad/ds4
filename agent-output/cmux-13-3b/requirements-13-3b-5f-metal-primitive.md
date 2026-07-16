# Story 13.3b-5f — opaque packed-FP4 Metal/MLX training primitive (BA re-pin)

**Status:** GO to Architect for one durable primitive/VJP design and ADR, then one TDD RED → GREEN implementation slice within that accepted boundary. The superseded Python `forward_one` + nested `mx.vjp` Path A remains STOPPED. No real model, shard, or 4096 smoke is authorized before independent Reviewer PASS and Test Manager GREEN.

## User story

**As a** training engineer (WHO), **I want** a training-only opaque Metal/MLX primitive that consumes packed OCP E2M1 FP4 experts and exposes the exact first-order routed-MoE forward/input/score derivative contract (WHAT), **so that** outer MLX transforms retain only assignment-proportional tensors and a real 4096-token QLoRA backward can fit without changing FP4, routing, shared-expert, or LoRA gradient semantics (WHY).

## Supersession and authorization

- User authorization reopens Path A only around the lower-level opaque primitive defined here. It does not authorize another Python graph-lifetime workaround.
- Story 13.3b-5e remains the RED/root-cause authority: under MLX 0.31.2 outer transforms, `mx.eval`, `mx.stop_gradient`, `del`, `mx.clear_cache`, phase custom functions, and route chunks do not detach retained per-expert dequant/VJP graphs.
- Revoke the Story 13.3b-5d implementation authorization for the Python `forward_one` + nested `mx.vjp` composition. The real 4096 smoke remains forbidden.
- ADR 0025's 13.3b-5d amendment is superseded specifically where it claims that per-expert `mx.eval` barriers guarantee one-expert routed forward/backward lifetime under the trainer's outer transform.
- ADR 0024 remains unchanged and is the binding FP4 value, packing, scale, and independent-reference contract.
- Before implementation, Architect must create a new durable ADR for the opaque packed-FP4 training primitive, exact first-order derivative contract, MLX transform boundary, memory lifetime, and backend isolation. The ADR must explicitly amend or supersede the contradicted ADR 0025 claim.

## Authorized future implementation boundary

Authorize only the smallest training-only lower-level primitive and its narrow training integration after Architect acceptance:

- A new isolated Metal/MLX primitive may consume one selected expert/assignment group at a time on its unique routed token rows `x_e[R_e,H]`.
- It must consume packed OCP MXFP4 E2M1 weights directly as uint8 bytes, decoding low nibble first and high nibble second exactly as ADR 0024.
- It must apply BF16 linear-domain scales by direct multiply, one scale per 32 consecutive logical input features on axis 1, exactly as ADR 0024. Native E8M0/mxfp4 scale substitution is forbidden.
- It must dequantize bounded tiles internally. It must never materialize, retain, return, or expose full FP32 `w1`, `w3`, or `w2` expert matrices as MLX arrays.
- Its first-order leaf contract must provide:
  - expert output `y_e[R_e,H]`;
  - expert-input cotangent `dx_e[R_e,H]` from the normalized routed upstream cotangent;
  - rowwise `a_e[R_e] = dot(g_e, y_e)` for normalized router-score gradients.
- The normalized score derivative must use the exact simplified identity:

  ```text
  dscore_e = rsf*a_e/D' - dot(g,routed)/D'
  ```

  Here `D'` is the duplicate-collapsed selected-score denominator, `rsf` is the selected routed score factor, `g` is the routed-output cotangent, and `routed` is the normalized routed output. No Q recomputation pass is authorized.
- Packed weights and BF16 scales are frozen inputs. The primitive returns no packed-weight or scale cotangents and makes no higher-order-gradient claim.
- The primitive must be an opaque custom-VJP leaf under the installed MLX 0.31.2 transform stack. Outer `mx.grad` / trainer `mx.value_and_grad` may retain assignment-proportional inputs, outputs, cotangents, accumulators, and bounded tile scratch, but not expert dequant graphs proportional to `E*H*I`.
- Preserve exact clip behavior, clamp-boundary derivative behavior, sigmoid-SwiGLU, `w1`/`w3`/`w2` ordering, normalized routing, learned/hash selection, lower-index stable ties, correction-bias selection-only semantics, duplicate expert collapse per token, empty-expert skip, token order, one full-input shared-expert call, and LoRA gradient semantics.
- Host materialization remains limited to `mx.stop_gradient`-detached integer route metadata. Activations, scores, gates, packed weights, scales, outputs, and cotangents stay on MLX/Metal.
- Integration remains training-only and isolated from production inference, existing production Metal kernels/wrappers, SSD streaming, CUDA, ROCm, distributed inference, default Metal inference, CPU reference inference, model loading, datasets, checkpoint/model shards, and site-packages.
- A new training-only Metal source and narrow MLX wrapper plus one training-sibling dispatch are permissible only if Architect proves that they do not enter production build/runtime paths. Existing `metal/*.metal`, `ds4_metal.m`, runtime C/CUDA/ROCm files, and FROZEN `deepseek_v4.py` remain byte-intact.
- No C++, approximation, changed FP4 values, token dropping, capacity limit, changed normalization, permanent semantic variant flag, host activation/cotangent round trip, or fallback to the stopped Python graph composition.

## Required RED → GREEN acceptance criteria

Coder must write tracked RED tests before implementation, prove the stopped current path RED, then make the opaque primitive GREEN without loading a real model or any model shard.

1. **Current Python-path RED.** In fresh processes with compile disabled, fixed `K=2`, `T=1024`, `H=1024`, `I=512`, all experts non-empty, and identical total assignments, the existing Python custom-VJP path at `E=2/4/8` reproduces a materially E-scaled operation peak. Record active/cache/peak telemetry at expert boundaries. Historical locked evidence is `584,225,254`, `779,041,700`, and `1,179,289,384` bytes; the load-bearing RED is growth with `E` at fixed `T*K`, not allocator-byte identity.
2. **Packed forward parity.** Deterministic nonuniform token rows and distinct packed expert bytes/scales exercise both nibbles and multiple 32-logical-input scale blocks. Primitive `y_e` matches an independent ADR 0024 reference with `max_abs <= 1e-6`; shape, dtype contract, token-row order, and finiteness match. Test expectations must not derive from the implementation primitive.
3. **First-order VJP parity.** Primitive `dx_e` and whole sparse input plus score/gate cotangents match independent ordinary-MLX/dense references at `atol=rtol=1e-5`. Include positive, negative, and exact clip/clamp boundaries, nonuniform upstream cotangents, duplicate routes, and expected non-zero finite leaves.
4. **Score identity parity.** Independently compare `a_e = rowwise_dot(g_e,y_e)` and `dscore_e = rsf*a_e/D' - dot(g,routed)/D'` against the prior exact Q-form derivative at `atol=rtol=1e-5`, including duplicate-collapsed denominators and clamp-boundary fixtures.
5. **No frozen-parameter cotangents.** Transform/API tests prove the leaf differentiates only activation and score paths, returns no packed-weight or scale cotangents, and preserves focused LoRA propagation.
6. **Semantic invariants.** Learned and hash routes, lower-index stable ties, correction-bias selection-only behavior, duplicate collapse, empty-expert skip, exact normalized contribution weights, shared-expert behavior, token order, clip, sigmoid-SwiGLU, and `w1`/`w3`/`w2` ordering remain unchanged.
7. **Opaque/no-dense boundary.** Source/API and operation telemetry prove packed weights/scales enter the leaf directly, dequantization remains tile-local, no full FP32 expert matrix is returned or retained as an MLX array, and outer transforms expose no `E*H*I` dequant graph.
8. **Fixed-assignment memory GREEN.** Fresh-process representative outer-transform probes at fixed `K=2`, `T=1024`, `H=1024`, `I=512`, all experts non-empty, and `E=2/4/8` record active/cache/operation peak at every expert boundary. `max(peak)-min(peak) <= 64 MiB`; fitted/structural evidence shows no `E*H*I` slope. Final flat active memory alone is not evidence.
9. **Real-dimension no-shard GREEN.** A synthetic `H=4096`, `I=2048` no-shard representative outer-transform operation returns finite `y_e`, `dx_e`, and `a_e` with operation peak `<2 GiB`. Packed payload setup is outside the measured operation baseline; no 256-expert real payload allocation is required.
10. **Real-config formula bound.** For `E=256`, `K=6`, `T=4096`, `H=4096`, the derivation pins `sum_e R_e = T*K` and one FP32 assignment-proportional `[T*K,H]` tensor to `T*K*H*4 = 384 MiB`. The complete simultaneous assignment tensors, accumulators, metadata, and bounded tile scratch must have a documented routed-primitive operation bound `<=2 GiB`, with no `E*H*I` term.
11. **Focused LoRA gradients.** The tiny trainer-contract backward produces a non-empty LoRA gradient tree; every expected leaf is finite; at least one expected LoRA gradient/update is non-zero. No route, expert-weight, or scale gradient is invented.
12. **Regression and scope.** Existing tracked FP4, sparse routed-MoE, remap, attention/LoRA, and full non-live regressions remain green. Direct hashes/source checks prove ADR 0024, FROZEN `deepseek_v4.py` except its already-sanctioned 13.3b-5b state, and all production backends byte-intact.
13. **Tracking and reproducibility.** Every test file contributing to RED/GREEN, Reviewer, or Test Manager verdicts must return a non-empty `git ls-files -- <path>`. Any touched untracked verdict test must be tracked and included in the slice's staged/committed chain; a local pass count is not GREEN.
14. **Canonical documentation.** The new ADR is accepted, ADR 0025's disproved lifetime claim is explicitly revoked/superseded, ADR 0024 stays unchanged, and `docs/technical-spec.md` records the primitive gate plus the still-forbidden-until-double-green smoke policy.
15. **Chain of custody.** Preserve the Story 13.3b-5b sanctioned `_csa_block_bias_mlx` `mx.stop_gradient` fix and approved FROZEN SHA `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b`. Complete the tracked/staged predecessor chain for the three hash-pin advances, ADR 0026 amendment, and passthrough chat-template script, or STOP with an explicit split/commit dependency before verdict.

## Double-green gate

After Coder completes the tracked RED → GREEN slice:

1. Reviewer independently verifies ADR 0024 packing/scale fidelity, forward and first-order VJP math, score identity, clamp boundaries, opacity/no-dense exposure, outer-transform operation peaks, absence of an `E*H*I` slope, backend/FROZEN isolation, canonical ADR updates, chain of custody, and tracked-test reproducibility. Required verdict: **PASS**.
2. Test Manager independently reruns the RED baseline, primitive parity/VJP/score tests, fixed-assignment `E=2/4/8` operation-peak probes, real-dimension no-shard probe, formula bound, focused LoRA gradients, regressions, scope hashes, and `git ls-files` checks. Required verdict: **GREEN**.

Both verdicts are mandatory. Any FAIL, NO-GO, RED, NEEDS-INFO, untracked verdict test, parity drift, dense exposure, positive `E*H*I` slope, or peak failure returns to Architect/Coder and authorizes no smoke.

## Later real 4096 eager-smoke policy

Only after Reviewer PASS and Test Manager GREEN may exactly one real 4096-token smoke run. No shorter fallback or second smoke is authorized.

The wrapper remains:

```python
import mlx.core as mx
mx.disable_compile()
mx.set_memory_limit(400_000_000_000)
from mlx_lm.lora import main
main()
```

Retain the existing pinned 13.3b-5c/5d command paths and arguments: `model-4bit`, `mlx-4096`, `--iters 20`, `--batch-size 1`, `--learning-rate 1e-5`, `--max-seq-length 4096`, `--mask-prompt`, `--grad-checkpoint`, `--val-batches 1`, and `--steps-per-report 1`.

Before launch, verify no concurrent large model process, all model/data/tokenizer/LoRA gates, and an absent or empty adapter output path. Log eager policy, the `400_000_000_000`-byte graph limit, start time, PID, exact command, and MLX cache/active/peak memory after validation and every completed step. Exactly one validation batch must complete with finite loss. STOP before first backward if peak is `>=340 GB`. First backward/optimizer step must complete with finite loss, non-empty all-finite LoRA gradients, at least one non-zero expected gradient/update, and peak `<340 GB` before continuing the same smoke. No automatic 3072/2048/1536/1024 fallback.

## STOP conditions carried forward exactly from Architect 13.3b-5e §10

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

## Additional Story 13.3b-5f STOP conditions

STOP and return to Architect if:

- the primitive requires any production inference or additional FROZEN edit;
- any packed FP4 decoded value, LSB-first ordering, BF16 scale value, or per-32 logical-input scale application changes;
- any full dense expert matrix is allocated, returned, or retained, including behind an MLX wrapper;
- installed MLX cannot expose the exact first-order custom-VJP leaf contract to the trainer's outer transform;
- any measured `<=64 MiB`, `<2 GiB`, or formula-backed `<=2 GiB` peak gate cannot be met;
- the new primitive cannot remain training-only and backend-isolated;
- the predecessor chain of custody cannot be made reproducible without changing the sanctioned 13.3b-5b fix;
- the later real smoke reaches `>=340 GB` before backward, OOMs, produces non-finite loss/gradients, lacks a non-zero expected LoRA update, or fails adapter validation.

## Run and evidence policy

- No full model, model shard, dataset training run, or 4096 smoke during implementation, review, or test-manager validation.
- Use only deterministic tiny/synthetic fixtures and no-shard memory probes.
- Coder evidence must include RED-before-GREEN commands/results, exact parity maxima, operation active/cache/peak values, E-spread calculation, real-dimension peak, formula derivation, LoRA-gradient checks, direct scope hashes, `git diff --check`, and `git ls-files` output for every verdict test.
- Reviewer and Test Manager must produce independent evidence and may not infer operation memory safety from final active memory.
