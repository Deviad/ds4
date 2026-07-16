# Story 13.3b-5d — sparse routed-token FP4 backward architecture

**Status:** GO for a BA re-pin, one TDD Coder slice, then Reviewer + Test Manager.  
**Selected design:** training-only eager sparse dispatch with a custom first-order VJP. Materialize only detached routing metadata on the host, evaluate each selected expert on its unique routed token rows, scatter-add in original token order, and place an `mx.eval` memory barrier after every expert in both forward and VJP. Run this training path with MLX compilation globally disabled.  
**Scope:** `deepseek_v4_nn.py` training sibling only, plus tests and an ADR 0025 amendment. No `deepseek_v4.py` body, model shard, production inference, SSD, CUDA, distributed, or default Metal edit.

## 1. Root cause

Real configuration:

```text
B=1, S=4096, T=B*S=4096
H=4096, I=2048, E=256, K=6, hc_mult=4
43 layers: 3 hash_moe + 40 learned moe
FP4: two LSB-first E2M1 values/byte, BF16 linear scale per 32 logical inputs
```

`SparseMoeBlockNN.__call__` currently loops `eid=0..255`, calls `forward_one(eid, x)` on the complete `[B,S,H]` input, and masks only after the expert result exists. `DeepseekV4FP4Experts.forward_one` dequantizes all three matrices before its matmuls. Layer checkpointing bounds retained work across decoder layers, but not the 256 expert graphs inside the recomputed layer.

Exact FP32 lower bounds per recomputed layer:

| Component | Formula | Bytes |
|---|---:|---:|
| One expert dequantized `w1+w3+w2` | `(I*H + I*H + H*I)*4` | 96 MiB |
| All expert dequant outputs | `E*96 MiB` | **24 GiB** |
| One expert full-sequence `gate+up+hidden+out` | `(3*T*I + T*H)*4` | 160 MiB |
| All expert full-sequence activations | `E*160 MiB` | **40 GiB** |

The 24 GiB and 40 GiB values exclude LUT indices/values, expanded scales, attention, checkpoint inputs, residuals, logits, weights, and allocator scratch. The clean 400,000,000,000-byte scheduling run therefore still OOMed at first backward after finite validation 18.110. Sequence reduction cannot remove the 24 GiB term, and nested checkpointing without sparse token dispatch does not remove the 40 GiB term.

## 2. Routing contract to preserve

For input `x: [D..., H]`, let `T=product(D...)`.

```text
scores      = sqrt(softplus(x @ gate_weight.T))          [D..., E]
select_from = scores + e_score_correction_bias           learned moe only
indices     = stable argsort(-select_from)[..., :K]      [D..., K]
indices     = tid2eid[input_ids]                          hash_moe only
a_t,e       = 1 if e occurs at least once in indices[t]  [T, E]
d_t         = sum_e a_t,e * scores[t,e]
f_t,e       = routed_scaling_factor * a_t,e * scores[t,e] / (d_t + 1e-20)
routed[t]   = sum_e f_t,e * expert_e(x[t])
out          = routed + shared_experts(x)
```

Required consequences:

- Learned top-k assignments and lower-index stable tie behavior remain in `_select_indices`; the sparse dispatcher does not reselect or reorder experts.
- `e_score_correction_bias` affects selection only. It never enters contribution weights.
- Hash routing still computes `scores`; those scores normalize and weight the fixed `tid2eid` selections.
- Duplicate expert ids within one token are collapsed exactly once because the current implementation uses `mx.any(indices == eid, axis=-1)`. Assignment-slot summation would be wrong.
- No auxiliary router loss exists in this port. The selected change adds none and changes no return signature.
- Shared expert execution remains outside the custom routed operation, over the original full input, and is added once.
- Output is scattered to flat token indices and reshaped to `D...`; token order is unchanged.

## 3. Installed MLX findings and feasibility

Installed versions are MLX 0.31.2 and MLX-LM 0.31.3.

- `mx.take` and array indexing have input VJPs.
- `array.at[rows].add(values)` is differentiable and correctly sums duplicate-row updates.
- `mx.stop_gradient` is available for every discrete index tensor.
- `mx.gather_mm` and `mx.gather_qmm` exist, but do not solve this format:
  - `gather_mm` requires dense matrices, preserving the 24 GiB all-expert dequant floor if all weights are expanded.
  - `gather_qmm(mode="mxfp4")` requires E8M0 scales. ADR 0024 requires BF16 linear-domain scales. Substitution would change values and is forbidden.
- MLX 0.31.2 exposes no `nonzero`, `argwhere`, `segment_sum`, or Python `scatter_add` operation that can produce value-dependent compact per-expert rows entirely in a compiled graph.
- Host materialization of detached indices works under ordinary `mx.value_and_grad`, but a tiny probe under `mx.compile` fails exactly with:

```text
ValueError: [eval] Attempting to eval an array during function transformations like compile or vmap is not allowed.
```

- Calling `mx.disable_compile()` before the trainer creates/calls its compiled step makes that wrapper execute eagerly; the same host-routed value-and-gradient probe passes.

**Conclusion:** selected-token evaluation is feasible without unsupported index VJPs, but not inside the current compiled step. Correctness-first sparse dispatch must deliberately use eager execution. This avoids dynamic-shape recompilation; there is no compiled dynamic graph to recompile. Route indices are evaluated once per layer/batch as detached metadata.

## 4. Selected implementation

### 4.1 Boundary

Add one private routed operation used only by `SparseMoeBlockNN.__call__` in:

```text
python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py
```

Despite its directory, ADR 0025 defines this file as the trainable sibling/shim. `deepseek_v4.py` remains FROZEN and unchanged. The selected operation continues to call the ADR 0024 `_dequantize_fp4_block_scale_mlx` primitive through `forward_one`; no FP4 math is copied or altered.

### 4.2 Host routing plan

Flatten only leading token axes:

```text
x_flat       [T,H]
scores_flat  [T,E]
indices_flat [T,K]
```

Apply `mx.stop_gradient(indices_flat)`, evaluate only that integer tensor, and build Python `rows_by_expert: list[list[int]]`. For each token, append its flat row once to each distinct selected expert, preserving token order. Empty experts receive an empty list and are never called.

No activation, score, gate weight, FP4 weight, or scale is converted to NumPy/Python. Only `T*K` integers cross the host boundary: 24,576 int32 values, 96 KiB at the real shape.

### 4.3 Forward custom function

Create an `mx.custom_function` for routed output with differentiable positional inputs `(x_flat, scores_flat)` and detached `indices_flat`; capture the current frozen expert tensors through the closure. Construct it at call time so load-side replacement of module arrays cannot leave stale captured tensors.

Pseudocode:

```python
@mx.custom_function
def routed_fp4(x_flat, scores_flat, indices_flat):
    rows_by_expert = materialize_unique_rows(mx.stop_gradient(indices_flat))

    denom = zeros([T], scores.dtype)
    for eid, rows in enumerate(rows_by_expert):
        if rows:
            denom = denom.at[rows].add(scores_flat[rows, eid])
    mx.eval(denom)

    routed = zeros_like(x_flat)
    for eid, rows in enumerate(rows_by_expert):
        if not rows:
            continue
        x_e = mx.take(x_flat, rows, axis=0)                 # [R_e,H]
        y_e = experts.forward_one(eid, x_e)                # [R_e,H]
        f_e = scale * scores_flat[rows, eid] / (denom[rows] + 1e-20)
        routed = routed.at[rows].add(y_e * f_e[:, None])
        mx.eval(routed)                                    # release expert graph
    return routed
```

The per-expert `mx.eval` is load-bearing. Because the custom VJP defines the mathematical backward, the materialized forward accumulator does not need to retain 256 expert graphs.

### 4.4 Exact custom VJP

Weights/scales are frozen. The VJP returns cotangents only for `x_flat` and `scores_flat`; the integer index cotangent is zero/ignored. Higher-order gradients are out of scope; this is a first-order LoRA training primitive.

For output cotangent `g_t` and selected expert output `y_t,e`, define:

```text
a_t,e = dot(g_t, y_t,e)
Q_t   = sum_e selected scores[t,e] * a_t,e
D_t   = sum_e selected scores[t,e]
D'_t  = D_t + 1e-20
```

Then the exact score cotangent is:

```text
dL/dscores[t,e] = routed_scaling_factor * (a_t,e * D'_t - Q_t) / (D'_t * D'_t)
```

for selected unique experts, zero otherwise.

Expert-input cotangent must reuse MLX's VJP over the existing `forward_one` rather than hand-code clip/SwiGLU boundary derivatives:

```python
_, (dx_e,) = mx.vjp(
    lambda z: experts.forward_one(eid, z),
    [x_e],
    [f_e[:, None] * g_e],
)
dx = dx.at[rows].add(dx_e)
```

Evaluate `Q`, then `dx` and `dscores` after each expert. This bounds the backward graph to one expert. Returning `dscores` lets ordinary MLX autodiff propagate through `sqrt(softplus(x @ gate_weight.T))`; returning `dx` preserves the direct expert-input path. MLX sums both paths at `x`, so upstream attention LoRA gradients remain exact.

### 4.5 Required execution mode

The trainer must be launched with compilation disabled before importing/calling `mlx_lm.lora.main`, while retaining the 400 GB graph limit:

```python
import mlx.core as mx
mx.disable_compile()
mx.set_memory_limit(400_000_000_000)
from mlx_lm.lora import main
main()
```

`MLX_DISABLE_COMPILE=1` is an equivalent operator setting, but the wrapper call is preferred because the log can print both selected policies. Do not patch site-packages. Do not globally disable compilation from module import or model construction; normal inference and unrelated MLX users must remain untouched.

## 5. Probe evidence

Tiny deterministic probes used the installed production FP4 dequant primitive, multi-token/multi-expert routing, duplicate ids, an empty-expert-capable route plan, gathered token rows, and `array.at.add`.

Observed:

```text
forward max_abs                         0.0
input-gradient max_abs                  0.0
gate-weight-gradient max_abs            0.0
custom-VJP input-gradient max_abs       1.862645149230957e-09
custom-VJP score-gradient max_abs       3.814697265625e-06
all custom gradients finite/non-zero    true
compiled host routing                   FAIL (eval forbidden during compile)
compile globally disabled               PASS
```

The duplicate scatter probe also returned the expected summed cotangent for two updates to row zero. No model shard was loaded.

These probes establish API/VJP feasibility, not the final implementation verdict. Coder must first encode independent RED tests.

## 6. Memory bound

### 6.1 Routed operation hard bound

The custom function evaluates and releases one expert before the next. `R_e` is the number of unique tokens assigned to expert `e`; `0 <= R_e <= T`, and `sum_e R_e <= T*K = 24,576`.

Worst case for one live expert at `R_e=T=4096`:

| Live item | Upper size |
|---|---:|
| Dense FP32 `w1+w3+w2` outputs | 96 MiB |
| Gathered input `[R_e,H]` | 64 MiB |
| `gate+up+hidden` | 96 MiB |
| Expert output | 64 MiB |
| Routed/output cotangents and accumulators | <=256 MiB |
| Scores + score cotangent + route metadata | <9 MiB |
| FP4 LUT indices/values, expanded scales, products, matmul scratch | conservatively <=512 MiB |
| **Selected routed-op active bound** | **<1.1 GiB; budget 1.25 GiB** |

Average balanced routing is `T*K/E = 96` rows/expert; dequant scratch dominates there. The bound remains valid for adversarial concentration because experts execute serially. It removes both retained terms from §1: 24 GiB all-expert dequant and 40 GiB all-expert full-sequence activations.

### 6.2 Whole-process envelope

- Actual `model-4bit` safetensors total: 160,062,732,830 bytes = 149.07 GiB. Count the whole mapping as resident.
- Decoder checkpoint inputs, pessimistically FP32 `[1,4096,4,4096]` for all 43 layers: 11.0 GiB.
- Final FP32 logits `[1,4096,129280]`: 1.97 GiB.
- Selected routed operation: 1.25 GiB.
- LoRA rank-8 parameters/gradients are small relative to one GiB.
- Reserve 86.7 GiB for one attention/CSA/shared-expert recomputation, loss, transient casts, and MLX allocator scratch.
- Reserve another 50 GiB for unmodelled/runtime variance.

This gives a deliberately conservative **300 GiB process design ceiling**. The selected MLX limit is 400,000,000,000 bytes = 372.53 GiB, leaving **72.5 GiB safety margin**. Physical memory is 512 GiB and the recommended working set is about 464 GiB, so the 400 GB scheduler remains below both.

This is an architecture budget, not a claim that file bytes equal peak allocator bytes. The Coder memory test must prove the routed operation's bounded-growth shape; the first real smoke must log `mx.get_peak_memory()`. STOP if the implementation cannot hold the routed-op active delta below 2 GiB on the approved lightweight real-dimension/no-shard probe, or if the real process exceeds 340 GB before first backward completion. The latter preserves at least 60 GB decimal headroom to the selected limit.

## 7. Candidate adjudication

| Candidate | Verdict | Reason |
|---|---|---|
| Per-expert host gather → FP4 expert → weighted scatter-add, custom eager VJP | **SELECTED** | Exact current semantics; only routed rows; no unsupported index VJP; one-expert active bound; isolated training shim |
| Plain per-expert gather/scatter under ordinary autodiff | Reject | Removes full-sequence activations but may retain 24 GiB of dequant outputs for backward |
| Native `gather_qmm` / `SwitchGLU` | Reject | Native mxfp4 requires E8M0 scales; ADR 0024 requires BF16 linear scales |
| `gather_mm` after all-expert dequant | Reject | Keeps 24 GiB dense dequant floor |
| Fixed-capacity routing | Reject | Exact no-drop capacity must handle `R_e=T`; recreates dense storage/compute and fails routed-row instrumentation |
| Assignment chunks with gathered packed matrices | Reject | Replicates large packed/dequantized matrices per assignment or requires a new optimized custom Metal GEMM/VJP; not the smallest correctness slice |
| Expert-weight chunking with dense token masks | Reject | Still evaluates unassigned tokens and does not satisfy routed-token-count invariant |
| Dequant stop-gradient or nested per-expert checkpoint alone | Reject | Prior probes showed no improvement/regression; neither gives the explicit one-expert memory barrier |
| FROZEN vendor primitive edit | Forbidden | Unnecessary; would require separate ADR authorization and re-entry |

## 8. File/function edit map

Production/training shim edit, one file:

- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py`
  - add private unique-route-plan helper;
  - add private custom routed FP4 function/factory and exact VJP;
  - change only `SparseMoeBlockNN.__call__` routed branch to use it;
  - keep `_scores`, `_select_indices`, shared expert, FP4 storage, and `forward_one` math unchanged;
  - optionally retain dense implementation as a test-only helper outside production, not as a permanent runtime flag. Preferred: put independent dense reference in tests and delete no production behavior behind a flag.

Tests:

- New tracked test file, suggested `tests/test_deepseek_v4_nn_sparse_routed_backward.py`.
- Existing focused tests remain unchanged unless a stale expectation is demonstrably wrong. Every test contributing to the verdict must pass `git ls-files -- <path>` before Coder/Reviewer/Test Manager verdicts.

Canonical docs in the coding slice:

- Amend `docs/adr/0025-real-trainable-nn-module-deepseek-v4-port.md` with the training-only eager sparse custom-VJP contract and compile-disabled requirement.
- Update `docs/technical-spec.md` with the exact 4096 smoke wrapper and memory telemetry requirements.
- BA owns the `docs/backlog.md` re-pin.
- ADR 0024 is consumed unchanged because packing, LUT, nibble order, scale type/domain, and dequant math do not change.

## 9. RED → GREEN TDD plan

Write all RED tests before implementation.

1. **Independent forward parity**
   - Tiny `T>=5`, `E>=4`, `K>=2`, nonuniform packed FP4 bytes/scales, nonuniform tokens/gate weights.
   - Compare sparse output to an independent dense NumPy reference or the frozen dense MLX formula.
   - Assert shape, order, finiteness, and `max_abs <= 1e-6` for float32 deterministic fixtures.

2. **Ties, duplicates, empty experts**
   - Learned equal `score+bias` tie keeps lower expert ids from existing stable argsort.
   - Hash `tid2eid` duplicate slots count one expert contribution and one denominator term, matching current `mx.any` semantics.
   - Empty expert is never called and changes neither denominator nor output.

3. **Gradient parity**
   - Compare dense ordinary-autodiff and sparse custom-VJP gradients for input and trainable gate weight.
   - Use loss `sum(out*out)` and non-saturated fixtures.
   - Tolerance: input `atol=rtol=1e-5`; gate/score `atol=rtol=1e-5` (probe worst score difference `3.82e-6`).
   - Add one clamp-near fixture; custom expert-input VJP must match MLX because it calls `mx.vjp(forward_one)` rather than duplicating derivatives.

4. **Detached index contract / no unsupported scatter VJP**
   - Assert route indices pass through `mx.stop_gradient` before host materialization.
   - Differentiate only `(x,scores)`; no gradient transform is requested for integer indices.
   - Exercise duplicate `array.at.add` rows and prove finite expected cotangents.
   - Fail on any `scatter_axis`/index VJP exception.

5. **Routed-token instrumentation**
   - Wrap `forward_one` and record every input first dimension.
   - Assert call sizes equal exact unique `R_e` values, never `T` for experts with proper subsets, and no call for `R_e=0`.
   - Assert `sum R_e <= T*K`.

6. **Bounded-growth memory probe**
   - Separate process per dense/sparse case; reset peak, clear cache, execute forward+VJP, `mx.eval`, report peak delta.
   - Increase `E` while keeping `T,K,H,I` tiny. Dense peak must grow with `E*T`; sparse custom peak must stay governed by max `R_e` plus output accumulators.
   - Add an approved no-shard real-dimension single-live-expert probe; routed-op active delta must be <2 GiB. Do not load model shards.

7. **Real-config shape/contract test without shards**
   - `H=4096,I=2048,E=256,K=6,T=4096` formula-only assertions: route metadata 96 KiB, one-expert dequant 96 MiB, old retained floors 24/40 GiB, new budget 1.25 GiB.
   - Verify flatten/reshape and hash/learned config fields from a synthetic `ModelArgs`; do not instantiate all real expert payloads if that would allocate GiBs.

8. **LoRA regression**
   - Existing tiny two-layer trainer-contract backward remains finite, gradient tree nonempty, every LoRA leaf finite, and at least one LoRA gradient nonzero.
   - Existing FP4 parity, learned MoE, hash MoE, remap, and full focused suite remain green.

9. **Eager execution gate**
   - Tiny probe documents compiled host routing fails with MLX's exact eval-during-transform error.
   - With `mx.disable_compile()` before creation/call of the trainer-style wrapper, value+gradient passes.
   - No module-import global compile side effect.

## 10. Run plan

1. BA re-pins Story 13.3b-5 to authorize one implementation slice, eager training, the custom first-order VJP, exact tests, ADR 0025 amendment, and the revised command. Previous 13.3b-5c was run-only and cannot authorize this edit.
2. Coder follows RED → GREEN, records tiny probes and tracked-test checks, and makes no model/full-run attempt.
3. Reviewer and Test Manager independently verify math, route uniqueness, custom-VJP formula, tracking, memory formula, and scope. Reviewer PASS and Tester GREEN are mandatory.
4. Resource preflight as in 13.3b-5c: no concurrent large model process; clean adapter path; model/dataset/LoRA/tokenizer gates pass.
5. Re-run 4096 only, with one validation batch, 400 GB memory limit, and compilation disabled before `mlx_lm.lora.main()`. Print selected compile mode, graph limit, cache/active/peak memory after validation and after each completed training step.
6. First gate: finite validation. Second gate: first backward + optimizer step completes with finite loss and peak <340 GB. If green, continue through iteration 20 and validate fresh adapter safetensors exactly as 13.3b-5c required.
7. No automatic 3072/2048/1536/1024 retry.

## 11. Invariants and forbidden alternatives

- Exact top-k/hash indices, normalized score weights, duplicate collapse, stable tie behavior, output order, and shared expert behavior.
- Gradients through token activations and selected gate scores; discrete route indices detached.
- Frozen experts remain frozen; no gradient allocation for packed weights/scales.
- FP4 remains ADR 0024 LSB-first E2M1 with BF16 linear per-32 scales.
- First-order gradients only. Do not silently claim higher-order custom-VJP support.
- No permanent semantic flag selecting dense versus sparse behavior.
- No native mxfp4 scale substitution, approximation, token dropping, capacity overflow, aux-loss invention, or routing renormalization change.
- No FROZEN `deepseek_v4.py` body edit.
- No production inference, SSD streaming, CUDA, distributed inference, default Metal, dataset, checkpoint, or site-package edit.
- No full model or shard load in the coding/tests slice.

## 12. STOP conditions

STOP and return to Architect if any occurs:

- Host route materialization cannot run under globally disabled compile in the actual trainer wrapper.
- Forward or input/gate gradient parity exceeds pinned tolerances.
- Duplicate hash routes are counted per slot rather than per unique expert.
- Any integer-index/scatter VJP error remains.
- The custom VJP requires a `deepseek_v4.py` FROZEN-body change.
- A no-shard real-dimension probe exceeds 2 GiB active delta or shows expert graphs accumulating across loop iterations.
- Import/model construction changes global MLX compile state.
- Existing LoRA backward loses finite nonzero gradients.
- Real pre-backward peak reaches 340 GB, first backward OOMs, loss becomes non-finite, or adapter validation fails.
- Reviewer or Test Manager finds untracked verdict-participating tests.

## 13. BA and ADR verdict

- **BA re-pin required: YES.** The current requirement is run-only, fixes the exact compiled command contract, and authorizes no implementation edit. BA must add eager execution, custom-VJP parity, bounded-memory, and first-step peak gates.
- **ADR update required: YES.** Amend ADR 0025 because it explicitly records the original dense correctness-first loop as a deferred optimization and defines the trainable sibling boundary. Record sparse eager dispatch/custom first-order VJP there. **Do not amend ADR 0024**; FP4 format/dequant contract is unchanged.
