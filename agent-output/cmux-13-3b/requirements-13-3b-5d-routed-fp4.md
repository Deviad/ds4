# Story 13.3b-5d — sparse routed-token FP4 backward (BA re-pin)

**Status:** GO for exactly one RED → GREEN implementation slice, then independent Reviewer PASS + Test Manager GREEN. The real 4096 smoke remains forbidden until both gates are green.

## User story

**As a** training engineer (WHO), **I want** the training-only DeepSeek V4 NN sibling to execute each FP4 routed expert on only its unique selected token rows and provide an exact custom first-order VJP for token activations and router scores (WHAT), **so that** the real 4096-token LoRA backward can fit within a bounded memory envelope without changing routing, FP4, shared-expert, or gradient semantics (WHY).

## Authorized implementation boundary

Authorize one TDD implementation slice with one production/training-shim edit:

- Edit only `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py` in the implementation path.
- Replace dense all-256-expert/full-token routed execution in `SparseMoeBlockNN.__call__` with eager per-expert routed-token gather/scatter.
- Preserve `_scores`, `_select_indices`, exact learned top-k and hash routes, lower-index stable ties, duplicate-per-token expert collapse, normalized score weights, `e_score_correction_bias` selection-only behavior, token order, return shape, and one full-input shared-expert contribution.
- Materialize only `mx.stop_gradient`-detached integer routing metadata on the host. Do not convert activations, scores, gate weights, packed FP4 weights, or scales to NumPy/Python.
- Use a custom first-order VJP whose differentiable inputs are token activations and scores. Return cotangents only for those inputs; discrete route indices remain detached and frozen expert weights/scales receive no gradients.
- Reuse existing `DeepseekV4FP4Experts.forward_one` inside `mx.vjp` for exact FP4 expert-input derivatives. Do not copy or replace ADR 0024 FP4 math.
- Place per-expert `mx.eval` barriers in routed forward and custom VJP so one expert graph is active at a time. Skip empty experts.
- Disable MLX compilation only in the explicit training wrapper before `mlx_lm.lora.main()`. Importing or constructing the model must not change global compile state.
- Add tracked tests, amend `docs/adr/0025-real-trainable-nn-module-deepseek-v4-port.md`, and update the exact eager-smoke command and telemetry contract in `docs/technical-spec.md`.
- Keep `docs/adr/0024-fp4-e2m1-on-the-fly-moe-dequant-contract.md` unchanged.
- Do not edit the FROZEN `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` body.
- Do not edit production inference, SSD streaming, CUDA, distributed inference, default Metal, datasets, checkpoint/model shards, or site-packages.
- Do not add a permanent dense/sparse semantic flag, native mxfp4 scale substitution, approximation, token dropping, capacity overflow, auxiliary router loss, changed renormalization, or higher-order-gradient claim.

## Required RED → GREEN acceptance criteria

Coder must write the RED tests before implementation and make all criteria GREEN without loading model shards or the full model.

1. **Independent tiny forward parity.** A deterministic float32 fixture with `T>=5`, `E>=4`, `K>=2`, nonuniform tokens, gate weights, packed FP4 bytes, and scales matches an independent dense reference in shape and token order; all values are finite and maximum absolute error is `<=1e-6`.
2. **Routing semantics.** Learned equal `score+bias` ties retain lower expert ids; duplicate hash-route expert ids for one token contribute exactly once to numerator and denominator; empty experts are never called; score-correction bias remains selection-only; shared expert behavior is unchanged.
3. **Gradient parity.** Sparse custom-VJP gradients match dense ordinary-autodiff gradients for token input and gate/score paths with `atol=rtol=1e-5`, including a clamp-near fixture. Every compared gradient is finite and expected non-zero leaves are non-zero.
4. **Detached-index contract.** Integer routes pass through `mx.stop_gradient` before host materialization; differentiation targets only token activations and scores; duplicate-row scatter-add cotangents are correct; no scatter/index VJP failure remains.
5. **Exact routed-token instrumentation.** Every called expert receives exactly its unique routed-token count `R_e`, never full `T` for a proper subset; experts with `R_e=0` are not called; token order is preserved; `sum_e R_e <= T*K`.
6. **Bounded-growth probes.** Separate-process dense and sparse forward+VJP probes show dense growth with `E*T` while sparse growth is governed by maximum `R_e` plus accumulators. An approved no-shard real-dimension routed-operation probe has active-memory delta `<2 GiB` and shows no cross-expert graph accumulation.
7. **Real-config formula contract.** Formula-only checks for `T=4096`, `H=4096`, `I=2048`, `E=256`, `K=6` pin route metadata to 96 KiB, one-expert FP32 dequant output to 96 MiB, old retained floors to 24 GiB and 40 GiB, selected routed-operation budget to 1.25 GiB, and whole-process design ceiling to 300 GiB.
8. **Existing LoRA backward.** The focused tiny trainer-contract backward returns a non-empty gradient tree; every LoRA leaf is finite; at least one expected LoRA gradient is non-zero. Existing FP4 parity, learned-MoE, hash-MoE, remap, and focused regression suites remain green.
9. **Tracking and reproducibility.** Every test file contributing to any RED/GREEN, Reviewer, or Test Manager verdict is tracked according to `git ls-files -- <path>`. Any touched untracked verdict-participating test is committed in the same slice or the slice STOPs; a local-only pass count is not GREEN.
10. **Canonical documentation and scope.** ADR 0025 records the eager sparse custom-first-order-VJP decision and wrapper-local compile disable; `docs/technical-spec.md` records the exact later 4096 command and telemetry policy; ADR 0024 and unrelated paths remain unchanged; direct checks prove the FROZEN `deepseek_v4.py` body unchanged.

## Double-green gate

Coder may not load model shards or run the full model. After Coder completes the tracked RED → GREEN slice:

1. Reviewer independently verifies the forward and VJP math, unique-route semantics, per-expert barriers, scope, FROZEN-body integrity, canonical documentation, and tracked-test reproducibility. Required verdict: **PASS**.
2. Test Manager independently reruns the focused and regression checks, separate-process memory probes, real-dimension/no-shard `<2 GiB` probe, formula contract, LoRA-gradient contract, scope checks, and `git ls-files` tracking checks. Required verdict: **GREEN**.

Both verdicts are mandatory. Any NO-GO, RED, NEEDS-INFO, untracked test, parity drift, or memory failure returns to Coder or Architect; it does not authorize a smoke.

## Later real 4096 eager-smoke policy

Only after Reviewer PASS and Test Manager GREEN, authorize exactly one real smoke at sequence length 4096. No shorter fallback is authorized.

The explicit wrapper policy is:

```python
import mlx.core as mx
mx.disable_compile()
mx.set_memory_limit(400_000_000_000)
from mlx_lm.lora import main
main()
```

The final command must retain the proven 13.3b-5c arguments and paths:

```text
--config '/Volumes/Data NVME/mlx-ft/ds4/lora-config.json'
--model '/Volumes/Data NVME/mlx-ft/ds4/model-4bit'
--train
--data '/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096'
--adapter-path '/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke-memory400'
--fine-tune-type lora
--iters 20
--batch-size 1
--learning-rate 1e-5
--max-seq-length 4096
--mask-prompt
--grad-checkpoint
--val-batches 1
--steps-per-report 1
```

Before launch: verify no concurrent large model process; verify model, dataset, tokenizer, and LoRA-target gates; make the adapter path absent or empty; capture the exact command, start time, PID, and complete log. The wrapper/log must report eager compilation policy and selected `400_000_000_000`-byte limit. Record MLX cache/active/peak memory after validation and after every completed training iteration.

Gate order:

1. One validation batch completes with finite loss.
2. Before starting first backward, STOP if peak memory is `>=340 GB`.
3. First backward and optimizer step complete with finite loss, a non-empty finite LoRA gradient tree, at least one non-zero expected gradient/update, and peak `<340 GB`.
4. If green, continue the same single smoke through iteration 20 with per-iteration reporting and memory telemetry.
5. Fresh `adapters.safetensors` must load, contain only expected finite LoRA tensors, and include at least one non-zero delta.

No automatic 3072/2048/1536/1024 retry; no second smoke without a new re-pin.

## STOP conditions

STOP and return to Architect if any occurs:

- Host route materialization cannot run under globally disabled compile in the actual trainer wrapper.
- Forward or input/gate gradient parity exceeds the pinned tolerances.
- Duplicate hash routes are counted per slot rather than per unique expert.
- Any integer-index/scatter VJP error remains.
- The custom VJP requires a `deepseek_v4.py` FROZEN-body change.
- A no-shard real-dimension probe exceeds 2 GiB active delta or shows expert graphs accumulating across loop iterations.
- Import or model construction changes global MLX compile state.
- Existing LoRA backward loses finite non-zero gradients.
- Reviewer or Test Manager finds an untracked verdict-participating test.
- Real pre-backward peak reaches 340 GB, first backward OOMs, loss becomes non-finite, or adapter validation fails.

## Required evidence

Coder handoff must include RED evidence, GREEN evidence, exact test commands, exact separate-process memory numbers, real-config formula results, direct FROZEN-body proof, `git diff --check`, and `git ls-files` output for every verdict-participating test. Reviewer and Test Manager must independently record their tracking checks and verdicts. Historical 13.3b-5c finite validation 18.110 and first-backward OOM/exit 134 remain preserved as failure evidence, not a current GREEN claim.
