# Story 13.3b-5d — BA re-pin: sparse routed-token FP4 backward

## Goal
Re-pin Story 13.3b-5 requirements to authorize the Architect-selected training-shim implementation after the 400GB run still OOMed at first backward.

## User model/budget override
Use current BA pane model `openai-codex/gpt-5.6-sol` high. Do not switch providers/models.

## Read first
1. `agent-output/cmux-13-3b/architecture-13-3b-5d-routed-fp4-backward.md`
2. `agent-output/cmux-13-3b/coder-13-3b-5c-memory400-stop.md`
3. `agent-output/cmux-13-3b/requirements-13-3b-5c-memory400.md`
4. Story 13.3b-5 section in `docs/backlog.md`

## Locked architecture decision
Authorize one TDD implementation slice:
- edit only training sibling `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py`;
- replace dense all-256-expert/full-token routed execution with eager per-expert routed-token gather/scatter;
- preserve exact top-k/hash route selection, duplicate collapse, normalized score weights, stable ties, token order, and shared expert behavior;
- use detached integer routing metadata and a custom first-order VJP returning gradients for token activations and scores only;
- use existing `forward_one` inside `mx.vjp` for exact FP4 expert-input derivatives;
- place per-expert `mx.eval` barriers in forward and VJP;
- disable MLX compilation only in the explicit training wrapper before `mlx_lm.lora.main()`; no module-import global side effect;
- amend ADR 0025; ADR 0024 unchanged;
- no `deepseek_v4.py` FROZEN-body edit.

## Required RED→GREEN acceptance criteria
Pin checkable AC for:
1. Independent tiny forward parity, float32 max absolute error `<=1e-6`.
2. Learned stable ties, duplicate hash routes counted once, empty experts skipped.
3. Input and gate/score gradient parity with `atol=rtol=1e-5`; all finite and non-zero where expected.
4. Detached index contract; no scatter/index VJP failure.
5. Instrumentation proving each expert sees exact unique routed-token count `R_e`, not full `T`; empty experts never called; `sum R_e <= T*K`.
6. Separate-process bounded-growth probe and no-shard real-dimension routed-op active delta `<2 GiB`.
7. Real-config formula contract: old retained floors 24/40 GiB, new routed-op budget 1.25 GiB, process ceiling 300 GiB.
8. Existing LoRA backward: non-empty tree, every leaf finite, at least one non-zero gradient.
9. Every verdict-participating test tracked by git.
10. ADR 0025 and technical-spec command updated without touching unrelated paths.

## Full-model boundary
The implementation Coder slice must not load model shards or run the full model. Reviewer PASS + Tester GREEN must precede any real smoke.

After double-green, authorize exactly one 4096 smoke:
- `mx.disable_compile()` before trainer wrapper;
- `mx.set_memory_limit(400_000_000_000)`;
- one validation batch;
- per-iteration reporting and memory telemetry;
- pre-backward peak STOP threshold 340GB;
- no shorter fallback.

## STOP conditions
Carry forward Architect §12 exactly, including parity/tolerance failures, duplicate-route semantics, unsupported index VJP, need for FROZEN edit, memory probe >=2GiB, global compile side effect, LoRA-gradient regression, untracked test, real peak >=340GB, OOM/non-finite loss/adapter failure.

## Canonical docs
Update only the relevant Story 13.3b-5 requirements/AC in `docs/backlog.md`. Preserve historical failed-run evidence and unrelated backlog content.

## Deliverable
Write `agent-output/cmux-13-3b/requirements-13-3b-5d-routed-fp4.md` with exact user story, implementation boundary, RED/GREEN AC, review/test gate, real-smoke command policy, STOP conditions, and tracking requirement.

Create `.cmux-status/ba.done` only on successful completion. End with `{"status":"ok","role":"BA"}` or error JSON.
