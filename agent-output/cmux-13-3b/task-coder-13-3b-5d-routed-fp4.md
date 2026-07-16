# Story 13.3b-5d — Coder TDD: sparse routed-token FP4 backward

## Goal
Implement the Architect-selected training-only eager sparse routed FP4 operation with exact custom first-order VJP. Remove the all-256-expert/full-sequence backward graph while preserving routing, FP4, shared-expert, and gradient semantics.

## User model/budget override
Use current supported `opencode-go/kimi-k2.6` high. Do not switch providers/models.

## Read first
1. `agent-output/cmux-13-3b/requirements-13-3b-5d-routed-fp4.md`
2. `agent-output/cmux-13-3b/architecture-13-3b-5d-routed-fp4-backward.md`
3. `AGENTS.md`
4. `docs/architecture.md`, `docs/technical-spec.md`, ADR 0024, ADR 0025, ADR 0026
5. Existing `deepseek_v4_nn.py` expert/MoE tests and trainer-contract tests.

## Baseline integrity
At dispatch:
- FROZEN `deepseek_v4.py` SHA256: `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b`.
- `deepseek_v4_nn.py` SHA256: `ea181743e939ba3309a445f238cd6974a73b4ca834a9219aa0973010d3c08f3f`.
- Existing staged Story 13.3b-5b files must not be reverted or silently altered outside this task.

## TDD order — mandatory
1. Create tracked `tests/test_deepseek_v4_nn_sparse_routed_backward.py` with RED tests before production edit.
2. Run RED tests and record exact failures.
3. Implement minimal GREEN in `deepseek_v4_nn.py` only.
4. Refactor only while tests remain green.
5. Run focused regression, bounded-memory probes, then broader regression.

## Authorized production design
Implement exactly Architect §§2–4 and BA boundary:
- flatten leading token axes only;
- preserve existing `_scores` and `_select_indices`;
- stop-gradient and host-materialize only integer route metadata;
- build unique token rows per expert, collapsing duplicate expert ids within each token;
- skip empty experts;
- custom first-order routed function with differentiable `(x_flat, scores_flat)` and detached indices;
- forward: denominator from selected unique scores; evaluate one expert on `[R_e,H]`; weighted scatter-add in original token order; `mx.eval` after every expert;
- VJP: exact score derivative from Architect §4.4; use `mx.vjp(lambda z: experts.forward_one(eid,z), ...)` for expert-input derivative; accumulate `dx` and `dscores`; `mx.eval` after each expert;
- preserve shared expert over original full input, added once;
- construct closure at call time so replaced module arrays are not stale;
- no compile disable at module import/model construction.

Do not edit the FROZEN `deepseek_v4.py`, ADR 0024, model shards, datasets, site-packages, inference, SSD, CUDA, distributed, or default Metal paths.

## Required tests and proof
All BA AC apply. At minimum:
- independent deterministic forward parity `max_abs <=1e-6`;
- learned ties, duplicate hash routes once, empty expert skipped, correction bias selection-only, shared expert unchanged;
- input and score/gate gradient parity `atol=rtol=1e-5`, finite/non-zero;
- detached integer indices and duplicate-row scatter cotangent; no scatter VJP error;
- exact routed-token call counts and `sum R_e <= T*K`;
- separate-process dense versus sparse bounded-growth probe;
- no-shard real-dimension active delta `<2 GiB`;
- formula contract: route metadata 96KiB, one-expert dequant 96MiB, old 24/40GiB floors, new 1.25GiB budget, 300GiB process ceiling;
- existing focused LoRA backward non-empty/all-finite/at-least-one-nonzero;
- compile-gate test: compiled host routing fails as documented; wrapper-local `mx.disable_compile()` passes without import side effect.

## Documentation
- Amend `docs/adr/0025-real-trainable-nn-module-deepseek-v4-port.md` with sparse eager custom-first-order-VJP decision and wrapper-local compile disable.
- Update `docs/technical-spec.md` with exact later 4096 wrapper, 400GB memory limit, one validation batch, per-step telemetry, and 340GB STOP threshold.
- Do not edit `docs/backlog.md`; BA already owns that re-pin.

## Verification
No full model or shard load.

Run:
- new RED/GREEN test file;
- existing focused FP4/MoE/remap/LoRA trainer-contract suites;
- separate-process memory probes;
- full regression suite if feasible;
- `git diff --check`;
- direct SHA check proving FROZEN `deepseek_v4.py` remains exactly `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b`;
- `git ls-files --` for every verdict-participating test. The new test MUST be `git add`ed before completion.

STOP on any Architect §12 condition. Do not weaken tolerances or tests.

## Staging and handoff
Do not commit.

Stage only:
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py`;
- new tracked sparse-routed test;
- ADR 0025;
- `docs/technical-spec.md`;
- BA-owned `docs/backlog.md` if already changed by BA;
- preserve already staged 13.3b-5b files without modifying their content.

Write `agent-output/cmux-13-3b/coder-13-3b-5d-notes.md` with RED evidence, GREEN evidence, exact commands/counts, memory numbers, tracking proof, SHA/scope proof, and staged-file list.

On success create `.cmux-status/coder.done` and end with `{"status":"ok","role":"Coder"}`. On STOP write `coder-13-3b-5d-stop.md`, do not create success marker, and emit error JSON.
