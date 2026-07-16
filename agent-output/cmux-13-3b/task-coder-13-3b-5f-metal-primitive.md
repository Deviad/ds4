# Story 13.3b-5f — Coder TDD: opaque packed-FP4 Metal/MLX primitive

## Goal
Implement the exact Architect-selected six-kernel training-only primitive and replace the STOPPED Python nested-VJP routed path. No full model/shards/smoke.

## Model
Use user-authorized `openai-codex/gpt-5.5` high. Do not switch.

## Read in full
- `requirements-13-3b-5f-metal-primitive.md`
- `architecture-13-3b-5f-metal-primitive.md`
- ADR 0024, amended ADR 0025, ADR 0026, new ADR 0028
- current `deepseek_v4_nn.py`, `routed_fp4` tests, package config.

## TDD sequence
1. Record current Python E=2/4/8 E-scaled memory RED.
2. Add tracked `tests/test_deepseek_v4_nn_routed_fp4_metal.py` and primitive-absent RED tests before implementation.
3. Implement exact minimal GREEN.
4. Strengthen existing sparse test without weakening expectations.
5. Run focused, memory, regression, scope/tracking checks.

## Exact edit scope
Implement only:
- `python-envs/mlx/src/ds4_ft_mlx/metal/ds4_routed_fp4_train.metal`
- `python-envs/mlx/src/ds4_ft_mlx/routed_fp4_metal.py`
- `python-envs/mlx/pyproject.toml`
- routed branch only in `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py`
- new tracked Metal primitive test
- strengthened tracked sparse routed test
- `docs/technical-spec.md`
- stage BA/Architect canonical docs: backlog, architecture, ADR 0025, ADR 0028
- complete/stage the full approved 13.3b-5b predecessor chain.

No root `metal/*.metal`, C/CUDA/ROCm/SSD/distributed/default-Metal/inference, FROZEN `deepseek_v4.py` beyond the already-sanctioned staged 13.3b-5b state, ADR 0024, model shard, dataset, or site-package edit.

## Locked implementation
Follow Architect §§5–6 exactly:
- package-local `.metal` source with marked common header and six bodies:
  1. `ds4_fp4_pair_swiglu_forward`
  2. `ds4_fp4_down_forward`
  3. `ds4_fp4_down_input_vjp`
  4. `ds4_fp4_pair_swiglu_vjp_terms`
  5. `ds4_fp4_pair_input_vjp`
  6. `ds4_fp4_reduce_a`
- 8×8×32 tiling, 256-thread groups, FP32 accumulation, exact LSB-first LUT and BF16 direct scales;
- stride-aware `ensure_row_contiguous=False`, internal row gather, padded bounds;
- no full dense weight output/scratch;
- exact clip and clamp-boundary derivatives;
- narrow wrapper with MLX 0.31.2/Apple Metal fail-closed guard, source marker parser, lazy kernel cache, shape/dtype guards;
- one-expert forward and `(dx_e,a_e)` backward API;
- routed custom function with assignment concatenation, one denominator/routed/dx/score scatter, simplified score identity;
- packed/scales/rows captured constants, differentiable primals only `(x32,scores)`, no frozen cotangents;
- shared expert and route selection unchanged;
- no fallback to stopped Python `mx.vjp(forward_one)` path.

## GREEN gates
All BA/Architect gates, including:
- forward `max_abs<=1e-6` independent reference;
- dx/a and whole input/gate/score parity `1e-5`, exact clamp boundaries;
- no dense matrix exposure, DOT graph opaque CustomKernel;
- fixed A, E=2/4/8 peak spread `<=64 MiB`, fresh processes, operation peak not final active only;
- real H=4096/I=2048 no-shard outer-transform operation `<2 GiB` with finite y/dx/a;
- formula envelope 1504 MiB / <=2GiB;
- focused LoRA gradients non-empty/all-finite/non-zero;
- representative R={1,8,32,96} timing and smoke-feasibility estimate; STOP if plainly infeasible;
- full tracked non-live regressions;
- `make` production build check;
- direct protected-file/root-Metal hash manifest unchanged;
- every verdict test tracked by `git ls-files`;
- old hash `96c39168c78e5fd9` zero tracked hits;
- FROZEN SHA `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b`.

## Chain of custody/staging
Stage complete intended 13.3b-5b + 5f chain, including sanctioned FROZEN fix, three pins, ADR 0026, passthrough script, new primitive/code/tests/docs. Do not stage unrelated files. Do not commit.

## STOP
Honor every BA/Architect STOP. Do not relax tolerances, memory gates, or backend isolation. If custom kernel/API cannot pass, write `coder-13-3b-5f-stop.md`, no success marker, error JSON.

## Handoff
Write `agent-output/cmux-13-3b/coder-13-3b-5f-notes.md` with RED evidence, kernel/API details, exact parity maxima, E peaks/spread, real peak, timing, tests/counts, tracking, hashes, staged list.

Create `.cmux-status/coder.done` only on success. End with one unwrapped JSON line.
