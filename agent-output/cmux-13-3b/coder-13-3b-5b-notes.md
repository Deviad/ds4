# Story 13.3b-5b — Coder notes (clean-resource resume)

## Edit applied (pre-existing from prior session)
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py:902`
  - `scatter_indices = mx.stop_gradient(mx.expand_dims(safe_indices, 1))`
  - Verified present at launch.

## sha-pin cascade (pre-existing from prior session)
- `tests/test_numpy_real_forward_reference_composition.py:397` → `5e11a9c4d82aeb24`
- `tests/test_deepseek_v4_real_config_reference_forward.py:383` → `5e11a9c4d82aeb24`
- `python-envs/mlx/src/ds4_ft_mlx/real_forward_intermediate_dump.py:61` → `5e11a9c4d82aeb24`
- Old hash `96c39168c78e5fd9`: zero tracked hits at launch.

## ADR 0026 Amendment 1 (pre-existing from prior session)
- Present in `docs/adr/0026-csa-hca-indexer-mlx-real-port.md`.

## Resource preflight
- No competing processes at launch.
- >300GiB free RAM, 182GiB free on `/Volumes/Data NVME`.

## 4096 smoke-train
- Launched 12:20:46, PID 7472.
- Validation: `Iter 1: Val loss 19.116, Val took 2245.586s` (finite).
- OOM at first training step: `[METAL] Command buffer execution failed: Insufficient Memory`.
- Exit 134.

## 2048 fallback
- Launched 13:06:59, PID 7114.
- Validation: `Iter 1: Val loss nan, Val took 1748.508s` (NaN — new).
- OOM at first training step: same Metal Insufficient Memory.
- Exit 134.

## AC verdict
- AC 1 (regression): preflight 50/2 green; full suite not re-run (no code change in resume).
- AC 2 (finite loss): 4096 val finite; 2048 val NaN; training loss never reached.
- AC 3 (no VJP failure): NOT PROVEN — backward never reached.
- AC 4 (iter 20): FAILED — both OOM.
- AC 5 (gradients): NOT PROVEN.
- AC 6 (adapter saved): FAILED — no safetensors.
- AC 7 (hash pins): PASS.
- AC 8 (tracked tests): `git ls-files` shows all four test files tracked.

## Outcome
STOP-ESCALATE — both 4096 and 2048 OOM under clean resources.
Architect r2 required for next steps.
