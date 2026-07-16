# Story 13.3b-5b — Coder STOP-ESCALATE

## Reason
Both 4096 and 2048 seq-length smoke-train OOM under verified clean resources.

## Resource preflight (verified clean)
- No competing mlx_lm/quantization/DeepSeek/GLM processes.
- >300GiB free RAM, 182GiB free on `/Volumes/Data NVME`.
- No instance lock bypassed; no unrelated processes killed.

## 4096 run
- Launched: 2026-07-13 12:20:46
- Validation: `Iter 1: Val loss 19.116, Val took 2245.586s` (finite, matches prior clean forward)
- Crash: first training step OOM
- Error: `[METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)`
- Exit: 134 (Abort trap: 6)

## 2048 fallback run
- Launched: 2026-07-13 13:06:59
- Validation: `Iter 1: Val loss nan, Val took 1748.508s` (NaN — new signal, not seen in prior 4096 runs)
- Crash: first training step OOM
- Error: same `[METAL] Command buffer execution failed: Insufficient Memory`
- Exit: 134 (Abort trap: 6)

## Observations
- 4096 forward/validation is correct (finite loss 19.116), proving `stop_gradient` did not corrupt forward.
- Backward path never reached: both runs OOM before first optimizer step completes.
- 2048 validation produced NaN, which was not observed in the earlier 4096 clean forward (19.116). This may be truncation-related or a separate data/model issue at 2048.
- The prior resource-contention 2048 run (13.3b-5b previous attempt) was user-terminated at val step 2/25; it never reached training or OOM.

## Required escalation
Architect r2 needed to determine:
1. Whether a smaller seq-length (1024?) or batch-size reduction should be attempted.
2. Whether the 2048 NaN validation is a genuine new correctness issue or a truncation artifact.
3. Whether Metal memory pressure can be further reduced (e.g., disabling grad-checkpoint, though that increases RAM; or using CPU fallback for validation).

## AC status at STOP
- AC 1 (regression green): NOT RE-RUN — tests were green at preflight (50/2), but full suite not re-run after this resume because no code changed.
- AC 2 (finite losses): PARTIAL — 4096 val finite, 2048 val NaN, neither training loss reached.
- AC 3 (no scatter_axis VJP failure): NOT PROVEN — backward never reached.
- AC 4 (iter 20 complete): FAILED — both runs OOM before training step 1.
- AC 5 (gradient tree non-empty): NOT PROVEN.
- AC 6 (adapter saved): FAILED — no safetensors produced.
- AC 7 (vendor hash/pins): PASS — `5e11a9c4d82aeb24` correct, old hash `96c39168c78e5fd9` has zero hits.
- AC 8 (tracked test files): NOT VERIFIED in this resume — no new test files modified.

## Code state
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py:902` has `mx.stop_gradient` wrap.
- No other production edits made in this resume.
- Old hash `96c39168c78e5fd9` fully retired from tracked code/tests.
