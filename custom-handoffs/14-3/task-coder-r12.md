# Story 14.3 — Coder r12 final five blockers

Read `custom-handoffs/standby/review.md` r11 in full, requirements.md, architecture.md, coder-notes-r11.

Close exactly these blockers:

1. Registry: stage `custom-handoffs/14-3/.pipeline-private/implementation-state.v1.tsv` with revision `14-3-coder-r11` and current functional hash before final gates; after any code/test changes supervisor will compute/register final deterministic four-file hash.
2. Exact paths: at direct activation entry point enforce pinned Phase-1 paths exactly: model `/Volumes/Data NVME/mlx-ft/ds4/model-4bit`, data `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096`, adapter `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-smoke`, config `/Volumes/Data NVME/mlx-ft/ds4/lora-config.json`. Keep synthetic unit helpers able to exercise preflight without real assets, but real activation/main must reject arbitrary paths with `failure_code: pinned-paths` before preflight. Add tracked test.
3. LoRA values: reject bool and nonfinite NaN/Inf for rank/scale/dropout, preserve exact canonical key set/ranges. Add direct NaN/bool regressions.
4. Argparse: capture useful parser error text (e.g. invalid segment-size) into marker/report while preserving return code 2 and `failure_code: argparse`; avoid relying on `str(SystemExit(2))`.
5. Cleanup warning: modify tracked test to inspect durable `agent-output/cmux-14-3/cleanup-warnings.json` bytes or stderr, using unchanged control/mutant oracle. Ensure warning sidecar/report is written deterministically even when report writing fails; do not claim pre-cleanup report contains late warnings unless it actually does.

No real assets, Phase 2, smoke/training/inference/backend/install/network, commit/push. Preserve Path A/provider/vendor bytes and exact-fork behavior. Run focused/finetune/source/provider suites and py_compile. Write `custom-handoffs/14-3/coder-notes-r12.md`; marker only complete, otherwise STOP.