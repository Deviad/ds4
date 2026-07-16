# Story 13.3b-5c — BA thin re-pin: 400GB MLX memory-budgeted smoke

## Goal
Re-pin Story 13.3b-5 acceptance criteria and run authorization to Architect r2's evidence-backed memory-safe command. No new product scope or design work.

## User model/budget override
Use the current BA pane model `openai-codex/gpt-5.6-sol` high. The user selected current pane models because other provider budgets are exhausted. Do not switch models.

## Read first
1. `agent-output/cmux-13-3b/architecture-13-3b-5c-memory-safe-backward.md`
2. `agent-output/cmux-13-3b/coder-13-3b-5b-stop.md`
3. `agent-output/cmux-13-3b/requirements-13-3b-5.md`
4. Relevant Story 13.3b-5 section in `docs/backlog.md`

## Locked evidence
- 4096 validation has zero zero-target validation records and finite loss `19.116`.
- 2048 has 174/819 zero-target validation records after truncation; its NaN is proven completion-mask division by zero.
- Validation clears MLX cache after every batch; reducing validation batches shortens runtime but is not the OOM fix.
- First-backward memory floor is dominated by all-256-expert FP4 work inside a decoder-layer checkpoint.
- Installed MLX supports `mx.set_memory_limit`; a real-shaped FP4 backward probe showed lower graph peaks while preserving finite gradients.
- Architect chooses exactly `mx.set_memory_limit(400_000_000_000)` with max length 4096, `--val-batches 1`, and `--steps-per-report 1`.

## Re-pin only these requirements
1. Authorize explicit MLX graph memory limit `400_000_000_000` before `mlx_lm.lora.main()`.
2. Retain max sequence length 4096 and completion-only `--mask-prompt` objective.
3. Authorize `--val-batches 1 --steps-per-report 1` for this smoke.
4. Remove/deprecate the blind 2048 fallback; no shorter-length retry in this slice.
5. Retain original backward GREEN contract unchanged:
   - finite validation/training loss;
   - non-empty gradient tree;
   - every LoRA leaf finite;
   - at least one non-zero LoRA gradient/update;
   - iteration 20 completes without OOM/crash;
   - fresh loadable adapter safetensors saved.
6. Output path is `/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke-memory400` and must be absent/empty before launch.
7. No production/FROZEN/dataset/model-shard edit is authorized.
8. If the 400GB run OOMs, emits NaN/Inf, or fails adapter validation: STOP; no automatic 3072/2048/1024 retry.

## Canonical docs
Update only the Story 13.3b-5 requirements/acceptance text in `docs/backlog.md` as needed to replace the stale 2048 fallback with this locked run. Preserve historical evidence and unrelated backlog entries.

## Deliverable
Write `agent-output/cmux-13-3b/requirements-13-3b-5c-memory400.md` with:
- exact user story and checkable AC;
- exact authorized command contract copied from Architect §3.2;
- resource preflight;
- STOP conditions;
- explicit statement that this is a run-only slice unless a new correctness failure forces re-entry through Architect.

Create `.cmux-status/ba.done` only on successful completion. End with terminal JSON in this pane only: `{"status":"ok","role":"BA"}` or error JSON.
