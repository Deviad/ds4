# Story 13.3b-5c — Coder finalize-only STOP evidence

## Goal
Finalize the already-terminal 400GB smoke result. Do not run any model, test suite, context-mode poll, or retry.

## User model/budget override
Use current `opencode-go/kimi-k2.6` high. Do not switch providers.

## Read
- `agent-output/cmux-13-3b/coder-13-3b-5c-memory400.log`
- `agent-output/cmux-13-3b/coder-13-3b-5c-memory400.rc`
- `agent-output/cmux-13-3b/requirements-13-3b-5c-memory400.md`
- `agent-output/cmux-13-3b/architecture-13-3b-5c-memory-safe-backward.md`

## Locked observed facts to verify
- Selected limit line: `MLX memory limit: previous=522268023193 selected=400000000000`.
- One-batch validation completed: `Iter 1: Val loss 18.110, Val took 98.094s`.
- First backward then failed with `[METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)`.
- RC file is `134`.
- Trainer process is gone.
- `/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke-memory400` contains only `adapter_config.json`; no `adapters.safetensors`.

## Required action
1. Verify facts with short direct reads only.
2. Write `agent-output/cmux-13-3b/coder-13-3b-5c-memory400-stop.md` containing:
   - clean preflight summary;
   - exact command/memory limit;
   - finite validation evidence;
   - exact OOM/exit evidence;
   - AC-by-AC verdict;
   - statement: no shorter fallback authorized or attempted;
   - required return to Architect for first-step memory telemetry and all-expert FP4 chunking redesign.
3. Do not create `.cmux-status/coder.done`; remove it if stale.
4. Do not edit or stage production/test files.
5. End with exactly one terminal JSON line in this pane: `{"status":"error","error":"400GB 4096 smoke OOM at first backward after finite validation loss 18.110; no adapter saved","role":"Coder"}`.
