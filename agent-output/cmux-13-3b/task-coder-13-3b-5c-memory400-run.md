# Story 13.3b-5c — Coder run-only: 400GB memory-budgeted LoRA smoke

## Goal
Execute and monitor the exact BA/Architect-authorized 4096 LoRA smoke with MLX graph memory limited to 400,000,000,000 bytes. Prove first backward and iteration 20 without changing model math, data, site-packages, or production code.

## User model/budget override
Use the current supported OpenCode Go model in this pane: `opencode-go/kimi-k2.6` high. Do not switch to Codex, Anthropic, or Neuralwatt.

## Read first
1. `agent-output/cmux-13-3b/requirements-13-3b-5c-memory400.md`
2. `agent-output/cmux-13-3b/architecture-13-3b-5c-memory-safe-backward.md`
3. `agent-output/cmux-13-3b/coder-13-3b-5b-stop.md`
4. `AGENTS.md`

## Scope
Run-only. No production/FROZEN/test/dataset/model-shard/site-packages edit.

### Resource preflight
Record before launch:
- no concurrent full-model MLX/DS4/quantization/conversion process;
- Apple M3 Ultra 512GiB, recommended working set 464GiB;
- free disk on `/Volumes/Data NVME`;
- output path absent/empty;
- model/dataset/LoRA-target/chat-template gates present;
- tokenizer-only probe confirms zero zero-target validation records at 4096.

If another full-model process exists: STOP as `PENDING-RESOURCE`; do not kill it.

### Output cleanup
Remove only stale `/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke-memory400` before launch. Do not touch `model-4bit`, dataset, GLM artifacts, or historical smoke outputs.

## Exact authorized command
Run exactly the command in BA requirements, with complete log/PID/RC capture. Suggested wrapper:

```bash
cd /Users/spotted/projects/ds4-finetuning
LOG=agent-output/cmux-13-3b/coder-13-3b-5c-memory400.log
PID=agent-output/cmux-13-3b/coder-13-3b-5c-memory400.pid
RC=agent-output/cmux-13-3b/coder-13-3b-5c-memory400.rc
rm -f "$LOG" "$PID" "$RC"
rm -rf '/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke-memory400'
nohup bash -c '
  cd "/Volumes/Data NVME/mlx-ft/ds4" &&
  unset SSLKEYLOGFILE &&
  . .venv/bin/activate &&
  python -c '\''import mlx.core as mx; limit=400_000_000_000; previous=mx.set_memory_limit(limit); print(f"MLX memory limit: previous={previous} selected={limit}", flush=True); from mlx_lm.lora import main; main()'\'' \
    --config lora-config.json \
    --model model-4bit \
    --train \
    --data "/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096" \
    --adapter-path adapters-smoke-memory400 \
    --fine-tune-type lora \
    --iters 20 \
    --batch-size 1 \
    --learning-rate 1e-5 \
    --max-seq-length 4096 \
    --mask-prompt \
    --grad-checkpoint \
    --val-batches 1 \
    --steps-per-report 1
  rc=$?
  echo "$rc" > "/Users/spotted/projects/ds4-finetuning/agent-output/cmux-13-3b/coder-13-3b-5c-memory400.rc"
  exit "$rc"
' > "$LOG" 2>&1 &
echo $! > "$PID"
caffeinate -i -w "$(cat "$PID")" &
```

Correct shell quoting if needed without changing command arguments. Poll with short bounded checks. Never start a second run concurrently.

## GREEN acceptance criteria
All required:
1. Log shows selected memory limit `400000000000`.
2. One-batch validation loss finite.
3. Iteration-1 backward/optimizer update completes; training loss finite.
4. Finite training loss reported every iteration through 20.
5. No Metal OOM, NaN/Inf, scatter/index VJP, or other correctness exception.
6. Gradient contract non-empty, all LoRA leaves finite, at least one update non-zero. Use existing focused backward tests plus saved-adapter inspection as concrete evidence.
7. Fresh `/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke-memory400/adapters.safetensors` exists, loads, contains expected LoRA keys only, all values finite, at least one non-zero tensor.
8. Focused tests remain green; run full regression suite and record exact counts if runtime permits after the successful smoke.
9. No production inference/FROZEN/dataset/model-shard/SSD/CUDA/distributed/default-Metal path changed by this run.

## STOP conditions
STOP immediately; no 3072/2048/1536/1024 fallback:
- OOM before/during first backward;
- NaN/Inf validation or training loss;
- scatter/index VJP or another correctness error;
- adapter absent/stale/unloadable/non-finite/all-zero;
- competing full-model process appears.

Write `agent-output/cmux-13-3b/coder-13-3b-5c-memory400-stop.md` with exact evidence and emit error JSON.

## Completion and staging
On GREEN:
- write `agent-output/cmux-13-3b/coder-13-3b-5c-memory400-notes.md` with per-AC proof;
- stage only the already-approved Story 13.3b-5b production fix/pins/ADR/template script plus current canonical notes required for review; do not stage unrelated dirty/untracked files;
- do not commit;
- create `.cmux-status/coder.done`;
- emit `{"status":"ok","role":"Coder"}` in this pane only.
