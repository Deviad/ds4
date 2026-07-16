# Story 13.3b-5c — Coder STOP evidence (400 GB 4096 smoke)

## Preflight summary

- Device: Apple M3 Ultra, 512 GiB physical, 464 GiB recommended working set
- No concurrent full-model mlx_lm/DS4/conversion/quantization process detected
- Adapter output path `/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke-memory400` was absent before launch
- Model, dataset, and LoRA-target gates passed pre-launch

## Exact command and memory limit

```bash
cd '/Volumes/Data NVME/mlx-ft/ds4' && \
unset SSLKEYLOGFILE && \
. '/Volumes/Data NVME/mlx-ft/ds4/.venv/bin/activate' && \
python -c 'import mlx.core as mx; limit=400_000_000_000; previous=mx.set_memory_limit(limit); print(f"MLX memory limit: previous={previous} selected={limit}", flush=True); from mlx_lm.lora import main; main()' \
  --config '/Volumes/Data NVME/mlx-ft/ds4/lora-config.json' \
  --model '/Volumes/Data NVME/mlx-ft/ds4/model-4bit' \
  --train \
  --data '/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096' \
  --adapter-path '/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke-memory400' \
  --fine-tune-type lora \
  --iters 20 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --max-seq-length 4096 \
  --mask-prompt \
  --grad-checkpoint \
  --val-batches 1 \
  --steps-per-report 1
```

MLX memory limit line: `MLX memory limit: previous=522268023193 selected=400000000000`

## Finite validation evidence

One-batch validation completed:

```
Iter 1: Val loss 18.110, Val took 98.094s
```

Loss is finite (18.110). No NaN/Inf.

## Exact OOM and exit evidence

First backward then failed:

```
libc++abi: terminating due to uncaught exception of type std::runtime_error: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
bash: line 18: 85314 Abort trap: 6           python -c '...'
```

- Process exit: `Abort trap: 6` (signal SIGABRT)
- RC file: `134`
- Trainer process is gone; PID 85314 terminated

## Adapter output state

`/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke-memory400` contains only `adapter_config.json`; no `adapters.safetensors` was written.

## AC-by-AC verdict

| AC | Verdict | Evidence |
|---|---|---|
| 1. Initial validation loss finite | PASS | `Iter 1: Val loss 18.110` |
| 2. First `step()` completes backward and optimizer update; iteration-1 training loss finite | **FAIL / STOP** | OOM during first backward; no training loss produced |
| 3. Gradient tree is non-empty | **FAIL / STOP** | No gradient tree; process aborted before backward completion |
| 4. Every LoRA gradient leaf is finite | **FAIL / STOP** | No gradients computed |
| 5. At least one LoRA gradient/update is non-zero | **FAIL / STOP** | No gradients computed |
| 6. Iteration 20 completes without OOM or crash | **FAIL / STOP** | Process aborted at first backward, iteration 1 never completed |
| 7. `adapters.safetensors` freshly written, loads, expected keys, all finite, at least one non-zero delta | **FAIL / STOP** | `adapters.safetensors` absent; only `adapter_config.json` present |
| 8. No production/FROZEN/dataset/model-shard/SSD/CUDA/distributed/default-Metal path changes | PASS | Run-only slice; no files edited |

## No shorter fallback authorized or attempted

As per requirements §4 and §6: no 3072/2048/1536/1024 retry was authorized in this slice, and none was attempted after the 400 GB OOM. The 2048 fallback was explicitly removed from the requirements because 2048 validation NaN was proven to be zero-target masking/truncation, not a valid shorter-length path.

## Required next step

Return to **Architect** for a micro-slice focused on:
1. First-step memory telemetry (full per-op peak breakdown during the first backward)
2. Proof-oriented redesign of all-expert FP4 execution/chunking (the 256-expert dense dequant graph inside each layer is the identified dominant memory consumer, ~24 GiB/layer floor independent of sequence length)

Do not apply dequant `stop_gradient`, nested per-expert checkpointing, dataset filtering, shorter-length fallback, or FROZEN-body edit without new architecture and ADR authorization.
