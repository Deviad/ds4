# Story 13.3b-5c — 400 GB MLX memory-budgeted smoke (BA thin re-pin)

**Status:** GO — run-only Coder slice authorized. No new product scope or design work.

## User story

**As a** training engineer (WHO), **I want** `mlx_lm.lora --train --iters 20` to run against the remapped+quantized real 43-layer DeepSeek V4 Flash checkpoint (nn port + FP4 experts) with LoRA on attention q_a/q_b/kv under an explicit 400 GB MLX graph-memory limit (WHAT), **so that** the first real gradient flow through the DeepSeek V4 nn port is proven end-to-end without changing the completion-only objective (WHY).

## Locked run requirements

1. Set the MLX graph memory limit to exactly `400_000_000_000` bytes before `mlx_lm.lora.main()`.
2. Retain `--max-seq-length 4096` and completion-only `--mask-prompt`.
3. Use `--val-batches 1 --steps-per-report 1` for this smoke.
4. Do not use the stale blind 2048 fallback. No 3072/2048/1536/1024 retry is authorized in this slice.
5. Use `/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke-memory400`; it must be absent or empty before launch.
6. Do not edit production code, FROZEN primitives, dataset files, model shards, site-packages, SSD streaming, CUDA, distributed inference, or the default Metal inference path.

## Exact authorized command contract

Copied from Architect §3.2 without modification:

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

The wrapper must preserve `mlx_lm.lora` argument parsing. Do not patch site-packages or add a permanent orchestrator default before this run proves the setting.

## Resource preflight and evidence capture

Before launch, verify and record:

- no concurrent full-model `mlx_lm`, DS4, conversion, or quantization process;
- Apple M3 Ultra, 512 GiB physical memory, and 464 GiB recommended working set;
- `/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke-memory400` is absent or empty, with no stale safetensors accepted;
- existing model, dataset, and LoRA-target gates pass;
- tokenizer preflight at 4096 finds zero zero-target records for the selected validation record and zero across all 819 validation records;
- exact command, start time, PID, and complete stdout/stderr log are captured.

Required run evidence:

- `MLX memory limit` line showing selected limit `400000000000`;
- finite one-batch validation loss;
- finite training loss at iteration 1 and every iteration through 20;
- peak-memory reports;
- no Metal OOM, scatter/index VJP failure, NaN, or Inf;
- final adapter-save line.

## Acceptance criteria

Run is GREEN only if all criteria hold:

1. Initial validation loss is finite.
2. First `step()` completes backward and optimizer update; iteration-1 training loss is finite.
3. Gradient tree is non-empty.
4. Every LoRA gradient leaf is finite.
5. At least one LoRA gradient/update is non-zero.
6. Iteration 20 completes without OOM or crash, with finite training losses throughout.
7. `/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke-memory400/adapters.safetensors` is freshly written, loads successfully, contains only expected LoRA adapter tensors, all values are finite, and at least one delta is non-zero.
8. Existing focused backward tests remain green, and no production/FROZEN/dataset/model-shard/SSD/CUDA/distributed/default-Metal path changes occur.

These criteria preserve the original backward GREEN contract: finite validation/training loss; non-empty gradient tree; every LoRA leaf finite; at least one non-zero LoRA gradient/update; iteration 20 completes without OOM/crash; and a fresh loadable adapter safetensors file is saved.

## STOP conditions

STOP immediately and do not retry at 3072/2048/1536/1024 if any condition occurs:

- the 400 GB run OOMs before or during first backward;
- validation or training loss is NaN/Inf;
- scatter/index VJP or another correctness exception appears;
- adapter output is absent, stale, unloadable, non-finite, or all zero;
- resource preflight finds another full-model process;
- any existing model, dataset, LoRA-target, or tokenizer preflight gate fails.

After STOP, return through Architect with first-step memory telemetry and failure evidence. Do not apply dequant `stop_gradient`, nested per-expert checkpointing, dataset filtering, shorter-length fallback, or a FROZEN-body edit without new architecture and ADR authorization.

## Slice boundary

This is a run-only slice. No implementation edit is authorized unless a new correctness failure forces re-entry through Architect and a newly approved coding slice.
