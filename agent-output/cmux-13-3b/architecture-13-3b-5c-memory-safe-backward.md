# Story 13.3b-5c — memory-safe first backward architecture

**Status:** GO for one thin BA re-pin, then one run-only Coder slice.  
**Decision:** Option E — keep the real 4096 completion-only objective and run MLX with an explicit `mx.set_memory_limit(400_000_000_000)` before `mlx_lm.lora.main()`. Use one validation batch and per-iteration reporting only to reduce wall time and expose the first completed backward; neither is the OOM fix. Do not edit model math, MLX-LM, dataset, or model shards.

## 1. Evidence and root-cause findings

### 1.1 Observed clean-resource failures

- 4096: validation finite at `19.116` over 25 batches; first training step aborts with Metal insufficient memory, exit 134.
- 2048: validation `NaN`; first training step aborts with the same Metal insufficient-memory error, exit 134.
- Batch size 1 and layer gradient checkpointing were already active. No adapter or training loss was produced.

These runs exclude ordinary resource contention and show that sequence reduction to 2048 is insufficient.

### 1.2 Validation does not leave the allocator cache full

Installed versions: MLX-LM `0.31.3`, MLX `0.31.2`.

- `mlx_lm/tuner/trainer.py:20-22` calls `mx.clear_cache()` whenever cache memory exceeds the threshold.
- Evaluation calls that helper after every evaluated batch (`trainer.py:176-209`). Its default threshold is zero (`:184`), so every validation batch ends with a cache clear.
- Training also clears after every completed step (`trainer.py:329`).
- `mlx_lm/lora.py:195` exposes `--clear-cache-threshold`, but `train_model()` omits `args.clear_cache_threshold` when constructing `TrainingArgs` (`lora.py:261-276`). The effective training threshold therefore remains zero. This omission makes the flag ineffective, but accidentally gives the most aggressive cache clearing already.
- Validation is an ordinary uncompiled forward. Only the training `step` is wrapped in `mx.compile` (`trainer.py:246-260`).

Conclusion: 25 validation batches cost about 37 minutes but are not supported as the cause of the first-backward peak. A post-validation `mx.clear_cache()` is already performed. `--val-batches 1` is useful for run cost, not as the memory fix.

`--val-batches 0` is invalid as a validation-disable mechanism: `evaluate()` executes zero batches, then computes `0 / 0` at `trainer.py:211-214`, yielding NaN. Validation still runs because the validation dataset itself remains truthy (`trainer.py:282-299`).

### 1.3 2048 NaN is all-masked-token division by zero

Exact tokenizer/dataset-only reproduction used the installed tokenizer, `CompletionsDataset`, `--mask-prompt`, and trainer truncation/mask algebra. No model was loaded.

`default_loss()` constructs the completion mask and divides by `ntoks = mask.sum()` without a zero guard (`trainer.py:86-99`). `iterate_batches()` truncates token arrays and total lengths to `max_seq_length` but does not clamp prompt offsets (`trainer.py:118-170`). For a tokenized record of total length `T`, prompt offset `O`, and limit `L`, effective target tokens are:

```text
max(0, min(T, L) - O)
```

Validation-set results (819 records):

| max length | zero-target records | truncated records |
|---:|---:|---:|
| 4096 | 0 | 31 |
| 3072 | 2 | 96 |
| 2048 | 174 | 285 |
| 1536 | 229 | 331 |
| 1024 | 234 | 382 |

The 2048 log contains 12 validation truncation warnings. Six warning lengths map uniquely to sampled zero-target records: `(total, offset)` = `(2717,2375)`, `(2696,2181)`, `(3482,2326)`, `(3327,2075)`, `(3363,2228)`, `(3766,2314)`. Each has `offset > 2048`. Its per-batch loss divides zero by zero; `all_losses += loss * toks` cannot recover because `NaN * 0` remains NaN. This proves the observed 2048 validation NaN is masking/truncation, not model numeric drift.

4096 has no zero-target validation record, so retaining 4096 preserves the intended completion-only objective without filtering.

### 1.4 Backward peak is active graph memory, dominated by all-expert FP4 work

Current FP4 routed-expert path:

- `_dequantize_fp4_block_scale_mlx()` expands packed FP4 and BF16 scales into FP32 dense arrays (`deepseek_v4.py:163-203`).
- `DeepseekV4FP4Experts.forward_one()` dequantizes all three matrices (`w1`, `w2`, `w3`) (`deepseek_v4_nn.py:416-425`).
- `SparseMoeBlockNN.__call__()` loops over all 256 experts and computes every expert before masking by routing selection (`deepseek_v4_nn.py:500-511`).
- MLX-LM checkpoints `type(model.layers[0]).__call__` (`trainer.py:25-38,238`), so the existing boundary is one complete decoder layer. It bounds retained intermediates across layers, but not the 256-expert graph inside the layer during recomputation/backward.

Verified lower bounds from real dimensions, excluding dequant temporary arrays, attention, residuals, model weights, and allocator overhead:

- One FP32 dense 2048×4096 matrix: 32 MiB.
- Three dense matrices × 256 experts: **24 GiB per layer**, independent of sequence length.
- Gate/up/hidden plus output activation lower bound: **20 GiB/layer at 2048**, **40 GiB/layer at 4096**.
- The dequant primitive additionally constructs nibble indices, LUT outputs, stacked FP4 values, repeated FP32 scales, and the FP32 product; 24 GiB is therefore only the dense-output floor.

This explains why halving sequence length did not remove the OOM: it halves sequence-dependent activations but leaves the expert dequant/model component unchanged.

A light exact-function probe (4 real-shaped FP4 experts, hidden 4096, intermediate 2048, sequence 2048, checkpointed backward) measured 1.544 GB peak under the default memory policy. `mx.stop_gradient` around dequant outputs remained 1.544 GB: no benefit. A nested per-expert checkpoint probe was worse at sequence 128 (1.422 GB versus 1.167 GB). These probes do not predict the full-model absolute peak, but they reject speculative stop-gradient/per-expert-checkpoint edits as the next slice.

### 1.5 Supported memory control

Installed MLX exposes and documents:

- `mx.set_memory_limit(limit)`: graph-evaluation memory guideline.
- `mx.set_cache_limit(limit)`: free-cache limit.
- `mx.clear_cache()`: releases inactive cached buffers only.

No installed Python/native-library evidence was found for `MLX_METAL_CACHE_LIMIT` or another `MLX_*CACHE/MEMORY` environment variable. Do not rely on an undocumented environment variable.

The process default reported by `mx.set_memory_limit()` was `522,268,023,193` bytes. Device facts:

- physical: `549,755,813,888` bytes (512 GiB);
- recommended working set: `498,216,206,336` bytes (464 GiB);
- trainer wires the recommended limit at `trainer.py:229`, but does not lower the graph memory limit.

A synthetic exact-FP4 backward probe showed the supported API actively changes graph scheduling while preserving finite gradients:

| memory limit | measured peak |
|---:|---:|
| default | 1.544 GB |
| 1.2 GB | 1.238 GB |
| 0.8 GB | 0.879 GB |
| 0.5 GB | 0.837 GB |

Thus an explicit 400 GB graph limit is the smallest evidence-backed intervention. It leaves 98.2 GB (91.5 GiB) below the recommended working set and 149.8 GB (139.5 GiB) below physical memory. It may increase recomputation/runtime, but does not change model math, data, gradients, or optimizer semantics.

## 2. Option adjudication

| Option | Minimality | Semantic safety | Expected memory impact | Decision |
|---|---|---|---|---|
| A. 4096, fewer validation batches/cache clear | High | `val_batches=1` safe; `0` produces NaN | Low: cache already cleared after every batch | Use `1` only to shorten run; reject as OOM fix |
| B. Intermediate sequence length | High | Unsafe without zero-target handling; 3072 already has 2 invalid validation records | Uncertain; 2048 still OOM | Reject |
| C. Filter/preprocess zero-target examples | Medium | Changes sampled objective/population; unnecessary at 4096 | Fixes NaN only, not OOM | Reject |
| D. Finer checkpoint or stop-gradient dequant | Medium/high | Requires new proof; FROZEN dequant edit requires new ADR sanction | Probes show no benefit or regression for proposed forms | Reject for next slice |
| E. Supported MLX memory limit | Highest for run-only change | Safe scheduling/resource policy | Probe-confirmed peak reduction | **Choose** |

## 3. One next slice: 4096 memory-budgeted run

### 3.1 BA re-pin

BA must thinly re-pin requirements before Coder because current §7 prescribes 2048 as the OOM fallback, but 2048 is now proven to corrupt completion-only validation via zero-target truncation. Re-pin only:

1. authorize the explicit 400 GB MLX graph memory limit;
2. retain max sequence length 4096;
3. authorize `--val-batches 1 --steps-per-report 1` for this smoke;
4. remove the blind 2048 fallback;
5. retain AC §3(a)-(f) unchanged.

### 3.2 Exact run command

No production/FROZEN/dataset/model-shard edit. Run after normal resource preflight and existing dataset/model/LoRA-target gates:

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

The wrapper form is proven to preserve `mlx_lm.lora` argument parsing. Do not patch site-packages. If reproducibility requires an orchestrator entry later, add a project-owned wrapper only after this run proves the setting; do not turn an unproven budget into a permanent default.

## 4. Resource preflight and evidence capture

Before launch:

- no concurrent full-model `mlx_lm`, DS4, conversion, or quantization process;
- device remains Apple M3 Ultra, physical 512 GiB, recommended working set 464 GiB;
- adapter output path absent or empty; never accept stale safetensors;
- current model/dataset/LoRA-target gates pass;
- capture command, start time, PID, and complete stdout/stderr log.

Required log evidence:

- selected memory-limit line;
- finite one-batch validation loss;
- finite training loss at iteration 1 and every iteration through 20;
- peak memory reports;
- no Metal OOM, scatter VJP, NaN, or Inf;
- final adapter-save line.

## 5. Acceptance criteria

1. Tokenizer preflight at 4096 reports zero zero-target records for the selected validation record and zero across all 819 validation records.
2. Initial validation loss finite.
3. First `step()` completes backward and optimizer update; iteration-1 training loss finite.
4. Iteration 20 completes without OOM/crash.
5. Gradient contract remains non-empty, all LoRA leaves finite, and at least one LoRA gradient non-zero. Existing focused backward tests remain green; post-run adapter inspection must show expected LoRA keys, all finite values, and at least one non-zero delta as the full-run concrete proof.
6. `/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke-memory400/adapters.safetensors` exists, is newly written, loads successfully, and contains only expected adapter tensors.
7. No production inference, FROZEN primitive, dataset, model shard, SSD, CUDA, distributed, or default Metal path changed.

## 6. STOP conditions and fallback

STOP; do not try 3072/2048/1536/1024 if any occurs:

- 400 GB run OOMs before or during first backward;
- validation or training loss is NaN/Inf;
- scatter/index VJP or another correctness exception appears;
- adapter is absent, stale, unloadable, non-finite, or all zero;
- resource preflight finds another full-model process.

Fallback after STOP: new Architect micro-slice with first-step memory telemetry and a proof-oriented redesign of all-expert FP4 execution/chunking. Do not apply dequant `stop_gradient`, nested per-expert checkpointing, dataset filtering, or a FROZEN-body edit based on current evidence. Any FROZEN-body proposal requires explicit new ADR sanction.
