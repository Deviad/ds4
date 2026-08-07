# DeepSeek V4 Flash 0731 — DSpark MTP Quantization + Joint LoRA Fine-Tuning Runbook

Complete reproducible steps from raw HF checkpoint to a quantized GGUF with inline
MTP drafter, ready for joint LoRA fine-tuning.

**Machine:** Mac Studio M3 Ultra, 512 GB unified memory
**Disk:** 3.6 TB NVME (`/Volumes/Data NVME`)
**Date:** 2026-08-01

---

## Prerequisites

| Tool | Location | Purpose |
|------|----------|---------|
| `deepseek4-quantize` | `/Users/spotted/projects/ds4/gguf-tools/deepseek4-quantize` | HF safetensors → GGUF quantizer |
| `ds4` engine (fc9 build) | `/Users/spotted/projects/ds4-story2-fc9/ds4` | Imatrix collection + inference |
| HF checkpoint (0731) | `/Volumes/Data NVME/huggingface/DeepSeek-V4-Flash-0731/` | Source weights (FP8, 166.9 GB) |
| Template GGUF (Layers37-42) | `/Users/spotted/projects/ds4/gguf/DeepSeek-V4-Flash-Layers37-42Q4KExperts-...gguf` | Quantization template (91 GB) |
| Template GGUF (Q4KExperts) | `/Users/spotted/projects/ds4/gguf/DeepSeek-V4-Flash-Q4KExperts-...gguf` | Imatrix source template (153 GB) |
| MTP drafter GGUF (old) | `/Users/spotted/projects/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` | MTP tensor reference (3.5 GB) |
| Imatrix calibration dataset | `/Users/spotted/projects/ds4/gguf-tools/imatrix/dataset/rendered_prompts.txt` | 4682 prompts, ~2.91M tokens |
| MLX venv | `/Users/spotted/projects/ds4-finetuning/python-envs/mlx/.venv/bin/python3` | MLX 0.31.2 + mlx-lm 0.31.3 |
| ds4-finetuning worktree | `/Users/spotted/projects/ds4-finetuning/` | LoRA training pipeline |

---

## Step 0: Fix config.json (one-time, manual)

The 0731 HF checkpoint's `config.json` is missing `layer_types` and `mlp_layer_types`
fields that the MLX model definition requires. Derive them from `compress_ratios`:

```python
import json

config_path = "/Volumes/Data NVME/huggingface/DeepSeek-V4-Flash-0731/config.json"
c = json.load(open(config_path))

COMPRESS_RATIO_TO_LAYER_TYPE = {
    0: "sliding_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}
ratios = c["compress_ratios"]
n = c["num_hidden_layers"]  # 43
c["layer_types"] = [COMPRESS_RATIO_TO_LAYER_TYPE[r] for r in ratios[:n]]

n_hash = c.get("num_hash_layers", 0)
c["mlp_layer_types"] = (["hash_moe"] * min(n, n_hash) + ["moe"] * max(0, n - n_hash))[:n]

with open(config_path, "w") as f:
    json.dump(c, f, indent=2)
```

**Result:** `layer_types` (43 entries: 2 sliding, 21 compressed_sparse, 20 heavily_compressed)
and `mlp_layer_types` (43 entries: 3 hash_moe, 40 moe) added to config.json.

---

## Step 1: Quantize Q4_K imatrix source GGUF (~4 hours)

Creates a high-quality Q4_K experts GGUF from the new 0731 model. Used as the
model for imatrix collection (less quantized = more accurate activation statistics).

```bash
/Users/spotted/projects/ds4/gguf-tools/deepseek4-quantize \
  --hf "/Volumes/Data NVME/huggingface/DeepSeek-V4-Flash-0731" \
  --template "/Users/spotted/projects/ds4/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf" \
  --out "/Volumes/Data NVME/mlx-ft/ds4/ds4flash-0731-q4k-imatrix-src.gguf" \
  --overwrite
```

**Output:** `ds4flash-0731-q4k-imatrix-src.gguf` (~153 GB)
**Template provides:** Q4_K experts, F16 HC/compressor/indexer, Q8 attention/shared/output

---

## Step 2: Collect imatrix (~1.5 hours)

Runs the ds4 engine on the calibration dataset, collecting routed-MoE activation
statistics (sum of squared activations per expert column).

```bash
/Users/spotted/projects/ds4-story2-fc9/ds4 \
  -m "/Volumes/Data NVME/mlx-ft/ds4/ds4flash-0731-q4k-imatrix-src.gguf" \
  --imatrix-dataset /Users/spotted/projects/ds4/gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out "/Volumes/Data NVME/mlx-ft/ds4/ds4flash-0731-routed-moe-imatrix.dat" \
  --ctx 32768
```

**Output:** `ds4flash-0731-routed-moe-imatrix.dat` (~450 MB, 129 entries, 112.7M values)

### Step 2a: Fix non-finite imatrix values (manual, required)

The imatrix collection produces non-finite values (NaN/Inf) in early layers
(attention sink overflow). The quantizer rejects these. Fix by replacing with 0.0:

```python
import numpy as np
import struct

path = "/Volumes/Data NVME/mlx-ft/ds4/ds4flash-0731-routed-moe-imatrix.dat"
data = open(path, "rb").read()

offset = 0
n_entries = struct.unpack_from("<i", data, offset)[0]
offset += 4

fixed_data = bytearray()
fixed_data += struct.pack("<i", n_entries)
n_fixed = 0

for i in range(n_entries):
    name_len = struct.unpack_from("<i", data, offset)[0]; offset += 4
    name = data[offset:offset+name_len]; offset += name_len
    n_rows = struct.unpack_from("<i", data, offset)[0]; offset += 4
    n_values = struct.unpack_from("<i", data, offset)[0]; offset += 4
    total_vals = n_rows * n_values
    values = np.frombuffer(data, dtype=np.float32, count=total_vals, offset=offset).copy()
    offset += total_vals * 4

    bad = ~np.isfinite(values)
    n_fixed += bad.sum()
    values[bad] = 0.0

    fixed_data += struct.pack("<i", name_len) + name
    fixed_data += struct.pack("<i", n_rows) + struct.pack("<i", n_values)
    fixed_data += values.tobytes()

with open(path, "wb") as f:
    f.write(fixed_data)
print(f"Fixed {n_fixed} non-finite values")
```

**Observed:** 7,700,484 non-finite values (6.83%), concentrated in layers 1-2 (100%)
and partially in layer 3. Caused by attention sink overflow during collection.

---

## Step 3: Quantize with Layers37-42 recipe + imatrix (~7.5 hours)

Produces the final production GGUF with the mixed-precision recipe:
- Layers 37-42: Q4_K experts (higher quality, closest to output)
- Layers 0-36: IQ2_XXS gate/up + Q2_K down (aggressive compression)
- Attention projections: Q8_0
- Shared experts: Q8_0
- Output head: Q8_0

```bash
/Users/spotted/projects/ds4/gguf-tools/deepseek4-quantize \
  --hf "/Volumes/Data NVME/huggingface/DeepSeek-V4-Flash-0731" \
  --template "/Users/spotted/projects/ds4/gguf/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf" \
  --out "/Volumes/Data NVME/mlx-ft/ds4/ds4flash-0731-layers37-42-imatrix.gguf" \
  --imatrix "/Volumes/Data NVME/mlx-ft/ds4/ds4flash-0731-routed-moe-imatrix.dat" \
  --overwrite
```

**Output:** `ds4flash-0731-layers37-42-imatrix.gguf` (~91 GB)

---

## Step 4: Cleanup Q4_K imatrix source (reclaim ~153 GB)

```bash
rm "/Volumes/Data NVME/mlx-ft/ds4/ds4flash-0731-q4k-imatrix-src.gguf"
```

---

## Step 5: Create MTP drafter GGUF from 0731 HF checkpoint (TODO)

Extract the 4705 `mtp.*` tensors from the 0731 HF checkpoint, dequantize FP8,
quantize to GGUF types (Q4_K experts, Q8_0 attention, F32 norms/HC), and write
a standalone drafter GGUF using the old MTP GGUF as structural template.

**Input:** HF checkpoint `mtp.*` tensors + old MTP GGUF template (32 tensors)
**Output:** `ds4flash-0731-dspark-support.gguf` (~3.5 GB)

---

## Step 6: Sidecar test (TODO)

Test the new target + drafter sidecar to validate MTP tensors work and measure
baseline acceptance rate.

```bash
DS4_DSPARK_ENABLE=1 DS4_DSPARK_STATS=1 DS4_DSPARK_CONFIDENCE_THRESHOLD=0 \
/Users/spotted/projects/ds4-story2-fc9/ds4 \
  -m "/Volumes/Data NVME/mlx-ft/ds4/ds4flash-0731-layers37-42-imatrix.gguf" \
  --mtp "/Volumes/Data NVME/mlx-ft/ds4/ds4flash-0731-dspark-support.gguf" \
  --prompt "Write a Python function that implements a binary search tree." \
  --nothink -n 256 --temp 0 --seed 42
```

**Expected:** DSpark stats with acceptance rate (target: 60-70% for matched pair)

---

## Step 7: Inline MTP engine modification (TODO)

Modify the ds4 engine to check the main model for `mtp.*` tensors. If found,
bind them directly — no `--mtp` sidecar needed. Backward compatible:
- `--mtp` provided → sidecar (existing behavior)
- No `--mtp` → scan main model for MTP → inline if found
- Neither → no DSpark (existing behavior)

---

## Step 8: Splice MTP into main GGUF (TODO)

Append MTP tensors into the main GGUF so the engine loads target + drafter
from a single file.

---

## Step 9: Joint QLoRA fine-tuning (TODO)

Fine-tune target + drafter together using the ds4-finetuning pipeline:

```bash
cd /Users/spotted/projects/ds4-finetuning
python scripts/finetune_e2e.py \
  --mlx-work "/Volumes/Data NVME/mlx-ft/ds4" \
  --iters 400 --lr 5e-5 --rank 8
```

Proven parameters (from 2026-07-25 run):
- 400 iterations (overfitting starts after ~400 at lr 5e-5)
- Learning rate: 5e-5, rank 8, scale 20
- Targets: q_a_proj, q_b_proj, kv_proj
- Batch size 1, max sequence 4096, prompt masking, gradient checkpointing

---

## Quantization recipe reference

From `quants.c` — the quantizer supports 4 quantized output types:

| Type | Block | Bytes/block | ~Bits/weight | Imatrix |
|------|-------|-------------|--------------|---------|
| `q8_0` | 32 | 34 | 8.5 | No |
| `q4_k` | 256 | 144 | 4.5 | No |
| `q2_k` | 256 | 84 | 2.6 | No |
| `iq2_xxs` | 256 | 66 | 2.06 | **Yes** |

Passthrough: `f32` (4B), `f16` (2B), `bf16` (2B), `i32` (4B)

Per-component flags: `--routed-w1`, `--routed-w2`, `--routed-w3`, `--experts`,
`--attention-proj`, `--attention`, `--shared`, `--embedding`, `--output`,
`--dense`, `--tensor-type PFX=TYPE`

---

## Disk budget

| Artifact | Size | Status |
|----------|------|--------|
| HF checkpoint (0731, FP8) | 166.9 GB | ✅ downloaded |
| Q4_K imatrix source | 153 GB | ✅ created → deleted after step 4 |
| Imatrix .dat | 450 MB | ✅ created + fixed |
| Layers37-42 final GGUF | ~91 GB | ⏳ quantizing |
| MTP drafter GGUF | ~3.5 GB | TODO |
| Peak disk usage | ~411 GB | during steps 1-3 |
| Final disk usage | ~261 GB | after cleanup |

---

## Key findings from this session

1. **Metal MoE bug (fixed):** Stale Metal shader cache on macOS 26 caused zero MoE
   output. Fixed by touching `metal/moe.metal` to invalidate the cache.

2. **Imatrix non-finite values:** Layers 1-2 produce 100% non-finite imatrix values
   (attention sink overflow). Must be replaced with 0.0 before quantization.

3. **config.json missing fields:** The 0731 config.json lacks `layer_types` and
   `mlp_layer_types`. Must be derived from `compress_ratios` before MLX loading.

4. **Sidecar vs inline:** Sidecar pattern uses two separate memory mappings,
   causing TLB pressure and per-dispatch buffer switching overhead. Inline MTP
   (single GGUF) eliminates this.

5. **Drafter acceptance ceiling:** LoRA on frozen experts caps at ~19% acceptance.
   Joint training (target + drafter together) is needed for 60-70%.

---

## Joint trainer adapter readiness (implementation verified)

The orchestration spine's `train-joint` adapter invokes the actual component with
one direct argv vector:

```text
< pinned interpreter > -m ds4_ft_mlx.joint_train --config < absolute config >
```

`--dry-run` appears only for `dry-run` previews or an explicitly marked rehearsal.
No shell command string, nested quoting, or inherited environment is used.
The adapter records component repository branch, staged-diff digest, interpreter
identity, component-config identity, source-mode/tool identity, expected report
path, and report schema before launch.

Dry-run completion requires the component report to prove:
`execution_allowed=false`, `payload_loaded=false`, and `training_started=false`.
Production completion additionally requires report schema agreement, `iters=400`,
non-empty main and MTP trainable groups, full-tuned MTP tensor evidence,
completion evidence, and immutable identity agreement. Missing, stale, malformed,
nonzero, or mismatched reports fail closed.

Verified harmless rehearsal:
- actual `ds4_ft_mlx.joint_train` entrypoint;
- temporary JSON-only joint-base metadata and dataset manifest;
- no safetensors payload, model load, or training;
- no component report/checkpoint/output write;
- child exit and launch-before-log ordering observed.

Implementation status: adapter and rehearsal tests green; actual 400-iteration
training, fusion, export, imatrix, quantization, and final GGUF remain unexecuted.
