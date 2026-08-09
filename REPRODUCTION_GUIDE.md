# DeepSeek V4 Flash 0731 — MTP Finetuning & Speed Research: Reproduction Guide

A consolidated guide to reproducing the work done on DeepSeek V4 Flash 0731:
the mixed-quantization GGUF, the joint main + MTP drafter finetune, the
inference speed research, and the published Hugging Face model.

- **Published model:** [`Deviad/DeepSeek-V4-Flash-0731-MTP-JointFT-GGUF`](https://huggingface.co/Deviad/DeepSeek-V4-Flash-0731-MTP-JointFT-GGUF)
- **Model card:** [`MODEL_CARD_Flash_L37-42Q4K_MTP_JointFT.md`](MODEL_CARD_Flash_L37-42Q4K_MTP_JointFT.md)
- **Quantize + finetune runbook:** [`runbook-0731-dspark-quantize-finetune.md`](runbook-0731-dspark-quantize-finetune.md)
- **Research results:** [`agent-output/mtp-40tps/results.md`](agent-output/mtp-40tps/results.md)

Machine used: **Mac Studio M3 Ultra, 512 GB unified memory**, 3.6 TB NVMe
(`/Volumes/Data NVME`).

---

## 1. What was done

1. **Mixed-quantization GGUF** of DeepSeek V4 Flash 0731 (imatrix-tuned).
2. **Joint main + MTP finetune** (LoRA on the main model and the 3-stage MTP
   drafter together), then spliced inline into a single GGUF (no sidecar).
3. **Inference speed research** on the DwarfStar engine (Metal).
4. **Published** the resulting model to Hugging Face.

## 2. What improved (measured)

| Change | Effect | How |
|---|---|---|
| Indexer sparse threshold `1024 → 4096` | **+24% decode** (24.3 → 28.9 t/s) at 4.4k–16.5k ctx | `DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD` default raised |
| KV checkpoint alignment `2048 → 0` | Warm-reuse prefill **36.8 s → 3.4 s** | `KV_CACHE_DEFAULT_BOUNDARY_ALIGN_TOKENS` set to 0 |
| Inline 3-stage MTP drafter | No separate `--mtp` sidecar needed | drafter embedded in the GGUF |
| Default MTP draft depth `2 → 1` | Plain decoding by default (fastest) | drafting is currently net-negative |

## 3. What did NOT improve

- **40 t/s at 18k context is not reachable** with the current kernels. The
  attention kernel (~9.69 ms, ~26% of decode) is the bottleneck and is
  latency/occupancy-bound, not bandwidth-bound.
- **MTP speculative drafting is net-negative** on this finetune: draft-2 runs
  ~31 t/s vs ~37 t/s plain at 13k context (verification costs more than
  accepted drafts save). Hence draft depth defaults to 1.
- **Fusion A/B sweep** found existing fusions worth **<3%** — nearly exhausted.

## 4. Reproduce the model (quantize + finetune)

Follow the runbook end-to-end:
[`runbook-0731-dspark-quantize-finetune.md`](runbook-0731-dspark-quantize-finetune.md).
Summary of the stages:

1. **Fix `config.json`** (one-time): derive `layer_types` / `mlp_layer_types`
   from `compress_ratios` (the 0731 checkpoint omits them).
2. **Quantize a Q4_K imatrix source** GGUF (~4 h).
3. **Collect imatrix** (~1.5 h), then repair non-finite values (layers 1–2).
4. **Quantize with the Layers37-42 recipe + imatrix** (~7.5 h):
   - layers 37–42 routed experts → `Q4_K`
   - layers 0–36 gate/up → `IQ2_XXS`, down → `Q2_K`
   - attention / shared / output → `Q8_0`
5. **Joint main + MTP LoRA finetune**, fuse, re-export, re-quantize, embed the
   MTP stages inline.

**Finetuning dataset:** derived from questions posed to **Fable 5**,
**Opus 4.8**, and **Opus 4.7**, together with coding repositories from
**GitHub**.

### MLX LoRA finetune command

The joint main + MTP LoRA finetune runs in the separate `ds4-finetuning`
worktree via `scripts/finetune_e2e.py` (an end-to-end pipeline: train → fuse →
MLX→HF → hybrid → quantize → verify). Proven parameters from the 2026-07-25
run:

```bash
cd /Users/spotted/projects/ds4-finetuning
python scripts/finetune_e2e.py \
  --mlx-work "/Volumes/Data NVME/mlx-ft/ds4" \
  --iters 400 --lr 5e-5 --rank 8
```

Key arguments (defaults shown):

| Flag | Default | Notes |
|---|---|---|
| `--iters` | 400 | overfitting starts after ~400 at lr 5e-5 |
| `--lr` | 5e-5 | learning rate |
| `--rank` | 8 | LoRA rank (scale 20) |
| `--targets` | `self_attn.q_a_proj self_attn.q_b_proj self_attn.kv_proj` | LoRA target modules (scale = rank × 2.5 = 20) |
| `--batch` | 1 | batch size |
| `--seq-len` | 4096 | max sequence length |
| `--dataset-root` | `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge` | dataset root |
| `--stop-after` | (all) | stop after `train`/`fuse`/`mlx-to-hf`/`hybrid`/`quantize`/`verify` |
| `--dry-run` | off | preview without running |

## 5. Reproduce the speed research

Build the engine and run the benchmarks that produced the numbers above:

```sh
make                                   # build (Metal)

# Indexer threshold A/B (13k prompt): sparse threshold effect
./ds4 --metal -m "$MODEL" --prompt-file "$PROMPT" -n 16 --temp 0 --ctx 32768
DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD=1024 \
  ./ds4 --metal -m "$MODEL" --prompt-file "$PROMPT" -n 16 --temp 0 --ctx 32768

# KV warm-reuse (disk checkpoint, alignment off): run twice, compare cold vs warm
./ds4-server --metal -m "$MODEL" --ctx 32768 --kv-disk-dir /tmp/kv --kv-disk-space-mb 8192
# then send the same prompt twice; the second hit should be much faster

# MTP drafting A/B (currently net-negative)
./ds4 --metal -m "$MODEL" --mtp-draft 2 --prompt-file "$PROMPT" -n 16 --temp 0 --ctx 32768
./ds4 --metal -m "$MODEL" --mtp-draft 1 --prompt-file "$PROMPT" -n 16 --temp 0 --ctx 32768
```

Full methodology and numbers: [`agent-output/mtp-40tps/results.md`](agent-output/mtp-40tps/results.md).

## 6. Run the published model

```sh
./download_model.sh   # or fetch from the HF repo directly
./ds4 -m DeepSeek-V4-Flash-IQ2_XXS-L37-42Q4K-MTP-JointFT-imatrix-0731.gguf --temp 0
# MTP drafting (optional, currently net-negative)
./ds4 -m DeepSeek-V4-Flash-IQ2_XXS-L37-42Q4K-MTP-JointFT-imatrix-0731.gguf --mtp-draft 2 --temp 0
```

## 7. Known limitations

- Requires the **DwarfStar** engine (this repo); generic GGUF loaders will not
  load this layout.
- Fits **128 GB** unified-memory machines (96.5 GiB weights + KV/scratch).
- MTP drafting is experimental and currently slower than plain decoding.
- License: MIT (base DeepSeek-V4-Flash is MIT).
