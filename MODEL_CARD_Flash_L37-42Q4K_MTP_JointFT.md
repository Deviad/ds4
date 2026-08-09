---
license: mit
library_name: gguf
pipeline_tag: text-generation
language:
- en
tags:
- deepseek
- deepseek-v4
- deepseek-v4-flash
- gguf
- mtp
- multi-token-prediction
- speculative-decoding
- imatrix
- finetuned
base_model: deepseek-ai/DeepSeek-V4-Flash
---

# DeepSeek V4 Flash 0731 — L37-42 Q4_K, Joint-Finetuned MTP (GGUF)

A quantized GGUF of **DeepSeek V4 Flash (0731 checkpoint)** with a **jointly
finetuned inline Multi-Token Prediction (MTP) drafter**. This is the **0731**
revision of DeepSeek V4 Flash. Unlike the released Flash GGUFs, this build
embeds the 3-stage MTP drafter directly in the main model — no separate
support/sidecar file is required. It is intended for the
[DwarfStar](https://github.com/antirez/ds4) inference engine, which is the only
loader that understands this layout.

**File name (HF convention):**
`DeepSeek-V4-Flash-IQ2_XXS-L37-42Q4K-MTP-JointFT-imatrix-0731.gguf`

## Fine-tuning data

The joint fine-tuning dataset is derived from questions posed to **Fable 5**,
**Opus 4.8**, and **Opus 4.7**, together with coding repositories from
**GitHub**.

## Model details

| | |
|---|---|
| Architecture | DeepSeek V4 Flash (`deepseek4`), 43 layers |
| Base | `deepseek-ai/DeepSeek-V4-Flash`, **0731 checkpoint** |
| Revision | **0731** |
| Parameters | 284B total / 13B active (base); ~304B logical incl. MTP stages |
| Context length | 1,048,576 tokens (train) |
| Attention | 64 heads, 1 KV head, head_dim 512, sliding window 128 |
| Experts | 256 routed, 6 used per token |
| MTP drafter | 3 inline stages, block size 5, target layers 40–42 |
| File size | 96.47 GiB |
| License | MIT (base DeepSeek-V4-Flash is MIT) |

## Quantization recipe

Asymmetric mixed quantization: only the routed MoE experts are aggressively
compressed; projections, routing, and output are kept high-precision to
preserve quality. imatrix-tuned.

| Component | Type |
|---|---|
| Layers 37–42 routed experts | `Q4_K` (closest to output, higher quality) |
| Layers 0–36 routed gate/up experts | `IQ2_XXS` |
| Layers 0–36 routed down experts | `Q2_K` |
| Attention projections | `Q8_0` |
| Shared experts | `Q8_0` |
| Output head | `Q8_0` |
| Norms / HC / compressor / indexer | `F16` / `F32` |

Tensor-type breakdown (from `ds4 --inspect`):

| Type | Tensors | Size |
|---|---:|---:|
| `iq2_xxs` | 80 | 41.25 GiB |
| `q2_k` | 40 | 26.25 GiB |
| `q4_k` | 18 | 20.25 GiB |
| `q8_0` | 376 | 6.66 GiB |
| `f16` | 366 | 2.05 GiB |
| `f32` | 527 | ~0 GiB |
| `i32` | 3 | ~0 GiB |

## Inline MTP drafter (the distinguishing feature)

The 3-stage MTP drafter is **embedded in the main GGUF** and jointly finetuned
with the main model (`mtp/dspark` metadata: `stages=3`, `block=5`,
`target_layers=40,41,42`). Because the drafter ships inside the model, you do
**not** need a separate `--mtp` support GGUF. DwarfStar binds it automatically
at load:

```text
ds4: embedded_mtp stages=3 bound_stages=3 source=main_model block_size=5 draft=1
```

**Current status of MTP drafting on this build:** speculative decoding is
available via `--mtp-draft N` but is presently *net-negative* on this
finetuned checkpoint — verification costs more than accepted drafts save
(measured ~31 t/s with draft-2 vs ~37 t/s plain at 13k context, byte-identical
output). The engine therefore defaults to `--mtp-draft 1` (plain decoding).
Drafting remains opt-in for experimentation as drafter acceptance improves.

## Intended use

- Local inference of DeepSeek V4 Flash on high-memory Apple Silicon (Metal),
  NVIDIA CUDA, or ROCm, via the DwarfStar engine.
- Fits 128 GB unified-memory machines (96.47 GiB weights + KV/scratch).
- Research into inline/joint MTP speculative decoding.

**Out of scope:** use with generic GGUF loaders (llama.cpp etc.) — this file
requires the DwarfStar engine's tensor layout and inline-MTP handling.

## Performance (DwarfStar, Metal)

Single-run, `--ctx 32768`, greedy, Mac Studio M3 Ultra 512 GB:

| Context | Prefill | Generation (default) | Generation (draft-2) |
|---|---:|---:|---:|
| ~13k tokens | ~470 t/s | ~37 t/s | ~31 t/s |

Warm KV-cache reuse (disk checkpoint, alignment off) reduces a repeat 18k-turn
prefill from ~36.8 s to ~3.4 s.

## How to run

```sh
# Download (once available in antirez/deepseek-v4-gguf)
./download_model.sh q2-q4-mtp-imatrix

# Plain decoding (default, currently fastest)
./ds4 -m gguf/DeepSeek-V4-Flash-IQ2_XXS-L37-42Q4K-MTP-JointFT-imatrix-0731.gguf --temp 0

# Experiment with MTP drafting (currently net-negative, opt-in)
./ds4 -m gguf/DeepSeek-V4-Flash-IQ2_XXS-L37-42Q4K-MTP-JointFT-imatrix-0731.gguf \
      --mtp-draft 2 --temp 0
```

## Limitations & biases

Inherits DeepSeek V4 Flash's limitations and biases. Quantization is lossy;
the 2-bit routed-expert compression is imatrix-tuned to preserve quality but
is not bit-exact with the FP8 base. MTP drafting is experimental and currently
slower than plain decoding on this checkpoint.

## License

MIT. Derived from `deepseek-ai/DeepSeek-V4-Flash` (MIT).

## Citation

```bibtex
@misc{deepseekai2026deepseekv4,
      title={DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
      author={DeepSeek-AI},
      year={2026},
      url={https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash}
}
```
