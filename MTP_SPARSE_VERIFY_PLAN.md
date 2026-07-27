# MTP Sparse Verify for DS4 Flash

## Goal

Add MTP (Multi-Token Prediction) speculative decoding with **sparse verify**
to the DS4 engine, targeting 1.3–1.5× generation speedup (34 → 45–50 t/s)
on Apple Silicon Metal.

## Prior art

| Project | Approach | Result |
|---|---|---|
| GLM 5.2 llama.cpp (Deviad fork) | MTP gamma=1 + sparse-gather verify, cache-order sorted | 9.05 → 13.92 t/s (+54%) at 18.7k context |
| DS4 DSpark PR #502 | MTP gamma=5 + dense batch verify + rejection sampling | −30% on prose (dense verify kills bandwidth) |
| DS4 Flash baseline | No MTP, plain decode | 34 t/s on M3 Ultra |

**Key insight from GLM 5.2** (commit `5217c92fa`): the verify step must use
the same sparse attention as normal decode (gather only top-k KV rows, attend
unmasked). Dense batch verify over the full KV cache is guaranteed slower on
bandwidth-bound unified memory.

## Architecture context

DS4 Flash has:
- 43 decoder layers, 256 routed experts (top-6 active), 1 shared expert
- CSA (Compressed Sparse Attention): compressor + indexer, `index_topk=512`, `sliding_window=128`
- 1 MTP layer (`num_nextn_predict_layers=1`) — co-trained draft head
- MTP tensors in original HF checkpoint: 4,705 tensors in shards 46–48
- Current GGUF strips MTP tensors; quantizer has no MTP support

The CSA indexer already selects top-k KV rows per query — the same "gather
only relevant KV rows" pattern that made GLM 5.2's sparse verify work.

---

## User stories

### Story 1 — Include MTP tensors in GGUF

As a DS4 operator (WHO), I want the GGUF to contain the quantized MTP layer
(WHAT), so that the engine can use it for speculative draft prediction (WHY).

**Acceptance criteria:**

- (a) `deepseek4-quantize` accepts a `--dspark` or `--mtp` flag that reads
  MTP tensors from the HF checkpoint (shards 46–48, `mtp.0.*` prefix).
- (b) MTP attention projections quantized at Q8_0 (same as main model attention).
- (c) MTP expert weights quantized using the same profile as the main model's
  expert layers (IQ2XXS or Q4K, matching the template GGUF).
- (d) MTP norms, sinks, and compressor/indexer weights stored at F16/F32.
- (e) Output GGUF contains `dspark.kind`, `dspark.draft_layers=1`,
  `dspark.runtime_mtp=yes` metadata keys.
- (f) GGUF size increases by ~10–12 GB over the base model.
- (g) Base model tensors are byte-identical to a non-MTP quantization
  (MTP is additive only).

### Story 2 — Load MTP layer in ds4.c

As a DS4 engine (WHO), I want to load and initialize the MTP layer from the
GGUF at startup (WHAT), so that it is available for draft prediction during
generation (WHY).

**Acceptance criteria:**

- (a) `ds4_load_model` detects MTP metadata and allocates the MTP decoder
  layer (attention + MoE + norms + compressor/indexer).
- (b) MTP layer shares the tokenizer and embedding with the main model
  (no duplicate embedding weights).
- (c) MTP KV cache is allocated separately from the main model KV cache,
  sized for 1-token draft (not full context).
- (d) Memory overhead is ≤12 GB over the base model.
- (e) If the GGUF has no MTP tensors, the engine falls back to plain decode
  with zero overhead (no MTP code paths executed).
- (f) `./ds4 --inspect` reports MTP layer presence and metadata.

### Story 3 — MTP draft prediction (gamma=1)

As a DS4 engine (WHO), I want the MTP layer to predict the next token after
each committed token (WHAT), so that correct predictions skip the full
43-layer forward pass (WHY).

**Acceptance criteria:**

- (a) After each committed token, the MTP layer runs a single forward pass
  (1 decoder layer: attention + MoE + norms) to produce a draft logit vector.
- (b) The draft token is the argmax of the MTP logits (temp=0) or sampled
  from the MTP distribution (temp>0).
- (c) MTP forward pass uses the MTP KV cache (1 cached row per committed
  token), not the main model KV cache.
- (d) MTP forward pass latency is ≤5% of a full main-model forward pass.
- (e) Draft prediction is disabled during prefill (only active during
  autoregressive decode).
- (f) Opt-in via `DS4_MTP=1` env var or `--mtp` CLI flag; disabled by default.

### Story 4 — Sparse verify (the critical path)

As a DS4 engine (WHO), I want to verify the MTP draft token using sparse
attention — gathering only the CSA indexer's top-k KV rows, not the full
KV cache (WHAT), so that the verify step costs the same as a normal decode
step and does not increase bandwidth (WHY).

**Acceptance criteria:**

- (a) The verify step runs the main model's full 43-layer forward pass for
  the draft token, but attention at each layer uses the CSA indexer to
  gather only `index_topk` (512) KV rows + `sliding_window` (128) recent
  rows — identical to normal single-token decode.
- (b) No dense/masked attention over the full KV cache during verify.
- (c) Gathered KV indices are sorted in ascending cache order before the
  gather (prevents FP summation order changes that flip near-tie tokens —
  correctness guard from GLM 5.2 commit `5217c92fa`).
- (d) Verify produces the main model's logit vector for the draft token.
- (e) If `argmax(verify_logits) == draft_token`: commit the token, update
  both main and MTP KV caches, skip to next MTP draft. **Net effect: one
  cheap MTP pass + one normal decode pass = 2 passes for 2 tokens instead
  of 2 passes for 1 token.**
- (f) If `argmax(verify_logits) != draft_token`: commit the verify token
  (the main model's argmax), discard the draft, reset MTP KV cache for the
  new token. **Net effect: one cheap MTP pass wasted + one normal decode
  pass = same cost as plain decode.**
- (g) Sparse verify requires `n_kv > index_topk + 1`; below that threshold,
  fall back to plain decode (dense fallback is cheap at short context).
- (h) Correctness: temp=0 output is token-identical to plain decode
  (the verify step always produces the main model's argmax).

### Story 5 — Metal kernel support

As a Metal backend (WHO), I want optimized kernels for the MTP forward pass
and sparse verify gather (WHAT), so that the MTP path does not introduce
kernel-level bottlenecks (WHY).

**Acceptance criteria:**

- (a) MTP attention reuses existing CSA Metal kernels (compressor, indexer,
  flash attention) — no new attention kernels needed if the MTP layer has
  the same CSA architecture.
- (b) MTP MoE reuses existing `MUL_MAT_ID` expert dispatch kernels.
- (c) Sparse verify gather uses the existing CSA indexer's top-k gather
  path — no new gather kernel needed.
- (d) If new kernels are needed (e.g., MTP-specific fused ops), they are
  added to `metal/` with the same coding standards as existing kernels.
- (e) No regression in non-MTP decode performance (kernels are unchanged
  when MTP is disabled).

### Story 6 — Benchmarking and quality gates

As a DS4 operator (WHO), I want benchmark results proving the MTP sparse
verify path is faster than plain decode without quality regression (WHAT),
so that I can enable it with confidence (WHY).

**Acceptance criteria:**

- (a) Benchmark on M3 Ultra 512 GB with `ds4flash.gguf` + MTP GGUF:
  - Short prompt (≤1K context): measure t/s with MTP vs plain decode
  - Medium prompt (~10K context): measure t/s with MTP vs plain decode
  - Long prompt (~50K context): measure t/s with MTP vs plain decode
- (b) Target: ≥1.2× speedup at all context lengths (no regression).
- (c) Correctness gate: temp=0 output is token-identical to plain decode
  on a 10-prompt test battery (factual, code, creative, structured).
- (d) MTP acceptance rate measured and reported per prompt category.
- (e) Memory overhead measured and reported (≤12 GB over base model).
- (f) Results documented in a benchmark report committed to the repo.

### Story 7 — Fine-tuned GGUF with MTP

As a DS4 fine-tuning operator (WHO), I want the fine-tuning pipeline
(`finetune_e2e.py`) to produce a GGUF that includes MTP tensors (WHAT),
so that fine-tuned models also benefit from MTP sparse verify (WHY).

**Acceptance criteria:**

- (a) `finetune_e2e.py` gains a `--mtp` flag that includes MTP tensors
  in the hybrid directory and quantization step.
- (b) MTP tensors come from the original HF checkpoint (not affected by
  LoRA — the LoRA only targets main model attention projections).
- (c) The fine-tuned GGUF with MTP produces correct output in the DS4
  engine with `DS4_MTP=1`.
- (d) `docs/finetuning-guide.md` updated with MTP instructions.

---

## Execution order

```
Story 1 (quantizer MTP) → Story 2 (engine load) → Story 3 (draft)
    → Story 4 (sparse verify) → Story 5 (Metal kernels)
    → Story 6 (benchmarks) → Story 7 (fine-tuning integration)
```

Stories 1–2 are prerequisites. Story 4 is the critical path — the sparse
verify is what makes MTP net-positive on bandwidth-bound hardware. Stories
3 and 5 can partially overlap. Story 6 gates the merge. Story 7 is
independent and can run in parallel with Stories 3–6.

## Key design decisions

1. **gamma=1 only** (not gamma=5). Gamma=1 minimizes overhead: the MTP
   pass is cheap, and rejection costs nothing extra (same as plain decode).
   Higher gamma multiplies expert reads and was proven net-negative on
   this hardware (DS4 DSpark PR #502).

2. **Sparse verify, not dense verify.** The verify step uses the same CSA
   sparse attention as normal decode. This is the lesson from GLM 5.2
   (commit `5217c92fa`): dense verify over the full KV cache is −30% on
   bandwidth-bound hardware.

3. **Opt-in, not default.** `DS4_MTP=1` or `--mtp` flag. Zero overhead
   when disabled. Promote to default only after Story 6 benchmarks prove
   ≥1.2× speedup across all context lengths.

4. **MTP tensors from original checkpoint.** The LoRA fine-tuning only
   modifies main model attention projections. MTP weights are co-trained
   with the base model and should not be modified by LoRA.

## References

- GLM 5.2 sparse verify: `Deviad/llama.cpp` commits `5217c92fa`, `aa3a9cf17`, `0cf6cb228`
- DS4 DSpark PR (what failed): `antirez/ds4#502`
- DS4 Flash MTP tensors: `/Volumes/Backup/Models/DeepSeek-V4-Flash-DSpark` shards 46–48
- GLM 5.2 D5 GGUF: `Deviad/GLM-5.2-D5-IQ2S-Q2K-last6-IQ4NL-original-imatrix-expert`
- Fine-tuning pipeline: `ds4-finetuning` worktree, `scripts/finetune_e2e.py`
