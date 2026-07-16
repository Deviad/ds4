# ADR 0023 — Post-fuse generation coherence cross-check (Track-B safety net after parity-marker retirement)

- **Date:** 2026-06-22 (Head-of-AI due-diligence pivot)
- **Status:** Accepted
- **Supersedes / amends:** None (companion to ADR 0022; amends ADR 0008 by modifying the
  Track-B exit gate from a forward-parity marker to a post-fuse generation coherence
  cross-check — explicitly does NOT weaken Track-A or Track-B independence from ADR 0008).
- **Related:** ADR 0001 (Metal production path), ADR 0007 §4 (anti-circularity), ADR 0008
  (two-track markers), ADR 0017 (FROZEN dequant primitives), ADR 0019 (fusion bridge),
  ADR 0022 (parity-marker gate retired; strategic pivot)

## Context

ADR 0002 (parity-first / fail-closed) and ADR 0008 (Track-B marker
`.deepseek-v4-forward-parity-ok`) prescribed that MLX have a **bit-exact forward
reference** before training, to catch silent subtle forward bugs (wrong rope_theta,
SwiGLU clamp sign error, FP4 dequant mis-order) that would poison the gradient
landscape. The marker was the gate that protected against this risk.

The marker is **structurally unattainable on M3 Ultra** (HD-of-AI due-diligence
finding, 2026-06-22):
- HF Transformers ≥5.0 forces FP4→BF16 dequant on CPU: requires ~570GB resident
  for DeepSeek-V4-Flash (284B params × 2 bytes) > 512GB Mac unified RAM.
  `from_pretrained` process killed at 20% load (LIVE-confirmed 11.15h-r5 probe (c)).
- HF Transformers <5.0 does not ship `DeepseekV4ForCausalLM` class (LIVE-confirmed
  probe (b)).
- MLX `mlx_lm.load()` vendor port has strict-key enforcement (L1804) that rejects
  shimmed-ckpt key vocabulary; no upstream remap mechanism (LIVE-confirmed probe (B)).
- Any locally-written numpy re-impl would be **circular** under ADR 0007 §4 (cannot
  self-validate) — confirmed by r4/r5/r6 STOP chain (6 consecutive STOPs).

Six STOPs (11.15h, then -r2, -r3, -r4, -r5, -r6) established no hardware path to a
trusted independent gold forward on the M3 Ultra. Continuing the parity-marker
gate is sunk-cost-fallacy that blocks the project's core deliverable (local QLoRA).

## Decision

Replace the forward-parity safety net with an equivalent (slightly weaker but
sound) **post-fuse generation coherence cross-check**:

```
1. Train LoRA adapter locally (MLX `mlx_lm.lora --train`) on the shimmed ckpt
   converted to `model-4bit` via `convert-shimmed -q`.
2. Fuse: adapter → numpy-delta on shimmed HF ckpt → fused HF safetensors
   (per ADR 0019). Shared numpy-delta path is B-track-internal and needs NO
   gold-forward reference (it is byte-arithmetic: delta = (alpha/rank) * B·A).
3. Quantize fused safetensors → fused GGUF via `deepseek4-quantize --hf`
   (Track-A primitive; unchanged).
4. Sanity-generate on N fixed canonical prompts via BOTH:
     (a) Track-A C engine on the fused GGUF: `ds4 -m fused.gguf --metal ...`
     (b) Track-A C engine on the IMMUTABLE base GGUF: `ds4 -m ds4flash.gguf --metal ...` (no adapter)
5. Coherence assertion: the fused-model output must:
     (a) differ from base-model output on prompts where the adapter was trained
         to influence the answer; AND
     (b) remain coherent (no NaN, no degenerate repetition, valid tokens only,
         similar token entropy to base).
```

### Properties

- **Catch forward-bug risk class**: if the MLX forward path had a subtle bug
  (wrong rope, SwiGLU, FP4 dequant), the adapter optimizes against a wrong
  landscape. The fused model would then exhibit either (a) nonsensical
  generation vs base (forward bug visible) or (b) negligible difference from
  base (gradients flowed but converged poorly — visible as adapter no-op).
  Either symptom surfaces in cross-check generation ≠ base on trained prompts.
- **Non-circular per ADR 0007 §4**: Track-A C engine is bit-trusted per
  ADR 0001 / `.ds4-gguf-generate-ok` (PRESENT). The C-engine path is
  implemented independently from the MLX forward path; comparing their outputs
  is a real two-implementation cross-check, not a self-comparison.
- **Cheaper than gold forward**: the gate lifts at end-of-pipeline (post-train
  + post-fuse + post-quantize + post-serve), not at every pre-train gate.
- **Replaces 6-slice detour with 1 comparison**: this sanity check IS the
  forward-parity safety net, but bounded to known prompts + known-trained
  deltas instead of golden layer-0 hidden states.

## Consequences

- ADR 0002's "parity-first / fail-closed" principle is honored via COHERENCE
  cross-check on the SERVED model, not pre-training gold forward.
- ADR 0008's Track-A/Track-B independence is preserved: Track-A produces no
  new artifact for the cross-check; Track-B failure of gold-forward no longer
  blocks Track-A serving of base GGUF.
- `.deepseek-v4-forward-parity-ok` marker is RETIRED as a training gate
  (ADR 0022). It remains absent; future auths MAY write it honestly only via
  the post-fuse coherence cross-check passing on N≥4 canonical prompts.
- Training forward is permitted to proceed without gold forward reference.
- If post-fuse coherence cross-check FAILS (fused == base on trained prompts,
  OR fused produces incoherent generation), debugging narrows to:
  1. MLX forward bug (catch via 11.54 attention composition tests — 23/2 GREEN
     private-suite already validates attention correctness on a partial slice).
  2. Adapter fuse / quant path bug (catch via `ds4 --inspect --lora` + a
     single-expert delta verification).
  3. Adapter trained too few iters or LR too small (catch via loss curves).
- If no direction can be identified from symptoms, fall back to Path B
  (remote CUDA + HF Transformers 5.x) using the ADR 0022 Path B exit ramp.

## Definition of done for ADR 0023 cross-check

A canonical-prompts gate is implemented in `scripts/finetune_ds4.py` (new
command `post-fuse-coherence-check`, Story 13.5). It runs:

1. `ds4 -m fused.gguf --metal --prompt <canonical-P_i>` (i ∈ 1..N, N≥4)
2. `ds4 -m ds4flash.gguf --metal --prompt <canonical-P_i>` (identical prompts)
3. Asserts fused[i] ≠ base[i] for at least ⌈N/2⌉ prompts AND all fused[i] are
   non-degenerate (entropy ≥ threshold, no NaN tokens, gen did not crash).
4. Writes `.deepseek-v4-post-fuse-coherence-ok` marker when ALL assertions pass.

This new marker SUPERSEDES `.deepseek-v4-forward-parity-ok` as the Track-B
exit gate. The latter remains absent; `convert-shimmed` no longer checks for it
(per Story 13.0 gate-lift edit).
