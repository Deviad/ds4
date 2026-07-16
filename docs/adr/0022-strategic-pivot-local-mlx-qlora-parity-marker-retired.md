# ADR 0022 — Strategic pivot: local MLX QLoRA on M3 Ultra (Path A); forward-parity marker gate retired

- **Date:** 2026-06-22 (Head-of-AI due-diligence pivot)
- **Status:** Accepted (supersedes the prior version of this ADR authored by
  Architect in Story 11.15h-r5; the prior "key-remap adapter + fp4 packing
  contract" version is now SUBSUMED under this pivot ADR because the pivot
  removes the parity-marker gate that ADR 0022 was originally written to
  sanction via a numpy MoE re-impl.)
- **Supersedes:** The numpy↔gold single-dequant parity variant of ADR 0022
  authored 2026-06-25 in Story 11.15h-r5 (`agent-output/cmux-11-15h-r6/
  adr-0022-final.md` held as Deferred — that variant is now also SUPERSEDED).
- **Amends:** ADR 0008 (Track-B exit gate relaxed from forward-parity marker to
  post-fuse generation coherence cross-check per ADR 0023; Track-A/Track-B
  independence INVIOLATE — Track-A produces nothing new; Track-B no longer
  gated).
- **Related:** ADR 0001 (Metal production path), ADR 0002 (parity-first — now
  honored via coherence cross-check), ADR 0007 §4 (anti-circularity),
  ADR 0008 (two-track markers — Track-B exit gate amended), ADR 0017 (FROZEN
  dequant primitives), ADR 0019 (fusion bridge), ADR 0023 (coherence cross-check)

## Context

A new Head of AI joined the project 2026-06-22 and ran due diligence: read the
codebase fresh, reviewed all plans, double-checked, and modified the plan where
necessary. Findings:

1. **The project's serving foundation is NOT blocked.** The C engine already
   ships THREE production backends today: Metal (`ds4_metal.m`, primary),
   NVIDIA CUDA (`ds4_cuda.cu`), and Strix Halo ROCm (`ds4_rocm.cu` + `rocm/`).
   The project goal "leave the door open to other architectures such as CUDA"
   is ALREADY satisfied — CUDA backend is shipped. The fused GGUF is
   backend-agnostic; `ds4 -m fused.gguf` chooses backend via runtime config.

2. **The training path pivot to local MLX (Epic 12) contradicted the project's
   own feasibility report.** `torch-real-v4-feasibility` (status verified at
   `training-next-status.md`) already concluded
   `local_training_feasible: false` for raw Torch/MPS PEFT because the
   checkpoint contains F8_E4M3/F8_E8M0 quantized tensors that cannot be trained
   raw on MPS. The bakeoff report
   `training-backend-bakeoff-gpt55.md` explicitly recommended Path B
   (remote NVIDIA CUDA + HF Transformers + PEFT + TRL) and labeled MLX
   "not a credible first path" — Epic 12 pivoted AWAY from this recommendation
   without resolving the underlying blocker.

3. **The Track-B forward-parity gold reference is HARDWARE-INFEASIBLE on M3
   Ultra.** ADR 0007 §4 requires the reference to be non-circular (cannot be
   self-expressing the same formula). LIVE probes through 6 STOPs (11.15h +
   r2/r3/r4/r5/r6) established:
   - HF Transformers 5.12.1 forces FP4→BF16 expansion (~570GB) > M3 Ultra 512GB
     unified RAM → OOM kill at 20% load (248/1285 weights). FALSIFIED.
   - HF Transformers <5.0 does not ship `DeepseekV4ForCausalLM`. FALSIFIED.
   - MLX `mlx_lm.load()` vendor port crashes at strict-key enforcement L1804
     on shimmed-ckpt key vocabulary. FALSIFIED.
   - Locally-written numpy re-impl would be circular under ADR 0007 §4.

4. **The actual training blockers (Code gaps, not hardware wall):**
   - **Blocker 1**: vendor mlx_lm port `deepseek_v4.py` `_load_real_weights` at
     L1804 strict-key asserts the HF-canonical key vocabulary
     (`model.embed_tokens.weight`, `model.layers.X.self_attn.*`,
     `lm_head.weight`). The shimmed ckpt uses the deepseek-ai release key
     vocabulary (`embed.weight`, `layers.X.attn.*`, `head.weight`). Both
     vocabularies are INHERENT to deepseek-ai's release — neither is a shim
     artifact. No upstream remap mechanism exists.
   - **Blocker 2**: vendor `_moe_mlx` defaults `block_size=16` dequant + does
     full single-expert BF16 expansion, but shimmed ckpt routed-expert I8
     tensors are **FP4 byte-packed** (2 nibbles/byte along axis=1,
     block_size=32, BF16 scales ~135-160GB total). Naive expansion to BF16
     would OOM at ~570GB; the kernel must unpack-and-dequant **on-the-fly
     during matmul without materializing BF16**.

5. **Prior Approach Was Sunk-Cost Syndrome.** Six consecutive STOPs narrowed
   the parity scope each time. The architectural contradiction (no hardware
   path to gold reference) cannot be solved by narrowing scope. The pivot
   replaces gold-forward with a post-fuse generation coherence cross-check
   (ADR 0023) that catches the same risk class at lower cost.

## Decision

1. **Strategic pivot to "Path A — Local MLX QLoRA on M3 Ultra."** Epic 13
   (new) replaces Epic 12's MLX-port-as-gold approach: build a working local
   MLX forward+backward+LoRA path through TWO concrete code fixes
   (key-remap + FP4 on-the-fly dequant kernel), then run `convert-shimmed -q`
   + `mlx_lm.lora --train` locally. Real LoRA training proceeds on the local
   hardware. ~7-10 dev days R&D investment. Door to CUDA already open
   (backend shipped).

2. **Forward-parity marker `.deepseek-v4-forward-parity-ok` is RETIRED as a
   training gate.** It remains ABSENT. The marker MAY be written honestly
   in the future ONLY via post-fuse coherence cross-check passing (per
   ADR 0023); not via gold forward reference.

3. **`convert-shimmed` is UNBLOCKED at the parity-marker gate.**
   `scripts/finetune_ds4.py:_validate_forward_parity_marker` no longer gates
   `convert-shimmed` execution. The two upstream gates
   (architecture-key / LoRA-target) remain unchanged.

4. **Story 11.15h / -i / -j arc CLOSED AS DEFERRED.** The 6-slice parity
   arc is superseded by Epic 13. Provenance preserved in
   `agent-output/cmux-11-15h*/` (read-only archive); Story 11.15h reconciliation
   now lives at `docs/adr/0020-...` (prior retrospective reasoning is honored as
   PROVENANCE for what was tried and why it failed — not as a forward plan).

5. **ADR 0019 fusion contract is RE-AFFIRMED as the serving bridge.**
   adapter → numpy-delta on shimmed HF ckpt → fused safetensors →
   `deepseek4-quantize --hf` → fused GGUF → `ds4 -m fused.gguf --metal`
   (or `--cuda` backend via same engine).

6. **Subagent model routing UNCHANGED** — kimi-k2.6 Coder, qwen3.6-35b Tester,
   opus-4-8 Reviewer/xhigh, glm-5.2-short BA/Architect. Caveman ultra default
   for all role panes + parent/supervisor.

## Consequences

- ADR 0002 (parity-first) is honored via ADR 0023 post-fuse coherence
  cross-check (NOT gold forward).
- ADR 0008 Track-A/Track-B independence INVIOLATE — Track-A produces no new
  artifact; Track-B no longer gated on a structurally-unattainable marker.
- ADR 0017 FROZEN primitives (`dequantize_i8_block_scale`, `dequantize_i8_e8m0_block_scale`)
  UNCHANGED — they remain Track-A primitives. New numpy MoE / FP4 on-the-fly
  dequant code lives in the MLX Python port, parallel to (not replacing) the
  Track-A primitive.
- ADR 0007 §4 (anti-circularity) HONORED — the post-fuse coherence
  cross-check compares the FUSED-MODEL output against the BASE-MODEL output,
  both served by the SAME Track-A C engine. It is NOT a self-comparison of
  MLX-vs-numpy (circular): it is MLX-trained-adapter-via-C-engine vs
  base-weights-via-C-engine, a real two-condition comparison on the
  bit-trusted C-engine path.
- All Epic 13 slices run the cmux multi-agent pipeline (BA → Architect →
  Coder → Reviewer + Test Manager in parallel after Coder) per AGENTS.md
  HARD STARTUP GATE.
- Per-slice git commit discipline honored. docs/agent-output/.cmux-status/.pi/
  remain untracked per project convention.
- If local MLX QLoRA proves harder than ~10 dev days (FP4 kernel R&D drift),
  Path B exit ramp remains available: rent remote CUDA, run HF Transformers
  5.x + PEFT + TRL, save adapter, numpy-delta fuse, serve locally. No
  architectural rework needed — same fusion bridge (ADR 0019) consumes
  either-trained adapter.
