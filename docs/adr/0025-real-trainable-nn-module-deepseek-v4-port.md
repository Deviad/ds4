# ADR 0025 — Real trainable `mlx.nn.Module` DeepSeek V4 port

- **Date:** 2026-06-26 (Story 13.3a, RE-SCOPE from 13.3)
- **Status:** Accepted
- **Supersedes / amends:** Does NOT supersede the parity-fixture `Model`
  (`vendor/mlx_lm_models/deepseek_v4.py:1685`) — that class STAYS for its 27
  parity tests. ADDS a sibling real-nn.Module port in a NEW file. Does NOT amend
  ADR 0017 (boundary X / 9 forbidden symbols) or ADR 0024 (FP4 dequant contract);
  the port CONSUMES both unchanged.
- **Related:** ADR 0007 §4 (anti-circularity), ADR 0015 (routed-dequant readiness),
  ADR 0022 (strategic pivot to local MLX QLoRA), ADR 0023 (post-fuse coherence
  gate), ADR 0024 (FP4 on-the-fly MoE dequant), ADR 0028 (opaque packed-FP4
  Metal training primitive).

## Context

Epic 13 Path A (local MLX QLoRA) requires `mlx_lm.convert -q` to produce a
`model-4bit/` and `mlx_lm.lora --train` to flow gradients. Story 13.3 (RUN the
wired commands) STOP-ESCALATED (BA, `agent-output/cmux-13-3/requirements.md`) on
two live-probe-proven hard blockers:

- **Blocker A** — the vendor `class Model` is a hand-rolled parity-fixture
  scaffold subclassing `object`, NOT `mlx.nn.Module`. It has no `leaf_modules` /
  `named_modules` / `trainable_parameters`. Experts are raw `mx.array` in a flat
  `self._real_weights` dict (no `nn.Linear` leaves). `__call__(list) -> list`
  returns `.tolist()` (no autograd graph). ⟹ `nn.quantize` and
  `linear_to_lora_layers` both crash before touching experts/forward.
- **Blocker B** — real config has 3×`hash_moe` + 40×`moe` layers; vendor
  `_real_layer_forward:1955` calls `_moe_mlx` for every layer, with NO hash_moe
  branch. hash_moe forward was UNIMPLEMENTED.

User picked Option 1 (hand-port a real nn.Module; no pivot to torch/GPT2). 13.2's
FP4 dequant math is correct but UNREACHABLE by the training path (BA Q2) because
the forward it lives in is never run by the trainer.

## Decision

Port a real trainable `mlx.nn.Module` DeepSeek V4 into a **NEW file**
`vendor/mlx_lm_models/deepseek_v4_nn.py` with **NEW `model_type="deepseek_v4_nn"`**,
starting from the mlx-lm `deepseek_v3.py` nn.Module skeleton and adding the v4
delta from the torch `modeling_deepseek_v4.py` reference + the FROZEN vendor MLX
math (13.1/13.2).

1. **Why the parity-fixture was insufficient** — Blockers A+B above. The fixture
   proves forward MATH (parity vs torch) but is structurally untrainable
   (`object` base, dict experts, list forward). Replacing it in-place would break
   27 dependent parity tests; instead the real module is a sibling.

2. **v3-base + v4-delta strategy** — `deepseek_v3.py` supplies the trainable
   skeleton (`nn.Embedding`/`nn.RMSNorm`/`nn.Linear` leaves, `__call__→mx.array`,
   `PipelineMixin`, `.layers`, `sanitize`). The v4 delta — `hc_mult`
   HyperConnection (torch `:876`), HyperHead (torch `:955`), hash_moe router
   (torch `DeepseekV4HashRouter:1054`, `tid2eid` token→expert lookup), MLA +
   grouped-output + sink-logits attention (torch `:755` + vendor `_attention_mlx`),
   FP4 experts — is ported as nn.Module submodules REUSING the FROZEN vendor MLX
   primitives (imported, never copied), so 13.1/13.2 math stays byte-exact.

3. **hash_moe resolution (Blocker B)** — hash_moe == moe with the same expert-call
   + shared-expert path; ONLY the routing-index source differs (static
   `tid2eid[input_ids]` vs learned `topk`). One `SparseMoeBlockNN` with an
   `is_hash` flag; `input_ids` threaded through the decoder to hash layers.

4. **FP4-under-nn.quantize decision — (c) on-the-fly via FROZEN primitive** — FP4
   experts are a custom `DeepseekV4FP4Experts(nn.Module)` holding uint8 packed
   `.weight` + bf16 `.scale`, `freeze()`d, dequantizing on-the-fly in `__call__`
   via the FROZEN `_dequantize_fp4_block_scale_mlx` (ADR 0024). It has no
   `to_quantized` → `nn.quantize` SKIPS it (experts stay FP4, no BF16 blow-up; ~555GB
   transient avoided). It is not `nn.Linear`/`SwitchLinear` → `linear_to_lora_layers`
   SKIPS it (DS4 LoRA targets attention only). `nn.quantize` quantizes the non-expert
   `nn.Linear` leaves (attention/lm_head) to 4-bit. **This makes the 13.2 FP4 primitive
   the LIVE training-forward path (resolves BA Q2).** Rejected: (b) load-time
   FP4→BF16 re-quantize (defeats the FP4 memory design).

5. **Coexistence** — new file + new model_type routes convert/train via the plugin
   (`mlx_lm_plugin.py` appends vendor dir to `mlx_lm.models.__path__`;
   `_get_classes` resolves `arch.Model` by `model_type`). The FROZEN parity-fixture
   `deepseek_v4.py` and all 13.2 tests are byte-untouched (`9 passed` baseline holds).

6. **Sequencing** — full tiny-config port is ~8.75 dev days (> 7-day STOP threshold),
   so it SPLITS into 13.3a-1 (skeleton + attention), 13.3a-2 (MoE + hash_moe + FP4),
   13.3a-3 (integration backward AC + train wiring). CSA-at-scale + hc_mult=4 is
   EXCLUDED and FLAGGED for 13.3b (vendor CSA is proven only for tiny `compression_ratio=4`,
   `hc_mult=1` fixtures).

## Amendment — Story 13.3b-5d sparse routed-token FP4 backward

Story 13.3b-5d changes only the training sibling's routed FP4 execution policy in
`deepseek_v4_nn.py`. The dense all-expert/full-token loop is replaced by a
training-only eager sparse dispatcher: route indices are computed by the existing
learned/hash router, passed through `mx.stop_gradient`, materialized on the host
only as integer token-row metadata, and collapsed so duplicate expert ids within
one token count once. Activations, router scores, FP4 payloads, and scales never
cross the host boundary.

The dispatcher evaluates each non-empty expert on its unique routed token rows
`R_e`, uses the existing `DeepseekV4FP4Experts.forward_one` and ADR 0024 FP4
dequant primitive unchanged, weighted-scatter-adds back to original token order,
and places `mx.eval` barriers after every expert in both routed forward and
custom backward. Empty experts are skipped. The shared expert still runs once on
the original full input outside the custom routed operation.

Backward is an exact custom **first-order** VJP for differentiable inputs
`(x_flat, scores_flat)`. Discrete indices remain detached; frozen expert
weights/scales receive no cotangents. Score cotangents use the normalized routed
weight derivative, and expert-input cotangents call `mx.vjp` over `forward_one`
rather than copying FP4/SwiGLU derivatives. Higher-order gradients remain out of
scope.

MLX compilation is not disabled at module import or model construction. The
trainer wrapper must disable compilation locally before calling `mlx_lm.lora.main`
because host route materialization inside `mx.compile` raises MLX's documented
"eval during function transformations" error:

```python
import mlx.core as mx
mx.disable_compile()
mx.set_memory_limit(400_000_000_000)
from mlx_lm.lora import main
main()
```

ADR 0024 remains unchanged; `deepseek_v4.py` remains FROZEN and imported only.

## Amendment — Story 13.3b-5f opaque packed-FP4 Metal primitive

Story 13.3b-5e disproved one consequence of the 13.3b-5d amendment under MLX
0.31.2 outer transforms: per-expert `mx.eval`, `mx.stop_gradient`, deletion, and
cache clearing do **not** bound the lifetime of the Python `forward_one` + nested
`mx.vjp` graph. Fresh-process E=2/4/8 telemetry showed active operation memory
rising after every expert until the complete outer gradient was evaluated.

ADR 0028 therefore supersedes the 13.3b-5d amendment only in these respects:

- the Python `forward_one` + nested `mx.vjp` routed backward is no longer an
  authorized implementation path;
- per-expert `mx.eval` barriers make no one-expert lifetime guarantee under
  `mx.grad` / trainer `mx.value_and_grad`;
- the routed training path must use the ADR 0028 opaque packed-FP4 Metal forward
  and exact first-order input-VJP boundary, with no fallback to the stopped path.

The remaining 13.3b-5d contracts continue unchanged: detached integer route
metadata, duplicate collapse, empty-expert skip, exact normalized weighting,
learned/hash routing semantics, frozen expert weights/scales, one full-input
shared-expert call, local compile disablement, and no higher-order-gradient claim.
ADR 0024 remains the binding packing/value/scale oracle. Production inference,
FROZEN `deepseek_v4.py`, and non-MLX backends remain outside this amendment.

## Consequences

- **Positive**: unblocks the entire Path A endgame (13.3b→13.3c→13.4→13.5→13.6).
  13.2 FP4 dequant becomes training-path-proven. Zero risk to 13.1/13.2 (sibling file).
  Native MLX `mx.quantize(mode="mxfp4")` confirmed for the real expert layout
  (group_size=32, 4-bit).
- **Negative / risk**: real-config CSA-at-scale + hc_mult=4 is an unproven parity
  gap deferred to 13.3b (potential additional multi-day spike). `tid2eid` provenance
  in the shimmed ckpt must be confirmed by 13.3b recon. The dense per-expert forward
  loop is correctness-first; a `gather_mm`/sorted-dispatch speedup is a later slice.
- **Invariant**: any future edit to the FROZEN `deepseek_v4.py` primitives or the
  parity-fixture `Model` is OUT OF SCOPE for the port; the port IMPORTS, never edits
  (STOP-rule S1).
