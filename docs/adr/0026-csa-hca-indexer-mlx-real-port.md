# ADR 0026 — CSA + HCA + Indexer real-dim MLX port (FROZEN contract additive expansion)

- **Date:** 2026-06-26 (Story/Epic 13.3b)
- **Status:** Accepted
- **Supersedes / amends:** Does NOT supersede ADR 0017 (boundary X / 9 forbidden
  symbols), ADR 0024 (FP4 on-the-fly dequant), or ADR 0025 (nn.Module sibling port).
  This ADR ADDITIVELY EXPANDS the FROZEN MLX attention contract introduced by Story 11
  (`deepseek_v4.py`) to cover the real 43-layer DeepSeek V4-Flash dims.
- **Related:** ADR 0025 (real nn.Module port — this consumes + extends it), Story
  11.14/11.15 (tiny CSA recon — guards preserved here), task brief
  `agent-output/cmux-13-3b/task-architect.md`.

## Context

Story 11 shipped FROZEN MLX helpers proven ONLY for a tiny CSA subset:
`_csa_config_error` (`deepseek_v4.py:317`) HARD-BLOCKS real dims (requires
`compression_ratio==4`, `num_attention_heads==1`, `o_groups==1`,
`num_key_value_heads==1`, `hc_mult==1`, `hidden_size==head_dim`,
`q_lora_rank==hidden_size`). The real Flash ckpt violates ALL of these:
`head_dim=512`, `hidden=4096`, `heads=64`, `o_groups=8`, `hc_mult=4`, plus CSA at
rate=4 (21 layers) AND HCA at rate=128 (20 layers, which has NO FROZEN path at all).

Recon (verified from ckpt index, `architecture.md §1`) shows the real ckpt uses
DeepSeek-native key names (`attn.compressor.wkv/wgate/ape/norm`,
`attn.indexer.compressor.*`, `attn.indexer.weights_proj`, `attn.indexer.wq_b`) that
already align to FROZEN `_csa_windowed_compressor_mlx` / `_csa_indexer_mlx` SEMANTICS.
The real architecture (torch `modeling_deepseek_v4.py:805`) is multi-head MLA
sliding attention with compressed-KV entries appended to the KV axis plus a
`block_bias` appended to the mask — i.e. the EXISTING real-dim-capable cr=0
`_attention_mlx:632` path PLUS a compressed-KV/block-bias addendum. So the port is a
RENAME + ADD, not an algorithm re-derivation.

Operator picked Option B: full CSA+HCA+Indexer real-dim port (multi-week).

## Decision

1. **Additive-only FROZEN expansion.** The sole edit to FROZEN `deepseek_v4.py` is a
   purely additive dispatch branch in `_attention_mlx` (`:640`):
   ```python
   if args.compression_ratio != 0:
       if _csa_config_error(args) is None:    # tiny proven subset → UNCHANGED
           return _csa_attention_mlx(args, x, weights, index_topk=index_topk)
       return _attention_real_mlx(args, x, weights, index_topk=index_topk)   # NEW
   ```
   Plus NEW appended symbols: `_hca_compressor_mlx`, `_indexer_mlx`,
   `_indexer_scorer_mlx`, `_attention_real_mlx`, and (if GroupedLinear parity
   requires) `_grouped_linear_mlx`. No FROZEN helper BODY is edited.

2. **Tiny guards stay live, tiny path stays byte-identical.** `_csa_config_error`,
   `_require_csa_config`, `_csa_attention_mlx`, `_csa_compressor_mlx`,
   `_csa_indexer_mlx`, `_csa_windowed_compressor_mlx` are UNCHANGED. Tiny inputs still
   satisfy `_csa_config_error(args) is None` and hit the identical return — Story
   11.14/11.15 parity fixtures MUST stay GREEN (regression gate). Only inputs that
   PREVIOUSLY raised `NotImplementedError` (real dims) route to the new symbol; no
   existing test input changes branch.

3. **Reuse FROZEN dim-generic primitives.** The CSA compressor + indexer compressor
   reuse FROZEN `_csa_windowed_compressor_mlx` (out_dim/rate parametric). HC stream +
   HyperHead reuse FROZEN `_hyperconnection_mlx`/`_hyperhead_mlx` (hc_mult=4 already
   validated). FP4 experts reuse ADR 0024 dequant. NEW helpers are written only where
   FROZEN cannot serve additively: HCA single-window compressor (no Ca/Cb overlap),
   real multi-head compressed attention (FROZEN tiny is single-head collapse), and the
   real indexer top-k → block_bias path (FROZEN tiny returns a valid_mask for
   single-head attention, not a `[B,1,S,T]` mask with `-1` sentinel + clamp).

4. **nn-file guard lift is SANCTIONED, not a FROZEN edit.** `AttentionNN:153`
   (`compression_ratio != 0 → NotImplementedError`) is lifted in `deepseek_v4_nn.py`,
   which ADR 0025 already owns as the editable sibling port. Real layers construct
   compressor/indexer submodules; sliding layers keep the cr=0 path unchanged.

5. **Convert-side remap is OUT of the FROZEN contract.** Ckpt-native → nn-port key
   remap (rename, ape transpose, per-expert→stacked, FP8-scale drop, `model.` prefix)
   lives in a NEW convert script, not in FROZEN load. `sanitize_weights` stays
   mtp-strip pass-through.

## Consequences

- FROZEN tiny behavior and 27 parity tests + 11.14/11.15 tiny CSA tests are
  protected by construction (additive branch + unchanged guards).
- The real path is parity-graded against torch `modeling_deepseek_v4.py` (per-helper
  oracles) AND must keep backward AC (13.3a-3) GREEN.
- RoPE parity for compress layers (theta=160000 + yarn, full vs trailing slice) is the
  primary technical risk (`architecture.md §3.5`); 13.3b-1 builds a rope oracle first
  and STOP-ESCALATES if a non-additive math/kernel primitive is required.
- Total envelope (~4–4.6 weeks) exceeds the 3-week single-story ceiling, so 13.3b is
  formally re-scoped to an Epic with five gated stories (`architecture.md §10`).
- If any tiny parity test flips RED, or a FROZEN symbol would need a semantic (not
  additive) change, the work STOPS — the contract is additive-only.

## Amendment 1 — 2026-06-27 (Story 13.3b-5b): backward-safe block_bias

**Status:** Accepted. **Trigger:** first real 43-layer `mlx_lm.lora --train` run. Forward PROVEN on
real bytes (Iter 1 Val loss 19.116, finite, all cr=4 CSA + cr=128 HCA + 256 FP4 experts dequant
forward-correct); first BACKWARD step crashed `ValueError: [scatter_axis] Cannot calculate VJP with
respect to indices` at `mlx_lm/tuner/trainer.py:250 loss_value_and_grad`.

**Decision.** The additive-only FROZEN rule (Decision §1) is amended for ONE precise case: a REAL
backward correctness bug in a FROZEN helper BODY may be fixed with a forward-identity transform. The
sole sanctioned body edit is in `_csa_block_bias_mlx` (`deepseek_v4.py:902`):
`scatter_indices = mx.stop_gradient(mx.expand_dims(safe_indices, 1))`.

**Why this is contract-consistent, not a contract break.**
- `mx.stop_gradient` is IDENTITY in the forward pass → block_bias output is byte-identical (probe-verified).
  All tiny CSA parity fixtures (Story 11.14/11.15) and 13.3b-2b AC2 (CSA-vs-torch forward parity) stay GREEN.
- block_bias is a DISCRETE top-k routing mask whose indices come from argsort over indexer-weight scores;
  those weights are NOT LoRA targets → no gradient is required there. The mask is non-differentiable by
  design (like MoE expert routing); gradients flow through attention values, not the discrete selection.
- The fix is the minimal statement of that fact: stop grad ONLY on the `indices` argument MLX cannot
  differentiate. No algorithm, shape, dtype, or numeric value changes.

**Hash consequence (the single sanctioned source-hash change).** This edit shifts `deepseek_v4.py`
sha256 from the pre-amendment frozen baseline to `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b`
(sha16 `5e11a9c4d82aeb24`). Per AGENTS.md SLICE-INVARIANT, ALL sha-pin sites MUST advance to the new hash
in the SAME slice (cascade enumerated in 13.3b-5b architecture §4). This is the documented case where the
FROZEN hash legitimately changes, with this ADR amendment as the justification.

**Scope limit.** This amendment authorizes ONLY the `_csa_block_bias_mlx:902` stop_gradient wrap. No other
FROZEN body edit is sanctioned. Any further FROZEN body change still STOP-ESCALATES.
