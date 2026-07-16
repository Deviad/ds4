# Progress — DS4 Finetuning (Story 11)

## Hard restart reminder

For every fresh-memory restart that will make implementation/code changes, do
not edit first. Spawn the full agent pipeline first: **BA, Architect, Coder,
Reviewer, Tester/Test Manager**. Required flow: `BA → Architect → Coder →
Reviewer + Test Manager`, with file handoff and `.cmux-status/*.done` markers as
specified in `AGENTS.md`. If cmux is unavailable, stop and ask before using a
fallback.

## Current state (2026-06-18)

Story 11 slices 1–12 are implemented locally. Slice 12 / Story 11.14 ports
the proven `compression_ratio=4` CSA compressor/indexer attention path to MLX
for the tiny cache-less single-head subset. `.deepseek-v4-forward-parity-ok` is
still correctly absent; `model-4bit` is still correctly absent.

`make test` is green when run with `DS4_TEST_MODEL=/Users/spotted/projects/ds4/ds4flash.gguf`.
`make finetune-test` is green: **250 OK (60 skipped)**. MLX-specific
`tests.test_deepseek_v4_mlx_port` is green: **53 OK (2 skipped)** in the MLX
venv.

## Slice 11.15a — Real expert packing classifier — IMPLEMENTED (awaiting review/test manager)

- Added a header-only `classify_checkpoint_expert_packing()` wrapper that keeps
  `classify_expert_metadata_from_header()` unchanged and classifies the actual
  checkpoint expert storage observed by the Architect: routed `I8` weights with
  `F8_E8M0` scales, shared `F8_E4M3` weights with `F8_E8M0` scales, FP4 verified
  absent, `can_decode_payload=False`.
- The classifier reconciles observed metadata with declared FP8
  `quantization_config`, flags routed block-size discrepancy (`axis=1`,
  `block_size=16` vs declared `[128,128]`), and preserves shared 128×128 2-D
  block ambiguity instead of guessing an axis.
- New tests cover real-shaped metadata classification, FP4 absence, unknowns,
  config discrepancy, payload-free operation, determinism, optional real shard
  header classification, and forward-parity marker absence. `dequantize_expert_packed`
  remains fail-closed and `.deepseek-v4-forward-parity-ok` / `model-4bit` remain
  absent.
- Documentation follow-up: plan wording that still says "packed FP4" should be
  corrected by the doc owner to the verified real packing
  `I8/F8_E4M3 + F8_E8M0`; this slice only flags the premise correction.

## Slice 11.14 — MLX compressor/indexer port — IMPLEMENTED (review follow-up fixes applied)

- **Scout done:** `agent-output/story-11-14-mlx-compressor-scout.md`
  (supervisor-authored after the dispatched `scout` stalled on local Kimi K2.7
  and was interrupted). Maps the pure-Python
  `tiny_compressor_indexer_attention_reference` → MLX port, lists the 10
  compressor/indexer weight keys, the 3 fail-closed insertion points, reusable
  MLX helpers, the RoPE/pool gap, and a 3-slice TDD breakdown at ≤1e-5.
- **Plan sanity check done:** `review-xhigh-plan-sanity-check.md` — all 9 doc
  claims OK, no blockers.
- **Implemented TDD slices:**
  - **11.14a** CSA compressor MLX port: `_csa_compressor_mlx` matches
    `tiny_csa_compressor_forward` within ≤1e-5 using full-length RoPE and
    softmax-gated Ca/Cb pooling.
  - **11.14b** Lightning indexer MLX port: `_csa_indexer_mlx` matches
    `tiny_csa_indexer_forward` within ≤1e-5 for scores and exactly for
    indices/mask.
  - **11.14c** Integrated CSA attention + Model relaxation:
    `_csa_attention_mlx` matches `tiny_compressor_indexer_attention_reference`
    within ≤1e-5 using `compress_rope_theta` for CSA/indexer RoPE, including
    all-masked and sub-window rows → zeros; `_attention_mlx` routes to it for the
    proven tiny subset only (`hc_mult=1` included); `Model._validate_real_mode`
    and `load_weights` accept the proven tiny CSA subset only.
- **Resolved 11.14c decision:** `index_topk` is a forward kwarg/default, not a
  `ModelArgs` field.
- **Scope guard:** 11.14 is the proven tiny subset only
  (`num_attention_heads=1`, `o_groups=1`, cache-less single forward). It does
  **not** write `.deepseek-v4-forward-parity-ok` — that needs 11.15 + 11.16 too.

## Slice 11 — Compressor/indexer reconciliation — DONE

The divergent synthetic `compression_ratio=2` MLX block-sparse path
(`_compressed_attention_mlx` + 3 helpers + `ModelArgs.index_topk`) was removed
because it broke the fail-closed invariant and did not match the intended
ratio=4 CSA reference. The reviewed-clean ratio=4 pure-Python
`tiny_compressor_indexer_attention_reference` (softmax gating, matching
Transformers `DeepseekV4CSACompressor`) was kept. All three compression
fail-closed gates restored: `_attention_mlx`, the integrated-layer fixture, and
`Model._validate_real_mode`. Reconciliation review: no blocker; the stale
"ratio=2 structural analog" doc wording was fixed. 8 ratio=4 pure-Python tests
pass; `make finetune-test` 236 OK (54 skipped).

## Slice 10 — Multi-layer synthetic — DONE + REVIEWED CLEAN

- `Model._validate_real_mode` accepts `num_hidden_layers∈{1,2,3}` (hc_mult=1).
- Stacked multi-layer forward with per-layer `layers.{i}.*` weights; final
  norm/lm_head collapse applied once after the loop.
- `hc_mult>1` × multi-layer is dual-side fail-closed (`_validate_real_mode`
  raises `NotImplementedError`; `bounded_model_expected_shapes` raises
  `CheckpointTensorLoadError`).
- Follow-up review clean.

## Earlier Story 11 slices (all reviewed clean)

- Slices 1–6: multi-head/grouped attention (reference, RoPE, sinks, groups,
  Model integration).
- Slice 7: multi-head/grouped checkpoint shape compatibility — real Flash
  attention shapes match.
- Slices 8–9: synthetic top-k MoE routing (`n_routed_experts≤4`) + shape compat.
- Checkpoint header validation, selected-tensor loader, bounded
  checkpoint→Model load, `e_score_correction_bias` zero-default, hc_mult>1,
  final HyperHead collapse — all reviewed clean.
