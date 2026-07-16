# ADR 0009: Stateful/decode readiness is recorded as a fail-closed diagnostic

Date: 2026-06-20
Status: Accepted

## Context

Stories 11.16 and 11.27–11.29 prove stateful/decode behavior only in the
spec/reference layer (`StatefulCSACache`, `IncrementalSlidingKVCache`, and tiny
reference decode/fusion helpers). That evidence is valuable, but it is not the
vendor MLX production runtime.

A live, introspection-only probe of
`ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4` found that `Model.__call__`,
`Model._real_forward`, and `Model._real_layer_forward` are stateless one-shot
interfaces with no cache/state/offset parameter. The module exposes no real
cache/decode/stateful/generate/step/incremental symbol, and `_attention_mlx` /
`_csa_attention_mlx` expose only cache-less `index_topk` wiring. A real-mode
stateful proof would require a reviewed production cache/decode API, which is
out of scope for this slice.

The remaining production decode questions are larger than seam detection:
MLA latent `c_t^KV` plus decoupled-RoPE cache lifetime, DSA/FlashMLA FP8
KV-cache decode, and the independent B1/B2/B3 parity gates.

## Decision

Add an additive `stateful_decode_readiness` block to the
`deepseek-v4-forward-parity-readiness` JSON report. The block is produced by a
pure builder fed by an introspection-only probe and records the structural seam
result, the proven-but-non-production spec seam, and the future proof criteria.

The diagnostic is fail-closed: `decision="not-ready"`, `status="fail-closed"`,
`fail_closed=true`, and `ready=false` are pinned. Structural seam detection is
necessary-but-insufficient; even if a future probe finds a cache/decode token,
readiness does not flip until a reviewed production cache/decode API and
real-mode stateful parity proof land.

The top-level readiness schema stays `1` because this is a backward-compatible
additive key. No marker/gate writer, conversion path, training path, generation
path, vendor forward API, or production math changes.

## Consequences

The readiness report now machine-readably prevents the proven spec/reference
stateful CSA work from being overclaimed as real-mode decode/stateful parity.
The missing production cache/decode seam and future proof criteria are pinned
without relaxing `.deepseek-v4-forward-parity-ok`, `convert-shimmed`, or any
honesty counter. `full_forward_parity=false`, `marker_earned=false`,
`blockers_count=4`, `real_mode_proofs.proofs_total=5`, and
`coverage.fixtures_total=19` remain the expected state until later reviewed
proof slices close B0/B1/B2/B3.
