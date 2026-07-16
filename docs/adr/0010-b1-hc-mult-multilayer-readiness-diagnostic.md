# ADR 0010: B1 (hc_mult>1 multi-layer / hyperhead) readiness is recorded as a fail-closed diagnostic

Date: 2026-06-20
Status: Accepted

## Context

B1 is forward-parity blocker #2: full decoder-layer hyperconnection residual
mixing and final hyperhead parity. Story 11.32 proved adjacent B0a evidence:
single-layer `hc_mult=1`, multi-layer `hc_mult=1`, and single-layer
`hc_mult=2` hyperhead real-mode proofs. Those proofs are valuable but do not
compose into a proven stacked multi-layer `hc_mult>1` residual-mixing plus final
hyperhead end-to-end real-mode forward.

A live introspection-only probe found the multi-layer `hc_mult>1` cell is
double-blocked. The production `Model._validate_real_mode` gate rejects
`num_hidden_layers>1 and hc_mult!=1`, and `Model.__init__` calls that gate for
every real-mode model, so there is no harness-only seam to construct the
forbidden configuration. The trusted pure-Python reference
`_integrated_multilayer_forward` also rejects `hc_mult>1`, and the transformers
reference setter `set_transformers_integrated_weights` is single-layer only.

The existing `_real_forward` layer loop and final `hc_head` path already exist,
and `_integrated_layer_residual_streams` is already `hc_mult`-agnostic. The gap
is therefore a reviewed production gate lift plus a trusted multi-layer
`hc_mult>1` reference, not a new diagnostic-only forward-code change. A B1 proof
is larger than this slice. B2/B3 remain independent and larger still.

## Decision

Add an additive `b1_hc_mult_multilayer_readiness` block to the
`deepseek-v4-forward-parity-readiness` JSON report. The block is produced by a
pure fail-closed builder fed by an introspection/guard-only probe. It records
`proof_available=false`, the production and reference rejection evidence, the
proven-but-insufficient B0a evidence, and ordered future proof criteria.

The diagnostic is fail-closed and no-false-positive: `decision="not-ready"`,
`status="fail-closed"`, `fail_closed=true`, `ready=false`, and
`proof_available=false` are pinned regardless of probe outcome. If a future
probe finds the production gate lifted, `hc_mult_multi_layer_allowed=true` is
only structural evidence; readiness still does not flip until a reviewed
`real_mode_proofs` B1 proof compares real-mode `Model(num_hidden_layers in
{2,3}, hc_mult=2)` against a trusted multi-layer `hc_mult>1` reference at
`<=1e-5`.

No marker writer, conversion path, training path, quantization path, generation
path, production API, vendor gate, or production math changes. The top-level
readiness schema stays `1` because this is a backward-compatible additive key.

## Consequences

The readiness report now machine-readably prevents the B0a hyperconnection and
hyperhead evidence from being overclaimed as multi-layer `hc_mult>1` / final
hyperhead end-to-end parity. The exact missing gate-lift plus trusted-reference
requirements are pinned without relaxing `.deepseek-v4-forward-parity-ok`,
`convert-shimmed`, or any honesty counter.

`full_forward_parity=false`, `marker_earned=false`, `blockers_count=4`,
`real_mode_proofs.proofs_total=5`, and `coverage.fixtures_total=19` remain the
expected state until later reviewed proof slices close B1, B2, B3, and the final
marker-write gate.
