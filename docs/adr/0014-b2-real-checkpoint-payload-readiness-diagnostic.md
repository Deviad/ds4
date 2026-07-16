# ADR 0014: B2 real-checkpoint-payload decode readiness recorded as fail-closed diagnostic

Date: 2026-06-20
Status: Accepted

## Context

B2 is the forward-parity blocker for full MoE parity with packed FP4/I8 expert
dequant and expert kernels (`forward_parity_blockers()[2]`). Stories 11.37/11.38/
11.39 (ADRs 0011/0012/0013) added `real_mode_proofs` entries B2-a-1/B2-a-2/B2-a-3
— **synthetic** I8 block-scale dequant integration proofs through the real-mode
`_moe_mlx` branch. These are partial B2 evidence only: they use *our own*
dequantized-float experts (isolation gate) plus a pure-Python reference at
`≤1e-3`, which is **circular** for the *real routed payload* decode question
(ADR 0007 §4).

ADR 0007 §4 is binding on this sub-blocker: routed I8 real-payload
`dequantize_expert_packed("i8")` decode stays gated until an **independent,
trusted full-path reference** (the official DeepseekV4 routed-expert dequant or
a DS4-CPU harness) confirms the consumer convention (scale-application
order/orientation) for the real routed I8+F8_E8M0 1-D `block_size=16 axis=1`
layout. A torch/numpy reconstruction of our own `int8 * decode_e8m0(scale)`
formula is circular; the FP8 shim's `f8_e8m0_to_bf16` is our own `decode_e8m0`
→ circular; Transformers ships `Fp8Dequantize` (e4m3) and `Mxfp4Dequantize`
(FP4) only — neither defines the routed I8+UE8M0 micro-block dequant. FP4 is
absent from the real checkpoint expert families (ADR 0007 §Context table).

This sub-blocker therefore has **no synthetic proof seam and no locally
attainable independent trusted reference** — exactly the situation ADRs 0009
(stateful decode) and 0010 (B1 hc_mult>1 multi-layer) faced. The honest next
slice is a fail-closed *readiness diagnostic* (Option B in BA requirements and
`docs/backlog.md` Story 11.40), not a proof.

## Decision

Add an additive `b2_real_checkpoint_payload_readiness` block to the
`deepseek-v4-forward-parity-readiness` JSON report. The block is produced by a
pure fail-closed builder fed an introspection / header-only, read-only probe over
the **original** real DeepSeek V4 Flash F8 checkpoint (`HF_MODEL` /
`DS4_HF_MODEL`). (BA AC1 said `hf-f8shim`, but `shim_ds4_safetensors.py` rewrites
only `F8_E4M3`/`F8_E8M0` to BF16/F32 and leaves `I8` routed weights untouched,
so the shimmed snapshot cannot reproduce the ADR-0007 `I8`+`F8_E8M0` /
`F8_E4M3`+`F8_E8M0` classification; the probe therefore targets the original
checkpoint. This is an Architect refinement of BA AC1's path wording, flagged in
`agent-output/cmux-11-40/architecture.md` §0.4 + §Q3 — binding intent preserved.)

The probe re-derives ADR-0007 facts **live** via the existing Story 11.15a
proven-safe header-only path: `read_safetensors_header` (reads only dtype/shape;
never payload bytes) + `classify_checkpoint_expert_packing` +
`resolve_routed_block_layout`/`reconcile_routed_block_layout` (all in
`ds4_ft_mlx.deepseek_v4_dequant`, pure-Python, no mlx/torch import). It is bounded
to a fixed expert-family tensor set, pins the index SHA-256 for drift detection
(mirrors `.deepseek-v4-mapping-ok`), and records `adr_0007_binding=true`. It
skips cleanly (fail-closed default) when the checkpoint is absent.

The block records the ADR-0007-verified classification (routed `I8`+`F8_E8M0`
1-D `axis=1 block_size=16`; shared `F8_E4M3`+`F8_E8M0` 2-D 128×128; `fp4_absent`;
declared `[128,128]` advisory; routed block-size discrepancy) and the exact
missing-trusted-reference requirement, and emits
`decode_trusted_reference_available=false`, `real_payload_decoded=false`,
`proof_available=false`, `can_decode_payload=false`,
`dequantize_expert_packed_i8_status="NotImplementedError (raises)"`,
`dequantize_expert_packed_fp4_status="NotImplementedError (raises)"`. A
companion structured field `i8_dispatch_evidence` records the
synthetic-proven / real-unproven / shim-circular split, so the literal i8 status
string (scoped to the ungated discriminator form that spec §10.9 / the
`assertRaises` tests gate) cannot be misread as either "i8 fully fail-closed" or
"i8 fully proven".

The verdict stays fail-closed unconditionally: `status="fail-closed"`,
`decision="not-ready"`, `fail_closed=true`, `ready=false`. Successful real-payload
classification (or a future probe that finds an independent trusted reference) is
**never sufficient** — `decision/status/fail_closed/ready` stay
not-ready/fail-closed/true/false until BOTH (i) an independent trusted routed
I8+F8_E8M0 full-path reference lands AND (ii) a reviewed real-mode real-payload
decode proof (decoded real routed I8 expert == trusted reference at `≤1e-5` under
the venv python) is accepted as a `real_mode_proofs` entry. Neither happens in
11.40.

The block is written via additive DATA (`_write_json_atomic`), **not**
`_write_gate_marker`. It is not loadable/interpretable as a gate marker by
`convert-shimmed` or `_validate_forward_parity_marker`. The top-level readiness
schema stays `1` (backward-compatible additive key, mirroring ADRs 0009/0010).

## Consequences

- The readiness report now machine-readably prevents B2-a-1/B2-a-2/B2-a-3
  synthetic I8 partial-evidence proofs (and the FP8 shim path) from overclaiming
  real-checkpoint-payload routed I8 decode parity — the same value-prop 11.36
  provides for B0a hyperconnection evidence vs B1, and 11.35 provides for the
  stateful/decode seam.
- `real_mode_proofs.proofs_total` stays 8 (no new entry). The hard counters
  `full_forward_parity=false`, `marker_earned=false`, `blockers_count=4`,
  `status="not-ready"`, `coverage.fixtures_total=19` are unchanged.
- `.deepseek-v4-forward-parity-ok` and `model-4bit` stay absent;
  `convert-shimmed` still hard-requires the absent forward-parity marker;
  `dequantize_expert_packed("fp4",...)` and the ungated
  `dequantize_expert_packed("i8", b"\x00", scales=b"\x7f", shape=(1,))`
  discriminator form still raise `NotImplementedError`.
- No production/vendor/spec math edit; no gate lift; no new math, no new
  classifier (reuses `classify_checkpoint_expert_packing`), no parallel
  implementation (ADR 0002 anti-slop).
- Out of scope: rewriting spec §10.9's `dequantize_expert_packed("fp4" | "i8")`
  sentence (which predates the 11.22/11.24 metadata-gated i8 router) — the i8
  nuance is recorded in the block's `i8_dispatch_evidence` and here, not by
  editing spec math semantics in this slice; a future docs-cleanup slice may
  reconcile §10.9 if explicitly tasked.
