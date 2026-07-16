# Story 13.3b-5g — Architect re-entry after packed-FP4 real-smoke OOM

## Goal
Adjudicate the single authorized real 4096 smoke RED. No model/shards/dataset run and no second smoke authorization.

## Read
- commit `5dee4ce`
- ADR 0028, r3 reduction architecture, BA requirements
- final r4 review/test reports
- `agent-output/cmux-13-3b/smoke-4096-routed-fp4-20260714-135103.log`
- run-only wrappers `run_smoke_4096_routed_fp4.py` / `.sh`
- current routed integration/custom VJP and trainer/checkpoint behavior.

## Authoritative evidence
- Exact pinned 4096/20 command; `mx.disable_compile()`; graph limit `400_000_000_000`.
- No concurrent heavy process.
- Validation finite: `18.109817504882812` in `82.645s`.
- Post-validation active `160,085,895,190`; peak `205,523,058,182`; cache `9,252` bytes.
- Snapshot of 96 trainable tensors changed active by only ~22.3MB.
- `SparseMoeBlockNN.__call__` unconditionally calls package-local `routed_fp4`; no dense fallback.
- First backward aborted before gradient callback:
  `[METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)`
- exit `134`; only `adapter_config.json`; no `adapters.safetensors`; no process remains.
- No retry/shorter fallback authorized.

## Required analysis
1. Explain why one-layer/operation memory probes did not predict 43-layer first-backward peak.
2. Separate retained activation/checkpoint graph, attention/shared-expert paths, routed primitive VJP lifetime, and Metal command-buffer transient hypotheses.
3. Inspect whether `grad_checkpoint(model.layers[0])` actually checkpoints all 43 layers for this model structure.
4. Define lightweight tracked diagnostics that can falsify each hypothesis without loading real shards/model.
5. Decide next minimal architecture: multi-layer synthetic peak proof, checkpoint integration fix, custom primitive lifetime redesign, layer-serial backward, or STOP Path A.
6. State exact acceptance and STOP gates. No real smoke authorization.

## Deliverable
Write `agent-output/cmux-13-3b/architecture-13-3b-5g-first-backward-oom.md`. Amend durable docs only if needed for the decision. Marker only success; unwrapped JSON.