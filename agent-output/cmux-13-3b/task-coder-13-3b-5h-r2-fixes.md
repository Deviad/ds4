# Story 13.3b-5h r2 — Coder fixes for Reviewer FAIL

Use `openai-codex/gpt-5.5` high. TDD RED→GREEN. Diagnostic-only; no production/vendor/primitive/Metal edits, real assets, classification, redesign, smoke, training, or commit.

Read `review-13-3b-5h.md` in full.

Required:
1. Record `active_baseline` immediately before differentiated graph construction with no pending graph. Do not evaluate a separate forward after baseline and before VJP. Sequential rows measure per-stage immediate baseline corresponding to each stage peak, then aggregate with documented matching semantics; no pre-loop reuse.
2. Add independent mutation-sensitive VJP-leg tests:
   - attention-forward+routed-VJP proves attention LoRA-gradient partition stopped and expected routed nodes/gradients remain;
   - routed-forward+attention-VJP proves routed-output VJP stopped while attention LoRA gradients remain;
   - tests must fail when either stop_gradient is removed. Do not use helper summary booleans as oracle.
3. Replace global-max materialized parity predicate with true elementwise `atol=rtol=1e-5` for loss, every LoRA gradient leaf, and input gradient; preserve exact key/shape checks. Add counterexample where old predicate passes but elementwise allclose fails.
4. Regenerate all 25 rows fresh after telemetry/parity changes. No patch/reuse.
5. Run focused both orders, prior 5g suite, full 46-file canonical baseline, make/diffs/tracking/protected hashes/exact staged bytes.

STOP on parity/nonfinite/missing peak/cell failure. Update `coder-13-3b-5h-notes.md`; stage refreshed report/tests/helper/logs. Marker only complete GREEN; unwrapped JSON.