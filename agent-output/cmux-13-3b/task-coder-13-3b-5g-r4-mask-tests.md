# Story 13.3b-5g r4 — independent masked-component test oracles

Use `openai-codex/gpt-5.5` high. TDD RED→GREEN. Test-only; no helper/production/report semantic edits unless a genuine implementation defect appears, in which case STOP.

Read `review-13-3b-5g-r3.md` and Architect r2 mask contract.

Add load-bearing runtime tests whose expected values do NOT call `_routed_only`, `_zero_residual_identity_like`, or reuse `_masked_layer_forward` composition:
1. Direct zero-valued identity expression: exact zero forward, exact shape/dtype, gradient exactly ones/nonzero.
2. Ordinary model forward equals masked `full` forward.
3. All five masks: exact output shape/dtype, finite forward and input gradients, required nonzero input-gradient behavior.
4. Direct routed result derived independently from ordinary full MLP output minus shared-expert output; compare to selected routed branch forward and input gradient.
5. Exact LoRA trainable key set unchanged before/after every mask.
6. Mutation-sensitive proof: in-memory replace routed/identity helpers with `zeros_like`; independent oracle must detect both semantic failures. Normal suite remains GREEN.

Do not build expected values from helper under test. Preserve raw-gradient/report bytes if helper unchanged. Run focused suite, both compile orders, tracked-only baseline, make/diff/tracking/hash/index checks. Stage exact tests/logs/updated notes.

Update `agent-output/cmux-13-3b/coder-13-3b-5g-r3-notes.md` with r4 evidence. Marker only full GREEN; unwrapped JSON.