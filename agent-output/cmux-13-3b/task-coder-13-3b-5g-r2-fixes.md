# Story 13.3b-5g r2 — Coder diagnostic contract remediation

Use user-pinned `openai-codex/gpt-5.5` high. TDD RED→GREEN. No commit, production edits, real assets, redesign, classification, or smoke.

Read in full:
- `architecture-13-3b-5g-r2-contract.md`
- `requirements-13-3b-5g-r2-contract.md`
- `review-13-3b-5g.md`
- existing helper/test/fixture/report/5g notes.

Repair all Reviewer blockers exactly:
1. Probe B/D: synthetic packed payload, quantize applicable base leaves group32/bits4, freeze, trainer-equivalent rank8/scale20/dropout0 LoRA only q_a/q_b/kv on last min(D,16); exact 6L key/shape/gradient assertions.
2. Bind each differentiated forward once. Remove doubled model/layer/chain calls.
3. Measure real active_after_graph at Architect-defined graph-ready boundary; sequential row uses max over measured per-layer boundaries.
4. Measure actual DOT CustomKernel calls and exact `6*D*N` Probe C node counts.
5. Keep exact 64 rows/order. E=2 low-E non-comparable control; normalized E=8/32/128/256 uses K=6 and 384 assignments. Add exact comparison metadata; no mixed trend.
6. Make 43-layer fixture load-bearing: exact schema, validated config derivation, repository-relative provenance/source hashes, fixture hash, malformed-copy failures.
7. Implement test-only masked Probe D execution without mutating LoRA registration; exact branch shapes/dtypes, full equality, zero-valued nonzero-input-gradient replacements, routed branch parity.
8. Probe A exact nonempty gradient key/shape equality and separate loss/gradient finiteness.
9. Add all 15 Architect-required behavioral gates: row identity/order, fresh-child nonces, serial intervals, plan/report hashes, failures/signals/timeouts, no real access, 8GB/180s, tracking/hashes.
10. Fully regenerate report; never patch/reuse old numeric rows.

Run focused tests, complete helper serially, tracked-only baseline, make, diff checks, tracking and protected hashes. Preserve explicit STOP on any missing cell or timeout.

Write `agent-output/cmux-13-3b/coder-13-3b-5g-r2-notes.md`. Marker only full GREEN; unwrapped JSON.