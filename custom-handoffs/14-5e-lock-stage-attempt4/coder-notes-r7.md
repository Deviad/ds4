# Story 14.5e — Coder r7 raw-production closure

## Scope

Implemented the exact `task-coder-r7.md` retained-root/order closure against `architecture-r3` and Reviewer r6 B1-B7. Production paths remain synthetic-only: no model, dataset, provider, training, inference, cleanup, commit, or push.

## Production corrections

- B4 `run_phase()` now retains independent attempt-1/2/3 roots, strictly parses separate Phase A/B authorizations, captures Phase A report and complete resume-bound marker bytes once, then runs fresh runtime preflight and authorization/policy validation before strict Phase A/dependency validation.
- Runtime identity capture retains the exact preflight object, canonical compact bytes, SHA-256, phase, and lifecycle. Writers and validators reject caller reconstruction or post-capture mutation.
- Identity snapshots use duplicate-key rejection and require exact canonical compact sorted JSON bytes and semantic equality.
- Final validation strictly decodes retained Phase A/Phase B report bytes and requires exact semantic/hash equality to the published phase objects and retained lineage.
- Phase success markers persist the full strict resume binding; final markers retain their exact final schema.
- PRE_LOCK cleanup now aggregates primary, unlock, and close failures and records the same truth in the uncertain-FD registry. Uncertain cleanup remains publication-blocking.
- Canonical docs now describe the implemented B4 order, retained identity roots, complete markers, and strict publication behavior.

## TDD evidence

RED-first r7 tests were added to tracked `tests/test_ds4_segmented_pilot.py` for complete Phase A marker persistence, canonical strict identity bytes, writer retained-preflight enforcement, and PRE_LOCK cleanup error truth. Existing mutation matrices remain raw production validator/writer tests.

- Focused r7 RED baseline: 4 failures before implementation.
- Focused r7 regression: `4 passed, 653 deselected`.
- Full pilot: `657 passed, 1 warning`.
- Exact documented canonical six:
  `903 passed, 3 skipped, 1 warning, 2 subtests passed`.
- Changed-source compile: passed for `scripts/ds4_segmented_pilot.py` and `scripts/finetune_ds4.py`.
- `git diff --check` and `git diff --cached --check`: passed before final staging refresh.
- All six canonical verdict-contributing test files are tracked.

## Gate boundary

No completion marker was created until the exact canonical six passed. No real A4/B4 execution is authorized by this coder slice. Reviewer r6 and Test Manager r6 remain independent gates; this handoff is coder evidence only.
