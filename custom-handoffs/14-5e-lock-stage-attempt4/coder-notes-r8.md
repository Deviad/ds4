# Story 14.5e — Coder r8 raw-production trust-binding + mandatory matrices

## Scope

Implemented exact `task-coder-r8.md` against `architecture-r3` and Reviewer r7 standby requirements. Production remains synthetic-only: no model, dataset, provider, training, inference, cleanup, commit, or push.

## Production corrections

- B4 captures the Phase A identity root once from the independently captured strict report bytes before Phase A validation. The candidate report cannot mint its own trusted identity root.
- Canonical report validation binds candidate identity to the explicit trusted object plus canonical identity bytes/hash and retained runtime/report roots. Final validation binds `trusted_identity` to retained Phase B identity and validates canonical snapshot hashes without reopening phase reports.
- Success publication passes retained Phase A and Phase B roots explicitly; phase report hash values come from the one readback snapshot.
- Owner and candidate cleanup records preserve primary, unlock, and close outcomes. Unproven close retains descriptor ownership/flock truth and blocks later lock/publication operations; successful close clears owner state even when unlock failed.

## Matrix evidence

- Identity/report root mutation matrix: independent trusted object, bytes, hash, candidate identity, candidate snapshot bytes, and candidate snapshot hash.
- Candidate cleanup matrix: release/unlock × close outcomes, including primary-error binding and uncertain registry truth.
- Existing tracked B4 order, authorization/policy, report/marker schema, publication seam, historical, artifact, lock, and catalog matrices remain green.

## Validation

- Full pilot: `671 passed, 1 warning`.
- Exact documented six-file suite: `917 passed, 3 skipped, 1 warning, 2 subtests passed`.
- `python3 -m py_compile scripts/ds4_segmented_pilot.py tests/test_ds4_segmented_pilot.py`: PASS.
- `git diff --check`: PASS.
- All verdict-contributing test files are tracked.
- No `.cmux-status` marker was staged before this handoff.

## Gate boundary

Coder evidence only. Reviewer r8 and Test Manager r8 remain independent gates. Real A4/B4 execution remains unauthorized.
