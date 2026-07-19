# Story 14.5e — Coder r14 trusted dispatch + coherent staging

## Result

Trusted explicit `phase_spec` dispatch now owns attempt and namespace. Report `phase`, typed `attempt`, and `namespace` must match the trusted spec exactly; mismatch rejection occurs after cleanup state is initialized and removes all attempt-4 success markers. No real A4/B4 model, dataset, provider, training, inference, cleanup, commit, or push ran.

## RED-first corrections

- Added the r14 mismatch matrix inside each existing Phase A authorization, Phase A admission-lineage, and Phase B authorization disagreement test branch. Cases cover report attempt `1`, `3`, wrong type, namespace mismatch, and phase mismatch with pre-created A4 success markers; every case asserts phase-A marker, phase-B marker, final marker, and final report absence.
- RED was observed before the production correction: report/spec mutation reached the unrelated root-disagreement error, so report/spec dispatch was not authoritative.

## Production correction

- `_write_success_evidence()` no longer reclassifies an explicit `phase_spec` from report fields.
- Explicit attempt-4 specs are the sole dispatch source for attempt and namespace. Report phase, attempt type/value, and namespace are strict-checked against that spec.
- Cleanup certainty is checked before report/spec rejection. Rollback remains initialized before all early rejection branches, including incomplete test specs; marker removal falls back safely when a malformed spec lacks `namespace_paths`.

## Validation

- Focused `tests/test_ds4_segmented_pilot.py`: **735 passed, 1 warning**.
- Exact documented six-file command: **981 passed, 3 skipped, 1 warning, 2 subtests passed**.
- r13 publication-byte and early-root-disagreement tests remain green.
- Independent Reviewer and Test Manager gates are not claimed here; review r13 remains standby.
- No marker was written. All intended files are staged as one coherent index; `.cmux-status/*` is not staged; tracked worktree drift is empty after staging.
