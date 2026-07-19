# Story 14.5e — Coder r9 mandatory-root + named matrix completion

## Result

GREEN synthetic closure.

## Production

- `_validate_attempt4_report` now requires `trusted_identity_snapshot_bytes` and `trusted_identity_snapshot_sha256`.
- Rejects omission, partial omission, wrong type, malformed/value SHA, canonical-byte mismatch, and coordinated identity substitution.
- `validate_phase_b_dependency` carries the explicit snapshot object/bytes/SHA roots.

## Named tests

All nine names verified by grep and all passed:

- `test_r9e_identity_snapshot_roots_mandatory_matrix`
- `test_r9e_real_b4_run_phase_order`
- `test_r9e_dual_fresh_a4_b4_lifecycle_distinct_dynamic`
- `test_r9e_final_every_root_omission_and_substitution_matrix`
- `test_r9e_writer_final_post_capture_mutation_matrix`
- `test_r9e_release_four_outcomes_prior_error_competitor_matrix`
- `test_r9e_rollback_four_outcomes_prior_error_competitor_matrix`
- `test_r9e_candidate_four_outcomes_prior_error_competitor_matrix`
- `test_r9e_production_writer_persisted_marker_strict_b4_admission`

## Validation

- Focused r9e: `9 passed`.
- Full pilot: `680 passed`.
- Canonical six-file suite: `926 passed, 3 skipped, 1 warning, 2 subtests passed`.
- `py_compile`: PASS.
- `git diff --check`: PASS.
- All verdict-contributing test files tracked.
- No real A4/B4 execution, model/data/provider/training/inference execution, cleanup, commit, or push performed.
