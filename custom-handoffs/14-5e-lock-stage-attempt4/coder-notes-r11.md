# Story 14.5e — Coder r11 retained-history and unpatched lifecycle closure

## Result

GREEN synthetic closure on the current staged candidate. Production lifecycle remains unpatched in the new B4 coverage: the real lock acquisition/release path, raw report/marker/final validators, dependency admission, success writer, and final publication were exercised with only API/training seams faked.

## Production correction

- `run_phase()` now assigns `report["attempt_3_historical_evidence"]` from the original pre-training `historical_attempt3_evidence` object.
- Removed the API-only `{}` substitution and removed the post-training attempt-3 re-verification path.
- Attempt-4 report validation now runs regardless of whether a hardware API seam is supplied.
- Existing writer/final-root arguments continue receiving the same retained history object/value.

## Tracked coverage

- `test_r11_api_hardware_seam_retains_pretraining_attempt3_history`: red before the production correction; proves one POST_LOCK history verification, report identity continuity, writer argument continuity, and raw report validation with fake API/training seams.
- `test_r11_persisted_phase_a_real_b4_production_lineage`: persists synthetic Phase A report/OK marker, then runs real B4 admission/dependency/validator/writer/final publication with distinct policy-valid dynamic observation and fake API/training only.
- `test_r11_cleanup_uncertain_state_blocks_all_lifecycle_and_publication`: verifies cleanup-uncertain rejection for PRE_LOCK, acquisition, failure writer, success writer, and publication surfaces; confirms close-success releases the candidate flock and leaves no uncertain registry.
- Existing r10/r9 matrices remain enabled for every explicit final root, post-capture mutation, marker field, caller failure seam, release/rollback/candidate truth table, and final publication.

## Validation

- Focused r11 plus terminal mutation gate: `10 passed, 1 warning`.
- Full pilot: `701 passed, 1 warning`.
- Exact documented vendor-first six-file suite: `947 passed, 3 skipped, 1 warning, 2 subtests passed`.
- `py_compile`: PASS.
- `git diff --check`: PASS.
- `git diff --cached --check`: PASS before final r11 staging.
- All six canonical test files are tracked via `git ls-files`.

No real A4/B4 invocation, model/data/provider/training/inference execution, cleanup, commit, or push occurred. No completion marker is created until exact staging/no-drift checks pass.
