# Story 14.5c Coder r3 Notes

## Scope

Direct RED-first r3. No delegate, cmux, real assets, model, dataset, adapter, provider, training, inference, cleanup, commit, or push.

## RED evidence

Added production-bound r3 tests before repair. Initial focused run was `4 failed`: catalog had no external authorization gate, reserved B2 output path was wrong, B3 compared A3 authorization to B3 authorization, and final publication accepted mutated Phase A lineage.

## Repairs

- Catalog no longer calls or mints `canonical_attempt3_authorization()`. Runnable A3/B3 commands require externally supplied reviewed JSON/file values. Added phase-specific A3/B3 authorization CLI inputs; both phase authorizations are embedded separately and runtime validation remains authoritative.
- B3 dependency validation no longer compares A3 authorization with B3 authorization. A3 report validation consumes A3's own command-bound authorization; B3 launch validates separate B3 authorization.
- Corrected reserved attempt-2 Phase B output absence path.
- Final attempt-3 publication re-reads and hashes A3 report/OK marker and B3 report, strictly validates both, and emits final lineage binding attempt, namespace, both histories, both report hashes/paths/contracts, and both phase authorizations. Phase mutation after admission fails closed.
- Added invalid metadata type × every Phase A checkpoint matrix, separate phase authorization catalog test, B3 lineage test, final publication mutation test, and retained synthetic artifact/parser/loadability coverage.
- Updated architecture, technical spec, and backlog after behavior became true.

## Verification

- Focused r3 selection: `208 passed, 126 deselected, 1 warning`.
- Full `tests/test_ds4_segmented_pilot.py`: `334 passed, 1 warning`.
- Full `tests/test_finetune_ds4.py`: `51 passed, 1 warning`.
- Canonical six-file regression: `580 passed, 3 skipped, 1 warning, 2 subtests passed`.
- `py_compile`: passed.
- `git diff --check`: passed.
- Verdict tests tracked: `tests/test_ds4_segmented_pilot.py`, `tests/test_finetune_ds4.py`.

Real A3/B3 remains blocked pending independent Reviewer PASS, Test Manager GREEN on the exact staged revision, and fresh explicit phase authorization.
