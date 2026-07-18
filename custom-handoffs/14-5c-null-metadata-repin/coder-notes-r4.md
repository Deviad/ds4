# Story 14.5c — Coder r4

## Scope

Direct RED-first synthetic implementation only. No delegate, cmux, real assets, training, cleanup, commit, or push.

## RED

Added r4 tests for exact attempt-3 marker schema, mandatory B3 Phase-A admission lineage, and post-admission A3 report mutation. Initial focused run: `3 failed, 334 deselected`.

## Implementation

- Added immutable A3 admission lineage capture containing exact report/marker bytes (base64), hashes, schemas, authorization, histories, identity, contract, artifacts, and resume source.
- Added strict phase-A/B3/final marker validation and report/marker re-read checks.
- Added B3 report lineage to contract digest and canonical report validation.
- Added separate Phase-A authorization propagation into B3 catalog wrappers and launch checks; B3 remains externally authorized separately.
- Added import-time frozen attempt-2 semantic, manifest-hash, file, and absence roots; mutable expected dictionaries cannot substitute historical evidence.
- Preserved null/absent/object metadata compatibility and all tensor/digest/layout checks.
- Updated `docs/architecture.md` and `docs/technical-spec.md` after behavior landed.

## Validation

`PYTHONPATH=. uv run --with pytest pytest -q tests/test_ds4_segmented_pilot.py tests/test_finetune_ds4.py`

Result: `388 passed, 2 warnings`.

Also passed structural compilation and focused r4 tests. No `.cmux-status` marker created because this direct run was explicitly requested without cmux.
