# Story 14.5c Coder Notes

## Scope

Synthetic TDD implementation only. No model, dataset, adapter, provider, training, inference, cleanup, commit, push, delegate, or cmux access.

## Implementation

- `canonical_tensor_digest()` now accepts absent, JSON `null`, and object `__metadata__`; all other JSON types still fail closed. Duplicate-key, tensor schema/layout, bounded payload, manifest, physical SHA-256, and `canonical_tensor_digest_v1` behavior remain strict.
- Repinned live pilot helpers and runtime routing to fixed attempt-3 namespace and `--attempt 3`.
- Retired attempt-2 runnable catalog entries; added explicit non-default A3/B3 catalog entries with FD-attested `noclobber` logging and `tee /dev/fd/3`.
- Added immutable attempt-2 historical evidence constants/verifier and attempt-3 collision/publication routing.
- Updated tracked synthetic tests and canonical documentation already present in the handoff.

## TDD evidence

- RED: focused null-metadata and attempt-3 tests failed against the old parser/attempt-2 surface.
- GREEN: focused parser/namespace tests passed.
- Focused tracked suites: `341 passed`.
- Canonical six-file regression: `536 passed, 3 skipped, 2 subtests passed`.
- Structural checks: `py_compile`, `git diff --check`, and `git ls-files --error-unmatch` passed.

Reviewer/Test Manager gates and any real A3/B3 authorization remain outstanding and were not performed.
