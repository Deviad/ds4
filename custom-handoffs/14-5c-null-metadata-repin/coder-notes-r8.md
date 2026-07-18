# Story 14.5c — Coder r8 trusted-root and snapshot-hash closure

## RED-first

- Reproduced the prior final-validation boundary failures before the r8 implementation.
- Targeted regression baseline was red once independent B3/cross-phase roots became mandatory.

## Production changes

- `validate_final_attempt3_report()` now requires independently retained Phase A3 authorization, Phase B3 authorization, cross-phase identity, attempt-1 history, attempt-2 history, and Phase A admission lineage.
- Final phase snapshots pass those retained roots into canonical validation; snapshot fields are never promoted to trust roots.
- Final derivation uses the SHA-256 values returned by the strict byte snapshots. It no longer reopens phase report paths through `file_sha256()`.
- Phase-A admission marker validation accepts the already-captured report snapshot hash, avoiding a second phase-report hash read during final validation.
- `_write_success_evidence()` and `run_phase()` carry the independently retained B3 authorization and runtime identity/history roots into final publication.

## Tests

- Added trusted-root substitution rejection coverage.
- Added an instrumented one-read/hash test that fails if final validation reopens either phase report through `file_sha256()`.
- Updated existing final-publication tests for the explicit retained-root contract.
- `PYTHONPATH=/Users/spotted/projects/ds4-finetuning uv run --with pytest pytest -q tests/test_ds4_segmented_pilot.py`: **392 passed**.
- `python3 -m py_compile scripts/ds4_segmented_pilot.py`: **PASS**.
- `git diff --check`: **PASS**.
- Full `tests/` suite: **1210 passed, 22 failed, 18 skipped, 89 subtests passed**; failures are outside this slice in existing MLX/Transformers parity/readiness coverage and one missing exported Transformers symbol.

## Boundary

Synthetic tests only. No real model, dataset, adapter, training, inference, CUDA, distributed execution, cleanup, commit, push, delegate, cmux operation, or `.cmux-status` marker was performed or created.
