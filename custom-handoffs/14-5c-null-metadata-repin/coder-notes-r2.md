# Story 14.5c Coder r2 Notes

## Scope

Synthetic RED-first closure only. No real model, dataset, adapter, provider, training, inference, cleanup, commit, push, delegate, or cmux access.

## Blocker closure

- Added exact six-key compact authorization JSON validation and wrapper binding. Authorization binds revision, canonical A3 command SHA-256, pilot/catalog source SHA-256, protected 43-file manifest SHA-256, and immutable attempt-2 manifest SHA-256. Admission validates before attempt-3 writes and repeats after lock/runtime identity checks.
- Repaired attempt-2 historical verification against the nested `lock_lifecycle` manifest schema, complete private expected snapshot, exact absolute ten-entry paths/hashes, strict outer manifest bytes, failure/absence facts, and hash-pinned report identity.
- Repinned contract/report lineage to attempt 3 and added both attempt-1 and attempt-2 historical objects plus authorization to canonical reports and contract digests. B3 consumes only exact A3 evidence.
- Added synthetic artifact/publication boundary tests for absent/null/object metadata, invalid metadata byte preservation/no-success, MLX-written null metadata loadability, authorization mutation, and historical verifier behavior.
- Updated canonical architecture, technical specification, and BA backlog status after implementation.

## Verification

- Focused selection: `172 passed, 176 deselected, 1 warning`.
- Canonical six-file regression: `543 passed, 3 skipped, 1 warning, 2 subtests passed`.
- `py_compile`: passed.
- `git diff --check`: passed.
- Verdict tests tracked: `tests/test_ds4_segmented_pilot.py`, `tests/test_finetune_ds4.py`.
- Protected smoke/provider hashes and 43-file protected manifest checked; protected bytes unchanged.

Reviewer/Test Manager independent gates remain outstanding. Real A3/B3 remains blocked; no authorization to execute is implied.
