# Story 14.5c — Coder r7 snapshot + byte-verifier closure

## Scope

Direct TDD-only synthetic revision from `task-coder-r7.md` and `custom-handoffs/standby/review.md`. No delegate, nested agent, cmux, real model/dataset/adapter access, training, inference, CUDA, distributed execution, cleanup, commit, or push.

## RED-first changes

- Added a coordinated embedded-phase mutation regression: mutating the supplied Phase A object while re-deriving final aggregates is rejected against the immutable report snapshot.
- Added exact tensor descriptor-key and independent schema/value/payload mutation coverage.
- Replaced the r6 attempt-2 sparse/hash-spoof fixture with byte-authentic synthetic files and JSON. The fixture exercises `_verify_historical_snapshot()` and `_verify_historical_evidence_snapshot()` over real bytes, SHA-256, strict duplicate-key parsing, ten file descriptors, and eight absence facts. The production attempt-2 wrapper retains its local immutable trust literals.

## Production changes

- `_validate_final_phase_snapshot()` reads each canonical A3/B3 report once, hashes and strict-parses that byte snapshot, requires supplied phase objects to equal it exactly, then invokes the canonical phase validator with trusted identity, authorization, historical evidence, and Phase-A lineage.
- Final publication binds the returned snapshot hashes instead of reopening report paths through `file_sha256()` during aggregation.
- Safetensors tensor entries now require exact keys `{dtype, shape, data_offsets}`.
- Canonical docs now describe r7 as synthetic-only and keep A3/B3 authorization blocked.

## Verification

- R7 focused tests: `12 passed`.
- Pilot regression suite: `390 passed`.
- Exact documented six-file suite:

```text
636 passed, 3 skipped, 1 warning, 2 subtests passed in 101.02s
```

- `python3 -m py_compile scripts/ds4_segmented_pilot.py tests/test_ds4_segmented_pilot.py`: PASS.
- Modified test file is tracked; `git diff --check` and staged diff whitespace checks remain required.

## Boundary

Synthetic tests only. No real model, dataset, adapter, provider, training, inference, CUDA, distributed, network, A3, or B3 execution performed. No `.cmux-status` marker staged. Reviewer and Test Manager gates remain external requirements.
