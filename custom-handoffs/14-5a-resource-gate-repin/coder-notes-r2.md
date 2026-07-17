# Story 14.5a — Coder r2 notes

## Closure

- Added strict Phase B2 pre-log validation through the shared `validate_phase_b_dependency()` path.
- B2 launch-check now strict-reads the A2 report and OK marker, checks attempt/namespace, two-step evidence, provider/update cardinality, immutable identity, active log binding, contract digest, resume file SHA, and canonical tensor digest before any B2 namespace mutation.
- Attempt-2 contract digest now binds the complete namespace path map, active log, effective values, immutable identity, and `ATTEMPT1_EVIDENCE_SHA256` manifest.
- Retired `ds4-segmented-pilot-phase-a` and `ds4-segmented-pilot-phase-b` from runnable MLX/BACKEND/COMMAND catalogs. Attempt 2 remains explicit and non-default.
- Preserved attempt-1 effective/contract behavior by keeping `attempt` and `log_path` out of default attempt-1 effective values and applying the expanded digest payload only to attempt 2.
- Added collision, dependency-mutation, manifest, log-FD, exact-namespace, catalog, and resource-gate synthetic coverage.
- Updated `docs/architecture.md`, `docs/technical-spec.md`, and `docs/backlog.md` to describe the repaired pending-gate contract.

## Synthetic validation

- `tests/test_ds4_segmented_pilot.py`: 106 passed, 1 warning.
- No model, dataset, adapter, MLX training, inference, smoke, CUDA, distributed, network, cleanup, commit, or push execution performed.
- No `.cmux-status` marker staged.

## Remaining gates

Fresh independent Reviewer PASS and Test Manager GREEN are required. Real Phase A2 remains unauthorized; Phase B2 requires separate authorization after verified A2 evidence.
