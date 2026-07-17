# Story 14.5 — Coder r4 terminal-safety closure

## Scope

Direct TDD synthetic-only fix from `task-coder-r4.md` and `custom-handoffs/standby/review.md`. No delegate, real model/dataset/adapter access, training, inference, CUDA, distributed execution, commit, or push.

## Changes

- `_acquire_ft_lock(workspace=None)` resolves the current `PILOT_WORKSPACE` at call time; `run_phase()` passes the current workspace explicitly and records the same lock path.
- Added success-evidence rollback: any report or marker publication failure removes all success markers before failure evidence can replace reports. Rollback falls back to renaming a marker out of the evidence namespace if unlink itself fails. Failure markers bind to the report they name; marker-write failures are surfaced without leaving OK evidence.
- Existing one-attempt reports/markers are preserved byte-for-byte before watchdog/config work and when a second invocation is rejected; watchdog-cancellation failures are captured while lock release and failure evidence still run.
- Partial lock acquisition cleans its unowned file on owner-write/fsync failure; watchdog installation disarms SIGALRM if backup-thread setup fails.
- Added terminal mutation-point inventory covering parser/config, preflight, start-save, provider, callback, checkpoint validation, report/marker writes, timeout/watchdog, lock release, Phase B success/failure, and final aggregation, with executable dynamic injections for terminal failures and report/marker boundaries.
- Added byte-exact Phase A/Phase B catalog command oracles and default `smoke-train` / `continue-train` command guards.
- Qualified architecture/technical/backlog claims: r4 remains synthetic and real execution remains blocked pending independent gates.

## TDD evidence

RED first:

- lock default test accessed the import-time `/Volumes/Data NVME` default after monkeypatching `PILOT_WORKSPACE`;
- final-marker injection left `.ds4-segmented-pilot-phase-b-ok` behind.

GREEN after implementation:

```text
59 passed, 1 warning in 0.66s

Exact six-file synthetic/protected suite:

```text
305 passed, 3 skipped, 1 warning, 2 subtests passed in 21.75s
```
```

Focused command:

```bash
PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" \
PYTHONDONTWRITEBYTECODE=1 \
python-envs/mlx/.venv/bin/python -m pytest -q tests/test_ds4_segmented_pilot.py
```

Coverage added includes all four success evidence write boundaries, path-isolated lock acquisition, report/marker hash binding, watchdog install/cancel, terminal mutation inventory, Phase B evidence, and byte-exact catalog/default command guards.

## Boundaries

No real path was opened or mutated by implementation or synthetic tests. No `.cmux-status` marker is staged in git. Independent Reviewer PASS and Test Manager GREEN remain required before any real authorization.
