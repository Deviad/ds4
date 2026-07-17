# Story 14.5a A2 pre-log failure closeout requirements

As a DS4 fine-tuning operator (WHO), I want the authorized Phase A2 pre-log failure recorded exactly and closed without repair or retry (WHAT), so that the consumed authorization, clean attempt-2 namespace, and Phase B2 block remain auditable (WHY).

## Observed result

- Revision: `c910d1ba33912236ee87f3f9bdfb5b31edece6e7`.
- Exit: `1`.
- Terminal failure: `pilot launch check failed closed: MLX version mismatch: None`.
- Canonical interpreter: `importlib.metadata.version("mlx") == "0.31.2"`; `mlx.__version__ is None`.
- Current preflight: `scripts/ds4_segmented_pilot.py::_runtime_preflight()` validates `getattr(mlx, "__version__", None)` against `0.31.2`.
- Training/provider calls/updates: `0`.
- Phase A2 log/report/OK/fail paths: absent.
- Adapter output: absent.
- Attempt-2 final OK/fail paths: absent.
- Retry: not performed.
- Attempt-2 namespace: otherwise clean; no cleanup or overwrite occurred.

## Disposition

The one-attempt A2 authorization is consumed. This closeout authorizes documentation only: no repair, code edit, cleanup, commit, push, retry, fallback, alternate namespace, or attempt-3 allocation. Any repair requires a fresh reviewed repair/repin and fresh explicit operator authorization before another A2 invocation. Phase B2 remains blocked until a newly authorized A2 succeeds, its exact evidence verifies, and B2 receives separate explicit operator authorization.

## Acceptance criteria

1. `docs/backlog.md` records the exact revision, exit, terminal failure, version mismatch diagnosis, zero training/provider calls/updates, absent A2/final artifacts, no retry, otherwise-clean namespace, and consumed authorization.
2. `training-next-status.md` states the same operational result and current gate.
3. No technical specification change is made because this closeout changes execution status only, not the runtime contract.
4. No repair, code edit, test run, training/provider call, cleanup, commit, or push occurs.
5. B2 remains blocked; fresh reviewed repair/repin plus fresh explicit authorization is required before any new A2 invocation.
