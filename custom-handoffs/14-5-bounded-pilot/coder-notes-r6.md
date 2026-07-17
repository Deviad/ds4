# Story 14.5 — Coder r6 production-path closure

## Scope

Direct TDD synthetic-only work from `task-coder-r6.md` and `custom-handoffs/standby/review.md`. No role dispatch, delegation, real model/dataset/adapter access, training, inference, CUDA, distributed execution, commit, or push.

## Changes

- `run_phase()` now recognizes retained partial lock ownership when `_acquire_ft_lock()` raises before `lock_acquired=True`. It performs exactly one release attempt and atomically quarantines a surviving partial lock outside `.ds4-ft.lock` if release unlink fails; ownership globals are cleared only after release or quarantine.
- Phase B dependency validation now requires marker `report_path` to resolve exactly to the canonical Phase A report, verifies the marker hash against that path, and requires the loaded hashed JSON payload to equal the report object supplied to validation.
- Added path-substitution and report-content-substitution mutations.
- Added end-to-end Phase A partial-progress failure coverage with one recorded update and ordered training failure → watchdog cancel → release → failure report → fail-marker evidence.
- Applied the resolved temporary-root filesystem guard to direct success/failure evidence boundaries, one-attempt preservation, and partial-lock terminal tests.
- No canonical docs were changed for this pending-gate slice. No `.cmux-status` marker was staged.

## TDD evidence

RED first:

```text
Phase B path substitution test failed before canonical-path binding was implemented.
```

Focused GREEN:

```text
71 passed, 1 warning-free test-suite output
```

Canonical real-trainer targets:

```text
2 passed, 45 deselected, 1 warning
```

Exact six-file synthetic/protected suite:

```text
318 passed, 1 warning, 5 subtests passed in 22.07s
```

Additional gates:

- `python3 -m py_compile scripts/ds4_segmented_pilot.py tests/test_ds4_segmented_pilot.py`: PASS.
- `git diff --check` and `git diff --cached --check`: PASS.
- Modified test file is tracked: PASS.
- Frozen provider SHA-256: `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518`.
- Frozen smoke SHA-256: `ec17950d985578f3cd97b51734527bfc47e6c399ebe84fad8ffcb12beae18ad8`.
- Vendor inner HEAD: `80fab4e419a57f9465bb9e2f4e90010d645e124c`; vendor clean.

## Boundary

Synthetic only. No real model, dataset, adapter, training, inference, CUDA, distributed, or network execution performed. Independent review and Test Manager gates remain external requirements.
