# Story 14.5 — Coder r7 final TOCTOU/tmp-guard closure

## Scope

Direct TDD-only synthetic fix from `task-coder-r7.md` and `custom-handoffs/standby/review.md`. No delegate, real model/dataset/adapter access, training, inference, CUDA, distributed execution, commit, or push.

## Changes

- Phase B now reads the canonical Phase A report bytes once, computes SHA-256 from that exact byte snapshot, strict-parses the same bytes with duplicate-key rejection, and compares the parsed object with the supplied report.
- Added a mutation-sensitive hash/read-boundary test: a replacement performed after the legacy hash hook cannot make the supplied report bind to a different payload.
- Extended the dependency test with duplicate-report-key rejection.
- Installed `install_tmp_fs_guard()` in every terminal test touching `run_phase()`, evidence writers, lock acquire/release, or watchdog installation/cancellation.
- Added an AST coverage assertion to prevent new unguarded terminal tests.
- Test count preserved by folding new checks into existing terminal/dependency tests.

## TDD evidence

RED first:

- Legacy Phase B implementation accepted the hash/read-boundary replacement (`test_phase_b_dependency_requires_complete_binding_and_exact_report_identity` failed because no `PilotError` was raised).
- Legacy implementation did not reject duplicate Phase A report keys.

GREEN:

- Focused pilot suite: `71 passed`.
- Exact documented six-file suite:

```text
317 passed, 3 skipped, 1 warning, 2 subtests passed in 21.59s
```

- Direct real-trainer targets:

```text
2 passed, 45 deselected, 1 warning in 0.06s
```

- `python3 -m py_compile scripts/ds4_segmented_pilot.py tests/test_ds4_segmented_pilot.py`: PASS.
- `git diff --check`: PASS.
- `git diff --cached --check`: PASS.
- Modified verdict test tracking: PASS (`tests/test_ds4_segmented_pilot.py` tracked).
- Final AST containment inventory: PASS (`uncovered=[]`).

## Boundary

Synthetic tests only. No real model, dataset, adapter, training, inference, CUDA, distributed, network, or protected GGUF execution performed. No `.cmux-status` marker staged. Reviewer and Test Manager gates remain external requirements.
