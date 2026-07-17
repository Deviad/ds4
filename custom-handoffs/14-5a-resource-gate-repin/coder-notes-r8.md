# Story 14.5a — Coder r8

## Scope

Implemented direct, isolated, temp-only consumer oracles in tracked `tests/test_ds4_segmented_pilot.py` only. No production/runtime/vendor files changed by r8. No real model, dataset, provider, training, inference, cleanup, commit, push, or marker staging performed.

Added:

- Parameterized A2 canonical-validation matrix for `phase-a-start-checkpoint`, `phase-a-step1-checkpoint`, `phase-a-step2-checkpoint`, `phase-a-final-checkpoint`, and `phase-a-config`.
- Parameterized B2 canonical-validation matrix for `phase-b-start-checkpoint`, `phase-b-step1-checkpoint`, `phase-b-final-checkpoint`, `phase-b-config`, and `phase-b-resume`.
- Coherent namespace/report/contract rebinding helper; exact named failure boundaries; generic namespace/digest failures explicitly rejected.
- A2/B2 success-admission publication/no-write matrices with writer-call count zero and phase/final destination absence. B2 negatives preserve A2 report and marker bytes.
- Phase-B dependency matrix for all five A2 artifact/config keys plus direct `phase-b-resume` binding coverage. B2-produced start/step-1/final/config cells remain `N/A — produced after dependency admission` per r8 architecture.
- Executable catalog baseline-vs-one-key mutation matrix for all ten logical keys using captured launch argv, training argv, order, FD-backed log, and status propagation; no command-string/assignment-only oracle.
- Positive B2 publication assertions for phase report/OK and final report/OK, exact report paths/SHA-256 bindings, contract/namespace/attempt/output fields, final progression, and absent fail markers.

## TDD / verification

- Targeted non-catalog r8 matrices: `27 passed, 234 deselected, 1 warning`.
- Targeted complete r8 matrices: `34 passed, 3 failed, 224 deselected, 1 warning`.
- Canonical six-file suite after r8: `503 passed, 3 failed, 3 skipped, 1 warning, 2 subtests passed`.
- RED failures are the required direct executable-consumer defects:
  - `phase-a-step1-checkpoint`: mutation changes no observed launch argv, training argv, or runtime evidence.
  - `phase-a-step2-checkpoint`: mutation changes no observed launch argv, training argv, or runtime evidence.
  - `phase-b-step1-checkpoint`: mutation changes no observed launch argv, training argv, or runtime evidence.
- These are not fixture/generic-guard failures. The catalog stub observed launch-check then training, propagated status, created the FD-backed log, and showed changed values for all other applicable keys. Per r8 architecture, stop and escalate these missing-consumer defects; do not add production behavior in a test-only repin.
- `py_compile`: PASS for touched tests and protected scripts.
- `git diff --check`: PASS.
- `git diff --cached --check`: PASS.
- `git ls-files --error-unmatch tests/test_ds4_segmented_pilot.py`: PASS; verdict-contributing test is tracked.
- Protected hashes remain unchanged from r7:
  - `scripts/ds4_segmented_pilot.py` `02355fa330173951416e0e50652aa40e688f5601f79be3c3cc826d7160a10c31`
  - `scripts/finetune_ds4.py` `de303d1166f2ecd2428dc15f63c8be8ed22e9f30274008afd35be458ef4b188b`

## Gate

STOP/ESCALATE for fresh Architect/Reviewer/Test Manager disposition of the three real executable-catalog missing-consumer REDs. Real Phase A2/B2 execution remains blocked. Role marker is workflow status only and must not be staged.
