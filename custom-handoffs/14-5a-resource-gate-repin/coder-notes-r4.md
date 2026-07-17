# Story 14.5a — Coder r4 trusted canonical admission

## Verdict

Implementation complete for synthetic-only TDD gate. Real Phase A2/B2 remain unauthorized. No model, dataset, adapter, training, inference, cleanup, commit, or push performed.

## Red → green

- Added positive valid-A2-to-B2 admission coverage before B2 destination creation.
- Added exact phase-B collision reporting for B2/final destinations while accepting all verified A2 inputs.
- Changed canonical fixture resource evidence to use production `_resource_evidence()` output.
- Added NoSuchProcess/ZombieProcess followed by later competitor tests and PID/RSS integer-normalization failure tests.
- Added independent command/identity/provider coordinated-substitution tests and centralized artifact namespace mutation coverage.
- Focused pilot suite: `149 passed, 1 warning`.

## Implementation

- B2 collision ownership is phase-specific: A2 output/report/log/OK/start/step1/step2/final/config paths are dependencies, not collisions; only B2/final destinations must be absent.
- Canonical resource validation now consumes the aggregate production schema (`total`, `counts_by_type`, `pids_by_type`, `unknown_pid_counts_by_type`) with exact key/type/order/count invariants.
- Added trusted canonical attempt-2 command construction and compare it independently against both effective pins and report commands.
- Added trusted immutable identity plumbing through canonical validation, B2 dependency admission, and runtime pre-publication revalidation.
- Provider evidence now requires exact schema and gradient path/shape/dtype pairings.
- `_validate_artifacts`, marker/report/final publication, and phase specs consume centralized namespace entries.

## Verification

- Focused exact pilot suite: `149 passed, 1 warning`.
- Six-file canonical suite: `393 passed, 3 skipped, 2 failed, 1 warning, 2 subtests passed`; two failures are pre-existing MLX compatibility failures because installed `mlx_lm.tuner.trainer` has no `TrainUI`, in `tests/test_ds4_segmented_loss_and_grad.py`.
- `git diff --check`: PASS; `git diff --cached --check`: PASS.
- `py_compile` pilot/test: PASS.
- All six verdict-contributing test files are tracked by `git ls-files`.

## Gate boundary

Reviewer PASS and Test Manager GREEN remain required. Canonical docs remain pending until both independent gates pass. Real Phase A2/B2 require separate fresh explicit operator authorization after those gates.
