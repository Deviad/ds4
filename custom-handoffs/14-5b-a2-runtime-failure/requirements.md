# Story 14.5b A2 runtime failure closeout requirements

As a DS4 fine-tuning operator (WHO), I want the authorized Phase A2 runtime failure recorded exactly and closed without cleanup or retry (WHAT), so that its completed updates, fail-closed artifact verdict, consumed namespace, and Phase B2 block remain auditable (WHY).

## Observed result

- Revision: `e6d34fa03479316720430f35cc94d4606a45ef96` (`e6d34fa`).
- Authorized invocations: one; authorization consumed.
- Exit: `1`.
- Provider calls / optimizer updates: `2/2`.
- Checkpoints saved: `0000001_adapters.safetensors` and `0000002_adapters.safetensors`; `phase-a-start.safetensors` and final `adapters.safetensors` also exist.
- Iteration 1 loss / validation loss: `19.334` / `19.553`.
- Iteration 2 loss / validation loss: `17.648` / `19.841`.
- Wall time: `2444.2739184170496s` (`2444.27s`), within the `2700s` Phase A2 budget.
- Post-training validation failure: `safetensors __metadata__ must be an object`.
- Direct header result: all four safetensors artifacts have `__metadata__: null`.
- Failure evidence exists: `agent-output/cmux-14-5-attempt-2/phase-a-report.json`, `agent-output/cmux-14-5-attempt-2/pilot-report.json`, `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-a-fail`, and `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-fail`.
- Phase A2 and final OK markers: absent.
- Lock: acquired once, released once, absent after exit.
- Watchdog: cancelled.
- Retry: not performed.

## Disposition

The attempt-2 namespace contains failed-run artifacts and is consumed and immutable. No cleanup, deletion, rename, overwrite, reuse, fallback, dynamic suffix, or automatic retry is permitted. Phase B2 remains blocked and unauthorized.

Any future repair requires a fresh full requirements, architecture, TDD, independent Reviewer, and Test Manager cycle; an explicit fixed new namespace decision that does not clean or reuse attempt 2; and fresh explicit operator authorization for one visible non-retryable invocation. B2 remains separately authorized only after exact A-phase success is independently verified.

The observed losses, validation losses, checkpoints, and elapsed time are execution facts only. They do not establish convergence, model quality, optimizer/RNG/dataset-cursor/scheduler/global-step continuity, resume continuity, or full-training readiness.

## Acceptance criteria

1. `docs/backlog.md` records the exact revision, one consumed authorization, `2/2` calls/updates, saved checkpoints, exact losses/validation losses, bounded wall time, fail-closed metadata error, four null metadata headers, failure/OK-marker state, lock release, cancelled watchdog, and no retry.
2. `training-next-status.md` states the same operational result and current gate.
3. Attempt-2 namespace and artifacts remain immutable failed-run evidence; no cleanup or reuse occurs.
4. B2 remains blocked. Any future A-phase attempt requires fresh full review, a fixed new namespace, and fresh explicit authorization.
5. No convergence, quality, continuity, resume, or readiness claim is made.
6. No code edit, runtime artifact mutation, test/training/provider invocation, cleanup, commit, or push occurs in this closeout.
