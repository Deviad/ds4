# Story 14.5c — Coder r12 final two-oracle closure

## Scope

Closed standby review r11 F1/F2 with tracked tests and canonical documentation updates. No production-code changes were needed in r12; the staged pilot/catalog changes remain the prior r11 implementation. No real model/data/provider/training/inference/Phase A3/Phase B3/cleanup execution, commit, push, delegation, or cmux operation was performed.

## Changes

- `tests/test_ds4_segmented_pilot.py`
  - Strengthened every caller-level `run_phase()` publication seam: both Phase-A writes and all four Phase-B/final writes now compare the exact returned phase failure report with the persisted phase report, compare the final failure report including its `phase_report` binding, and validate both fail markers for existence, report path, report SHA-256, contract digest, namespace, phase, attempt `3`, exit code, and failure status. All three OK markers remain asserted absent.
  - Reworked the weakened attempt-2 verifier mutant to use one coordinated path/size/SHA-256/payload substitution. The weakened lower-level verifier accepts the substituted fixture while retaining the valid report snapshot; the public production immutable-target guard rejects the same substitution with exact `attempt-2 runtime target bindings are not immutable`, with no unrelated missing-report rejection.
- `docs/architecture.md`, `docs/backlog.md`, `docs/technical-spec.md`
  - Updated r11 staging claims to r12 evidence and recorded the exact two-oracle closure while preserving the synthetic-only/A3-B3-blocked boundary.

## Verification

Focused:

```text
3 passed, 622 deselected, 1 warning
```

Canonical vendor-first six-file suite:

```text
871 passed, 3 skipped, 1 warning, 2 subtests passed in 49.50s
```

Also passed:

```text
python3 -m py_compile tests/test_ds4_segmented_pilot.py scripts/ds4_segmented_pilot.py scripts/finetune_ds4.py
git diff --cached --check
git ls-files --error-unmatch tests/test_ds4_segmented_pilot.py
```

The modified test and documentation files are staged. No staged success/failure markers were added. Reviewer PASS and same-tree Test Manager GREEN remain required before any fresh phase authorization.
