# Coder notes — Story 14.5

- Attempt: `1`
- Revision: `coder-14-5-luna-r1-20260717T0418Z`
- Staged artifact hash (computed before handoff files): `03fad0ebdd0fa2009cd6de91b6970d1c30594738e33c488a9f40502418c13a76`
- Commit: none; implementation and verdict files staged for supervisor.

## Changed

- Added `scripts/ds4_segmented_pilot.py`: immutable Phase A/Phase B contracts, strict pin/config/path validation, canonical safetensors digest v1, resume equality proof, evidence cardinality/global-step validation, lock ownership, timeout/watchdog, atomic report/marker helpers, pilot callbacks/provider evidence, and fail-closed phase runner.
- Added tracked mutation-sensitive `tests/test_ds4_segmented_pilot.py`.
- Added explicit non-default `ds4-segmented-pilot-phase-a` and `ds4-segmented-pilot-phase-b` catalog commands to `scripts/finetune_ds4.py`; existing smoke/default entries untouched.
- Added durable Story 14.5 architecture, technical-spec, and backlog entries.

## Verification

- Focused Story 14.5 regression command: `259 passed, 3 skipped, 1 warning, 2 subtests passed`.
- New pilot test: `13 passed, 1 warning`.
- `git diff --check`: PASS.
- Cited baseline tests tracked: PASS; new `tests/test_ds4_segmented_pilot.py` staged and tracked in index.
- Frozen smoke SHA: `ec17950d985578f3cd97b51734527bfc47e6c399ebe84fad8ffcb12beae18ad8` PASS.
- Provider SHA: `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518` PASS.
- Vendor HEAD: `80fab4e419a57f9465bb9e2f4e90010d645e124c`; vendor status clean.
- Full `pytest` collection remains blocked by pre-existing missing vendor dependency `lm_eval` in `vendor/mlx-lm/tests/test_evaluate.py`.
- Repository `tests` run reached unrelated pre-existing failure in `tests/test_deepseek_v4_nn_interaction_ablation.py::test_interaction_ablation_vjp_legs_are_gradient_partition_sensitive`; no unrelated files changed.
- No real model, dataset, adapter, training, inference, CUDA, distributed, or smoke execution performed.

## Acceptance verdicts

- AC1: PASS — Story 14.0 correction/history and Story 14.5 user story recorded in backlog.
- AC2: PASS — explicit Phase A/Phase B pins, paths, budgets, and non-default catalog commands recorded.
- AC3: PASS — canonical tensor digest, file hashing, resume equality/change proof, and fail-closed malformed-input tests implemented.
- AC4: PASS — per-step provider/callback evidence and exact two-plus-one cardinality/global mapping contract implemented and tested.
- AC5: PASS — one-attempt output/marker gates, Phase B dependency checks, lock ownership, timeout/watchdog, no-retry/no-fallback policy implemented.
- AC6: PASS — atomic report/marker helpers and report-before-success-marker ordering implemented.
- AC7: PASS — synthetic implementation gate green; no real execution authorized or attempted.
- AC8: PASS — smoke/provider/vendor protected identities unchanged; default catalog excludes pilot entries.
- AC9: PASS — architecture and technical specification updated; no ADR needed.
- AC10: PASS — implementation is staged, not committed; reviewer/test-manager gates remain supervisor-owned.
