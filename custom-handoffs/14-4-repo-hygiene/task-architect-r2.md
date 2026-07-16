# Story 14.4 — Architect r2 fresh-clone test defect micro-revision

Read Reviewer r2 BLOCKED report at `custom-handoffs/standby/review.md`.

Design minimal closure for fresh synthetic clone failure:
`tests/test_ds4_segmented_loss_and_grad.py::test_active_memory_matrix` writes `agent-output/cmux-14-2/memory-r4.log` without ensuring parent exists. Local pass depended on stale untracked directory.

Specify exact minimal fix (test creates output parent, or cleaner temp/output strategy), TDD evidence using untouched synthetic staged clone, and required protected-hash cascade into `tests/test_ds4_segmented_smoke.py::TestProtected.test_provider_test`. Preserve Path A 365/digest and production provider bytes. Also specify final handoff force-add and manifest regeneration ordering so final gate reports are not omitted. No production edit, no commit/push. Write architecture-r2.md and marker.