# Story 14.5 — Architect bounded two-phase pilot

Read requirements.md, AGENTS.md, docs architecture/technical spec/backlog, Story 14.3 smoke script/report/log, `scripts/ds4_segmented_smoke.py`, `scripts/finetune_ds4.py`, vendor trainer at 80fab4e, and MLX-LM adapter save/load behavior.

Produce architecture.md determining whether current paths can prove:
- Phase A: 2 optimization updates, save every step, 2700s hard timeout;
- Phase B: load exact Phase A step-2 adapter, 1 resumed update, 1500s hard timeout;
- total 4200s, one attempt each, no retry/fallback;
- exact provider call counts 2 then 1, finite loss/gradients, step progression, checkpoint hashes and adapter-weight continuity;
- no optimizer/RNG/data-cursor continuity claim unless implementation truly preserves them.

Specify exact CLI/config/output/checkpoint semantics, locks/markers/reports, abort behavior, cleanup ownership, and visible execution. Identify minimal implementation/test changes required before authorization, including fail-closed preflight proving Phase B loaded the intended checkpoint rather than restarting. Preserve existing smoke command and defaults, Path A, vendor pin, CUDA/distributed/Metal default paths. Update durable architecture/technical spec or ADR only if required. No real assets/training/inference/commit/push. Write architecture.md and marker.