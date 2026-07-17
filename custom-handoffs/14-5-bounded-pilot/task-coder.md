# Story 14.5 — Coder synthetic implementation

Read requirements.md and architecture.md completely. Implement via TDD red→green; no real assets/training/inference, no commit/push.

Scope exactly:
- New tracked `scripts/ds4_segmented_pilot.py` dedicated Phase A/B entry point.
- Add exact non-default catalog commands in `scripts/finetune_ds4.py`.
- New tracked `tests/test_ds4_segmented_pilot.py` covering architecture T1-T18 with mutation-sensitive oracles.
- Update docs/architecture.md and docs/technical-spec.md bounded-pilot sections; update docs/backlog.md actual implementation state only.
- Preserve `scripts/ds4_segmented_smoke.py` byte-for-byte, Story 14.3 evidence, Path A 365/digest, provider source/test except any explicitly reviewed sentinel needed, vendor gitlink 80fab4e, default catalogs, CUDA/distributed/Metal paths.

Implement exact phase specs:
- A: iters 2, eval/save every 1 as architecture binds, timeout 2700, output `adapters-segmented-pilot-phase-a`.
- B: iters 1, timeout 1500, resume exact Phase A `0000002_adapters.safetensors`, output `adapters-segmented-pilot-phase-b`.
- total active budget 4200.
- same model/data/config/max-seq/batch/LR/mask/grad-checkpoint/segment-size.
- one attempt per phase; fail on any output/marker; no retry/fallback.
- exact provider/update/checkpoint cardinality; finite evidence; canonical tensor digest/schema; pre-update resume equality and post-update change; fresh optimizer and explicit non-claims.
- durable atomic reports/markers and lock/timeout/abort cleanup.
- exact asset/source/config/model manifest contract without accessing real paths synthetically.

All verdict tests must be tracked. Run focused red evidence, focused green, exact Epic 14 regression, broader relevant suite, py_compile, cached diff check, protected hashes, Path A, vendor pin, smoke byte identity, source import/fresh-clone checks. Write coder-notes.md, stage exact slice, marker.