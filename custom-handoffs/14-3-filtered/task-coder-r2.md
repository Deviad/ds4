# Story 14.3a — Coder r2 reviewer blocker closure

Read custom-handoffs/14-3-filtered/review.md (or custom-handoffs/standby/review.md if canonical handoff is stale), requirements.md, architecture.md, coder-notes.md.

Close exactly three blockers:

1. Canonical `custom-handoffs/14-3/requirements.md` must say only the filtered dataset `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke` is authorized for this smoke; original `mlx-4096` is immutable and rejected. Remove contradictory old sentence without changing 4096/safety contract.
2. Add/stage `agent-output/cmux-14-3/filtered-dataset-provenance.json` (already created by supervisor) and reference it consistently in filtered requirements, architecture, docs/backlog. Preserve exact counts: input/kept/excluded train 15170/15108/62, valid 819/816/3, test 824/823/1; total excluded 66; rule and source/filtered hashes. Do not alter real datasets.
3. Strengthen tracked `test_default_catalog_entries_unchanged` to prove exact byte identity of `smoke-train`, `full-train`, and `continue-train` command strings against an approved immutable baseline (not merely absence of `-smoke`). Keep segmented entry explicitly filtered and all defaults unchanged.

Stage touched files; register new functional hash/revision `14-3a-coder-r2` after edits. Run exact-fork focused suites, py_compile, Path A/pins/protected checks. No real smoke/training/inference/assets beyond using the already-created provenance artifact; no commit/push. Write `custom-handoffs/14-3-filtered/coder-notes-r2.md` and marker.