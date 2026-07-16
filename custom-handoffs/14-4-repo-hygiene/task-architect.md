# Story 14.4 — Architect reproducibility/cleanup design

Read requirements.md, AGENTS.md, canonical architecture/technical spec/backlog, commit af08c25, current tracked/untracked inventory, and inner vendor/mlx-lm staged diff.

Produce architecture.md with an exact, safe execution plan:
- deterministic keep/commit manifest for all current implementation files needed by fresh clone;
- deterministic delete manifest/rules for stale/generated/ephemeral files;
- explicit quarantine/do-not-touch set for ambiguous user files unless requirements authorize deletion;
- vendor inner commit + outer gitlink/pin/reference update strategy, or STOP if contract conflict cannot be safely resolved;
- tracking of every test contributing to verdict;
- gitignore rules preventing recurrence without hiding required evidence;
- serial mutation order, rollback points, and final clean-worktree criterion;
- targeted/full tests and fresh-clone/submodule reproducibility checks;
- preserve Path A permanent STOP and successful Story 14.3 smoke evidence.

No file mutations except architecture.md and durable architecture/ADR docs if required. No deletion/commit/push. Marker when complete.