# Story 14.5 — bounded multi-iteration training + resume pilot

User authorized proceeding after Story 14.4. Read AGENTS.md, Story 14.3/14.4 evidence, smoke report/log, current activation script/catalog/config, architecture/technical spec/backlog, and adapter artifacts. Update docs/backlog.md:
1. Correct stale Story 14.0 status consistently with completed downstream work and existing evidence; do not rewrite history.
2. Add Story 14.5 user story and testable acceptance criteria for smallest meaningful bounded real pilot demonstrating more than one optimization step plus checkpoint/resume continuity.

Requirements must explicitly pin:
- same model `/Volumes/Data NVME/mlx-ft/ds4/model-4bit`;
- filtered dataset `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke`;
- same segmented provider, max seq 4096, batch 1, LR 1e-5, mask-prompt, grad-checkpoint, segment-size 1;
- exact phase iteration counts, save/checkpoint cadence, resume source/output paths;
- hard per-phase and total wall-clock budgets derived from Story 14.3 result (1025.602482s for one iteration, initial validation 816.30s); propose conservative smallest bounded budget;
- one attempt per phase, no automatic retry/fallback, lock/abort behavior;
- measurable evidence: finite loss/gradients, exact provider calls, step progression, adapter/checkpoint existence and hashes, resume starts from prior state rather than restart;
- Reviewer PASS + Tester GREEN before real execution;
- no convergence/quality/throughput/full-readiness claim;
- no Path A reopening; no CUDA/distributed changes; no smoke rerun;
- visible cmux/panel execution and final report.

If resume semantics cannot be proven from current CLI/config without implementation, require a separate minimal implementation phase before authorization. Write requirements.md and marker. No training/inference/real-asset access/commit/push.