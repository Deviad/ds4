# Story 14.3b — BA closeout note (2026-07-16)

## Summary

Story 14.3 (activation wiring), 14.3a (filtered dataset repin), and 14.3b
(timeout repin 600s→1200s) are all COMPLETE. Reviewer PASS (r4), Test Manager
GREEN (r4), and bounded real smoke succeeded.

## Reviewer PASS (r4)

- Task prompt: `custom-handoffs/14-3-timeout/task-reviewer-r4.md`
- Registered coder slice: `14-3b-coder-r4`
- Functional hash: `8bb8dd5640f11e6082d1e58c33e63c10221c67ead14fec2b44e60efd8cd25a82`
- Verified: 1200s timeout sole production change; watchdog mutation oracle
  exact; Path A 365-file digest intact; vendor gitlink pinned; all protected
  source/test hashes intact; ADR 0029 staged.

## Test Manager GREEN (r4)

- Report: `custom-handoffs/14-3-timeout/test-report.md`
- Exact-fork pytest: 246 passed, 3 skipped, 1 warning, 2 subtests passed
- py_compile passed on tracked Python files
- Path A: 365 rows, digest intact
- Provenance excluded total: 66

## Bounded real smoke result

- Report: `agent-output/cmux-14-3/smoke-report.json` — status `ok`
- Loss: 19.33367156982422 (finite float32)
- Token count: 1102 (positive, matches default_loss mask sum)
- Gradient leaf count: 96
- Finite: true
- Provider call count: 1
- Wall clock: 1025.6024819999002 seconds (under 1200s timeout)
- Gradient paths: model.layers[27-42].self_attn.{q_a_proj,q_b_proj,kv_proj}.lora_{a,b}
- Gradient dtypes: all mlx.core.float32
- Smoke log: `agent-output/cmux-14-3/smoke-log.txt` — 100% complete, 1/1 iters
- Success marker: `.ds4-segmented-smoke-ok`
- Adapter checkpoint: `adapters.safetensors` + `adapter_config.json` created
  (evidence only, not promoted)

## Explicit non-claims

This smoke does NOT prove: convergence, loss quality, generalization, OOM
repair, command-buffer lifetime fix, throughput, speed, full-training
readiness, real-data correctness of segmented math (synthetic equivalence
proven in Story 14.2), or any result beyond one bounded 4096-token microbatch.

## Backlog updates applied

- Story 14.3 status → COMPLETE (activation wiring + bounded real smoke succeeded)
- Story 14.3a status → COMPLETE (filter applied, smoke reached trainer, 14.3b
  timeout repin resolved)
- Story 14.3b status → COMPLETE (Reviewer PASS + Test Manager GREEN + bounded
  real smoke succeeded)
- Closeout evidence block added before EOF Epic 14

## No commit/push

All changes remain staged in the local worktree. Supervisor/operator owns the
commit/push decision.