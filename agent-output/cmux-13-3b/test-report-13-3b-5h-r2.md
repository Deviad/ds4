# Story 13.3b-5h r2 — Tester independent validation

## Verdict
GREEN.

## Fresh execution
- focused 5h + 5g order A: rc=0, `22 passed, 1 warning in 26.18s`
- focused 5g + 5h order B: rc=0, `22 passed, 1 warning in 26.25s`
- canonical tracked baseline: rc=0, `498 passed, 15 skipped, 2 warnings, 86 subtests passed in 82.16s`
- `make`: rc=0, `Nothing to be done for 'all'.`
- `git diff --check`: rc=0
- `git diff --cached --check`: rc=0

## Report audit
- `agent-output/cmux-13-3b/interaction-ablation-report.json` validated: 25 rows, exact required order, 25 cell-evidence entries, 25 unique child PIDs, 25 unique child nonces, non-overlapping monotonic child intervals.
- `agent-output/cmux-13-3b/multilayer-peak-report.json` validated: 64 rows, exact required order, 64 cell-evidence entries, 64 unique child nonces, non-overlapping monotonic child intervals.
- metadata validated: `serial=true`, `max_concurrency=1`, `memory_limit=8000000000`, `timeout_s=180`, `real_assets_accessed=false`, no classification drift.
- required tracked files confirmed with `git ls-files`:
  - `tests/fixtures/deepseek_v4_nn_multilayer_peak_topology.json`
  - `tests/helpers/routed_fp4_multilayer_peak_probe.py`
  - `tests/test_deepseek_v4_nn_interaction_ablation.py`
  - `tests/test_deepseek_v4_nn_multilayer_peak.py`
- working-tree bytes match staged bytes for the report, helper, fixture, and both focused tests.

## Critical D43 controls
- `composed_checkpoint_on` D=43: delta `1,881,949,460 B`, peak-forward delta `39,013,308 B`, peak-backward delta `1,882,121,016 B`, nodes `516`
- `attention_forward_routed_vjp` D=43: delta `1,894,229,012 B`, nodes `516`
- `routed_forward_attention_vjp` D=43: delta `1,882,958,368 B`, nodes `172`
- `materialized_attention_routed_boundary` D=43: delta `1,881,949,460 B`, elementwise allclose `true`, nodes `516`
- `composed_layer_sequential` D=43: delta `41,973,728 B`, nodes `516`

## Conclusion
- no missing rows
- no subprocess failures
- no parity drift
- no tracking drift
- no protected-source drift
- no baseline regressions
