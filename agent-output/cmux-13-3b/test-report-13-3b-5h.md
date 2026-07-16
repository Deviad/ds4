# Story 13.3b-5h — Tester independent validation

## Verdict
GREEN.

## Fresh execution
- focused 5h + 5g order A: rc=0, `19 passed, 1 warning in 23.44s`
- focused 5g + 5h order B: rc=0, `19 passed, 1 warning in 23.11s`
- canonical tracked baseline: rc=0, `495 passed, 15 skipped, 2 warnings, 86 subtests passed in 79.98s`
- `make`: rc=0, `Nothing to be done for 'all'.`
- `git diff --check`: rc=0
- `git diff --cached --check`: rc=0

## Report audit
- `agent-output/cmux-13-3b/interaction-ablation-report.json` validated: 25 rows, exact required order, 25 cell-evidence entries, 25 unique child PIDs, 25 unique child nonces, and non-overlapping monotonic child intervals.
- report metadata validated: `serial=true`, `max_concurrency=1`, `memory_limit=8000000000`, `timeout_s=180`, `real_assets_accessed=false`, `classification=not-classified-by-coder`.
- required tracked files confirmed with `git ls-files`:
  - `tests/fixtures/deepseek_v4_nn_multilayer_peak_topology.json`
  - `tests/helpers/routed_fp4_multilayer_peak_probe.py`
  - `tests/test_deepseek_v4_nn_interaction_ablation.py`
  - `tests/test_deepseek_v4_nn_multilayer_peak.py`
- protected hashes and fixture hash verified against current bytes.

## Critical D43 controls
- `composed_checkpoint_on` D=43: delta `1,883,265,644 B`
- `attention_forward_routed_vjp` D=43: `forward_exact=true`, delta `1,894,521,052 B`
- `routed_forward_attention_vjp` D=43: `forward_exact=true`, delta `1,883,290,164 B`
- `materialized_attention_routed_boundary` D=43: `materialized_allclose_atol_rtol_1e-5=true`, parity diffs `0.0`, delta `1,883,194,248 B`
- `composed_layer_sequential` D=43: delta `43,293,044 B`

## Conclusion
- no missing rows
- no subprocess failures
- no parity drift
- no tracking drift
- no protected-source drift
- no baseline regressions
