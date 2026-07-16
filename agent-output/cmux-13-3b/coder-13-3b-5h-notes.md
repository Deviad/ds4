# Story 13.3b-5h — Coder notes

Status: GREEN.

## Scope

Implemented diagnostic-only 25-row synthetic interaction ablation. No production, vendor, primitive, Metal, real asset, smoke, training, classification, or commit performed.

## Changed / added

- Extended `tests/helpers/routed_fp4_multilayer_peak_probe.py` with 13.3b-5h interaction-plan, interaction-child, and interaction-run paths while preserving existing 13.3b-5g plan/report semantics.
- Added `tests/test_deepseek_v4_nn_interaction_ablation.py`.
- Generated `agent-output/cmux-13-3b/interaction-ablation-report.json`.
- Added validation logs:
  - `agent-output/cmux-13-3b/coder-13-3b-5h-focused-and-compile-orders.log`
  - `agent-output/cmux-13-3b/coder-13-3b-5h-tracked-baseline.log`
  - `agent-output/cmux-13-3b/coder-13-3b-5h-make.log`
  - `agent-output/cmux-13-3b/coder-13-3b-5h-diff-tracking-hashes.log`

## TDD

RED observed before implementation:

```text
tests/test_deepseek_v4_nn_interaction_ablation.py::test_interaction_ablation_plan_exact_25_rows_order_thresholds_and_identity
failed: helper CLI did not support interaction-plan
```

GREEN after implementation.

## Report

`agent-output/cmux-13-3b/interaction-ablation-report.json` contains exactly 25 rows in the required order:

1. composed checkpoint-on D=1,2,4,8,16,43.
2. composed checkpoint-off D=1,4,8.
3. attention-forward+routed-VJP D=1,8,16,43.
4. routed-forward+attention-VJP D=1,8,16,43.
5. same-semantics materialized boundary D=1,8,16,43.
6. per-layer sequential D=1,8,16,43.

All rows use cr4/no_shared/T64/H128/I64/E8/K2/hc4/rank8 last min(D,16), fixture-derived topology, raw gradients, 8 GB memory limit, 180 second child timeout, fresh subprocess per row, serial execution, unique PID/nonce, and non-overlapping monotonic intervals.

Report metadata records classification thresholds only:

```json
{"collapsed_max_delta": 227479736, "persistent_min_delta": 941632822}
```

No automatic classification is made (`classification: not-classified-by-coder`).

Parent-observed report summary:

```text
rows 25
hash_match True
failures []
```

## Validation

Commands run in canonical MLX venv with `PYTHONPATH=python-envs/mlx/src`:

```text
python-envs/mlx/.venv/bin/python -m pytest -q tests/test_deepseek_v4_nn_interaction_ablation.py
5 passed, 1 warning

python-envs/mlx/.venv/bin/python -m pytest -q tests/test_deepseek_v4_nn_multilayer_peak.py
14 passed, 1 warning

python-envs/mlx/.venv/bin/python -m pytest -q tests/test_deepseek_v4_nn_interaction_ablation.py tests/test_deepseek_v4_nn_multilayer_peak.py
19 passed, 1 warning

python-envs/mlx/.venv/bin/python -m pytest -q tests/test_deepseek_v4_nn_multilayer_peak.py tests/test_deepseek_v4_nn_interaction_ablation.py
19 passed, 1 warning

python-envs/mlx/.venv/bin/python -m pytest -q $(git ls-files -- tests/test_deepseek_v4_nn_interaction_ablation.py tests/test_deepseek_v4_nn_multilayer_peak.py)
19 passed, 1 warning

make
Nothing to be done for `all`.

git diff --check
git diff --cached --check
clean
```

Full tracked baseline closure requested by `task-coder-13-3b-5h-full-baseline.md`:

```text
tracked_test_file_count=46
python-envs/mlx/.venv/bin/python -m pytest -q $(git ls-files 'tests/test_*.py')
495 passed, 15 skipped, 2 warnings, 86 subtests passed in 84.53s (0:01:24)
pytest_exit_code=0
git_diff_check_exit_code=0
git_diff_cached_check_exit_code=0
all_contributing_test_files_tracked=yes
```

Complete output recorded and staged in `agent-output/cmux-13-3b/coder-13-3b-5h-full-tracked-baseline.log`.

## Tracking and hashes

Tracked verdict/evidence files verified with `git ls-files`:

- `agent-output/cmux-13-3b/interaction-ablation-report.json`
- `agent-output/cmux-13-3b/multilayer-peak-report.json`
- `tests/fixtures/deepseek_v4_nn_multilayer_peak_topology.json`
- `tests/helpers/routed_fp4_multilayer_peak_probe.py`
- `tests/test_deepseek_v4_nn_interaction_ablation.py`
- `tests/test_deepseek_v4_nn_multilayer_peak.py`

Direct protected-source hashes recorded in `agent-output/cmux-13-3b/coder-13-3b-5h-diff-tracking-hashes.log` for production/vendor/Metal guard files.

## Staging

Staged for supervisor/reviewer/tester, no commit:

- helper extension
- new focused test
- 25-row report
- 5h validation logs
- this notes file

## STOP gates

No STOP gate triggered. No non-finite raw loss/gradient, missing peak, parity failure, row omission/reorder, child timeout/OOM/signal/nonzero, concurrency, real asset access, or protected-source edit observed.

## R2 fixes for Reviewer FAIL

Status: GREEN.

### Scope

Diagnostic/test/helper/report/log repair only. No production, vendor, primitive, Metal, real asset, classification, smoke, training, commit, or redesign action performed.

### TDD RED

Added mutation-sensitive checks before helper repair. RED recorded in `agent-output/cmux-13-3b/coder-13-3b-5h-r2-red.log`:

```text
FAILED tests/test_deepseek_v4_nn_interaction_ablation.py::test_interaction_ablation_active_baseline_is_pinned_immediately_before_vjp_graph
FAILED tests/test_deepseek_v4_nn_interaction_ablation.py::test_interaction_ablation_materialized_parity_uses_elementwise_allclose_counterexample
2 failed, 6 passed, 1 warning
```

### Repairs

- Moved 5h `active_baseline` capture to after diagnostic forward/comparison graph evaluation and release, immediately before differentiated VJP graph construction. No separate forward evaluation remains between baseline capture and VJP construction.
- Updated sequential rows to capture per-stage immediate baselines and report the baseline matching the stage that produced the aggregated `max(per_stage_backward_peak)`.
- Replaced materialized parity's global-max tolerance predicate with true elementwise `atol=rtol=1e-5` checks for scalar loss, every LoRA gradient leaf, and input gradient while preserving exact key and shape checks.
- Added counterexample coverage where the old global-max predicate passed but elementwise allclose fails: `a=[1.5e-5, 1.0]`, `b=[0.0, 1.0]`.
- Added independent VJP-leg gradient partition tests:
  - `attention_forward_routed_vjp`: routed VJP remains (`nodes=12`, nonzero input gradient) and attention LoRA-gradient partition is stopped (all LoRA leaf gradients zero).
  - `routed_forward_attention_vjp`: routed-output VJP is stopped (`nodes=4`) while attention gradients remain (nonzero LoRA leaf gradients and nonzero input gradient).

### Fresh 25-row regeneration

Regenerated from scratch, no patch/reuse:

```text
python-envs/mlx/.venv/bin/python tests/helpers/routed_fp4_multilayer_peak_probe.py interaction-run --output agent-output/cmux-13-3b/interaction-ablation-report.json
{"output": "agent-output/cmux-13-3b/interaction-ablation-report.json", "rows": 25}
```

Fresh report evidence:

```text
rows=25
created_at=2026-07-14T17:50:23Z
executed_cell_list_sha256=a5c3474c04ab8b0edb3aa03af806cf0830ebe73c76c9cb6077d42559cfa6a356
all rows exit_code=0, error=None, finite_loss=True, finite_gradients=True
classification=not-classified-by-coder
protected_hashes_match_report=True
```

### Validation

```text
python-envs/mlx/.venv/bin/python -m pytest -q tests/test_deepseek_v4_nn_interaction_ablation.py
8 passed, 1 warning

python-envs/mlx/.venv/bin/python -m pytest -q tests/test_deepseek_v4_nn_multilayer_peak.py
14 passed, 1 warning

python-envs/mlx/.venv/bin/python -m pytest -q tests/test_deepseek_v4_nn_interaction_ablation.py tests/test_deepseek_v4_nn_multilayer_peak.py
22 passed, 1 warning

python-envs/mlx/.venv/bin/python -m pytest -q tests/test_deepseek_v4_nn_multilayer_peak.py tests/test_deepseek_v4_nn_interaction_ablation.py
22 passed, 1 warning

tracked_test_file_count=46
python-envs/mlx/.venv/bin/python -m pytest -q $(git ls-files 'tests/test_*.py')
498 passed, 15 skipped, 2 warnings, 86 subtests passed in 78.07s

make
Nothing to be done for `all`.

git diff --check
git diff --cached --check
clean
```

### R2 artifacts

- `agent-output/cmux-13-3b/coder-13-3b-5h-r2-red.log`
- `agent-output/cmux-13-3b/coder-13-3b-5h-r2-focused.log`
- `agent-output/cmux-13-3b/coder-13-3b-5h-r2-regenerate-25.log`
- `agent-output/cmux-13-3b/coder-13-3b-5h-r2-focused-and-orders.log`
- `agent-output/cmux-13-3b/coder-13-3b-5h-r2-full-tracked-baseline.log`
- `agent-output/cmux-13-3b/coder-13-3b-5h-r2-make.log`
- `agent-output/cmux-13-3b/coder-13-3b-5h-r2-diff-tracking-hashes-prestage.log`

### STOP gates

No STOP gate triggered. No parity failure, nonfinite loss/gradient, missing peak/cell, report row omission/reorder, real asset access, smoke/classification/training, protected-source edit, production/vendor/primitive/Metal edit, or commit performed.
