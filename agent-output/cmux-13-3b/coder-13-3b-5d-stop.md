# Story 13.3b-5d review-fix STOP — memory peak still scales with E

## Result
STOP-ESCALATE. Review-fix implementation removed `_y_cache` / `_f_cache` all-expert VJP retention and strengthened the tracked sparse tests, but the required fixed-K all-experts-nonempty memory proof still shows operation peak growth with expert count. No full model, shard load, or real smoke was run. No commit.

## RED evidence before production correction
Command:

```bash
source '/Volumes/Data NVME/mlx-ft/ds4/.venv/bin/activate'
python -m pytest tests/test_deepseek_v4_nn_sparse_routed_backward.py::test_indices_are_stop_gradient_before_host_materialization -q --tb=short
```

Result: `FAILED` because `SparseMoeBlockNN.__call__` still contained `_y_cache` and `_f_cache` in the custom VJP.

## Implementation attempted
File changed: `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py`.

Changes:
- Removed `_y_cache` and `_f_cache`.
- Q pass now recomputes one expert at a time, evaluates `_q_e` / `_Q`, stops internal graph, clears cache, and releases locals before the next expert.
- Gradient pass now recomputes only the current expert, derives score cotangent and direct input VJP, evaluates/stops `_dscore_e`, `_dx_e`, `_dx`, `_dscores`, clears cache, and releases locals before the next expert.
- Forward routed accumulation evaluates/stops each per-expert contribution and accumulator before the next expert.

Tracked tests strengthened in `tests/test_deepseek_v4_nn_sparse_routed_backward.py`:
- nonuniform token fixture;
- distinct packed FP4 bytes and scales per expert;
- sparse gate gradient compared with dense ordinary autodiff and finite differences, including dense non-zero assertion;
- clamp-near sparse-vs-dense input and gate gradient parity at `atol=rtol=1e-5`;
- non-zero shared expert fixture that detects omission/double-add;
- AST/source contract that `_host_unique_rows_per_expert` receives `mx.stop_gradient(...)` and no cache symbols remain;
- duplicate route slot collapse compared with single-slot output/cotangent;
- dense and sparse memory cases in separate fresh child processes;
- real-dimension no-shard probe uses two non-empty experts.

## GREEN tests that passed
Sparse suite:

```text
21 passed, 1 warning in 5.80s
```

Focused FP4/MoE/remap/LoRA suite:

```text
117 passed, 8 skipped, 1 warning, 16 subtests passed in 26.90s
```

Full tracked non-live regression excluding real-shard/live smoke files:

```text
364 passed, 29 skipped, 1 warning, 77 subtests passed in 25.28s
```

## Required fixed-K all-experts-nonempty memory proof — FAIL/STOP
Fresh separate Python processes. No model shards. Fixed `K=2`, `T=1024`, `H=1024`, `I=512`. Hash routing makes every expert non-empty.

```text
E=2 active_delta=60,325,900  peak_delta=584,225,254  finite_g=True
E=4 active_delta=60,358,668  peak_delta=779,041,700  finite_g=True
E=8 active_delta=60,424,204  peak_delta=1,179,289,384 finite_g=True
```

Active retained delta is flat, but operation peak still grows materially with E. Per task STOP condition, this remains unresolved memory-peak evidence and needs Architect/Reviewer adjudication or a different VJP boundary.

Earlier forward+VJP probe including output evaluation also showed flat active delta but growing peak:

```text
E=2 active_delta=14,688,256 peak_delta=601,022,962 finite_y=True finite_g=True nonempty=2 min_rows=1024 max_rows=1024
E=4 active_delta=14,688,264 peak_delta=791,659,440 finite_y=True finite_g=True nonempty=4 min_rows=512  max_rows=512
E=8 active_delta=14,688,280 peak_delta=1,188,759,340 finite_y=True finite_g=True nonempty=8 min_rows=256 max_rows=256
```

## Chain-of-custody checks
- FROZEN SHA: `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b`.
- Superseded pre-13.3b-5b hash literal: zero tracked `git grep` hits after removing stale historical-note literals.
- Verdict test tracking: all cited focused/regression test files are tracked by `git ls-files`.

## STOP status
No `.cmux-status/coder.done` marker created. Existing stale marker remains deleted. No full model/shard run. No commit.
