# Story 13.3b-5g r3 — Coder notes

## Status

GREEN.

Raw-gradient semantic repair remains intact and the follow-up compile-state contamination is fixed. The diagnostic focused suite no longer disables MLX compile state in the pytest parent process; runtime MLX behavioral checks execute in subprocess snippets, and the compile-behavior baseline expectation is unchanged.

No commit.
No real assets.
No production/vendor/primitive/Metal/inference/SSD/CUDA/ROCm/distributed edits.

## Files changed

- `tests/helpers/routed_fp4_multilayer_peak_probe.py`
- `tests/test_deepseek_v4_nn_multilayer_peak.py`
- `tests/fixtures/deepseek_v4_nn_multilayer_peak_topology.json`
- `agent-output/cmux-13-3b/multilayer-peak-report.json`
- `agent-output/cmux-13-3b/coder-13-3b-5g-r3-notes.md`
- staged verification logs under `agent-output/cmux-13-3b/test-13-3b-5g-r3-isolation-*.log`

## Raw-gradient repair retained

- No input-gradient `mx.nan_to_num` before DOT export, evaluation, finiteness validation, or evidence.
- Raw input gradients drive DOT export and `mx.eval` directly.
- Any raw non-finite LoRA or input gradient raises `FloatingPointError` and records the cell as failed.
- `routed_only` no longer stops gradient on selected `mlp_input`.
- Removed attention/shared branches remain zero-valued with residual-stream identity-gradient replacement.
- Probe A records separate child PID, nonce, monotonic interval, and parent-observed exit/signal/error.
- Protected hashes include `ds4_cli.c`, `ds4_server.c`, `deepseek_v4_nn.py`, root `metal/*.metal`, and prior protected files.

## Compile-state isolation repair

- Runtime diagnostic tests no longer call `h._setup_mlx()` / `mx.disable_compile()` in the pytest parent.
- MLX runtime checks execute through `_run_helper_snippet(...)`, a fresh Python subprocess per check.
- Added `test_multilayer_peak_compile_state_isolated_from_diagnostic_subprocesses`, which proves the compile-baseline expectation before and after a diagnostic subprocess cell.
- Baseline expectation in `tests/test_deepseek_v4_nn_sparse_routed_backward.py::test_compiled_host_routing_fails_as_documented` was not changed.

## Regenerated report audit

Report was regenerated from scratch during r3 raw-gradient repair with:

```bash
python-envs/mlx/.venv/bin/python tests/helpers/routed_fp4_multilayer_peak_probe.py run --output agent-output/cmux-13-3b/multilayer-peak-report.json
```

Audit result:

```text
rows=64
errors_or_nonfinite=0
evidence=64
unique_pids=64
unique_nonces=64
serial_non_overlap=true
input_nonzero_failures=0
protected_hash_count=30
root_metal_count=19
```

No report regeneration was required by the isolation follow-up because only pytest-parent isolation changed; helper semantics/report bytes from the raw-gradient repair remain staged.

## Verification

Focused diagnostic suite:

```text
python-envs/mlx/.venv/bin/python -m pytest tests/test_deepseek_v4_nn_multilayer_peak.py -q
14 passed, 1 warning in 18.45s
```

Explicit order checks:

```text
compile test alone: 1 passed
focused diagnostic suite then compile test: 15 passed
compile test then focused diagnostic suite: 15 passed
```

Tracked Python baseline:

```text
tracked_test_files=45
490 passed, 15 skipped, 2 warnings, 86 subtests passed in 74.85s
```

Build:

```text
make
make: Nothing to be done for `all'.
```

Final chain checks after staging:

```text
index==worktree hashes: PASS for updated/staged r3 assets and logs
git diff --check: PASS
git diff --cached --check: PASS
protected production diffs: empty
```

## r4 mask-test repair

Reviewer r3 FAIL was addressed with test-only changes in `tests/test_deepseek_v4_nn_multilayer_peak.py`.

Added independent, mutation-sensitive masked-component runtime oracles:

- Direct `x - mx.stop_gradient(x)` zero expression: exact zero forward, exact shape/dtype, exact all-ones gradient.
- `_zero_residual_identity_like` checked against an independently written zero-mean residual expression and gradient, not against `_masked_layer_forward` composition.
- Ordinary `model(ids)` forward equals masked `full` forward exactly.
- All five masks (`full`, `no_routed`, `no_shared`, `no_attention`, `routed_only`) prove exact output shape/dtype, finite forward, finite input gradients, nonzero input gradients, and unchanged LoRA trainable key set.
- `_routed_only` checked against independent `layer.mlp(x, input_ids=ids) - layer.mlp.shared_experts(x)` forward and input-gradient oracle.
- In-memory mutation proof replaces `_routed_only` and `_zero_residual_identity_like` with `zeros_like`; the new oracle detects identity-gradient loss and routed forward/gradient semantic failure.

No helper, report, fixture, production, vendor, primitive, Metal, inference, SSD, CUDA, ROCm, or distributed semantics changed in r4.

## r4 verification

Focused diagnostic suite:

```text
/Volumes/Data NVME/mlx-ft/ds4/.venv/bin/pytest tests/test_deepseek_v4_nn_multilayer_peak.py -q
14 passed, 1 warning in 7.82s
```

Compile-order checks:

```text
focused suite then compiled-host-routing baseline: 15 passed, 1 warning in 7.89s
compiled-host-routing baseline then focused suite: 15 passed, 1 warning in 7.79s
```

Tracked Python baseline:

```text
tracked_test_files=45
PYTHONPATH=.
466 passed, 38 skipped, 2 warnings, 79 subtests passed in 49.32s
```

Build and hygiene:

```text
make: Nothing to be done for `all'.
git diff --check: PASS
```

Staged r4 evidence logs:

- `agent-output/cmux-13-3b/coder-13-3b-5g-r4-focused-and-compile-orders.log`
- `agent-output/cmux-13-3b/coder-13-3b-5g-r4-tracked-baseline.log`
- `agent-output/cmux-13-3b/coder-13-3b-5g-r4-make-diff-tracking.log`

## Completion

Full tracked baseline is GREEN. Marker written only after exact-byte staging and final checks.
