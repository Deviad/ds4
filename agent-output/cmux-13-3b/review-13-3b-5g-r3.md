# Story 13.3b-5g r3 — final independent review

## Verdict

**FAIL / STOP — Reviewer PASS withheld.**

No production/test/source edits, real assets, classification, redesign, or smoke. Reviewer marker intentionally absent.

## Blocking finding

### HIGH — required masked-component tests remain non-load-bearing

The binding contract requires a focused behavioral oracle for exact zero-valued identity-gradient behavior, full masked-path equality with ordinary model forward, every mask's shape/dtype and input-gradient semantics, direct routed parity against `full_mlp_output - shared_output`, and LoRA registration stability across every mask (`agent-output/cmux-13-3b/architecture-13-3b-5g-r2-contract.md:138-160`, `:409-427`).

The only r3 masked runtime test constructs its expected result with the same helper functions under test: `h._zero_residual_identity_like` and `h._routed_only` at `tests/test_deepseek_v4_nn_multilayer_peak.py:314-347`. It therefore checks that `_masked_layer_forward` repeats the same helper composition, not that either helper implements the required identity-gradient or current routed branch. No focused test compares `_routed_only` with `layer.mlp(...) - layer.mlp.shared_experts(...)`, compares masked `full` with ordinary `model(ids)`, exercises all five masks for shape/dtype and gradient semantics, proves exact zero forward plus identity derivative, or checks trainable keys before/after every mask.

Independent in-memory mutation proof reproduced the defect in the test oracle without editing files:

```text
Deliberately replace _routed_only with zeros_like and
_zero_residual_identity_like with zeros_like, then execute the exact current
actual/expected parity construction:
current_test_forward_diff = 0.0
current_test_gradient_diff = 0.0
required_direct_routed_diff = 0.007146671414375305
required_identity_grad_max = 0.0
```

Thus the current test still reports perfect forward/gradient parity while both required semantics are broken. The focused suite is GREEN but acceptance criterion 13 remains unmet. Required repair: add independent runtime oracles for all contract items above; they must fail under these in-memory semantic mutations.

## Independent gate audit

### 1. Raw input-gradient integrity — PASS

`tests/helpers/routed_fp4_multilayer_peak_probe.py:558-575` sends the raw `input_grad` to DOT export and `mx.eval`, tests `_finite_array(input_grad)`, and raises `FloatingPointError` on any non-finite value. No `nan_to_num` or other input-gradient replacement exists.

Critical prior-RED cell reproduced with `mx.nan_to_num` replaced by a throwing sentinel:

```text
Probe D routed_only, D=43, compression_ratio=4:
error=null
finite_gradients=true
input_gradient_nonzero=true
input_gradient_shape=[1,64,4,128]
```

### 2. Routed-only selected branch runtime semantics — implementation PASS, test gate FAIL

The selected branch receives live `mlp_input` at `tests/helpers/routed_fp4_multilayer_peak_probe.py:256-270,273-300`; only routed indices are stopped for host row materialization. Independent bounded D=1 checks:

```text
ordinary model vs masked full max_abs = 0.0
direct routed vs full MLP minus shared max_abs = 6.984919309616089e-09
zero identity forward max_abs = 0.0
zero identity gradient min/max = 1.0/1.0
all five masks: shape=[1,4,4,128], dtype=float32, finite=true
LoRA keys unchanged before/after masks=true
```

Current bytes are correct, but the load-bearing test requirement is the blocking finding above.

### 3. Runtime behavioral tests — FAIL

Raw-gradient, LoRA, DOT, compile-isolation, and process-normalization runtime checks execute in fresh subprocesses (`tests/test_deepseek_v4_nn_multilayer_peak.py:252-430`). Masked-component independent-oracle coverage required by the contract is absent/non-load-bearing.

### 4. Probe A fresh-child evidence — PASS

Checked-in report contains separate PID, nonce, monotonic interval, and parent-observed outcome. Fresh independent execution returned:

```text
parent_pid=35750
child_pid=payload_pid=35901
nonce_match=true
interval_valid=true
parent_observed_exit_code=0
signal=null
error=null
layer/pipeline/decoder/checkpointed_calls=[43,43,43,43]
key/shape/parity/finite checks=true
```

### 5. Compile-state isolation and order stability — PASS

Independent commands:

```text
pytest tests/test_deepseek_v4_nn_multilayer_peak.py -q
14 passed, 1 warning in 18.61s

pytest focused then test_compiled_host_routing_fails_as_documented
15 passed, 1 warning in 18.49s

pytest test_compiled_host_routing_fails_as_documented then focused
15 passed, 1 warning in 18.50s
```

The before/after child-isolation test is at `tests/test_deepseek_v4_nn_multilayer_peak.py:372-392`. Baseline compile expectation was unchanged.

### 6. Regenerated report, evidence, nodes, LoRA, and fixture — PASS

Independent complete JSON audit:

```text
rows=64; evidence=64; schema/order/outcome issues=0
unique child PIDs=64; unique nonces=64; parent PID reused=false
serial non-overlap=true; max_concurrency=1
memory_limit=8000000000; timeout_s=180
all rows exit_code=0, signal/error=null, finite loss/gradients=true
all model cells: 6*min(D,16) trainable leaves, exact input shape, non-zero input gradient
all Probe C rows: measured nodes=6*D*nonempty_experts
fixture: 43 ordered uniform MoE entries; exact source/fixture hashes and derivation metadata valid
real_assets_accessed=false; classification=not-authorized-by-diagnostic
```

Critical Probe C cells reproduced independently:

```text
D=43, E=256, one-graph: nodes=66048/66048, finite=true, error=null
D=43, E=256, sequential: nodes=66048/66048, finite=true, error=null
```

### 7. Exact bytes, staged logs, and protected hashes — PASS

Worktree and index bytes match for helper, focused test, fixture, report, r3 notes, and all four final `test-13-3b-5g-r3-isolation-*.log` files. All are tracked. Direct protected-hash audit found exactly 30 expected paths, including `ds4_cli.c`, `ds4_server.c`, `deepseek_v4_nn.py`, package routed/Metal files, and all 19 root `metal/*.metal`; every hash matches current bytes. Staged and unstaged protected diff-path counts are zero.

### 8. Focused, tracked baseline, make, diff, and tracking — PASS

Independent results:

```text
focused: 14 passed, 1 warning
tracked baseline: 45 tracked test files; 490 passed, 15 skipped,
                  2 warnings, 86 subtests passed in 78.23s
make: Nothing to be done for `all'.
git diff --check: PASS
git diff --cached --check: PASS
helper/test/fixture/report git ls-files: PASS
all 45 baseline test files git ls-files: PASS
```

## Required re-review gate

Add independent, mutation-sensitive runtime tests for the complete masked-component contract: zero identity forward/derivative, ordinary-vs-masked full equality, all five masks' shape/dtype/gradient behavior, direct routed-vs-full-minus-shared parity, and key-registration stability before/after every mask. Rerun focused and tracked baseline checks, then return to Reviewer and Test Manager.
