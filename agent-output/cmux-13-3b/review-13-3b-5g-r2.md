# Story 13.3b-5g r2 — independent diagnostic re-review

## Verdict

**FAIL / STOP — Reviewer PASS withheld.**

Prior RED largely repaired.
Corrected report still contract-invalid.
No classification, production redesign, real-asset access, smoke, or real training authorized.

## Blocking findings

### CRITICAL — non-finite input gradients sanitized into successful evidence

`tests/helpers/routed_fp4_multilayer_peak_probe.py:560-567` replaces the differentiated input gradient with `mx.nan_to_num(input_grad)` before:

- DOT export;
- evaluation;
- finiteness validation;
- `input_gradient_nonzero` evidence.

The contract requires the actual input gradient to be finite. Sanitization can convert NaN/Inf into finite values and then report `finite_gradients=true`.

Independent bounded reproduction of the checked-in Probe D `routed_only`, `D=43`, `compression_ratio=4` construction:

```text
raw_input_finite False
nonfinite_count 6144
raw_nonzero True
sanitized_input_finite True
sanitized_nonzero True
```

The regenerated report records this cell as successful with `finite_gradients=true` and non-zero input evidence. That evidence is false for the raw differentiated graph. Immediate STOP gate triggered.

Required repair: never normalize the measured gradient. Export, evaluate, validate, and record the raw input gradient. Capture any non-finite cell as failure and fully regenerate all 64 rows.

### HIGH — `routed_only` detaches the selected routed branch from input backward

`tests/helpers/routed_fp4_multilayer_peak_probe.py:289-290` calls:

```python
_routed_only(layer.mlp, mx.stop_gradient(mlp_input), input_ids)
```

The contract requires the routed branch to execute normally while only removed attention/shared branches use zero-valued identity-gradient replacements. The stop-gradient detaches the selected routed operation from the input path. The added `_zero_identity_like(mlp_input)` keeps a non-zero gradient alive, so the current non-zero evidence does not prove routed-branch backward participation.

Independent bounded D=1 comparison against the same forward without that stop-gradient:

```text
forward_max_abs 0.0
actual_grad_finite True
actual_nonzero True
correct_grad_finite True
correct_nonzero True
gradient_max_abs_diff 0.0020038634538650513
```

Identical forward, materially different input gradient. Probe D component semantics and routed-only telemetry remain invalid.

### HIGH — focused tests remain non-load-bearing for the defects above

`tests/test_deepseek_v4_nn_multilayer_peak.py:113-205` primarily validates the checked-in report. `:208-216` checks source substrings. `:219-229` tests direct normalization calls only.

No focused test executes and asserts:

- raw, unsanitized model-cell input-gradient finiteness;
- routed-only backward parity with the direct current routed branch;
- runtime LoRA conversion/key/shape behavior;
- masked full/routed parity and gradient semantics;
- DOT measurement boundary from a live model cell;
- parent handling of malformed child output and child attempts to override observed process outcomes.

Focused suite passes despite both semantic blockers:

```text
8 passed, 1 warning in 2.05s
```

Acceptance criterion 13 and the Architect-required behavioral gates remain unmet.

### MEDIUM — Probe A lacks required fresh-child execution evidence

`tests/helpers/routed_fp4_multilayer_peak_probe.py:865-871` launches Probe A in a subprocess, but records no child PID, nonce, or monotonic start/finish interval. `agent-output/cmux-13-3b/multilayer-peak-report.json:1243-1264` contains parity fields plus a self-asserted `monkeypatch_subprocess_only=true`, not the separate fresh-child execution evidence required by the r2 contract.

### MEDIUM — current r2 bytes differ from staged bytes

All four verdict assets pass `git ls-files`, but each has status `AM`. Worktree and index SHA-256 differ for helper, test, fixture, and report. Consequently `git diff --cached --check` validates older staged content, not the exact r2 worktree bytes reviewed here. Tracking itself passes; r2 chain-of-custody remains incomplete.

### MEDIUM — direct protected-hash proof remains incomplete

The eight report hashes match current worktree bytes. The protected list omits at least:

- `ds4_cli.c`;
- `ds4_server.c`;
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py`;
- root `metal/*.metal` coverage.

No protected production-like path appears in current staged or unstaged `git diff --name-only`, but this checkout explicitly forbids relying on a vacuous diff as the load-bearing integrity proof.

## Prior RED closure audit

Confirmed repaired:

- Probe B/D quantize → freeze → one rank-8, scale-20, dropout-0 LoRA conversion;
- exact last-`min(D,16)` q_a/q_b/kv key set and `6L` leaf count;
- single-bound differentiated forwards;
- DOT-derived Probe C node counts;
- measured `active_after_graph` boundary and sequential aggregation;
- exact 64 rows and order;
- E=2 low-volume control separated from the 384-assignment E>=8 axis;
- fixture-derived Probe D configuration, exact 43 layers, fixture/source hashes, malformed-fixture rejection;
- exact Probe A key/shape/parity fields;
- 64 fresh measurement children, unique PID/nonce, serial non-overlap, identity alignment;
- normalized serialized outcomes for the checked-in successful rows.

Not closed:

- Probe D selected-component input-gradient semantics;
- actual raw input-gradient finiteness;
- load-bearing behavioral tests;
- separate Probe A fresh-child evidence;
- complete r2/protected-byte chain-of-custody.

## Independent reproductions

```text
Probe B D=1 checkpoint-on:
  exit=0, trainable_keys=6, custom_nodes=12, input_nonzero=true

Probe C D=43/E=256 one-graph:
  exit=0, measured_nodes=66048, expected 6*43*256=66048

Probe C D=43/E=256 sequential:
  exit=0, measured_nodes=66048, expected 6*43*256=66048

Probe D routed_only D=43/cr4:
  exit=0, trainable_keys=96, custom_nodes=172
  reported input_nonzero=true
  raw input gradient contains 6144 non-finite values
```

Bounded D=2 topology check:

```text
12 observed keys
12 expected keys
exact key equality=true
all LoRA ranks=8
unexpected non-LoRA trainables=0
```

Report audit:

```text
rows=64
evidence=64
unique child PIDs=64
unique nonces=64
serial non-overlap=true
Probe C node mismatches=0
model key-count mismatches=0
active_after_graph copied from baseline rows=0
Probe A child execution evidence=false
```

Integrity checks:

```text
git diff --check: PASS
git diff --cached --check: PASS
helper/test/fixture/report tracked: PASS
protected modified paths reported by git diff: none
listed report protected hashes match worktree: PASS
```

Full tracked baseline not rerun: unrestricted repository suite may access configured real assets, prohibited by this review task. Coder's r2 notes claim `629 passed, 18 skipped, 96 subtests`; no r2 baseline log accompanies those notes.

## Required re-review gate

1. Remove input-gradient sanitization and capture any raw non-finite result as failure.
2. Remove the routed-only selected-branch stop-gradient; prove backward parity for the current direct routed branch.
3. Add runtime behavioral tests for topology, masks, raw gradients, DOT boundaries, and subprocess outcome override paths.
4. Add separate Probe A PID/nonce/interval evidence.
5. Complete protected-hash and exact-r2-byte evidence.
6. Regenerate the full 64-row report from scratch.
7. Rerun focused and tracked baseline checks, then return to independent Reviewer and Test Manager.

Reviewer marker intentionally not written. PASS remains withheld.
