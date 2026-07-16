# Story 13.3b-5g r4 — final independent re-review

## Verdict

**PASS.**

The sole r3 blocker is closed. Reviewer marker is authorized.

No production/test/source edits, real assets, classification, redesign, or smoke were performed by Reviewer.

## Prior blocker closure

The r4 focused test is load-bearing and independent of the helper composition it verifies:

- It proves `x - mx.stop_gradient(x)` has exact zero forward value, exact shape/dtype, and an exact all-ones input gradient.
- It compares `_zero_residual_identity_like` against an independently written residual identity expression and independently differentiated gradient.
- It compares ordinary `model(ids)` output with masked `full` output exactly.
- It exercises all five masks (`full`, `no_routed`, `no_shared`, `no_attention`, `routed_only`) and checks exact output shape/dtype, finite forward values, finite exact-shape input gradients, nonzero input gradients, and unchanged LoRA trainable keys.
- It derives the routed oracle directly as `layer.mlp(x, input_ids=ids) - layer.mlp.shared_experts(x)` and compares both forward output and independently differentiated input gradient with `_routed_only`.
- It deliberately mutates both `_routed_only` and `_zero_residual_identity_like` to `zeros_like`. The independent oracle detects identity-gradient loss plus routed forward/gradient divergence while the unmutated path passes.

The expected values do not call the helper under test or repeat `_masked_layer_forward` composition. This directly resolves the r3 mutation-survival finding.

## Independent verification

Using the canonical project environment `python-envs/mlx/.venv`:

```text
focused: 14 passed, 1 warning in 18.57s
focused then compiled-host-routing: 15 passed, 1 warning in 18.72s
compiled-host-routing then focused: 15 passed, 1 warning in 18.76s
tracked baseline: 45 tracked test files; 490 passed, 15 skipped,
                  2 warnings, 86 subtests passed in 73.89s
make: PASS
```

Both `git diff --check` and `git diff --cached --check` pass.

## Venv count adjudication

Coder's alternate-environment result is legitimate but is not the canonical release verdict:

```text
/Volumes/Data NVME/mlx-ft/ds4/.venv:
466 passed, 38 skipped, 2 warnings, 79 subtests passed
```

The same exact list of 45 tracked test files was supplied in both runs. Environment inspection and JUnit comparison establish the difference:

- Canonical environment: Python 3.13.5; Torch import available.
- Alternate environment: Python 3.14.6; Torch absent.
- 14 canonical passing Torch-reference tests become explicit skips in the alternate environment.
- Nine Torch-dependent modules become collection-level skip placeholders, replacing ten canonical passing test cases.
- Exact transition accounting is 14 `passed -> skipped`, 10 `passed -> absent`, and 9 module-level `absent -> skipped`; this yields 490/15 versus 466/38 without any changed test-file input list.

Therefore the count difference reflects optional dependency/environment collection behavior, not a hidden baseline omission. The required canonical environment independently produces the prior 490 passed / 15 skipped GREEN result.

## Tracking, staging, and hashes

- All 45 test files contributing to the canonical baseline are tracked.
- The r4 test and evidence files are tracked and their worktree/index hashes match.
- The 30 protected-source hashes match current bytes; no protected path appears in staged or unstaged diffs.
- No untracked test contributes to the claimed tracked-only baseline. Existing unrelated untracked tests were not supplied to either baseline run and are not used by this slice's verdict.
- All substantive staged evidence paths are worktree/index identical. The only staged/worktree mismatch observed before this review was `.cmux-status/coder.done`, a status marker, not test/source/evidence content.
- `tests/helpers/routed_fp4_multilayer_peak_probe.py`, the fixture, and the report remain worktree/index byte-identical; r4 semantic edits are confined to the focused test and evidence.

## Final gate

No blocking finding remains. Story 13.3b-5g r4 earns Reviewer PASS. This verdict authorizes only the next contract-defined review/classification gate; it does not authorize production redesign, real assets, smoke, or training.
