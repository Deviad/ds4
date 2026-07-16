# Story 13.3b-5h — independent review

**Verdict: FAIL. Return to Coder.**

No production, vendor, primitive, Metal, real-asset, smoke, training, classification, redesign, or commit action performed.

## Blocking findings

### HIGH — `active_baseline` violates required measurement boundary

`tests/helpers/routed_fp4_multilayer_peak_probe.py:756` records `active_baseline`, then constructs and evaluates a separate forward graph at lines 761–764 before constructing the differentiated graph at line 784. The same ordering exists for materialized rows at lines 805, 817–819, and 823, and sequential rows record one baseline at line 860 before each stage's separate forward and differentiated graphs.

This is not “active memory immediately before graph construction with no pending graph.” It makes every reported baseline-subtracted backward delta contract-invalid.

Independent D=1 reproduction of the exact one-graph sequence measured:

```text
recorded_boundary_value=1,081,024 B
actual_immediate_pre_vjp=2,176,412 B
difference=1,095,388 B
active_after_graph=44,229,088 B
```

The report must be regenerated after capturing the baseline at the pinned boundary. Sequential telemetry must likewise define and measure the baseline corresponding to each pinned per-stage backward peak rather than reusing a pre-loop value across later stage graphs.

### HIGH — control tests do not prove either VJP leg was removed

`tests/test_deepseek_v4_nn_interaction_ablation.py:105-142` checks forward equality, finiteness, nonzero input gradient, trainable-key count, and positive node count. None distinguishes the required stopped VJP from the full composed VJP.

Two independent in-memory source mutations were run without changing project files:

1. Replaced the attention control's stopped boundary with `ffn_hidden = after_attention`.
2. Replaced the routed control's stopped routed output with `mlp_output = routed + ...`.

Every predicate used by the current runtime test still passed for both mutants:

```text
attention_forward_routed_vjp: error=None, finite=True, forward_exact=True,
input_gradient_nonzero=True, trainable_key_count=6, nodes=12

routed_forward_attention_vjp: error=None, finite=True, forward_exact=True,
input_gradient_nonzero=True, trainable_key_count=6, nodes=12
```

Current implementation behavior appears directionally correct: the attention-forward/routed-VJP control has zero attention-LoRA gradient values, while routed-forward/attention-VJP retains nonzero attention-LoRA gradients and D=1 differentiated custom-kernel nodes fall from 12 to 4. Required mutation-sensitive independent proof is absent. Add independent gradient-partition/VJP assertions that fail when either `stop_gradient` is removed; do not reuse helper summary booleans as the oracle.

### MEDIUM — materialized parity predicate is weaker than elementwise `atol=rtol=1e-5`

`_max_array_diff` and `_max_grad_diff` at `tests/helpers/routed_fp4_multilayer_peak_probe.py:292-311` compare global maximum error against a tolerance scaled by the global maximum reference magnitude. Elementwise allclose requires each element's error to be scaled by that element's reference magnitude.

Independent counterexample:

```text
a=[1.5e-5, 1.0]
b=[0.0,    1.0]
helper_allclose=True
elementwise_allclose=False
```

All four current materialized report rows record zero loss, LoRA-gradient, and input-gradient maximum differences, so no present drift was observed. Replace the predicate with per-element allclose for every gradient leaf and the input gradient, retaining exact key and shape checks, then add a mutation-sensitive counterexample test.

## Independently verified evidence

- Exact 25-row identities and order; plan/execution SHA-256 `a5c3474c04ab8b0edb3aa03af806cf0830ebe73c76c9cb6077d42559cfa6a356`.
- Fixed topology: cr4, no_shared, T64, H128, I64, E8, K2, hc4, rank-8 LoRA on last `min(D,16)` layers.
- Exact `8_000_000_000 B` memory limit, `180 s` timeout, serial fresh children, 25 unique PIDs/nonces, monotonic non-overlap, all parent-observed exits zero.
- Raw finite losses/gradients, required trainable-key counts/shapes, nonzero input gradients, measured positive DOT custom-kernel nodes.
- Both single-VJP controls preserve composed forward bytes exactly.
- Sequential implementation evaluates one layer at a time, carries evaluated output, deletes graph references, and clears cache; no training-equivalence or automatic-classification claim made.
- Classification metadata only: collapsed `<=227,479,736 B`, persistent `>=941,632,822 B`; `classification=not-classified-by-coder`.
- Fixture has 43 ordered MoE layers; fixture SHA-256 `c80c103d3f9b8a12b6e3de5fa9b62b3efd56dc321497911a34357cf8c90ad094`; recorded provenance hashes match current files.
- Protected production/vendor/Metal hashes match recorded evidence; no protected staged or unstaged changes found.
- Nine 5h verdict/evidence files tracked; staged bytes equal worktree bytes. Relevant staged diff: `192,140` bytes, SHA-256 `952c60ae2977986abcb5d0ea9942b40f19229733818539665558d2858cda0f9f`.
- `git diff --check`: clean.
- `git diff --cached --check`: clean.
- `make`: exit 0, nothing to rebuild.
- Focused compile orders independently rerun: `19 passed, 1 warning` in each order.
- Full canonical tracked baseline independently rerun across 46 files: `495 passed, 15 skipped, 2 warnings, 86 subtests passed`; exit 0.

## Required re-review evidence

1. Correct baseline placement and regenerated fresh 25-row report.
2. Mutation-sensitive independent VJP-leg tests that fail when either stop-gradient is removed.
3. Elementwise materialized loss/LoRA/input-gradient parity at `atol=rtol=1e-5`, with mutation-sensitive test.
4. Refreshed focused orders, full 46-file baseline, make/diff/tracking/protected-hash evidence, and exact staged-byte proof.

Reviewer PASS marker intentionally not written.
