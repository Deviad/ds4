# Story 13.3b-5h r2 — independent re-review

**Verdict: PASS.**

No blocking findings. No production, vendor, primitive, Metal, real-asset, smoke, training, classification, redesign, or commit action performed.

## Prior blockers re-adjudicated

### `active_baseline` boundary — resolved

- `_run_interaction_one_graph` evaluates and releases diagnostic forward/comparison graphs, clears cache, then records `active_baseline` immediately before `mx.value_and_grad` graph construction. No forward construction or evaluation remains between those operations.
- `_run_interaction_materialized` follows the same immediate boundary.
- `_run_interaction_sequential` records a fresh `stage_baseline` immediately before each stage VJP graph and reports the baseline belonging to the stage that produced `max(per_stage_backward_peak)`. No pre-loop baseline is reused.
- All relevant graphs are evaluated before release; no pending diagnostic graph remains at the recorded boundary.

### Independent VJP-partition proof — resolved

The focused test computes raw gradient leaves and DOT-node counts rather than trusting helper summary booleans.

Independent in-memory mutation replay, without project-file edits:

- Normal `attention_forward_routed_vjp`: `nodes=12`, nonzero input gradient, all attention-LoRA leaf gradients zero.
- Removing attention-boundary `stop_gradient`: attention-LoRA leaf gradients become nonzero; assertion fails.
- Normal `routed_forward_attention_vjp`: `nodes=4`, nonzero input gradient, attention-LoRA gradients nonzero.
- Removing routed-output `stop_gradient`: node count becomes `12`; assertion fails.

Both required stop-gradient mutations are therefore detected independently.

### Elementwise materialized parity — resolved

- `_max_array_diff` applies `abs(a-b) <= 1e-5 + 1e-5*abs(b)` elementwise.
- `_max_grad_diff` applies the same predicate to every exact-key, exact-shape LoRA gradient leaf.
- Materialized loss, LoRA gradients, and input gradient all use these predicates.
- Counterexample `a=[1.5e-5, 1.0]`, `b=[0.0, 1.0]` returns false, catching the former global-maximum predicate.
- All four regenerated materialized rows report exact key/shape parity and zero measured loss, LoRA-gradient, and input-gradient differences.

### Fresh 25-row regeneration — resolved

- RED evidence: `2026-07-14T17:46:59Z`.
- Helper/test fixes: `2026-07-14T17:48:09Z` / `17:49:01Z`.
- Report regeneration: `2026-07-14T17:50:23Z`, after both fixes.
- Exact 25 rows; 25 unique child PIDs; 25 unique nonces; strictly non-overlapping monotonic intervals; all parent-observed exits zero.
- Plan/execution SHA-256: `a5c3474c04ab8b0edb3aa03af806cf0830ebe73c76c9cb6077d42559cfa6a356`.
- Report SHA-256: `8c24d9350d263c14bcfe61d6e18557453711865f0be175b6adb2b33ff2fecd97`.

## Contract verification

- Exact matrix/order and fixed topology: cr4, `no_shared`, T64, H128, I64, E8, K2, hc4, rank-8 LoRA on last `min(D,16)` layers.
- Fixture: 43 ordered MoE layers; SHA-256 `c80c103d3f9b8a12b6e3de5fa9b62b3efd56dc321497911a34357cf8c90ad094`.
- Exact `8_000_000_000 B` memory limit, `180 s` timeout, serial maximum concurrency 1.
- All rows: finite raw loss/gradients, nonzero input gradient, measured positive forward/backward peaks and DOT nodes, complete parent/child outcomes, no sanitization.
- Both single-VJP controls: exact composed-forward bytes.
- Metadata: `classification=not-classified-by-coder`; no classification logic or redesign authorization added.
- D43 measured `(active_baseline, peak_backward, nodes)`:
  - composed checkpoint on: `(37,412,972, 1,919,533,988, 516)`
  - attention-forward/routed-VJP: `(36,399,936, 1,930,806,180, 516)`
  - routed-forward/attention-VJP: `(36,399,936, 1,919,484,416, 172)`
  - materialized boundary: `(37,412,972, 1,919,462,592, 516)`
  - layer sequential: `(37,557,788, 79,725,228, 516)`
- Thirty protected production/vendor/Metal hashes match current bytes; no protected staged or unstaged changes.
- All 11 refreshed r2 helper/test/report/log/note files are tracked, staged, byte-identical to worktree, and have no unstaged changes. All 46 canonical test files are tracked.

## Independent validation

- Focused 5h: `8 passed, 1 warning`.
- Prior 5g: `14 passed, 1 warning`.
- Order A, 5h then 5g: `22 passed, 1 warning`.
- Order B, 5g then 5h: `22 passed, 1 warning`.
- Canonical 46-file baseline: `498 passed, 15 skipped, 2 warnings, 86 subtests passed`.
- `make`: exit 0; nothing to rebuild.
- `git diff --check`: exit 0.
- `git diff --cached --check`: exit 0.

Reviewer PASS permits only independent Test Manager closeout and subsequent Architect re-entry under the binding contract.
