# Story 13.3b-5h — 25-row cr4 attention/routed interaction ablation

**Status:** GO for exactly one diagnostic-only synthetic interaction-ablation slice. Real-training Path A remains STOPPED. No production redesign, production or vendor edit, primitive change, real asset, smoke, or training run is authorized.

## User story

As a training engineer (WHO), I want one tracked 25-row no-model/no-shard synthetic ablation that separates the two backward legs, attention-to-routed materialization boundary, checkpoint behavior, and one-graph versus per-layer lifetime in the compression-ratio-4 attention/routed composition (WHAT), so that an Architect can classify the bounded synthetic spike or permanently stop Path A from controlled interaction evidence instead of attributing it to either bounded component alone (WHY).

## Authority and outcome boundary

- `agent-output/cmux-13-3b/architecture-13-3b-5h-classification.md` is binding.
- Corrected Story 13.3b-5g evidence and its Reviewer PASS plus Test Manager GREEN are accepted inputs and are not rerun.
- Selected classification entering this story: compression-ratio-4 attention/routed one-graph interaction.
- Checkpoint integration failure, routed primitive alone, attention path alone, shared expert path, and evaluation-time command-buffer behavior as the primary synthetic explanation are not established.
- No result from this story directly authorizes production work or a real smoke. Every successful diagnostic outcome returns to Architect for a production-boundary decision.

## Scope

This BA slice changes only:

- `docs/backlog.md`;
- `agent-output/cmux-13-3b/requirements-13-3b-5h-interaction-ablation.md`;
- the BA success marker.

A later Coder slice is conditionally authorized only for the tracked synthetic helper, focused verdict tests, consumption/validation of the current checked-in synthetic topology fixture, and generated Story 13.3b-5h diagnostic report. Reuse the corrected Story 13.3b-5g helper/test/report discipline and fixture-derived topology. Current production and vendor sources remain consumers under test, never implementation targets.

Forbidden throughout this story:

- production, vendor, primitive, package Metal, root Metal, inference, model-loader, SSD, CUDA/ROCm, distributed, CPU-reference, or default-Metal source edits;
- real model, shard, dataset, external config, real model directory, or site-packages access/edit;
- real smoke, second smoke, shorter fallback, full training, or any production redesign;
- semantic approximation, reduced matrix, substituted shape, concurrent children, higher memory limit, or longer timeout.

## Fixed topology and execution contract

Every row uses exactly:

- compression ratio `4`;
- shared expert disabled (`no_shared`);
- `T=64`, `H=128`, `I=64`, `E=8`, `K=2`, `hc_mult=4`;
- rank-8 LoRA on the last `min(D,16)` layers;
- the exact quantize → freeze → LoRA topology already proven by corrected Story 13.3b-5g;
- raw, unsanitized losses and gradients;
- fixture-derived layer prefixes from the existing checked-in 43-entry synthetic topology fixture;
- MLX memory limit exactly `8_000_000_000 B`;
- one fresh subprocess per row, at most `180` seconds per child, serial execution, maximum concurrency `1`, unique PID and nonce, and non-overlapping monotonic intervals.

Existing cr0/cr128 and standalone rows remain controls by reference. Do not regenerate them, add them to the 25 rows, or mix their values into new measurements.

## Exact 25-row matrix and order

Ordering and identity are load-bearing. Execute exactly these rows:

1. `composed_checkpoint_on`, `D=1`.
2. `composed_checkpoint_on`, `D=2`.
3. `composed_checkpoint_on`, `D=4`.
4. `composed_checkpoint_on`, `D=8`.
5. `composed_checkpoint_on`, `D=16`.
6. `composed_checkpoint_on`, `D=43`.
7. `composed_checkpoint_off`, `D=1`.
8. `composed_checkpoint_off`, `D=4`.
9. `composed_checkpoint_off`, `D=8`.
10. `attention_forward_routed_vjp`, checkpoint on, `D=1`.
11. `attention_forward_routed_vjp`, checkpoint on, `D=8`.
12. `attention_forward_routed_vjp`, checkpoint on, `D=16`.
13. `attention_forward_routed_vjp`, checkpoint on, `D=43`.
14. `routed_forward_attention_vjp`, checkpoint on, `D=1`.
15. `routed_forward_attention_vjp`, checkpoint on, `D=8`.
16. `routed_forward_attention_vjp`, checkpoint on, `D=16`.
17. `routed_forward_attention_vjp`, checkpoint on, `D=43`.
18. `materialized_attention_routed_boundary`, checkpoint on, `D=1`.
19. `materialized_attention_routed_boundary`, checkpoint on, `D=8`.
20. `materialized_attention_routed_boundary`, checkpoint on, `D=16`.
21. `materialized_attention_routed_boundary`, checkpoint on, `D=43`.
22. `composed_layer_sequential`, checkpoint on, `D=1`.
23. `composed_layer_sequential`, checkpoint on, `D=8`.
24. `composed_layer_sequential`, checkpoint on, `D=16`.
25. `composed_layer_sequential`, checkpoint on, `D=43`.

Group count: `6 + 3 + 4 + 4 + 4 + 4 = 25`. No omitted, reduced, reordered, substituted, or extra row is permitted.

## Exact control semantics

### Composed checkpoint controls

`composed_checkpoint_on` and `composed_checkpoint_off` use the same no_shared cr4 attention+routed one-graph forward, loss, trainable tree, and input-gradient probe. Only checkpoint state differs. Checkpoint integration remains accepted as correct; these rows test checkpoint efficacy inside the composition, not checkpoint-call-site integration.

### Attention-forward plus routed-VJP

`attention_forward_routed_vjp` must:

- preserve exact composed-forward bytes;
- keep routed forward and routed VJP present;
- stop the attention VJP exactly at the attention-to-routed boundary;
- preserve a nonzero input-gradient path with the already-proven zero-forward identity-gradient expression;
- retain the exact LoRA key set and shapes; zero LoRA values are allowed, but no key may disappear.

### Routed-forward plus attention-VJP

`routed_forward_attention_vjp` must:

- preserve exact composed-forward bytes;
- keep routed forward present;
- stop the routed-output VJP;
- preserve the residual/input path;
- retain the exact LoRA key set and shapes; zero LoRA values are allowed, but no key may disappear.

### Same-semantics materialized boundary

`materialized_attention_routed_boundary` must materialize only between attention and routed work. Against the matched composed row it must preserve:

- forward output and loss;
- every LoRA gradient key and shape;
- every LoRA gradient value;
- input-gradient shape, nonzero status, and value;

with `atol=rtol=1e-5` for loss and gradient parity. Any forward, loss, key, shape, or gradient parity failure invalidates the control and triggers STOP.

### Composed per-layer sequential control

`composed_layer_sequential` is a memory-only control:

- evaluate one complete no_shared cr4 attention+routed layer at a time;
- carry only the evaluated stage output;
- delete prior graph references;
- clear cache between stages;
- report the pinned per-stage maximum;
- make no end-to-end training-equivalence or production-design claim.

## Required telemetry and evidence

Every row records, from real measurements rather than copied or arithmetic placeholders:

1. exact row index, case identity, checkpoint state, depth, and all fixed topology fields;
2. `active_baseline` immediately before graph construction with no pending graph;
3. `active_after_graph` immediately after constructing the unevaluated differentiated graph and before DOT export or evaluation;
4. pre-boundary and post-boundary active memory where a control has an explicit boundary;
5. `peak_forward` and `peak_backward`;
6. final active memory and cache memory, without using either as a substitute for peak;
7. measured DOT custom-kernel node count;
8. raw loss and gradient finiteness;
9. duration and normalized process outcome;
10. child PID, unique nonce, monotonic start/end interval, and proof of serial non-overlap;
11. exact trainable-key count and SHA-256 of newline-joined sorted keys;
12. input-gradient shape, finiteness, and nonzero status;
13. exact-forward equality evidence for both single-VJP controls;
14. materialized-boundary loss/gradient parity maxima at `atol=rtol=1e-5`;
15. plan identity and execution identity sufficient to prove all 25 planned rows executed exactly once in order.

Every attempted row remains in the generated report, including timeout, signal, OOM, and handled nonzero outcomes. Child output cannot override the parent-observed process outcome. No report row may be retained from an older run or patched by hand.

## Quantitative classification bands

Use only baseline-subtracted backward delta:

```text
delta = peak_backward - active_baseline
```

Accepted corrected reference:

```text
composed D=43 no_shared/cr4 delta = 1,883,265,644 B
```

Bounded ceiling:

```text
2 * max(76,945,772, 112,472,492, 113,739,868) = 227,479,736 B
```

Classification:

- `delta <= 227,479,736 B`: collapsed/bounded.
- `delta >= 941,632,822 B`: spike persists; this is 50% of the composed reference.
- `227,479,736 B < delta < 941,632,822 B`: ambiguous; no redesign authorization.

No root-cause claim may use no_routed or no_attention alone. Both are interaction-breaking controls, while no_shared is the positive interaction control that retains attention plus routed and preserves the corrected spike.

## Classification and Architect re-entry gates

After complete evidence, return to Architect and apply only these gates:

1. Select custom primitive lifetime redesign only if attention-VJP removal leaves the spike persistent, routed-VJP removal collapses it, and composed sequential stays bounded.
2. Select attention checkpoint/lifetime redesign only if routed-VJP removal leaves the spike persistent, attention-VJP removal collapses it, and checkpoint/materialization evidence points to attention recomputation or retention.
3. Select layer-serial backward research only if both single-VJP controls collapse, composed one-graph persists, composed sequential is bounded, and the materialized boundary preserves exact loss/gradient semantics while collapsing the spike.
4. Classify checkpoint interaction only if matched cr4 checkpoint-on/off rows diverge beyond ordinary measurement noise while semantic parity remains exact. This never reopens checkpoint integration failure.
5. Classify command-buffer/lazy-boundary interaction only if same-semantics materialization collapses the spike. `active_after_graph` remains load-bearing; Metal error text alone is insufficient.
6. Select permanent STOP Path A if controls are ambiguous, both single-VJP controls remain persistent, parity fails, any required `D=43` row cannot complete within bounds, or classification would require semantic approximation.

No gate authorizes automatic implementation, production redesign, primitive change, real asset access, or smoke. Architect must issue a separate production-boundary decision after successful Reviewer and Test Manager closeout.

## Acceptance criteria

All required:

1. Exact 25-row order and identities above; no omitted, reduced, reordered, substituted, or extra cell.
2. Every row uses cr4, no_shared, `T=64`, `H=128`, `I=64`, `E=8`, `K=2`, `hc_mult=4`, rank-8 LoRA on the last `min(D,16)` layers, raw gradients, and fixture-derived prefixes.
3. Fresh subprocess per row; unique PID/nonce; non-overlapping intervals; serial maximum concurrency `1`.
4. MLX memory limit exactly `8_000_000_000 B`; per-child timeout at most `180` seconds.
5. Existing fixture is validated and consumed; no real model, shard, dataset, external config, real model directory, or site-packages access.
6. Exact quantize → freeze → LoRA topology and exact tracked raw-gradient key set and shapes.
7. Both single-VJP controls preserve exact composed-forward bytes and required nonzero input-gradient path.
8. Materialized boundary preserves forward, loss, every LoRA gradient, and input gradient at `atol=rtol=1e-5` with exact key/shape equality.
9. Raw losses and gradients are finite; no `nan_to_num`, hidden sanitization, or dropped key.
10. Real `active_after_graph` boundary, real pre/post-boundary telemetry where applicable, real forward/backward peaks, and measured DOT custom-kernel nodes.
11. Complete parent-observed outcomes and aligned child identity evidence for all 25 rows; no hand-patched or reused report row.
12. Baseline-subtracted deltas use exactly `227,479,736 B` and `941,632,822 B` as bounded/persistent thresholds.
13. Helper, focused tests, fixture, and generated report are tracked by `git ls-files` before any verdict.
14. Focused and tracked applicable baseline tests are GREEN; direct protected-source hashes prove production/vendor/FROZEN integrity; `git diff --check` is clean.
15. Independent Reviewer PASS and independent Test Manager GREEN.
16. Double-GREEN authorizes only Architect re-entry; no automatic redesign, production edit, primitive change, real smoke, or training.

## Immediate STOP gates

Stop the diagnostic and return captured evidence to Architect on any:

- real asset, smoke, training, sequence fallback, production/vendor/primitive/Metal edit, or site-packages access/edit;
- matrix reduction, omission, reordering, extra row, substituted shape, concurrency, memory-limit increase, or timeout increase;
- topology, trainable-key, gradient-key, gradient-shape, or input-gradient-path mismatch;
- forward-byte mismatch in either single-VJP control;
- materialized-boundary forward, loss, gradient-key, gradient-shape, or value-parity failure;
- non-finite raw loss or gradient, hidden sanitization, or missing gradient key;
- missing `peak_backward`, copied/fabricated telemetry, or reliance on final active memory instead of peak;
- missing, reused, overlapping, or identity-inconsistent child evidence;
- any required `D=43` row that times out, OOMs, signals, exits nonzero, or cannot complete inside `8_000_000_000 B` and `180` seconds;
- any untracked verdict helper, focused test, fixture, or generated report;
- any root-cause claim from no_routed or no_attention alone;
- any attempt to authorize or perform production redesign before Architect re-entry.

A STOP cannot authorize a smaller matrix, reduced dimensions, omitted cell, concurrency, larger memory limit, longer timeout, approximation, production repair, real assets, or smoke.

## Review and handoff gates

- Coder follows TDD RED → GREEN and stays within the synthetic diagnostic scope.
- Reviewer independently verifies semantics, exact matrix/order, thresholds, tracking, protected-source hashes, and report legitimacy; Reviewer does not edit production code.
- Test Manager independently reruns the authorized validation, verifies every cited verdict file is tracked, and records the process/telemetry evidence.
- Reviewer PASS plus Test Manager GREEN are both mandatory.
- Architect re-entry is mandatory after double-GREEN and before any production-boundary decision.
