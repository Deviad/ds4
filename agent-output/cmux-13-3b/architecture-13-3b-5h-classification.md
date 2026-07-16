# Story 13.3b-5h — classification after diagnostic double-GREEN

**Decision: classify the synthetic spike as a compression-ratio-4 attention/routed one-graph interaction. Keep real-training Path A stopped. Select one minimal synthetic interaction ablation before any production redesign.**

No implementation was performed by this adjudication. No real asset, smoke, training, production source, vendor source, primitive, or Metal path was accessed or changed.

## Evidence accepted

The corrected Story 13.3b-5g evidence is contract-valid:

- Probe A: 43 decoder layers, 43 checkpointed forward calls, checkpoint wrapper `checkpointed_fn`, exact checkpoint-on/off gradient-key and shape equality, finite losses/gradients, and zero reported gradient difference.
- Probe B checkpoint-on D=43: `152,425,220 B` backward peak.
- Probe C D=43/E=256: `66,048` measured custom-kernel nodes; one-graph `5,491,948 B`; sequential `1,483,376 B`.
- Probe D cr4: full `1,918,140,864 B`; no_attention `148,773,608 B`; no_routed `113,283,804 B`; no_shared `1,919,566,760 B`; routed_only `150,040,984 B`.
- Probe D cr0/cr128 full: `152,425,480 B` / `153,084,344 B`.
- All 64 report rows succeeded with finite loss and raw gradients; 64 unique child PIDs/nonces ran serially; plan/execution hashes match.
- Helper, focused test, fixture, and report are tracked. Fixture and all recorded provenance hashes match current bytes.
- Reviewer r4 PASS and Test Manager r4 GREEN independently close the corrected diagnostic gate.

The corrected report supersedes all earlier 5g numeric peaks.

## Exact classification-gate adjudication

### 1. Checkpoint integration fix — rejected

Gate not met.

Probe A observes exactly 43 checkpointed calls across 43 decoder layers. Checkpoint-on/off gradient keys, shapes, losses, and gradients match. Probe B checkpoint-on remains finite and bounded through D=43. This rejects bypassed-layer, wrong-class, and checkpoint-call-site integration explanations.

Checkpoint efficacy inside a composed cr4 graph remains a possible interaction variable. That is not checkpoint integration failure.

### 2. Custom primitive lifetime redesign — not authorized

Gate conjunction not met.

Probe C proves routed one-graph lifetime exceeds its matched sequential control, but remains bounded. Across the normalized endpoints, measured custom-kernel nodes rise from 48 to 66,048 (`1376x`) while backward peak rises only about `7.25x`; D=43/E=256 remains `5,491,948 B`. This does not establish routed-only peak or failure tracking `depth x nonempty_experts` strongly enough to explain the cr4 `1.918 GB` composed spike.

At cr4, no_routed removes more than half of the full D=43 delta, but that predicate alone is insufficient. Routed_only remains bounded at `150,040,984 B`, and no_attention also collapses the peak. A routed primitive redesign cannot be selected from an interaction signature that disappears when either side is removed.

### 3. Attention/shared checkpoint redesign — not authorized

Gate not met.

At cr4, baseline-subtracted full delta is `1,881,839,748 B`:

- no_routed delta: `76,945,772 B`, removing `95.911%`;
- no_attention delta: `112,472,492 B`, removing `94.023%`;
- no_shared delta: `1,883,265,644 B`, removing nothing;
- routed_only delta: `113,739,868 B`, bounded.

The pinned gate requires no_attention or no_shared to remove more than no_routed while routed-only stays bounded. Neither does. Shared removal preserves the spike and slightly increases it. Attention-only behavior is bounded by no_routed. An attention-only or shared-path redesign is therefore not authorized.

### 4. Layer-serial backward research — not authorized yet

Gate not met.

Probe C supplies only a routed-only sequential control. It is not a matched sequential control for the cr4 attention+routed composition. Comparing full/no_shared cr4 against routed-only sequential would cross component topology and cannot satisfy the pinned gate.

Layer-serial research requires a same-topology composed sequential control plus semantic-parity evidence. Neither exists yet.

### 5. STOP Path A — active safety state; permanent abandonment not selected yet

Real-training Path A remains stopped. No real smoke is authorized.

The matrix does classify the bounded synthetic phenomenon more narrowly than inseparable whole-model behavior: removing shared leaves the spike intact, cr0/cr128 remain bounded, and removing either attention or routed collapses cr4. The isolated signature is therefore a cr4 attention/routed composition, not an unbounded shared path, checkpoint bypass, routed-only node-pressure failure, or arbitrary whole-model failure.

One bounded interaction control can still separate the two backward legs and outer-graph lifetime. Permanent Path A abandonment becomes mandatory if that control remains ambiguous, violates parity, or cannot classify under the existing 8 GB/180-second bounds.

## Culprit classification

**Selected category: routed/attention composition, specific to compression ratio 4.**

Rejected as sole causes:

- attention compression-ratio-4 path alone;
- routed primitive alone;
- shared expert path;
- checkpoint integration;
- command-buffer interaction as the primary synthetic explanation.

The cr4 full/no_shared allocation is already present at `active_after_graph`, before DOT export and `mx.eval`. Peak adds only a small amount afterward. The synthetic spike therefore points first to lazy differentiated-graph construction/liveness, not an evaluation-time Metal command-buffer transient. The real smoke's command-buffer OOM may still be the failure manifestation; synthetic evidence does not prove that driver-private resources are irrelevant.

No_routed and no_attention are both interaction-breaking controls. Each removes one required side of the attention+routed pair. Their collapse does not make both components independently causal, and the bounded standalone paths disprove any single-component attribution from these rows alone. No_shared is the positive interaction control: it retains attention plus routed, removes shared, and preserves the spike.

## Selected next architecture: minimal synthetic interaction ablation

Diagnostic-only extension. Reuse the corrected helper/test/report discipline and current checked-in synthetic topology fixture. No real asset and no production edit.

### Exact matrix: 25 fresh-process rows

All rows use cr4, no shared expert, `T=64`, `H=128`, `I=64`, `E=8`, `K=2`, `hc_mult=4`, rank-8 LoRA on the last `min(D,16)` layers, raw gradients, and fixture-derived layer prefixes.

1. Composed attention+routed one-graph, checkpoint on, `D=[1,2,4,8,16,43]`: 6 rows.
2. Same composed one-graph, checkpoint off, `D=[1,4,8]`: 3 rows.
3. Attention-forward plus routed-VJP control, checkpoint on, `D=[1,8,16,43]`: 4 rows.
4. Routed-forward plus attention-VJP control, checkpoint on, `D=[1,8,16,43]`: 4 rows.
5. Same-semantics materialized attention-to-routed boundary, checkpoint on, `D=[1,8,16,43]`: 4 rows.
6. Composed per-layer sequential memory control, checkpoint on, `D=[1,8,16,43]`: 4 rows.

Total: `6 + 3 + 4 + 4 + 4 + 4 = 25`.

Existing cr0/cr128 and standalone rows remain controls by reference; do not regenerate or mix their values into new rows.

### Control semantics

- Attention-forward plus routed-VJP: preserve exact composed forward bytes; stop the attention VJP at the attention-to-routed boundary; preserve a nonzero input-gradient path with the already-proven zero-forward identity-gradient expression. LoRA keys remain exact; zero LoRA values are permitted but no key may disappear.
- Routed-forward plus attention-VJP: preserve exact composed forward bytes; stop the routed-output VJP; preserve the residual/input path. Routed forward remains present.
- Materialized boundary: preserve composed forward, loss, LoRA gradients, and input gradient within `atol=rtol=1e-5`. Materialize only between attention and routed work. Any parity failure invalidates this control.
- Sequential: memory-only control. Evaluate one complete no_shared cr4 layer at a time, carry only evaluated stage output, delete graph references, and clear cache. Make no end-to-end training-equivalence claim.

Record `active_baseline`, `active_after_graph`, pre/post-boundary active memory where applicable, `peak_forward`, `peak_backward`, final/cache memory, measured custom-kernel nodes, finiteness, duration, process outcome, child identity, trainable-key hash, input-gradient shape/nonzero, and exact matrix identity.

### Quantitative classification bands

Use baseline-subtracted backward delta.

Reference composed D=43 no_shared/cr4 delta: `1,883,265,644 B`.

Bounded ceiling: twice the largest corrected D=43 cr4 standalone delta:

```text
2 * max(76,945,772, 112,472,492, 113,739,868) = 227,479,736 B
```

- `delta <= 227,479,736 B`: collapsed/bounded.
- `delta >= 941,632,822 B` (50% of composed reference): spike persists.
- Between thresholds: ambiguous; no redesign authorization.

### Next re-entry gates

- Select custom primitive lifetime redesign only if attention-VJP removal leaves the spike persistent, routed-VJP removal collapses it, and composed sequential stays bounded.
- Select attention checkpoint/lifetime redesign only if routed-VJP removal leaves the spike persistent, attention-VJP removal collapses it, and checkpoint/materialization evidence points to attention recomputation or retention.
- Select layer-serial backward research only if both single-VJP controls collapse, composed one-graph persists, composed sequential is bounded, and the materialized boundary preserves exact loss/gradient semantics while collapsing the spike.
- Classify checkpoint interaction only if matched cr4 checkpoint-on/off rows diverge beyond ordinary measurement noise while semantic parity remains exact. This still does not reopen checkpoint integration.
- Classify command-buffer/lazy-boundary interaction only if same-semantics materialization collapses the spike. Active-after-graph evidence remains load-bearing; Metal error text alone is insufficient.
- Select permanent STOP Path A if controls are ambiguous, both single-VJP controls remain persistent, parity fails, a required D=43 row cannot complete within bounds, or classification would require semantic approximation.

No outcome directly authorizes a production change or real smoke. Every successful diagnostic outcome returns to Architect for a production-boundary decision.

## Acceptance gates

All required:

1. Exact 25-row order and identities; no omitted or reduced cell.
2. Fresh subprocess per row, serial execution, unique PID/nonce, non-overlapping intervals.
3. MLX memory limit exactly `8_000_000_000 B`; per-child timeout at most 180 seconds.
4. Existing fixture validated and consumed; no real model, shard, dataset, external config, or site-packages access.
5. Exact quantize-freeze-LoRA topology and tracked raw gradient keys.
6. Exact forward equality for VJP-leg controls; materialized-boundary loss/gradient parity at `atol=rtol=1e-5`.
7. Finite raw losses and gradients; no `nan_to_num` or hidden sanitization.
8. Real `active_after_graph` boundary and measured DOT custom-kernel nodes.
9. Complete captured process outcomes; no hand-patched report rows.
10. Helper, tests, fixture, and report tracked before any verdict.
11. Direct protected-source hashes and clean `git diff --check`.
12. Independent Reviewer PASS and Test Manager GREEN.
13. Architect re-entry after double-GREEN; no automatic redesign or smoke.

## STOP conditions

Immediate STOP and return to Architect:

- any real asset, smoke, training, sequence fallback, or production/vendor/primitive/Metal edit;
- any matrix reduction, substituted shape, concurrency, limit increase, or timeout increase;
- any missing peak or reliance on final active memory;
- any forward, loss, gradient-key, gradient-shape, or parity failure;
- any non-finite raw gradient or hidden gradient sanitization;
- any untracked verdict helper/test/fixture/report;
- any root-cause claim from no_routed or no_attention alone;
- any production redesign before the next classification re-entry.

## Authorization and durable documentation

BA is authorized to pin this single diagnostic-only story and update `docs/backlog.md` with trackable requirements and acceptance criteria.

Coder is conditionally authorized after BA handoff, only for the synthetic helper, focused tests, fixture-consumption support, and generated diagnostic report named by that story. No production or real-training work is authorized.

Reviewer and Test Manager remain mandatory after Coder.

No durable architecture document or ADR change is made now. Production boundaries, ADR 0028 primitive semantics, and the real-training STOP remain unchanged. This classification adds evidence and one diagnostic gate; it does not change shipped architecture.
