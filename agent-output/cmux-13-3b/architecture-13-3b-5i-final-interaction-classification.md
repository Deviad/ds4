# Story 13.3b-5i — final interaction classification

**Decision: STOP Path A permanently. No production design slice, further diagnostic, real smoke, or training run is authorized.**

No code, production, vendor, primitive, Metal, real asset, smoke, training, or commit action was performed.

## Evidence validity

Accepted inputs:

- binding Story 13.3b-5h architecture and BA requirements;
- fresh corrected 25-row report created at `2026-07-14T17:50:23Z`;
- exact plan/execution SHA-256 `a5c3474c04ab8b0edb3aa03af806cf0830ebe73c76c9cb6077d42559cfa6a356`;
- 25 successful rows, 25 unique child PIDs/nonces, serial non-overlapping intervals, exact `8_000_000_000 B` limit, and `180 s` timeout;
- finite raw losses/gradients, exact single-VJP forward bytes, nonzero input gradients, and materialized-boundary elementwise parity at `atol=rtol=1e-5`;
- tracked helper, tests, fixture, report, notes, and validation logs;
- protected-source hashes matching all 30 guarded production/vendor/Metal files;
- Reviewer r2 PASS and Test Manager r2 GREEN;
- canonical tracked baseline `498 passed, 15 skipped, 2 warnings, 86 subtests passed`.

No report legitimacy, semantic-parity, tracking, or protected-source gate remains open.

## Exact quantitative bands

All classifications use only:

```text
delta = peak_backward - active_baseline
collapsed: delta <= 227,479,736 B
persistent: delta >= 941,632,822 B
ambiguous: 227,479,736 B < delta < 941,632,822 B
```

`x D1` means each row's backward delta divided by its own family's D1 delta.

| Family | Checkpoint | D | Backward delta (B) | x D1 | Class |
|---|---:|---:|---:|---:|---|
| composed | on | 1 | 42,201,812 | 1.0000 | collapsed |
| composed | on | 2 | 86,037,976 | 2.0387 | collapsed |
| composed | on | 4 | 173,709,792 | 4.1162 | collapsed |
| composed | on | 8 | 349,053,424 | 8.2711 | ambiguous |
| composed | on | 16 | 699,740,688 | 16.5808 | ambiguous |
| composed | on | 43 | 1,882,121,016 | 44.5981 | persistent |
| composed | off | 1 | 42,201,812 | 1.0000 | collapsed |
| composed | off | 4 | 173,709,792 | 4.1162 | collapsed |
| composed | off | 8 | 349,051,320 | 8.2710 | ambiguous |
| attention-forward + routed-VJP | on | 1 | 43,476,992 | 1.0000 | collapsed |
| attention-forward + routed-VJP | on | 8 | 352,163,612 | 8.1000 | ambiguous |
| attention-forward + routed-VJP | on | 16 | 704,947,996 | 16.2143 | ambiguous |
| attention-forward + routed-VJP | on | 43 | 1,894,406,244 | 43.5726 | persistent |
| routed-forward + attention-VJP | on | 1 | 43,215,104 | 1.0000 | collapsed |
| routed-forward + attention-VJP | on | 8 | 350,066,404 | 8.1006 | ambiguous |
| routed-forward + attention-VJP | on | 16 | 700,698,724 | 16.2142 | ambiguous |
| routed-forward + attention-VJP | on | 43 | 1,883,084,480 | 43.5747 | persistent |
| materialized attention→routed boundary | on | 1 | 42,130,672 | 1.0000 | collapsed |
| materialized attention→routed boundary | on | 8 | 348,982,028 | 8.2833 | ambiguous |
| materialized attention→routed boundary | on | 16 | 699,669,292 | 16.6071 | ambiguous |
| materialized attention→routed boundary | on | 43 | 1,882,049,620 | 44.6717 | persistent |
| composed layer-sequential | on | 1 | 42,161,808 | 1.0000 | collapsed |
| composed layer-sequential | on | 8 | 42,167,448 | 1.0001 | collapsed |
| composed layer-sequential | on | 16 | 42,161,800 | 1.0000 | collapsed |
| composed layer-sequential | on | 43 | 42,167,440 | 1.0001 | collapsed |

### D43 comparison

| Family | Backward delta (B) | Ratio to composed | Measured nodes | Class |
|---|---:|---:|---:|---|
| composed checkpoint-on | 1,882,121,016 | 1.000000 | 516 | persistent |
| attention-forward + routed-VJP | 1,894,406,244 | 1.006527 | 516 | persistent |
| routed-forward + attention-VJP | 1,883,084,480 | 1.000512 | 172 | persistent |
| materialized attention→routed boundary | 1,882,049,620 | 0.999962 | 516 | persistent |
| composed layer-sequential | 42,167,440 | 0.022404 | 516 | collapsed |

## Final adjudication

### 1. Checkpointed composition itself

Not established as causal.

Matched checkpoint-on/off composed rows are identical at D1 and D4. At D8, checkpoint-off differs by only `-2,104 B` (`0.999994x` checkpoint-on). No semantic drift exists. This rejects checkpoint state as the explanation over every matched depth and does not reopen the already-rejected checkpoint integration hypothesis.

Checkpoint-off D16/D43 was not part of the pinned bounded matrix. A D43-specific checkpoint interaction therefore remains unmeasured, but no positive checkpoint-causality evidence exists and no further diagnostic is authorized.

### 2. Attention-forward/routed-VJP leg

Persistent.

This control preserves exact composed forward bytes and routed VJP while stopping attention VJP. Its D43 delta is `1,894,406,244 B`, `1.006527x` composed, with `43.5726x` D1 scaling. Attention VJP is not necessary for the spike. This row cannot establish routed VJP as the sole cause because the complementary control also remains persistent.

### 3. Routed-forward/attention-VJP leg

Persistent.

This control preserves routed forward and attention VJP while stopping routed-output VJP. Its D43 delta is `1,883,084,480 B`, `1.000512x` composed, with `43.5747x` D1 scaling. Routed-output VJP and its additional custom-kernel nodes are not necessary for the spike: D43 remains persistent with only 172 measured nodes versus 516 composed.

Both single-VJP controls remaining persistent triggers the binding permanent-STOP gate. Neither component-specific lifetime redesign is evidence-supported.

### 4. Materialized attention→routed boundary

Does not collapse pressure.

Materialization preserves forward, loss, every LoRA gradient, and input gradient with exact key/shape equality and true elementwise `atol=rtol=1e-5` parity. D43 delta remains `1,882,049,620 B`, only `71,396 B` below composed (`0.999962x`). Same-semantics boundary materialization is therefore rejected as a production repair.

### 5. Sequential layer evaluation

Identifies graph-lifetime accumulation in the synthetic topology.

Sequential backward delta stays flat from `42,161,808 B` at D1 to `42,167,440 B` at D43 while total measured nodes still rise from 12 to 516. D43 is `2.2404%` of composed and collapsed. By contrast, every one-graph family grows approximately linearly with depth and becomes persistent at D43.

The allocation is already present immediately after differentiated-graph construction:

| D43 family | Active-after-graph delta (B) | Backward delta (B) |
|---|---:|---:|
| composed | 1,881,949,460 | 1,882,121,016 |
| attention-forward + routed-VJP | 1,894,229,012 | 1,894,406,244 |
| routed-forward + attention-VJP | 1,882,958,368 | 1,883,084,480 |
| materialized boundary | 1,881,949,460 | 1,882,049,620 |
| layer-sequential | 41,973,728 | 42,167,440 |

This is a differentiated-graph construction/liveness signature, not a peak created primarily by DOT export, evaluation, or node count alone.

The sequential control remains memory-only. It does not prove end-to-end loss, gradient, optimizer-order, accumulation, checkpoint/resume, or distributed training equivalence. The pinned gate for layer-serial backward research also required both single-VJP controls to collapse; both instead persist. No production layer-serial design is authorized.

### 6. Relation to prior cr4 evidence and real first-backward OOM

The current composed/no_shared D43 delta `1,882,121,016 B` reproduces the corrected prior no_shared/cr4 delta `1,883,265,644 B` within `1,144,628 B` (`99.9392%`). Prior D43 evidence remains coherent:

| Prior D43 control | Backward delta (B) | Class under final bands |
|---|---:|---|
| cr4 full | 1,881,839,748 | persistent |
| cr4 no_shared | 1,883,265,644 | persistent |
| cr4 no_attention | 112,472,492 | collapsed |
| cr4 no_routed | 76,945,772 | collapsed |
| cr4 routed_only | 113,739,868 | collapsed |
| cr0 full | 141,533,580 | collapsed |
| cr128 full | 141,705,340 | collapsed |

Removing either complete forward component breaks the cr4 interaction, while merely stopping either backward leg does not. Removing shared leaves it intact. Materializing the attention→routed boundary leaves it intact. Sequential graph disposal removes the depth scaling. Final synthetic classification: **compression-ratio-4 attention+routed one-graph lifetime accumulation, not a checkpoint integration defect, standalone attention defect, standalone routed primitive defect, shared-expert defect, single VJP-leg defect, boundary-materialization defect, or custom-node-count-only defect.**

The real sequence-4096 run remains authoritative RED: finite validation loss `18.109817504882812`, validation active `160,085,895,190 B`, validation peak `205,523,058,182 B`, then first backward abort before gradient callback with exit `134` and:

```text
[METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

The synthetic result supplies a structurally relevant one-graph depth-lifetime mechanism but does not reproduce the real shape, absolute memory, driver-private resources, or command-buffer failure. It cannot prove that the real Metal OOM is caused solely by MLX active-memory graph allocations. It does prove that local primitive bounds, adapter size, bounded standalone components, and kernel-node count alone cannot justify reopening Path A.

## Selected outcome: STOP Path A

Permanent STOP under the binding gate because both single-VJP controls remain persistent.

No smallest evidence-supported production slice exists:

- checkpoint redesign lacks matched causal evidence;
- routed primitive redesign is contradicted by persistence after routed-output VJP removal and by bounded routed-only evidence;
- attention lifetime redesign is contradicted by persistence after attention-VJP removal and by bounded attention-only behavior;
- boundary materialization preserves the spike;
- layer-serial execution collapses synthetic memory but lacks training-semantic equivalence and fails its pinned authorization predicate;
- a production change would therefore require speculation or semantic approximation.

### Remaining uncertainty

- Exact MLX graph object, saved value, or allocator resource responsible for the cr4 per-layer retained allocation remains unidentified.
- Checkpoint-off behavior at D16/D43 remains unmeasured.
- Compression-ratio-4 specificity is classified empirically but not reduced to one primitive buffer or source-level mechanism.
- Sequential control has no end-to-end training-equivalence proof.
- Mapping from the shape-reduced synthetic active-memory signature to real sequence-4096 Metal command-buffer private-resource exhaustion remains unquantified.

These uncertainties do not authorize another diagnostic. They are the explicit reason no safe production repair can be selected.

## Boundary and authorization state

- Path A: permanently stopped.
- Further synthetic diagnostic: not authorized.
- Production/vendor/primitive/Metal redesign: not authorized.
- Real asset access, smoke, shorter fallback, or training: not authorized.
- Existing inference, SSD streaming, CUDA/ROCm, distributed, CPU-reference, default Metal, model-loading, and ADR 0028 boundaries: unchanged.
- Any future smoke would require a separately reviewed implementation and explicit operator authorization; no such implementation is authorized by this adjudication.
