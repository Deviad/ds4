# Story 13.3b-5g — first-backward OOM adjudication

**Decision: STOP current real-training Path A. Select one diagnostic-only multi-layer synthetic peak proof as next slice.**

No second real smoke. No shorter fallback. No model, shard, or dataset run. No production fix authorized yet.

## Evidence reviewed

- Commit `5dee4ce7037902c9787b523373ca46d40aaf6de0` and its tracked manifest.
- ADR 0028, Architect r3 reduction decision, BA r3 requirements.
- Final r4 Reviewer PASS and Test Manager GREEN.
- `smoke-4096-routed-fp4-20260714-135103.log`.
- `run_smoke_4096_routed_fp4.py` and `run-smoke-4096-routed-fp4.sh`.
- Current `SparseMoeBlockNN`, `routed_fp4`, MLX-LM trainer/checkpoint path, default loss, validation, LoRA setup.
- Existing fixed-expert, real-dimension, formula, opacity, parity, and benchmark probes.
- New no-model checkpoint-coverage probe: `architect-13-3b-5g-checkpoint-probe.log`.

No real asset loaded by this adjudication.

## Authoritative RED

Pinned smoke conditions held:

- exact 4096/20 command;
- `mx.disable_compile()`;
- graph limit `400_000_000_000`;
- no concurrent heavy model process;
- finite validation loss `18.109817504882812` in `82.645s`;
- validation active `160,085,895,190` bytes;
- validation peak `205,523,058,182` bytes;
- validation cache `9,252` bytes;
- 96-tensor trainable snapshot increased active memory by `22,282,292` bytes, about `0.0208 GiB`;
- first backward aborted before the wrapper's gradient callback;
- Metal command-buffer out-of-memory, exit `134`;
- only `adapter_config.json`; no `adapters.safetensors`; no surviving process.

Last completed telemetry does not reveal the failing backward peak. The Metal error does not prove that MLX's reported peak crossed either `340_000_000_000` or `400_000_000_000`. Driver/command-buffer resource exhaustion below those counters remains possible.

## Why previous probes did not predict this failure

Previous GREEN evidence proved local primitive properties, not whole-trainer lifetime.

| Probe | Covered | Missing from real first backward |
|---|---|---|
| Fixed-E fresh processes | One routed block; `E=2/4/8`; fixed assignments; small dimensions | 43-layer depth, `E=256`, attention, shared expert, loss/head, trainer parameter gradients |
| Real-dimension helper | One routed block; `H=4096`, `I=2048`; two non-empty experts | 256 non-empty expert segments, 43 checkpoint frames, ordinary module backward |
| `T=4096` test | One block and two experts | Same omissions above |
| Formula envelope | One routed operation, assignment-proportional tensors | Outer graph metadata, cross-layer liveness, command-buffer scheduling, ordinary attention/shared paths |
| Primitive benchmark | Independent forward/input-VJP launches | One lazy `value_and_grad` graph and allocator reuse across 43 layers |

The helper's direct real-dimension case also uses `T=512`; the separate test reaches `T=4096` but still uses `E=2` and one layer. Neither exercises the real `43 x 256` expert-segment launch topology.

ADR 0028's `1504 MiB` envelope is a per-routed-operation bound. It cannot be treated as the whole-model additive peak without a scheduler-lifetime proof. Forty-three such envelopes equal `63.16 GiB`; this is not a claim that all remain simultaneously live, only the size of the untested depth-lifetime risk.

Validation already demonstrated a forward peak `45,437,162,992` bytes (`42.32 GiB`) above its post-validation active state. First backward adds layer recomputation, activation cotangents, parameter cotangents, and command-buffer transients. The former 300 GiB process budget therefore remains an estimate, not accepted evidence.

## Checkpoint adjudication

Current MLX-LM behavior:

1. `trainer.train` calls `grad_checkpoint(model.layers[0])` before validation and training.
2. `grad_checkpoint` replaces `type(layer).__call__` with `checkpointed_fn`.
3. `Model.layers` returns `self.model.pipeline_layers`.
4. In the non-distributed smoke, that slice contains all 43 layers.
5. All 43 are instances of the same `DecoderLayerNN` class.

No-model 43-layer probe result:

```text
layer_count=43
pipeline_layer_count=43
unique_layer_types=['DecoderLayerNN']
all_decoder_layers=True
class_call_changed=True
all_layers_share_checkpointed_class_call=True
checkpoint_wrapper_name=checkpointed_fn
```

**Conclusion:** `grad_checkpoint(model.layers[0])` does install the checkpoint wrapper for all 43 decoder layers. A checkpoint call-site integration fix is rejected as the next architecture.

This proves coverage, not efficacy. `mx.checkpoint` still leaves one outer lazy gradient graph. Each layer input remains a checkpoint boundary, and each backward layer recomputes attention, routed expert, shared expert, and hyperconnection work. Pessimistic FP32 checkpoint inputs alone are `10.75 GiB` for `[1,4096,4,4096] x 43`; BF16 inputs are `5.375 GiB`. Scheduler lifetime inside the complete gradient evaluation remains unproven.

## Hypothesis separation

### H1 — retained activation/checkpoint graph

**Status: plausible; coverage proven, lifetime unproven.**

Checkpointing removes the requirement to retain every ordinary forward intermediate. It does not reduce the model to one independently evaluated layer backward. The outer `nn.value_and_grad` graph still contains 43 checkpoint VJPs, inter-layer cotangents, route metadata, and trainable-parameter paths. `mx.eval(*gradients)` triggers that graph as one backward evaluation.

Falsifier: depth sweep with checkpoint on/off and one-graph versus per-layer sequential controls. Peak growth near explicit checkpoint-input bytes with flat sequential control weakens this hypothesis. Large unexplained depth growth supports it.

### H2 — attention and shared-expert paths

**Status: plausible and unmeasured by routed primitive tests.**

Validation's `42.32 GiB` transient above post-validation active state proves substantial full-model forward work outside the frozen payload baseline. First backward recomputes ordinary attention and shared SwiGLU paths. LoRA gradients exist only for the last 16 layers by MLX-LM's default `num_layers=16`; 96 trainable tensors equal `16 layers x 3 targets x 2 LoRA tensors`. Small adapter storage does not imply small activation backward.

Falsifier: test-only `no_attention`, `no_shared`, and `no_routed` ablations on the same shape-reduced 43-layer topology. Each ablation preserves residual shape and differentiable input flow. Compare fresh-process backward peaks, not final active memory.

### H3 — routed primitive VJP lifetime

**Status: plausible; local opacity proven, depth/expert composition unproven.**

One `routed_fp4` call creates expert-segment forward outputs, concatenates them, then its VJP creates all expert `dx_e`/`a_e` parts and concatenates them before scatter. Packed matrices stay opaque, but one real layer can still contribute up to 256 expert segments and six kernel families. The outer 43-layer graph can therefore contain tens of thousands of `CustomKernel` nodes even though each local tensor is assignment-proportional.

Falsifier: routed-only residual chain over independent depths and expert counts, with tensor dimensions fixed while `depth x nonempty_experts` changes. A `no_routed` full-layer control isolates routed contribution.

### H4 — Metal command-buffer transient/resource pressure

**Status: plausible; not proven solely by error text.**

Failure source is a Metal command-buffer callback. Last successful MLX telemetry precedes backward, so driver-private scratch, command encoding resources, or allocations attached to the failed command buffer may be absent from recorded peak. Many small opaque kernels can stress this path independently from packed-payload bytes.

Falsifier: fixed-small-tensor node-pressure sweep and a matched sequential-evaluation control. If one lazy graph grows or aborts with `depth x experts` while sequential evaluation remains flat, command-buffer/whole-graph lifetime becomes load-bearing. If both track only tensor volume, node pressure weakens.

### H5 — trainable snapshot or adapter storage

**Status: falsified as primary cause.**

Snapshot delta was only `22,282,292` bytes. Failure occurred before optimizer update or adapter serialization. Keep snapshot code out of future peak proof to avoid a confound, but do not redesign adapter storage.

## Selected next architecture: tracked multi-layer synthetic peak proof

Diagnostic-only slice. No production source edit. No site-packages edit. No real asset path access.

### Files

Suggested tracked additions:

- `tests/helpers/routed_fp4_multilayer_peak_probe.py`
- `tests/test_deepseek_v4_nn_multilayer_peak.py`
- `agent-output/cmux-13-3b/multilayer-peak-report.json`

Generated report may be handoff evidence. Helper and tests must be tracked before any verdict.

### Probe A — checkpoint coverage

Fresh subprocess. Tiny 43-layer `Model`.

Require:

- `len(model.layers) == 43`;
- 43 `DecoderLayerNN` instances;
- one instrumented forward observes exactly 43 calls through `checkpointed_fn`;
- checkpoint-on and checkpoint-off tiny gradients finite and equal at `atol=rtol=1e-5`;
- class monkeypatch confined to subprocess.

Any count other than 43 selects checkpoint integration repair and stops all other architecture selection.

### Probe B — depth/activation volume

Fresh process per cell; `mx.disable_compile()`.

Pinned shape-reduced full-model axis:

```text
D = {1, 2, 4, 8, 16, 43}
T = 64
H = 128
I = 64
E = 8
K = 2
hc_mult = 4
LoRA targets = q_a_proj, q_b_proj, kv_proj
trainable layers = min(D, 16)
```

Run checkpoint on for every depth. Run checkpoint off for `D={1,4,8}` as bounded controls. Record baseline, after graph construction, forward peak, backward peak, final active/cache, duration, finiteness, and exit status.

### Probe C — expert/node pressure

Routed-only residual chain. Frozen synthetic packed payload; no shared expert or attention.

```text
D = {1, 8, 16, 43}
E = {2, 8, 32, 128, 256}
T = 64
H = 64
I = 32
K = min(6, E)
```

Deterministic routes must make every expert non-empty where `T*K >= E`. Hold logical tensor dimensions and total assignments fixed where comparing E. Record `depth x nonempty_experts`, kernel-node count on bounded cells, and memory telemetry.

Matched control: same total one-layer/expert work evaluated sequentially with an evaluation/cache-clear boundary after each layer. This control is memory-only; no claim of mathematical equivalence to end-to-end training.

### Probe D — component ablations

Shape-reduced current `DecoderLayerNN` topology. Cases:

- `full`;
- `no_routed`;
- `no_shared`;
- `no_attention`;
- `routed_only`.

Test-only replacements return zero-like outputs with exact expected shape and preserve an input-gradient path. Include representative compression ratios `0`, `4`, and `128`; include one checked-in 43-entry shape-reduced topology fixture derived from canonical prior config evidence, not from the real model directory.

### Telemetry schema

Each JSON row must include:

```text
case, depth, experts, nonempty_experts, tokens, hidden, intermediate, top_k,
hc_mult, compression_ratio, checkpoint, component_mask,
active_baseline, active_after_graph, peak_forward, peak_backward,
active_final, cache_final, custom_kernel_nodes, duration_s,
finite_loss, finite_gradients, exit_code, signal, error
```

Peak values are load-bearing. Final active values alone are not verdict evidence.

### Lightweight bounds

- No real model/shard/dataset/config path.
- Fresh subprocess per measurement cell.
- One cell hard timeout: 180 seconds.
- Diagnostic MLX memory limit: `8_000_000_000` bytes.
- No concurrent probe processes.
- Any intentional subprocess OOM captured as evidence; parent report still completes.
- If a pinned cell exceeds time or memory, STOP and return to Architect. Do not silently reduce depth, expert count, or omit the cell.

## Classification gates after report

These gates select the following architecture; they do not authorize code by themselves.

1. **Checkpoint integration fix**
   - Select only if Probe A observes fewer than 43 checkpointed calls, a bypassed layer class, or checkpoint-on gradient divergence.

2. **Custom primitive lifetime redesign**
   - Select if routed-only one-graph peak or failure tracks `depth x nonempty_experts`, sequential control remains bounded, and `no_routed` removes at least half of the full-model D=43 backward peak delta.
   - Next design target: bounded stage count across all assignment segments, direct assignment buffers, no Python `y_parts`/`dx_parts`/`a_parts` concatenation forest. Exact ADR 0028 math and opacity remain unchanged.

3. **Attention/shared checkpoint redesign**
   - Select if `no_attention` or `no_shared` removes more D=43 peak than `no_routed`, while routed-only depth/node probes stay bounded.
   - Requires separate component-specific Architect re-entry. Do not change routed primitive spec opportunistically.

4. **Layer-serial backward research**
   - Consider only if each component and sequential control is bounded but the composed one-graph case alone fails or exceeds twice the matched sequential peak delta.
   - Requires a new training-loop ADR, exact loss/gradient/update parity, accumulation semantics, optimizer-state ordering, and checkpoint/resume proof. Not authorized now.

5. **STOP Path A**
   - Keep real training permanently stopped on MLX 0.31.2 if the bounded synthetic matrix cannot classify the failure, if multiple components remain inseparable, or if the only passing approach needs semantic approximation, expert dropping, sequence shortening, or a permanent alternate path flag.

## Diagnostic slice acceptance gates

All required for PASS:

1. Probe helper and every test contributing to verdict tracked by Git.
2. Complete JSON report for every pinned cell or explicit captured subprocess failure.
3. Probe A exact 43-layer checkpoint coverage result.
4. Finite checkpoint-on/off parity for bounded tiny case at `atol=rtol=1e-5`.
5. Fresh-process peak telemetry for depth, node-pressure, sequential-control, and component-ablation matrices.
6. No model, shard, dataset, external config, real smoke, or full training access.
7. No production inference, root Metal, SSD, CUDA, ROCm, distributed, model-loader, FROZEN parity body, or current primitive edit.
8. `git diff --check` clean; protected production hashes recorded directly.
9. Independent Reviewer PASS and Test Manager GREEN.
10. New Architect re-entry selects or rejects a production redesign. Diagnostic GREEN alone authorizes no redesign and no smoke.

## STOP gates

Immediate STOP:

- any second real smoke or shorter-sequence fallback;
- any real asset load;
- missing peak telemetry or reliance on final active memory;
- omitted `D=43` or `E=256` node-pressure cell without Architect re-pin;
- diagnostic limit above 8 GB or concurrent cells;
- checkpoint count not exactly 43;
- checkpoint-on/off semantic drift above `atol=rtol=1e-5`;
- production/source edit in diagnostic slice;
- untracked verdict test/helper;
- inferred root cause from Metal error text without ablation/control evidence;
- primitive redesign or layer-serial training before the required re-entry;
- any attempt to authorize real smoke from synthetic diagnostics alone.

## Durable documentation

Amended:

- `docs/adr/0028-opaque-packed-fp4-metal-training-primitive.md` — primitive correctness retained; real-training memory path stopped; one-layer gates no longer whole-model evidence.
- `docs/technical-spec.md` — first-backward RED and mandatory multi-layer synthetic gate.

No `docs/architecture.md` change: package/backend boundary unchanged. No `docs/backlog.md` change: BA owns canonical backlog re-pin.
