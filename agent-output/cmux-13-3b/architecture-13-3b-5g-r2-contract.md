# Story 13.3b-5g r2 — diagnostic contract remediation

**Decision: repair and rerun the existing diagnostic only. Keep the 64-row matrix. Do not classify the OOM, redesign production, access real assets, or run a smoke.**

Reviewer RED is accepted in full. The serialized 64 rows are reproducible but not contract-valid because Probe B/D did not use the trainer LoRA topology, differentiated losses duplicated each forward, Probe C copied rather than measured `active_after_graph`, the `E=2` cell was implicitly compared against a different assignment volume, and the 43-entry fixture was not load-bearing.

This document is the complete implementation contract for the remediation slice. It refines the synthetic diagnostic only. It does not change the package/backend architecture, ADR 0028 primitive semantics, or the real-training STOP.

## Scope and authority

Allowed edits remain limited to:

- `tests/helpers/routed_fp4_multilayer_peak_probe.py`;
- `tests/test_deepseek_v4_nn_multilayer_peak.py`;
- `tests/fixtures/deepseek_v4_nn_multilayer_peak_topology.json`;
- regenerated `agent-output/cmux-13-3b/multilayer-peak-report.json`;
- slice-local coder/review/test evidence.

No production, vendor, primitive, package Metal, root Metal, inference, model-loader, SSD, CUDA, ROCm, distributed, or real-training source edit is authorized. The current `deepseek_v4_nn.py`, `routed_fp4_metal.py`, and package Metal kernels are consumers under test, not implementation targets.

No model, shard, dataset, external config, real model directory, site-packages edit, second smoke, shorter fallback, or full training run is authorized. The exact limits remain `8_000_000_000` bytes, at most `180` seconds per child, and one child at a time.

## Matrix decision: retain exactly 64 rows

The matrix does not expand.

Exact order is load-bearing:

1. Probe B checkpoint-on, in depth order `D=[1,2,4,8,16,43]`: 6 rows.
2. Probe B checkpoint-off, in depth order `D=[1,4,8]`: 3 rows.
3. Probe C, outer order `D=[1,8,16,43]`, then `E=[2,8,32,128,256]`, then `case=[probe_c_routed_one_graph, probe_c_routed_sequential]`: 40 rows.
4. Probe D, outer order `component_mask=[full,no_routed,no_shared,no_attention,routed_only]`, then `compression_ratio=[0,4,128]`: 15 rows.

Total: `6 + 3 + 40 + 15 = 64`.

The report row schema remains exactly the already pinned 25 keys and adds no row-local fields. Additional proof belongs in top-level metadata/cell evidence so existing schema consumers remain stable.

## Probe B and Probe D trainer-equivalent trainable topology

Every Probe B and Probe D child must create the same trainable topology before constructing either the forward or differentiated graph.

### Setup sequence

1. Construct the synthetic `Model` from the cell or validated fixture-derived config.
2. Fill the synthetic packed routed-expert payload.
3. Quantize applicable base linear leaves with `mlx.nn.quantize(model, group_size=32, bits=4)`. This mirrors the 4-bit base presented to the real trainer; it must not alter the already packed/frozen routed-expert contract.
4. Call `model.freeze()`.
5. Call `mlx_lm.tuner.utils.linear_to_lora_layers` once with:

```python
num_layers = min(depth, 16)
config = {
    "rank": 8,
    "scale": 20.0,
    "dropout": 0.0,
    "keys": {
        "self_attn.q_a_proj",
        "self_attn.q_b_proj",
        "self_attn.kv_proj",
    },
}
```

`rank=8`, `scale=20.0`, and `dropout=0.0` are the repository's real-training defaults, not the smaller values used by unrelated tiny tests.

### Exact trainable-key contract

Let `L=min(D,16)` and `first=D-L`. The flattened trainable key set must equal, not merely contain:

```text
model.layers.{i}.self_attn.{projection}.{leaf}
```

for every:

- `i in range(first, D)`;
- `projection in [q_a_proj, q_b_proj, kv_proj]`;
- `leaf in [lora_a, lora_b]`.

Therefore:

- exact trainable leaf count is `6 * min(D,16)`;
- no embedding, norm, hyperconnection, output projection, router gate, routed expert, shared expert, `lm_head`, compressor, or indexer leaf may be trainable;
- every expected key must also appear in the differentiated LoRA gradient tree, including ablations where its value may legitimately be zero;
- no unexpected gradient key is permitted.

For each converted projection with logical base dimensions `[out_dim, in_dim]`:

```text
lora_a.shape == [in_dim, 8]
lora_b.shape == [8, out_dim]
```

The child records expected/observed key count and SHA-256 of the newline-joined sorted keys in aligned top-level cell evidence. A mismatch is a captured cell error and invalidates the report; measurement must not continue under the wrong topology.

## Bind each forward exactly once

Every differentiated loss must assign its forward result once and reuse it.

Model cells:

```python
def loss_for(params, input_probe):
    model.update(params)
    y = forward_once(model, ids, input_probe, component_mask)
    return mx.mean(y * y)
```

Probe C one-graph:

```python
def loss_for(z):
    y = chain(z)
    return mx.sum(y * y)
```

Probe C sequential stage:

```python
def layer_loss(z):
    y = layer(z)
    return mx.sum(y * y)
```

Expressions such as `model(ids) * model(ids)`, `chain(z) * chain(z)`, or `layer(z) * layer(z)` are forbidden. Separate forward-only and differentiated measurements may each construct their own single forward; the prohibition is duplicate construction inside one loss graph.

## Differentiated input path for Probe B/D

Probe B/D retain the exact LoRA trainable tree above and additionally differentiate one separate synthetic input probe. The input probe is not inserted into `model.trainable_parameters()` and is not a trainable-key exception.

Use the normal token embedding and HC expansion, then inject a zero-valued identity-gradient term before the first decoder layer:

```python
h = broadcast_embedded_tokens
h = h + (input_probe - mx.stop_gradient(input_probe))
```

`input_probe` has exactly the HC residual-stream shape `[1,T,hc_mult,H]` and the same floating dtype as `h`. Its forward contribution is exactly zero; its derivative is identity. Differentiate `(lora_params, input_probe)` together. This forces the activation backward through the selected component topology, including `no_attention` and `routed_only`, without adding a fake model parameter or changing the forward value.

Every successful Probe B/D cell must report in aligned cell evidence that the input gradient has the exact input-probe shape, is finite, and has at least one non-zero value. `finite_gradients` is true only when both the exact LoRA gradient tree and input gradient are finite.

A focused test must prove that the zero-valued identity-gradient expression has exact zero forward value, preserves shape/dtype, and yields a non-zero identity input gradient under MLX 0.31.2.

## Probe D component execution

Do not replace registered model modules in a way that removes or renames LoRA leaves. Keep the model parameter tree intact and implement test-only masked decoder execution around the existing modules.

The `full` masked executor must be numerically identical to the ordinary model forward for a zero-valued input probe.

Each removed branch returns an exact zero-valued tensor with the branch's expected `[B,S,H]` shape and dtype while preserving an input-gradient path with `x - stop_gradient(x)`:

- `no_attention`: skip the attention call; use zero-valued identity-gradient attention output; execute routed plus shared MLP normally.
- `no_routed`: execute shared expert only; add the zero-valued identity-gradient routed replacement.
- `no_shared`: execute the existing routed path directly; add the zero-valued identity-gradient shared replacement.
- `routed_only`: use the zero-valued identity-gradient attention replacement and execute only the existing routed path, with the shared replacement zero-valued.
- `full`: execute attention and the existing combined routed/shared MLP unchanged.

The direct routed branch must use the current `SparseMoeBlockNN` routing methods, `_host_unique_rows_per_expert`, and `routed_fp4`; it must not reimplement FP4 math, routing scores, or kernels. Tests must compare the test-only routed branch against `full_mlp_output - shared_output` on a bounded deterministic fixture within the existing numerical tolerance and must verify output shape/dtype.

The registered LoRA key set is asserted before and after selecting the mask. No mask may mutate module registration or parameter paths.

## Actual custom-kernel node measurement

`custom_kernel_nodes` is no longer an arithmetic placeholder and is non-null for every successful measurement row.

After constructing the unevaluated differentiated outputs and before `mx.eval`, export the graph with all outputs that drive backward:

```python
mx.export_to_dot(path, value, *flattened_lora_gradients, input_gradient)
```

Probe C exports `value` and the input gradient. Count graph nodes as the number of DOT node lines containing `CustomKernel`; do not count a formula, source registrations, or substring occurrences outside node lines. Query `active_after_graph` before DOT export so DOT traversal cannot affect the memory boundary.

The metadata pins:

```text
custom_kernel_nodes_scope = differentiated graph: value plus every evaluated gradient leaf
custom_kernel_nodes_method = mx.export_to_dot; count DOT node lines containing CustomKernel
```

For Probe C, deterministic routing makes the count independently predictable. Let `N=nonempty_experts`:

- one `routed_fp4` differentiated invocation constructs two forward plus four VJP custom kernels per non-empty expert;
- one-graph expected count is exactly `6 * D * N`;
- sequential expected total count is the sum of the `D` separately exported stage graphs and is also exactly `6 * D * N`;
- the simultaneous graph distinction is carried by the case and telemetry, not by pretending the sequential total is concurrently resident.

A mismatch between measured and expected Probe C node count is a captured error. Probe B/D counts are measured, not assigned a formula, because their graphs may contain custom kernels outside the routed branch.

Temporary DOT files remain child-local and are deleted after counting. They are diagnostic intermediates, not new report assets.

## Real `active_after_graph` boundary

`active_after_graph` always means active MLX memory immediately after construction of the unevaluated differentiated graph and immediately before DOT export or `mx.eval`. It never means post-model setup, post-forward evaluation, or a copy of baseline.

### Common baseline

Before recording `active_baseline`, materialize the synthetic model/payload/input state needed by the cell, call `mx.eval` on resident arrays, clear cache, and then query active memory. Thus baseline includes resident synthetic state but no pending forward or backward graph.

### One-graph Probe B/C/D

1. Reset peak memory.
2. Construct `value` and all requested gradients with the single-bound loss.
3. Query `active_after_graph` immediately.
4. Export/count nodes.
5. Evaluate `value` and every gradient leaf.
6. Record `peak_backward` from that same construction/evaluation interval.

Forward peak is measured separately: reset peak, construct one forward once, evaluate it, and record `peak_forward`.

### Sequential Probe C

For each of the `D` stages:

1. reset peak memory;
2. construct one per-layer differentiated graph;
3. query active memory before export/evaluation;
4. export/count that stage's nodes;
5. evaluate value/gradient;
6. delete graph references, carry only the evaluated stage output, and clear cache.

The row records:

- `active_after_graph = max(per_stage_active_after_graph)`;
- `peak_forward = max(per_stage_forward_peak)`;
- `peak_backward = max(per_stage_backward_peak)`;
- `custom_kernel_nodes = sum(per_stage_measured_nodes)`.

Metadata must state these aggregation rules verbatim. Sequential remains a memory-only control and makes no end-to-end mathematical-equivalence claim.

## Probe C assignment/comparison contract

The pinned cells and `K=min(6,E)` remain unchanged. The conflict is resolved by separating comparison authority, not by dropping or resizing a cell.

Deterministic route construction remains:

```text
eid = (token_index * K + slot_index) mod E
```

Exact groups:

| Group | Experts | K | Total assignments `T*K` | Comparison authority |
|---|---:|---:|---:|---|
| low-E control | 2 | 2 | 128 | non-comparable to E>=8 expert-count axis |
| normalized expert-count axis | 8,32,128,256 | 6 | 384 | comparable within same D and same execution mode |

Rules:

1. Keep every `E=2` one-graph and sequential row. It is a bounded low-E control and may be compared only with its matched execution-mode control at the same `D,E,K` or across depth while `E=2,K=2` stay fixed.
2. Expert-count claims use only `E=[8,32,128,256]`, where `T=64`, `K=6`, and assignments are exactly `384`.
3. Comparisons never cross depth or execution mode unless explicitly evaluating depth or one-graph-versus-matched-sequential behavior.
4. `E=2` is never plotted, fitted, or cited as part of an expert-count trend with `E>=8`.
5. Every expert must be non-empty when `T*K>=E`; the report's `nonempty_experts` must equal the deterministic route result.

Metadata adds a machine-readable `probe_c_comparison_contract` containing the route formula, the `E=2` control declaration, normalized experts, exact K/assignment counts, and allowed grouping keys. Tests compare it exactly.

No production classification may use the old mixed-volume `E=2 -> E>=8` relationship.

## Load-bearing 43-entry topology fixture

Probe D must consume, validate, and derive its model config from `tests/fixtures/deepseek_v4_nn_multilayer_peak_topology.json`.

### Fixture schema

Top-level schema version is integer `1`. Required top-level keys are exactly:

```text
schema_version, story, purpose, real_assets_accessed, model_defaults, provenance, layers
```

Required constants:

```text
story = "13.3b-5g"
real_assets_accessed = false
len(layers) = 43
```

`model_defaults` carries every non-layer-varying `ModelArgs` field needed by Probe D, including model type, vocabulary/tokens, hash-layer count, shared-expert count, expert dtype, attention head dimensions/counts, LoRA ranks, output groups, RoPE head dimension, sinkhorn iterations, scoring function, routed scale, SwiGLU limit, and RMS epsilon. Compression ratio remains the report axis and is not silently taken from the fixture.

Every layer object has exactly:

```text
index, layer_type, hidden_size, intermediate_size, experts, top_k, hc_mult
```

Validation requires:

- `index` is an integer and indices are exactly `0..42` in array order;
- `layer_type` is exactly `"moe"` for this fixture;
- all numeric shape/topology fields are integers, not booleans, and strictly positive;
- `top_k <= experts`;
- hidden/intermediate dimensions satisfy the current FP4 block/attention constructor constraints;
- the uniform fields required by current `ModelArgs` are equal across all 43 entries;
- `model_defaults` attention/head/group dimensions are internally valid for the layer hidden size;
- no path or provenance source escapes repository root or names a real model/shard/dataset/config location.

Unknown/missing keys, duplicate/skipped indices, wrong types, invalid shapes, hash mismatch, or inconsistent layers fail before any MLX model is constructed.

### Config derivation

Probe D derives, rather than hard-codes:

```text
num_hidden_layers = len(layers)
mlp_layer_types = [layer.layer_type for layer in layers]
hidden_size = validated common layer.hidden_size
moe_intermediate_size = validated common layer.intermediate_size
n_routed_experts = validated common layer.experts
num_experts_per_tok = validated common layer.top_k
hc_mult = validated common layer.hc_mult
```

All other Probe D constructor values come from validated `model_defaults`; only the cell's `compression_ratio` is applied as the matrix axis. Probe D report dimensions must equal the derived values. The old hard-coded Probe D `_config(...)` path is forbidden.

### Provenance and hashes

`provenance` distinguishes durable authority from direct synthetic derivation evidence and contains repository-relative path plus lowercase SHA-256 for exactly:

Durable authority:

- `docs/technical-spec.md` — section 10.12;
- `docs/adr/0028-opaque-packed-fp4-metal-training-primitive.md` — Story 13.3b-5g amendment.

Direct shape/matrix derivation evidence:

- `agent-output/cmux-13-3b/architecture-13-3b-5g-first-backward-oom.md`;
- `agent-output/cmux-13-3b/requirements-13-3b-5g-synthetic-peak.md`.

The fixture also records a plain derivation map from those pins to the 43 shape-reduced entries. The helper recomputes every source hash before use. Report metadata records:

- repository-relative fixture path;
- SHA-256 of exact fixture bytes;
- schema version;
- layer count;
- validated config derivation;
- the validated provenance object and source hashes.

Tests recompute all hashes independently. Merely reporting `len(fixture["layers"])` is insufficient.

A load-bearing consumption test must show that the Probe D derived config comes from validated fixture content, and malformed temporary copies must fail for at least missing key, discontinuous index, wrong type, invalid shape, inconsistent layer, and provenance/hash mismatch cases.

## Report-level execution evidence

Keep the exact row schema and add one aligned top-level `cell_evidence` entry per row. Each evidence entry includes:

```text
ordinal, cell_identity_sha256, child_pid, child_nonce,
started_monotonic_ns, finished_monotonic_ns,
trainable_key_count, trainable_keys_sha256,
input_gradient_shape, input_gradient_nonzero,
measured_custom_kernel_nodes
```

Probe C uses null trainable-key fields. Model cells use the exact LoRA evidence. `measured_custom_kernel_nodes` must equal the row field.

The parent must use a fresh `Popen` child per cell so the child PID is observable. Generate a unique nonce per launch and have the child echo it outside the row schema. Metadata records parent PID and:

```text
execution_mode = "serial_fresh_subprocess_per_cell"
max_concurrency = 1
planned_cell_count = 64
executed_cell_count = 64
```

Tests require:

- 64 evidence entries aligned one-to-one with the 64 exact ordered rows;
- each child PID differs from the parent PID;
- each nonce is unique;
- each interval has `start < finish`;
- `start[i] >= finish[i-1]`, proving no overlap;
- planned identity SHA-256 equals row/evidence identity SHA-256 at every ordinal;
- planned-cell-list SHA-256 equals executed-row-identity-list SHA-256.

Probe A has a separate fresh-child execution record and remains outside the 64 measurement rows.

## Exit, signal, timeout, and error consistency

Normalize every row to exactly one process outcome:

- success: `exit_code=0`, `signal=null`, `error=null`, all required telemetry populated;
- handled child failure: positive `exit_code`, `signal=null`, bounded non-empty error;
- signal termination: `exit_code=null`, canonical signal name, bounded non-empty error;
- timeout: `exit_code=null`, `signal="TIMEOUT"`, bounded non-empty error;
- never set both `exit_code` and `signal`;
- error is UTF-8 text bounded to 500 characters;
- any non-success may use null measurement/finiteness fields but may not disappear from the matrix;
- parent always emits valid JSON with all 64 rows.

A parsed child row cannot override the actual parent-observed process return code/signal. Tests cover success, handled non-zero, signal, timeout, malformed child output, and report-row consistency without waiting 180 seconds.

## Probe A remediation

Probe A retains exact 43-layer checkpoint coverage and subprocess-confined class restoration. Strengthen parity as follows:

1. Flatten checkpoint-off and checkpoint-on gradient trees.
2. Require both key sets to be non-empty and exactly equal before calculating any numerical difference.
3. Require every corresponding shape to match.
4. Record off/on key counts and SHA-256 of newline-joined sorted keys.
5. Record loss finiteness separately from gradient-tree finiteness; do not derive both flags from one combined boolean.
6. Compare every matching gradient at `atol=rtol=1e-5`; intersection-only comparison is forbidden.
7. Preserve the reported max absolute/relative differences as summaries after exact key/shape checks.
8. Restore `DecoderLayerNN.__call__` in `finally` and prove a fresh process owns the monkeypatch.

Tests require `layer_count`, `pipeline_layer_count`, `decoder_layer_count`, and checkpointed forward calls all equal `43`; wrapper name remains `checkpointed_fn`; both losses and both gradient trees are finite; key sets/hashes/counts match and are non-empty; all gradient comparisons pass.

## Required focused tests

Coder uses TDD and adds behavioral tests that fail against the current implementation. At minimum, tests must pin:

1. Exact 64-row plan, exact ordering, exact dimensions, exact row-key set, uniqueness, and no extras.
2. Exact Probe B/D quantize-freeze-LoRA sequence outcome, last-`min(D,16)` keys, `6L` count, rank-8 shapes, and forbidden non-target keys.
3. Single-bound forward construction in model, one-graph routed, and sequential routed differentiated paths.
4. Actual DOT-derived custom-kernel counts; Probe C exact `6*D*N` counts on every successful row.
5. Real `active_after_graph` query after differentiated graph construction and sequential max aggregation, never baseline assignment.
6. Exact Probe C route formula, non-empty expert counts, `E=2` control metadata, `384`-assignment normalized axis, and forbidden mixed-volume comparisons.
7. Fixture top-level/layer schema, index continuity, types, shapes, source hashes, fixture hash, derived config, and report consumption.
8. Full/masked forward shape and dtype; full-path equality; zero-value/non-zero-input-gradient replacements; direct routed-branch parity.
9. Fresh-child boundaries, unique nonces, exact serial non-overlap evidence, plan/report identity hashes, and one evidence entry per row.
10. Exit/signal/timeout/error normalization and complete captured-failure rows.
11. Probe A exact non-empty gradient-key equality, shape equality, separate finiteness, and parity.
12. Every successful row has finite loss/gradients, non-null real telemetry, non-null measured node count, `exit_code=0`, null signal/error, and aligned evidence.
13. No real-asset path/access, exact 8 GB limit, timeout at most 180 seconds, and no concurrent children.
14. Helper, fixture, focused test, and report tracked by `git ls-files` before any pass-count claim.
15. Direct protected source hashes plus `git diff --check`; no reliance on a vacuous zero-line diff.

The regenerated report supersedes every old numeric peak. No old row may be retained, patched by hand, or mixed with the corrected run.

## Completion and review gate

Coder completion requires the corrected 64-row report under the same serial/8 GB/180-second bounds, focused GREEN, tracked baseline GREEN, direct protected hashes, and tracked-file verification.

Then Reviewer and Test Manager rerun independently in parallel. Reviewer PASS and Test Manager GREEN are both mandatory. Any missing pinned row, timeout/OOM, topology mismatch, telemetry failure, node-count mismatch, fixture/provenance failure, untracked test, or child-outcome inconsistency returns to Architect; it does not authorize reducing the matrix.

Even double-GREEN authorizes only a later Architect classification re-entry. It does not authorize a production redesign, primitive change, layer-serial backward, real asset, second smoke, or real training.

## Durable documentation decision

No durable documentation edit is required in this remediation. `docs/technical-spec.md` section 10.12 and ADR 0028 already pin the diagnostic-only gate and real-training STOP. This document corrects how that already-authorized synthetic evidence must be produced; it introduces no durable production architecture or backend boundary.
