# Story 13.3b-5g r2 — canonical diagnostic contract remediation

**Status:** GO for one repair rerun of the existing synthetic diagnostic only. Real-training Path A remains STOPPED. The prior serialized 64-row report is reproducible but contract-invalid and supplies no production classification authority. No production redesign, source edit, primitive change, real-asset access, smoke, or real training is authorized.

## User story

As a training engineer (WHO), I want the existing 64-cell no-model/no-shard synthetic peak diagnostic repaired to use trainer-equivalent LoRA topology, single-bound differentiated forwards, measured graph telemetry, normalized comparison groups, and load-bearing fixture provenance (WHAT), so that an Architect can later classify or reject a production direction from contract-valid evidence rather than from accidental graph construction or copied metadata (WHY).

## Authority and outcome boundary

- `agent-output/cmux-13-3b/architecture-13-3b-5g-r2-contract.md` is the binding remediation architecture.
- `agent-output/cmux-13-3b/review-13-3b-5g.md` is the accepted Reviewer RED that invalidated the prior diagnostic contract.
- This is a repair rerun of the already-authorized synthetic diagnostic, not a new architecture or production classification slice.
- The corrected report fully supersedes every old peak, node-count, and graph-memory value. No old row may be retained, patched by hand, or mixed into the corrected report.
- Even Reviewer PASS plus Test Manager GREEN authorizes only later Architect classification re-entry. It does not authorize production redesign, a primitive change, layer-serial backward, real assets, a second smoke, or real training.

## Scope

Allowed implementation assets for the later Coder slice remain exactly:

- `tests/helpers/routed_fp4_multilayer_peak_probe.py`;
- `tests/test_deepseek_v4_nn_multilayer_peak.py`;
- `tests/fixtures/deepseek_v4_nn_multilayer_peak_topology.json`;
- regenerated `agent-output/cmux-13-3b/multilayer-peak-report.json`;
- slice-local Coder, Reviewer, and Test Manager evidence.

No production, vendor, primitive, package Metal, root Metal, inference, model-loader, SSD, CUDA, ROCm, distributed, or real-training source edit is authorized. Current `deepseek_v4_nn.py`, `routed_fp4_metal.py`, and package Metal kernels remain consumers under test, not implementation targets.

No model, shard, dataset, external config, real model directory, site-packages edit, second smoke, shorter fallback, or full training run is authorized. Each measurement child retains the exact `8_000_000_000`-byte MLX limit, timeout at most `180` seconds, and serial execution with at most one child alive at a time.

## Exact 64-row matrix and ordering

The matrix does not expand or shrink. Ordering is load-bearing:

1. Probe B checkpoint-on, depth order `D=[1,2,4,8,16,43]`: 6 rows.
2. Probe B checkpoint-off, depth order `D=[1,4,8]`: 3 rows.
3. Probe C, outer order `D=[1,8,16,43]`, then `E=[2,8,32,128,256]`, then `case=[probe_c_routed_one_graph,probe_c_routed_sequential]`: 40 rows.
4. Probe D, outer order `component_mask=[full,no_routed,no_shared,no_attention,routed_only]`, then `compression_ratio=[0,4,128]`: 15 rows.

Total: `6 + 3 + 40 + 15 = 64`.

The existing 25-key row schema remains exact. Additional proof belongs in top-level metadata and aligned `cell_evidence`, never extra row-local keys.

## Probe A — exact checkpoint parity remediation

Probe A remains outside the 64 measurement rows and runs in a fresh subprocess with subprocess-confined class restoration.

Required evidence:

1. `layer_count`, `pipeline_layer_count`, `decoder_layer_count`, and checkpointed forward calls all equal `43`.
2. Wrapper name remains `checkpointed_fn`.
3. Checkpoint-off and checkpoint-on gradient trees are flattened; both key sets are non-empty and exactly equal before numerical comparison.
4. Every corresponding gradient shape matches.
5. Off/on key counts and SHA-256 of newline-joined sorted keys are recorded and equal.
6. Loss finiteness and gradient-tree finiteness are recorded separately for each mode.
7. Every matching gradient passes `atol=rtol=1e-5`; intersection-only comparison is forbidden.
8. Maximum absolute and relative differences remain summary fields only after exact key/shape checks.
9. `DecoderLayerNN.__call__` is restored in `finally`, and fresh-process ownership of the monkeypatch is proven.

## Probe B/D — trainer-equivalent trainable topology

Every Probe B and Probe D child completes this sequence before forward or differentiated graph construction:

1. Build the synthetic `Model` from validated configuration. Probe D must use fixture-derived configuration.
2. Fill the synthetic packed routed-expert payload.
3. Quantize applicable base linear leaves with `mlx.nn.quantize(model, group_size=32, bits=4)` without altering the already packed/frozen routed-expert contract.
4. Call `model.freeze()`.
5. Call `mlx_lm.tuner.utils.linear_to_lora_layers` exactly once with `num_layers=min(D,16)`, `rank=8`, `scale=20.0`, `dropout=0.0`, and keys exactly `self_attn.q_a_proj`, `self_attn.q_b_proj`, `self_attn.kv_proj`.

Let `L=min(D,16)` and `first=D-L`. The flattened trainable key set must equal exactly every:

```text
model.layers.{i}.self_attn.{projection}.{leaf}
```

where `i in range(first,D)`, `projection in [q_a_proj,q_b_proj,kv_proj]`, and `leaf in [lora_a,lora_b]`.

Consequences:

- exact trainable leaf count is `6 * min(D,16)`;
- no embedding, norm, hyperconnection, output projection, router gate, routed expert, shared expert, `lm_head`, compressor, or indexer leaf is trainable;
- every expected key appears in the differentiated LoRA gradient tree, including ablations whose value may legitimately be zero;
- no unexpected gradient key is permitted;
- for logical base dimensions `[out_dim,in_dim]`, `lora_a.shape == [in_dim,8]` and `lora_b.shape == [8,out_dim]`.

Each model child records expected/observed trainable-key count and SHA-256 of newline-joined sorted keys. Any mismatch is a captured cell error; measurement must not continue under the wrong topology.

## Single-bound differentiated forwards

Every differentiated loss binds its forward result once and reuses it. Duplicate expressions equivalent to `model(ids) * model(ids)`, `chain(z) * chain(z)`, or `layer(z) * layer(z)` are forbidden.

Separate forward-only and differentiated measurements may each construct one forward. The prohibition applies to duplicate forward construction inside a single loss graph.

## Differentiated input path for Probe B/D

Probe B/D differentiate the exact LoRA tree plus one separate synthetic `input_probe`. The input probe is not added to `model.trainable_parameters()` and is not a trainable-key exception.

After normal token embedding and HC expansion, inject before the first decoder layer:

```text
h = h + (input_probe - stop_gradient(input_probe))
```

`input_probe` has exact shape `[1,T,hc_mult,H]` and the same floating dtype as `h`. Its forward contribution is exactly zero and its derivative is identity.

Every successful Probe B/D cell records an input gradient with exact input-probe shape, finite values, and at least one non-zero value. `finite_gradients=true` only when both the exact LoRA gradient tree and input gradient are finite.

A focused MLX 0.31.2 test must prove exact zero forward contribution, preserved shape/dtype, and non-zero identity input gradient.

## Probe D — component execution contract

Test-only masking must preserve the registered model parameter tree and exact LoRA paths.

- `full`: existing attention and combined routed/shared MLP; numerically equal to ordinary model forward with zero-valued input probe.
- `no_attention`: skip attention; substitute exact zero-valued identity-gradient attention output; execute routed and shared MLP normally.
- `no_routed`: execute shared expert only; substitute exact zero-valued identity-gradient routed output.
- `no_shared`: execute current routed path directly; substitute exact zero-valued identity-gradient shared output.
- `routed_only`: substitute zero-valued identity-gradient attention and shared outputs; execute only current routed path.

Every replacement preserves expected `[B,S,H]` shape and dtype. The direct routed branch uses current `SparseMoeBlockNN` routing methods, `_host_unique_rows_per_expert`, and `routed_fp4`; it must not reimplement FP4 math, routing scores, or kernels.

Focused tests compare the direct routed result with `full_mlp_output - shared_output` on a bounded deterministic fixture within existing tolerance, verify shape/dtype, and assert the LoRA key set before and after every mask.

## Measured custom-kernel nodes

`custom_kernel_nodes` is non-null for every successful measurement row and is measured from the actual differentiated graph, never assigned from an arithmetic placeholder.

Before `mx.eval`, export all outputs that drive backward: value plus every flattened LoRA gradient and input gradient for Probe B/D; value plus input gradient for Probe C. Count only DOT node lines containing `CustomKernel`. Query `active_after_graph` before DOT export. Child-local DOT files are deleted after counting.

Metadata pins:

```text
custom_kernel_nodes_scope = differentiated graph: value plus evaluated gradient leaf
custom_kernel_nodes_method = mx.export_to_dot; count DOT node lines containing CustomKernel
```

For Probe C, with `N=nonempty_experts`, measured counts must equal:

- one-graph: `6 * D * N`;
- sequential total across separately exported stage graphs: `6 * D * N`.

A mismatch is a captured cell error. Probe B/D counts remain measured and may include custom kernels outside the routed branch.

## Measured `active_after_graph` boundary

`active_after_graph` means active MLX memory immediately after constructing the unevaluated differentiated graph and immediately before DOT export or `mx.eval`. It may not be copied from baseline or measured after setup/forward evaluation.

Before `active_baseline`, materialize all resident synthetic model, payload, and input state, evaluate resident arrays, clear cache, then query active memory. Baseline includes resident synthetic state and no pending forward/backward graph.

For Probe B/C/D one-graph cells:

1. reset peak memory;
2. construct value and all requested gradients from a single-bound loss;
3. query `active_after_graph` immediately;
4. export and count nodes;
5. evaluate value and every gradient leaf;
6. record `peak_backward` from the same construction/evaluation interval.

Forward peak is separate: reset peak, construct one forward, evaluate it, record `peak_forward`.

For sequential Probe C, each of `D` stages resets peak, constructs one per-layer differentiated graph, queries active memory before export/evaluation, exports/counts, evaluates, and clears at the specified boundary. The row records the maximum active-after-graph and peak values across stages and sums measured node counts. Sequential control makes no end-to-end mathematical-equivalence claim.

## Probe C — route and comparison contract

Pinned cells retain `T=64`, `H=64`, `I=32`, and `K=min(6,E)`. Deterministic route construction remains:

```text
eid = (token_index * K + slot_index) mod E
```

Comparison groups are exact:

| Group | Experts | K | Assignments `T*K` | Authority |
|---|---:|---:|---:|---|
| low-E control | 2 | 2 | 128 | non-comparable to the E>=8 expert-count axis |
| normalized expert-count axis | 8,32,128,256 | 6 | 384 | comparable only within the same D and execution mode |

Rules:

1. Keep every `E=2` one-graph and sequential row.
2. `E=2` may be compared only with its matched execution-mode control at the same `D,E,K`, or across depth while `E=2,K=2` remain fixed.
3. Expert-count claims use only `E=[8,32,128,256]`, with exactly 384 assignments.
4. Comparisons do not cross depth or execution mode except an explicit depth or matched one-graph/sequential analysis.
5. `E=2` is never plotted, fitted, or cited in an expert-count trend with `E>=8`.
6. Every expert is non-empty when `T*K>=E`, and `nonempty_experts` equals the deterministic route result.

Top-level metadata contains an exact machine-readable `probe_c_comparison_contract` with route formula, low-E declaration, normalized experts, K/assignment counts, and allowed grouping keys. Tests compare it exactly. No production classification may use the old mixed-volume `E=2` to `E>=8` relationship.

## Load-bearing Probe D topology fixture

Probe D consumes, validates, and derives configuration from `tests/fixtures/deepseek_v4_nn_multilayer_peak_topology.json` before constructing any MLX model.

Required top-level keys are exactly:

```text
schema_version, story, purpose, real_assets_accessed, model_defaults, provenance, layers
```

Required constants are integer `schema_version=1`, `story="13.3b-5g"`, `real_assets_accessed=false`, and exactly 43 layers.

Every layer object has exactly:

```text
index, layer_type, hidden_size, intermediate_size, experts, top_k, hc_mult
```

Validation requires indices `0..42` in order; `layer_type="moe"`; positive integer, non-boolean shape/topology values; `top_k<=experts`; valid FP4 block/attention constructor dimensions; uniform fields required by current `ModelArgs`; internally valid `model_defaults`; and no path or provenance source outside the repository or naming a real model/shard/dataset/config location.

`model_defaults` carries every non-layer-varying `ModelArgs` field Probe D needs, including model type, vocabulary/tokens, hash-layer count, shared-expert count, expert dtype, attention dimensions/counts, LoRA ranks, output groups, RoPE head dimension, sinkhorn iterations, scoring function, routed scale, SwiGLU limit, and RMS epsilon. Compression ratio remains the report axis.

Probe D derives `num_hidden_layers`, `mlp_layer_types`, `hidden_size`, `moe_intermediate_size`, `n_routed_experts`, `num_experts_per_tok`, and `hc_mult` from validated layer entries. All remaining constructor values come from validated `model_defaults`. The old hard-coded Probe D configuration path is forbidden.

Fixture provenance records repository-relative path plus lowercase SHA-256 for exactly:

Durable authority:

- `docs/technical-spec.md` section 10.12;
- `docs/adr/0028-opaque-packed-fp4-metal-training-primitive.md` Story 13.3b-5g amendment.

Direct synthetic derivation evidence:

- `agent-output/cmux-13-3b/architecture-13-3b-5g-first-backward-oom.md`;
- `agent-output/cmux-13-3b/requirements-13-3b-5g-synthetic-peak.md`.

The fixture records a plain derivation map. The helper recomputes source hashes before use. Report metadata records fixture path, exact-byte fixture SHA-256, schema version, layer count, validated configuration derivation, provenance object, and source hashes. Tests recompute all hashes independently.

Malformed temporary fixtures must fail before model construction for at least missing key, discontinuous index, wrong type, invalid shape, inconsistent layer, and provenance/hash mismatch.

## Report and execution evidence

Every row retains exactly these 25 keys, with explicit `null` for non-applicable values:

```text
case, depth, experts, nonempty_experts, tokens, hidden, intermediate, top_k,
hc_mult, compression_ratio, checkpoint, component_mask,
active_baseline, active_after_graph, peak_forward, peak_backward,
active_final, cache_final, custom_kernel_nodes, duration_s,
finite_loss, finite_gradients, exit_code, signal, error
```

Top-level `cell_evidence` has exactly one aligned entry per row containing:

```text
ordinal, cell_identity_sha256, child_pid, child_nonce,
started_monotonic_ns, finished_monotonic_ns,
trainable_key_count, trainable_keys_sha256,
input_gradient_shape, input_gradient_nonzero,
measured_custom_kernel_nodes
```

Probe C uses null trainable-key fields. Model cells use exact LoRA evidence. `measured_custom_kernel_nodes` equals the row field.

The parent uses fresh `Popen` children and unique nonces. Metadata records parent PID plus:

```text
execution_mode = serial_fresh_subprocess_per_cell
max_concurrency = 1
planned_cell_count = 64
executed_cell_count = 64
```

Required proof:

- 64 evidence entries align one-to-one with the exact ordered rows;
- every child PID differs from parent PID;
- every nonce is unique;
- every interval has `start < finish`;
- `start[i] >= finish[i-1]` proves no overlap;
- planned, row, and evidence identity SHA-256 values match at every ordinal;
- planned-cell-list SHA-256 equals executed-row-identity-list SHA-256.

Probe A has separate fresh-child execution evidence outside the 64 rows.

## Child outcome normalization

Each row has exactly one normalized process outcome:

- success: `exit_code=0`, `signal=null`, `error=null`, and all required telemetry populated;
- handled child failure: positive `exit_code`, `signal=null`, bounded non-empty error;
- signal termination: `exit_code=null`, canonical signal name, bounded non-empty error;
- timeout: `exit_code=null`, `signal="TIMEOUT"`, bounded non-empty error.

`exit_code` and `signal` are never both set. Error is UTF-8 text bounded to 500 characters. Non-success rows may null measurement/finiteness fields but remain in the matrix. Parent always emits valid JSON with all 64 rows, and parsed child output cannot override the parent-observed return code or signal.

## Acceptance criteria

All are required:

1. Exact 64-row plan, exact ordering, exact dimensions, exact 25-key row schema, uniqueness, and no extras are tested and reported.
2. Probe B/D use quantize → freeze → exactly one rank-8 LoRA conversion with repository defaults and exact last-`min(D,16)` q_a/q_b/kv key set, `6L` leaves, correct shapes, and no forbidden trainables.
3. Every model, one-graph routed, and sequential routed differentiated loss binds one forward exactly once.
4. Probe B/D differentiate the exact LoRA tree plus the zero-valued identity-gradient input probe; every successful cell has exact-shape, finite, non-zero input gradient.
5. Every successful row has DOT-derived `custom_kernel_nodes`; Probe C measured counts equal `6*D*N` for one-graph and sequential totals.
6. `active_after_graph` is measured immediately after differentiated graph construction; sequential cells use the pinned per-stage/max aggregation and never copy baseline.
7. Probe C keeps all `E=2` rows as non-comparable low-E controls and permits expert-count claims only on `E=8/32/128/256`, `K=6`, exactly 384 assignments, within matched depth/execution mode.
8. Probe D consumes and validates the exact 43-entry fixture, derives configuration from it, verifies exact-byte fixture and canonical-source hashes, records provenance/derivation metadata, and rejects malformed copies before model construction.
9. Probe D full/masked execution preserves shape, dtype, parameter registration, identity-gradient paths, full-path equality, and current routed-branch parity without reimplementing production math.
10. Top-level evidence proves fresh child per cell, unique nonce, exact serial non-overlap, identity-hash alignment, planned/executed count `64`, and maximum concurrency `1`.
11. Exit/signal/timeout/error normalization preserves every captured-failure row and parent-observed process result.
12. Probe A proves exact 43-layer/checkpoint coverage, non-empty equal gradient keys/shapes/hashes, separate loss/gradient finiteness, and full-key parity at `atol=rtol=1e-5`.
13. Behavioral tests fail against the Reviewer-RED implementation and pin every corrected invariant; the fully regenerated report supersedes all old numeric values.
14. Helper, fixture, focused test, and report are tracked by `git ls-files`; focused and tracked baseline are GREEN; `git diff --check` is clean; direct hashes prove protected source byte integrity.
15. No real assets, real model directory, site-packages edit, second smoke, shorter fallback, full training, production redesign/classification, or production/vendor/primitive/Metal/inference/SSD/CUDA/ROCm/distributed/model-loader edit occurs.

## Immediate STOP gates

STOP and return to Architect on any of:

- row count, identity, order, dimension, or 25-key schema drift;
- omitted or reduced pinned cell, including `D=43`, `E=256`, or any retained `E=2` control;
- topology mismatch, unexpected trainable/gradient key, wrong rank/shape, or measurement continuing after mismatch;
- duplicate forward construction in one differentiated loss;
- copied/fabricated `active_after_graph` or arithmetic-placeholder node count;
- Probe C node-count mismatch or mixed-volume `E=2`/`E>=8` expert-count claim;
- fixture schema/config/provenance/hash failure or hard-coded Probe D configuration;
- missing, overlapping, reused-child, or inconsistent cell evidence;
- unnormalized, omitted, or child-overridden process outcome;
- Probe A count/key/shape/finiteness/parity failure;
- any pinned timeout, OOM, signal, or non-zero child result;
- memory limit above `8_000_000_000` bytes, timeout above `180` seconds, or concurrent children;
- any untracked verdict helper/test/fixture/report;
- any real asset load, real model directory, site-packages edit, second smoke, shorter fallback, or full training;
- any production/vendor/primitive/Metal/inference/model-loader/SSD/CUDA/ROCm/distributed source edit;
- any production root-cause classification, redesign authorization, layer-serial backward, or smoke authorization from this diagnostic.

A STOP is captured evidence. It does not authorize a smaller matrix, reduced dimensions, omitted cell, concurrency, larger memory limit, shorter fallback, real assets, production repair, or smoke.

## Review and handoff gates

- Coder: TDD RED → GREEN within the narrow diagnostic asset boundary, then regenerate all 64 rows from scratch under serial/8 GB/180-second guards.
- Reviewer: independently verify semantic contract, actual graph measurements, matrix/evidence completeness, comparison validity, provenance, tracking, and protected hashes; required verdict PASS.
- Test Manager: independently rerun focused and tracked baseline checks, validate report and process evidence, and verify every cited helper/test/fixture/report with `git ls-files`; required verdict GREEN.
- Architect: only after double-GREEN, re-enter for classification. No production redesign, primitive change, layer-serial backward, real asset, second smoke, or real training is pre-authorized.
