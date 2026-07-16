# Story 13.3b-5g — independent diagnostic review

## Verdict

**FAIL / STOP — Reviewer PASS withheld.**

Synthetic report reproducible as implemented.
Binding diagnostic contract not implemented.
No production redesign, Architect classification, real-asset access, or smoke authorized.

## Blocking findings

### HIGH — Probe B does not model pinned LoRA trainability

`tests/helpers/routed_fp4_multilayer_peak_probe.py:242-270` constructs a plain `Model`, never calls `model.freeze()`, never calls `linear_to_lora_layers`, and differentiates `model.trainable_parameters()` directly.

Independent D=2 inspection:

```text
trainable_count=46
lora_count=0
q_target_base_weight_count=6
non_target_count=40
```

Non-target gradients include embeddings, norms, hyperconnections, output projections, router gate, shared experts, and LM head. All D layers remain trainable. Required contract: LoRA targets `q_a_proj,q_b_proj,kv_proj`; trainable layers `min(D,16)`. Probe B depth peaks therefore do not measure pinned topology. Probe D uses same invalid trainable tree, weakening every component comparison.

Required fix: freeze base model; attach trainer-equivalent LoRA to exactly pinned keys and last `min(D,16)` layers; assert trainable key set/count before measuring; regenerate report.

### HIGH — Backward cells build two full forward graphs while node telemetry counts one

`tests/helpers/routed_fp4_multilayer_peak_probe.py:264` evaluates `model(ids)` twice inside one loss.

`tests/helpers/routed_fp4_multilayer_peak_probe.py:327,333` evaluates `layer(z)` / `chain(z)` twice inside each differentiated loss.

These expressions create two independent model/routed graphs. They are not equivalent to `y=model(ids); mean(y*y)` or `y=chain(z); sum(y*y)` for graph-lifetime and command-buffer-node measurement.

`tests/helpers/routed_fp4_multilayer_peak_probe.py:334` still records `D * nonempty_experts * 6`. Actual differentiated Probe C graph contains two routed calls per layer. Six primitive kernels per call implies 12 per layer/expert, not 6. Example D=43/E=256 row reports `66,048`; implemented loss constructs `132,096` primitive kernel nodes.

Reported peaks and node-pressure relationship therefore describe an accidental doubled graph while labeling a single graph.

Required fix: bind each forward once inside every loss; count actual constructed primitive calls/nodes or derive count from verified graph construction; regenerate all affected rows.

### HIGH — Probe C `active_after_graph` is fabricated from baseline

`tests/helpers/routed_fp4_multilayer_peak_probe.py:313-314` assigns:

```python
row["active_after_graph"] = row["active_baseline"]
```

No graph has been constructed or measured. All 40 Probe C rows consequently report exact baseline equality. Binding schema requires active memory after graph construction, not a copied placeholder. This is load-bearing telemetry failure.

Required fix: construct relevant one-graph or per-layer graph, then query telemetry at defined boundary. Document sequential-control aggregation semantics.

### HIGH — Probe C does not hold assignments fixed across expert-count comparison

Pinned report values:

```text
E=2   -> T*K = 64*2 = 128 assignments
E=8   -> T*K = 64*6 = 384 assignments
E=32  -> 384 assignments
E=128 -> 384 assignments
E=256 -> 384 assignments
```

Canonical AC requires fixed logical dimensions/assignments while comparing expert count. Current E=2 point changes both expert count and assignment volume. `K=min(6,E)` creates a contract tension that implementation neither resolves nor discloses.

Required fix: Architect/BA re-pin exact normalization or define E=2 as a non-comparable control and add a fixed-assignment expert-count axis. Do not classify E=2 versus E>=8 from current matrix.

### HIGH — Checked-in topology fixture does not drive Probe D

Fixture read occurs only at `tests/helpers/routed_fp4_multilayer_peak_probe.py:393`; only `len(fixture["layers"])` reaches metadata at line 410. All model cells use hard-coded `_config(...)` values at line 250. Layer entries, indices, shapes, types, and topology are never validated or consumed.

Fixture provenance is a self-asserted string. No canonical source path/hash or derivation record exists. A malformed 43-entry fixture with arbitrary values would leave every measurement unchanged.

Required fix: validate fixture schema/index continuity; derive Probe D model configuration from fixture; record fixture SHA-256 and canonical provenance evidence in metadata/tests.

### MEDIUM — Focused tests are materially non-load-bearing

`tests/test_deepseek_v4_nn_multilayer_peak.py` passes despite every blocker above.

Missing assertions:

- exact 64 rows, uniqueness, deterministic ordering, and no extras;
- full pinned dimensions per cell;
- serial execution evidence;
- actual fresh-child execution boundary;
- LoRA target/trainable-layer contract;
- fixed assignment count;
- measured/accurate custom-kernel nodes;
- real `active_after_graph` boundary;
- component replacement shape and nonzero input-gradient path;
- fixture contents/provenance/consumption;
- bounded errors and exit/signal consistency.

The path check at lines 84-89 only rejects literal `/Volumes/` strings in plan cells. It does not prove runtime no-real-asset behavior. Static source audit currently finds no real-asset path, but test itself is weak.

TDD RED claim remains plausible but not independently auditable: no RED command output/log staged. Green tests validate report shape/presence more than helper semantics.

### LOW — Probe A comparison can ignore gradient-tree mismatch

`_max_grad_diff` compares only intersection of checkpoint-on/off keys. It does not assert identical key sets. `finite_loss_*` fields are assigned the combined loss-plus-gradient finiteness booleans. Current run has matching deterministic models and zero measured difference, but test should pin equal nonempty key sets and loss finiteness separately.

## Evidence confirmed

### Report completeness as serialized

```text
rows=64
Probe B=9
Probe C one-graph=20
Probe C sequential=20
Probe D=15
duplicate matrix keys=0
captured failures=0
schema-invalid rows=0
max duration=8.450049457838759s
memory_limit=8_000_000_000
timeout_s=180
serial metadata=true
classification=not-authorized-by-diagnostic
```

All required row keys present. Every successful row includes finite flags and peak fields. Metadata alone does not cure semantic/telemetry defects above.

### Probe A independent reproduction

```text
layer_count=43
pipeline_layer_count=43
decoder_layer_count=43
checkpointed_forward_calls=43
checkpoint_wrapper_name=checkpointed_fn
finite checkpoint-on/off loss and gradients=true
max_gradient_abs_diff=0.0
max_gradient_rel_diff=0.0
```

Subprocess wrapper and `finally` restoration visible in source. Coverage/parity gate passes for implemented tiny case.

### Critical-cell independent reproduction

Successful serial fresh-child reruns:

```text
Probe B D=43 checkpoint-on peak_backward=492,694,696
Probe B D=8 checkpoint-off peak_backward=53,531,060
Probe C D=43/E=256 one-graph peak_backward=6,613,048
Probe C D=43/E=256 sequential peak_backward=1,544,424
Probe D full/cr4 peak_backward=7,443,025,396
Probe D no_routed/cr4 peak_backward=131,443,616
Probe D no_shared/cr4 peak_backward=7,427,475,272
Probe D no_attention/cr4 peak_backward=388,821,144
Probe D routed_only/cr4 peak_backward=373,320,172
```

Values reproduce report closely. Reproducibility does not establish contract validity.

### Tests/build

```text
focused: 4 passed, 1 warning in 2.00s
tracked baseline: 480 passed, 15 skipped, 86 subtests passed in 63.00s
make: Nothing to be done for `all'.
git diff --check: clean
git diff --cached --check: clean
```

Every file contributing to cited focused verdict tracked by `git ls-files`:

```text
tests/helpers/routed_fp4_multilayer_peak_probe.py
tests/test_deepseek_v4_nn_multilayer_peak.py
tests/fixtures/deepseek_v4_nn_multilayer_peak_topology.json
agent-output/cmux-13-3b/multilayer-peak-report.json
```

### Isolation/scope

Source shows serial `subprocess.run` per planned cell, hard `timeout=180`, `mx.disable_compile()`, and `mx.set_memory_limit(8_000_000_000)` in child setup.

No diagnostic source path loads model, shard, dataset, external config, or real model directory. No real smoke executed during review.

Protected worktree hashes match commit `5dee4ce7037902c9787b523373ca46d40aaf6de0` for:

```text
ds4.c
ds4_metal.m
ds4_cuda.cu
ds4_distributed.c
ds4_ssd.c
python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py
python-envs/mlx/src/ds4_ft_mlx/routed_fp4_metal.py
python-envs/mlx/src/ds4_ft_mlx/metal/ds4_routed_fp4_train.metal
```

No production, primitive, root Metal, inference, SSD, CUDA, distributed, or FROZEN edit attributable to staged 5g diagnostic slice.

## Required re-review gate

1. Repair Probe B LoRA/trainable-layer topology.
2. Eliminate duplicate forward construction.
3. Replace copied Probe C telemetry with measured boundary.
4. Resolve fixed-assignment/E=2 contract tension through Architect/BA re-pin.
5. Make fixture load-bearing with provenance.
6. Add behavioral tests covering these invariants.
7. Regenerate complete report under same exact serial/8GB/180s guards.
8. Rerun focused and tracked baseline.
9. Return to independent Reviewer and Test Manager.

Reviewer PASS withheld. Architect re-entry for classification premature. Real-training Path A remains STOPPED.
