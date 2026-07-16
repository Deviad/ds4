# Story 13.3b-5g — multi-layer synthetic peak diagnostics (BA re-pin)

**Status:** GO for one diagnostic-only tracked helper/test/report slice. Real-training Path A remains STOPPED. No production or primitive redesign is authorized. No real model, shard, dataset, external config, full training run, second smoke, or shorter-sequence fallback is authorized.

## User story

As a training engineer (WHO), I want a tracked no-model/no-shard multi-layer synthetic peak diagnostic that separates checkpoint depth, routed-expert node pressure, attention/shared-expert cost, and whole-graph lifetime (WHAT), so that an Architect can select or reject the next production design from controlled peak-memory evidence instead of inferring root cause from one Metal OOM (WHY).

## Authority and outcome boundary

- `agent-output/cmux-13-3b/architecture-13-3b-5g-first-backward-oom.md` is the binding diagnostic architecture.
- The real 4096/20 smoke is authoritative RED: first backward aborted with Metal command-buffer OOM, exit `134`, before the gradient callback, optimizer update, or adapter payload.
- Existing checkpoint coverage evidence shows `grad_checkpoint(model.layers[0])` patches the shared `DecoderLayerNN.__call__` used by all 43 layers. Coverage is proven; scheduler lifetime and efficacy are not.
- This slice may add only tracked diagnostic helpers, tests, a shape-reduced synthetic topology fixture, and the JSON report.
- Diagnostic GREEN does not authorize a production edit, primitive redesign, layer-serial backward, another smoke, or any real-asset access.
- After independent Reviewer PASS and Test Manager GREEN, Architect re-entry is mandatory and solely owns classification and any later design authorization.

## In scope

Suggested tracked additions, with equivalent names allowed only if the same narrow boundary remains obvious:

- `tests/helpers/routed_fp4_multilayer_peak_probe.py`;
- `tests/test_deepseek_v4_nn_multilayer_peak.py`;
- one checked-in 43-entry shape-reduced topology fixture derived from canonical prior configuration evidence, never by opening a real model directory;
- `agent-output/cmux-13-3b/multilayer-peak-report.json`.

The helper and every test contributing to a verdict must be tracked by Git before Coder, Reviewer, or Test Manager cites a pass count or legitimacy verdict.

## Probe A — exact checkpoint coverage and parity

Run in a fresh subprocess using a tiny 43-layer `Model` and a subprocess-confined class monkeypatch.

Required evidence:

1. `len(model.layers) == 43`.
2. Exactly 43 instances are `DecoderLayerNN`.
3. One instrumented forward observes exactly 43 calls through `checkpointed_fn`; a count other than 43 is an immediate STOP and selects checkpoint-integration repair for Architect adjudication.
4. A bounded tiny checkpoint-on and checkpoint-off case produces finite loss and finite gradients.
5. Checkpoint-on and checkpoint-off gradients match at `atol=rtol=1e-5`.
6. The class monkeypatch cannot escape its subprocess.

## Probe B — depth and activation-volume matrix

Use a fresh subprocess per cell with `mx.disable_compile()`.

Pinned shape-reduced full-model parameters:

```text
T = 64
H = 128
I = 64
E = 8
K = 2
hc_mult = 4
LoRA targets = q_a_proj, q_b_proj, kv_proj
trainable layers = min(D, 16)
```

Mandatory decision depths are `D = {1, 8, 16, 43}`. To preserve the Architect's complete pinned depth sweep, checkpoint-on report coverage is `D = {1, 2, 4, 8, 16, 43}`; `D={2,4}` are supporting interpolation cells and cannot replace or weaken the mandatory decision depths. Checkpoint-off bounded controls are `D = {1, 4, 8}`. No depth may be silently removed or reduced.

Each cell records active baseline, active after graph construction, forward peak, backward peak, final active/cache, duration, loss/gradient finiteness, process exit status, signal, and error.

## Probe C — routed expert/node-pressure matrix

Use a routed-only residual chain with frozen synthetic packed payload and no shared expert or attention.

```text
D = {1, 8, 16, 43}
E = {2, 8, 32, 128, 256}
T = 64
H = 64
I = 32
K = min(6, E)
```

Required controls and invariants:

1. Deterministic routes make every expert non-empty whenever `T*K >= E`.
2. Logical tensor dimensions and total assignments remain fixed while comparing expert counts.
3. Record `depth x nonempty_experts`, memory telemetry, and custom-kernel node count on bounded cells.
4. Every one-graph cell has a matched memory-only sequential control that executes the same total one-layer/expert work with an evaluation/cache-clear boundary after each layer.
5. The sequential control makes no claim of mathematical equivalence to end-to-end training.
6. `D=43` and `E=256` are mandatory. No omission, lower replacement, or inferred value is allowed without Architect re-pin.

## Probe D — component ablation matrix

Use the shape-reduced current `DecoderLayerNN` topology and the checked-in synthetic 43-entry fixture.

Required cases:

```text
full
no_routed
no_shared
no_attention
routed_only
```

Required representative compression ratios:

```text
0
4
128
```

Test-only replacements return zero-like outputs with exact expected shape and preserve an input-gradient path. Component comparisons use fresh-process backward peaks at `D=43`; final active memory alone is not verdict evidence.

## Process isolation and resource bounds

- Fresh subprocess per measurement cell.
- Hard timeout `<=180` seconds per cell.
- MLX diagnostic memory limit exactly `8_000_000_000` bytes; never higher.
- No concurrent probe processes.
- No model, shard, dataset, external config, real model directory, real smoke, or full-training access.
- An intentional subprocess OOM, timeout, signal, or non-zero exit is captured as evidence in its row; the parent report must still be valid JSON and must not crash or hide the failure.
- Any pinned cell exceeding time or memory makes the diagnostic verdict STOP for Architect re-entry. The cell may not be replaced by a smaller depth, expert count, token count, hidden size, intermediate size, or shorter fallback.

## JSON report contract

Every measurement row must include every field below. A non-applicable value is explicit `null`; no key is omitted.

```text
case, depth, experts, nonempty_experts, tokens, hidden, intermediate, top_k,
hc_mult, compression_ratio, checkpoint, component_mask,
active_baseline, active_after_graph, peak_forward, peak_backward,
active_final, cache_final, custom_kernel_nodes, duration_s,
finite_loss, finite_gradients, exit_code, signal, error
```

Report requirements:

- one row for every pinned cell or an explicit captured-failure row for that cell;
- peaks are load-bearing; final active/cache values cannot substitute for `peak_forward` or `peak_backward`;
- process failures retain `exit_code`, `signal`, and a bounded error string instead of disappearing from the matrix;
- report identifies probe/control kind sufficiently through `case`, `checkpoint`, and `component_mask` to distinguish one-graph, sequential, coverage, parity, and ablation rows;
- deterministic ordering and enough run metadata to reproduce the synthetic matrix without real assets.

## Classification evidence for Architect re-entry

The report provides evidence for these existing Architect gates; this BA slice does not select among them:

1. **Checkpoint integration fix:** only if Probe A observes fewer than 43 checkpointed calls, a bypassed layer class, or checkpoint-on gradient divergence.
2. **Custom primitive lifetime redesign:** only if routed-only one-graph peak or failure tracks `depth x nonempty_experts`, sequential control remains bounded, and `no_routed` removes at least half of the full-model `D=43` backward-peak delta.
3. **Attention/shared checkpoint redesign:** only if `no_attention` or `no_shared` removes more `D=43` peak than `no_routed`, while routed-only depth/node probes stay bounded.
4. **Layer-serial backward research:** only if each component and sequential control is bounded but the composed one-graph case alone fails or exceeds twice the matched sequential peak delta. A new training-loop ADR, exact loss/gradient/update parity, accumulation semantics, optimizer-state ordering, and checkpoint/resume proof would be required.
5. **STOP Path A:** retain the permanent stop on MLX 0.31.2 if the bounded matrix cannot classify the failure, multiple components remain inseparable, or the only passing approach needs semantic approximation, expert dropping, sequence shortening, or a permanent alternate-path flag.

## Acceptance criteria

All are required:

1. Probe helper, synthetic topology fixture, and every verdict test are tracked by Git.
2. Complete valid JSON report exists for every pinned cell or contains an explicit captured subprocess failure for that cell.
3. Probe A reports exact 43-layer checkpoint coverage and exactly 43 instrumented checkpoint calls.
4. Bounded tiny checkpoint-on/off loss and gradients are finite and match at `atol=rtol=1e-5`.
5. Probe B records fresh-process graph-construction, forward-peak, backward-peak, final-memory, duration, finiteness, and exit telemetry for the full pinned checkpoint-on sweep and bounded checkpoint-off controls.
6. Probe C completes the pinned `D={1,8,16,43}` by `E={2,8,32,128,256}` one-graph matrix and matched sequential controls, including `D=43`, `E=256`.
7. Probe D covers `full`, `no_routed`, `no_shared`, `no_attention`, and `routed_only` at compression ratios `0`, `4`, and `128`, using fresh-process `D=43` backward peaks.
8. Every row carries the complete telemetry schema; peak values, explicit process failures, node pressure, and control identity are preserved.
9. No real model/shard/dataset/config, real model directory, real smoke, full training, or site-packages access occurs.
10. No production inference, root Metal, SSD, CUDA, ROCm, distributed, model-loader, FROZEN parity body, or current routed primitive source is edited.
11. `git diff --check` is clean, and direct hashes record protected production/FROZEN sources because zero `git diff` output is not sufficient in this checkout.
12. Independent Reviewer verdict is PASS and independent Test Manager verdict is GREEN, each after verifying every cited test/helper is tracked with `git ls-files`.
13. Architect re-entry consumes the report and explicitly selects or rejects a production redesign. Diagnostic GREEN alone authorizes no redesign and no smoke.

## Immediate STOP gates

Carry unchanged from the Architect adjudication:

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

Any STOP produces an explicit captured finding and returns to Architect. It does not authorize a fallback, omitted cell, real-asset experiment, production fix, or second smoke.

## Review and handoff gates

- Coder: TDD RED → GREEN only within the tracked diagnostic boundary; no production or real-asset access.
- Reviewer: independently verify matrix completeness, telemetry integrity, control validity, tracking, protected hashes, and no causal overclaim; required verdict PASS.
- Test Manager: independently rerun bounded synthetic validation serially, verify resource limits and captured failures, and check every verdict test/helper with `git ls-files`; required verdict GREEN.
- Architect: mandatory re-entry after double-green; owns the next classification and any durable design decision.
