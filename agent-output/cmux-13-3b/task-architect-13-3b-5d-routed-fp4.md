# Story 13.3b-5d — Architect micro-slice: sparse routed FP4 backward

## Goal
Design the smallest correctness-first training-shim change that removes the all-256-expert/full-sequence FP4 activation/dequant backward peak while preserving exact routed-MoE semantics and LoRA gradient flow. The 400GB MLX scheduling limit still OOMed at first backward after finite validation.

## User model/budget override
Use current Architect pane model `openai-codex/gpt-5.6-sol` high. Do not switch providers/models.

## Read first
1. `agent-output/cmux-13-3b/coder-13-3b-5c-memory400-stop.md`
2. `agent-output/cmux-13-3b/architecture-13-3b-5c-memory-safe-backward.md`
3. `agent-output/cmux-13-3b/requirements-13-3b-5c-memory400.md`
4. Current implementations in:
   - `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_nn.py`
   - `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`
   - installed MLX/MLX-LM APIs under `/Volumes/Data NVME/mlx-ft/ds4/.venv/lib/python3.14/site-packages/`
5. Existing tiny/real forward and backward tests for FP4 experts and `SparseMoeBlockNN`.

## Locked evidence
- Clean 4096 default memory run: finite validation 19.116, first-backward OOM.
- Clean 4096 `mx.set_memory_limit(400_000_000_000)` run: finite one-batch validation 18.110, first-backward OOM, RC 134.
- Clean 2048 run also OOMed; NaN was proven zero-target masking for 174/819 validation records.
- Validation cache is cleared; the OOM is active backward graph memory.
- Current routed path loops all 256 experts and computes each expert over the full sequence before masking/accumulation.
- Dense dequant output floor is ~24GiB/layer independent of sequence length; full-sequence expert activations add ~40GiB/layer at 4096.
- Existing layer checkpointing bounds layers but not the all-expert graph within a layer.
- Memory-limit scheduling, dequant stop-gradient, nested per-expert checkpoint probes did not solve the issue.

## Required investigation — no full model run
1. Trace exact `SparseMoeBlockNN` forward/backward shapes and routing contract: top-k count, gate weights, shared expert, selected routed experts, token flattening, output ordering, aux-loss implications.
2. Inspect MLX APIs and VJP support for a sparse implementation:
   - gather/take token rows by routed assignment;
   - differentiable scatter-add back to token positions;
   - detached discrete assignment/expert indices;
   - sorting/grouping or fixed-capacity routing;
   - `mx.gather_mm`, segmented operations, custom functions, or chunked accumulation if applicable.
3. Determine whether only selected token/expert assignments can be evaluated without dynamic-shape recompilation or unsupported index VJPs. Explicitly account for top-k gate-weight gradients.
4. Compare candidate designs:
   - per-expert gather → FP4 expert on assigned tokens → weighted scatter-add;
   - assignment/expert chunking with fixed shapes;
   - grouped/gather matmul route;
   - expert-weight dequant chunking while retaining dense token masks.
5. Estimate peak memory for real dimensions and 4096 tokens. Provide a credible bound below 400GB with safety margin, not just relative claims.
6. Use tiny/light probes only to verify MLX VJP support and numerical equivalence against the existing dense-mask implementation. No model shard loading.

## Design requirements
Choose one implementation; no unranked menu.

Must preserve:
- exact top-k expert assignments and normalized gate weights;
- differentiability through token activations and gate weights;
- discrete routing indices detached from VJP;
- shared expert behavior;
- FP4 LSB-first/BF16 scale contract;
- forward equivalence to current dense-mask reference on tiny deterministic fixtures;
- LoRA gradient equivalence/finite non-zero behavior;
- production inference/SSD/CUDA/distributed/default Metal paths untouched.

Prefer a change isolated to `deepseek_v4_nn.py` training shim. If a FROZEN vendor body edit is unavoidable, STOP and identify the exact ADR amendment required before coding.

## TDD requirements
Specify RED tests before implementation:
- tiny multi-token/multi-expert routed forward parity against an independent dense NumPy or frozen MLX reference;
- top-k duplicates/ties/empty-expert behavior;
- gradient parity for input and gate weights within explicit tolerances;
- every index tensor detached; no `scatter_axis` VJP failure;
- instrumentation proving expert computation receives routed-token count, not full sequence length;
- peak-memory probe showing bounded growth versus dense reference;
- real-config lightweight shape/contract checks without loading shards.

## Deliverable
Write `agent-output/cmux-13-3b/architecture-13-3b-5d-routed-fp4-backward.md` with:
- exact root cause and selected design;
- APIs/shape algebra/pseudocode;
- memory estimate and expected safety margin;
- file/function edit map;
- RED→GREEN test plan;
- invariants and forbidden alternatives;
- run plan returning to 4096/400GB smoke;
- STOP conditions;
- whether BA re-pin and ADR update are required.

No production edits. Create `.cmux-status/architect.done` only on success. End with `{"status":"ok","role":"Architect"}` or error JSON.
