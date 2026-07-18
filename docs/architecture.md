# DS4 Project Architecture

This document is the high-level architecture map for the DS4 / DS4 fine-tuning project. It complements `docs/technical-spec.md`, which remains the operational build/run/test specification.

## Purpose

DS4 is a DeepSeek V4 Flash specific runtime and fine-tuning toolchain. It is not a generic GGUF runner. The project optimizes for a small, readable, high-performance implementation with correctness gates before speed-oriented relaxations.

## Architectural goals

- Production inference runs through whole-model Metal graph execution on Apple Silicon.
- SSD streaming, CUDA, distributed inference, and default Metal inference stay isolated from unrelated fixes.
- The default Metal model loader remains mmap-backed; full GGUF payloads are not eagerly copied.
- SSD streaming explicitly owns routed-expert buffers and hides missing expert/layer reads behind useful inference work where possible.
- CPU code is reference/debug only, not a production fallback.
- Fine-tuning produces portable LoRA artifacts that can be validated and converted into DS4 runtime-compatible adapter payloads.
- Long local agent sessions should remain practical through live KV reuse and disk KV checkpoints.

## Repository scaffolding

```text
ds4-finetuning/
├── AGENTS.md                         # Agent instructions, workflow gates, model routing
├── docs/backlog.md                 # Canonical requirements, user stories, acceptance criteria
├── training-next-status.md           # Current fine-tuning status and next blockers
├── progress.md                       # Chronological implementation progress
├── docs/
│   ├── architecture.md               # This high-level architecture/scaffolding document
│   ├── technical-spec.md             # Build/run/test/specification and invariants
│   ├── deepseek-v4-architecture-dossier.md
│   ├── deepseek-v4-mtp-policy.md
│   └── adr/                          # Architecture Decision Records
├── .pi/
│   └── agents/                       # Reusable project-local cmux/Pi role agents and launchers
├── agent-output/                     # Per-slice handoff artifacts, logs, reviews, reports
├── python-envs/
│   ├── mlx/                          # MLX fine-tuning / parity environment definition and source
│   └── torch/                        # Torch/PEFT/TRL/Accelerate environment definition
├── scripts/                          # Fine-tuning, conversion, inspection, and workflow helpers
├── tests/                            # Python unit/parity/regression tests
├── ds4.c                             # Runtime: loading, tokenizer, CPU reference, Metal scheduling, sessions
├── ds4_cli.c                         # CLI and REPL entrypoint
├── ds4_server.c                      # OpenAI/Anthropic-compatible HTTP server
├── ds4_metal.m                       # Objective-C Metal runtime/wrapper layer
└── metal/                            # Metal compute kernels
```

## Major subsystems

### Runtime core

Primary files: `ds4.c`, `ds4.h`, `ds4_cli.c`, `ds4_server.c`, `ds4_metal.m`, `metal/*.metal`.

Responsibilities:

- Load DS4 GGUF/checkpoint metadata and tensors.
- Run tokenization, prompt/session handling, and generation orchestration.
- Execute whole-model Metal graph inference in the production path.
- Maintain CPU reference/debug paths without turning them into hidden production fallbacks.
- Serve CLI and HTTP APIs without exposing tensor internals through broad public APIs.

### Fine-tuning toolchain

Primary files: `scripts/finetune_ds4.py`, `python-envs/mlx/`, `python-envs/torch/`, `vendor/mlx-lm`, `tests/test_finetune_ds4.py`, `tests/test_mlx_lm_source.py`.

Responsibilities:

- Prepare isolated MLX and Torch environments.
- Keep released `mlx-lm==0.31.3` as the default MLX-LM source while allowing an explicit, verified opt-in to the pinned `Deviad/mlx-lm` submodule at `vendor/mlx-lm` (`80fab4e419a57f9465bb9e2f4e90010d645e124c`, advanced from `15b522f5...` by Story 14.4).
- Run cheap validation gates before expensive training/conversion work.
- Convert portable LoRA safetensors into DS4 canonical adapter artifacts.
- Keep marker files meaningful: no full-forward marker is written until the required parity stories are complete.
- Treat MLX-LM source selection as environment plumbing only: switch with `python -m pip --no-deps`, verify source identity with module path plus PEP 610 metadata, and never edit fork source or claim a trainer/Metal lifetime fix in the bootstrap story.
- **DS4 segmented activation** (Story 14.3): `scripts/ds4_segmented_smoke.py` is a standalone thin-wiring entry point that reuses the MLX-LM fork's public `train()` signature and the existing `lora.build_parser()`/`CONFIG_DEFAULTS`/`train_model`-body convention. It constructs the DS4 segmented loss-and-gradient provider from `ds4_ft_mlx.segmented_loss_and_grad.make_ds4_segmented_loss_and_grad` and passes it explicitly as `loss_and_grad=provider`. This step (`ds4-segmented-smoke`) is a non-default MLX step; the default `smoke-train`, `full-train`, and all other MLX-LM CLI commands remain byte-unchanged. The trainer retains ownership of accumulation, optimization, distributed averaging, callbacks, UI, validation, and checkpoint save — the provider owns only the bounded forward/reverse transform per microbatch. The activation is a local tool, not a durable architectural boundary change; no ADR is created. The pinned dataset path was repinned from `mlx-4096` to `mlx-4096-smoke` (filtered copy, 66 rows excluded by the same whitespace-token conservative bound the smoke preflight uses) after the original failed preflight on a row exceeding 4096 tokens. The filtered-dataset provenance is recorded in `agent-output/cmux-14-3/filtered-dataset-provenance.json` (input/kept/excluded: train 15170/15108/62, valid 819/816/3, test 824/823/1; total excluded 66; source and filtered SHA-256 hashes per split; excluded-row manifest). The original dataset remains immutable and is rejected by the pinned-path gate. The hard timeout was repinned from 600s to 1200s after the 14.3a filtered-dataset smoke reached trainer startup but timed out with zero completed iterations. The timeout constant <code>SMOKE_TIMEOUT_SECONDS</code> is the single source of truth: the signal alarm, backup watchdog thread, and report message all read it at runtime.

### DeepSeek V4 MLX/reference port

Primary files: `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`, `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py`, `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_*_spec.py`, parity tests.

Training-only packed-FP4 Metal component (ADR 0028): `python-envs/mlx/src/ds4_ft_mlx/routed_fp4_metal.py` plus package data `python-envs/mlx/src/ds4_ft_mlx/metal/ds4_routed_fp4_train.metal`. This source stays outside root `metal/`, `ds4_metal.m`, and the production Makefile/runtime source list.

Responsibilities:

- Provide bounded MLX implementations and pure-Python references for DeepSeek V4 Flash mechanics.
- Keep the real trainable `mlx.nn.Module` sibling separate from the FROZEN parity-fixture module.
- Execute routed packed-FP4 training forward/input-VJP through opaque assignment-proportional Metal kernels on supported Apple MLX hosts, without exposing dense expert dequant matrices to outer transforms.
- Prove each model subsystem by tiny deterministic fixtures before relaxing fail-closed gates.
- Fail closed on unsupported training platforms; never fall back to the superseded Python nested-VJP path or alter production inference backends.

### Agent workflow and handoff artifacts

Primary files: `AGENTS.md`, `.pi/agents/`, `agent-output/`, `.cmux-status/`.

Responsibilities:

- Use reusable project-local Pi agent definitions under `.pi/agents/`.
- Store per-slice outputs under `agent-output/<slice>/`.
- Use marker files under `.cmux-status/` as the authoritative completion signal.
- Launch cmux role agents in visible interactive mode by default.

## Data and marker policy

- Do not mutate the base GGUF/checkpoint.
- Do not create `.deepseek-v4-forward-parity-ok` speculatively.
- Do not create `model-4bit` until the relevant conversion/parity gate is proven.
- Test fixtures should be tiny, deterministic, and shaped to exercise real semantics without loading large model payloads.
- Real-model tests must skip cleanly when required local weights/binaries are unavailable.

### Two independent correctness tracks (ADR 0008)

- **Two independent parity tracks have distinct markers** (ADR 0008):
  - `.deepseek-v4-forward-parity-ok` gates the **MLX** track — shimmed-checkpoint
    forward parity → `convert-shimmed` → `model-4bit` → MLX adapter training. It
    stays absent until routed expert-decode dispatch (11.15c follow-up),
    `hc_mult>1` multi-layer forward (11.11), and real-scale integration are all
    proven, bound to the parity-report hash.
  - `.ds4-gguf-generate-ok` gates the **DS4/Metal base-generation** track — the
    real `ds4` runtime smoke-generates from the immutable base `ds4flash.gguf`,
    **independent** of MLX / conversion / `model-4bit` / adapter (Story 11.19).
- **Track-independence invariant:** neither track creates or validates the other's
  marker. Track A (`.ds4-gguf-generate-ok`) success never creates `model-4bit` or
  `.deepseek-v4-forward-parity-ok`; Track B success never creates
  `.ds4-gguf-generate-ok`. `convert-shimmed` is gated on Track B only.
- Do not create `.ds4-gguf-generate-ok` speculatively; it is written **only** by
  Story 11.19's proven DS4 base-generation gate, bound to recorded
  prompt/output + `ds4` binary/version + determinism evidence.
- **Pinned marker path (Story 11.19):** `.ds4-gguf-generate-ok` lives at
  **`/Users/spotted/projects/ds4/.ds4-gguf-generate-ok`** (`DS4_ROOT`, home of the
  binary + base `ds4flash.gguf`) — separate from the MLX-track marker dir
  `/Volumes/Data NVME/mlx-ft/ds4/`. The **only** writer is Story 11.19's gate
  helper `ds4_gguf_base_smoke_check(args)` in `scripts/finetune_ds4.py`, which
  runs one bounded deterministic production-Metal invocation (`ds4 -m
  ds4flash.gguf -p <prompt> -n 12 --temp 0 --metal`) under an opt-in env gate
  (`DS4_GGUF_BASE_SMOKE=1`), clears stale markers first, asserts `returncode==0`
  + real generated text + base-GGUF immutability, and writes the marker bound to a
  sidecar evidence record. It skips cleanly (skip ≠ proof, no marker) when the
  env gate is unset or either artifact is missing; it never imports torch/MLX.
- Rewording the `forward_parity_blockers()` code tuple (to drop DS4-GGUF base
  generation as a *forward-parity* blocker, keeping only the genuine Track-B
  blockers) was **completed by Story 11.20**: the 4th tuple item is now
  MLX/shimmed-Track-B scoped — `"full MLX shimmed-checkpoint load/forward and MLX
  generation smoke (Track B only; DS4-GGUF base generation is its own Track-A
  gate, .ds4-gguf-generate-ok, not a forward-parity blocker)"` — and the other 3
  items, the tuple length (4), and order are byte-unchanged.

## Canonical documentation policy

`agent-output/<slice>/` is handoff evidence, not canonical architecture. It may contain requirements, slice architecture, coder notes, review findings, and test reports, but durable design choices must be consolidated into `docs/` in the same slice.

Canonical locations:

- Whole-project scaffolding and subsystem boundaries: `docs/architecture.md`.
- Durable decisions and workflow policy: `docs/adr/`.
- Build/run/test commands and invariants: `docs/technical-spec.md`.
- DeepSeek V4 tensor/model mapping: `docs/deepseek-v4-architecture-dossier.md`.
- MTP policy: `docs/deepseek-v4-mtp-policy.md`.

Default agent rule: if a slice `agent-output/<slice>/architecture.md` introduces a durable architecture decision, the Architect must also update `docs/architecture.md` or create/update an ADR. Reviewer must treat stale docs as a finding unless the slice explicitly marks the handoff as exploratory/non-canonical.

Historical `agent-output` technical themes already consolidated here:

- Story 11.x MLX/reference work belongs to the **DeepSeek V4 MLX/reference port** subsystem.
- Story 11.14 CSA compressor/indexer attention remains a bounded, fail-closed tiny subset until real-scale and KV/generation stories land.
- Story 11.15 expert packing/decode work belongs to the **DeepSeek V4 MLX/reference port** plus fine-tuning parity gates; real checkpoint metadata supersedes earlier fictional-FP4 assumptions when observed.
- Story 11.15d: 256-expert / top-6 routing parity is proven in the pure-Python spec module (`deepseek_v4_moe_spec.tiny_topk_moe_routing`) **without relaxing vendor caps** — the `n_routed_experts<=4` and `num_hidden_layers∈{1,2,3}` caps live only in the vendor model's `_validate_real_mode`/`bounded_model_expected_shapes` and stay untouched; the routing reference uses `MoEConfig` (whose only guard is `num_experts_per_tok<=n_routed_experts`). Durable correctness invariants: the top-k **tie-break is lower-index-wins on equal `score+correction_bias`** (Python stable-sort semantics), `e_score_correction_bias` affects **selection only** (never the contribution weight), and routing is **decode-independent** (combine uses expert outputs and never calls `dequantize_expert_packed`, which after Story 11.22 dispatches `i8` along **both** proven block-scale paths — BF16 (2 bytes/scale, PyTorch-parity-proven, Story 11.21) and **E8M0/UE8M0 (1 byte/scale, proven via an independent closed-form reference + shim-pipeline (`f8_e8m0_to_bf16`) self-consistency, Story 11.22)** — bitwise-identical routing to the proven primitives, no new math. `i8` without `block_size`+`scale_axis` metadata, an ambiguous scale byte length, and `fp4` **stay fail-closed** per ADR 0002. Routed-I8 E8M0 **raw-decode** is proven and is **NOT a Track-B forward blocker** (the MLX forward path consumes BF16-shimmed scales); the genuine Track-B forward blockers are multi-head CSA (`hc_mult>1`/`o_groups>1`/`num_key_value_heads>1`), 43-layer integration + vendor `_moe_mlx` tie-break, the shared-expert 2-D 128×128 axis, and real `model-4bit` MLX conversion → generation smoke). The MLX `_moe_mlx` `mx.argsort` tie-break vs this lower-index rule must be reconciled at full-model integration (11.15e/11.16). The **shared-expert** F8_E4M3 + E8M0 **clean 2-D `[rows/128, cols/128]` decode** is proven (`dequantize_f8_e4m3_e8m0_2d_block_scale`, Story 11.24) bit-identical to the closed form `e4m3 × 2^(scale−127)` on the real `[2048,4096]+[16,32]` geometry + a NaN-aware sweep (scale `0xFF` → NaN); the 2-D scale tensor itself determines the tiling, so the square-`[128,128]`-block 1-D-inference "axis ambiguity" is a classification artifact, not a decode problem. The classifier clears the shared-F8_E4M3 application-order + 2-D-axis unknowns for the observed clean 2-D tiling; FP4, unrecognized dtypes, routed discrepancy, and the genuinely-ambiguous shared 2-D axis (no observed 2-D scale) **stay fail-closed** per ADR 0002. The MLX runtime shared forward is still open; this is a decode-primitive + classifier proof, **NOT** a shared-expert forward-parity or Track-B unblock claim.
- Story 11.17: stateful **CSA Ca-carry** is proven for the tiny single-head `compression_ratio=4` `hc_mult=1` subset via `StatefulCSACache` in `deepseek_v4_attention_spec.py`, which carries **exactly one completed compression window (the Ca overlap) plus a partial pending window** across arbitrary (incl. non-window-aligned) forward calls, and is proven equal to the cache-less full-sequence `tiny_compressor_indexer_attention_reference` (`compressed_kv` exact; `attended` ≤1e-5; `index_scores`/`block_bias` ≤1e-5 on causally-valid entries; top-k exact). The main compressor and the indexer's downscaled compressor share the same one-window Ca/Cb overlap; the per-window pool and per-token scoring are shared (extract-method, byte-identical) with the cache-less reference. Everything outside the proven stateful subset (ratio≠4, `num_key_value_heads>1`, real scale) **stays fail-closed** with specific messages — **except that Story 11.23 lifted the `hc_mult>1` axis in the stateful cache, Story 11.28 lifted the stateful `num_attention_heads>1` gate for the proven fusion primitive, and Story 11.29 lifted the stateful `num_output_groups>1` gate for the proven multi-group fusion primitive**. `hc_mult` is orthogonal to the cached KV windows (the cache compresses the post-collapse hidden stream and never reads `hc_mult`), proven by single-chunk equivalence + cross-split determinism at `hc_mult=2` with the cache body unchanged. The remaining stateful CSA gates (`num_key_value_heads`) + compression + index stay fail-closed in the stateful cache, and the **MLX runtime** `hc_mult`/`o_groups` gates (`vendor _csa_config_error`) stay closed (untestable here); the 11.16 compressor-free `IncrementalSlidingKVCache` is a separate path and is unchanged. Real `smoke-generate` stays blocked and the final `.deepseek-v4-forward-parity-ok` marker stays absent (still needs real expert-decode dispatch, `hc_mult>1` multi-layer, and real-scale integration); `forward_parity_blockers()` is unchanged.
- Story 11.25/11.26/11.27/11.28/11.29: the **stateless single-chunk multi-head CSA fusion reference** (`tiny_multihead_csa_fusion_reference`) is proven vs the real `DeepseekV4Attention` module under the venv python (torch 2.12.1 + transformers 5.12.1) for `num_attention_heads>1` and, at 11.26, `o_groups>1` via `_grouped_output_projection` at ≤1e-5. Story 11.27 added the **stateful single-head CSA fusion primitive**: `StatefulCSACache` carries a sliding-window per-token KV stream alongside the unchanged compressed-KV Ca/Cb carry, and `tiny_stateful_csa_fusion_reference` fuses `[sliding_kv ∥ compressed_kv]` per chunk with strict `0/-inf block_bias`, sink softmax, inverse-RoPE, and grouped output. **Story 11.28 closes the multi-head × stateful intersection cell** with zero new math by removing the 2-line `StatefulCSACache.__init__` `num_attention_heads != 1`/`single-head CSA required` gate and parametrizing the fusion parity test over `num_attention_heads ∈ {1,2,4}`. **Story 11.29 closes the FINAL multi-group × stateful cell** with zero new math by removing exactly the 2-line `StatefulCSACache.__init__` `num_output_groups != 1`/`o_groups=1 required` gate and parametrizing the same test over `(num_heads,o_groups) ∈ {(1,1),(4,1),(4,2),(2,2)}`. `step_fusion`, `_project_stateful_fusion_token`, `_grouped_output_projection`, `tiny_stateful_csa_fusion_reference`, and `tiny_multihead_csa_fusion_reference` stay byte-identical; `_grouped_output_projection` dynamically loops `for group in range(spec.num_output_groups)`, with the stateless `(4,2)` math already proven by 11.26. The AC5 11.29 probe and green test prove cross-split determinism across `[8]`, `[3,1,4]`, and `[2,2,2,2]` and real-module stateful parity at `o_groups>1` (probe: `(4,2)` spec-vs-real `[3,1,4]=2.236e-09`, `(2,2)` `[3,1,4]=2.603e-09`, determinism `0.0`, direct `o_a_proj`/`o_b_proj` layout, no transpose) under the venv interpreter. The runner's static provenance labels remain conservative/stale by design (`"stateful single-head CSA fusion"` covered, `"stateful multi-head CSA cache"`/`"stateful multi-group CSA cache"` not_covered); the authoritative multi-head/multi-group provenance is the numeric parity proof plus this document/backlog. The pure-Python stateful CSA fusion epic is now complete for the configurable proven axes: single-head stateful (11.27), multi-head stateful (11.28), and multi-group stateful (11.29). The proven boundary is still **proof primitive only**: the MLX runtime vendor gate stays closed per ADR 0002, and **axis (b) `num_key_value_heads>1` is a permanent MQA invariant**. Compression_ratio, `num_key_value_heads`, index, the coupled compressor-free gate, and all vendor `_csa_config_error` gates stay closed. This is NOT a full-model, MLX-runtime, or Track-B forward-parity claim. The resolved-semantics epic sequence is 11.25 (axis a, stateless) → 11.26 (axis c, stateless `o_groups>1` proof, no gate lift) → 11.27 (stateful single-head `[sliding∥compressed]` fusion) → 11.28 (stateful multi-head gate lift, DONE) → 11.29 (stateful multi-group gate lift, DONE).
- Story 11.35 / ADR 0009: the `deepseek-v4-forward-parity-readiness` report carries an additive `stateful_decode_readiness` diagnostic block. It is introspection-only and fail-closed: the live vendor MLX real-mode probe records no production cache/decode/KV/sliding-window-stateful seam (`seam_available=false`), while the proven spec-layer stateful seam is explicitly labelled non-production. The block changes no gate, no marker writer, no schema version, and no honesty counter (`full_forward_parity=false`, `marker_earned=false`, `blockers_count=4` remain binding).
- Story 11.36 / ADR 0010: the readiness report also carries an additive `b1_hc_mult_multilayer_readiness` diagnostic block. It is introspection/guard-only and fail-closed: `proof_available=false`, `hc_mult_multi_layer_allowed=false` under the current probe, and the multi-layer `hc_mult>1` cell is double-blocked by both the production `_validate_real_mode` gate and the trusted `_integrated_multilayer_forward` reference; `set_transformers_integrated_weights` is single-layer only. The block changes no gate, no production API, no schema version, and no honesty counter.
- Story 11.37 / ADR 0011: `real_mode_proofs` now includes `B2-a-1`, a synthetic single-layer `expert_dtype="i8"` integration proof that drives public `Model.load_weights()`/`__call__` through the real-mode `_real_forward` -> `_moe_mlx` I8 block-scale dequant branch. It is partial B2 evidence only: packed FP4 expert dequant, real checkpoint payload decode, expert parallel kernels, and multi-layer I8 stay fail-closed; no marker, gate, conversion, generation, or production/vendor math path changes.
- Story 11.38 / ADR 0012: `real_mode_proofs` now includes `B2-a-2`, the multi-layer counterpart to B2-a-1. It drives synthetic `expert_dtype="i8"`, `num_hidden_layers=2` weights through public `Model.load_weights()`/`__call__` so stacked `_real_forward` reaches `_moe_mlx` I8 block-scale dequant on every layer, with an NL=3 secondary check. This closes synthetic multi-layer I8 dequant integration as partial B2 evidence only; packed FP4 expert dequant, real checkpoint payload decode, expert kernels, `hc_mult>1` (B1), and B3 stay fail-closed. No marker, gate, conversion, generation, or production/vendor math path changes.
- Story 11.39 / ADR 0013: `real_mode_proofs` now includes `B2-a-3`, the top-k>1 multi-expert counterpart to B2-a-1/B2-a-2. It drives synthetic `expert_dtype="i8"`, `n_routed_experts=4`, `num_experts_per_tok=2` weights through public `Model.load_weights()`/`__call__` so `_moe_mlx` combines I8-dequantized routed experts via proper-subset `argsort`, multi-term `denom`, per-expert `factor=scores/denom`, and `factor=0` masks for unselected experts, with a top-k=3 secondary check. This closes synthetic top-k>1 multi-expert I8 dequant integration as partial B2 evidence only; packed FP4 expert dequant, real checkpoint payload decode, expert kernels, `hc_mult>1` (B1), and B3 stay fail-closed. No marker, gate, conversion, generation, or production/vendor math path changes.
- Story 11.40 / ADR 0014: the readiness report also carries an additive `b2_real_checkpoint_payload_readiness` fail-closed diagnostic block — the B2 real-payload-decode analogue of 11.35 `stateful_decode_readiness` and 11.36 `b1_hc_mult_multilayer_readiness`. A pure fail-closed builder is fed an introspection / header-only probe over the **original** real DeepSeek V4 Flash F8 checkpoint (`HF_MODEL`); the probe re-derives the ADR-0007-verified expert payload classification live via the existing Story 11.15a `read_safetensors_header` + `classify_checkpoint_expert_packing` path (dtype/shape only, never payload bytes — `payload_bytes_decoded=false`; `hf-f8shim` is not the target because the shim rewrites F8 dtypes and cannot reproduce the `I8`+`F8_E8M0`/`F8_E4M3`+`F8_E8M0` packing) and pins the index SHA-256 for drift detection. The block records `decode_trusted_reference_available=false`, `real_payload_decoded=false`, `proof_available=false`, and the exact ADR-0007 §4 missing-independent-reference requirement (our torch/numpy + the shim's `f8_e8m0_to_bf16` are circular; Transformers ships only `Fp8Dequantize`/`Mxfp4Dequantize`). Classification success is never a readiness flip (`status="fail-closed"`, `decision="not-ready"`, `fail_closed=true`, `ready=false`) until BOTH an independent trusted routed I8+F8_E8M0 full-path reference lands AND a reviewed real-mode real-payload decode proof is accepted as a `real_mode_proofs` entry. It is diagnostic only — NOT proof-delivering — so it adds no `real_mode_proofs` entry: `proofs_total` stays 8, `full_forward_parity=false`, `marker_earned=false`, `blockers_count=4`, `.deepseek-v4-forward-parity-ok` and `model-4bit` stay absent, no production/vendor/spec math is edited.
- Story 11.41 / ADR 0015: the readiness report also carries `b2_routed_dequant_trusted_reference_readiness`, a fail-closed per-candidate trusted-reference landscape block that complements 11.40's payload classification by recording the official convert.py other-packing mismatch, kernel.py no-I8-op gap, Transformers delegation gap, circular local reconstructions, and DS4-CPU-harness-not-yet-built independence criterion without adding a proof entry or changing counters/markers.
- Story 11.16: tiny **incremental KV-cache attention** is proven equal to the full-sequence prefill reference (≤1e-5, exact by shared math) in `deepseek_v4_attention_spec.py` via a single-shared-KV-head `IncrementalSlidingKVCache` that stores **position-baked RoPE'd KV rows** and persists exactly `sliding_window-1` rows between steps (full causal history when `sliding_window=0`); the incremental step and the full-sequence reference share the same projection/sink/RoPE/`o_a`/`o_b` helpers (extract-method, byte-identical). A tiny **weight-free greedy decode loop** (`tiny_greedy_decode`, in-fixture embed/vocab matrices, lower-index argmax tie-break) is deterministic and **drift-free vs full recompute** (≤1e-5). **CSA stateful Ca-carry across steps stays deferred / fail-closed** — incremental/decode with `compression_ratio!=0` raises `NotImplementedError`; the compressor-free sliding-window path is the binding gate. Real `smoke-generate` stays blocked (`model-4bit` absent, no large model process) and the final `.deepseek-v4-forward-parity-ok` marker stays absent (still needs real expert-decode dispatch, `hc_mult>1` multi-layer, and real-scale integration); `forward_parity_blockers()` is unchanged.
- Story 11.15e: 43-layer **structural loadability** is proven via a header-only stacked shape-compatibility validator in `deepseek_v4_checkpoint.py` (`stacked_shape_compatibility_report` / `assert_stacked_shape_compatible`): it checks contiguous layer coverage `0..num_hidden_layers-1`, discovers the per-layer family signatures (the real 43-layer Flash checkpoint has **4 distinct per-layer templates** — hash-routing layers `0..2`, alternating CSA/indexer layers `3..42`), pins canonical shapes for the proven core families (attention q/kv/o + norms + sinks, router gate/bias/tid2eid + `e_score_correction_bias`, shared + routed experts), validates the routed-expert index set per MoE layer, enforces intra-template shape consistency, and **fails closed with a specific named blocker** on any missing/renamed/mis-shaped core key or unclassified family — never a silent pass. It is **header/metadata-only** (no tensor payload, no full-weight load, no checkpoint mutation) and never constructs a forward. The vendor **forward gate stays triple-blocked** in `_validate_real_mode` (`num_hidden_layers=43`, `hc_mult>1` multi-layer, `n_routed_experts>4` forward) and is byte-untouched — the only “cap relaxation” is that the header-only validator accepts 43 layers as input. CSA/HCA per-family canonical shape pinning is **deferred** (consistency-checked, not hand-pinned). With 11.15e, Story 11.15 (a–e) is complete **except** the intentionally-deferred routed decode dispatch (11.15c follow-up) and `hc_mult>1` multi-layer forward (Story 11.11); `.deepseek-v4-forward-parity-ok` stays absent until those and Story 11.16 prove out.
- Reusable cmux/Pi role-agent scaffolding belongs under `.pi/agents/`; per-slice `agent-output` files are artifacts only.

## Design decision process

Architecture decisions that affect boundaries, runtime behavior, model parity gates, marker semantics, documentation ownership, or agent workflow should be recorded as ADRs in `docs/adr/`.

Use ADRs for durable decisions; use `progress.md` and `training-next-status.md` for current execution state.

## Related documents

- `docs/technical-spec.md` — operational technical specification.
- `docs/deepseek-v4-architecture-dossier.md` — model/tensor mapping dossier.
- `docs/deepseek-v4-mtp-policy.md` — MTP tensor policy.
- `docs/backlog.md` — canonical Requirements, trackable user stories, and acceptance criteria. BA owns updates. User story format: `As a [type of user] (WHO), I want [some goal] (WHAT), so that [some reason] (WHY).`
- `training-next-status.md` — current implementation status.
- `AGENTS.md` — agent workflow and quality rules.

### Story 14.5 bounded segmented pilot

r4 admission lineage is immutable: B3 stores exact A3 report and marker bytes plus SHA-256, strict schemas, A3 authorization, histories, identity, contract, artifacts, and resume evidence; final publication re-reads those bytes and validates them against independent phase authorization. A3 success markers require exact phase-A resume bindings; B3 and final markers use exact report/path/hash/contract schemas. Attempt-2 history verification uses import-time frozen semantic, file, manifest-hash, and absence roots.

The bounded pilot is a separate, non-default entry point in `scripts/ds4_segmented_pilot.py`; it does not alter the frozen Story 14.3 smoke or default training commands. MLX-LM remains owner of optimizer updates and checkpoint cadence. Phase A performs exactly two updates and Phase B performs one fresh-process update after loading the Phase A step-2 adapter. Resume evidence proves adapter-weight equality by canonical tensor digest only; it makes no optimizer-state, RNG, dataset-cursor, scheduler, global-step, convergence, or full-training-readiness claim.

Story 14.5c synthetic r12 coder evidence is complete; real A3/B3 remain blocked pending exact-revision independent review, Test Manager GREEN, and fresh explicit phase authorization. R12 closes the previously missing exact caller-level phase/final failure-report and fail-marker assertions at all two A3 and four B3/final publication writes, and proves one coordinated attempt-2 path/size/hash/payload substitution passes only under a deliberately weakened verifier while the public immutable-target guard rejects it with the targeted error. Earlier R11 closures remain: loaded-MLX negative schema/payload matrix, independent wrong-JSON-type coverage for every A3/B3/final marker field, and coordinated attempt-2 guard reachability. Attempt 2 remains consumed immutable failure evidence and the only runnable successor is fixed namespace `ds4-segmented-pilot-attempt-3`. Canonical safetensors identity accepts `__metadata__` only when absent, JSON null, or a JSON object. Metadata remains excluded from tensor entries and `canonical_tensor_digest_v1`; duplicate-key rejection, tensor schema/layout checks, bounded payload reads, tensor manifests, and physical file SHA-256 remain unchanged. A private expected snapshot binds the attempt-2 pre-log record, runtime manifest, ten exact runtime files, failure markers, revision, command, lock/watchdog lifecycle, and absent OK/B2 evidence; attempt-1 and attempt-2 evidence can satisfy historical admission only. A3/B3 paths, reports, markers, commands, contract digests, and resume validation consume one centralized fixed mapping. Pre-log admission consumes an externally reviewed phase authorization value supplied to the runnable catalog; catalog generation never mints permission. It checks exact authorization revision/command/source/protected identities, immutable history, and collision absence before parent/log/output/lock writes. A3 and B3 authorizations are distinct: B3 validates the A3 report against its own A3 authorization while launching under separate B3 authorization. Immediately before final publication, the exact A3 report/OK marker and canonical B3 report are re-read, hash/strict-validated, and bound into an attempt-3 final schema with both histories, phase paths/hashes/contracts, and both authorizations. B3 additionally requires canonical A3 success and exact step-2 resume bindings. Attempt-2 commands and `--attempt 2` become non-runnable; only explicit non-default A3/B3 catalog steps are added. Budgets remain `2/2700s` and `1/1500s`, total `3/4200s`, with retry/fallback `none`. No real A3/B3 execution is authorized by design or implementation work; exact same-revision Reviewer PASS, Test Manager GREEN, and fresh phase-specific operator authorization remain mandatory.
