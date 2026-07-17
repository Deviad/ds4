# DS4 Training Next Move

> **PIVOT — 2026-06-22 (Head-of-AI due-diligence).** Strategic pivot to Path A:
> local MLX QLoRA on M3 Ultra. Forward-parity marker `.deepseek-v4-forward-
> parity-ok` RETIRED (6-STOP chain established no hardware path on M3 Ultra — see
> `docs/adr/0022-strategic-pivot-local-mlx-qlora-parity-marker-retired.md`).
> Replaced by ADR 0023 post-fuse generation coherence cross-check
> (`docs/adr/0023-post-fuse-generation-coherence-cross-check.md`). Epic 12
> Track-B-as-gold arc superseded by Epic 13 Path A slices 13.0→13.6. Door to
> CUDA ALREADY OPEN: C engine ships `ds4_cuda.cu` (NVIDIA / DGX Spark) +
> `ds4_rocm.cu` + `rocm/` (Strix Halo / ROCm) backends today; fused GGUF is
> backend-agnostic. See `docs/backlog.md` Epic 13 for full plan.

## Current execution gate — Story 14.5a synthetic closeout (2026-07-17)

- Resource observer repair and immutable attempt-2 repin are synthetically closed: independent Reviewer r8b **PASS**, Test Manager r8b **GREEN**, canonical six-file suite `508 passed, 3 skipped` (`1 warning`, `2 subtests passed`).
- Attempt 1 remains immutable five-file historical evidence: `agent-output/cmux-14-5/phase-a-log.txt`, `agent-output/cmux-14-5/phase-a-report.json`, `agent-output/cmux-14-5/pilot-report.json`, `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-phase-a-fail`, and `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-fail`; manifest verification passed `5/5`. It cannot satisfy attempt-2 gates.
- Attempt 2 remains fixed A2/B2 namespaces with collision-before-write and one-attempt semantics. No cleanup, overwrite, suffix allocation, retry, fallback, or attempt 3 is allowed.
- Path A remains permanently frozen/STOP. Real A2 remains blocked until fresh explicit operator authorization. B2 remains blocked until A2 completes, its exact evidence is verified, and separate explicit operator authorization is given.
- No real model/dataset access, training, inference, commit, or push occurred. Do not claim convergence, quality, optimizer/RNG/dataset-cursor/scheduler/global-step continuity, or full-training readiness. Later claims remain limited to evidence actually observed, such as adapter-weight equality/continuity, bounded throughput/readiness, or real execution.

Status (prior to pivot, preserved for provenance): MLX FP8 safetensors emulation now decodes `F8_E4M3` weights and `F8_E8M0` scales to BF16/F32. The two-shard probe is MLX-loadable, and the full gated FP8 shim rewrite completed. `convert-shimmed` now reaches the next blocker: project-controlled `mlx_lm.models.deepseek_v4` exists with import/config parsing, tiny partial forward fixtures, and a bounded real-mode `Model` that can `load_weights()` from standard Transformers-compatible names and run forward for proven synthetic configs including multi-head/grouped attention (`num_key_value_heads=1`, `o_groups>=1`), synthetic top-k MoE (`n_routed_experts<=4`), hc_mult=4 stream mixing with synthetic final HyperHead collapse, stacked multi-layer forward (`num_hidden_layers∈{1,2,3}`, hc_mult=1 only), and the proven tiny cache-less single-head `compression_ratio=4` CSA compressor/indexer attention subset. Full DeepSeek V4 load/forward parity (real-scale multi-layer `num_hidden_layers=43`, multi-layer with hc_mult>1, real-scale packed FP4/256-expert top-6, and generation/KV cache smoke) is still absent. Local Torch/MPS one-step train/export and DS4 inspect gates also pass; raw Torch/MPS PEFT remains blocked for the current F8 checkpoint.

## Completed since last status

- **Plan sanity check (2026-06-18):** `xhigh-reviewer` verified all 9 Story 11.4/11.10/11.11/11.12/11.13/11.14/11.15/11.16 doc claims against actual code/tests — all OK, no blockers/majors/minors. Report at `review-xhigh-plan-sanity-check.md`.
- **Story 11.14 recon (2026-06-18):** supervisor-authored scout at `agent-output/story-11-14-mlx-compressor-scout.md` maps the pure-Python `tiny_compressor_indexer_attention_reference` → MLX port, lists all 10 compressor/indexer weight keys, the 3 fail-closed insertion points (`_attention_mlx:327`, integrated fixture `:511`, `_validate_real_mode:1282`), the 7 reusable MLX helpers, the gap to fill (full-length RoPE + softmax-gated windowed pool), and a 3-slice TDD breakdown (11.14a compressor / 11.14b indexer / 11.14c integrated + Model relaxation) at ≤1e-5 tolerance. CSA reference is stateless per-forward, so 11.14 can land cache-less (stateful Ca overlap → 11.16).
- **Subagent model routing (2026-06-18, revised):** BA/Architect = `anthropic/claude-opus-4-8`; Coding + code review = `openai-codex/gpt-5.5`; Utility = `anthropic/claude-sonnet-4-6`. Recorded in `AGENT.md` § Subagent model routing, `docs/backlog.md` Key decisions, and this file § Subagent model policy.
- **Plan sync (2026-06-18):** `docs/backlog.md` Story 11.4/11.10/11.11/11.12/11.13 status checkboxes refreshed to reflect slices 1–11; Stories 11.14/11.15/11.16 added with full user stories + Given/When/Then AC; Story 11.14 now carries the 11.14a/b/c TDD slice breakdown. `training-next-status.md` items 4/16/17/18 synced + recommended-next-slice pointer.
- **Story 11.15a (2026-06-18, awaiting review/test manager):** Added a deterministic header-only `classify_checkpoint_expert_packing()` wrapper for the verified real checkpoint packing: routed `I8 + F8_E8M0`, shared `F8_E4M3 + F8_E8M0`, FP4 absent, routed block-size discrepancy vs declared `[128,128]`, shared 128×128 2-D block ambiguity, and `can_decode_payload=False`. `dequantize_expert_packed()` and `.deepseek-v4-forward-parity-ok` remain fail-closed. Plan wording that still says "packed FP4" is now flagged for doc-owner correction to the verified real packing.
- Python environments are declarative under `python-envs/` and installed via `uv venv --seed` + `pip install -e`.
- `AGENT.md` records mandatory parallel GPT-5.5 xhigh reviewer subagent, TDD red-green-refactor, clean/decoupled/low-complexity code, and `uv venv --seed` policy.
- Shell command emission includes `set -euo pipefail`, per-step subshell grouping, and safe `setup-env` venv existence grouping after successful `cd`.
- `/Volumes/Data NVME/mlx-ft/ds4/.venv-torch` exists with Python 3.12.11, torch 2.12.0, PEFT 0.19.1, and MPS available.
- `torch-smoke` performs tiny PEFT attach/export plus DS4 converter dry-run.
- Added `torch-one-step-lora` to `local-torch-mps` backend.
- `local-torch-mps` gates now fail closed if `torch.backends.mps.is_available()` is false.
- `run-command --execute` now requires explicit CLI `--yes`; ambient `DS4_FT_YES=1` no longer bypasses approval.
- `cpu-check ds4-adapter-inspect` now executes real `./ds4 --inspect -m <ds4flash.gguf> --lora <adapter>` and validates adapter/GGUF/binary prerequisites instead of emitting comments.
- Added and ran `torch-real-v4-feasibility`, a safe real-checkpoint feasibility gate that uses `init_empty_weights`, real config/tokenizer/checkpoint metadata, MPS enforcement, and safetensors header scanning without full weight loading.
- Inventoried architecture references for recreating `mlx_lm.models.deepseek_v4` locally: installed MLX-LM includes `deepseek_v3.py`, `deepseek_v32.py`, and `mla.py`; installed Transformers includes `modeling_deepseek_v4.py`; DS4 provides tensor/layout/runtime reference code for Flash.
- Implemented MLX FP8 safetensors emulation in `scripts/shim_ds4_safetensors.py`:
  - `F8_E4M3` model weights decode to BF16/F32.
  - `F8_E8M0` scale tensors decode to BF16/F32.
  - Special values are handled: `F8_E8M0` byte `0` maps to `2^-127`, byte `255` maps to NaN, and `F8_E4M3FN` bytes `0x7f`/`0xff` map to NaN.
  - `local-mlx` default command emission routes through `fp8-shim-probe -> fp8-shim -> convert-shimmed` and omits raw `convert`; full `fp8-shim` requires `.fp8-shim-probe-ok`.
- Ran a real two-shard MLX FP8 emulation probe:
  - converted first two DeepSeek V4 Flash shards to BF16 at `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim-probe`
  - `mx.load()` succeeded on both rewritten shards
  - observed BF16 tensors: `embed.weight`, `layers.0.attn.wq_a.weight`, `layers.0.attn.wq_b.weight`, `layers.0.attn.wkv.weight`
- Ran the full gated MLX FP8 shim rewrite after the probe marker:
  - output: `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim`
  - shard count: 46
  - size: about 162.5 GiB
  - `model.safetensors.index.json` `metadata.total_size` updated to the rewritten data-byte total
- Implemented an I8 block-scale expert dequant helper (`dequantize_i8_block_scale`) in `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_dequant.py`, proven against a tiny PyTorch reference on synthetic data; `dequantize_expert_packed('i8')` remains fail-closed and no real checkpoint payload bytes were decoded.
- Added HCA/CSA compressor and CSA indexer scoring/top-k forward parity fixtures in `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_attention_spec.py`, compared against Transformers `DeepseekV4HCACompressor`, `DeepseekV4CSACompressor`, and `DeepseekV4Indexer` (max abs error within tolerance), integrated as additional partials in `deepseek-v4-forward-parity-check`; stateful cache/overlap and full layer integration remain fail-closed.
- Added hc_mult=2 hyperconnection residual mixing parity (now with non-symmetric hypernetwork weights and nonzero base/scale), final hyperhead collapse parity, and integrated one-layer hc_mult>1 attention+MoE parity against Transformers references.
- Added I8 block-scale routed-expert top-k MoE forward parity fixture, integrated into `deepseek-v4-forward-parity-check` as an additional partial; MLX real mode now also supports synthetic I8 routed experts with BF16 block scales in the bounded single-layer path, compared against a Transformers reference loaded with PyTorch-dequantized float expert weights. Story 11 slice8 extends pure MLX `_moe_mlx()` from top-1 to synthetic top-k routing (`num_experts_per_tok>1`) for small unquantized fixture weights and the existing I8 block-scale path, matching the pure-Python `tiny_topk_moe_forward()` selected-score normalization and `routed_scaling_factor`; slice9 extends checkpoint/Model shape compatibility to those same synthetic MoE bounds (`n_routed_experts<=4`, `num_experts_per_tok<=n_routed_experts`) for raw-weight `fp4` fixtures and I8 block-scale shapes, while packed FP4/real 256-expert top-6 remains fail-closed.
- Converted the bounded real `Model` to MLX array operations for the proven single-layer/sliding-attention/top-k-MoE path, now including positive `hc_mult` stream mixing with synthetic hc_mult=4 parity against the pure-Python integrated layer reference, final HyperHead collapse before optional final norm/lm_head, synthetic multi-head/grouped-attention Model forward parity for `num_key_value_heads=1` and `o_groups=2` with enabled sink logits, synthetic top-k MoE (`n_routed_experts<=4`, `num_experts_per_tok<=n_routed_experts`, unquantized/I8), and synthetic multi-layer (`num_hidden_layers∈{1,2,3}`) stacked forward parity with per-layer weights keyed `layers.{i}.*` and final collapse applied once. `load_weights()` keeps accepting MLX arrays and fail-closes incomplete `hc_head.*` sets and missing/wrong per-layer weights. hc_mult=1 single-head Transformers reference parity remains verified in the torch venv (max abs error ≤ 1e-5). Multi-layer with hc_mult>1 fails closed in both `Model._validate_real_mode()` and `bounded_model_expected_shapes()` because no stacked reference proves it. `import mlx.core` is lazy so the default `make finetune-test` environment still passes all 236 tests when mlx is absent.
- Started the multi-head/grouped attention port with a TDD-only Python reference slice: `DeepSeekV4AttentionSpec.flash_mlx_safetensors_shapes()` records Flash safetensors/MLX `[out, in]` formulas for the full dry-run-style config `hidden_size=4096`, `num_attention_heads=64`, `num_key_value_heads=1`, `head_dim=512`, `q_lora_rank=1024`, `o_lora_rank=1024`, `qk_rope_head_dim=64`, `o_groups=8`; tests explicitly contrast these with DS4 transposed matvec layout. The pure-Python synthetic reference covers q_a/q_b, per-head q RMSNorm (with a flattened-q-norm regression guard), single-KV broadcast, stable sink extra-bucket softmax, tail-only RoPE/inverse RoPE, independent grouped `o_a` blocks, and `o_b`; MLX now has isolated tail-only RoPE helpers that use `qk_rope_head_dim`, preserve the no-RoPE prefix, match pure-Python table generation, and broadcast over 4D multi-head-shaped tensors; `_attention_mlx()` now supports a bounded multi-head path for `num_key_value_heads=1`, per-head sink logits as an appended softmax bucket, tail-only RoPE/inverse RoPE, per-head q RMSNorm, KV broadcast, causal/sliding masks, and grouped output projection for `o_groups>1` using independent per-group `o_a` blocks followed by `o_b`; `Model._validate_real_mode()` now allows that proven synthetic subset while missing/wrong-shape sinks, compressors/indexers, non-1 KV heads, invalid head/group divisibility, unsupported real-scale MoE settings (including real 256-expert/top-6 FP4), and bad grouped projection shapes remain fail-closed. Checkpoint/Model shape compatibility now uses the proven multi-head/grouped attention formulas for `q_b`, one-KV `wkv`, per-head sinks, grouped `o_a`, and `o_b` while keeping full checkpoint payload execution blocked.
- Added checkpoint index/header-only validation for `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/model.safetensors.index.json`; it reads only JSON index plus safetensors headers, maps layer-0 checkpoint tensor names to bounded real-mode keys, checks expected dtype/shape metadata, and explicitly synthesizes missing `mlp.gate.e_score_correction_bias` as a zero buffer using Transformers proof. Header coverage for the bounded subset is now `coverage_ok=true`.
- Added selected-tensor payload loader planning/execution with strict byte budgets. Dry-run reads headers only; execute mode requires `max_bytes` and loads only selected tensors through a bounded safetensors `data_offsets` reader into MLX arrays. A real checkpoint selected-payload smoke loaded only `q_norm.weight` plus synthetic `mlp.gate.e_score_correction_bias` under a 4096-byte budget (3072 selected bytes, one shard header, no full-shard load). The bounded checkpoint-to-Model integration helper now dry-runs or execute-loads the complete bounded key set, computes expected canonical tensor shapes from `ModelArgs`, reports `model_shape_compatible`, and calls `Model.load_weights()` only after plan validation/budget/model-shape checks pass; tiny safetensors tests cover compatible execute mode and mismatch fail-closed behavior before payload/model construction. The real checkpoint was used only for header-only dry-runs estimating 1,356,878,296 selected bytes across 34 canonical keys (33 real tensors plus synthetic zero bias): the full config remains blocked by unbounded `num_hidden_layers=43`/full MoE, while a diagnostic real-attention/bounded-MoE shape config has attention shapes compatible and only MoE/gate mismatches. Full-load/forward blockers remain.
- Added hc_mult=2 hyperconnection parity fixture (`run_tiny_hyperconnection_hc2_transformers_fixture`) in `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_attention_spec.py`, compared against Transformers `DeepseekV4HyperConnection` (post/comb/collapsed outputs within tolerance), integrated as an additional partial in `deepseek-v4-forward-parity-check`; full decoder-layer residual placement remains fail-closed.
- Implemented Story 11.11 hyperconnection/hyperhead slice: generalized `_integrated_layer_forward` to hc_mult>1 stream mixing, added `run_integrated_layer_hc_mult_fixture` and `run_tiny_hyperhead_transformers_fixture` compared against Transformers `DeepseekV4Model` and `DeepseekV4HyperHead` (max abs error within tolerance), integrated as additional partials; full model load/forward remains the next blocker.
- Added CSA Lightning indexer scoring/top-k forward parity fixture (`tiny_csa_indexer_forward` / `run_tiny_csa_indexer_scorer_fixture`) in `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_attention_spec.py`, compared against Transformers `DeepseekV4Indexer`/`DeepseekV4IndexerScorer` (max abs error within tolerance), integrated as an additional partial in `deepseek-v4-forward-parity-check`; stateful cache overlap and full layer integration remain fail-closed.
- Added an integrated pure-Python CSA compressor/indexer attention reference (`tiny_compressor_indexer_attention_reference` / `run_tiny_compressor_indexer_attention_fixture`) in `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_attention_spec.py` that composes the validated CSA compressor + Lightning indexer components, applies the `hca_block_bias` causal compressed-block threshold combined with the indexer top-k gather mask, and produces a per-query compressed attention output using the spec's own `compressor_shapes()`/`indexer_shapes()` weight names. Gating follows the trusted Transformers softmax semantics (the brief's "sigmoid" was an error; `sigmoid` only appears in the HyperConnection/HyperHead path in Transformers, never the compressor). 8 TDD tests cover output shapes, causal masking, all-masked early-query zero output, compressor agreement with the validated helper, top-k restriction, and fail-closed rejection. The real compression_ratio=4 MLX port of the compressor/indexer is deferred to a follow-up slice; `_attention_mlx()`/`Model._validate_real_mode()` now fail closed for ALL nonzero compression ratios. (A divergent synthetic `compression_ratio=2` block-sparse MLX path that a parallel worker had added was reverted during reconciliation, since it did not correspond to the real ratio=4 architecture; `_attention_mlx` raises `NotImplementedError("... compressors/indexers")` for both ratio=2 and ratio=4, locked by regression tests.) Component Transformers parity (`DeepseekV4CSACompressor`, `DeepseekV4CSAIndexer`, `DeepseekV4HCACompressor`) re-confirmed in the torch venv.
- Reconciled Story 11 slice 11 after a duplicate-dispatch collision: a divergent synthetic `compression_ratio=2` MLX block-sparse path (`_compressed_attention_mlx` and three helpers) was removed; `_attention_mlx()`, the integrated-layer fixture, and `Model._validate_real_mode()` now fail closed for ALL nonzero compression ratios (both ratio=2 and ratio=4), locked by regression tests. The reviewed-clean ratio=4 pure-Python `tiny_compressor_indexer_attention_reference` and its 8 tests were kept; `make finetune-test` is now 236 OK (54 skipped).
- Updated the DS4 Fine-Tuning plan to match reality through Story 11.14: multi-head/grouped attention proven; sinks/inverse-RoPE/compressor-indexer pure-Python reference proven; hc_mult>1 + HyperHead + multi-layer hc_mult=1 proven; synthetic top-k MoE proven; and the tiny cache-less single-head `compression_ratio=4` CSA compressor/indexer path now runs in MLX with `compress_rope_theta` parity. Stories 11.15 (real-scale 43-layer/packed-FP4/256-expert/top-6 MoE) and 11.16 (KV cache + generation parity) remain before `.deepseek-v4-forward-parity-ok` can be written.
- Ran `convert-shimmed`; it no longer fails on FP8 safetensors loading or missing `mlx_lm.models.deepseek_v4` import, but fails on the remaining full forward-parity gate:

```text
.deepseek-v4-forward-parity-ok: missing DeepSeek V4 architecture gate marker
```
- Ran `torch-one-step-lora` with `--execute --yes`:
  - loaded real DeepSeek V4 Flash config/tokenizer locally
  - read real `anthropomorphic-frankenmerge/mlx-4096/train.jsonl` text
  - trained one tiny PEFT LoRA step on MPS (latest observed `loss 11.79633617401123`)
  - exported PEFT adapter via `save_pretrained`
  - wrote DS4-shaped internal-layer adapter
  - converted it to DS4 safetensors
- DS4 inspect passed against immutable base GGUF:

```text
ds4: LoRA adapter validated: /Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/torch-one-step-lora/adapter.ds4.safetensors matched_pairs=1 ignored_tensors=0 rank=8 alpha=16
```

## Real DeepSeek V4 Flash local feasibility result

Report:

```text
/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/torch-real-v4-feasibility/training_feasibility_report.json
```

Key evidence:

```text
local_training_feasible: false
mps_available: true
empty_real_model_constructed: true
target_modules_present: model.layers.0.self_attn.kv_proj, model.layers.0.self_attn.q_a_proj, model.layers.0.self_attn.q_b_proj
safetensors_shards: 46
safetensors_tensors: 69187
dtype_counts: BF16=433, F32=417, F8_E4M3=375, F8_E8M0=34167, I64=3, I8=33792
stored_bytes_estimate: 159609485896
bf16_base_bytes_estimate: 316138866596
mps_recommended_max_memory_bytes: 429496729600
full_weight_load_attempted: false
reason: checkpoint contains F8_E4M3, F8_E8M0 tensors; raw Torch/MPS PEFT cannot train these quantized weights directly
heuristic_limitations: this gate proves hard blockers such as unsupported F8 dtypes and missing modules; for dequantized/non-F8 checkpoints it is not a full memory proof because activations/optimizer state and MPS allocator fragmentation require follow-up execution gates
```

Conclusion: raw Transformers/PEFT local training of the real DeepSeek V4 Flash checkpoint is not feasible on Torch/MPS without an F8-aware training path or dequantized training checkpoint. Keep local DS4 runtime LoRA and converter work; use explicit `remote-cuda` or DS4-native training path for real training.

## Produced artifacts

- `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/torch-one-step-lora/peft-export/adapter_model.safetensors`
- `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/torch-one-step-lora/adapter.safetensors`
- `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/torch-one-step-lora/adapter.ds4.safetensors`

## Validation

- `torch-env-create`: OK
- `torch-env-check`: OK
- `torch-smoke`: OK
- `torch-one-step-lora`: OK, with mandatory MPS availability check
- `torch-real-v4-feasibility`: OK, result `local_training_feasible=false` with F8 dtype blocker
- `fp8-shim-probe`: OK, converted two shards and `mx.load()` read them
- `fp8-shim`: OK, full 46-shard BF16 shim rewrite completed
- `convert-shimmed`: BLOCKED by Story 11 forward parity gate; import/tiny-config, MTP exclusion, mapping, dequant parity, and LoRA target gates now exist; `.deepseek-v4-forward-parity-ok` is absent
- `./ds4 --inspect -m /Users/spotted/projects/ds4/ds4flash.gguf --lora .../adapter.ds4.safetensors`: OK
- `python3 scripts/finetune_ds4.py run-command --backend cpu-check ds4-adapter-inspect --adapter-ds4 ... --ds4-root . --ds4-gguf /Users/spotted/projects/ds4/ds4flash.gguf --execute --yes`: OK
- `make lora-test`: OK
- `make finetune-test`: 250 tests OK (60 skips when torch/transformers/mlx/safetensors unavailable)
- `make -B cpu`: OK
- `make -B ds4`: OK
- `git diff --check`: clean

## Remaining stories and next tasks

1. **Story 11.1 — Build a DeepSeek V4 architecture mapping dossier.**
   - Acceptance: compare Transformers DeepSeek V4, MLX-LM `deepseek_v3.py`/`deepseek_v32.py`/`mla.py`, DS4 tensor/layout code, and the shimmed checkpoint index; every tensor family is mapped or marked fail-closed. A shallow v3/v32 rename is explicitly rejected.
2. **Story 11.2 — Add a reproducible vendored MLX-LM patch path.**
   - Status: partially done. `setup-env` installs a reproducible `.pth` hook, and the real MLX venv now imports project-controlled `mlx_lm.models.deepseek_v4` from `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`.
   - Remaining acceptance: implementation must progress beyond partial fixture support only after real tiny load/forward parity tests for integrated attention with cache/sinks/compressor/indexer, full decoder-layer hyperconnection residual mixing, packed FP4/I8 expert dequant, and expert kernels. `deepseek-v4-forward-parity-check` now writes `deepseek-v4-forward-parity-partial.json` for proven embedding/RMSNorm/head, sliding-attention-no-compressor, tail-RoPE, attention sink extra-logit scoring, corrected sliding cache return/persist semantics, output inverse-RoPE tail, HCA block-bias/no-bias metadata, CSA top-k indexer gather-mask, final-hyperhead-hc2-collapse, hyperconnection-hc1, nonzero hc_mult=2 pre/post/comb math, top-k unquantized MoE, hash-router `tid2eid` unquantized MoE, and an **integrated one-layer attention+MoE fixture compared against Transformers `DeepseekV4Model`** (max abs error ~2.5e-6), fails closed, and does not write the final marker.
3. **Story 11.3 — Implement tiny-config construction tests before real checkpoint conversion.**
   - Status: partially done. `ModelArgs.from_dict` accepts tiny/real Flash config fields and fails closed for unsupported `expert_dtype`; `Model` now supports a bounded real-mode constructor/forward/load_weights for the proven synthetic subset including multi-head/grouped attention, synthetic top-k MoE for small unquantized/I8 configs, hc_mult=4 stream mixing, final HyperHead collapse, and stacked multi-layer forward for `num_hidden_layers∈{1,2,3}` with per-layer `layers.{i}.*` weights, and raises `NotImplementedError` for real configs outside that subset (real Flash `num_hidden_layers=43` remains fail-closed). Explicit fixture modes for embedding/RMSNorm/head and integrated-layer remain available.
4. **Story 11.4 — Port attention/compressor/hyperconnection semantics.**
   - Status (slices 1–11): multi-head/grouped attention fully proven — partial/tail RoPE, per-head sinks as extra softmax bucket, grouped `o_a`/`o_b` output projection, output inverse-RoPE, and real Flash attention shape compatibility. `hc_mult>1` hyperconnection, final HyperHead collapse, and `hc_mult=1` multi-layer residual stacking (slice 10) are proven. A pure-Python `compression_ratio=4` CSA compressor/indexer reference is proven (slice 11); hyperconnection/hyperhead fixtures moved to Story 11.11.
   - Acceptance remaining: real-scale integration (→ Story 11.15) and incremental-cache attention parity (→ Story 11.16) before the forward marker is written.
5. **Story 11.5 — Port MoE/router/expert semantics with explicit FP4/I8 risk handling.**
   - Acceptance: routed/shared expert module mapping is explicit; packed `I8`/FP4 expert families and companion `F8_E8M0` scales either match trusted DS4/Transformers dequant semantics or fail closed with specific diagnostics.
6. **Story 11.6 — Add conversion-level tensor mapping and dequantization tests for the shimmed checkpoint.**
   - Status: partially done. The tensor-name scanner classifies real checkpoint families, proof-gates `mtp.*` stripping via `.deepseek-v4-mtp-exclusion-ok`, removes stale `.deepseek-v4-mapping-ok` on failed reruns, writes structured JSON markers bound to the current shimmed index SHA-256, checks recognized MoE family completeness, and requires `.deepseek-v4-dequant-parity-ok` before `convert-shimmed`; dequant now includes FP8 shim parity, an explicit I8 affine primitive, packed FP4/I8 metadata risk accounting, and a real checkpoint header scan that classifies routed expert weights as `I8` with paired `BF16` scales (block size 16 along axis 1) and shared experts as `BF16` weights with `BF16` scales, while packed decode remains fail-closed.
   - Acceptance: packed I8/FP4 expert fixtures still require trusted parity before `convert-shimmed` is accepted; full forward parity remains required.
7. **Story 11.9 — Restrict MLX LoRA training targets to DS4-supported modules.**
   - Status: partially done. `mlx-lora-targets-check` writes `lora-targets.json`, DS4-safe `lora-config.json` with explicit `lora_parameters.keys`, and structured `.mlx-lora-targets-ok` bound to the config hash; MLX training commands now include `--config lora-config.json`, and prerequisites reject forged/stale allowlist markers.
   - Remaining acceptance: real MLX training still waits for a loadable DeepSeek V4 MLX model and conversion/forward parity gates.
8. **Story 10.1 / 11.7 — Add/vendor `mlx-lm` DeepSeek V4 support and rerun conversion.**
   - Acceptance: `convert-shimmed` no longer fails with `mlx_lm.models.deepseek_v4` missing; `/Volumes/Data NVME/mlx-ft/ds4/model-4bit` is created and loads through MLX/MLX-LM.
   - Current gate: `convert-shimmed` validates structured JSON gate markers for import, tiny-config, MTP exclusion, mapping, dequant parity, and forward parity; `.deepseek-v4-mapping-ok` must match the current `hf-f8shim/model.safetensors.index.json` SHA-256, `.deepseek-v4-mtp-exclusion-ok` is directly required when the index contains `mtp.*`, and any future `.deepseek-v4-forward-parity-ok` must be bound to a full parity report hash. The remaining blocker is full forward parity; the current `deepseek-v4-forward-parity-check` records partial fixture evidence, fails closed, and removes stale `.deepseek-v4-forward-parity-ok`.
9. **Story 10.2 — Convert the full shimmed checkpoint to a local MLX base.**
   - Acceptance: `/Volumes/Data NVME/mlx-ft/ds4/model-4bit` exists, loads through MLX/MLX-LM, and raw `convert` remains diagnostic-only.
10. **Story 10.3 — Run local MLX smoke training and generation.**
   - Acceptance: bounded `smoke-train` writes `$MLX_WORK/adapters-smoke`; smoke generation loads the adapter and produces non-empty output; 2048-token fallback is recorded separately if needed.
11. **Story 10.4 — Validate real adapter deployment through DS4 before full training.**
   - Acceptance: a real smoke adapter converts to DS4 canonical safetensors and `./ds4 --inspect -m ds4flash.gguf --lora adapter.ds4.safetensors` passes with matched internal target pairs.
12. **Story 5.1 / 10.4 — Produce a portable DS4 adapter release package.**
   - Acceptance: package includes original standard adapter safetensors, DS4-canonical adapter, config/metadata, rank/alpha/target coverage, checksums, dataset manifest reference, backend/version pins, DS4 inspect evidence, and DS4 generation smoke prompt/output.
13. **Story 10.5 — Keep remote CUDA as an explicit fallback path.**
   - Acceptance: remote commands require explicit `--backend remote-cuda --execute --yes`, prove DeepSeek V4 Flash architecture plus F8/F8_E8M0 scale compatibility, run a real one-step LoRA forward/backward/export smoke, and produce standard PEFT safetensors before any full run.
14. **Story 10.6 — Expand DS4 runtime LoRA coverage to trained internal targets.**
   - Acceptance: target metadata is table-driven; CPU prefill/decode tests cover each target; Metal parity is added or unsupported backend/target pairs fail closed.
15. **Story 10.7 — Preserve artifacts, logs, and cleanup decisions.**
   - Acceptance: `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim` remains until `model-4bit` is created or local MLX is abandoned; status docs record artifact sizes, logs, missing `model-4bit`, and safe deletion decisions.

16. **Story 11.14 — Port compressor/indexer CSA attention to MLX (implemented / slice 12).**
   - Status: implemented for the proven tiny subset. MLX `_attention_mlx` runs `compression_ratio=4` CSA forward matching the pure-Python `tiny_compressor_indexer_attention_reference` within ≤1e-5; `_validate_real_mode`/`load_weights` accept the proven tiny CSA subset; `compression_ratio=2`, multi-head compressed attention, real scale, and stateful cache remain fail-closed; `index_topk` is a forward kwarg, not `ModelArgs`.
17. **Story 11.15 — Real-scale MoE: 43-layer, expert packing, 256-expert, top-6 routing (in progress).**
   - Slice 11.15a: header-only real expert packing classification implemented; FP4 is verified absent in the actual checkpoint, routed experts are `I8 + F8_E8M0`, shared experts are `F8_E4M3 + F8_E8M0`, and decode remains fail-closed.
   - Remaining acceptance: trusted-reference decode for the verified real formats; top-6/256 routing parity; 43-layer stacked shape compatibility; `.deepseek-v4-forward-parity-ok` stays absent until proven. (Full user story + AC in `docs/backlog.md` Epic 11 → Story 11.15; wording still needs doc-owner correction from the old FP4 premise.)
18. **Story 11.16 — KV cache and generation path parity.**
   - Acceptance: incremental token-by-token KV cache update matches the full-sequence reference within ≤1e-5; CSA cache carry across calls matches or is explicitly deferred; `smoke-generate` produces a deterministic continuation without cache-shape errors once a converted base exists; the generation gate stays blocked until proven. (Full user story + AC in `docs/backlog.md` Epic 11 → Story 11.16.)

**Recommended next slice:** Story 11.15b (trusted-reference decode for the verified real expert formats: routed `I8 + F8_E8M0`, shared `F8_E4M3 + F8_E8M0`) or Story 11.16 (KV cache + generation), depending on which blocker the operator wants to attack first. `.deepseek-v4-forward-parity-ok` must remain absent until both remaining blockers are proven.

Stop rules:

- Do not run raw Torch/MPS real DeepSeek V4 Flash training against the current F8 checkpoint; the feasibility gate says it is not locally trainable by raw PEFT.
- Do not claim MLX training is unblocked until `convert-shimmed` creates `/Volumes/Data NVME/mlx-ft/ds4/model-4bit` and `smoke-train` starts from it.
- Do not run full training until a real smoke adapter validates through DS4 inspect.
- Do not run remote CUDA commands without explicit backend selection and execution approval.

### 2026-06-17 — Point 1 architecture dossier

- Added `docs/deepseek-v4-architecture-dossier.md` documenting DeepSeek V4 Flash architecture mapping across Transformers, MLX-LM references, DS4 layout/runtime code, and the shimmed checkpoint index.
- Dossier keeps fail-closed policy: MTP remains review-required, unknown non-MTP tensor families are currently zero, and conversion/training remain blocked pending MTP policy, dequant parity, and forward parity.

## Subagent model policy

Updated 2026-06-18. Pass an explicit `model:` on every subagent dispatch.

| Role | Model | Subscription |
|---|---|---|
| **BA / Architect** | `anthropic/claude-opus-4-8` | Anthropic |
| **Coding** | `openai-codex/gpt-5.5` | OpenAI |
| **Code reviewer** (`xhigh-reviewer` agent) | `openai-codex/gpt-5.5` | OpenAI |
| **Utility** (commands, codebase scan, info extraction, web search) | `anthropic/claude-sonnet-4-6` | Anthropic |

- **Tandem gate:** no new worker slice while the current slice is unreviewed. Keep file-mutating coding dispatches serial.
- **No silent substitution:** `xhigh-reviewer` always uses `openai-codex/gpt-5.5`; do not swap without explicit user instruction.
