# Story 13.3b-5 — Architect (decide model-4bit production path: Option A vs B)

## Context
BA 13.3b-5 STOPPED (legitimate): smoke-train needs on-disk `model-4bit/` dir, but the remap script (`scripts/remap_ds4_nn_weights.py`) only remaps IN-MEMORY (returns a dict). There is no path to produce on-disk `model-4bit/` that `mlx_lm.lora --model` can load. This is the "missing slice" between 13.3b-4 (remap exists, in-memory only) and 13.3b-5 (smoke-train).

## Architect job — pick ONE option + spec the exact wiring (micro-revision, not a full rewrite)

### Option A — convert-side `apply` subcommand (ADR 0026 convert-only lineage)
- Add an `apply`/`convert` subcommand to `scripts/remap_ds4_nn_weights.py` that reads `hf-f8shim`, runs `remap_weight_dict` over real payload, and WRITES a remapped on-disk dir (nn-port keys, `model_type=deepseek_v4_nn`, sidecars/index).
- Then `mlx_lm.convert -q --model <remapped-dir> --mlx-path model-4bit` quantizes the BF16 attention (skips FP4 experts via `to_quantized`-absent).
- NO FROZEN/nn-port edit. Convert-side only (ADR 0026 §5 lineage).

### Option B — load-side: wire remap into nn `Model.load_weights` (ADR 0025 nn-port territory)
- Add a `load_weights` override to `deepseek_v4_nn.Model` (line 528) that mirrors the frozen precedent at `deepseek_v4.py:2367` (frozen `Model.load_weights` calls `remap_shimmed_ckpt_keys` + canonicalize pre-pass).
- Call `remap_weight_dict` (or the rename logic) as a pre-pass, then delegate to `nn.Module.load_weights`.
- Then `mlx_lm.convert --model hf-f8shim -q --mlx-path model-4bit` works directly: plugin installs nn port → nn `Model.load_weights` remaps ckpt-native keys → quantize BF16 (skip FP4 experts) → write `model-4bit/`.
- One-step pipeline (no intermediate ~162GB dir).

## Recon already done by supervisor (facts for your decision)

1. **nn port `Model`** (`deepseek_v4_nn.py:528`) has `sanitize` (line 547) but **NO `load_weights` override** → inherits stock `nn.Module.load_weights` (strict). This is why ckpt-native keys fail to load.
2. **Frozen precedent** `deepseek_v4.py:2367` `Model.load_weights` DOES call `remap_shimmed_ckpt_keys` (line 2373-2374) + `_canonicalize_weight_key` pre-pass when `forward_parity_fixture is None`. The nn port should mirror this shape.
3. **`DeepseekV4FP4Experts`** (`deepseek_v4_nn.py:342`) extends `nn.Module` with NO `to_quantized` method → `mlx_lm.convert -q` / `nn.quantize` SKIPS it (experts stay FP4, dequant at forward per ADR 0024). So `mlx_lm.convert -q` correctly quantizes only BF16 attention, leaving FP4 experts intact. Confirmed.
4. **Disk**: only **210Gi** free on `/Volumes/Data NVME`. The shimmed ckpt is 162GB. Option A's intermediate remapped dir (~162GB BF16+FP4) likely WON'T FIT alongside the existing shimmed ckpt. **Option B avoids this** (no intermediate dir — remap happens in-memory at load time inside `mlx_lm.convert`).
5. **`scripts/remap_ds4_nn_weights.py:329`** `remap_weight_dict` is the in-memory remap (returns a dict, no `save_file`/`mkdir`). It already does the full §5.1 rename table (verified AC1 r2: 1460 keys match nn `model.parameters()` exactly).
6. **Shim** `scripts/shim_ds4_safetensors.py:361` `copy_sidecars` already writes `model_type="deepseek_v4_nn"` on the shimmed ckpt → `mlx_lm.load`/`mlx_lm.convert` sees `deepseek_v4_nn` and uses the nn port via the plugin.
7. **lora-config.json** keys (`self_attn.q_a_proj`/`q_b_proj`/`kv_proj`) match nn port leaf names exactly (BA Q4 resolved) → no lora-key remap needed.
8. **Dataset EXISTS** at `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096/` (train=15170/valid=819/test=824). BA Q2 resolved.

## Your decision should weigh
- **ADR consistency**: Option A matches ADR 0026 convert-only contract; Option B is an nn-port edit under ADR 0025.
- **Frozen precedent**: Option B mirrors the existing frozen `Model.load_weights:2367` shape — it's the established pattern.
- **Disk**: Option A needs ~162GB intermediate (210Gi free — tight/risky); Option B needs ~0 intermediate.
- **Pipeline steps**: Option A = 2-step (remap apply → convert -q); Option B = 1-step (convert -q does everything).
- **Correctness**: both end at the same `model-4bit/` (BF16 attention quantized to 4-bit, FP4 experts preserved).
- **Scope risk**: Option B touches `deepseek_v4_nn.py` (nn-port, owned by ADR 0025 — sanctioned). Option A is purely convert-side.

## Deliverable — ONE micro-revision file
Write `agent-output/cmux-13-3b/architecture-13-3b-5-model-4bit-path.md` with:
1. **Decision**: Option A or B (with rationale referencing the facts above).
2. **Exact wiring spec** for the Coder slice:
   - If Option A: the new `apply` subcommand signature, the writer (per-shard streaming? single-shard?), the output dir layout (config.json with `model_type=deepseek_v4_nn`, `model.safetensors.index.json`, shards), and how `mlx_lm.convert -q` then consumes it.
   - If Option B: the exact `load_weights` override to add to `deepseek_v4_nn.Model` (line ~528/547), which remap function to call (`remap_weight_dict` from the remap script, or a refactored shared helper), where it imports from, and how to mirror `deepseek_v4.py:2367`'s shape (pre-pass + canonicalize).
3. **Scope guard**: confirm NO FROZEN `deepseek_v4.py` body edit (ADR 0017/0024/0026). If Option B, confirm it's nn-port-only (ADR 0025 sanctioned).
4. **AC for the Coder slice**: `mlx_lm.convert --model /Volumes/Data NVME/mlx-ft/ds4/hf-f8shim -q --mlx-path /Volumes/Data NVME/mlx-ft/ds4/model-4bit` succeeds; `model-4bit/` exists with `config.json` (`model_type=deepseek_v4_nn`), quantized weights, FP4 experts preserved (not re-quantized); load `model-4bit` via `mlx_lm.load` + forward finite; FROZEN bodies byte-intact.
4. **New slice name**: e.g. 13.3b-4b (model-4bit production) — a prerequisite folded into 13.3b-5 before the smoke-train can run.

## Pre-flight read
- `agent-output/cmux-13-3b/ba-13-3b-5-stop.md` (full BA finding + recon evidence index).
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py:528` (Model), `:547` (sanitize), `:342` (FP4Experts), `:206` (AttentionNN).
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py:2367` (frozen `load_weights` precedent with `remap_shimmed_ckpt_keys`).
- `scripts/remap_ds4_nn_weights.py:329` (`remap_weight_dict` — the in-memory remap).
- `scripts/shim_ds4_safetensors.py:361` (`copy_sidecars` writes `model_type=deepseek_v4_nn`).
- `agent-output/cmux-13-3b/architecture.md` §5.1-§5.4 (convert rename table).
- ADR 0024 (FP4 dequant), ADR 0025 (nn-port ownership), ADR 0026 (additive FROZEN convert-only).

## Build
`source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first). Caveman ultra default; byte-exact exempt. NO test runs needed — this is a design decision + spec. Echo `{"status":"ok","role":"Architect"}` in surface:82 only when the micro-revision file is written.
