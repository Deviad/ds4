# Architecture micro-revision — 13.3b-5 model-4bit production path

Scope: ONE decision (Option A vs B) + exact Coder wiring. Micro-revision, not
rewrite. NO test runs. Design + spec only.

Caveman ultra. Byte-exact exempt: code, paths, commands, keys, ADR/line refs.

---

## 1. DECISION — Option B (load-side remap into nn `Model.load_weights`)

Reject A. Pick B.

Disk = hard blocker for A. `/Volumes/Data NVME` avail **226G**; `hf-f8shim`
already eats **163G** of used. Option A intermediate remapped dir = same payload
~163G (rename-only, FP4 experts copied). 226G − 163G = 63G left, then `model-4bit`
(FP4 experts preserved ≈ bulk of 163G) does NOT fit. **A is disk-infeasible.**
B = ~0 intermediate (remap in-memory inside `mlx_lm.convert`).

Other axes also favor B:

- **Frozen precedent shape.** Frozen `deepseek_v4.Model.load_weights`
  (`deepseek_v4.py:2367`) already does remap-pre-pass-then-load. B mirrors that
  shape on the nn sibling. Established pattern.
- **Vendor→package import precedent EXISTS.** Frozen `deepseek_v4.py:2372`:
  `from ds4_ft_mlx.shimmed_ckpt_key_remap import remap_shimmed_ckpt_keys`. So a
  vendor file importing a `ds4_ft_mlx.*` remap module is already sanctioned. B
  reuses this lane.
- **1-step pipeline.** B = `mlx_lm.convert -q` does everything. A = remap-apply
  writer (new mkdir/shard/index code) + convert = 2 steps, more code, more risk.
- **Single source of truth.** B shares ONE remap core; A risks rename-table drift
  (would break AC1 r2 1460-key exact match).
- **Correctness identical.** Both end at same `model-4bit` (BF16 attn → 4-bit,
  FP4 experts preserved). ADR 0024 dispatch unchanged.
- **ADR fit.** B = nn-port edit, ADR 0025 territory (sanctioned: 13.3b-5 is the
  nn-port training enablement). Confirmed nn-port-ONLY; FROZEN untouched (§3).

---

## 2. EXACT WIRING SPEC (Option B) — for Coder

### 2.0 PREREQ — flip hf-f8shim config model_type (NEW finding, blocking)

⚠️ Recon fact #6 is STALE. On-disk `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json`
has **`"model_type": "deepseek_v4"`** (verified), NOT `deepseek_v4_nn`. The dir
was shimmed Jun 17, before `copy_sidecars` (`shim_ds4_safetensors.py:357-362`)
started writing `deepseek_v4_nn`. Consequence: `mlx_lm.convert --model hf-f8shim`
routes to FROZEN object-based `deepseek_v4.Model` (`deepseek_v4.py:2247`
`class Model:` — NOT nn.Module, ADR 0025 Blocker A) → `nn.quantize` crashes
before B's override ever runs.

Fix (idempotent, one field, data not code):
```bash
python - <<'PY'
import json, pathlib
p = pathlib.Path("/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json")
c = json.loads(p.read_text())
if c.get("model_type") != "deepseek_v4_nn":
    c["model_type"] = "deepseek_v4_nn"
    p.write_text(json.dumps(c, indent=2, sort_keys=True) + "\n")
print("model_type =", c["model_type"])
PY
```
No 163G re-shim. Keys stay ckpt-native; B's override remaps at load. After flip,
plugin routes `hf-f8shim` → `deepseek_v4_nn.Model` (trainable nn port).

### 2.1 Extract shared remap core → NEW package module

`scripts/` is NOT importable from vendor (not on `sys.path`, verified). Move the
materialized in-memory remap core into a package module so BOTH the script and
the nn Model import ONE copy (no rename-table duplication):

- NEW file: `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_nn_remap.py`
  (mirror the existing `ds4_ft_mlx/shimmed_ckpt_key_remap.py` precedent file).
- Move from `scripts/remap_ds4_nn_weights.py`: `KeyRemap`, `_DROP_PATTERNS`,
  `_DIRECT_RULES`, `_EXPERT_RULE`, `_LAYER`/`_EXPERT` regex consts, `remap_key`
  (`:155`), `_zero_bias_keys` (`:188`), `_validate_expert_sources` (`:192`),
  `_get_mx` (`:139`), `_config_int` (`:147`), `_convert_value`, and
  `remap_weight_dict` (`:329`).
- `scripts/remap_ds4_nn_weights.py` re-imports them from the new module
  (`from ds4_ft_mlx.deepseek_v4_nn_remap import remap_key, remap_weight_dict, ...`)
  so the `plan` subcommand + header path (`remap_header_specs`) keep working
  unchanged. Header-only fns (`remap_header_specs`, `read_safetensors_header`,
  `_read_checkpoint_header_specs`) MAY stay in the script (they call the shared
  `remap_key`). Single source of truth for the rename table = preserved.

### 2.2 Add `load_weights` override to `deepseek_v4_nn.Model`

Target: `deepseek_v4_nn.py`, `class Model(nn.Module)` (`:528`). Currently NO
`load_weights` (inherits stock `nn.Module.load_weights` strict — verified
`Model.load_weights is nn.Module.load_weights == True`). Add an override next to
`sanitize` (`:547`). Mirror frozen `deepseek_v4.py:2367` SHAPE (remap pre-pass →
strict load), simpler (nn keys are final, no `_canonicalize_weight_key` needed):

```python
def load_weights(self, weights, strict: bool = True) -> None:
    # Story 13.3b-5, ADR 0025/0027: shimmed ckpt-native keys -> nn param tree.
    # Mirrors frozen deepseek_v4.Model.load_weights:2367 remap-pre-pass shape.
    from ds4_ft_mlx.deepseek_v4_nn_remap import remap_weight_dict
    mapping = dict(weights)
    # IDEMPOTENT GATE (critical): convert saves model-4bit with ALREADY-nn keys
    # (model.* / lm_head.*, plus nn.quantize .scales/.biases). Re-loading those
    # through remap_key would KeyError (orphan-STOP detector). Native ckpt keys
    # start with embed./head./norm./hc_head_./layers. (no model./lm_head. prefix).
    is_native = any(not k.startswith(("model.", "lm_head.")) for k in mapping)
    if is_native:
        cfg = {"n_routed_experts": self.args.n_routed_experts,
               "num_hash_layers": self.args.num_hash_layers}
        mapping, _report = remap_weight_dict(mapping, config=cfg)
    super().load_weights(list(mapping.items()), strict=strict)
```

Notes:
- `super()` of `Model` → `nn.Module.load_weights` (Model bases = `nn.Module`
  only; inner `DeepseekV4ModelNN` carries `PipelineMixin`, not `Model`).
- mlx_lm flow: `load_model` → `model.sanitize(weights)` (mtp-strip, leaves keys
  ckpt-native) → `model.load_weights(list(weights.items()), strict=strict)`. So
  override receives ckpt-native keys; `is_native` True; remap fires. ✓
- `remap_weight_dict` already drops `mtp.*` and synthesizes the 3 hash-layer
  `e_score_correction_bias` zero leaves; output = full nn tree (AC1 r2: 1460 keys
  == `model.parameters()`).
- Values arrive as `mx.array`; `_convert_value` transforms (`cast_int32`,
  `stack_uint8`, `stack`) operate on them as-is.

### 2.3 Convert consumption (no convert-side code change)

`mlx_lm.convert(--model hf-f8shim -q --mlx-path model-4bit)` (orchestrator
`finetune_ds4.py:945` `convert-shimmed`):
1. `load(hf-f8shim)` → config `deepseek_v4_nn` (after §2.0) → instantiate nn
   `Model` → `sanitize` → **§2.2 override remaps + strict-loads**.
2. `set_dtype` casts only floating (`cast_predicate` skips
   `e_score_correction_bias`); FP4 `*_weight` are uint8 → untouched.
3. `quantize_model` `wrapped_predicate`: `if not hasattr(module, "to_quantized")`
   → skip. `DeepseekV4FP4Experts` (`:342`, plain `nn.Module`, NO `to_quantized`)
   SKIPPED → experts stay FP4. Only `nn.Linear` (attn q_a/q_b/kv/o, gate,
   shared_experts) → 4-bit. ✓ ADR 0024 honored.
4. `save_model(mlx_path)` writes shards with nn keys + quant `.scales/.biases`;
   `save_config` writes `config.json` (`model_type=deepseek_v4_nn`, quantization
   block). → `model-4bit/` on disk.
5. smoke-train `mlx_lm.lora --model model-4bit` reloads → nn `Model.load_weights`
   → `is_native` False (keys already `model.*`) → bypass remap → strict load. ✓
   (This is why §2.2 idempotency is mandatory.)

---

## 3. SCOPE GUARD

- **NO FROZEN edit.** `deepseek_v4.py` (incl. `load_weights:2367`,
  `remap_shimmed_ckpt_keys`, `_attention_mlx`, FP4 dequant) BYTE-INTACT. ADR
  0017/0024/0026 untouched. Verify: `git diff -- .../deepseek_v4.py` empty AND
  source-hash unchanged (AGENTS.md: do not trust empty diff alone — hash it).
- **nn-port-only edit** = `deepseek_v4_nn.py` (`load_weights` add) → ADR 0025
  sanctioned (real trainable nn port is the training path enabler).
- **NEW files** = `ds4_ft_mlx/deepseek_v4_nn_remap.py` (extracted core) +
  edit `scripts/remap_ds4_nn_weights.py` (re-import). Neither frozen.
- **Data patch** = `hf-f8shim/config.json` model_type field (§2.0). Not code.
- New module participates in baseline → MUST be `git add`ed in the slice commit
  (AGENTS.md tracking hygiene; the moved remap core is verdict-bearing for AC1
  1460-key match).

---

## 4. ACCEPTANCE CRITERIA — Coder slice (model-4bit production)

1. `hf-f8shim/config.json` `model_type == "deepseek_v4_nn"` (§2.0 idempotent).
2. `ds4_ft_mlx/deepseek_v4_nn_remap.py` exists; `remap_weight_dict` importable
   from package; `scripts/remap_ds4_nn_weights.py plan` still works (re-import).
3. `deepseek_v4_nn.Model.load_weights` defined; native-key load remaps (1460
   keys == `model.parameters()`), nn-key load bypasses (idempotent reload).
4. `mlx_lm.convert --model "/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim" -q --mlx-path "/Volumes/Data NVME/mlx-ft/ds4/model-4bit"` SUCCEEDS;
   `model-4bit/` exists; `config.json` `model_type=deepseek_v4_nn` + quantization
   block; attn `nn.Linear` quantized 4-bit; FP4 experts PRESERVED (uint8
   `*_weight` + bf16 `*_scale`, NOT re-quantized — assert expert shards still FP4
   shape, no `.scales` on expert leaves).
5. Reload check: `mlx_lm.load("/Volumes/Data NVME/mlx-ft/ds4/model-4bit")` +
   one forward → finite (no NaN/Inf). Confirms §2.2 idempotent reload path.
6. FROZEN `deepseek_v4.py` byte-intact (diff empty + source hash unchanged).
7. New/moved files `git add`ed in slice commit (tracking hygiene).

Then 13.3b-5 BA re-pins runnable smoke-train AC; smoke-train runs against the now
on-disk `model-4bit/`.

---

## 5. NEW SLICE NAME

**13.3b-4b — model-4bit production (load-side remap into nn `Model.load_weights`)**.
Prerequisite, folded in BEFORE 13.3b-5 smoke-train. Coder serial; Reviewer
(`xhigh-reviewer`, `openai-codex/gpt-5.5`) + Test Manager after.

---

## 6. ADR follow-up

Add **ADR 0027 — load-side ckpt-native→nn remap in `deepseek_v4_nn.Model.load_weights`**
(extends ADR 0025; mirrors frozen `deepseek_v4.py:2367`; records idempotent
native-vs-nn gate + shared `deepseek_v4_nn_remap` extraction). BA/Architect
follow-up, not blocking Coder.
