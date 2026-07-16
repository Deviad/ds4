# Story 13.3b-4b — Coder (model-4bit production: load-side remap into nn Model.load_weights)

## Slice context
- **Story**: 13.3b-4b — model-4bit production. Prerequisite folded in BEFORE 13.3b-5 smoke-train. Architect decision = **Option B** (`agent-output/cmux-13-3b/architecture-13-3b-5-model-4bit-path.md`).
- **Predecessor**: 13.3b-4 complete (HEAD `fecc337`): `scripts/remap_ds4_nn_weights.py` remap exists but IN-MEMORY only.
- **Problem**: smoke-train needs on-disk `model-4bit/` dir; `mlx_lm.convert --model hf-f8shim -q` fails because nn `Model.load_weights` inherits stock strict load (ckpt-native keys don't match nn leaf names).
- **Fix**: Option B — wire remap into nn `Model.load_weights` (mirrors frozen `deepseek_v4.py:2367`). 1-step pipeline, no intermediate 163GB dir, single source of truth for rename table.

## SCOPE (Architect §2 + §3)

### 1. Data patch — flip hf-f8shim config model_type (§2.0)
The shimmed ckpt was created Jun 17 BEFORE `copy_sidecars` started writing `model_type=deepseek_v4_nn`. On-disk `config.json` has `model_type=deepseek_v4` → `mlx_lm.convert` would route to FROZEN object-based `deepseek_v4.Model` (NOT nn.Module) → crash.

Run the idempotent patch (Architect §2.0 snippet):
```python
import json, pathlib
p = pathlib.Path("/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json")
c = json.loads(p.read_text())
if c.get("model_type") != "deepseek_v4_nn":
    c["model_type"] = "deepseek_v4_nn"
    p.write_text(json.dumps(c, indent=2, sort_keys=True) + "\n")
print("model_type =", c["model_type"])
```
Data patch, not code. NOTE: this config.json lives on `/Volumes/Data NVME/` — it is NOT in the git repo, so it isn't `git add`ed. Document the flip in your notes.

### 2. Extract remap core → NEW package module (§2.1)
`scripts/` is NOT on `sys.path` (vendor can't import from it). Extract the remap core into an importable package module so BOTH the script AND the nn `Model.load_weights` import ONE copy.

- **NEW file**: `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_nn_remap.py` (mirror `ds4_ft_mlx/shimmed_ckpt_key_remap.py` precedent).
- **MOVE** from `scripts/remap_ds4_nn_weights.py`: `KeyRemap`, `_DROP_PATTERNS`, `_DIRECT_RULES`, `_EXPERT_RULE`, `_LAYER`/`_EXPERT` regex consts, `remap_key` (`:155`), `_zero_bias_keys` (`:188`), `_validate_expert_sources` (`:192`), `_get_mx` (`:139`), `_config_int` (`:147`), `_convert_value`, and `remap_weight_dict` (`:329`).
- **RE-IMPORT** in `scripts/remap_ds4_nn_weights.py`: `from ds4_ft_mlx.deepseek_v4_nn_remap import remap_key, remap_weight_dict, ...` so the `plan` subcommand + header path (`remap_header_specs`) keep working unchanged. Header-only fns (`remap_header_specs`, `read_safetensors_header`, `_read_checkpoint_header_specs`) MAY stay in the script (they call the shared `remap_key`).
- Single source of truth for the rename table = preserved (AC1 r2 1460-key exact match must still hold).

### 3. Add `load_weights` override to deepseek_v4_nn.Model (§2.2)
Target: `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py`, `class Model(nn.Module)` (`:528`). Currently NO `load_weights` (inherits stock `nn.Module.load_weights` strict). Add override next to `sanitize` (`:547`).

Use the Architect's exact spec (§2.2):
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

### 4. Convert consumption (§2.3) — NO convert-side code change
`mlx_lm.convert(--model hf-f8shim -q --mlx-path model-4bit)` does everything once §1-3 land:
1. `load(hf-f8shim)` → config `deepseek_v4_nn` (after §1) → instantiate nn `Model` → `sanitize` → **§3 override remaps + strict-loads**.
2. `set_dtype` casts only floating; FP4 `*_weight` are uint8 → untouched.
3. `quantize_model`: `wrapped_predicate` skips modules without `to_quantized`; `DeepseekV4FP4Experts` (`:342`, plain `nn.Module`, NO `to_quantized`) SKIPPED → experts stay FP4. Only `nn.Linear` (attn q_a/q_b/kv/o, gate, shared_experts) → 4-bit. ADR 0024 honored.
4. `save_model(mlx_path)` writes `model-4bit/` with nn keys + quant `.scales/.biases`; `save_config` writes `config.json` (`model_type=deepseek_v4_nn`, quantization block).
5. smoke-train `mlx_lm.lora --model model-4bit` reloads → §3 `is_native` False → bypass remap → strict load (idempotency gate is WHY §3.5 mandatory).

## AC (Architect §4 — 7 acceptance criteria)
1. `hf-f8shim/config.json` `model_type == "deepseek_v4_nn"` (§2.0 idempotent).
2. `ds4_ft_mlx/deepseek_v4_nn_remap.py` exists; `remap_weight_dict` importable from package; `scripts/remap_ds4_nn_weights.py plan` still works (re-import).
3. `deepseek_v4_nn.Model.load_weights` defined; native-key load remaps (1460 keys == `model.parameters()`), nn-key load bypasses (idempotent reload).
4. `mlx_lm.convert --model "/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim" -q --mlx-path "/Volumes/Data NVME/mlx-ft/ds4/model-4bit"` **SUCCEEDS**; `model-4bit/` exists; `config.json` `model_type=deepseek_v4_nn` + quantization block; attn `nn.Linear` quantized 4-bit; **FP4 experts PRESERVED** (uint8 `*_weight` + bf16 `*_scale`, NOT re-quantized — assert expert shards still FP4 shape, no `.scales` on expert leaves).
5. Reload check: `mlx_lm.load("/Volumes/Data NVME/mlx-ft/ds4/model-4bit")` + one forward → finite (no NaN/Inf). Confirms §2.2 idempotent reload path.
6. FROZEN `deepseek_v4.py` byte-intact (diff empty + source hash unchanged).
7. New/moved files `git add`ed in slice commit (tracking hygiene HARD RULE `58194a9`).

## TDD red→green
- AC1: verify config flip (data patch).
- AC2 RED→GREEN: write test `test_deepseek_v4_nn_remap_module.py` — import `remap_weight_dict` from `ds4_ft_mlx.deepseek_v4_nn_remap`; run `scripts/remap_ds4_nn_weights.py plan` (header path) still produces 1460-key match.
- AC3 RED→GREEN: write test `test_deepseek_v4_nn_load_weights_override.py` — native-key load remaps (1460 keys, 0 missing/extra); nn-key load (`model.*` prefix) bypasses remap (idempotent).
- AC4: run `mlx_lm.convert --model hf-f8shim -q --mlx-path model-4bit`. **This is the core deliverable** — the actual 163GB convert. May take significant time (real I/O). Verify `model-4bit/` exists with correct config + FP4 experts preserved (assert expert leaves still uint8 `*_weight`, no `.scales` on expert leaves).
- AC5: reload + forward finite (tiny forward, seq_len 8-16).
- AC6: FROZEN `deepseek_v4.py` source-hash + AST byte-intact (AGENTS.md: do NOT trust empty git diff alone — hash it).
- AC7: `git ls-files` + `git add` new module + modified script + new tests. Commit-gating: `git add` (stage) ONLY — do NOT `git commit` (supervisor commits post-double-green).

## MUST NOT
- Edit FROZEN `deepseek_v4.py` (ADR 0017/0024/0026 — `load_weights:2367`, `remap_shimmed_ckpt_keys`, `_attention_mlx`, FP4 dequant all BYTE-INTACT).
- Run `git commit` (commit-gating — supervisor commits post-green-gates).
- Transpose ape (BA §0.4 from 13.3b-4 — the moved `remap_weight_dict` core preserves the no-transpose contract; do NOT alter the rename table).
- Re-quantize FP4 experts — `mlx_lm.convert -q` must SKIP them (verify via `to_quantized`-absent on `DeepseekV4FP4Experts`).
- Introduce C++.
- Modify the rename table (the move is byte-for-byte; AC1 r2 1460-key match must still hold).

## COMMIT GATING (user rule)
- `git add` (stage) the new module + modified script + new tests. Do NOT commit.
- Staged-but-uncommitted is correct end state.
- Reviewer + Test Manager validate via `git diff --cached`.
- Supervisor commits only after both return GREEN.

## STOP-ESCALATE
Write `agent-output/cmux-13-3b/coder-13-3b-4b-stop.md` if:
- AC4 convert fails on a real BUG in the nn port or remap on the FULL 43-layer real ckpt (forward NaN on a real cr=4/cr=128 layer that synthetic/tiny parity missed, OR strict-load fails on a nn param key the remap table doesn't cover) — STOP, record the failing key/layer + error, escalate. Genuine correctness issue, NOT a test fix.
- Disk runs out during AC4 convert (model-4bit + temp buffers exceed 226G free) — STOP, escalate Architect (may need streaming convert or different disk).
- The extract (§2.1) breaks AC1 r2 1460-key match (rename table drift during the move) — STOP, the move must be byte-for-byte.
- `to_quantized`-absent check fails (experts get re-quantized by `mlx_lm.convert -q`) — STOP, ADR 0024 violation.

## Pre-flight read
1. `agent-output/cmux-13-3b/architecture-13-3b-5-model-4bit-path.md` (Architect §1-§6 — the SPEC, read in full).
2. `agent-output/cmux-13-3b/ba-13-3b-5-stop.md` (BA finding + recon evidence index).
3. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py:528` (Model), `:547` (sanitize), `:342` (FP4Experts), `:206` (AttentionNN).
4. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py:2367` (frozen `load_weights` precedent with `remap_shimmed_ckpt_keys`).
5. `python-envs/mlx/src/ds4_ft_mlx/shimmed_ckpt_key_remap.py` (the precedent package module to mirror for the new `deepseek_v4_nn_remap.py`).
6. `scripts/remap_ds4_nn_weights.py` (the script to extract FROM — `remap_key:155`, `remap_weight_dict:329`, `plan:450/454`).
7. `tests/test_remap_ds4_nn_weights_keyset.py` (AC1 from 13.3b-4 — the 1460-key match must still pass after the extract).
8. `scripts/shim_ds4_safetensors.py:361` (`copy_sidecars` — how `model_type=deepseek_v4_nn` gets written on NEW shims; the existing hf-f8shim predates this).
9. ADR 0024 (FP4 dequant — experts STAY FP4), ADR 0025 (nn-port ownership), ADR 0026 (additive FROZEN).
10. `AGENTS.md` (tracking hygiene HARD RULE `58194a9` + FROZEN byte-intactness via source-hash+AST not git diff).

## Build
`source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first). Use `ctx_execute` for bounded probes (header scans, param-tree probes — NOT long polls). The AC4 convert on the 163GB shimmed ckpt may take significant time (real I/O — budget 10-30 min for the actual convert run). Caveman ultra default; byte-exact exempt. Echo `{"status":"ok","role":"Coder"}` in surface:83 ONLY.

## Deliverables
1. NEW `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_nn_remap.py` (extracted remap core).
2. MODIFIED `scripts/remap_ds4_nn_weights.py` (re-import from new module; `plan` still works).
3. MODIFIED `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py` (`load_weights` override at Model).
4. NEW tests: `test_deepseek_v4_nn_remap_module.py` (AC2) + `test_deepseek_v4_nn_load_weights_override.py` (AC3).
5. On-disk `model-4bit/` at `/Volumes/Data NVME/mlx-ft/ds4/model-4bit/` (AC4 — the actual convert output; NOT in git, lives on NVME).
6. `agent-output/cmux-13-3b/coder-13-3b-4b-notes.md` (config flip §2.0 done + extract §2.1 approach + override §2.2 + convert AC4 result: model-4bit dir stats + FP4 expert preservation evidence + reload+forward finite + any bugs found).
7. `.cmux-status/coder.done` marker + in-pane JSON.
8. `git add` staged (NOT committed): new module + modified script + modified nn-port + 2 new tests.

## Regression target
- AC1 r2 1460-key match still holds (extract byte-for-byte).
- 13.3b-1/2a/2b/3 forward parity intact (FROZEN byte-intact).
- Full suite: 572 + 2 new / 13 / 0 RED.
- FROZEN `deepseek_v4.py` source-hash unchanged.

Echo `{"status":"ok","role":"Coder"}` in this pane only. BEGIN NOW.
