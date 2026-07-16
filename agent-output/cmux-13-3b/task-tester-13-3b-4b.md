# Story 13.3b-4b — Test Manager (validate Coder model-4bit production)

## Slice context
- **Story**: 13.3b-4b — model-4bit production (load-side remap into nn `Model.load_weights`, Architect Option B). Prerequisite before 13.3b-5 smoke-train.
- **Coder done**: 10:03. HEAD still `fecc337` (NO commit per commit-gating). All deliverables `git add`-staged.
- **Predecessor**: 13.3b-4 (`fecc337`).

## What Coder staged (validate via `git diff --cached`)
```
.cmake-status/coder.done
agent-output/cmux-13-3b/coder-13-3b-4b-notes.md          (131L, coder notes)
docs/adr/0027-load-side-ckpt-native-to-nn-remap.md       (41L, NEW ADR)
python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_nn_remap.py    (228L, NEW extracted remap core)
python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py (58L added, load_weights override + AutoConfig registration)
scripts/remap_ds4_nn_weights.py                           (222L rewired, re-import from new module)
tests/test_deepseek_v4_attention_parity.py               (1647L, NEWLY-TRACKED baseline)
tests/test_deepseek_v4_checkpoint.py                     (1179L, NEWLY-TRACKED baseline)
tests/test_deepseek_v4_dequant_parity.py                 (1238L, NEWLY-TRACKED baseline)
tests/test_deepseek_v4_moe_parity.py                     (508L, NEWLY-TRACKED baseline)
tests/test_deepseek_v4_nn_load_weights_override.py       (108L, NEW test AC3)
tests/test_deepseek_v4_nn_remap_module.py                (51L, NEW test AC2)
tests/test_deepseek_v4_real_config_reference_forward.py  (2L, tiny update)
tests/test_ds4_gguf_base_smoke.py                        (238L, NEWLY-TRACKED baseline)
```

## Your job — independent validation

### 1. Run the slice's tests + full suite
```bash
unset SSLKEYLOGFILE
source python-envs/mlx/.venv/bin/activate
PYTHONPATH="$PWD:$PWD/python-envs/mlx/src" pytest -q tests/test_deepseek_v4_nn_remap_module.py tests/test_deepseek_v4_nn_load_weights_override.py tests/test_remap_ds4_nn_weights_keyset.py -v
PYTHONPATH="$PWD:$PWD/python-envs/mlx/src" pytest -q  # full suite — expect 578 / 13 / 0 RED
```

### 2. AC1-AC7 verification

**AC1** — config flip: `python3 -c "import json; print(json.load(open('/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json'))['model_type'])"` → must be `deepseek_v4_nn`.

**AC2** — extract module: `python3 -c "from ds4_ft_mlx.deepseek_v4_nn_remap import remap_weight_dict, remap_key; print('importable')"`; `python3 scripts/remap_ds4_nn_weights.py plan --src /Volumes/Data NVME/mlx-ft/ds4/hf-f8shim` (header path still works). AC1 r2 1460-key match must hold (Coder claims `expected_params 1460, remapped_specs 1460, missing 0, extra 0`).

**AC3** — load_weights override: `grep -n 'def load_weights' python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py` → defined. Run `test_deepseek_v4_nn_load_weights_override.py` — native-key load remaps (1460 keys), nn-key load (`model.*` prefix) bypasses remap (idempotent). **CRITICAL**: verify the `is_native = any(not k.startswith(("model.", "lm_head.")) for k in mapping)` gate is present + correct — this is what makes model-4bit reload idempotent.

**AC4** — convert SUCCEEDS + FP4 experts preserved (CRITICAL):
- `/Volumes/Data NVME/mlx-ft/ds4/model-4bit/` exists; `config.json` `model_type=deepseek_v4_nn` + `quantization: {group_size:64, bits:4, mode:affine}`.
- **FP4 experts PRESERVED**: `grep` model-4bit index for `model.layers.0.mlp.experts.w1_weight` (uint8) + `w1_scale` (BF16 sidecar) — MUST exist. Expert leaves MUST NOT have `.scales`/`.biases` (those are 4-bit nn.Linear quant artifacts). Verify: expert keys present (~258), expert `.scales/.biases` = 0, attn `.scales`/`.biases` present (~762).
- attn `nn.Linear` quantized 4-bit: `q_a_proj.weight` should be U32 (packed) + `q_a_proj.scales`/`.biases` BF16.
- **If experts got re-quantized** (expert `.scales`/`.biases` present, or `w1_weight` not uint8) → ADR 0024 violation, BLOCKED.

**AC5** — reload + forward finite:
```python
from mlx_lm import load
model, tok = load("/Volumes/Data NVME/mlx-ft/ds4/model-4bit", lazy=True)
import mlx.core as mx
logits = model(mx.array([[1,2,3,4,5,6,7,8]]))
mx.eval(logits)
# assert finite, shape (1, 8, vocab_size), 0 NaN/Inf
```

**AC6** — FROZEN `deepseek_v4.py` byte-intact:
```bash
git diff --cached -- python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py  # empty
sha256sum python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py  # 96c39168c78e5fd9158039707334f92206e8bb39f7e718e0d9064a7356809a44 per Coder — verify
```
AGENTS.md: do NOT trust empty `git diff` alone — also source-hash + AST.

**AC7** — tracking hygiene HARD RULE (`58194a9`):
```bash
git ls-files tests/test_deepseek_v4_nn_remap_module.py tests/test_deepseek_v4_nn_load_weights_override.py tests/test_deepseek_v4_attention_parity.py tests/test_deepseek_v4_checkpoint.py tests/test_deepseek_v4_dequant_parity.py tests/test_deepseek_v4_moe_parity.py tests/test_ds4_gguf_base_smoke.py
```
All MUST be staged `A` (not `??`). Coder did NOT `git commit` (HEAD still `fecc337`).

### 3. WATCH-ITEM — 5 newly-tracked historical test files (HARD RULE scrutiny)

Coder staged 5 previously-untracked baseline test files (`test_deepseek_v4_attention_parity.py` 1647L, `test_deepseek_v4_checkpoint.py` 1179L, `test_deepseek_v4_dequant_parity.py` 1238L, `test_deepseek_v4_moe_parity.py` 508L, `test_ds4_gguf_base_smoke.py` 238L). Coder's notes say: "Updated stale historical tests that asserted `/Volumes/Data NVME/mlx-ft/ds4/model-4bit` must be absent; they now accept absence OR a valid current `deepseek_v4_nn` 4-bit artifact."

**You MUST read the actual edits in these files** (via `git diff --cached` on each) and verify the "absent → absent OR valid" updates are LEGITIMATE contract updates (the model-4bit now exists legitimately as a `deepseek_v4_nn` 4-bit artifact, so "must be absent" is genuinely stale), NOT relaxations that hide regressions. This is exactly the 13.3b-3 chain-of-custody gap that triggered HARD RULE `58194a9` — untracked test files participating in the baseline MUST be tracked, and the legitimacy of any contract update MUST be diff-checked.

### 4. WATCH-ITEM — AutoConfig registration (narrow tokenizer tolerance)

Coder added a "narrow optional `AutoConfig` registration in `deepseek_v4_nn.py` so `AutoTokenizer` tolerates MLX-only `model_type=deepseek_v4_nn` during `mlx_lm.convert`/`mlx_lm.load`." Verify via `git diff --cached python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py` that:
- The registration is narrow (only config-level, discards DS4 layer/rope fields generic Transformers validation rejects).
- It does NOT touch FROZEN `deepseek_v4.py`.
- It does NOT alter the nn port's training behavior.

### 5. Regression
Full suite 578 / 13 / 0 RED. No scope creep. No C++.

## Deliverables
1. `agent-output/cmux-13-3b/test-manager-13-3b-4b-report.md` — full validation report (AC1-AC7 + 5 newly-tracked files legitimacy + AutoConfig + regression).
2. `.cmux-status/test-manager.done` marker.
3. In-pane JSON `{"status":"ok","role":"Test Manager"}` in surface:85 ONLY.

## STOP-ESCALATE
Write `agent-output/cmux-13-3b/test-manager-13-3b-4b-stop.md` if:
- Any AC RED (convert didn't actually succeed, FP4 experts re-quantized, reload NaN, FROZEN edited).
- The "absent → absent OR valid" updates in the 5 historical test files are relaxations hiding regressions (not legitimate contract updates).
- The `AutoConfig` registration is broad / touches FROZEN / alters training.
- Coder committed (HEAD advanced past `fecc337`).
- Test file untracked/unstaged (HARD RULE violation).

## Pre-flight read
1. `agent-output/cmux-13-3b/coder-13-3b-4b-notes.md` (Coder summary — AC4 convert result, AC5 reload, AC6 FROZEN hash).
2. `agent-output/cmux-13-3b/architecture-13-3b-5-model-4bit-path.md` (Architect §1-§6 SPEC).
3. `agent-output/cmux-13-3b/ba-13-3b-5-stop.md` (BA finding).
4. `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_nn_remap.py` (the NEW extracted module).
5. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py:528` (Model), `load_weights` override (the 58L addition).
6. `AGENTS.md` (tracking hygiene HARD RULE `58194a9`).

## Build
`source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first). Use `ctx_execute` for bounded header/index scans (NOT long polls). The AC5 reload + forward takes ~77s (real I/O). Caveman ultra default; byte-exact exempt. BEGIN NOW. Echo `{"status":"ok","role":"Test Manager"}` in this pane only.
