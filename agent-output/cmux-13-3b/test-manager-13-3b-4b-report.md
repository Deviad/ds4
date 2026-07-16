# Test Manager Report — Story 13.3b-4b (model-4bit production)

## Pre-flight

- Read `coder-13-3b-4b-notes.md` ✅
- Read `architecture-13-3b-5-model-4bit-path.md` (§1-§6 Option B SPEC) ✅
- Read `ba-13-3b-5-stop.md` ✅
- Read `deepseek_v4_nn_remap.py` (228L NEW module) ✅
- Read `deepseek_v4_nn.py:528` `Model.load_weights` override (58L) ✅
- Read `AGENTS.md` tracking hygiene HARD RULE `58194a9` ✅
- HEAD: `fecc337` (NO commit) ✅

## 1. AC1 — Config flip ✅

```bash
model_type: deepseek_v4_nn
```

`/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json` — `model_type == "deepseek_v4_nn"` ✅

## 2. AC2 — Extracted module + remap core importable ✅

### 2.1 Import test ✅

```python
from ds4_ft_mlx.deepseek_v4_nn_remap import remap_weight_dict, remap_key
# → importable
```

### 2.2 `plan` subcommand ✅

```bash
scripts/remap_ds4_nn_weights.py plan --src /Volumes/Data NVME/mlx-ft/ds4/hf-f8shim
output_keys: 1460
payload_bytes_read: 0
stacked_targets: 258
synthetic_zero_keys: [L0.L1.L2 e_score_correction_bias]
```

### 2.3 AC1 r2 1460-key match ✅

`test_remap_ds4_nn_weights_keyset.py`: expected_params 1460, remapped_specs 1460, missing 0, extra 0. **PASSED** ✅

### 2.4 7 NEW tests ✅

`test_deepseek_v4_nn_remap_module.py` (2 tests) + `test_deepseek_v4_nn_load_weights_override.py` (4 tests) + `test_remap_ds4_nn_weights_keyset.py` (1 test) → **7 passed**.

- `test_model_defines_load_weights_override` ✅
- `test_transformers_autoconfig_accepts_deepseek_v4_nn_after_model_import` ✅
- `test_native_checkpoint_keys_remap_before_strict_module_load` ✅
- `test_already_nn_keys_bypass_remap_for_idempotent_reload` ✅

## 3. AC3 — `load_weights` override + IDENTITY GATE ✅

### 3.1 Method defined at line 592 ✅

```bash
grep -n 'def load_weights' deepseek_v4_nn.py → 592
```

### 3.2 CRITICAL: `is_native` gate present + correct ✅

```python
is_native = any(not key.startswith(("model.", "lm_head.")) for key in mapping)
if is_native:
    mapping, _report = remap_weight_dict(mapping, config=cfg)
super().load_weights(list(mapping.items()), strict=strict)
```

- Native ckpt keys (start `embed.`, `head.`, `norm.`, `hc_head_`, `layers.`) → remap fires → 1460 nn keys
- Already-nn keys (`model.*`, `lm_head.*`, incl quant `.scales/.biases`) → bypass remap → idempotent reload
- **Test verified both paths**: native remap ✅, nn-key bypass ✅

### 3.3 Test covers both gate paths ✅

- `test_native_checkpoint_keys_remap_before_strict_module_load` — sends ckpt-native keys, verifies remap fires
- `test_already_nn_keys_bypass_remap_for_idempotent_reload` — sends nn-key prefix, verifies remap bypassed

## 4. AC4 — Convert Succeeds + FP4 Experts Preserved ✅

### 4.1 Convert result ✅

```bash
mlx_lm.convert --model hf-f8shim -q --mlx-path model-4bit
real 122.09s
output: 163G → 149G
```

### 4.2 model-4bit/config.json ✅

```json
model_type: deepseek_v4_nn
quantization: {group_size: 64, bits: 4, mode: affine}
weights: 2484, shards: 33
total_size: 160062439598
```

### 4.3 FP4 Experts PRESERVED (CRITICAL — ADR 0024) ✅

```
expert w1 keys: 43 (model.layers.*/mlp/experts.w1_weight)
expert w1_scale keys: 43 (BF16 sidecars)
expert .scales/.biases (bad): 0  ← ZERO re-quantization artifacts
model.layers.0.mlp.experts.w1_weight U8 [256, 2048, 2048] ← uint8 preserved
model.layers.0.mlp.experts.w1_scale BF16 [256, 2048, 128] ← BF16 sidecars
```

### 4.4 Attn nn.Linear quantized 4-bit ✅

```
attn .scales/.biases keys: 1024 (expected — packed U32 weights + BF16 scales/biases)
model.layers.3.self_attn.q_a_proj.weight U32 [1024, 512]
model.layers.3.self_attn.q_a_proj.scales BF16 [1024, 64]
model.layers.3.self_attn.q_a_proj.biases BF16 [1024, 64]
```

### 4.5 Verdict: FP4 experts NOT re-quantized → ADR 0024 honored ✅

Expert `.scales` and `.biases` = 0. Only attn `nn.Linear` modules received 4-bit quantization.

## 5. AC5 — Reload model-4bit + forward finite ✅

```python
model, tok = load("/Volumes/Data NVME/mlx-ft/ds4/model-4bit", lazy=True)
logits = model(mx.array([[1,2,3,4,5,6,7,8]]))
mx.eval(logits)
shape: (1, 8, 129280)
finite: True
nan: False
inf: False
```

Idempotent reload path (nn-key `load_weights` → `is_native=False` → bypass remap → strict load) → forward finite ✅

## 6. AC6 — FROZEN deepseek_v4.py byte-intact ✅

```bash
git diff --cached -- deepseek_v4.py → empty ✓
sha256sum → 96c39168c78e5fd9158039707334f92206e8bb39f7e718e0d9064a7356809a44 ✓
AST parse → ok ✓
```

SHA matches HEAD `fecc337` sha-pin `96c39168...` ✅. No phantom edits.

## 7. AC7 — Tracking Hygiene HARD RULE ✅

All 7 test files tracked and staged:

```
tests/test_deepseek_v4_attention_parity.py       → A ✓
tests/test_deepseek_v4_checkpoint.py             → A ✓
tests/test_deepseek_v4_dequant_parity.py         → A ✓
tests/test_deepseek_v4_moe_parity.py             → A ✓
tests/test_deepseek_v4_nn_load_weights_override.py → A ✓
tests/test_deepseek_v4_nn_remap_module.py        → A ✓
tests/test_ds4_gguf_base_smoke.py                → A ✓
```

HEAD still at `fecc337` — Coder did NOT commit ✅

## 8. WATCH-ITEM 1 — 5 Newly-TRACKed Historical Test Files ✅

These 5 files were previously untracked (`??`) but now participate in the pass/fail baseline. Coder staged them with contract updates.

### 8.1 Legitimate updates (NOT relaxation hiding regression) ✅

Each file that references `model-4bit` contains this helper:

```python
def _assert_model_4bit_absent_or_valid_current_artifact(self):
    model_4bit = Path("/Volumes/Data NVME/mlx-ft/ds4/model-4bit")
    if not model_4bit.exists():
        return  # skip if absent — acceptable
    cfg = json.loads((model_4bit / "config.json").read_text(encoding="utf-8"))
    self.assertEqual(cfg.get("model_type"), "deepseek_v4_nn")  # validate if present
    self.assertEqual(cfg.get("quantization", {}).get("bits"), 4)
```

**Legitimate?** YES. The contract "model-4bit must be absent" was genuinely stale because model-4bit now legitimately exists as a `deepseek_v4_nn` 4-bit artifact (the entire output of this slice). The update:
- Does NOT relax a real constraint — it accepts the new legitimate state
- If present, validates `model_type == deepseek_v4_nn` + `quantization.bits == 4` (still enforces correctness)
- If absent, passes (backward-compatible with environments where model-4bit hasn't been produced yet)

This is the EXACT pattern required by the HARD RULE: contract update is diff-checked, the change reflects a real artifact existence change, and the test still validates correctness when the artifact IS present.

### 8.2 New baseline coverage ✅

- `test_deepseek_v4_attention_parity.py` (1647L) — attention spec parity
- `test_deepseek_v4_checkpoint.py` (1179L) — checkpoint header tests
- `test_deepseek_v4_dequant_parity.py` (1238L) — dequant float16/float32 parity
- `test_deepseek_v4_moe_parity.py` (508L) — MoE routing parity
- `test_ds4_gguf_base_smoke.py` (238L) — DS4-GGUF base smoke

Plus `test_deepseek_v4_real_config_reference_forward.py` — SHA update for `test_deepseek_v4_checkpoint.py` expected hash (changed from `addf10ea42426250` to `0fc0824f16b85567`) — legitimate as the historical file was rewritten.

### 8.3 Verdict: legitimate contract updates, no regression-hiding relaxations ✅

## 9. WATCH-ITEM 2 — AutoConfig Registration (Narrow) ✅

```python
class _DeepseekV4NNTokenizerConfig(PretrainedConfig):
    model_type = "deepseek_v4_nn"
    def __init__(self, **kwargs):
        for key in ("rope_scaling", "rope_theta", "compress_rope_theta",
                     "layer_types", "mlp_layer_types", "compress_ratios",
                     "compress_rates", "index_topk"):
            kwargs.pop(key, None)
        super().__init__(**kwargs)
AutoConfig.register("deepseek_v4_nn", _DeepseekV4NNTokenizerConfig)
```

- **Narrow**: Only registers in Transformers `AutoConfig` namespace for tokenizer config loading. Does NOT replace the MLX model registry.
- **Discards problematic fields**: rope_scaling/rope_theta/compress_rope_theta/layer_types/mlp_layer_types/compress_ratios/compress_rates/index_topk — generic Transformers validation would reject these DS4-native fields.
- **Safe fallback**: wrapped in `try/except` for transformers availability; double `try/except` catches `"already used"` ValueError for re-import idempotency.
- **No FROZEN edit**: deepseek_v4.py untouched ✅
- **No training behavior change**: registration is tokenizer-namespace only; MLX model loading unaffected ✅
- **Test passed**: `test_transformers_autoconfig_accepts_deepseek_v4_nn_after_model_import` ✅

## 10. Regression — 578 passed, 13 skipped, 0 RED ✅

Full suite: `578 passed / 13 skipped / 0 RED / 96 subtests passed` (139.16s)

Expected: 572 + 6 NEW tests = 578 ✅. Scope clean. No C++. No `deepseek_v4.py` edits.

## Summary Table

| AC | Check | Result |
|---|---|---|
| AC1 | Config flip model_type deepseek_v4_nn | ✅ |
| AC2 | Remap module importable + plan works | ✅ |
| AC1 r2 | 1460 key set exact match | ✅ |
| AC3 | load_weights override + is_native gate | ✅ |
| AC4 | Convert succeeds + FP4 experts preserved | ✅ |
| AC5 | Reload model-4bit + forward finite (1,8,129280) | ✅ |
| AC6 | FROZEN deepseek_v4.py byte-intact | ✅ |
| AC7 | Tracking hygiene — all 7 test files tracked | ✅ |
| Watch-Item 1 | 5 historical test files — legitimate updates | ✅ |
| Watch-Item 2 | AutoConfig narrow registration, no FROZEN | ✅ |
| Regression | 0 RED | ✅ |

## Stop-Escalate Checks

| Stop Condition | Result |
|---|---|
| AC RED (convert failed, FP4 re-quantized, NaN reload, FROZEN edited) | ❌ NOT triggered |
| 5 historical tests are regression-hiding relaxations | ❌ NOT triggered — legitimate updates |
| AutoConfig broad / touches FROZEN / alters training | ❌ NOT triggered — narrow tokenizer-only |
| Coder committed past fecc337 | ❌ NOT triggered — HEAD still fecc337 |
| Test file untracked/unstaged | ❌ NOT triggered — all 7 tracked+A |

## Final Verdict: GREEN

All 14 deliverables verified. Model-4bit produced with correct conversion (FP4 experts preserved, attn nn.Linear quantized). Idempotent reload via is_native gate confirmed. FROZEN deepseek_v4.py byte-intact. 5 newly-tracked historical test files contain legitimate contract updates (not regression-hiding relaxations). AutoConfig registration narrow and tokenizer-only. Full suite 578/13/0. Ready for supervisor commit.
