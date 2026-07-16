# Review 13.3b-4b — model-4bit production

Verdict: **PASS**

Reviewed staged work at `HEAD fecc33778dd7739e3667c0ccbf270ed7be024afa` via `git diff --cached`.
No production/test/ADR edits made by Reviewer.

## Axis 1 — Scope / blast radius: GREEN

Evidence:

- `git rev-parse --short HEAD` → `fecc337`; Coder did not commit past gate.
- Staged set is the expected 14-file slice: coder marker/notes, ADR 0027, new `deepseek_v4_nn_remap.py`, `deepseek_v4_nn.py`, `scripts/remap_ds4_nn_weights.py`, 5 historical tests, 2 new AC tests, 1 sentinel update, GGUF smoke test.
- No staged C/C++/Metal files.
- Frozen file diff: `git diff --cached -- python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` → 0 bytes.

## Axis 2 — FROZEN byte + AST identity: GREEN

`python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`:

```text
sha256        96c39168c78e5fd9158039707334f92206e8bb39f7e718e0d9064a7356809a44
head_sha256   96c39168c78e5fd9158039707334f92206e8bb39f7e718e0d9064a7356809a44
same_bytes    True
ast_sha256    c53646c8a5719f9b6e530394af8351b0c977292cebfa7047a151cf82f591ea0a
same_ast      True
cached_diff   0 bytes
unstaged_diff 0 bytes
```

Protected landmarks still in frozen file:

```text
def load_weights                         line 2367
from ds4_ft_mlx.shimmed_ckpt_key_remap   line 2373
def _attention_mlx                       line 1190
def _dequantize_fp4_block_scale_mlx      line 163
```

Forbidden new remap symbols absent from frozen file: `deepseek_v4_nn_remap`, `remap_weight_dict`, `KeyRemap`, `RemapReport`, `_DROP_PATTERNS`, `_DIRECT_RULES`, `_EXPERT_RULE`, `_validate_expert_sources`, `_zero_bias_keys` all count `0`.

## Axis 3 — Correctness vs Architect §2: GREEN

### §2.0 config flip

```text
/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json model_type = deepseek_v4_nn
```

### §2.1 extracted remap core

`python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_nn_remap.py` contains the required shared core:

```text
KeyRemap
_DROP_PATTERNS
_DIRECT_RULES
_EXPERT_RULE
remap_key
_zero_bias_keys
_validate_expert_sources
_get_mx
_config_int
_convert_value
remap_weight_dict
```

`scripts/remap_ds4_nn_weights.py` re-imports that core from `ds4_ft_mlx.deepseek_v4_nn_remap`.

Header/keyset evidence:

```text
scripts/remap_ds4_nn_weights.py plan --src /Volumes/Data NVME/mlx-ft/ds4/hf-f8shim
output_keys        1460
payload_bytes_read 0
stacked_targets    258
synthetic_zero_keys model.layers.0/1/2.mlp.e_score_correction_bias
```

Anti-circular keyset probe:

```text
expected_params 1460   # from tree_flatten(Model(ModelArgs.from_dict(cfg)).parameters())
remapped_specs  1460
missing         0
extra           0
payload_bytes_read 0
```

### §2.2 `Model.load_weights` override

`deepseek_v4_nn.Model.load_weights` is present at line 592. Critical idempotent gate present:

```python
is_native = any(not key.startswith(("model.", "lm_head.")) for key in mapping)
if is_native:
    cfg = {
        "n_routed_experts": self.args.n_routed_experts,
        "num_hash_layers": self.args.num_hash_layers,
    }
    mapping, _report = remap_weight_dict(mapping, config=cfg)
super().load_weights(list(mapping.items()), strict=strict)
```

Native ckpt keys remap. Already-nn keys (`model.*`, `lm_head.*`, quant `.scales` / `.biases`) bypass remap.

### §2.3 convert-side consumption

No convert-side code change found. `DeepseekV4FP4Experts` remains non-quantizable:

```text
DeepseekV4FP4Experts line 383
methods ['__init__', '_dequant', 'forward_one', '__call__']
has_to_quantized False
cached diff matching to_quantized|DeepseekV4FP4Experts: none
```

## Axis 4 — AC4 model-4bit FP4 expert preservation: GREEN

Header-only scan of `/Volumes/Data NVME/mlx-ft/ds4/model-4bit`:

```text
model-4bit/config.json model_type deepseek_v4_nn
quantization {'group_size': 64, 'bits': 4, 'mode': 'affine'}
safetensors_files 33
tensor_count 2484
```

FP4 routed experts preserved, not re-quantized:

```text
expert_w1_weight count 43
  model.layers.0.mlp.experts.w1_weight U8   (256, 2048, 2048)
expert_w1_scale count 43
  model.layers.0.mlp.experts.w1_scale  BF16 (256, 2048, 128)
expert_scales_biases count 0

expert w1_weight unique [('U8',   (256, 2048, 2048))]
expert w1_scale  unique [('BF16', (256, 2048, 128))]
expert w2_weight unique [('U8',   (256, 4096, 1024))]
expert w2_scale  unique [('BF16', (256, 4096, 64))]
expert w3_weight unique [('U8',   (256, 2048, 2048))]
expert w3_scale  unique [('BF16', (256, 2048, 128))]
```

Regular attention `nn.Linear` leaves are 4-bit quantized:

```text
attn_scales_biases count 762
model.layers.0.self_attn.q_a_proj.weight U32  (1024, 512)
model.layers.0.self_attn.q_a_proj.scales BF16 (1024, 64)
model.layers.0.self_attn.q_a_proj.biases BF16 (1024, 64)
```

ADR 0024 preserved: no expert `.scales` / `.biases` artifacts.

## Axis 5 — AC2 / AC3 legitimacy, anti-tautology: GREEN

Test evidence:

```text
pytest -q tests/test_deepseek_v4_nn_remap_module.py tests/test_deepseek_v4_nn_load_weights_override.py
6 passed, 1 warning in 0.10s

pytest -q tests/test_remap_ds4_nn_weights_keyset.py
1 passed, 1 warning in 0.58s
```

AC2 legitimacy:

- New test imports `ds4_ft_mlx.deepseek_v4_nn_remap` from package.
- Script re-export check asserts `script_remap.remap_key is package_remap.remap_key` and `script_remap.remap_weight_dict is package_remap.remap_weight_dict`.
- 1460-key match comes from probed `model.parameters()` via existing keyset test, not a hand-written expected list.

AC3 legitimacy:

- `test_native_checkpoint_keys_remap_before_strict_module_load` monkeypatches `deepseek_v4_nn_remap.remap_weight_dict`, loads native `embed.weight`, verifies remap call/config and loaded value.
- `test_already_nn_keys_bypass_remap_for_idempotent_reload` monkeypatches remap to raise, loads `model.embed_tokens.weight`, verifies bypass and loaded value.
- Both remap path and bypass path are asserted.

## Axis 6 — WATCH: 5 newly-tracked historical baseline tests: GREEN

All five watched files are staged/tracked in the index (`git ls-files` true):

| File | Staged | Relevant helper | Call sites | Adjudication |
|---|---:|---:|---:|---|
| `tests/test_deepseek_v4_attention_parity.py` | `A` | line 21 | 420, 1100 | legitimate stale-contract update |
| `tests/test_deepseek_v4_checkpoint.py` | `A` | line 16 | 1153 | legitimate stale-contract update |
| `tests/test_deepseek_v4_dequant_parity.py` | `A` | line 135 | 547 | legitimate stale-contract update |
| `tests/test_deepseek_v4_moe_parity.py` | `A` | line 47 | 306 | legitimate stale-contract update |
| `tests/test_ds4_gguf_base_smoke.py` | `A` | line 69 | 202, 234 | legitimate stale-contract update |

Representative helper body in each file:

```python
if not model_4bit.exists():
    return
cfg = json.loads((model_4bit / "config.json").read_text(encoding="utf-8"))
self.assertEqual(cfg.get("model_type"), "deepseek_v4_nn")
self.assertEqual(cfg.get("quantization", {}).get("bits"), 4)
```

Adjudication:

- Old contract was “`model-4bit` must be absent” for earlier routing/parity slices.
- Story 13.3b-4b legitimately creates `/Volumes/Data NVME/mlx-ft/ds4/model-4bit`; the old absence assertion is now stale.
- The relaxed guard still rejects stale `deepseek_v4` or non-4-bit artifacts.
- Current artifact validity is independently proven by Axis 4 header scan, including FP4 expert preservation and zero expert `.scales` / `.biases`.
- No relaxation observed that masks the current regression under review.

Sentinel update is consistent:

```text
tests/test_deepseek_v4_checkpoint.py staged sha16 0fc0824f16b85567
EXPECTED_SHAS updated to 0fc0824f16b85567
```

## Axis 7 — WATCH: AutoConfig registration narrowness: GREEN

Staged `deepseek_v4_nn.py` adds one optional tokenizer-tolerance helper at module top:

```python
def _register_transformers_tokenizer_config() -> None:
    try:
        from transformers import AutoConfig, PretrainedConfig
    except Exception:
        return

    class _DeepseekV4NNTokenizerConfig(PretrainedConfig):
        model_type = "deepseek_v4_nn"

        def __init__(self, **kwargs: Any):
            for key in (
                "rope_scaling",
                "rope_theta",
                "compress_rope_theta",
                "layer_types",
                "mlp_layer_types",
                "compress_ratios",
                "compress_rates",
                "index_topk",
            ):
                kwargs.pop(key, None)
            super().__init__(**kwargs)

    try:
        AutoConfig.register("deepseek_v4_nn", _DeepseekV4NNTokenizerConfig)
    except ValueError as exc:
        if "already used" not in str(exc):
            raise
```

Adjudication:

- Exact `model_type` only: `deepseek_v4_nn`.
- Config-level only; no tokenizer behavior override, no model load override beyond the intended `load_weights` change.
- Does not touch frozen `deepseek_v4.py`.
- Does not alter `ModelArgs`, training layers, forward path, sanitize semantics, or quantization behavior.
- DS4 model validation still happens through `ModelArgs.from_dict(...).validate()` for actual model construction; this is only to let `AutoTokenizer` tolerate the MLX-only model type.

## Axis 8 — sha-pin cascade + regression: GREEN

Sha-pin cascade:

- Frozen `deepseek_v4.py` unchanged; no cascade needed.
- Current sha16 pin sites in source/tests remain `96c39168c78e5fd9`.
- Old stale pins (`dc5aaaab9bb079d2`, `e4cafc52970b4617`) not present in active `tests/`, `python-envs/mlx/src`, or `scripts` sources except pycache / historical backlog prose.

Regression:

```text
pytest -q
578 passed, 13 skipped, 2 warnings, 96 subtests passed in 118.51s
```

Whitespace:

```text
git diff --cached --check
# no output
```

No C/C++/Metal scope creep:

```text
git diff --cached --name-only | grep -E '\.(c|h|m|mm|metal|cpp|cc|hpp)$'
# no output
```

## Final verdict

**PASS** — staged 13.3b-4b implements Architect Option B correctly. Frozen `deepseek_v4.py` is byte/AST intact, load-side remap is single-source and idempotent, `model-4bit` preserves FP4 routed experts, the five historical tests are now tracked and legitimately updated for the new artifact, and AutoConfig registration is narrow tokenizer tolerance only.
