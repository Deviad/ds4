# Coder notes — Story 13.3b-4b model-4bit production

## Scope delivered

- Data patch: `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json` now has `model_type=deepseek_v4_nn`.
- Added shared remap core: `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_nn_remap.py`.
- Rewired `scripts/remap_ds4_nn_weights.py` to import the shared remap table/core; header-only `plan` path still uses shared `remap_key`.
- Added `deepseek_v4_nn.Model.load_weights` override:
  - ckpt-native keys remap through `remap_weight_dict` before strict `nn.Module.load_weights`.
  - already-nn keys (`model.*` / `lm_head.*`, including quant `.scales` / `.biases`) bypass remap for idempotent `model-4bit/` reload.
- Added narrow Transformers `AutoConfig` registration in `deepseek_v4_nn.py` so `AutoTokenizer` tolerates MLX-only `model_type=deepseek_v4_nn` during `mlx_lm.convert` / `mlx_lm.load`.
- Added tests:
  - `tests/test_deepseek_v4_nn_remap_module.py`
  - `tests/test_deepseek_v4_nn_load_weights_override.py`
- Updated stale historical tests that asserted `/Volumes/Data NVME/mlx-ft/ds4/model-4bit` must be absent; they now accept absence OR a valid current `deepseek_v4_nn` 4-bit artifact.
- Tracking hygiene: the stale historical test files were untracked but now participate in the pass/fail baseline, so they are staged with this slice.
- Added ADR: `docs/adr/0027-load-side-ckpt-native-to-nn-remap.md`.

## TDD evidence

Initial RED:

```text
pytest -q tests/test_deepseek_v4_nn_remap_module.py tests/test_deepseek_v4_nn_load_weights_override.py
5 failed
```

GREEN after implementation:

```text
pytest -q tests/test_deepseek_v4_nn_remap_module.py tests/test_deepseek_v4_nn_load_weights_override.py tests/test_remap_ds4_nn_weights_keyset.py
7 passed
```

AC1 / AC2 keyset:

```text
expected_params 1460
remapped_specs 1460
missing 0
extra 0
payload_bytes_read 0
```

## Convert AC4

Command:

```bash
mlx_lm.convert --model '/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim' -q --mlx-path '/Volumes/Data NVME/mlx-ft/ds4/model-4bit'
```

Result:

```text
[INFO] Loading
[INFO] Using dtype: bfloat16
[INFO] Quantizing
[INFO] Quantized model with 8.292 bits per weight.
real 122.09
```

Output:

```text
/Volumes/Data NVME/mlx-ft/ds4/model-4bit exists
size: 149G
config model_type: deepseek_v4_nn
quantization: {'group_size': 64, 'bits': 4, 'mode': 'affine'}
weights: 2484
shards: 33
metadata total_size: 160062439598
```

FP4 expert preservation:

```text
DeepseekV4FP4Experts_has_to_quantized False
expert keys: 258
expert .scales/.biases: []
model.layers.0.mlp.experts.w1_weight U8 [256, 2048, 2048]
model.layers.0.mlp.experts.w1_scale BF16 [256, 2048, 128]
model.layers.3.self_attn.q_a_proj.weight U32 [1024, 512]
model.layers.3.self_attn.q_a_proj.scales BF16 [1024, 64]
model.layers.3.self_attn.q_a_proj.biases BF16 [1024, 64]
```

## Reload + forward AC5

Command loads `model-4bit` with `mlx_lm.load(..., lazy=True)` and forwards seq_len=8.

```text
logits_shape (1, 8, 129280)
finite True
nan_count 0
inf_count 0
real 77.64
```

## Regression

Full suite with repo + MLX src on `PYTHONPATH`:

```text
PYTHONPATH="$PWD:$PWD/python-envs/mlx/src" pytest -q
578 passed, 13 skipped, 2 warnings, 96 subtests passed in 118.06s
```

FROZEN `deepseek_v4.py`:

```text
deepseek_v4_sha256 96c39168c78e5fd9158039707334f92206e8bb39f7e718e0d9064a7356809a44
deepseek_v4_sha16 96c39168c78e5fd9
deepseek_v4_ast_parse ok
git diff -- deepseek_v4.py: empty
```

## Logs

- `agent-output/cmux-13-3b/coder-13-3b-4b-convert.log`
- `agent-output/cmux-13-3b/coder-13-3b-4b-load-forward.log`
- `agent-output/cmux-13-3b/coder-13-3b-4b-pytest-pypath-r3.log`

## Notes

First convert attempt after flipping `model_type` exposed a tokenizer-side issue:
Transformers `AutoTokenizer` tried `AutoConfig` on `deepseek_v4_nn`, which is an
MLX-only model type. The model itself had already loaded through `mlx_lm`'s model
registry. A narrow optional `AutoConfig` registration in `deepseek_v4_nn.py` fixes
only tokenizer config loading and discards DS4 layer/rope fields that generic
Transformers validation rejects. No FROZEN `deepseek_v4.py` changes.
