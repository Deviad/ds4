# Coder r2 notes — Story 13.3b-4 AC2 real payload fix

## What was wrong in r0
- Reviewer Axis 5 was correct: `tests/test_remap_ds4_nn_weights_load.py` used a tiny synthetic topology plus fabricated zero/one tensors.
- That proved key shape plumbing only; it did not load safetensors payload bytes from `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim` and could not catch value-corrupting remaps.
- Test Manager r0 AC2 GREEN was a methodology miss.

## r2 change
- Rewrote only `tests/test_remap_ds4_nn_weights_load.py`.
- AC1 and AC3 tests left worktree-byte-unchanged.
- New AC2 test loads real payload tensors for:
  - globals: `embed.weight`, `head.weight`, `norm.weight`, `hc_head_{fn,base,scale}`
  - original L0 sliding, original L2 CSA, original L3 HCA
- Compacts real layers `0,2,3` into local nn layers `0,1,2` for a 3-layer model with `layer_types=[sliding, CSA, HCA]` and `compress_ratios=[0,4,128]`.
- Real original L2 is still hash-routed (`num_hash_layers=3` in full config), so compact model uses `mlp_layer_types=[hash_moe, hash_moe, moe]` / `num_hash_layers=2`; this preserves real L2 `tid2eid` payload instead of fabricating a MoE gate bias.

## Real payload / anti-circularity
- Test references `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim`.
- Test imports/uses `safe_open`, `get_tensor`, and `load_file`.
- `safe_open(...).get_tensor` materializes non-BF16 real tensors (FP4 packed `I8` expert bytes, F32 ape/HC/etc.).
- Local safetensors MLX `get_tensor(BF16)` raises `TypeError: data type 'bfloat16' not understood`; BF16 tensors use `mx.load` shard-mmap fallback while retaining only requested keys. `load_file` is also exercised on a tiny real `hc_head_base` probe written under `tmp_path`.
- No `_zeros`, `_ones`, `mx.zeros`, or `mx.ones` exist in the test file.
- Real FP4 stack verified:
  - `model.layers.1.mlp.experts.w1_weight` shape `(256, 2048, 2048)`
  - dtype `mx.uint8`
  - nonzero sample
  - sample equals original `layers.2.ffn.experts.0.w1.weight` bytes after uint8 cast

## RAM budget
- Selected payload keys: globals + L0/L2/L3 only, not full 46-shard / 162GB checkpoint.
- Header-counted selected payload: ~12.906 GiB, under 16 GiB budget.
- Prototype strict-load+forward peak max RSS observed ~13.64 GiB with BF16 `mx.load` mmap fallback.

## Remap bugs found
- No remap script changes required.
- Strict key set for compact model: 103 expected keys, 0 missing, 0 extra.
- `model.load_weights(list(remapped.items()), strict=True)` succeeds.
- Forward at `seq_len=8` finite with shape `(1, 8, 129280)`.

## Validation
- `pytest -q tests/test_remap_ds4_nn_weights_load.py` → `1 passed, 1 warning in 4.35s`.
- `pytest -q tests/test_remap_ds4_nn_weights_keyset.py tests/test_remap_ds4_nn_weights_load.py tests/test_remap_ds4_nn_weights_regression.py` → `3 passed, 1 warning in 4.84s`.
- `pytest -q` → `572 passed, 13 skipped, 2 warnings, 96 subtests passed in 113.09s`.
- `python -m py_compile scripts/remap_ds4_nn_weights.py tests/test_remap_ds4_nn_weights_keyset.py tests/test_remap_ds4_nn_weights_load.py tests/test_remap_ds4_nn_weights_regression.py` → OK.
- `git diff --check` → OK.
- Anti-circularity grep: required tokens present; forbidden synthetic zero/one helpers absent.

## Protected / scope checks
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`: byte-identical to HEAD, AST-identical, sha256 prefix `96c39168c78e5fd9`.
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py`: byte-identical to HEAD, AST-identical, sha256 prefix `994980c4ac41e4db`.
- `scripts/shim_ds4_safetensors.py`: byte-identical to HEAD, AST-identical, sha256 prefix `bcde9485a0a0bcdf`.
- No C++.
- No git commit run.

## Staging / tracking
- Staged modified `tests/test_remap_ds4_nn_weights_load.py` plus this r2 notes file and fresh `.cmux-status/coder.done`.
- Existing staged r0 deliverables remain staged for supervisor commit gating.
