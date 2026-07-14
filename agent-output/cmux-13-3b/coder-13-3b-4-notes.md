# Coder notes — Story 13.3b-4

## Summary
- Added `scripts/remap_ds4_nn_weights.py` as a standalone convert-side remapper for the already-shimmed DS4 checkpoint.
- Added AC tests for header-only real keyset, strict sub-checkpoint load/forward, and tiny CSA regression.
- No FROZEN body, nn-port, or shim edits.
- No `git commit` run; supervisor owns commit after Reviewer + Test Manager GREEN.

## Ape orientation
- BA §0.4 applied: CSA, indexer, and HCA `ape` are copied verbatim in token-major layout.
- No `.T` / transpose path exists in the remapper.
- Tests pin:
  - `model.layers.2.self_attn.compressor.ape == (4, 1024)`
  - `model.layers.2.self_attn.indexer.compressor.ape == (4, 256)`
  - `model.layers.3.self_attn.compressor.ape == (128, 512)`
  - transform report for all three is `identity`.

## Key-remap coverage
- Globals: `embed.weight`, `head.weight`, `norm.weight`, `hc_head_{fn,base,scale}`.
- Per-layer norms/HC: `attn_norm`, `ffn_norm`, `hc_attn_*`, `hc_ffn_*`.
- Attention core: `wq_a/wq_b/wkv/wo_a/wo_b`, q/kv norms, sinks.
- CSA/HCA compressors and CSA indexer: compressor/indexer keys mapped to nn submodule leaves; ape identity.
- MLP: `gate.weight -> gate_weight`, `gate.bias -> e_score_correction_bias`, hash `tid2eid`, shared experts `w1/w3/w2 -> gate/up/down`.
- Routed experts: `layers.N.ffn.experts.M.w{1,2,3}.{weight,scale}` stacked by ascending expert id into `model.layers.N.mlp.experts.w{1,2,3}_{weight,scale}`; FP4 stays packed, weights cast/reported as uint8, scales preserved.
- Drops: `mtp.*`, dense BF16 attention `.scale` sidecars, `indexer.wq_b.scale`, shared-expert `.scale` sidecars.
- Hash-layer bias gap: synthetic zero `model.layers.{0,1,2}.mlp.e_score_correction_bias` for real config.
- Real checkpoint plan summary: `output_keys=1460`, `dropped_keys=1940`, `stacked_targets=258`, `payload_bytes_read=0`.

## Sub-checkpoint validation choice
- Full 162GB materialization intentionally not run per BA Q2.
- AC1 uses real checkpoint index+safetensors headers only; expected set comes from actual `deepseek_v4_nn` `model.parameters()` probe.
- AC2 uses a tiny but real-topology sub-checkpoint: 3 layers = sliding + CSA + HCA, ckpt-native raw keys, strict `Model.load_weights`, finite forward at sequence length 129.

## Validation
- RED first: new tests failed before implementation with `ModuleNotFoundError: No module named 'scripts.remap_ds4_nn_weights'`.
- Targeted AC: `pytest -q tests/test_remap_ds4_nn_weights_keyset.py tests/test_remap_ds4_nn_weights_load.py tests/test_remap_ds4_nn_weights_regression.py` → `3 passed`.
- Targeted nn/regression set: new tests + `test_13_3b_1_dispatch_additive.py` + `test_13_3b_3_nn_mixed_layers_forward.py` + `test_deepseek_v4_nn_fp4_parity.py` + `test_deepseek_v4_nn_backward.py` → `9 passed`.
- Full suite with project import path: `PYTHONPATH="$PWD:python-envs/mlx/src" pytest -q` → `572 passed, 13 skipped, 96 subtests passed`.
- `python scripts/remap_ds4_nn_weights.py plan --src '/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim'` → `output_keys=1460`, `payload_bytes_read=0`.
- `git diff --check` clean.
- `python -m py_compile` clean for script + new tests.

## Protected-file verification
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`: byte-identical to HEAD, AST-identical, sha256 prefix `pre-13.3b-5b-superseded-sha16`.
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py`: byte-identical to HEAD, AST-identical, sha256 prefix `994980c4ac41e4db`.
- `scripts/shim_ds4_safetensors.py`: byte-identical to HEAD, AST-identical, sha256 prefix `bcde9485a0a0bcdf`.
- sha-pin cascade no-op: no diff in `tests/test_numpy_real_forward_reference_composition.py` or `tests/test_deepseek_v4_real_config_reference_forward.py`.

## File tracking declaration
`git ls-files -- <slice files>` after staging confirms every new test file and deliverable is in the index:

```text
.cmux-status/coder.done
agent-output/cmux-13-3b/coder-13-3b-4-notes.md
scripts/remap_ds4_nn_weights.py
tests/test_remap_ds4_nn_weights_keyset.py
tests/test_remap_ds4_nn_weights_load.py
tests/test_remap_ds4_nn_weights_regression.py
```

Modified/created files in this slice:
- `.cmux-status/coder.done` — staged marker.
- `agent-output/cmux-13-3b/coder-13-3b-4-notes.md` — staged handoff notes.
- `scripts/remap_ds4_nn_weights.py` — staged new remap script.
- `tests/test_remap_ds4_nn_weights_keyset.py` — staged new AC1 test.
- `tests/test_remap_ds4_nn_weights_load.py` — staged new AC2 test.
- `tests/test_remap_ds4_nn_weights_regression.py` — staged new AC3 test.

No existing test file modified. No untracked test file left as a baseline dependency.
