# Test report — Story 13.3b-5d

Status: GREEN

Commands:
- `uv run --directory python-envs/mlx pytest ../../tests/test_deepseek_v4_nn_sparse_routed_backward.py -v --tb=short`
- `uv run --directory python-envs/mlx bash -lc 'cd /Users/spotted/projects/ds4-finetuning && PYTHONPATH=/Users/spotted/projects/ds4-finetuning pytest $(git ls-files "tests/test*.py") -q'`
- `git ls-files -- tests/test_deepseek_v4_nn_sparse_routed_backward.py tests/test_deepseek_v4_nn_backward.py tests/test_deepseek_v4_nn_moe_forward.py tests/test_deepseek_v4_nn_fp4_parity.py tests/test_deepseek_v4_fp4_dequant_mlx.py tests/test_deepseek_v4_fp4_dequant_python_ref.py tests/test_deepseek_v4_dequant_parity.py tests/test_deepseek_v4_moe_parity.py tests/test_deepseek_v4_nn_moe_construct.py tests/test_deepseek_v4_nn_moe_quantize.py tests/test_deepseek_v4_nn_hash_forward.py tests/test_deepseek_v4_nn_remap_module.py tests/test_remap_ds4_nn_weights_keyset.py tests/test_remap_ds4_nn_weights_load.py tests/test_remap_ds4_nn_weights_regression.py tests/test_deepseek_v4_nn_lora.py`
- `git diff --check`

Counts:
- sparse routed backward suite: `21 passed, 1 warning`
- focused FP4/MoE/remap/LoRA trainer-contract suites: `122 passed, 3 skipped, 1 warning, 16 subtests passed`
- full non-live regression: `449 passed, 15 skipped, 2 warnings, 86 subtests passed`
- tracked verdict-participating tests: `16/16` focused files tracked; `44/44` tracked `tests/test*.py`

Memory numbers:
- dense/sparse probe, `E=4`: `(193101.0, 179815.0)`
- dense/sparse probe, `E=8`: `(253215.0, 125221.0)`
- real-dimension no-shard probe: `active_delta=268500996.0`, `peak_bytes=4076454246.0`, `finite_y=True`, `finite_g=True`
- formula-contract constants verified in test: `route_meta_bytes=98304`, `one_expert_dequant=100663296`, `old_all_dequant=25769803776`, `old_full_seq_act=42949672960`, `routed_op_budget=1342177280`, `margin=77877452800`

Tracking / scope:
- all verdict-participating test files were tracked
- `git diff --check` clean
- no code/docs/tests edited
- no model shards loaded during these runs; shard-gated cases stayed skip-gated / no-shard
- post-run process check showed no lingering pytest/test worker processes

Verdict: GREEN
