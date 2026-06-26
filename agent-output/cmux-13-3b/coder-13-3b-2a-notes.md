# Coder notes — Story 13.3b-2a

## Result

Status: OK.

Implemented additive-only real indexer helpers in `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`:

- `_indexer_scorer_mlx`
- `_indexer_mlx`

Added tests:

- `tests/test_indexer_scorer_mlx_parity.py`
- `tests/test_indexer_mlx_parity.py`

Advanced sha16 pin sites for additive `deepseek_v4.py` baseline:

- `tests/test_deepseek_v4_real_config_reference_forward.py`
- `tests/test_numpy_real_forward_reference_composition.py`
- `python-envs/mlx/src/ds4_ft_mlx/real_forward_intermediate_dump.py`

New vendor sha16: `812df0f7a34c0f07`.

## Red-first evidence

Initial AC1/AC2 run before implementation:

```text
2 failed
ImportError: cannot import name '_indexer_scorer_mlx'
ImportError: cannot import name '_indexer_mlx'
```

## Rope-landmine resolution

Probe result for FROZEN `_csa_windowed_compressor_mlx:347`:

```text
has suppress param? False
uses full tables? True
uses apply full? True
uses tail tables? False
```

Resolution: did NOT reuse it end-to-end.

Implemented Ca/Cb overlap pooling inline inside new `_indexer_mlx`, then applied 13.3b-1 helpers:

- `_compress_rope_yarn_tail_tables_mlx`
- `_apply_rope_tail_mlx`

Channel scope: trailing `qk_rope_head_dim=64` of `index_head_dim=128`.

Positions:

- compressed_kv: `arange(n_windows) * rate`
- q: `position_ids`, broadcast per index head before tail-RoPE apply

FROZEN `_csa_windowed_compressor_mlx` body left byte-identical.
FROZEN tiny `_csa_indexer_mlx` body left byte-identical.

## AC results

```text
python-envs/mlx/.venv/bin/python3 -m pytest tests/test_indexer_scorer_mlx_parity.py tests/test_indexer_mlx_parity.py --no-header -q
2 passed
```

AC1:

- scorer output shape `[1,17,9]`
- fp32 return
- matches torch `DeepseekV4IndexerScorer.forward` at `atol=1e-3`
- `weights_proj` shape `(64,4096)` asserted

AC2:

- real header shapes asserted for layer 2 indexer tensors
- score tensor matches torch at `atol=1e-3`
- output shape `[1,257,64]`
- `-1` sentinel mask exact
- non-sentinel valid pick set exact per query
- tie order not asserted

## Regression results

13.3b-1:

```text
4 passed
```

13.3a nn:

```text
12 passed
```

13.2 FP4 selected regression:

```text
10 passed
```

Tiny CSA targeted 11.14/11.15:

```text
24 passed, 94 deselected, 8 subtests passed
```

Full suite:

```text
1 failed, 563 passed, 13 skipped, 96 subtests passed
```

Single failure:

```text
tests/test_deepseek_v4_mlx_port.py::DeepSeekV4MlxPortTests::test_finetune_readiness_report_records_b0b_a_without_unblocking_b0
AssertionError: 9 != 8
```

This matches expected one-red carve-out shape: `561+2new = 563` pass / `1` red / `13` skip.
No introduced red observed.

## Byte-intactness

Frozen-body proof vs `HEAD` (`1f5d8e4`): source hash + AST identical for protected symbols:

- `_csa_config_error`
- `_require_csa_config`
- `_csa_windowed_compressor_mlx`
- `_csa_indexer_mlx`
- `_csa_attention_mlx`
- `_csa_compressor_mlx`
- `_attention_mlx`
- `_attention_real_mlx`
- `_hca_compressor_mlx`
- `_compress_rope_yarn_tail_tables_mlx`
- `_apply_rope_tail_mlx`
- `_apply_rope_full_mlx`
- `_rope_full_tables_mlx`
- `_rope_tail_tables_mlx`
- `_hyperconnection_mlx`
- `_hyperhead_mlx`
- `sanitize_weights`

Changed protected list: `[]`.
New functions only: `['_indexer_scorer_mlx', '_indexer_mlx']`.

`git diff --check`: clean.

## Blast-radius notes

No CSA attention wiring.
No `_attention_real_mlx` edit.
No nn-port edit.
No convert script edit.
No FROZEN body edit.
No C++.
