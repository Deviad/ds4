# Coder notes — Story 13.3b-3

Status: GREEN.

## Scope delivered

- `deepseek_v4.py`
  - Additive `_attention_mlx` dispatch split only:
    - tiny cr=4 + `_csa_config_error(args) is None` → `_csa_attention_mlx` unchanged.
    - real cr=4 → `_csa_attention_real_mlx`.
    - cr=128 fallback → `_attention_real_mlx` unchanged.
  - No FROZEN helper body edits.

- `deepseek_v4_nn.py`
  - Lifted `AttentionNN` `compression_ratio != 0` guard.
  - Added per-layer ratio support from `compress_ratios` / `compress_rates` / `layer_types`.
  - Normalizes stale all-`sliding_attention` real config via `compress_ratios`.
  - Added `AttentionCompressorNN` and `AttentionIndexerNN` submodules:
    - CSA: `compressor.{wkv,wgate,ape,norm.weight}` + `indexer.compressor.{wkv,wgate,ape,norm.weight}` + `indexer.weights_proj` + `indexer.wq_b`.
    - HCA: `compressor.{wkv,wgate,ape,norm.weight}` only.
  - `AttentionNN.__call__` builds functional weight dict with `linear_weight(...)` for QuantizedLinear/LoRA-aware leaves and delegates to `_attention_mlx`.

- Tests
  - Extended dispatch routing legs.
  - Extended CSA/HCA parity tests with direct-vs-dispatch route-equivalence.
  - Added `tests/test_13_3b_3_nn_mixed_layers_forward.py` for stale config normalization + mixed sliding/CSA/HCA AttentionNN wiring.
  - Updated stale fail-closed expectations for real cr=4 dispatch now reaching real CSA validation.
  - Advanced sha pins for additive dispatch baseline.

## Dispatch shape chosen

Chose BA-pinned `cr == 4` CSA branch before HCA fallback:

```python
if args.compression_ratio != 0:
    if _csa_config_error(args) is None:
        return _csa_attention_mlx(args, x, weights, index_topk=index_topk)
    if args.compression_ratio == 4:
        return _csa_attention_real_mlx(args, x, weights, index_topk=index_topk)
    return _attention_real_mlx(args, x, weights, index_topk=index_topk)
```

Reason: keeps tiny return byte-identical and current cr=128 fallback byte-identical; routes real cr=4 to CSA helper.

## TDD / validation

Red first:

- New dispatch cr=4-real spy failed before branch: routed to HCA fallback.
- CSA parity-through-dispatch failed before branch: `_attention_mlx` reached HCA-only path.
- AttentionNN mixed-layer test failed before guard lift / submodule wiring.

Green runs:

- Targeted integration:
  - `5 passed` — dispatch + CSA/HCA route-equivalence + AttentionNN mixed layers.
- 13.3a nn regression:
  - `12 passed`.
- 13.3b regression:
  - `11 passed`.
- 13.2 FP4 regression:
  - `71 passed, 1 skipped, 16 subtests passed`.
- Tiny CSA / legacy MLX port:
  - `124 passed, 21 subtests passed`.
- Full suite:
  - `569 passed, 13 skipped, 2 warnings, 96 subtests passed`.

## Byte-intactness

AST/source proof against HEAD:

- FROZEN helpers all byte-identical: `_csa_config_error`, `_require_csa_config`, `_csa_attention_mlx`, `_csa_compressor_mlx`, `_csa_indexer_mlx`, `_csa_windowed_compressor_mlx`, `_csa_compressor_real_mlx`, `_csa_block_bias_mlx`, `_csa_attention_real_mlx`, `_indexer_mlx`, `_indexer_scorer_mlx`, `_hca_compressor_mlx`, `_attention_real_mlx`, `_hyperconnection_mlx`, `_hyperhead_mlx`, `sanitize_weights`.
- `_attention_mlx` cr=0 suffix sha16 stayed `5b0849ad8172420e`.
- `deepseek_v4.py` new sha16: `96c39168c78e5fd9`.
- `git diff --check` clean.

## Blast-radius notes

- No convert/remap touched.
- No `_grouped_linear_mlx` introduced.
- No C++ touched.
- `real_forward_intermediate_dump.py` vendor sha sentinel advanced to the sanctioned additive dispatch baseline.
- `test_deepseek_v4_mlx_port.py` is untracked in this checkout; local expectations updated so full workspace suite reflects new real cr=4 dispatch contract.
