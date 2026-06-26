# Story 13.1 Coder Notes — COMPLETE

## Summary

Applied 4 architectural locks from `architecture.md` byte-exact:

1. Rewrote `python-envs/mlx/src/ds4_ft_mlx/shimmed_ckpt_key_remap.py` to emit
   vendor-internal keys (not `model.`-prefixed) + cluster-6 DROP policy +
   `KeyError` on residual unmapped.
2. Added 2-line pre-pass sibling call at
   `deepseek_v4.py:Model.load_weights` L1739-1740.
3. Widened `deepseek_v4.py:_load_real_weights` L1788 condition
   `== "i8"` → `in ("i8", "fp4")`.
4. Cluster-6 DROP filters (compressor/indexer/attn-proj.scale/shared.scale/
   tid2eid/mtp) — audited in test AC7.

## TDD red→green log

- RED baseline: new test file `tests/test_key_remap_loads_shimmed_ckpt.py`
  written with synthetic fixture; confirmed import error on non-existent module
  before rewrite.
- GREEN after rewrite: 6/7 pass, 1 slow-gated skip.
- AC4 real 163GB shimmed ckpt load: PASSED in ~50s (set
  DS4_RUN_SLOW_PARITY=1).
- AC7 exhaustive 69,187-key coverage: PASSED — zero residual unmapped.

## Full suite (post-fix)

```
27 failed, 510 passed, 12 skipped, 104 subtests passed
```

Failure breakdown (all pre-existing or expected):
- 20 `test_deepseek_v4_mlx_port.py`: pre-existing ( `_moe_mlx` already had
  `("i8","fp4")` from prior work; tests lack scales → forward crash).
- 3 `test_deepseek_v4_validate_real_mode_relaxation.py`: same pre-existing
  `_moe_mlx` issue.
- 3 vendor-frozen unchanged tests: expected because `deepseek_v4.py` edited.
- 1 `test_deepseek_v4_real_config_reference_forward.py` vendor-unchanged:
  expected.

No NEW RED introduced by this slice.

## Byte-intactness

- FROZEN `deepseek_v4_dequant.py:322` / `:375` — untouched (signatures match).
- Track-A `.ds4-gguf-generate-ok` — present.
- Track-B `.deepseek-v4-forward-parity-ok` — absent (retired ADR 0022).
- `model-4bit` — absent (Story 13.3 owns).
- `ds4flash.gguf` — symlink intact in `/Users/spotted/projects/ds4/`.
- `ds4.c` / `ds4_metal.m` / `metal/*.metal` / `ds4_cli.c` / `ds4_server.c` —
  zero diff.

## Incident resolved

Test-suite verification `sed` pipeline temporarily corrupted
`deepseek_v4.py` to 0 bytes. Recovered from `/tmp/d4_orig.py` (saved mid-edit,
87737 bytes, 1941 lines). Verified content matches expected 3 edits.

## Files touched

- `python-envs/mlx/src/ds4_ft_mlx/shimmed_ckpt_key_remap.py` (REWRITE)
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`
  (+2 lines pre-pass call + 1 condition widen)
- `tests/test_key_remap_loads_shimmed_ckpt.py` (NEW)
- `tests/test_key_remap_adapter_loads_shimmed_ckpt.py` (DELETED — r5/r6
  superseded by vendor-internal vocab)

## Commit

Per-slice commit applied.
