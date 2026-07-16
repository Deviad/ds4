# Review — Story 13.3b-3

Verdict: PASS / GREEN.

Reviewer: xhigh-reviewer. Commit reviewed: `d1f1488` vs parent `aa93c5d`.

## 4-axis audit

| Axis | Verdict | Evidence |
|---|---|---|
| Scope / blast radius | GREEN | No convert/remap/smoke-train files touched. `deepseek_v4.py` changed only dispatch branch. `deepseek_v4_nn.py` changed nn wiring. Test changes match AC1-AC4 plus sha-pin cascade. `real_forward_intermediate_dump.py` only advanced the vendor sha sentinel found by whole-tree pin scan. |
| Frozen-body integrity | GREEN | AST/source extraction from `HEAD~1` and `HEAD`: only top-level changed symbol in `deepseek_v4.py` is `_attention_mlx`; all protected helpers/classes byte-identical. `_attention_mlx` cr=0 suffix sha16 stayed `5b0849ad8172420e`. |
| Correctness / architecture | GREEN | Dispatch shape matches BA Q1 / ADR 0026 Decision-1. No `_grouped_linear_mlx`. `AttentionNN` guard lifted; CSA/HCA compressor/indexer leaves present; `__call__` exports real weights through `linear_weight` and delegates to `_attention_mlx`. |
| Validation / regression | GREEN | Targeted reviewer run: `10 passed`. Full suite reviewer run: `569 passed, 13 skipped, 2 warnings, 96 subtests passed`. `git diff --check` clean. |

## 8-axis review

### Axis 1 — Scope additive only

Verdict: GREEN.

`git show --stat --name-status d1f1488` changed:

- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py`
- AC test files:
  - `tests/test_13_3b_1_dispatch_additive.py`
  - `tests/test_13_3b_3_nn_mixed_layers_forward.py`
  - `tests/test_attention_real_mlx_hca_parity.py`
  - `tests/test_csa_attention_real_mlx_parity.py`
  - `tests/test_deepseek_v4_real_config_reference_forward.py`
  - `tests/test_numpy_real_forward_reference_composition.py`
- `python-envs/mlx/src/ds4_ft_mlx/real_forward_intermediate_dump.py` — sha sentinel only, required by Axis 7 whole-tree scan.
- `agent-output/cmux-13-3b/coder-13-3b-3-notes.md` — handoff artifact.

No convert/remap/shim script touched. No new real-branch helpers in `deepseek_v4.py`; AST top-level symbol count stayed 53 → 53.

`deepseek_v4.py` diff is exactly the two-line dispatch insertion before HCA fallback.

### Axis 2 — FROZEN bodies byte-intact

Verdict: GREEN.

Independent AST/source extraction from `HEAD~1` and `HEAD`:

- top-level changed symbols in `deepseek_v4.py`: only `_attention_mlx` (`04ad4ab7923ccef3` → `96dea976b44639f7`).
- `_attention_mlx` cr=0 body after the compression dispatch stayed byte-identical: sha16 `5b0849ad8172420e` both revs.
- Protected symbols byte-identical, including:
  - `_csa_config_error` `d0c1651d480e06b0`
  - `_csa_attention_mlx` `84f205c0274606f5`
  - `_attention_real_mlx` `d6370a97a0be03c2`
  - `_csa_attention_real_mlx` `b5bba4f3d2b56491`
  - `_indexer_mlx` `caf1d778abd6d224`
  - `_indexer_scorer_mlx` `dc8d82183a8da49d`
  - `_csa_compressor_real_mlx` `95842e1d6e9bec72`
  - `_csa_block_bias_mlx` `8aa86eaa9b91bfaf`
  - `_hca_compressor_mlx` `09bcfab4b6cb7d6b`
  - `_compress_rope_yarn_tail_tables_mlx` `39e77a9b245e8dfc`
  - `_apply_rope_*`, `_rope_*_tables_mlx`, `_hyperconnection_mlx`, `_hyperhead_mlx`, `_causal_sliding_mask_mlx`, `sanitize_weights`, `Model`, FP4/i8 dequant helpers.

### Axis 3 — Dispatch additive / ADR 0026 Decision-1

Verdict: GREEN.

Current `_attention_mlx:1199-1204` exact shape:

```python
if args.compression_ratio != 0:
    if _csa_config_error(args) is None:
        return _csa_attention_mlx(args, x, weights, index_topk=index_topk)
    if args.compression_ratio == 4:
        return _csa_attention_real_mlx(args, x, weights, index_topk=index_topk)
    return _attention_real_mlx(args, x, weights, index_topk=index_topk)
```

Tiny return line byte-identical. HCA fallback line byte-identical. cr=0 body byte-identical via Axis 2.

### Axis 4 — GroupedLinear absent

Verdict: GREEN.

`_grouped_linear_mlx` absent across `python-envs/mlx/src`, `tests`, and `scripts`.

Grouped-o loop remains inline:

- `_attention_mlx:1249-1259`
- `_attention_real_mlx:1086-1096`
- `_csa_attention_real_mlx:992-1002`

No extraction; no FROZEN cr=0 body edit.

### Axis 5 — `AttentionNN` §4.1 wiring / AC4

Verdict: GREEN.

Confirmed in `deepseek_v4_nn.py`:

- old guard `compression_ratio != 0 → NotImplementedError("13.3a-1 AttentionNN supports only compression_ratio=0")` removed.
- new ratio selection from `compress_ratios` / `compress_rates` / `layer_types`.
- `AttentionCompressorNN` leaves: `wkv`, `wgate`, `ape`, `norm.weight`.
- `AttentionIndexerNN` leaves: nested `compressor.{wkv,wgate,ape,norm.weight}`, `weights_proj`, `wq_b`.
- CSA `AttentionNN` creates compressor + indexer.
- HCA `AttentionNN` creates compressor only.
- `__call__` keeps cr=0 path as `_attention_mlx(self.config, x, weights)`.
- real paths build extended weights via `linear_weight` (QuantizedLinear + LoRA-aware) and delegate to `_attention_mlx`; CSA passes `index_topk`.

AC4 test `tests/test_13_3b_3_nn_mixed_layers_forward.py` covers stale all-sliding config normalization plus mixed sliding/CSA/HCA `AttentionNN` layers against functional dispatch. Math anti-circularity is covered transitively by direct-helper torch parity in the CSA/HCA parity tests and existing cr=0 tests.

### Axis 6 — AC1-AC3 dispatch routing / parity-through-dispatch

Verdict: GREEN.

`tests/test_13_3b_1_dispatch_additive.py` extended:

- tiny cr=4 + `_csa_config_error is None` still routes to `_csa_attention_mlx` and equals direct tiny result byte-for-byte.
- real cr=4 spy routes to `_csa_attention_real_mlx` with `index_topk=7`.
- real cr=128 spy routes to `_attention_real_mlx` with `index_topk=None`.

`tests/test_csa_attention_real_mlx_parity.py` and `tests/test_attention_real_mlx_hca_parity.py` extended:

- direct helper still parity-checked to torch.
- `_attention_mlx` routed output equals direct helper output exactly (`atol=0.0, rtol=0.0`).

### Axis 7 — sha-pin cascade / Test #8 scrutiny

Verdict: GREEN.

Whole-tree scan roots: `tests/`, `scripts/finetune_ds4.py`, `python-envs/mlx/src/`.

`deepseek_v4.py` whole-file sha16:

- parent `aa93c5d`: `dc5aaaab9bb079d2`
- head `d1f1488`: `96c39168c78e5fd9`

Old `dc5aaaab9bb079d2` absent in scanned non-pycache files. New `96c39168c78e5fd9` appears in all legitimate pin sites:

- `tests/test_numpy_real_forward_reference_composition.py:397`
- `tests/test_deepseek_v4_real_config_reference_forward.py:383`
- `python-envs/mlx/src/ds4_ft_mlx/real_forward_intermediate_dump.py:61`

`test_deepseek_v4_real_config_reference_forward.py` also updates `tests/test_deepseek_v4_mlx_port.py` expected sha from stale `bde0fa84f281e422` to current `bf65d624186fc129`. Scrutiny result: legitimate baseline update, not tautology/relaxation. The invariant still asserts exact `_sha16(p) == EXPECTED_SHAS[rel]`; no assertion was removed or loosened. Current `tests/test_deepseek_v4_mlx_port.py` hash is `bf65d624186fc129`, and the file contains the updated real cr=4 fail-closed expectation for the new dispatch contract.

### Axis 8 — Regression / no scope explosion

Verdict: GREEN.

Reviewer targeted run:

```text
10 passed, 1 warning in 3.53s
```

Reviewer full suite:

```text
569 passed, 13 skipped, 2 warnings, 96 subtests passed in 121.04s
```

`git diff --check` clean for reviewed files.

No convert/remap, no new real-branch helpers, no `_grouped_linear_mlx`, no C/C++/Metal changes.

## Final verdict

All 8 axes GREEN.

`{"status":"ok","role":"Reviewer"}`
