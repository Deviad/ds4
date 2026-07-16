# Review — Story 13.3b-4

Verdict: BLOCKED.

Reviewer: xhigh-reviewer. Staged work reviewed at `HEAD=58194a9` via `git diff --cached`.

Blocking finding: AC2 legitimacy RED. `tests/test_remap_ds4_nn_weights_load.py` uses synthetic tiny zero/one tensors, not a real checkpoint sub-checkpoint. Task/BA axis requires real sub-checkpoint load+forward finite.

## 4-axis audit

| Axis | Verdict | Evidence |
|---|---|---|
| Scope / blast radius | GREEN | Cached diff contains exactly 6 staged files: `.cmux-status/coder.done`, `agent-output/cmux-13-3b/coder-13-3b-4-notes.md`, `scripts/remap_ds4_nn_weights.py`, and 3 new `tests/test_remap_ds4_nn_weights_*.py`. No staged protected files. HEAD still `58194a9`; Coder did not commit. |
| Frozen / protected integrity | GREEN | Direct `git show HEAD:<path>` byte+AST comparison: `deepseek_v4.py` sha16 `96c39168c78e5fd9`, `deepseek_v4_nn.py` sha16 `994980c4ac41e4db`, `scripts/shim_ds4_safetensors.py` sha16 `bcde9485a0a0bcdf`; all `bytes_same=True`, `ast_same=True`. |
| Correctness / architecture | GREEN for implementation mechanics | Real header probe: expected nn keys `1460`, remapped specs `1460`, missing `0`, extra `0`; dropped `1940`, stacked targets `258`, synthetic zeros for layers `0,1,2`, payload bytes read `0`. Ape specs remain identity: `(4,1024)`, `(4,256)`, `(128,512)`. Routed experts stack to `(256,2048,2048)` / `(256,2048,128)` with U8/BF16. |
| Validation / regression | BLOCKED | Reviewer ran targeted new tests: `3 passed`. Reviewer ran full suite: `572 passed, 13 skipped, 2 warnings, 96 subtests passed`. But AC2 test is not the required real sub-checkpoint load; it fabricates `_tiny_mixed_config()` and `_tiny_ckpt_style_weights()` from zeros/ones, then loads those. Passing suite cannot close AC2. |

## 8-axis review

### Axis 1 — Scope / blast radius

Verdict: GREEN.

Evidence:

```text
HEAD=58194a9
cached name-status:
A .cmux-status/coder.done
A agent-output/cmux-13-3b/coder-13-3b-4-notes.md
A scripts/remap_ds4_nn_weights.py
A tests/test_remap_ds4_nn_weights_keyset.py
A tests/test_remap_ds4_nn_weights_load.py
A tests/test_remap_ds4_nn_weights_regression.py
```

No staged diff for:

- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py`
- `scripts/shim_ds4_safetensors.py`
- ADRs / architecture / technical spec / backlog

Unrelated dirty/untracked workspace files exist, but not in staged slice scope.

### Axis 2 — FROZEN / nn-port / shim byte-intactness

Verdict: GREEN.

Direct byte+AST comparison against `HEAD=58194a9`:

```text
python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py
  tracked=True bytes_same=True sha256_head=96c39168c78e5fd9 sha256_wt=96c39168c78e5fd9 ast_same=True
python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py
  tracked=True bytes_same=True sha256_head=994980c4ac41e4db sha256_wt=994980c4ac41e4db ast_same=True
scripts/shim_ds4_safetensors.py
  tracked=True bytes_same=True sha256_head=bcde9485a0a0bcdf sha256_wt=bcde9485a0a0bcdf ast_same=True
```

Full-file byte identity covers the FROZEN bodies, nn-port ownership, and shim non-extension requirement.

### Axis 3 — Correctness / architecture

Verdict: GREEN for remap implementation.

Evidence from real checkpoint header/index probe:

```text
ckpt_exists True index_exists True
expected_keys 1460 spec_keys 1460 missing 0 extra 0
dropped 1940 stacked_targets 258
synthetic_zero ['model.layers.0.mlp.e_score_correction_bias', 'model.layers.1.mlp.e_score_correction_bias', 'model.layers.2.mlp.e_score_correction_bias']
payload_bytes_read 0
```

Sample checked mappings:

```text
embed.weight => model.embed_tokens.weight
head.weight => lm_head.weight
layers.2.attn.wq_a.weight => model.layers.2.self_attn.q_a_proj.weight
layers.2.attn.wq_a.scale => drop
layers.2.attn.indexer.wq_b.scale => drop
layers.2.ffn.shared_experts.w1.scale => drop
layers.2.ffn.experts.0.w1.weight => stack model.layers.2.mlp.experts.w1_weight
mtp.0.embed.weight => drop
```

Routed FP4 stays packed:

```text
model.layers.2.mlp.experts.w1_weight U8 (256, 2048, 2048) stack_uint8
model.layers.2.mlp.experts.w1_scale  BF16 (256, 2048, 128) stack
```

Shared experts map w1/w3/w2 correctly:

```text
w1 -> shared_experts.gate_proj.weight
w3 -> shared_experts.up_proj.weight
w2 -> shared_experts.down_proj.weight
```

### Axis 4 — ape-verbatim critical path

Verdict: GREEN.

`grep -nE '\bape\b|\.T\b|transpose|swapaxes|permute|moveaxis' scripts/remap_ds4_nn_weights.py tests/test_remap_ds4_nn_weights_*.py` found only comments/assertions mentioning no transpose and direct ape rename rules. No `.T`, `transpose`, `swapaxes`, `permute`, or `moveaxis` operation in remap code.

Real header probe:

```text
model.layers.2.self_attn.compressor.ape (4, 1024) identity
model.layers.2.self_attn.indexer.compressor.ape (4, 256) identity
model.layers.3.self_attn.compressor.ape (128, 512) identity
```

### Axis 5 — AC1/AC2/AC3 legitimacy / anti-tautology

Verdict: BLOCKED.

AC1: GREEN.

- Expected keyset comes from real `deepseek_v4_nn.Model(...).parameters()` via `tree_flatten`, not handwritten list.
- Real checkpoint header remap equals probed nn tree: `1460 == 1460`, missing `0`, extra `0`.

AC2: RED / BLOCKER.

Task requirement: real sub-checkpoint load+forward finite.

Observed test is synthetic tiny fixture:

```text
tests/test_remap_ds4_nn_weights_load.py:25  def _tiny_mixed_config() -> dict[str, object]:
tests/test_remap_ds4_nn_weights_load.py:80  def _tiny_ckpt_style_weights(cfg: dict[str, object]) -> dict[str, mx.array]:
tests/test_remap_ds4_nn_weights_load.py:100-170 raw[...] = _zeros/_ones(...)
tests/test_remap_ds4_nn_weights_load.py:182 remapped, report = remap.remap_weight_dict(_tiny_ckpt_style_weights(cfg), config=cfg)
tests/test_remap_ds4_nn_weights_load.py:192 model.load_weights(list(remapped.items()), strict=True)
```

`grep` evidence:

```text
AC2 load test has no /Volumes path, no safetensors.safe_open/load_file, no model.safetensors index/header payload load.
Only AC1 keyset test touches /Volumes/Data NVME/mlx-ft/ds4/hf-f8shim, and only header/index data.
```

This proves strict load + forward for a fabricated 3-layer zero/one topology, not for any real checkpoint tensor payload. That fails the stated AC2 legitimacy axis.

AC3: GREEN.

- Regression test imports remap module, runs tiny CSA direct vs routed path, checks byte/list equality, and confirms ape remap identity.
- Production protected files byte-identical, so tiny path not mutated.

Required fix before GREEN:

- Add reviewer-visible AC2 coverage that loads actual safetensors payload tensors from `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim` for at least one sliding + one CSA + one HCA sub-checkpoint path, remaps with `remap_weight_dict`, strict-loads nn module, runs finite forward; or update BA/Architect acceptance if synthetic tiny topology is intentionally sufficient. Current test can stay as unit coverage, but cannot satisfy AC2 as written.

### Axis 6 — Tracking hygiene / commit gating

Verdict: GREEN.

Evidence:

```text
git ls-files --stage -- tests/test_remap_ds4_nn_weights_keyset.py tests/test_remap_ds4_nn_weights_load.py tests/test_remap_ds4_nn_weights_regression.py
100644 ed086578f1c86af7fd91e85d3e3e348899a3ddb5 0 tests/test_remap_ds4_nn_weights_keyset.py
100644 458ea47b9ae9c9f435e9e283cd6fed0627114662 0 tests/test_remap_ds4_nn_weights_load.py
100644 5a8b0e485538ac62c3ab3c8252668aa9fee1c0f0 0 tests/test_remap_ds4_nn_weights_regression.py
```

`git status --short -- tests/test_remap_ds4_nn_weights_*.py` shows `A`, not `??`.

HEAD remains:

```text
58194a9ce571e21dbe666109fa77d72cca3e80b6
58194a9 docs: add HARd tracking rule for baseline test files (13.3b-3 lesson)
```

Coder did not commit.

### Axis 7 — sha-pin cascade slice invariant

Verdict: GREEN.

Evidence:

- No staged diff for `deepseek_v4.py`, `deepseek_v4_nn.py`, `scripts/finetune_ds4.py`, or known sha-pin tests.
- Grep for new remap symbols inside protected files returned no hits.
- Staged diff contains no phantom production sha-pin advancement; only coder notes mention protected sha prefixes.

### Axis 8 — Regression / no scope explosion

Verdict: GREEN for executed regression, BLOCKED overall by Axis 5.

Reviewer validation:

```text
python -m py_compile scripts/remap_ds4_nn_weights.py tests/test_remap_ds4_nn_weights_keyset.py tests/test_remap_ds4_nn_weights_load.py tests/test_remap_ds4_nn_weights_regression.py
py_compile ok

PYTHONPATH="$PWD:python-envs/mlx/src" pytest -q tests/test_remap_ds4_nn_weights_keyset.py tests/test_remap_ds4_nn_weights_load.py tests/test_remap_ds4_nn_weights_regression.py
3 passed, 1 warning in 0.49s

PYTHONPATH="$PWD:python-envs/mlx/src" pytest -q
572 passed, 13 skipped, 2 warnings, 96 subtests passed in 113.17s
```

No C++; no dependency changes; no smoke-train scope creep detected.

## Final verdict

BLOCKED.

Single blocking issue: AC2 test under review is synthetic tiny zero/one fixture, not real sub-checkpoint load+forward finite as required by task/BA. All other axes GREEN.
