# Review — Story 13.3b-4 r2

Verdict: PASS / GREEN.

Reviewer: xhigh-reviewer. Staged work reviewed at `HEAD=58194a9ce571e21dbe666109fa77d72cca3e80b6` via `git diff --cached`.

r0 blocker re-check: RESOLVED. AC2 now loads real safetensors payload tensors from `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim`, strict-loads the compact nn model, verifies real FP4 bytes, and runs finite forward.

## 8-axis audit

### Axis 1 — Scope / blast radius

Verdict: GREEN.

Evidence:

```text
HEAD 58194a9
cached stat: 7 files changed, 969 insertions(+)
A .cmux-status/coder.done
A agent-output/cmux-13-3b/coder-13-3b-4-notes.md
A agent-output/cmux-13-3b/coder-13-3b-4-r2-notes.md
A scripts/remap_ds4_nn_weights.py
A tests/test_remap_ds4_nn_weights_keyset.py
A tests/test_remap_ds4_nn_weights_load.py
A tests/test_remap_ds4_nn_weights_regression.py
```

No staged protected files:

```text
git diff --cached --name-only -- deepseek_v4.py deepseek_v4_nn.py shim_ds4_safetensors.py docs/... => empty
```

No staged `deepseek_v4.py`, `deepseek_v4_nn.py`, `scripts/shim_ds4_safetensors.py`, ADR, architecture, technical spec, or backlog edits.

### Axis 2 — FROZEN / nn-port / shim byte+AST identity

Verdict: GREEN.

Direct worktree-vs-HEAD byte and AST comparison:

```text
python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py
  tracked=True bytes_same=True sha_head=96c39168c78e5fd9 sha_wt=96c39168c78e5fd9 ast_same=True
python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py
  tracked=True bytes_same=True sha_head=994980c4ac41e4db sha_wt=994980c4ac41e4db ast_same=True
scripts/shim_ds4_safetensors.py
  tracked=True bytes_same=True sha_head=bcde9485a0a0bcdf sha_wt=bcde9485a0a0bcdf ast_same=True
```

### Axis 3 — Correctness / §5.1 remap mechanics

Verdict: GREEN.

Staged remap script remains the 465-line convert-side script and was re-verified against real checkpoint headers and the probed nn parameter tree:

```text
expected 1460 remapped 1460 missing 0 extra 0
dropped 1940 stacked_targets 258
synthetic_zero ['model.layers.0.mlp.e_score_correction_bias', 'model.layers.1.mlp.e_score_correction_bias', 'model.layers.2.mlp.e_score_correction_bias']
```

Representative real header mappings:

```text
model.layers.2.self_attn.compressor.ape F32 (4, 1024) identity
model.layers.2.self_attn.indexer.compressor.ape F32 (4, 256) identity
model.layers.3.self_attn.compressor.ape F32 (128, 512) identity
model.layers.2.mlp.experts.w1_weight U8 (256, 2048, 2048) stack_uint8 256 sources
model.layers.2.mlp.experts.w1_scale BF16 (256, 2048, 128) stack 256 sources
```

No missing/extra remapped keys. No remap bug exposed by r2 real-load test.

### Axis 4 — ape-verbatim / BA §0.4

Verdict: GREEN.

Forbidden transpose grep in staged remap script found no operation:

```text
grep -nE '\.T\b|transpose|swapaxes|permute|moveaxis' scripts/remap_ds4_nn_weights.py
16: copied verbatim in checkpoint token-major layout. No transpose is performed.
```

Ape transforms remain `identity` for CSA compressor, CSA indexer compressor, and HCA compressor. No `.T`, `transpose`, `swapaxes`, `permute`, or `moveaxis` operation applies to ape tensors.

### Axis 5 — AC2 legitimacy / anti-tautology r2 focus

Verdict: GREEN.

r0 blocker was synthetic `_zeros/_ones` with no safetensors payload. r2 resolves this.

Grep evidence on staged `tests/test_remap_ds4_nn_weights_load.py`:

```text
safe_open|load_file|get_tensor|safetensors hits: 17
_zeros|_ones|mx.zeros|mx.ones hits: 0
/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim refs: 1
line count: 268
```

Real payload evidence:

```text
CKPT = Path("/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim")
REAL_TO_LOCAL_LAYER = {0: 0, 2: 1, 3: 2}
```

The AC2 test:

- loads globals: `embed.weight`, `head.weight`, `norm.weight`, `hc_head_{fn,base,scale}`
- loads original L0 sliding, original L2 CSA, original L3 HCA
- compacts real layers `0,2,3` into local layers `0,1,2`
- uses `safe_open(...).get_tensor` for non-BF16 real tensors
- uses bounded BF16 `mx.load` fallback only because local safetensors MLX `get_tensor(BF16)` fails in this venv
- asserts compact model remapped keyset equals probed `model.parameters()` keyset
- asserts `model.load_weights(list(remapped.items()), strict=True)` succeeds
- runs `seq_len = 8` forward
- asserts output shape `(1, seq_len, vocab_size)`
- asserts all logits finite
- verifies routed FP4 expert stack is real:
  - `model.layers.1.mlp.experts.w1_weight`
  - shape `(256, 2048, 2048)`
  - dtype `mx.uint8`
  - nonzero sample
  - sample equals original `layers.2.ffn.experts.0.w1.weight` bytes read via `safe_open(...).get_tensor`

RAM / payload scope evidence:

```text
selected_keys=4716 selected_shards=5 selected_payload_gib=12.906
model-00001-of-00046.safetensors keys=1 payload_gib=0.986
model-00002-of-00046.safetensors keys=1565 payload_gib=3.632
model-00004-of-00046.safetensors keys=1576 payload_gib=3.667
model-00005-of-00046.safetensors keys=1569 payload_gib=3.634
model-00045-of-00046.safetensors keys=5 payload_gib=0.987
```

Measured AC2 test memory:

```text
1 passed, 1 warning in 4.08s
13,619,789,824 maximum resident set size
```

This is under the 16 GiB budget and does not materialize the full 46-shard / ~162GB checkpoint.

### Axis 6 — Tracking hygiene HARD RULE

Verdict: GREEN.

All baseline-participating remap tests are tracked in the index, not `??`:

```text
100644 ed086578f1c86af7fd91e85d3e3e348899a3ddb5 0 tests/test_remap_ds4_nn_weights_keyset.py
100644 f08d921146c7fb2efc7771bce0456ae0dbd33f2a 0 tests/test_remap_ds4_nn_weights_load.py
100644 5a8b0e485538ac62c3ab3c8252668aa9fee1c0f0 0 tests/test_remap_ds4_nn_weights_regression.py
```

`git status --short -- tests/test_remap_ds4_nn_weights_*.py` shows `A`, not `??`.

HEAD remains `58194a9`; Coder did not commit before review.

### Axis 7 — sha-pin cascade / stale pin no-op

Verdict: GREEN.

No staged changes to protected FROZEN file, nn port, shim, sha-pin tests, or docs pins. No phantom/stale pin advancement observed. Direct byte+AST identity in Axis 2 is stronger than a vacuous diff guard.

### Axis 8 — Regression / no scope explosion

Verdict: GREEN.

Validation run by Reviewer:

```text
python -m py_compile scripts/remap_ds4_nn_weights.py \
  tests/test_remap_ds4_nn_weights_keyset.py \
  tests/test_remap_ds4_nn_weights_load.py \
  tests/test_remap_ds4_nn_weights_regression.py
py_compile ok

git diff --check --cached
diff_check ok

PYTHONPATH="$PWD:python-envs/mlx/src" pytest -q \
  tests/test_remap_ds4_nn_weights_keyset.py \
  tests/test_remap_ds4_nn_weights_load.py \
  tests/test_remap_ds4_nn_weights_regression.py
3 passed, 1 warning in 5.52s

PYTHONPATH="$PWD:python-envs/mlx/src" pytest -q
572 passed, 13 skipped, 2 warnings, 96 subtests passed in 113.02s
```

No C++ files, dependency changes, FROZEN edits, nn-port edits, shim edits, or training/smoke scope creep detected.

## Final verdict

PASS / GREEN.

Axis 5 r0 blocker is closed. r2 AC2 is no longer synthetic; it exercises real safetensors payload, real FP4 packed bytes, strict load, and finite forward under the 16 GiB budget.
