# Story 13.3b-5d independent review

## Verdict: FAIL

Real 4096 smoke remains forbidden. Sparse forward/VJP math appears numerically correct, but backward implementation violates one-expert graph-lifetime contract and independent memory probe shows peak growth with number of non-empty experts. Acceptance tests also fail to establish several pinned contracts. Intended pre-existing 13.3b-5/5b changes remain outside index/tracking scope.

## Findings

### 1. BLOCKER — custom VJP accumulates every non-empty expert graph before first barrier

`python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py:592-609` builds `_Q` across all experts, stores each `_y_e` and `_f_e` in `_y_cache`/`_f_cache`, then calls `mx.eval(_Q)` only after loop. This is opposite of required per-expert evaluation/release. At real balanced routing, all 256 experts can be non-empty even though `K=6`; Q graph can retain/recompute all expert dequant graphs together, recreating 24 GiB all-expert dequant floor that 13.3b-5d exists to remove. Cached routed outputs also retain all-expert selected activations through score pass.

Independent fixed-`K=2` hash-routing forward+VJP probes, separate fresh processes, `T=1024,H=1024,I=512`, all experts non-empty:

```text
E=2  temp_peak_delta=474,644,864 bytes
E=4  temp_peak_delta=612,254,436 bytes
E=8  temp_peak_delta=897,009,068 bytes
```

Final retained delta stayed ~8.4 MiB, while operation peak grew with expert count. This directly disproves claimed one-expert active bound.

Required correction: evaluate Q accumulator after each expert and do not cache expert outputs across experts. Recompute one expert in later score/direct-input passes as needed, evaluate/release each pass per expert, then rerun fixed-`K`, all-experts-nonempty memory proof.

### 2. BLOCKER — new tests do not prove pinned parity and memory contracts

`tests/test_deepseek_v4_nn_sparse_routed_backward.py` has material legitimacy gaps:

- `:301-350` test named `test_sparse_gate_weight_gradient_matches_dense` never computes dense autodiff and never compares `g_sparse` with `g_num`; `g_num` is constructed but unused. Comment claiming dense autodiff is zero is false. Independent probe found dense gate gradients non-zero.
- `:353-379` clamp-near test checks only finite/non-zero, not sparse-vs-dense parity at `atol=rtol=1e-5`.
- `:123-143` forward fixture uses uniform tokens and identical packed bytes/scales for every expert, contrary to pinned nonuniform fixture. Routing/scatter mistakes can be masked.
- `:242-261` “shared expert identity” keeps gate/up weights zero, so shared output remains zero; test cannot detect omission, duplication, or wrong-input shared execution.
- `:386-404` detached-index test applies `mx.stop_gradient` inside test but does not observe implementation boundary.
- `:407-421` duplicate-row cotangent test does not construct duplicate route slots or assert expected summed cotangent.
- `:563-616` dense and sparse cases run sequentially in same child and do not reset peak between them; not pinned separate-process-per-case measurement.
- `:708-799` real-dimension probe uses `E=1,K=1`, so it cannot show cross-expert graph release. It asserts only post-evaluation active delta and ignores 4,076,454,246-byte operation peak.

Independent discriminating fixture with distinct expert packed bytes/scales and nonuniform tokens did support implementation math:

```text
forward_max_abs  1.4901161193847656e-08
xgrad_max_abs    2.384185791015625e-07
gategrad_max_abs 1.4901161193847656e-08
gate_dense_nonzero True
```

Those ad-hoc results do not replace required tracked tests.

### 3. BLOCKER — intended 13.3b-5/5b scope not reproducible from index/fresh clone

Current scope state:

```text
docs/adr/0026-csa-hca-indexer-mlx-real-port.md                         UNTRACKED
scripts/apply_passthrough_chat_template.py                             UNTRACKED
python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py     TRACKED+UNSTAGED
python-envs/mlx/src/ds4_ft_mlx/real_forward_intermediate_dump.py       TRACKED+UNSTAGED
tests/test_deepseek_v4_real_config_reference_forward.py                TRACKED+UNSTAGED
tests/test_numpy_real_forward_reference_composition.py                 TRACKED+UNSTAGED
```

Prior 13.3b-5 task explicitly required passthrough script tracked/staged. Current staged slice omits script, ADR 0026 amendment, sanctioned FROZEN stop-gradient fix, and hash-pin cascade. Committing index as reviewed would omit required backward fix and its architectural/reproducibility evidence.

Required correction: resolve full intended chain of custody before next review; include sanctioned FROZEN one-line fix, all three pin advances, ADR 0026 amendment, and passthrough script in tracked/staged scope or explicitly split/commit them with reproducible predecessor evidence.

## Audit

### Handoffs read in full

- `agent-output/cmux-13-3b/task-reviewer-13-3b-5d.md`
- `agent-output/cmux-13-3b/requirements-13-3b-5d-routed-fp4.md`
- `agent-output/cmux-13-3b/architecture-13-3b-5d-routed-fp4-backward.md`
- `agent-output/cmux-13-3b/coder-13-3b-5d-notes.md`
- `agent-output/cmux-13-3b/coder-13-3b-5b-stop.md`
- `agent-output/cmux-13-3b/coder-13-3b-5c-memory400-stop.md`
- 13.3b-5/5b coder and architecture evidence needed for FROZEN/hash/template scope

### Positive checks

- Routed forward semantics, duplicate collapse, score-only weighting, correction-bias selection-only, token reshape/order, and one shared-expert call: code inspection consistent.
- Custom score derivative algebra correct; direct expert-input cotangent correctly uses `mx.vjp(forward_one)`.
- Integer route metadata passed through `mx.stop_gradient` before `.tolist()` host materialization.
- No module-import `mx.disable_compile()` side effect; focused import test green.
- Exact FROZEN working-tree SHA reproduced:
  `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b`.
- FROZEN diff contains only sanctioned `_csa_block_bias_mlx` `mx.stop_gradient` wrap.
- ADR 0024 unchanged.
- ADR 0025, backlog 13.3b-5d, technical-spec wrapper/arguments/telemetry, and ADR 0026 amendment semantically consistent.
- Hash pins use `5e11a9c4d82aeb24`; old hash appears only as ADR history.
- Passthrough template implementation preserves exact concatenation and idempotent/force guard semantics.
- Formula values correct: 96 KiB route metadata, 96 MiB one-expert dequant, 24/40 GiB old floors, 1.25 GiB selected budget, 300 GiB process ceiling.
- Independent real-dimension no-shard probe reproduced:
  `active_delta=201,326,592`, `peak_bytes=4,076,454,246`, finite output/cotangent.
- `git diff --check` and `git diff --cached --check`: clean.

### Test execution

```text
tests/test_deepseek_v4_nn_sparse_routed_backward.py
21 passed, 1 warning in 4.63s

16-file focused regression
117 passed, 8 skipped, 1 warning, 16 subtests passed in 24.44s
```

All 16 cited focused test files returned non-empty `git ls-files` results. Pass counts remain insufficient because findings 1-2 expose contract/coverage failures.

### Safety/scope

- No model shards loaded.
- No full model or real smoke run.
- No production, test, ADR, backlog, or technical-spec edits made by Reviewer.
