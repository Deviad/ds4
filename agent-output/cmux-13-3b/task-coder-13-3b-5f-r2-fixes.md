# Story 13.3b-5f — Coder round-2 Reviewer fixes

## Goal
Address every finding in `review-13-3b-5f-r2.md`. No full model/shards/smoke.

## Model
Use user-authorized `openai-codex/gpt-5.5` high.

## Required fixes

1. **Real activation dtype path**
   - Implement ADR 0028 explicit `x32 = x_flat.astype(mx.float32)` before the routed custom-function boundary.
   - Preserve ordinary cast VJP back to BF16/FP16 upstream dtype.
   - Add BF16 and FP16 `SparseMoeBlockNN` forward + first-order gradient tests with finite/non-zero expected LoRA path.
   - One-expert wrapper validation must not cast then reject the original array inconsistently.

2. **Single tiled decomposition**
   - Remove unconditional `hidden_size<=32` and `intermediate_size<=32` serial overwrite branches.
   - Keep one BM8/BN8/BK32 tiled path.
   - Do not weaken 1e-6/1e-5 parity. If one-path tiled reduction cannot meet pinned parity, STOP and return to Architect—no hidden shape variant.

3. **Load-bearing derivative/score/frozen/opacity tests**
   - Exact `u1==+limit` fixture must have nonzero u3/upstream/dgate sensitivity and fail if equality mask is wrong.
   - Add exact `u3==-limit` and `u3==+limit` sensitivity fixtures.
   - Score test must build actual scores, duplicate-collapsed assignment plan, expert outputs, upstream g, old Q-form derivative, and simplified identity independently; assert support and 1e-5 parity.
   - Outer transform test must exercise `routed_fp4` under `mx.grad`/`mx.value_and_grad`, prove packed/scales are captured constants/no cotangent, and prove x/score cotangents finite/non-zero.
   - DOT/export test must target routed custom function under outer transform and assert opaque `CustomKernel` nodes/no dequant arithmetic.

4. **Expert-id guard**
   - Validate every host-materialized route id before indexing: negative or `>=n_experts` raises deterministic `ValueError` before kernel launch.
   - Add `-1`, upper-bound, valid-boundary, duplicate tests.

5. **Tracked memory proof expansion**
   - File-backed tracked helper must run `value_and_grad(x,scores)`, assert x and score gradients finite/non-zero, and report operation peak.
   - Real-dimension case must directly exercise finite `y_e/dx_e/a_e` shapes and whole routed x/score gradient peak `<2GiB`.
   - Keep E=2/4/8 spread gate and expert-boundary telemetry.

6. **Tracked performance harness**
   - Move/encode exact R=1/8/32/96 p50/p95 benchmark into a tracked helper under `tests/helpers/`.
   - Stage it and provide reproducible command/extrapolation.
   - Do not cite untracked `agent-output/...bench*.py` as verdict evidence.

7. **Baseline and chain**
   - Use only `git ls-files 'tests/test*.py'` for full baseline.
   - Track every verdict helper/test.
   - Preserve complete staged 13.3b-5b+5f chain, protected hashes, ADR0024, production isolation.

## Verification
Run RED regressions before each correction where practical, then focused tests, tracked helpers, tracked-only full suite, make, diff/tracking/hash checks. Update `coder-13-3b-5f-notes.md` with exact evidence and remove stale claims.

No commit. On GREEN marker+success JSON; on parity/architecture STOP write `coder-13-3b-5f-stop.md`, no success marker, error JSON.
