# Story 13.3b-5d — Coder review-fix slice

## Goal
Address every blocker in `agent-output/cmux-13-3b/review-13-3b-5d.md`. No full model/shard run.

## Model
Use user-authorized `openai-codex/gpt-5.5` high.

## Read first
- Reviewer report in full
- 13.3b-5d requirements/architecture/Coder notes
- current implementation/tests and intended 13.3b-5b chain.

## Blocker 1 — eliminate cross-expert VJP graph accumulation
Write discriminating RED tests first using fixed K and all experts non-empty.

Refactor custom VJP so no `_y_cache`/`_f_cache` or equivalent retains expert graphs across experts:
1. Q pass: for each expert, gather/recompute `y_e`, update Q, `mx.eval(Q)` before next expert, then release locals.
2. Gradient pass: recompute only current expert; derive score cotangent and direct input VJP; update `dx`/`dscores`; `mx.eval(dx, dscores)` before next expert; release locals.
3. Keep detached indices, exact unique-route semantics, denominator, gate gradients, and first-order contract.

Do not weaken per-expert barrier semantics. No FROZEN `deepseek_v4.py` body edit beyond existing sanctioned stop-gradient line.

Required memory proof in fresh separate processes, fixed K=2, T=1024,H=1024,I=512, all experts non-empty, E=2/4/8:
- no retained/execution graph growth proportional to E;
- record exact active and peak deltas;
- STOP if growth still demonstrates all-expert accumulation.

## Blocker 2 — legitimate tracked tests
Correct tests so they prove:
- nonuniform tokens and distinct packed FP4 bytes/scales per expert;
- sparse gate gradient compared to dense ordinary autodiff and numerical reference; assert dense gradient non-zero where expected;
- clamp-near sparse-vs-dense input and gate/score gradient parity at `atol=rtol=1e-5`;
- non-zero shared expert output detects omission, duplication, and wrong-input execution;
- implementation boundary passes stopped indices to host helper (use safe instrumentation/source/AST contract, not a tautological test-local stop only);
- duplicate route slots exercise implementation and assert expected summed cotangent;
- dense and sparse memory cases run in separate fresh processes with independent peak reset;
- real-dimension no-shard probe reports active delta and operation peak, finite output/cotangents, and at least two non-empty experts or an additional cross-expert probe that proves release;
- formula/budget claims accurately distinguish active retained delta from transient operation peak.

Do not lower tolerances or rename weak assertions as proof.

## Blocker 3 — complete chain of custody
Stage the complete intended 13.3b-5b predecessor chain plus 5d:
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` sanctioned one-line stop-gradient fix;
- `python-envs/mlx/src/ds4_ft_mlx/real_forward_intermediate_dump.py` pin;
- `tests/test_deepseek_v4_real_config_reference_forward.py` pin;
- `tests/test_numpy_real_forward_reference_composition.py` pin;
- `docs/adr/0026-csa-hca-indexer-mlx-real-port.md` amendment;
- `scripts/apply_passthrough_chat_template.py`;
- all existing 5d code/test/ADR/backlog/technical-spec files.

Verify old hash `96c39168c78e5fd9` has zero tracked hits and FROZEN SHA remains exactly `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b`.

Every verdict-participating test must be tracked with `git ls-files`. Any untracked baseline test is STOP. No commit.

## Verification
- RED evidence before production correction.
- New sparse suite.
- focused FP4/MoE/remap/LoRA suites.
- fixed-K E=2/4/8 memory probes in separate processes.
- no-shard real-dimension probe.
- full tracked non-live regression.
- `git diff --check`, staged scope, tracking, SHA/hash checks.

Update `coder-13-3b-5d-notes.md` with exact numbers/counts and staged list. Create `.cmux-status/coder.done` only on success. On unresolved memory accumulation write `coder-13-3b-5d-stop.md` and error JSON. End with one unwrapped JSON line.
