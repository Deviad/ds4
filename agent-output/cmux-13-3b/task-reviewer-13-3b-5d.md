# Story 13.3b-5d — Reviewer gate

Independently review sparse routed-token FP4 backward implementation against:
- `requirements-13-3b-5d-routed-fp4.md`
- `architecture-13-3b-5d-routed-fp4-backward.md`
- `coder-13-3b-5d-notes.md`
- historical 13.3b-5b/5c STOP evidence.

Use current user-selected `openai-codex/gpt-5.6-sol` high. Do not edit code/docs/tests.

Inspect complete intended working-tree/index scope, including pre-existing 13.3b-5b stop-gradient/hash-pin/template changes and new 13.3b-5d files. Required review:
- exact routed forward semantics, duplicate collapse, stable ties, correction-bias selection-only, shared expert once;
- custom first-order VJP derivation for x and scores, gate-gradient propagation, clamp/SwiGLU derivative reuse through `mx.vjp(forward_one)`;
- detached integer route indices and no unsupported index VJP;
- per-expert graph release and no full-sequence/all-expert activation retention;
- host routing only under wrapper-local eager mode; no import global side effect;
- independent test legitimacy, not tautological implementation-copy;
- measured 201,326,592-byte no-shard real-dimension delta and formula contract;
- tracked-test reproducibility via `git ls-files` for every cited test;
- FROZEN `deepseek_v4.py` SHA exactly `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b` and no unauthorized body drift beyond sanctioned 13.3b-5b fix;
- ADR 0025, backlog, technical spec, ADR 0026, hash pins, passthrough template, scope;
- no model shards/full smoke in this gate.

Run focused probes/tests as useful. Write exactly `agent-output/cmux-13-3b/review-13-3b-5d.md` with findings and PASS/FAIL/NEEDS-INFO. Create `.cmux-status/reviewer.done` only on PASS. End with one unwrapped JSON line.