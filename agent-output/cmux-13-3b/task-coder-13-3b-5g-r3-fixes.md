# Story 13.3b-5g r3 — Coder fixes for Reviewer r2 RED

Use `openai-codex/gpt-5.5` high. TDD RED→GREEN. No commit, production edits, real assets, classification, redesign, or smoke.

Read `review-13-3b-5g-r2.md`, Architect/BA r2, current helper/tests/fixture/report.

Required repairs:
1. Remove every input-gradient `nan_to_num`/normalization before DOT/eval/finiteness/evidence. Measure raw gradient only. Any raw NaN/Inf is a captured failed cell and prevents GREEN.
2. Remove `mx.stop_gradient(mlp_input)` from selected `routed_only` branch. Routed executes normally; only removed attention/shared branches use zero-valued identity-gradient replacements.
3. Add runtime behavioral tests, not source-substring/report-only checks:
   - raw unsanitized model-cell gradient finiteness;
   - routed-only forward and input-gradient parity against current direct routed branch;
   - runtime LoRA conversion keys/shapes;
   - full/masked forward and gradient semantics;
   - live DOT boundary/node measurement;
   - malformed child output and attempted outcome override handling.
4. Probe A records separate fresh-child PID, nonce, monotonic start/finish, parent-observed exit/signal/error; test uniqueness/interval validity.
5. Stage exact r3 worktree bytes for helper/test/fixture/report and notes; verify index==worktree hashes.
6. Expand direct protected hashes to ds4_cli.c, ds4_server.c, deepseek_v4_nn.py, all root metal/*.metal, plus prior protected files.
7. Regenerate all 64 rows from scratch. No row patch/reuse.
8. Focused tests, tracked-only baseline, make, diff checks; produce logs/evidence.

STOP if any raw non-finite cell remains, cell fails, timeout occurs, or semantic parity fails. Do not sanitize or weaken acceptance.

Write `agent-output/cmux-13-3b/coder-13-3b-5g-r3-notes.md`. Marker only full GREEN; unwrapped JSON.