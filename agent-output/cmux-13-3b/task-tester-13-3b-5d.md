# Story 13.3b-5d — Test Manager gate

Independently validate sparse routed-token FP4 backward against requirements, architecture, and Coder notes.

Use current user-selected `openai-codex/gpt-5.4-mini` high. Do not edit code/docs/tests.

Required checks without loading model shards:
- new sparse routed test suite;
- focused FP4/MoE/remap/LoRA trainer-contract suites;
- tracked full non-live regression and exact counts;
- separate-process dense-vs-sparse bounded-growth probes;
- real-dimension single-expert no-shard forward+VJP delta `<2 GiB`, finite values, no graph accumulation;
- formula contract values;
- eager/compile gate and no import side effect;
- duplicate hash routes, ties, empty expert, exact R_e instrumentation;
- input and gate/score gradient parity tolerances;
- `git ls-files` every verdict-participating test; untracked means NEEDS-INFO, never GREEN;
- FROZEN SHA exactly `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b`;
- `git diff --check` and scope guard;
- no active heavy model process and no model shard load.

Write exactly `agent-output/cmux-13-3b/test-report-13-3b-5d.md` with commands, counts, memory numbers, tracking proof, and GREEN/RED/NEEDS-INFO. Create `.cmux-status/test-manager.done` only on GREEN. End with one unwrapped JSON line.