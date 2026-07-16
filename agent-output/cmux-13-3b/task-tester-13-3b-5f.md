# Story 13.3b-5f — Test Manager gate

Independently validate the opaque packed-FP4 Metal/MLX primitive against requirements, architecture, ADR 0028, and Coder notes.

Use `openai-codex/gpt-5.4-mini` high. Do not edit files or run full model/shards.

Run and record:
- new primitive tests and strengthened sparse tests;
- independent forward/dx/a and whole routed score/gate parity, clamp boundaries, duplicate/empty/tie/shared semantics;
- opacity/no-dense/frozen-cotangent/platform-fail-closed guards;
- fresh-process E=2/4/8 fixed-assignment operation peaks and spread <=64MiB;
- H=4096/I=2048 no-shard operation peak <2GiB and R=1/8/32/96 timings;
- formula 1504MiB envelope;
- focused FP4/MoE/remap/LoRA gradients;
- full tracked non-live suite;
- `make`, `git diff --check`, `git ls-files` every verdict test;
- protected backend/root Metal/FROZEN/ADR0024 hashes and staged chain;
- no lingering process/no shard access.

Write exactly `agent-output/cmux-13-3b/test-report-13-3b-5f.md` with GREEN/RED/NEEDS-INFO. Create `.cmux-status/test-manager.done` only on GREEN. End with one unwrapped JSON line.