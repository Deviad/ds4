# Story 13.3b-5f — Reviewer gate

Independently review the opaque packed-FP4 Metal/MLX primitive against 13.3b-5f requirements, architecture, ADR 0028, and Coder notes.

Use `openai-codex/gpt-5.6-sol` high. Do not edit files or run full model/shards.

Required review:
- all six Metal kernels: LUT/nibble order, BF16 scale blocks, tiling, padding, strides, FP32 reduction, clip/SwiGLU and exact boundary derivatives, w1/w3/w2 ordering;
- no hidden full dense matrices or O(H²I) work;
- wrapper source parsing/cache/platform/version/shape guards and package data;
- routed custom VJP exact dx/a and simplified score derivative, duplicate collapse, routing/shared expert semantics, no frozen cotangents/fallback;
- independent test legitimacy and parity tolerances;
- fresh-process E=2/4/8 peak-spread proof and real-dimension operation peak legitimacy—not final active only;
- timing/smoke feasibility;
- package-local source excluded from root production build/runtime;
- make/build, root Metal and protected backend hashes;
- full 13.3b-5b + 5f staged chain, tracked tests, old hash removal, FROZEN SHA;
- ADR 0024 unchanged; ADR 0025 supersession and ADR 0028 consistency.

Reproduce focused probes/tests as useful. Write exactly `agent-output/cmux-13-3b/review-13-3b-5f.md` with findings and PASS/FAIL/NEEDS-INFO. Create `.cmux-status/reviewer.done` only on PASS. End with one unwrapped JSON line.