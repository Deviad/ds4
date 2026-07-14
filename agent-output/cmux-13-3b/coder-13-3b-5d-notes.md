# Coder 13.3b-5d — review-fix notes

Current outcome: STOP-ESCALATE. See `agent-output/cmux-13-3b/coder-13-3b-5d-stop.md` for full evidence.

Summary:
- RED first reproduced Review blocker: `_y_cache` / `_f_cache` source-contract test failed before production correction.
- Removed all-expert VJP caches and recomputed one expert at a time in Q and gradient passes.
- Strengthened sparse tests for nonuniform/distinct FP4 fixtures, dense/numerical gate gradients, clamp parity, shared-expert nonzero detection, stopped-index boundary, duplicate-route cotangent, separate-process memory cases, and two-nonempty-expert real-dim probe.
- Sparse suite: `21 passed, 1 warning in 5.80s`.
- Focused FP4/MoE/remap/LoRA suite: `117 passed, 8 skipped, 1 warning, 16 subtests passed in 26.90s`.
- Full tracked non-live regression: `364 passed, 29 skipped, 1 warning, 77 subtests passed in 25.28s`.
- STOP reason: fixed `K=2`, `T=1024`, `H=1024`, `I=512`, all-experts-nonempty fresh-process probe still shows peak growth with E: `584,225,254 -> 779,041,700 -> 1,179,289,384` bytes for `E=2/4/8` while active delta stays flat around `60.3 MiB`.
- No full model/shard run. No commit. No `.cmux-status/coder.done` success marker.
