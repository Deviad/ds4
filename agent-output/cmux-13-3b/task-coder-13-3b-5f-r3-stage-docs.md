# Story 13.3b-5f — Coder staging-only remediation

Reviewer r3 found one blocker only: current correct r3 amendments are unstaged in:
- `docs/adr/0028-opaque-packed-fp4-metal-training-primitive.md`
- `docs/backlog.md`
- `docs/technical-spec.md`

Use gpt-5.5 high.

Do not edit content or code/tests. Stage exactly these three current worktree files. Verify staged versions contain:
- combined forward bound `2e-6 + 1e-6*abs(ref)` and NRMSE;
- SIMD K1-K5 single-order contract;
- downstream proxy gates;
- R=96 performance gates;
- no-smoke policy.

Run `git diff --cached --check`, report staged hashes/status, preserve all existing intended staged files, no commit. Update Coder notes with staging-only closure. Marker only success; unwrapped JSON.