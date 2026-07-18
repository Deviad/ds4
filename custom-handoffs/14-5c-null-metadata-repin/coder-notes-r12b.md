# Story 14.5c — Coder r12b exact wording correction

## Scope

Docs-only correction. No production-code, test, marker, execution, commit, push, delegation, or cmux changes.

## Change

- `docs/technical-spec.md`
  - Corrected the exact error text from `attempt-2 runtime target bindings not immutable` to `attempt-2 runtime target bindings are not immutable`.
- Staged `task-coder-r12b.md` and this coder note as handoff evidence.

## Verification

- Canonical staged technical spec contains new phrase exactly once and old phrase zero times.
- `git diff --check`: PASS.
- `git diff --cached --check`: PASS.
- No new `.cmux-status` marker was created; pre-existing staged tree and markers were left unchanged.
