# Story 13.3b-5g r3 — Coder compile-state isolation follow-up

Use `openai-codex/gpt-5.5` high. No production edits, real assets, acceptance weakening, or commit.

Authoritative reproduction:
- `test_compiled_host_routing_fails_as_documented` alone: PASS.
- diagnostic focused suite then compile test: FAIL.
- compile test then diagnostic suite: PASS.

The new diagnostic suite leaks global `mx.disable_compile()` state. Fix contamination in new helper/tests only; do not alter the baseline compile-behavior test or its expectation.

Requirements:
1. Keep every compile-disabled diagnostic operation inside fresh subprocesses.
2. Runtime behavioral tests must not call child setup/`mx.disable_compile()` in pytest parent.
3. Add load-bearing order/isolation proof: compile semantics unchanged before and after diagnostic focused tests, or equivalent subprocess boundary proof.
4. Rerun both explicit orders above; both GREEN.
5. Rerun focused 13 tests, complete regenerated 64-row report if bytes/semantics require it, tracked-only baseline, make/diff/hashes.
6. Stage logs and exact updated bytes; update `coder-13-3b-5g-r3-notes.md`.
7. Marker and success JSON only if full tracked baseline GREEN. STOP otherwise.
