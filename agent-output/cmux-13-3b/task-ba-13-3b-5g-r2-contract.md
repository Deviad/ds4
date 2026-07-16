# Story 13.3b-5g r2 — BA canonical diagnostic contract re-pin

Use `openai-codex/gpt-5.6-sol` high.

Read `architecture-13-3b-5g-r2-contract.md`, Reviewer RED, current 5g requirements, and Story 13.3b-5g backlog block.

Update canonical `docs/backlog.md` acceptance/STOP text only as required to pin Architect r2:
- retain exact 64-row ordering;
- trainer-equivalent quantize→freeze→rank8 LoRA q_a/q_b/kv on last min(D,16), exact 6L keys;
- one bound forward per loss and actual DOT node counts;
- measured active_after_graph boundaries;
- E=2 non-comparable low-E control; normalized expert axis E=8/32/128/256 with K=6 and 384 assignments;
- fixture-driven Probe D with schema/config/provenance/source hashes;
- strengthened behavioral tests and fresh-child/serial evidence;
- regenerated report fully supersedes old values.

No production redesign/classification/real assets/smoke. No other canonical doc edits unless strictly needed. Write `agent-output/cmux-13-3b/requirements-13-3b-5g-r2-contract.md`. Marker only success; unwrapped JSON.