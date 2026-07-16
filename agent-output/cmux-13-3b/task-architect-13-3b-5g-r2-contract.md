# Story 13.3b-5g r2 — Architect contract repair after Reviewer RED

Use `openai-codex/gpt-5.6-sol` high. No production edits, real assets, or smoke.

Read `review-13-3b-5g.md`, prior 5g architecture/requirements, helper/tests/report.

Adjudicate and pin exact remediation:
1. Probe B/D trainer-equivalent topology: freeze base; attach LoRA only to q_a/q_b/kv on last min(D,16); exact trainable-key assertions.
2. Bind each differentiated forward once; pin actual custom-kernel call/node measurement.
3. Define real `active_after_graph` measurement boundary for one-graph and sequential cases.
4. Resolve E=2 versus fixed assignment volume with K=min(6,E): designate non-comparable control or define a separate normalized axis; state exact matrix/metadata/comparison rules without silently dropping pinned cells.
5. Make 43-entry topology fixture load-bearing: schema/index/type/shape validation, config derivation, canonical source/hash/provenance.
6. Strengthen tests for exact rows/order/dimensions, child boundaries, serial evidence, LoRA keys, assignment rules, node counts, telemetry, replacements/gradient path, fixture consumption, exit/signal consistency, and Probe A key-set equality.

Decide whether existing 64-row matrix remains or expands. No classification or redesign/smoke authorization.

Write `agent-output/cmux-13-3b/architecture-13-3b-5g-r2-contract.md`; amend durable docs if required. Marker only success; unwrapped JSON.