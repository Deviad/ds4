# Story 13.3b-5g — BA re-pin: multi-layer synthetic peak diagnostics

Use `openai-codex/gpt-5.6-sol` high.

Read `architecture-13-3b-5g-first-backward-oom.md` and canonical backlog/docs. Update `docs/backlog.md` with a trackable Story 13.3b-5g user story in exact WHO/WHAT/WHY format and acceptance/STOP gates matching Architect.

Required scope:
- diagnostic-only tracked helpers/tests/report;
- Probe A exact 43-layer checkpoint coverage + checkpoint on/off parity 1e-5;
- Probe B D={1,8,16,43};
- Probe C E={2,8,32,128,256}, T=64,H=64,I=32,K=min(6,E), node pressure + matched sequential control;
- Probe D full/no_routed/no_shared/no_attention/routed_only, compression ratios 0/4/128;
- fresh subprocess per cell, <=180s, memory limit 8GB, no concurrency;
- complete JSON telemetry schema and explicit captured failures;
- no real model/shard/dataset/config, no production or primitive edits;
- Reviewer PASS + Tester GREEN; Architect re-entry required afterward.

Carry all STOP gates exactly. Explicitly forbid second smoke, shorter fallback, real assets, and redesign authorization.

Write `agent-output/cmux-13-3b/requirements-13-3b-5g-synthetic-peak.md`. Marker only success; unwrapped JSON.