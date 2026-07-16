# Story 13.3b-5h — BA pin 25-row interaction ablation

Use `openai-codex/gpt-5.6-sol` high.

Read `architecture-13-3b-5h-classification.md` and canonical backlog. Add/update trackable Story 13.3b-5h in `docs/backlog.md` with exact WHO/WHAT/WHY and acceptance/STOP gates.

Pin exact 25 rows and order:
- composed checkpoint-on D=1,2,4,8,16,43;
- composed checkpoint-off D=1,4,8;
- attention-forward+routed-VJP D=1,8,16,43;
- routed-forward+attention-VJP D=1,8,16,43;
- same-semantics materialized boundary D=1,8,16,43;
- per-layer sequential control D=1,8,16,43.

All cr4/no_shared/T64/H128/I64/E8/K2/hc4/rank8. Pin exact control semantics, telemetry, thresholds 227,479,736 B and 941,632,822 B, classification/re-entry gates, 8GB/180s/fresh serial/no-real-assets constraints, Reviewer+Tester and Architect re-entry. No production redesign/smoke.

Write `agent-output/cmux-13-3b/requirements-13-3b-5h-interaction-ablation.md`. Marker only success; unwrapped JSON.