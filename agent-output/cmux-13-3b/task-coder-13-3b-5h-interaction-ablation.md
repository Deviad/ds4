# Story 13.3b-5h — Coder 25-row interaction ablation

Use `openai-codex/gpt-5.5` high. TDD RED→GREEN. No production/vendor/primitive/Metal edits, real assets, smoke, training, redesign, classification, or commit.

Read Architect 5h and BA 5h in full. Reuse corrected 5g helper/test/fixture discipline.

Implement tracked diagnostic artifacts, preferably:
- extend `tests/helpers/routed_fp4_multilayer_peak_probe.py` without changing existing 5g semantics;
- add `tests/test_deepseek_v4_nn_interaction_ablation.py`;
- generate `agent-output/cmux-13-3b/interaction-ablation-report.json`.

Exact 25 fresh serial rows and order:
1. composed checkpoint-on D=1,2,4,8,16,43;
2. composed checkpoint-off D=1,4,8;
3. attention-forward+routed-VJP D=1,8,16,43;
4. routed-forward+attention-VJP D=1,8,16,43;
5. same-semantics materialized boundary D=1,8,16,43;
6. per-layer sequential D=1,8,16,43.

All cr4/no_shared/T64/H128/I64/E8/K2/hc4/rank8 last min(D,16), fixture prefixes, raw gradients. Implement exact control semantics and parity. Record complete telemetry/identity/outcomes. 8GB, <=180s, no concurrency. Classification thresholds metadata: collapsed <=227,479,736 B; persistent >=941,632,822 B; middle ambiguous. Do not classify automatically.

Behavioral tests must independently prove forward equality, VJP leg removal semantics, materialized loss/LoRA/input-gradient parity 1e-5, exact rows/order, child seriality, measured graph/peaks/nodes, raw finiteness, fixture/tracking/protected hashes. No shared implementation oracle.

STOP on any missing/failed/nonfinite/parity cell. Run focused + prior 5g tests, both compile orders, tracked-only baseline in canonical project venv, make/diffs/hashes/tracking. Write `agent-output/cmux-13-3b/coder-13-3b-5h-notes.md`. Marker only complete GREEN; unwrapped JSON.