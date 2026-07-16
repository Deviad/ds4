# Story 13.3b-5i — Architect final interaction classification

Use `openai-codex/gpt-5.6-sol` high. Diagnostic adjudication only. No code edits, real assets, smoke, training, production redesign implementation, or commit.

Read in full:
- `architecture-13-3b-5h-classification.md`
- BA 5h requirements
- `interaction-ablation-report.json`
- Coder 5h r2 notes/logs
- `review-13-3b-5h-r2.md` PASS
- `test-report-13-3b-5h-r2.md` GREEN

Classify the corrected 25-row matrix. Compute and tabulate baseline-subtracted backward deltas and scaling for every family, especially D43. Apply thresholds exactly:
- collapsed: `<=227,479,736 B`
- persistent: `>=941,632,822 B`
- between: ambiguous

Adjudicate:
1. whether checkpointed composition itself is causal;
2. attention-forward/routed-VJP leg;
3. routed-forward/attention-VJP leg;
4. whether materializing attention→routed boundary collapses pressure without semantic drift;
5. whether sequential layer evaluation identifies graph-lifetime accumulation;
6. relation to prior cr4 full/no-attention/no-routed/no-shared evidence and real 4096 first-backward OOM.

Select exactly one:
- authorize the smallest evidence-supported production design slice, with invariants/acceptance criteria and affected boundaries; or
- STOP Path A with explicit remaining uncertainty and no further diagnostic/smoke.

Do not authorize a real smoke. Any future smoke remains separate explicit operator authorization after reviewed implementation.

Write `agent-output/cmux-13-3b/architecture-13-3b-5i-final-interaction-classification.md`. Marker only complete; unwrapped JSON.