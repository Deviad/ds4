# Story 13.3b-5h — Architect classification after diagnostic double-GREEN

Use `openai-codex/gpt-5.6-sol` high. No implementation, real assets, smoke, or training.

Read:
- `architecture-13-3b-5g-first-backward-oom.md` classification gates
- r2 contract, final report
- Reviewer r4 PASS and Tester r4 GREEN
- corrected helper/tests/fixture.

Authoritative corrected peaks:
- Probe A checkpoint calls 43; parity exact.
- Probe B D43 checkpoint-on: 152,425,220 B.
- Probe C D43/E256 routed one-graph: 5,491,948 B; sequential 1,483,376 B; nodes 66,048.
- Probe D cr4: full 1,918,140,864 B; no_attention 148,773,608 B; no_routed 113,283,804 B; no_shared 1,919,566,760 B; routed_only 150,040,984 B.
- cr0/cr128 full roughly 152–153 MB.

Required adjudication:
1. Apply previously pinned classification gates exactly.
2. Determine whether culprit is attention compression-ratio-4 path, routed/attention composition, shared path, checkpoint integration, command-buffer interaction, or inseparable multi-component behavior.
3. Explain no_routed and no_attention both collapsing peak while each standalone path remains bounded; do not infer single-component causality without interaction controls.
4. Decide one next architecture: minimal additional synthetic interaction ablation, attention checkpoint redesign, custom primitive lifetime redesign, layer-serial backward research, or STOP Path A.
5. Pin acceptance, STOP conditions, production boundaries, and whether BA/Coder work is authorized. No real smoke may be authorized from synthetic evidence alone.
6. Update durable docs/ADR only if decision changes architecture.

Write `agent-output/cmux-13-3b/architecture-13-3b-5h-classification.md`. Marker only success; unwrapped JSON.