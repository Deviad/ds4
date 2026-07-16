# Story 13.3b-5g — Coder diagnostic-only multi-layer peak proof

Use user-authorized `openai-codex/gpt-5.5` high. TDD RED→GREEN. No commit.

## Read
- `architecture-13-3b-5g-first-backward-oom.md`
- `requirements-13-3b-5g-synthetic-peak.md`
- current docs/backlog/technical spec
- existing routed primitive tests/helpers and MLX checkpoint code.

## Scope
Add and track only diagnostic/test artifacts:
- `tests/helpers/routed_fp4_multilayer_peak_probe.py`
- `tests/test_deepseek_v4_nn_multilayer_peak.py`
- `agent-output/cmux-13-3b/multilayer-peak-report.json`

No production source, primitive, site-packages, real model/shard/dataset/config access, or second smoke.

## Required probes
A. Tiny 43-layer Model: exact 43 DecoderLayerNN and checkpointed_fn calls; checkpoint on/off finite forward/grad parity `atol=rtol=1e-5`; subprocess monkeypatch only. Count !=43 is STOP.

B. Full shape-reduced topology, fresh process each: D={1,2,4,8,16,43}, T=64,H=128,I=64,E=8,K=2,hc=4, LoRA q_a/q_b/kv, trainable min(D,16); checkpoint on all; off D={1,4,8}.

C. Routed-only chain: D={1,8,16,43}, E={2,8,32,128,256}, T=64,H=64,I=32,K=min(6,E); deterministic nonempty experts where feasible; hold assignments fixed; node counts; matched sequential eval/cache-clear control.

D. Component ablations: full/no_routed/no_shared/no_attention/routed_only; compression ratios 0/4/128; checked-in 43-entry shape-reduced fixture from canonical evidence only.

Every cell: fresh subprocess, serial, mx.disable_compile, 8GB limit, <=180s, complete telemetry fields from architecture. Capture subprocess OOM/failure; never silently omit/reduce a cell.

## Gates
- tracked helper/tests/report;
- complete JSON matrix;
- direct production hashes and diff checks;
- no source/primitive edits;
- classify nothing beyond measurements; Architect re-entry required.

STOP immediately on architecture STOP conditions. Write/update `agent-output/cmux-13-3b/coder-13-3b-5g-notes.md` with commands, RED/GREEN, matrix completeness, failures, hashes, tracking. Marker only successful complete diagnostic; unwrapped JSON.