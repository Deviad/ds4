# Story 13.3b-5d — Coder follow-up: complete pinned AC evidence/docs

## Goal
Complete the missing BA/Architect acceptance criteria before Reviewer/Test Manager. Core sparse implementation is already GREEN; do not broaden or rewrite it unless a new RED test exposes a defect.

## Model/budget
Remain on OpenCode Go (`kimi-k2.6` or current OpenCode Go model) high. Do not switch to Codex, Anthropic, or Neuralwatt.

## Read
- `agent-output/cmux-13-3b/coder-13-3b-5d-notes.md`
- `agent-output/cmux-13-3b/requirements-13-3b-5d-routed-fp4.md`
- `agent-output/cmux-13-3b/architecture-13-3b-5d-routed-fp4-backward.md`
- Current staged diff and new test.

## Missing gates to complete

### 1. Separate-process bounded-growth evidence
Add tracked tests/probe support in the existing new test file or a new tracked helper/test. Run dense and sparse forward+VJP in separate processes after cache clear/peak reset. Record exact peak bytes/deltas.

Required proof:
- dense growth increases with `E*T` on a small deterministic fixture;
- sparse growth is governed by maximum `R_e` plus accumulators, not all `E*T`;
- no cross-expert graph accumulation.

### 2. No-shard real-dimension probe
Run an approved no-shard synthetic probe with real single-expert dimensions `H=4096`, `I=2048`, FP4 packed payload/scales, routed tokens, forward+VJP. Do not load model shards or instantiate 256 real expert payloads.

Required: routed-operation active-memory delta `<2 GiB`, finite output/cotangents, and exact measured number in notes. STOP if >=2 GiB.

### 3. Formula contract
Add/execute checks pinning all values:
- route metadata: 96 KiB;
- one-expert dense dequant output: 96 MiB;
- old retained dequant floor: 24 GiB;
- old full-sequence activation floor: 40 GiB;
- selected routed-op budget: 1.25 GiB;
- whole-process design ceiling: 300 GiB;
- 400GB scheduling limit leaves ~72.5 GiB margin.

### 4. Compile/eager gate
Add/execute evidence that:
- import/model construction does not change global compile mode;
- compiled host route materialization fails with the expected eval-during-transform behavior;
- calling `mx.disable_compile()` in a wrapper before the trainer-style value+gradient call passes;
- no module-import global side effect was introduced.

Do not patch site-packages or add compile-disable logic to module import.

### 5. Canonical docs
Complete and stage:
- amend `docs/adr/0025-real-trainable-nn-module-deepseek-v4-port.md` with training-only eager sparse custom first-order VJP, detached route metadata, per-expert barriers, and wrapper-local compile disable;
- update `docs/technical-spec.md` with exact later 4096 wrapper (`mx.disable_compile`, `mx.set_memory_limit(400_000_000_000)`), one validation batch, per-step cache/active/peak telemetry, and 340GB STOP threshold;
- stage BA-owned `docs/backlog.md` re-pin without changing its semantics;
- keep ADR 0024 unchanged.

### 6. Regression and tracking
- Re-run the new suite and all focused FP4/MoE/remap/LoRA suites.
- Run full non-live regression suite if it does not load model shards; record exact pass/skip/fail counts.
- `git diff --check`.
- `git ls-files --` every verdict-participating test.
- New/modified test files must be staged.
- FROZEN `deepseek_v4.py` SHA must remain `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b`.

## Scope
No full model/shard run. No `deepseek_v4.py`, ADR 0024, dataset, model shard, site-package, inference, SSD, CUDA, distributed, or default Metal edit. No commit.

## Handoff
Update `agent-output/cmux-13-3b/coder-13-3b-5d-notes.md` with exact probe commands/numbers, formula values, compile evidence, docs, test counts, tracking proof, FROZEN SHA, and full staged-file list.

On complete GREEN create `.cmux-status/coder.done` and emit `{"status":"ok","role":"Coder"}`. On any STOP condition write `coder-13-3b-5d-stop.md`, omit success marker, emit error JSON.
