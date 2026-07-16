# Story 13.3b-5c — Architect r2: memory-safe first backward + 2048 NaN diagnosis

## Goal
Design the minimal, evidence-based next slice that gets the real 43-layer DeepSeek V4 LoRA smoke through its first backward/optimizer step and iteration 20 on the 512GB M3 Ultra, without corrupting training semantics.

## User model/budget override
Use the model already running in this Architect pane: `openai-codex/gpt-5.6-sol` at high thinking. The user selected current pane models because other provider budgets are exhausted. Do not switch models.

## Trigger evidence
Read:
1. `agent-output/cmux-13-3b/coder-13-3b-5b-stop.md`
2. `agent-output/cmux-13-3b/coder-13-3b-5b-notes.md`
3. `agent-output/cmux-13-3b/coder-13-3b-5b-smoke-train.log`
4. Any archived/resource-contention and 2048 log/RC files in the same directory
5. `agent-output/cmux-13-3b/architecture-13-3b-5b-backward-safe-blockbias.md`
6. `agent-output/cmux-13-3b/requirements-13-3b-5.md`

Observed under verified clean resources:
- 4096 validation: finite `19.116`, 25 batches, 2245.586s; first training backward OOM, exit 134.
- 2048 validation: `NaN`, 25 batches, 1748.508s; first training backward OOM, exit 134.
- Batch size already 1; `--grad-checkpoint` already enabled.
- No adapter safetensors produced; no training loss; scatter-VJP fix not yet proven in full model backward.
- Focused forward/hash tests 50 passed / 2 skipped; direct block-bias VJP probe finite and non-zero.

## Required investigation — read-only/light probes only
Do NOT launch the full model.

1. Inspect the installed MLX-LM 0.31.3 trainer/lora/dataset implementation under:
   `/Volumes/Data NVME/mlx-ft/ds4/.venv/lib/python3.14/site-packages/mlx_lm/`
   Determine exact validation scheduling, cache clearing, gradient checkpoint placement, loss denominator/masking behavior, and available CLI/config flags.
2. Diagnose the 2048 NaN with a tokenizer/dataset-only probe:
   - For validation records, reproduce tokenization and prompt-mask offsets at max lengths 4096/3072/2048/1536/1024.
   - Count examples/batches with zero unmasked completion tokens after truncation.
   - Determine whether NaN is division by zero from all-masked tokens, not model numeric drift.
3. Diagnose memory shape before proposing sequence reduction:
   - Does initial 25-batch validation leave MLX Metal cache/compiled graphs resident before first backward?
   - Can `--val-batches 0/1`, validation scheduling, or an explicit safe cache clear avoid the first-step peak without changing training semantics?
   - Is there a supported `MLX_METAL_CACHE_LIMIT`/`mx.set_cache_limit` path and what is its correctness implication?
   - Confirm whether smaller max length alone can help, given 2048 still OOM and causes masked-token NaN.
4. Inspect custom FP4 expert/dequant and checkpoint boundaries for transient dense allocations retained through backward. Identify whether the current layer checkpointing actually bounds these allocations, without proposing speculative broad rewrites.
5. Rank options by minimality, semantic safety, memory impact, and proof required. Include exact command/config changes and rollback/STOP conditions.

## Decision requirements
Choose ONE next implementation/run slice, not a menu without a recommendation.

Explicitly adjudicate at least:
- A: keep 4096, reduce/disable initial validation batches and clear MLX cache before training;
- B: use an intermediate sequence length only after excluding zero-target masking;
- C: dataset preprocessing/filtering for zero-target examples;
- D: finer checkpoint/cache boundary or stop-gradient treatment for frozen dense dequant intermediates;
- E: any supported MLX memory-limit/environment setting.

Do not recommend 1024 blindly: shorter truncation may increase zero-target batches and preserve the same model-weight/backward peak.

## Boundaries
- Architecture/design/probes only. Do not edit production code, tests, dataset, model shards, or task configs.
- Do not launch a full model process.
- Preserve the real training objective: completion-only loss with prompt masking.
- Preserve all Metal/default/SSD/CUDA/distributed paths; any production fix must be narrow to MLX fine-tuning.
- No permanent semantic variants behind flags.
- Any proposed FROZEN-body edit requires explicit new ADR sanction and must be proven necessary.

## Deliverable
Write `agent-output/cmux-13-3b/architecture-13-3b-5c-memory-safe-backward.md` containing:
- root-cause findings for 4096 OOM and 2048 NaN, with source lines/probe evidence;
- chosen option and rejected alternatives;
- exact next command/code edit sites;
- resource budget/preflight;
- TDD/acceptance criteria, including finite non-empty completion loss, first backward, iter 20, gradients, adapter save;
- STOP conditions and fallback;
- whether BA requirements need a re-pin.

Create `.cmux-status/architect.done` only on successful completion. End with terminal JSON in this pane only: `{"status":"ok","role":"Architect"}` or error JSON.
