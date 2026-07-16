# Story 13.3b-5 — BA re-pin (smoke-train on real 43-layer model, now UNBLOCKED)

## Context — Q1 blocker RESOLVED
First BA run (13.3b-5 `ba-13-3b-5-stop.md`) STOPPED on Q1: smoke-train needs on-disk `model-4bit/` but remap was in-memory only. **That blocker is now resolved by Story 13.3b-4b** (commit `c1d3bf0`):
- `mlx_lm.convert --model /Volumes/Data NVME/mlx-ft/ds4/hf-f8shim -q --mlx-path /Volumes/Data NVME/mlx-ft/ds4/model-4bit` SUCCEEDED (122s).
- `/Volumes/Data NVME/mlx-ft/ds4/model-4bit/` EXISTS: 149G, 33 shards, 2484 weights.
- `config.json`: `model_type=deepseek_v4_nn`, `quantization={group_size:64, bits:4, mode:affine}`.
- FP4 experts PRESERVED (43 `w1_weight U8 [256,2048,2048]` + 43 `w1_scale BF16`), attn 4-bit (`q_a_proj.weight U32` + `.scales`/`.biases`).
- Reload verified finite (`(1,8,129280)`, 0 NaN/Inf).
- `deepseek_v4_nn.Model.load_weights` override has idempotent `is_native` gate (native ckpt keys remap, nn keys bypass).

So this is a THIN re-pin. BA does NOT re-do recon. BA confirms scope + re-pins runnable AC against the now-on-disk `model-4bit/` + calls `finetune_ds4.py smoke-train`.

## Q2-Q4 already resolved (no blockers — carry forward from first BA run)
- **Q2 training data**: EXISTS at `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096/` (train=15170/valid=819/test=824). No synthesis needed.
- **Q3 orchestrator**: USE `finetune_ds4.py smoke-train` command now that model-4bit exists (was the blocked path).
- **Q4 lora keys**: `lora-config.json` keys (`self_attn.q_a_proj`/`q_b_proj`/`kv_proj`) match nn port `AttentionCompressorNN`/`AttentionIndexerNN` leaf names exactly.
- **Q5 RAM feasibility**: deferred to run (2048 fallback exists if `--max-seq-length 4096` OOMs at 43 layers real dims hidden=4096 head_dim=512 n_routed_experts=256 on 512GB M3 Ultra).

## BA jobs (thin)
1. **Confirm scope**: smoke-train = `mlx_lm.lora --train` run on real 43-layer DeepSeek V4 with the nn port + remapped+quantized `model-4bit/`. First ACTUAL `--train` run on the real model. MILESTONE slice.
2. **Confirm orchestrator path**: `finetune_ds4.py smoke-train` command (at `:949`). Verify it references `mlx_work/model-4bit` (which now exists as `/Volumes/Data NVME/mlx-ft/ds4/model-4bit/`).
3. **Re-pin AC** (backward AC GREEN definition LOCKED from 13.3a-3 BA Q4): (a) loss finite; (b) grad tree non-empty; (c) every LoRA leaf grad finite; (d) >=1 LoRA grad non-zero. PLUS for 13.3b-5: (e) iter 20 completes no OOM no crash; (f) adapter saves to disk.
4. **Dataset wiring**: confirm `finetune_ds4.py smoke-train` resolves `split_dir` to `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096/` (train.jsonl + valid.jsonl). If `validate_dataset` gate needs a specific path or arg, pin it.
5. **lora-config**: verify `lora-config.json` (`/Volumes/Data NVME/mlx-ft/ds4/lora-config.json`) rank=8 keys=`[self_attn.q_a_proj, self_attn.q_b_proj, self_attn.kv_proj]` scale=20.0 dropout=0.0 + the `.mlx-lora-targets-ok` marker is present (if marker absent, BA pins it as a Coder pre-step or verifies lora-targets.json).
6. **STOP-RULE for Coder r0**: if smoke-train exposes a real bug in the nn port or remap on real 43-layer bytes during `--train` (forward NaN on a real cr=4/cr=128 layer that the 13.3b-4b reload-forward seq_len=8 may have missed, OR backward dies on real FP4 expert path, OR strict-load fails on the model-4bit nn keys, OR loss explodes), Coder STOPs — records the failing layer+iters+error, escalates to Architect/Coder. Genuine correctness issue, NOT a test fix.

## Deliverables
1. `agent-output/cmux-13-3b/requirements-13-3b-5.md` — thin SPEC with re-pinned AC + dataset wiring + lora-config + STOP-rule.
2. `docs/backlog.md` — update Story 13.3b-5 row (milestone: first real train).
3. `.cmux-status/ba.done` marker.
4. In-pane JSON `{"status":"ok","role":"BA"}` in surface:81 ONLY.

## STOP-ESCALATE
Write `agent-output/cmux-13-3b/ba-13-3b-5-r2-stop.md` if:
- `finetune_ds4.py smoke-train` orchestrator path can't resolve `model-4bit` or `split_dir` without a code change BA can't authorize (escalate split — Coder scope).
- lora-config keys diverge from nn port leaf names needing FROZEN/nn-port edit (escalate split).
- Q5 RAM infeasible at seq-len 4096 AND 2048 (escalate Architect for further reduction).

## Pre-flight read
- `agent-output/cmux-13-3b/ba-13-3b-5-stop.md` (first BA run Q1 STOP + Q2-Q4 resolutions).
- `agent-output/cmux-13-3b/architecture-13-3b-5-model-4bit-path.md` (Architect Option B decision + §4 AC for the 13.3b-4b prerequisite — DONE).
- `agent-output/cmux-13-3b/coder-13-3b-4b-notes.md` (model-4bit produced, reload+forward finite).
- `agent-output/cmux-13-3b/test-manager-13-3b-4b-report.md` + `review-13-3b-4b.md` (double GREEN on model-4bit).
- `scripts/finetune_ds4.py:949` (smoke-train command) + `:4498-4505` (`validate_dataset` gate) + `:56-82` (command list).
- `/Volumes/Data NVME/mlx-ft/ds4/lora-config.json` + `lora-targets.json` + `.mlx-lora-targets-ok` marker presence.
- `/Volumes/Data NVME/mlx-ft/ds4/model-4bit/config.json` (verify it's there).
- `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096/` (train=15170/valid=819/test=824).
- ADR 0024 (FP4 dequant — live training-forward path), ADR 0025 (nn-port ownership), ADR 0026 (additive FROZEN), ADR 0027 (load-side remap).
- `AGENTS.md` backward AC GREEN definition (13.3a-3 BA Q4 LOCKED).

## Build
`source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first). Use `ctx_execute` for bounded probes (orchestrator path resolution, lora-config read, dataset presence). Caveman ultra default; byte-exact exempt. BEGIN NOW. Echo `{"status":"ok","role":"BA"}` in surface:81 only.
