# Story 13.3b-5 — MILESTONE: smoke-train 43-layer real (thin re-pin)

**Status:** UNBLOCKED — BA re-pin r2 2026-06-27. Q1 prerequisite (on-disk `model-4bit/`)
RESOLVED by Story 13.3b-4b (`c1d3bf0`). No STOP. GO for Coder r0.

## User story
**As a** training engineer (WHO), **I want** `mlx_lm.lora --train --iters 20` to run against
the remapped+quantized real 43-layer DeepSeek V4 Flash ckpt (nn port + FP4 experts) with LoRA on
attention q_a/q_b/kv (WHAT), **so that** the first real gradient flow through the DeepSeek V4 nn port
is proven end-to-end (WHY).

## §0 — Scope (CONFIRMED)
- smoke-train = `mlx_lm.lora --config lora-config.json --model model-4bit --train` on the **real
  43-layer DeepSeek V4** with nn port + remapped+quantized `model-4bit/`.
- **FIRST ACTUAL `--train` run on the real model. MILESTONE slice.**
- Orchestrator command: `finetune_ds4.py smoke-train` (`scripts/finetune_ds4.py:949`).

## §1 — Prerequisite (DONE, verified this run)
`/Volumes/Data NVME/mlx-ft/ds4/model-4bit/` EXISTS — 149G, 33 shards.
`config.json`: `model_type=deepseek_v4_nn`, `quantization={group_size:64, bits:4, mode:affine}`,
`compress_ratios=[0,0,4,128,4,128,…]`. FP4 experts preserved + attn 4-bit. Reload finite
(13.3b-4b double-GREEN: `coder-13-3b-4b-notes.md`, `test-manager-13-3b-4b-report.md`,
`review-13-3b-4b.md`).

## §2 — Orchestrator path resolution (CONFIRMED, no code change)
`command_catalog` (`finetune_ds4.py:910-949`):
- `mlx_work = path_arg(args.mlx_work)` → env `MLX_WORK=/Volumes/Data NVME/mlx-ft/ds4` (`:462`).
- `--model {mlx_work / 'model-4bit'}` → `/Volumes/Data NVME/mlx-ft/ds4/model-4bit` ✓ EXISTS.
- `lora_config = mlx_work / "lora-config.json"` (`:929`) ✓ EXISTS.
- `split_dir = dataset_root / args.split_dir` (`:916`); `DATASET_ROOT=/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge`
  (`:461`), `DEFAULT_SPLIT_DIR="mlx-4096"` (`:37`) → `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096` ✓ EXISTS.
- `--adapter-path {mlx_work / 'adapters-smoke'}`, `--iters 20 --batch-size 1 --learning-rate 1e-5
  --max-seq-length 4096 --mask-prompt --grad-checkpoint`.

## §3 — AC (backward GREEN LOCKED 13.3a-3 BA Q4 + 13.3b-5 additions)
Run is GREEN iff ALL hold:
- (a) loss finite (no NaN/Inf).
- (b) grad tree non-empty.
- (c) every LoRA leaf grad finite.
- (d) ≥1 LoRA grad non-zero.
- (e) iter 20 completes — no OOM, no crash.
- (f) adapter saves to disk: `adapters-smoke/adapters.safetensors`.

## §4 — Dataset wiring (CONFIRMED)
`split_dir` resolves to `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096/`:
`train.jsonl` (15170), `valid.jsonl` (819), `test.jsonl` (824) — all present.
Dispatch gate `validate_dataset(args)` runs for `smoke-train` (`finetune_ds4.py:4501`). The
`model-4bit` existence gate (`:4503-4504`) and `validate_mlx_lora_target_gate(mlx_work)` (`:4498`)
also fire for `smoke-train`. All pass.

## §5 — lora-config + gate marker (CONFIRMED, no Coder pre-step)
`/Volumes/Data NVME/mlx-ft/ds4/lora-config.json`:
`rank=8`, `keys=[self_attn.q_a_proj, self_attn.q_b_proj, self_attn.kv_proj]`, `scale=20.0`,
`dropout=0.0`. Keys match nn port leaf names (`deepseek_v4_nn.py:237/239/240` under
`self_attn=AttentionNN:485`) exactly.
`.mlx-lora-targets-ok` marker PRESENT and **FRESH**: marker `config_sha256` =
`596b17548f87e335cbea13fac65888843a8f77113b08fc491410d3262c6508d0` = current
`shasum -a 256 lora-config.json`. `validate_mlx_lora_target_gate` will pass. **No re-run of
`mlx-lora-targets-check` required.**

## §6 — STOP-RULE for Coder r0 (genuine correctness, NOT test fix)
If smoke-train exposes a real bug in the nn port or remap on real 43-layer bytes during `--train`:
- forward NaN on a real cr=4/cr=128 layer that the 13.3b-4b reload-forward (seq_len=8) may have missed,
- backward dies on the real FP4 expert path,
- strict-load fails on the model-4bit nn keys,
- loss explodes,
→ Coder **STOPs**: record failing layer + iters + error, escalate to Architect/Coder. Genuine
correctness issue — do NOT patch tests or relax AC.

## §7 — Q5 RAM (deferred to run)
`--max-seq-length 4096` on 43 layers (hidden=4096, head_dim=512, n_routed_experts=256) on 512GB
M3 Ultra. If OOM: fall back to `smoke-train-2048` (`:950`, identical but `--max-seq-length 2048`,
`adapters-smoke-2048/`). If BOTH OOM → escalate Architect for further reduction.

## §8 — Pre-step (Coder)
1. `cd /Users/spotted/projects/ds4-finetuning`
2. `source python-envs/mlx/.venv/bin/activate` (unset `SSLKEYLOGFILE` first).
3. Run via orchestrator: `python3 scripts/finetune_ds4.py run-command smoke-train --execute --yes`
   (carries all gates). 4096 first; on OOM only, `smoke-train-2048`.
4. Verify AC (a)-(f). On any STOP-rule trigger → escalate, do not relax.

## Deliverables (Coder/Reviewer/Test-Manager, next)
- `adapters-smoke/adapters.safetensors` (or `adapters-smoke-2048/`).
- Coder notes: loss trace (iter 0…20), grad-tree summary, OOM/seq-len used.
- `review.md` + `test-report.md` double-GREEN against AC §3.
