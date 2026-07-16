# Story 13.3b-5 — BA (smoke-train on real 43-layer model)

## Slice context
- **Story**: 13.3b-5 — smoke-train on the real 43-layer DeepSeek V4 model. **MILESTONE** slice (first actual `mlx_lm.lora --train` run on the real model with the nn port + remapped ckpt).
- **Predecessor**: 13.3b-4 complete (HEAD `fecc337`): `scripts/remap_ds4_nn_weights.py` (465L NEW convert-side, key-remap + real payload load verified — AC2 r2 GREEN both gates).
- **All architectural foundations in place**:
  - nn port `deepseek_v4_nn.py` (AttentionNN guard lifted + §4.1 submodules `AttentionCompressorNN` + `AttentionIndexerNN`)
  - dispatch routing: cr=0 sliding / cr=4 CSA `_csa_attention_real_mlx` / cr=128 HCA `_attention_real_mlx`
  - FP4 dequant live (`_dequantize_fp4` at forward, ADR 0024)
  - plugin `ds4_ft_mlx.mlx_lm_plugin.install_deepseek_v4_plugin()` registers the nn port with `mlx_lm`

## BA role
**BA does NOT re-design the architecture.** BA confirms scope, resolves the open Qs below, and pins AC. Architecture docs (`docs/architecture.md`, ADRs) are the contract.

## Recon ALREADY DONE by supervisor (don't re-do)

1. **`finetune_ds4.py:949` smoke-train command** = `mlx_lm.lora --config lora-config --model mlx_work/model-4bit --train --data split_dir --adapter-path mlx_work/adapters-smoke --fine-tune-type lora --iters 20 --batch-size 1 --learning-rate 1e-5 --max-seq-length 4096 --mask-prompt --grad-checkpoint`
2. **remap script** `scripts/remap_ds4_nn_weights.py` has ONLY a `plan` subcommand (header-only) — NO `apply`/`convert` subcommand that writes a remapped ckpt dir to disk.
3. **`lora-config.json` READY** at `/Volumes/Data NVME/mlx-ft/ds4/lora-config.json`: `rank=8`, `keys=["self_attn.q_a_proj","self_attn.q_b_proj","self_attn.kv_proj"]`, `scale=20.0`, `dropout=0.0`. Plus `lora-targets.json` + `.mlx-lora-targets-ok` marker.
4. **NO `train.jsonl`/`valid.jsonl`/`splits` dir found anywhere** — `validate_dataset(args)` gate at `finetune_ds4.py:4501` needs data.
5. **`model-4bit` dir ABSENT** (session memory: "model-4bit stays absent — Story 13.3 owns creation" — 13.3b-5 IS that slice).

## Open questions BA MUST resolve

### Q1 — model-4bit resolution
Does `mlx_lm.lora --train` load the nn port via the plugin + remapped shimmed ckpt **IN-MEMORY** (no on-disk remapped dir needed), OR does smoke-train need an on-disk remapped ckpt dir?
- If **in-memory**: smoke-train uses a different invocation than the stock `mlx_lm.lora --model <dir>` (the plugin installs the nn port, and weights load via the remap at runtime).
- If **on-disk**: the remap script needs an `apply`/`convert` subcommand added → **Coder scope, NOT BA** → escalate a Coder sub-slice.
- Decide which. Probe the plugin (`ds4_ft_mlx.mlx_lm_plugin`) + the shimmed ckpt to determine if `mlx_lm.lora --model /Volumes/Data NVME/mlx-ft/ds4/hf-f8shim` (with plugin installed) loads correctly, OR if a remapped-dir step is required.

### Q2 — training data
`validate_dataset` gate needs a `split_dir` with `train.jsonl` + `valid.jsonl`. None currently exist. Does the user have a dataset, or do we **synthesize a tiny smoke dataset** (3-5 examples) for the `iters=20` smoke? For a smoke (not real training), a tiny synthesized dataset in the DeepSeek chat template (`BOS USER ... ASSISTANT_THINK ...`) is acceptable. Pin the exact data source.

### Q3 — orchestrator vs direct call
Use `finetune_ds4.py smoke-train` command (which references the absent `model-4bit`), OR a direct `mlx_lm.lora` call with the shimmed ckpt path + plugin install? The orchestrator has the command shape but its gate graph may block on `model-4bit` absence. Decide the cleanest path.

### Q4 — lora keys vs nn-port leaf names
`lora-config.json` keys are `self_attn.q_a_proj` / `self_attn.q_b_proj` / `self_attn.kv_proj`. Do these match the nn port `AttentionNN` submodule leaf names (`deepseek_v4_nn.py` `AttentionCompressorNN` / `AttentionIndexerNN`), OR do they need remapping to nn leaf names? Probe `model.parameters()` on the nn port to confirm the LoRA-eligible leaf names exactly. If they diverge and a FROZEN/nn-port edit is required → escalate (separate slice).

### Q5 — RAM + seq-length feasibility
`grad-checkpoint` + `max-seq-length 4096` — does this fit in 512GB M3 Ultra RAM at 43 real layers (hidden=4096, head_dim=512, n_routed_experts=256, num_experts_per_tok=6)? If 4096 is too aggressive for a smoke, reduce to 2048 or 1024. Pin the seq-length for the smoke.

## Pin AC (backward AC GREEN — from 13.3a-3 BA Q4 LOCKED)
- (a) loss finite (no NaN/Inf).
- (b) grad tree non-empty.
- (c) every LoRA leaf grad finite.
- (d) ≥1 LoRA grad non-zero.
- PLUS for smoke: (e) iter 20 completes — no OOM, no crash.
- (f) adapter saves to disk (`adapters-smoke/adapters.safetensors` + `adapter_config.json`).

## STOP-ESCALATE
Write `agent-output/cmux-13-3b/ba-13-3b-5-stop.md` if:
- **Q1**: on-disk remapped ckpt required → remap script needs `apply` subcommand = Coder scope, NOT BA → escalate a Coder sub-slice.
- **Q2**: no training data available AND cannot synthesize → escalate to user for a dataset.
- **Q4**: lora keys diverge from nn port leaf names in a way that needs a FROZEN/nn-port edit → escalate a separate slice.
- **Q5**: RAM infeasible at 4096 → escalate Architect for seq-length reduction (but BA can just pin a lower seq-length for the smoke — only STOP if even 1024 is infeasible).
- **Smoke exposes a real bug**: if the smoke-train run exposes a real bug in the nn port or remap on real 43-layer bytes (forward NaN on a real cr=4/cr=128 layer that synthetic parity tests missed, OR backward dies on a real FP4 expert path, OR strict-load fails on the full 43-layer real ckpt) → STOP, record the failing layer + error, escalate to Architect/Coder. This would be a genuine correctness issue, NOT a test fix.

## Tasks
1. Confirm scope: smoke-train only. Any new production code (e.g. remap `apply` subcommand, smoke-data synthesis helper, orchestrator tweak) is a SEPARATE Coder slice — BA does not write it.
2. Resolve Q1-Q5 above with concrete answers (probe the plugin + nn port as needed).
3. Pin AC (a-f above).
4. Verify backlog: update `docs/backlog.md` Story 13.3b-5 row.

## Deliverables
1. `agent-output/cmux-13-3b/requirements-13-3b-5.md` — SPEC with Q1-Q5 resolutions + AC + STOP-rules + scope.
2. `.cmux-status/ba.done` marker.
3. In-pane JSON `{"status":"ok","role":"BA"}` in surface:81 ONLY.
4. `docs/backlog.md` Story 13.3b-5 row updated.

## Pre-flight read
- `agent-output/cmux-13-3b/architecture.md` §5.1-§5.4 (convert rename table) + §6 if exists (smoke-train) + §0 (nn tree).
- `agent-output/cmux-13-3b/subslice-breakdown.md` 13.3b-5 row.
- `scripts/remap_ds4_nn_weights.py` (the `plan` subcommand — CHECK if `apply` is needed for Q1).
- `scripts/finetune_ds4.py:949` (smoke-train command) + `:4498-4505` (validate gates) + `:56-82` (command list).
- `deepseek_v4_nn.py` (AttentionNN / AttentionCompressorNN / AttentionIndexerNN leaf names — Q4).
- `/Volumes/Data NVME/mlx-ft/ds4/lora-config.json` + `lora-targets.json`.
- `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json` (43 layers real dims).
- `agent-output/cmux-13-3b/coder-13-3b-4-r2-notes.md` (remap r2 approach).
- ADR 0025 (nn file ownership) + ADR 0024 (FP4 dequant) + ADR 0026 (additive FROZEN).
- `AGENTS.md` backward AC GREEN definition (13.3a-3 BA Q4).

## Build
`source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first). Use `ctx_execute` for plugin/nn-leaf probes (bounded, short — NOT a long poll). Caveman ultra default; byte-exact exempt. BEGIN NOW. Echo `{"status":"ok","role":"BA"}` in this pane only.
