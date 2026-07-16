# Story 13.3b-5 — Coder STOP

## Verdict
STOP-ESCALATE.

Tokenizer `chat_template` wiring gap is fixed, but first real 43-layer `mlx_lm.lora --train` run now exposes a genuine backward/autograd blocker on real model bytes.

This is not OOM and not a tokenizer/data wiring failure.

## Completed before STOP

- Added reproducibility script: `scripts/apply_passthrough_chat_template.py`.
- Patched `/Volumes/Data NVME/mlx-ft/ds4/model-4bit/tokenizer_config.json`.
- Verified idempotence:
  - first run: `updated`
  - second run: `unchanged`
- Verified passthrough token stream on first `train.jsonl` row:
  - `full == prompt + completion`: `True`
  - `prompt_only == prompt`: `True`
  - no double template: BOS/User/Assistant counts each stay `1`
  - prompt tokens: `33`
  - full tokens: `531`
- Verified model weight shard metadata unchanged by tokenizer patch:
  - model shard mtime/size digest before patch: `241399eb97a7a76a6e6e73d389f500d875bd011ededfb37f31a4b70e483f9ae8`
  - after patch/train attempt: `241399eb97a7a76a6e6e73d389f500d875bd011ededfb37f31a4b70e483f9ae8`

## Smoke command

Log: `agent-output/cmux-13-3b/coder-13-3b-5-smoke-train.log`

Command launched through orchestrator:

```bash
cd /Users/spotted/projects/ds4-finetuning
unset SSLKEYLOGFILE
. python-envs/mlx/.venv/bin/activate
python3 scripts/finetune_ds4.py run-command smoke-train --execute --yes
```

Emitted training command:

```bash
cd '/Volumes/Data NVME/mlx-ft/ds4' && unset SSLKEYLOGFILE && . '/Volumes/Data NVME/mlx-ft/ds4/.venv/bin/activate' && mlx_lm.lora --config '/Volumes/Data NVME/mlx-ft/ds4/lora-config.json' --model '/Volumes/Data NVME/mlx-ft/ds4/model-4bit' --train --data '/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096' --adapter-path '/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke' --fine-tune-type lora --iters 20 --batch-size 1 --learning-rate 1e-5 --max-seq-length 4096 --mask-prompt --grad-checkpoint
```

## Failure

Run exit: `1`.

Model load succeeded:

```text
Trainable parameters: 0.004% (5.571M/154430.069M)
Starting training..., iters: 20
```

Validation forward succeeded, finite:

```text
Iter 1: Val loss 19.116, Val took 2381.597s
```

Failure happened immediately on first training backward step, before first optimizer update / training-loss report:

```text
File "/Volumes/Data NVME/mlx-ft/ds4/.venv/lib/python3.14/site-packages/mlx_lm/tuner/trainer.py", line 250, in step
    (lvalue, toks), grad = loss_value_and_grad(model, *batch)
File "/Volumes/Data NVME/mlx-ft/ds4/.venv/lib/python3.14/site-packages/mlx/nn/utils.py", line 35, in wrapped_value_grad_fn
    value, grad = value_grad_fn(model.trainable_parameters(), *args, **kwargs)
ValueError: [scatter_axis] Cannot calculate VJP with respect to indices.
```

## Classification

Genuine real-model backward blocker.

Likely source: CSA attention block-bias construction uses scatter/put over top-k indices in FROZEN `deepseek_v4.py`:

```text
_csa_block_bias_mlx(...)
return mx.put_along_axis(block_bias, scatter_indices, zeros, axis=-1)[..., :compressed_len]
```

First compressed layer in real config is layer index `2` (`compress_ratios[2]=4`), so failing layer is not printed by MLX but is most likely first CSA layer / layer `2` during backward.

No 2048 fallback run: Q5 fallback is for OOM only; this failure is deterministic backward VJP unsupported for scatter indices, not memory pressure.

## AC verdict

- (a) loss finite: partial only — validation loss `19.116`, no NaN/Inf before crash.
- (b) grad tree non-empty: NOT REACHED — backward raises before gradients return.
- (c) every LoRA leaf grad finite: NOT REACHED.
- (d) at least one LoRA grad non-zero: NOT REACHED.
- (e) iter 20 completes no OOM/no crash: FAIL — crash before first update.
- (f) adapter saves: FAIL — `/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke/adapters.safetensors` absent.

`/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke/` contains only `adapter_config.json`.

## Scope guard

No edits made to forbidden production/model files:

```text
96c39168c78e5fd9158039707334f92206e8bb39f7e718e0d9064a7356809a44  python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py
ea181743e939ba3309a445f238cd6974a73b4ca834a9219aa0973010d3c08f3f  python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py
13116c571be62f8fdfd65b326deec1ecc4e07685209b56df85f168c10d5bdc5b  python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_nn_remap.py
bcde9485a0a0bcdf2402bff0363e6159615daef380a1ba57cbddf67fd150bb64  scripts/shim_ds4_safetensors.py
```

Model-4bit weights untouched; only `tokenizer_config.json` changed on NVME.

## Ask

Architect/Coder follow-up needed: design a backward-safe CSA/indexer block-bias path (or equivalent gradient-safe masking) for real `deepseek_v4_nn` training. Current slice cannot patch it because scope explicitly forbids FROZEN `deepseek_v4.py` / nn-port edits beyond tokenizer data-config wiring.
