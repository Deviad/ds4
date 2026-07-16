# Story 13.3b-5 — Coder notes

## Status
STOP-ESCALATE after tokenizer fix.

## Tokenizer wiring fix

Added `scripts/apply_passthrough_chat_template.py`.

Template installed:

```jinja
{{ messages | map(attribute='content') | join('') }}
```

Patched file:

```text
/Volumes/Data NVME/mlx-ft/ds4/model-4bit/tokenizer_config.json
```

Verification:

```text
updated: /Volumes/Data NVME/mlx-ft/ds4/model-4bit/tokenizer_config.json
unchanged: /Volumes/Data NVME/mlx-ft/ds4/model-4bit/tokenizer_config.json
chat_template: {{ messages | map(attribute='content') | join('') }}
```

Passthrough token-stream probe on first training row:

```text
full_matches True
prompt_matches True
begin_count_full 1
user_count_full 1
assistant_count_full 1
full_tokens 531
prompt_tokens 33
```

No double templating detected.

## Smoke-train result

Log:

```text
agent-output/cmux-13-3b/coder-13-3b-5-smoke-train.log
```

Seq length used: `4096`.

Run loaded real 43-layer `model-4bit` and started training:

```text
Trainable parameters: 0.004% (5.571M/154430.069M)
Starting training..., iters: 20
```

Validation forward completed finite:

```text
Iter 1: Val loss 19.116, Val took 2381.597s
```

Backward failed before first optimizer update:

```text
ValueError: [scatter_axis] Cannot calculate VJP with respect to indices.
```

Full traceback in log.

## AC §3 verdict

- (a) loss finite: partial GREEN for validation forward (`19.116`); training loss not reached.
- (b) grad tree non-empty: RED / not reached; `loss_value_and_grad` raises.
- (c) every LoRA leaf grad finite: RED / not reached.
- (d) at least one LoRA grad non-zero: RED / not reached.
- (e) iter 20 completes: RED; crash before first update.
- (f) adapter saves: RED; `/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke/adapters.safetensors` absent.

`/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke/` contains only `adapter_config.json`.

## STOP classification

Not OOM; no 2048 fallback run.

Not tokenizer/data wiring; passthrough template verified.

Likely real backward gap in CSA attention scatter/put block-bias path. First compressed layer in config is layer index `2` (`compress_ratios[2]=4`); MLX traceback does not print layer, but error class points at scatter-index VJP, consistent with `_csa_block_bias_mlx` / `mx.put_along_axis`.

See stop handoff:

```text
agent-output/cmux-13-3b/coder-13-3b-5-stop.md
```

## Scope guards

Protected source hashes after run:

```text
96c39168c78e5fd9158039707334f92206e8bb39f7e718e0d9064a7356809a44  python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py
ea181743e939ba3309a445f238cd6974a73b4ca834a9219aa0973010d3c08f3f  python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py
13116c571be62f8fdfd65b326deec1ecc4e07685209b56df85f168c10d5bdc5b  python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_nn_remap.py
bcde9485a0a0bcdf2402bff0363e6159615daef380a1ba57cbddf67fd150bb64  scripts/shim_ds4_safetensors.py
```

Model shard count/size:

```text
33 shards, 149G total
```

Model shard mtime/size digest before tokenizer patch and after smoke attempt:

```text
241399eb97a7a76a6e6e73d389f500d875bd011ededfb37f31a4b70e483f9ae8
```

Only NVME model config changed:

```text
/Volumes/Data NVME/mlx-ft/ds4/model-4bit/tokenizer_config.json
sha256=8ecad81cce9fccdfed87f33215f4fa8ba1bbf7bbc68ed66706d10f4adfa5d347
```

## Local script validation

```text
python3 -m py_compile scripts/apply_passthrough_chat_template.py
```

Temp JSON fixture checks passed:

```text
updated
unchanged
temp-template-ok
different-template-guard-ok
force-template-ok
```
