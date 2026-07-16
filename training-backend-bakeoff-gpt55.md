# Research: DeepSeek V4 Flash LoRA training backend bakeoff

## Summary

DeepSeek V4 Flash LoRA training is gated first by model-code support, not by the trainer UI: any backend must successfully instantiate `model_type=deepseek_v4` / `DeepseekV4ForCausalLM`, preserve/read FP8 E4M3 weights plus UE8M0 scale tensors and FP4 expert tensors, attach PEFT LoRA to the real trainable projection modules, run backward, and export a safetensors adapter compatible with the DS4 runtime. The highest-probability path is **remote NVIDIA CUDA**, starting with **Transformers + PEFT + TRL** smoke tests, then Axolotl/LLaMA-Factory only if the same core gates pass; **Mac Studio M3 Ultra / MLX-lm is not a credible first path** unless the DS4 project ships a converted MLX model and architecture plugin.

## Research angles

1. **Architecture availability** — Does the stack know `deepseek_v4` / `DeepseekV4ForCausalLM`, or can it use `trust_remote_code` safely?
2. **Numeric format viability** — Can it load/train around FP8 E4M3, UE8M0 scales, and FP4 expert tensors without silently upcasting or failing?
3. **LoRA/adapter export** — Can it attach LoRA to the actual attention/MLP/MoE modules and write a runtime-usable safetensors adapter?
4. **Hardware/runtime fit** — Is Apple Silicon MPS/MLX viable, or is Hopper/Blackwell-class CUDA required?

## Compatibility gate table

Legend: **Pass** = likely viable now if DS4 model repo is correct; **Conditional** = viable only after smoke tests or custom integration; **Fail** = not a credible starting path.

| Stack | `deepseek_v4` / remote model code | FP8 E4M3 + UE8M0 + FP4 experts | PEFT/LoRA export | Operational risk | Gate result |
|---|---:|---:|---:|---|---:|
| **Transformers + PEFT + TRL** | Conditional: `AutoConfig`/`AutoModelForCausalLM` can use `trust_remote_code`; native support depends on installed Transformers containing or accepting DS4 code. | Conditional/high risk: PyTorch/CUDA has float8 support, but UE8M0/FP4 expert handling is model-code/kernel-specific. | Strongest: PEFT writes `adapter_model.safetensors`; TRL SFT works if model forward/backward works. | Best debugging surface; most direct path to DS4 `adapter.safetensors`. | **Best first candidate** |
| **Axolotl** | Conditional: wraps Transformers/PEFT; may need `trust_remote_code`, custom config, and target module overrides. | Conditional/high risk: inherits HF/PyTorch limits; YAML quantization paths are not evidence of DS4 FP8/FP4 MoE support. | Good if PEFT attaches; export normally PEFT-compatible. | Extra abstraction can hide model-load/module-name failures. | **Second candidate after HF smoke pass** |
| **LLaMA-Factory** | Conditional: broad HF model support but architecture whitelist/templates can lag. | Conditional/high risk: inherits HF/PyTorch limits; DS4 numeric formats unlikely to be first-class. | Good if PEFT path works. | Convenient UI/CLI, but less ideal for unknown architecture debugging. | **Second/third candidate after HF smoke pass** |
| **Unsloth** | Likely fail/conditional only if Unsloth explicitly added DS4. Optimized kernels usually target selected architectures. | Likely fail for custom FP8/FP4 MoE experts unless explicitly supported. | Good for supported models, not for unknown DS4. | Fast when supported; brittle when not. | **Do not start here** |
| **torchtune** | Likely fail: recipe/model registry oriented; unknown DS4 requires custom model builder. | Likely fail without custom kernels/format handling. | Supports LoRA concepts but not guaranteed PEFT runtime adapter format. | Good for supported families; poor fit for brand-new MoE architecture. | **Not viable initially** |
| **verl** | Conditional for RL/post-training, not the simplest SFT LoRA stack. May support DeepSeek-family distributed rollouts depending on version. | Conditional/high risk; depends on Megatron/FSDP/vLLM integration and model support. | Not the cleanest path to DS4 PEFT `adapter.safetensors`. | Use after SFT path works, especially for RLHF/GRPO. | **Not first for LoRA SFT** |
| **DeepSpeed / Megatron-LM** | Conditional: powerful if DS4 has Megatron model implementation. HF remote code alone is insufficient. | Best chance for large CUDA training, especially with Transformer Engine FP8 on Hopper/Blackwell, but FP4 experts still custom. | Weak for PEFT adapter export unless custom bridge writes PEFT-compatible safetensors. | High engineering cost; good for full/distributed training, not quick LoRA. | **Use only if HF path cannot scale** |
| **MLX-lm / Apple Silicon** | Likely fail unless DS4 architecture and conversion are explicitly implemented. | Likely fail: Apple MPS/MLX is not the natural target for FP8 E4M3 + UE8M0 + FP4 MoE training kernels. | MLX LoRA export format may need conversion to DS4 runtime PEFT safetensors. | Local convenience, but high incompatibility risk. | **Not viable as first path** |

## Findings

1. **The decisive first gate is whether `AutoConfig` recognizes or can remotely import `deepseek_v4`.** Transformers-based stacks can survive a new architecture only if either the installed package has native support or the model repository ships correct `auto_map` remote code for `DeepseekV4ForCausalLM`. If this fails, Axolotl, LLaMA-Factory, TRL, and PEFT will all fail upstream as well. Source: Hugging Face Transformers Auto classes and custom model documentation (`trust_remote_code`) — https://huggingface.co/docs/transformers/model_doc/auto and https://huggingface.co/docs/transformers/custom_models

2. **PEFT is the most plausible adapter-export layer, but only if DS4 exposes LoRA-targetable modules.** PEFT normally exports LoRA weights as safetensors, but the DS4 runtime target name `adapter.safetensors` may require either renaming `adapter_model.safetensors` or converting metadata/key names. The critical check is module census: whether projections are standard `torch.nn.Linear` or custom MoE/FP8/FP4 modules with supported LoRA injection points. Source: PEFT LoRA docs and checkpoint format — https://huggingface.co/docs/peft/task_guides/lora_based_methods and https://huggingface.co/docs/peft/developer_guides/checkpoint

3. **FP8 E4M3 is plausible on recent PyTorch/CUDA, but UE8M0 scales and FP4 experts are not generic trainer features.** PyTorch has float8 dtype work, NVIDIA Transformer Engine targets FP8 training primarily on Hopper+ GPUs, and Blackwell introduces stronger FP4/NVFP4 pathways; however, a trainer claiming LoRA support does not imply it understands DS4 scale packing or FP4 expert storage. Treat these as model-kernel compatibility gates, not YAML options. Sources: PyTorch dtype/tensor docs — https://pytorch.org/docs/stable/tensors.html ; NVIDIA Transformer Engine — https://github.com/NVIDIA/TransformerEngine ; NVIDIA Blackwell architecture materials — https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/

4. **Transformers + PEFT + TRL has the shortest path to a DS4 runtime adapter.** It minimizes abstraction: load DS4, attach LoRA, run one forward/backward, save PEFT adapter, inspect safetensors. TRL should be added only after base model + PEFT smoke tests pass. Source: TRL SFTTrainer docs — https://huggingface.co/docs/trl/sft_trainer

5. **Axolotl and LLaMA-Factory are productivity layers, not independent architecture enablers.** They can be useful after the raw HF/PEFT smoke test passes, but if DS4 model loading, target-module discovery, or FP8/FP4 handling fails in raw Transformers, these tools are unlikely to fix it. Sources: Axolotl repository/docs — https://github.com/axolotl-ai-cloud/axolotl ; LLaMA-Factory repository/docs — https://github.com/hiyouga/LLaMA-Factory

6. **Unsloth is not a safe first backend for an unknown DS4 FP8/FP4 MoE model.** Unsloth is excellent when the architecture is explicitly supported by its patched kernels, but the optimization layer is exactly what makes a new architecture risky. Use it only if its release notes explicitly mention DeepSeek V4 / DS4 Flash. Source: Unsloth docs/repository — https://github.com/unslothai/unsloth

7. **torchtune is a poor initial match.** It is recipe-oriented and strong for supported model families, but a brand-new `DeepseekV4ForCausalLM` plus unusual FP8/FP4 expert representation would require custom model/config/checkpoint code before LoRA can be trusted. Source: torchtune docs — https://pytorch.org/torchtune/stable/

8. **verl is better reserved for RL/post-training after SFT LoRA is proven.** verl can be attractive for large-scale RLHF/GRPO-style workflows, but it adds rollout, inference-engine, and distributed-training dependencies that obscure the core question: can DS4 load, train a LoRA delta, and export adapter safetensors? Source: verl repository/docs — https://github.com/volcengine/verl

9. **DeepSpeed/Megatron is the scale fallback, not the fastest adapter path.** If DS4 is too large for single-node HF/FSDP, Megatron/DeepSpeed with Transformer Engine may be the right CUDA route, but producing a DS4 runtime PEFT-style `adapter.safetensors` will likely require explicit conversion/export code. Sources: DeepSpeed docs — https://www.deepspeed.ai/ ; Megatron-LM — https://github.com/NVIDIA/Megatron-LM

10. **Mac Studio M3 Ultra should be treated as inference/dev only for this task unless DS4/MLX support is demonstrated.** Apple Silicon has high unified memory, but MLX-lm support depends on architecture conversion and MLX kernels; DS4’s FP8 E4M3 + UE8M0 + FP4 expert design points toward NVIDIA CUDA kernels, not local MPS/MLX training. Source: MLX-lm repository/docs — https://github.com/ml-explore/mlx-lm

## Exact smoke-test commands

Set a model id/path first:

```bash
export DS4_MODEL='REPLACE_WITH_DEEPSEEK_V4_FLASH_MODEL_ID_OR_LOCAL_PATH'
export HF_HOME="$PWD/.hf-cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 0. Create clean CUDA smoke-test environment

Use this on a remote CUDA host first. Prefer Python 3.11 unless a package explicitly requires newer.

```bash
python3.11 -m venv .venv-ds4-hf
source .venv-ds4-hf/bin/activate
python -m pip install -U pip wheel setuptools
# Install CUDA PyTorch matching the host driver/CUDA. Example for CUDA 12.4 wheels:
pip install --index-url https://download.pytorch.org/whl/cu124 'torch>=2.5' torchvision torchaudio
pip install -U 'transformers>=4.46' 'accelerate>=1.0' 'peft>=0.13' 'trl>=0.12' safetensors datasets sentencepiece protobuf einops
python - <<'PY'
import torch, transformers, peft, trl
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('peft', peft.__version__)
print('trl', trl.__version__)
print('float8_e4m3fn?', hasattr(torch, 'float8_e4m3fn'))
print('float4?', [x for x in dir(torch) if 'float4' in x.lower() or 'e2m1' in x.lower()])
PY
```

### 1. Inspect config and remote-code mapping

```bash
python - <<'PY'
import os, json
from transformers import AutoConfig
m=os.environ['DS4_MODEL']
cfg=AutoConfig.from_pretrained(m, trust_remote_code=True)
print('class:', type(cfg))
print('model_type:', getattr(cfg, 'model_type', None))
print('architectures:', getattr(cfg, 'architectures', None))
print('auto_map:', getattr(cfg, 'auto_map', None))
print('quantization_config:', json.dumps(getattr(cfg, 'quantization_config', None), indent=2, default=str))
assert getattr(cfg, 'model_type', None) in {'deepseek_v4','deepseek-v4','deepseek_v3','deepseek'}, 'Unexpected model_type; inspect before training'
assert any('DeepseekV4ForCausalLM' in str(x) or 'DeepSeekV4ForCausalLM' in str(x) for x in (getattr(cfg,'architectures',[]) or []) + [getattr(cfg,'auto_map',{})]), 'No visible DeepSeek V4 causal LM mapping'
PY
```

### 2. Instantiate on meta device without loading full weights

```bash
python - <<'PY'
import os
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights
m=os.environ['DS4_MODEL']
cfg=AutoConfig.from_pretrained(m, trust_remote_code=True)
with init_empty_weights():
    model=AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
print('model class:', type(model))
mods=[]
for name, module in model.named_modules():
    cls=module.__class__.__name__
    if any(k in name.lower() for k in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','expert','router','gate']):
        mods.append((name, cls))
print('candidate modules:', len(mods))
for x in mods[:300]: print(x)
PY
```

### 3. Inspect safetensors dtypes and scale tensors before training

```bash
python - <<'PY'
import os, glob, collections
from huggingface_hub import snapshot_download
from safetensors import safe_open
m=os.environ['DS4_MODEL']
path=snapshot_download(m) if not os.path.isdir(m) else m
files=glob.glob(path+'/**/*.safetensors', recursive=True)
print('safetensors files:', len(files))
assert files, 'No safetensors files found'
counts=collections.Counter(); scale_keys=[]; fp4_keys=[]; sample=[]
for f in files[:8]:
    with safe_open(f, framework='pt', device='cpu') as sf:
        for k in sf.keys():
            t=sf.get_tensor(k)
            counts[str(t.dtype)] += 1
            lk=k.lower()
            if 'scale' in lk or 'ue8' in lk: scale_keys.append((k, str(t.dtype), tuple(t.shape)))
            if 'fp4' in lk or 'e2m1' in lk or 'expert' in lk: fp4_keys.append((k, str(t.dtype), tuple(t.shape)))
            if len(sample)<30: sample.append((k, str(t.dtype), tuple(t.shape)))
print('dtype counts:', counts)
print('sample tensors:')
for x in sample: print(x)
print('scale-like keys:', len(scale_keys)); [print(x) for x in scale_keys[:80]]
print('fp4/expert-like keys:', len(fp4_keys)); [print(x) for x in fp4_keys[:80]]
PY
```

### 4. Attach PEFT LoRA to discovered modules

Start conservatively with attention projections. Add MLP/expert modules only after verifying class compatibility.

```bash
python - <<'PY'
import os, torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
m=os.environ['DS4_MODEL']
model=AutoModelForCausalLM.from_pretrained(
    m,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto',
    low_cpu_mem_usage=True,
)
# Adjust after module census if DS4 uses different names.
target_modules=['q_proj','k_proj','v_proj','o_proj']
conf=LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM', target_modules=target_modules)
model=get_peft_model(model, conf)
model.print_trainable_parameters()
trainable=[n for n,p in model.named_parameters() if p.requires_grad]
print('trainable count', len(trainable))
for n in trainable[:100]: print(n)
assert trainable, 'No LoRA trainable params attached'
PY
```

### 5. One-token forward/backward smoke test

```bash
python - <<'PY'
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
m=os.environ['DS4_MODEL']
tok=AutoTokenizer.from_pretrained(m, trust_remote_code=True, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model=AutoModelForCausalLM.from_pretrained(m, trust_remote_code=True, torch_dtype='auto', device_map='auto', low_cpu_mem_usage=True)
model=get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj','k_proj','v_proj','o_proj']))
model.train()
batch=tok(['hello world'], return_tensors='pt')
device=next(p.device for p in model.parameters() if p.requires_grad)
batch={k:v.to(device) for k,v in batch.items()}
batch['labels']=batch['input_ids'].clone()
out=model(**batch)
print('loss', out.loss)
out.loss.backward()
nonzero=0
for n,p in model.named_parameters():
    if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all():
        nonzero += 1
print('finite grad tensors', nonzero)
assert nonzero > 0, 'No finite LoRA gradients'
model.save_pretrained('ds4-lora-smoke-adapter', safe_serialization=True)
print('saved files:', __import__('os').listdir('ds4-lora-smoke-adapter'))
PY
```

### 6. Validate adapter safetensors and DS4 runtime filename

```bash
python - <<'PY'
from safetensors import safe_open
import os, shutil, json
src='ds4-lora-smoke-adapter/adapter_model.safetensors'
assert os.path.exists(src), 'PEFT did not write adapter_model.safetensors'
with safe_open(src, framework='pt', device='cpu') as sf:
    keys=list(sf.keys())
    print('adapter tensors', len(keys))
    for k in keys[:50]: print(k, sf.get_tensor(k).dtype, tuple(sf.get_tensor(k).shape))
assert any('lora_A' in k or 'lora_B' in k for k in keys), 'No LoRA keys found'
# Only do this if DS4 runtime expects exactly adapter.safetensors but accepts PEFT key schema.
shutil.copyfile(src, 'ds4-lora-smoke-adapter/adapter.safetensors')
print('wrote ds4-lora-smoke-adapter/adapter.safetensors')
print(open('ds4-lora-smoke-adapter/adapter_config.json').read()[:2000])
PY
```

### 7. Minimal TRL SFT smoke test after PEFT smoke passes

```bash
cat > /tmp/ds4_tiny.jsonl <<'EOF'
{"text":"### User: say hello\n### Assistant: hello"}
{"text":"### User: count to two\n### Assistant: one two"}
EOF
python - <<'PY'
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
m=os.environ['DS4_MODEL']
ds=load_dataset('json', data_files='/tmp/ds4_tiny.jsonl', split='train')
tok=AutoTokenizer.from_pretrained(m, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(m, trust_remote_code=True, torch_dtype='auto', device_map='auto', low_cpu_mem_usage=True)
args=SFTConfig(output_dir='ds4-trl-smoke', max_steps=1, per_device_train_batch_size=1, gradient_accumulation_steps=1, logging_steps=1, save_steps=1, max_seq_length=64, dataset_text_field='text')
peft=LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj','k_proj','v_proj','o_proj'])
trainer=SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=peft)
trainer.train()
trainer.save_model('ds4-trl-smoke/final-adapter')
PY
```

## Backend-specific smoke tests

### Axolotl

Only run after raw HF/PEFT smoke passes.

```bash
python3.11 -m venv .venv-ds4-axolotl
source .venv-ds4-axolotl/bin/activate
pip install -U pip wheel setuptools
pip install 'axolotl[deepspeed]'
cat > ds4_axolotl_smoke.yml <<EOF
base_model: ${DS4_MODEL}
trust_remote_code: true
strict: false
sequence_len: 64
sample_packing: false
pad_to_sequence_len: true
adapter: lora
lora_r: 4
lora_alpha: 8
lora_dropout: 0.0
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
datasets:
  - path: /tmp/ds4_tiny.jsonl
    type: completion
output_dir: ./ds4-axolotl-smoke
micro_batch_size: 1
gradient_accumulation_steps: 1
num_epochs: 1
max_steps: 1
learning_rate: 1e-4
optimizer: adamw_torch
bf16: auto
fp16: false
save_safetensors: true
EOF
python -m axolotl.cli.preprocess ds4_axolotl_smoke.yml
accelerate launch -m axolotl.cli.train ds4_axolotl_smoke.yml
find ds4-axolotl-smoke -maxdepth 3 -type f | sort
```

### LLaMA-Factory

```bash
python3.11 -m venv .venv-ds4-llamafactory
source .venv-ds4-llamafactory/bin/activate
pip install -U pip wheel setuptools
pip install -U llamafactory
cat > ds4_lf_dataset.json <<'EOF'
[
  {"instruction":"say hello", "input":"", "output":"hello"}
]
EOF
cat > ds4_lf_smoke.yaml <<EOF
model_name_or_path: ${DS4_MODEL}
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 4
lora_alpha: 8
lora_dropout: 0
lora_target: q_proj,k_proj,v_proj,o_proj
dataset: ds4_lf_smoke
 dataset_dir: .
template: default
cutoff_len: 64
max_samples: 1
overwrite_cache: true
preprocessing_num_workers: 1
output_dir: ds4-lf-smoke
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
max_steps: 1
learning_rate: 1.0e-4
logging_steps: 1
save_steps: 1
plot_loss: false
bf16: true
EOF
# Note: LLaMA-Factory dataset registration may require editing dataset_info.json; if so, this fails before model-specific gates.
llamafactory-cli train ds4_lf_smoke.yaml
```

### Unsloth

Use only if Unsloth explicitly claims DS4 support. Smoke check:

```bash
python3.11 -m venv .venv-ds4-unsloth
source .venv-ds4-unsloth/bin/activate
pip install -U pip wheel setuptools
pip install 'unsloth[colab-new]'
python - <<'PY'
import os
from unsloth import FastLanguageModel
m=os.environ['DS4_MODEL']
model, tok = FastLanguageModel.from_pretrained(model_name=m, max_seq_length=64, dtype=None, load_in_4bit=False, trust_remote_code=True)
model = FastLanguageModel.get_peft_model(model, r=4, target_modules=['q_proj','k_proj','v_proj','o_proj'], lora_alpha=8, lora_dropout=0, bias='none', use_gradient_checkpointing=False)
print(model)
PY
```

Expected result unless DS4 was added: unsupported architecture or patching failure.

### torchtune

Smoke gate is registry/config support. Expected fail without custom recipe.

```bash
tune ls | grep -i deepseek || true
python - <<'PY'
import torchtune, inspect
print('torchtune import ok', torchtune)
PY
```

### verl

Use after basic LoRA SFT works, mainly for RL. Smoke gate:

```bash
git clone https://github.com/volcengine/verl.git verl-src
cd verl-src
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python - <<'PY'
import verl, os
print('verl import ok')
print('DS4_MODEL', os.environ.get('DS4_MODEL'))
PY
# Then inspect examples/config for deepseek_v4/deepseek before attempting train.
grep -R "deepseek_v4\|DeepseekV4\|deepseek" -n examples verl | head -100
```

### DeepSpeed / Megatron

Use only if the HF/PEFT path works functionally but cannot fit/scale.

```bash
python3.11 -m venv .venv-ds4-mega
source .venv-ds4-mega/bin/activate
pip install -U pip wheel setuptools
pip install deepspeed transformer_engine[pytorch]
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
grep -R "DeepseekV4\|deepseek_v4\|DeepSeek" -n megatron | head -100 || true
python - <<'PY'
import deepspeed, transformer_engine, torch
print('deepspeed', deepspeed.__version__)
print('transformer_engine', transformer_engine.__version__)
print('cuda capability', torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
PY
```

### MLX-lm / Mac Studio

Run this only as a proof that local is not ready unless DS4 appears in supported conversions.

```bash
python3 -m venv .venv-ds4-mlx
source .venv-ds4-mlx/bin/activate
pip install -U pip wheel setuptools mlx mlx-lm
python - <<'PY'
import mlx.core as mx
print('mlx default device', mx.default_device())
PY
python -m mlx_lm.convert --hf-path "$DS4_MODEL" --mlx-path ds4-mlx-converted
python -m mlx_lm.lora --model ds4-mlx-converted --train --data /tmp --iters 1 --batch-size 1
```

Expected risk: conversion/model architecture failure or loss of FP8/FP4 semantics.

## Package and version risks

1. **Transformers version drift:** DS4 may require a bleeding-edge `transformers` commit, not PyPI. If config load fails, retry with `pip install git+https://github.com/huggingface/transformers.git` before abandoning HF.
2. **Remote code pinning:** Always pin the model repo revision/SHA once smoke tests pass. `trust_remote_code=True` without revision pinning is unreproducible and risky.
3. **PyTorch dtype support:** `torch.float8_e4m3fn` existing does not prove kernels support DS4 operations. Check actual forward/backward, not dtype names.
4. **CUDA architecture:** FP8 training is primarily Hopper/Ada-or-newer dependent, and FP4/NVFP4 strongly points to Blackwell-class support or custom emulation. An A100 may load after upcasting but fail to preserve intended memory/perf behavior.
5. **bitsandbytes confusion:** bitsandbytes 4-bit/NF4/QLoRA support is not the same as DS4 FP4 expert tensors. Do not treat `load_in_4bit=True` as DS4 FP4 compatibility.
6. **PEFT target modules:** Default `target_modules='all-linear'` can accidentally include router/expert internals or miss custom linear classes. Start with explicit attention projections, then expand.
7. **Adapter filename/schema:** PEFT writes `adapter_model.safetensors`; DS4 runtime asks for `adapter.safetensors`. Filename conversion is trivial; key-schema conversion may not be.
8. **TRL API churn:** `SFTTrainer` argument names have changed across TRL versions. Keep the raw PEFT smoke independent of TRL.
9. **Axolotl/LLaMA-Factory config churn:** Both move quickly; a YAML that parses today can break later. Pin commits after success.
10. **Apple local mismatch:** MPS/MLX may upcast or convert weights in ways that defeat DS4-specific FP8/FP4 storage assumptions.

## Recommendation

1. **Start on remote CUDA, not Mac Studio M3 Ultra.** Minimum practical first host: high-memory NVIDIA GPU with recent drivers; preferred: H100/H200 for FP8 exploration, B200/GB200 if FP4 expert kernels are required. If the model is very large MoE, plan multi-GPU or a provider with fast local NVMe and high RAM.
2. **Use raw Transformers + PEFT first.** Do not begin with Axolotl/LLaMA-Factory/Unsloth. Run gates 1-6 above. If they pass, you already have the target artifact path.
3. **Only then graduate to TRL/Axolotl/LLaMA-Factory for real SFT ergonomics.** Pick Axolotl if you want YAML + distributed recipes; pick LLaMA-Factory if its dataset/templates are convenient. Neither should be trusted until raw HF/PEFT proves DS4 compatibility.
4. **Avoid MLX-lm for initial training.** Use the Mac Studio for code prep, dataset formatting, safetensors inspection, and possibly tiny converted inference experiments, not DS4 Flash LoRA training.
5. **Escalate to DeepSpeed/Megatron only for scale.** If single-node HF works but cannot fit the real context/batch, then investigate FSDP/DeepSpeed/Megatron. Budget engineering time for adapter export conversion.

## Sources

- Kept: Hugging Face Transformers Auto classes and custom model docs (https://huggingface.co/docs/transformers/model_doc/auto, https://huggingface.co/docs/transformers/custom_models) — determines whether `deepseek_v4` can load at all.
- Kept: PEFT LoRA and checkpoint docs (https://huggingface.co/docs/peft/task_guides/lora_based_methods, https://huggingface.co/docs/peft/developer_guides/checkpoint) — governs LoRA injection and adapter safetensors export.
- Kept: TRL SFTTrainer docs (https://huggingface.co/docs/trl/sft_trainer) — relevant once model + PEFT gates pass.
- Kept: PyTorch tensor/dtype docs (https://pytorch.org/docs/stable/tensors.html) — baseline dtype support check.
- Kept: NVIDIA Transformer Engine (https://github.com/NVIDIA/TransformerEngine) — primary CUDA FP8 training library signal.
- Kept: NVIDIA Blackwell architecture materials (https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) — relevant for FP4/NVFP4 hardware direction.
- Kept: Axolotl repository/docs (https://github.com/axolotl-ai-cloud/axolotl) — HF/PEFT training wrapper candidate.
- Kept: LLaMA-Factory repository/docs (https://github.com/hiyouga/LLaMA-Factory) — HF/PEFT training wrapper candidate.
- Kept: Unsloth repository/docs (https://github.com/unslothai/unsloth) — optimized-kernel candidate but architecture-sensitive.
- Kept: torchtune docs (https://pytorch.org/torchtune/stable/) — recipe-based PyTorch tuning candidate.
- Kept: verl repository/docs (https://github.com/volcengine/verl) — RL/post-training candidate.
- Kept: DeepSpeed docs and Megatron-LM (https://www.deepspeed.ai/, https://github.com/NVIDIA/Megatron-LM) — distributed fallback path.
- Kept: MLX-lm repository/docs (https://github.com/ml-explore/mlx-lm) — Apple Silicon local candidate.
- Dropped: Generic blog posts and SEO tutorials — excluded because DS4 compatibility depends on exact architecture/kernels and adapter export behavior, not generic LoRA claims.
- Dropped: Generic QLoRA/bitsandbytes examples — excluded because NF4/INT4 quantized LoRA is not the same as DS4 FP4 expert tensor support.

## Gaps

- I could not verify a specific public DeepSeek V4 Flash model repository, commit SHA, or actual `config.json` because the task did not provide the model id/path and this environment exposes no web-search tool to me. The commands above are designed to close that gap quickly.
- Exact DS4 runtime adapter schema is unknown beyond the requested filename `adapter.safetensors`. Next step: compare a known-good DS4 adapter, if available, against PEFT output keys and metadata.
- FP4 expert behavior cannot be certified from package names. It must be tested on the target CUDA GPU with the actual DS4 checkpoint and remote model code.
