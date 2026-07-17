# DS4 Technical Specification

> Single source of truth for project structure, tech stack, build/run instructions,
> test commands, and coding rules. Referenced from `AGENTS.md`. Keep this file
> current when any of those areas change.

---

## 1. What this project is

**DS4** (`ds4.c`) is a purpose-built inference engine for **DeepSeek V4 Flash**
stored as `ds4flash.gguf`. It is not a generic GGUF runner. Goals:

- Whole-model Metal graph inference as the production path on Apple Silicon.
- SSD-streaming fallback for machines where the full model does not fit in RAM.
- CUDA / ROCm paths for Linux GPU servers.
- CPU reference path for correctness checks only (not production inference).
- DS4-native runtime LoRA: load `adapter.safetensors` beside the immutable GGUF;
  do not fuse or mutate `ds4flash.gguf`.
- A parallel Python fine-tuning toolchain (MLX on Apple Silicon, Torch/PEFT on
  CUDA) that produces portable standard LoRA adapter safetensors for DS4.

---

## 2. Repository layout

```
ds4-finetuning/
│
├── ds4.c / ds4.h              Core: model load, tokenizer, Metal graph, KV cache,
│                              SSD streaming, CPU reference forward
├── ds4_cli.c                  Interactive REPL + one-shot prompt entrypoint
├── ds4_server.c               OpenAI / Anthropic / Responses-compatible HTTP server
├── ds4_lora.c / ds4_lora.h   Runtime LoRA adapter loader and apply
├── ds4_metal.m                Objective-C Metal runtime + kernel wrappers (macOS only)
├── ds4_ssd.c / ds4_ssd.h     SSD-streaming expert paging
├── ds4_distributed.c/h        Multi-node tensor-parallel coordination
├── ds4_kvstore.c/h            Disk KV cache (session persistence)
├── ds4_web.c/h                HTTP helpers for the server
├── ds4_bench.c                Throughput benchmarking tool
├── ds4_eval.c                 Evaluation harness (perplexity, extractors, self-test)
├── ds4_agent.c                Autonomous agent loop
├── ds4_cuda.cu                CUDA kernel/dispatch layer (Linux)
├── ds4_rocm.cu / ds4_rocm.h  ROCm / HIP layer (AMD GPUs)
├── linenoise.c/h              Embedded readline alternative (REPL)
├── rax.c/h / rax_malloc.h    Embedded radix-tree (routing lookup, SSD hotlist)
│
├── metal/                     Metal compute kernels (.metal sources)
│   ├── flash_attn.metal       Flash attention
│   ├── moe.metal              Mixture-of-experts routing + compute
│   ├── dsv4_hc.metal          HyperConnection residual stream mixing
│   ├── dsv4_kv.metal          KV projection / normalisation
│   ├── dsv4_rope.metal        RoPE (partial + full)
│   ├── dsv4_misc.metal        Misc DS4-specific helpers
│   ├── norm.metal             RMSNorm
│   ├── dense.metal            General GEMM tiles
│   └── ...                    (argsort, bin, concat, cpy, glu, softmax, etc.)
│
├── scripts/
│   ├── finetune_ds4.py        Fine-tuning pipeline CLI (gates, dry-run, execute)
│   ├── convert_lora_to_ds4.py Convert standard LoRA safetensors → DS4 canonical format
│   └── shim_ds4_safetensors.py FP8 shim: rewrite F8_E4M3/F8_E8M0 → BF16/F32
│
├── python-envs/
│   ├── mlx/                   MLX fine-tuning environment (Apple Silicon)
│   │   ├── pyproject.toml     Declarative dependency set (mlx, mlx-lm, transformers…)
│   │   └── src/
│   │       ├── sitecustomize.py   Project-controlled MLX-LM plugin path injection
│   │       └── ds4_ft_mlx/
│   │           ├── deepseek_v4_attention_spec.py  Pure-Python parity reference (oracle)
│   │           ├── deepseek_v4_checkpoint.py      Safetensors header scan + shape compat
│   │           ├── deepseek_v4_dequant.py         Dequantisation helpers (I8, FP4, FP8)
│   │           ├── deepseek_v4_mapping.py         Tensor-name mapping + MTP gating
│   │           ├── deepseek_v4_moe_spec.py        Pure-Python MoE parity reference
│   │           ├── lora_targets.py                DS4-safe LoRA target allowlist
│   │           ├── mlx_lm_plugin.py               Plugin registration hook for mlx-lm
│   │           └── vendor/mlx_lm_models/
│   │               └── deepseek_v4.py             Project-controlled MLX model (vendored)
│   └── torch/
│       └── pyproject.toml     Torch/PEFT environment (Python 3.12, CUDA-ready)
│
├── tests/
│   ├── ds4_test.c             C integration tests (Metal path, extractors)
│   ├── ds4_lora_test.c        C LoRA adapter loader unit tests
│   ├── ds4_agent_test.c       Agent loop tests
│   ├── test_finetune_ds4.py   Pipeline CLI unit tests
│   ├── test_shim_ds4_safetensors.py   FP8 shim tests
│   ├── test_convert_lora_to_ds4.py    Adapter converter tests
│   ├── test_deepseek_v4_attention_parity.py  Pure-Python attention oracle tests
│   ├── test_deepseek_v4_moe_parity.py        Pure-Python MoE oracle tests
│   ├── test_deepseek_v4_dequant_parity.py    Dequant parity tests
│   ├── test_deepseek_v4_checkpoint.py        Checkpoint header/loader tests
│   ├── test_deepseek_v4_lora_targets.py      LoRA target allowlist tests
│   ├── test_deepseek_v4_mlx_port.py          MLX array implementation tests
│   └── test_deepseek_v4_forward_parity.py    Integrated forward parity gate
│
├── docs/
│   ├── technical-spec.md            ← this file
│   ├── deepseek-v4-architecture-dossier.md
│   └── deepseek-v4-mtp-policy.md
│
├── agent-output/              Slice plans, scout reports, reviewer findings
├── misc/                      Ignored notes and old planning material
├── AGENTS.md                  Agent rules, routing, quality gates (auto-loaded)
├── docs/backlog.md          Epic/story plan with acceptance criteria
├── training-next-status.md    Living status tracker and model-routing policy
└── progress.md                Current slice status and open items
```

External workspace (outside repo, never committed):

```
/Volumes/Data NVME/mlx-ft/ds4/
├── .venv/                  MLX Python 3.14 environment
├── .venv-torch/            Torch Python 3.12 environment
├── hf-f8shim/              FP8-shimmed HuggingFace checkpoint (BF16/F32)
├── dataset/                Built JSONL train/valid/test splits
└── model-4bit/             (blocked) Converted MLX quantised base — does not exist yet
```

---

## 3. Tech stack

| Layer | Technology | Notes |
|---|---|---|
| Inference engine | C99, `cc` / `clang` | `-O3 -ffast-math -std=c99`; macOS uses `-mcpu=native`, Linux `-march=native` |
| GPU kernels — Apple | Metal (`.metal`) + Objective-C (`-fobjc-arc`) | Compiled at build time via `xcrun metal` |
| GPU kernels — NVIDIA | CUDA (`.cu`), `nvcc` | `--use_fast_math`, configurable `CUDA_ARCH` |
| GPU kernels — AMD | ROCm / HIP (`.cu`), `hipcc` | `--offload-arch=gfx1151` default |
| Fine-tuning — Apple | Python ≥3.14, MLX ≥0.31.2, MLX-LM ≥0.31.3 | Project-controlled vendored `deepseek_v4.py` |
| Fine-tuning — CUDA | Python 3.12, PyTorch ≥2.3, PEFT, TRL, Accelerate | MPS available locally; CUDA requires remote machine |
| Adapter conversion | Python 3.x (`safetensors`, `numpy`) | `scripts/convert_lora_to_ds4.py` |
| Checkpoint shim | Python 3.x (`safetensors`) | `scripts/shim_ds4_safetensors.py` |
| Build system | GNU Make | `Makefile` with platform auto-detection |
| Package management | `uv` + `pip install -e` | Isolated venvs, declarative `pyproject.toml` |
| Embedded libs | `linenoise` (REPL), `rax` (radix tree) | Vendored in-tree |

---

## 4. Building

### 4.1 macOS — Metal (default)

```bash
# Requires Xcode command-line tools and macOS 14+
make            # builds ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, ./ds4-agent
```

To rebuild after changing Metal kernels, `make clean` first — Metal sources are
compiled as part of `ds4_metal.m` and do not have their own incremental rules.

```bash
make clean && make
```

### 4.2 macOS — CPU-only (no Metal, for reference/debug)

```bash
make cpu
```

Produces `ds4_cpu`, `ds4_server_cpu`, etc. Never use for production inference.

### 4.3 Linux — CUDA

```bash
make cuda                         # generic CUDA
make cuda-spark                   # Blackwell/Spark variant
CUDA_ARCH=sm_90 make cuda-generic # explicit arch
```

Requires `nvcc` on `PATH` (default: `/usr/local/cuda/bin/nvcc`). Override:

```bash
CUDA_HOME=/opt/cuda make cuda
```

### 4.4 Linux — ROCm / AMD (Strix Halo)

```bash
make strix-halo     # gfx1151 default
make rocm           # alias for strix-halo
```

---

## 5. Running the runtime

All runtime commands assume `ds4flash.gguf` is in `CWD` or pass `-m PATH`.

### 5.1 Interactive chat (Metal / default backend)

```bash
./ds4
./ds4 -m /path/to/ds4flash.gguf          # explicit model path
./ds4 --ssd-streaming                    # SSD-backed expert paging
./ds4 --lora adapter.safetensors         # load LoRA adapter (validated, then applied)
```

### 5.2 One-shot prompt

```bash
./ds4 -p "Your prompt here"
./ds4 --prompt-file long_prompt.txt
```

### 5.3 HTTP server (OpenAI / Anthropic-compatible)

```bash
./ds4-server -m ds4flash.gguf            # listens on default port
./ds4-server -m ds4flash.gguf --lora adapter.safetensors
./ds4-server --ssd-streaming -c 32768    # SSD mode, 32k context
```

The server exposes:
- `POST /v1/chat/completions` (OpenAI)
- `POST /v1/messages` (Anthropic)
- `POST /v1/responses` (Responses API)
- `POST /completion` (raw completion)

### 5.4 Inspect mode (no inference)

```bash
./ds4 --inspect -m ds4flash.gguf
./ds4 --inspect -m ds4flash.gguf --lora adapter.safetensors   # validate adapter
```

### 5.5 Diagnostics

```bash
./ds4 -p "prompt" --dump-logprobs out.json
./ds4 --perplexity-file text.txt
./ds4 --metal-graph-full-test            # GPU graph vs CPU reference
./ds4 --expert-profile profile.json      # MoE expert locality
```

### 5.6 Benchmarking

```bash
./ds4-bench -m ds4flash.gguf
```

---

## 6. Python environments

Environments live **outside the repo** at `/Volumes/Data NVME/mlx-ft/ds4/`. Do
not install fine-tuning packages into the global Python.

### 6.1 Create / recreate the MLX environment

```bash
cd /Volumes/Data\ NVME/mlx-ft/ds4
uv venv --seed .venv
source .venv/bin/activate
pip install -U pip
pip install -e /Users/spotted/projects/ds4-finetuning/python-envs/mlx
python /Users/spotted/projects/ds4-finetuning/scripts/finetune_ds4.py \
  run-command mlx-lm-source --mlx-lm-source release --execute --yes
```

- Python 3.14 (current: 3.14.5)
- Installs: `mlx==0.31.2`, `mlx-lm==0.31.3`, `transformers`, `datasets`, `safetensors`
- `sitecustomize.py` in the `src/` directory injects the project-controlled
  `mlx_lm.models.deepseek_v4` plugin path automatically on `import mlx_lm`.
- `setup-env` reconciles back to released MLX-LM by default. Selecting the fork is explicit and reversible; never rely on package version alone because the pinned fork also reports `0.31.3`.

### 6.2 Create / recreate the Torch/PEFT environment

```bash
cd /Volumes/Data\ NVME/mlx-ft/ds4
uv venv --seed --python 3.12 --clear .venv-torch
source .venv-torch/bin/activate
pip install -U pip
pip install -e /Users/spotted/projects/ds4-finetuning/python-envs/torch
```

- Python 3.12.11
- Installs: `torch`, `transformers`, `peft`, `trl`, `accelerate`, `datasets`
- MPS available locally; CUDA requires explicit `--backend remote-cuda`.

### 6.3 Activating for one-off commands

```bash
# MLX
source /Volumes/Data\ NVME/mlx-ft/ds4/.venv/bin/activate

# Torch
source /Volumes/Data\ NVME/mlx-ft/ds4/.venv-torch/bin/activate
```

### 6.4 MLX-LM source selection and verification

Story 14.0 adds `vendor/mlx-lm` as a real submodule pinned to `git@github.com:Deviad/mlx-lm.git`. The initial pin was `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`; Story 14.4 advanced it to `80fab4e419a57f9465bb9e2f4e90010d645e124c` to include the Story 14.1/14.2a trainer seam. The release package remains the default source. The fork is bootstrap-only and no file below `vendor/mlx-lm` may be edited in Story 14.0.

```bash
# Materialize and verify the source-control pin.
git submodule update --init --recursive vendor/mlx-lm
git ls-files --stage -- vendor/mlx-lm
git -C vendor/mlx-lm rev-parse HEAD
git -C vendor/mlx-lm remote get-url origin
git submodule status --recursive vendor/mlx-lm

# Dry-run selector commands.
python3 scripts/finetune_ds4.py run-command mlx-lm-source
python3 scripts/finetune_ds4.py run-command mlx-lm-source --mlx-lm-source fork

# Execute only after reviewing the dry run. Both paths use the isolated venv's
# absolute python, `python -m pip`, `--isolated`, `--require-virtualenv`, and
# `--no-deps`; fork mode also uses `--no-build-isolation -e`.
python3 scripts/finetune_ds4.py run-command mlx-lm-source --mlx-lm-source release --execute --yes
python3 scripts/finetune_ds4.py run-command mlx-lm-source --mlx-lm-source fork --execute --yes

# Read-only identity verification.
"$MLX_WORK/.venv/bin/python" scripts/finetune_ds4.py \
  mlx-lm-source-verify --mode release --scope all --mlx-work "$MLX_WORK"
"$MLX_WORK/.venv/bin/python" scripts/finetune_ds4.py \
  mlx-lm-source-verify --mode fork --scope all --mlx-work "$MLX_WORK"
```

The verifier fails closed on wrong venv, user-site enabled, `mlx` drift from `0.31.2`, MLX-LM drift from `0.31.3`, wrong submodule URL/SHA/HEAD/origin, dirty or uninitialized submodule state, and release/fork import identity mismatches. Release identity must resolve inside the isolated venv and outside `vendor/mlx-lm`; fork identity must resolve inside `vendor/mlx-lm` and carry editable PEP 610 metadata pointing at that path.

---

## 7. Fine-tuning pipeline

All fine-tuning operations go through `scripts/finetune_ds4.py`. Expensive steps
are dry-run-first and require `--execute --yes` to actually run.

### 7.1 Pipeline commands (cheapest → most expensive)

```bash
# Validation and gate checks (cheap, always safe)
python3 scripts/finetune_ds4.py preflight
python3 scripts/finetune_ds4.py build-dataset
python3 scripts/finetune_ds4.py validate-dataset
python3 scripts/finetune_ds4.py token-audit

# Architecture and checkpoint gates (dry-run safe)
python3 scripts/finetune_ds4.py deepseek-v4-import-check
python3 scripts/finetune_ds4.py deepseek-v4-tiny-config-check
python3 scripts/finetune_ds4.py deepseek-v4-mtp-exclusion-check
python3 scripts/finetune_ds4.py deepseek-v4-mapping-check
python3 scripts/finetune_ds4.py deepseek-v4-dequant-parity-check
python3 scripts/finetune_ds4.py deepseek-v4-forward-parity-check
python3 scripts/finetune_ds4.py model-4bit-conversion-plan
python3 scripts/finetune_ds4.py deepseek-v4-forward-parity-readiness
python3 scripts/finetune_ds4.py mlx-lora-targets-check

# Inspect what expensive commands would do (no execution)
python3 scripts/finetune_ds4.py emit-commands
python3 scripts/finetune_ds4.py emit-commands mlx-lm-source --mlx-lm-source fork

# Execute one expensive step (requires both flags)
python3 scripts/finetune_ds4.py run-command mlx-lm-source --mlx-lm-source release --execute --yes
python3 scripts/finetune_ds4.py run-command convert-shimmed --execute --yes
```

### 7.2 Gate marker system

Expensive steps are blocked behind structured JSON gate markers written to the
external workspace. Each marker is schema-validated and hash-bound to the
relevant checkpoint or config. Existence alone is not sufficient; the format and
hash must match. Example:

```
/Volumes/Data NVME/mlx-ft/ds4/.deepseek-v4-mapping-ok
/Volumes/Data NVME/mlx-ft/ds4/.deepseek-v4-mtp-exclusion-ok
/Volumes/Data NVME/mlx-ft/ds4/.deepseek-v4-dequant-parity-ok
/Volumes/Data NVME/mlx-ft/ds4/.deepseek-v4-forward-parity-ok   ← ABSENT (correct; not yet earned)
```

`convert-shimmed` requires `.deepseek-v4-forward-parity-ok`. That marker must not
be written until full architecture forward parity is proven (Stories 11.14–11.16).

### 7.3 Model-4bit conversion plan (read-only)

`model-4bit-conversion-plan` validates `$MLX_WORK/hf-f8shim`, the shimmed index
hash, and all non-forward DeepSeek V4 gates, then writes
`$MLX_WORK/model-4bit-conversion-plan.json` with `execution_allowed=false` and
the exact future `convert-shimmed` command. It never runs `mlx_lm.convert`, never
creates `model-4bit`, never writes `.deepseek-v4-forward-parity-ok`, and fails
closed on stale `model-4bit` output unless `--report-only` is passed. Real
conversion remains gated by `.deepseek-v4-forward-parity-ok`.

### 7.4 Forward-parity readiness report (read-only)

`deepseek-v4-forward-parity-readiness` is a fail-closed diagnostic. It writes
`$MLX_WORK/deepseek-v4-forward-parity-readiness.json` with
`full_forward_parity=false`, `marker_earned=false`, current 19-fixture coverage,
`forward_parity_blockers()`, non-forward gate status, and B0–B3+GATE criteria.
It also records `real_mode_proofs` (partial evidence): real-mode
`Model(forward_parity_fixture=None)` + public `load_weights()` + `__call__`
parity for compressor-free B0a tiny configs, single-layer `hc_mult=2` hyperhead,
B0b-a CSA compressed attention, `B0b-a-2` DSA top-k sparse-selection primitive,
and `B2-a-1`/`B2-a-2`/`B2-a-3`, synthetic single-layer, multi-layer, and
top-k>1 multi-expert I8 block-scale dequant integration proofs through the
real-mode MoE branch. The proof-count contract is now 8. The primitive proof
does not thread `index_topk` into production real-mode forward; B2-a-1/B2-a-2/
B2-a-3 do not cover packed FP4, real payload decode, expert kernels, or
`hc_mult>1` multi-layer stacking (B1). This is partial evidence only; B0/B2 are
not satisfied. It never writes
`.deepseek-v4-forward-parity-ok`, creates `model-4bit`, runs
conversion/training/generation, or weakens `convert-shimmed`. The report also
carries an additive `b2_real_checkpoint_payload_readiness` fail-closed diagnostic
block (ADR 0014) that records the ADR-0007 header-only-verified real-checkpoint
expert payload classification (via the existing Story 11.15a `read_safetensors_header`
path over the original HF F8 checkpoint; never payload bytes) and the exact
missing-independent-trusted-reference requirement gating routed I8 real-payload
decode; it is diagnostic only and adds no `real_mode_proofs` entry. The report
also carries `b2_routed_dequant_trusted_reference_readiness` (ADR 0015), which
records the per-candidate trusted-reference landscape (convert.py/kernel.py/
Transformers/our torch-numpy/shim/DS4-CPU harness) plus the convert.py per-32
FP4-vs-real-per-16 I8 mismatch machine-readably; it is fail-closed DATA only and
adds no proof, marker, conversion, payload decode, or gate lift.

### 7.5 Later 4096 eager smoke policy

The real DeepSeek V4 MLX smoke remains forbidden until the sparse routed FP4
backward slice has independent Reviewer PASS and Test Manager GREEN. When that
happens, launch exactly one 4096-token smoke through a wrapper that disables MLX
compilation locally and retains the 400 GB scheduler limit:

```python
import mlx.core as mx
mx.disable_compile()
mx.set_memory_limit(400_000_000_000)
from mlx_lm.lora import main
main()
```

The command arguments stay pinned to the Story 13.3b-5c/5d contract:

```text
--config '/Volumes/Data NVME/mlx-ft/ds4/lora-config.json'
--model '/Volumes/Data NVME/mlx-ft/ds4/model-4bit'
--train
--data '/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096'
--adapter-path '/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke-memory400'
--fine-tune-type lora
--iters 20
--batch-size 1
--learning-rate 1e-5
--max-seq-length 4096
--mask-prompt
--grad-checkpoint
--val-batches 1
--steps-per-report 1
```

The wrapper/log must print eager compile policy, selected
`400_000_000_000`-byte limit, start time, PID, exact command, and per-step MLX
cache/active/peak telemetry. One validation batch must complete with finite loss
before training. STOP before first backward if peak memory is `>=340 GB`. The
first backward/optimizer step must complete with finite loss, non-empty finite
LoRA gradients, at least one non-zero expected gradient/update, and peak
`<340 GB`; only then continue the same single smoke through iteration 20. No
shorter automatic fallback or second smoke is authorized without a new re-pin.

### 7.6 Adapter conversion

```bash
# Convert standard LoRA safetensors → DS4 canonical format
python3 scripts/convert_lora_to_ds4.py \
    --adapter adapters/adapter.safetensors \
    --out adapter.ds4.safetensors

# Validate in DS4
./ds4 --inspect -m ds4flash.gguf --lora adapter.ds4.safetensors
```

---

## 8. Testing

### 8.1 C integration tests

Requires the built binaries and model `ds4flash.gguf` present. Runs Metal
self-tests, extractor correctness checks, and agent loop tests.

```bash
make test
```

**Note:** `ds4: cannot open model 'ds4flash.gguf'` is expected when the model
file is absent. The C test suite is otherwise self-contained.

### 8.2 LoRA C unit tests

Lightweight; does not require a model file.

```bash
make lora-test
```

### 8.3 Python fine-tuning tests (primary suite)

Runs without the MLX or Torch environments activated. MLX-dependent tests skip
automatically when `import mlx.core` fails.

```bash
make finetune-test
# equivalent:
python3 -m unittest \
    tests/test_finetune_ds4.py \
    tests/test_shim_ds4_safetensors.py \
    tests/test_convert_lora_to_ds4.py \
    tests/test_deepseek_v4_mlx_port.py \
    tests/test_deepseek_v4_attention_parity.py \
    tests/test_deepseek_v4_moe_parity.py \
    tests/test_deepseek_v4_dequant_parity.py \
    tests/test_deepseek_v4_checkpoint.py \
    tests/test_deepseek_v4_lora_targets.py \
    tests/test_deepseek_v4_forward_parity.py
```

**Expected baseline:** 236 OK, 54 skipped (skips are MLX-only and
Transformers-only tests that guard themselves with `self.skipTest`).

### 8.4 MLX-specific tests (MLX venv required)

The MLX venv provides `mlx.core`. Activate it first, then run any single test
file or the full suite:

```bash
source /Volumes/Data\ NVME/mlx-ft/ds4/.venv/bin/activate
python3 -m unittest tests/test_deepseek_v4_mlx_port.py -v
```

### 8.5 Torch parity tests (Torch venv required)

```bash
source /Volumes/Data\ NVME/mlx-ft/ds4/.venv-torch/bin/activate
python3 -m unittest tests/test_deepseek_v4_forward_parity.py -v
```

**Critical:** never run Torch and MLX tests in the same Python process. The
`nanobind` and `torch` C-extensions conflict and can segfault. Run them in
separate invocations.

### 8.6 Quick sanity check before committing

```bash
make finetune-test   # must be 236 OK (54 skipped)
git diff --check     # must be CLEAN (no trailing whitespace / conflict markers)
```

---

## 9. Coding practices

### 9.1 Test-driven development (mandatory)

Every code change follows **red → green → refactor**:

1. Write a failing test that asserts the desired behaviour. Run it; confirm it
   fails (red).
2. Write the minimal implementation to make it pass (green).
3. Refactor for clarity and remove duplication without breaking the test.

Never merge an implementation before its tests are green. Never write tests after
the fact to paper over an existing implementation.

### 9.2 Clean, minimal code

- Write the most minimal, correct design. Explore alternatives before settling.
- Avoid slop: one-off patches for specific cases, dead code, unnecessary
  abstraction layers, over-engineered generality.
- Keep cyclomatic complexity low. Short functions with a single clear purpose.
- Use design patterns only where they genuinely reduce coupling or clarify intent.

### 9.3 Commenting policy

- Comment important inference code where model mechanics, cache lifetime, memory
  policy, or API orchestration are non-obvious.
- Comments explain **why** a shape, ordering, memory boundary, or routing
  decision exists — not what the code obviously already says.
- Prefer inline comments beside the code. Do not write separate design docs for
  things that fit in a well-placed comment.

### 9.4 API surface

- Keep public APIs narrow. CLI and server code must not reach into tensor
  internals.
- No permanent semantic variants behind flags. Diagnostic/debug switches that
  validate the one release path are acceptable.
- Do not introduce C++.

### 9.5 Fail-closed pattern

Every speculative or partially-proven path must raise or abort rather than
silently continue with undefined behaviour:

```python
# Good: explicit gate
if args.compression_ratio != 0:
    raise NotImplementedError("MLX attention does not support compressors yet")

# Bad: silent fallthrough that produces wrong results
```

This applies to: unsupported dtypes, unproven model subsets, unearned gate
markers, and any path that has not been validated against a trusted reference.

### 9.6 Parity-first on numerical code

New MLX or Torch implementations must be validated against a trusted reference
(pure-Python oracle, Transformers model, or DS4 CPU path) before the fail-closed
gate is relaxed. Tolerance for MLX array operations: ≤1e-5 max absolute error.

---

## 10. Architecture rules and invariants

### 10.1 Never mutate the base GGUF

`ds4flash.gguf` is immutable. LoRA adapters load separately via `--lora`. Do not
rewrite, quantise, or fuse into the base file.

### 10.2 Backend isolation

Each inference path (Metal, CUDA, ROCm, CPU) is self-contained. A fix for one
must not silently break another. After any Metal change verify speed is
maintained. After any structural change check if CUDA could be broken and, if
so, get access to the CUDA machine before merging.

### 10.3 SSD streaming latency hiding

Expert loads must be scheduled to overlap with the inference of the current
layer. Do not allow the next-layer load to serialize with inference.

### 10.4 Memory-mapped model loading

The default Metal path keeps the model mmap-backed. Do not eagerly copy the full
GGUF into RAM. SSD streaming uses explicit allocated buffers — that is intentional
and separate.

### 10.5 CPU path is reference only

Never use the CPU path for production or benchmarking. Large CPU inference runs
on macOS have previously triggered kernel VM failures with large mappings.

### 10.6 No concurrent large model processes

The instance lock is intentional. Do not start a second `ds4` or `ds4-server`
process while another is running with the full model loaded.

### 10.7 Python environment isolation

Fine-tuning packages must not be installed into the global Python or into a
shared environment. Each stack (MLX, Torch) gets its own `uv venv --seed`
environment. Package lists live in `pyproject.toml` files; never hard-code them
in scripts or commands.

### 10.8 MTP tensors are review-gated

`mtp.*` tensors in the checkpoint require an explicit review step before any
conversion or training that touches them. The `.deepseek-v4-mtp-exclusion-ok`
marker encodes the decision; do not forge or copy it.

### 10.9 Packed expert decode stays fail-closed

`dequantize_expert_packed("fp4" | "i8")` raises `NotImplementedError` until
trusted full-path parity with a reference (Transformers or DS4 CPU) is proven.

### 10.10 Full forward marker is not written speculatively

`.deepseek-v4-forward-parity-ok` must only be written after integrated
architecture parity is proven for: compressors/indexers (Story 11.14), real-scale
FP4/256-expert/top-6 MoE (Story 11.15), and KV cache / generation path (Story
11.16). Do not write it to unblock a downstream step prematurely.

### 10.11 Opaque packed-FP4 Metal training primitive gate

Story 13.3b-5f adds one MLX fine-tuning-only packed-FP4 primitive under
`python-envs/mlx/src/ds4_ft_mlx/metal/`. It is package-local data, loaded only by
`ds4_ft_mlx.routed_fp4_metal`, and must not be referenced by root `metal/*.metal`,
`ds4_metal.m`, or any C/CUDA/ROCm/SSD/distributed inference path. The primitive is
first-order only, Apple Metal + `mlx==0.31.2` fail-closed, and replaces the stopped
Python nested-`mx.vjp` routed FP4 training path. No real 4096-token training smoke is
authorized until independent Reviewer PASS and Test Manager GREEN for the primitive
parity, opacity, fixed-assignment memory, no-shard real-dimension memory, LoRA-gradient,
tracked-test, and production-hash gates.

ADR 0028's Architect-r3 reduction re-pin keeps SIMD reduction mandatory in all
five matrix kernels: lane `l` accumulates `l,l+32,...` in FP32, all lanes join
one final `simd_sum`, and lane 0 writes. Forward and VJP recomputation must use
the same lane/FMA/reduction order and clamp inputs; lane-0 serial tile dots and
shape-selected reduction variants are forbidden. Forward parity uses
`abs(got-ref) <= 2e-6 + 1e-6*abs(ref)` plus NRMSE `<=1e-6`; VJP/score parity
remains `atol=rtol=1e-5`, and exact clamp-boundary masks remain exact. Before
Reviewer round 3, the tracked 25-repeat `R=96` benchmark must report forward
p50 `<=0.0080s`, input-VJP p50 `<=0.0140s`, and `256x43x20` extrapolation
`<=1.25h` p50 / `<=1.35h` p95. These gates do not authorize the full
5,000-iteration run.

### 10.12 Story 13.3b-5g first-backward OOM re-entry

The single authorized 4096-token smoke at commit `5dee4ce` is an end-to-end
memory RED: finite validation completed, then the first backward evaluation
aborted with Metal command-buffer out-of-memory (exit `134`) before gradient
telemetry or an adapter payload. One-layer routed-operation memory probes do not
establish the peak of the 43-layer checkpointed trainer graph or cover ordinary
attention/shared-expert backward and command-buffer transients.

`grad_checkpoint(model.layers[0])` patches the shared `DecoderLayerNN.__call__`
class method and therefore covers all 43 current decoder-layer instances. Do not
apply a checkpoint call-site fix without contrary tracked evidence. Real training
stays STOPPED pending a tracked no-model/no-shard multi-layer synthetic peak
report with depth/expert sweeps, component ablations, and one lazy graph versus
sequential-evaluation controls. No second real smoke, shorter fallback, full run,
primitive redesign, or layer-serial backward is authorized without a new
BA/Architect re-pin plus independent Reviewer PASS and Test Manager GREEN.

---

## 11. Subagent model routing

Full routing table and rules are in `AGENTS.md` § **Subagent model routing**
(auto-loaded every session). Summary:

| Role | Model | Subscription |
|---|---|---|
| BA / Architect | `neuralwatt/glm-5.2` default; fallback `anthropic/claude-opus-4-8` | Neuralwatt / Anthropic |
| Coding + Code review | `openai-codex/gpt-5.5` | OpenAI |
| Tester / Test Manager | `neuralwatt/qwen3.6-35b` default; fallback `openai-codex/gpt-5.4-mini` | Neuralwatt / OpenAI |
| Utility (commands, scans, search) | `anthropic/claude-sonnet-4-6` | Anthropic |

Code review always goes through the dedicated `xhigh-reviewer` agent (fresh
context, `system-prompt: replace`, model `openai-codex/gpt-5.5`). File-mutating
coding dispatches are kept serial: one worker slice at a time, reviewed before
the next.

## Story 14.5 — bounded two-phase checkpoint/resume pilot

`ds4_segmented_pilot.py` is an explicit operator-only entry point. It is excluded from `DEFAULT_BACKEND_STEPS["local-mlx"]`; `ds4-segmented-smoke`, its report, marker, provider, and source bytes remain frozen.

Pinned phases:

- Phase A: `iters=2`, `steps_per_eval=2`, `save_every=1`, timeout `2700s`, output `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-a`, report `agent-output/cmux-14-5/phase-a-report.json`, log `agent-output/cmux-14-5/phase-a-log.txt`.
- Phase B: `iters=1`, `steps_per_eval=1`, `save_every=1`, timeout `1500s`, output `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-b`, resume source `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-a/0000002_adapters.safetensors`, report `agent-output/cmux-14-5/phase-b-report.json`, log `agent-output/cmux-14-5/phase-b-log.txt`.
- Common contract: `max_seq_length=4096`, `batch_size=1`, `learning_rate=1e-5`, prompt masking and gradient checkpointing enabled, `segment_size=1`, gradient accumulation `1`, seed `0`, LoRA, Adam, `num_layers=16`, `val_batches=25`, report/save cadence `1`, total active budget `4200s`.

Catalog commands use the pinned interpreter, `bash -lc`, `set -o pipefail`, unbuffered output, and `tee`. Both phase output directories and markers are one-attempt gates: existing paths fail closed and are never deleted. Phase B requires a durable Phase A OK report/marker binding and exact resume checkpoint file and canonical tensor digest. Safetensors identity uses chunked file SHA-256 plus canonical tensor digest v1 over sorted tensor name, dtype, shape, and raw bytes, excluding metadata.

Each phase must record finite per-step loss, positive token count, gradient schema/finiteness, provider-call and callback cardinality, checkpoint hashes, and local/global mapping (`1,2` → `1,2`; resumed `1` → `3`). Reports and markers use same-directory temporary files, `fsync`, and `os.replace`; success markers follow durable reports. Success publication is fail-atomic at the evidence level: any report or marker write failure removes every success marker before failure reports are written, and surviving failure markers bind to the report they name. Any lock, timeout, identity, schema, checkpoint, resource, report, or marker failure is terminal. There is no retry, fallback, reduced-length rerun, alternate backend, or smoke rerun. Story 14.5's synthetic implementation gate is closed by its prior independent verdicts. Story 14.5a's repaired resource/namespace gate is closed by Reviewer r8b PASS and Test Manager r8b GREEN; real Phase A2 still requires fresh explicit operator authorization. This section does not authorize real execution.

The synthetic gate is:

```bash
PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" PYTHONDONTWRITEBYTECODE=1 python-envs/mlx/.venv/bin/python -m pytest -q tests/test_ds4_segmented_pilot.py tests/test_ds4_segmented_smoke.py tests/test_finetune_ds4.py tests/test_mlx_lm_source.py tests/test_ds4_segmented_loss_and_grad.py tests/test_ds4_gguf_base_smoke.py
```

No real model, dataset, adapter, training, inference, CUDA, or distributed execution is part of implementation or review. Future execution requires separate operator authorization for each phase in a visible panel, with report/marker/checkpoint verification between phases. Adapter-weight continuity does not imply optimizer, RNG, dataset-cursor, scheduler, or trainer-global-step continuity, and the pilot makes no quality, convergence, throughput, or full-training-readiness claim.

## Story 14.5a — resource observer repair and attempt-2 namespace

Story 14.5a synthetic closeout is complete: Reviewer r8b **PASS**, Test Manager r8b **GREEN**, and the canonical six-file suite passed `508 passed, 3 skipped` (`1 warning`, `2 subtests passed`). Real execution remains unauthorized. `_resource_gate()` uses per-process observation and skips only `psutil.AccessDenied`, `psutil.NoSuchProcess`, and `psutil.ZombieProcess`. It records `allowed_process_skips.total`, fixed `counts_by_type`, sorted `pids_by_type`, sorted `unknown_pid_counts_by_type`, and all successfully observed non-current `[pid, rss]` pairs. Unknown enumeration, PID, RSS, virtual-memory, disk, or evidence errors fail closed with `resource_observer_error={stage, exception_type, pid}`. Thresholds remain strict RSS `>50 GiB`, available memory `>=32 GiB`, and disk free `>=1 GiB`.

Attempt 1 remains immutable historical evidence. Its five-file size/SHA-256 manifest covers `agent-output/cmux-14-5/phase-a-log.txt`, `agent-output/cmux-14-5/phase-a-report.json`, `agent-output/cmux-14-5/pilot-report.json`, `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-phase-a-fail`, and `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-fail`; verification closed `5/5`. The manifest is informational only. These files and all other attempt-1 reports, markers, checkpoints, logs, and output names remain immutable and cannot satisfy attempt-2 dependencies. Attempt 2 is the fixed `ds4-segmented-pilot-attempt-2` namespace:

- Phase A2 output `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a`, log/report `agent-output/cmux-14-5-attempt-2/phase-a-{log.txt,report.json}`, markers `.ds4-segmented-pilot-attempt-2-phase-a-{ok,fail}`.
- Phase B2 output `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-b`, resume `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/0000002_adapters.safetensors`, log/report `agent-output/cmux-14-5-attempt-2/phase-b-{log.txt,report.json}`, markers `.ds4-segmented-pilot-attempt-2-phase-b-{ok,fail}`.
- Final report `agent-output/cmux-14-5-attempt-2/pilot-report.json`, markers `.ds4-segmented-pilot-attempt-2-{ok,fail}`.

A2 launch checks all attempt-2 destinations before log/output mutation; B2 first strict-reads and validates the canonical A2 report and OK marker through the shared `validate_canonical_attempt2_report()` validator, requiring exact effective pins/command, immutable identity/resource schema, complete provider/update/validation/global evidence, checkpoint/config artifact cardinality, every output/log/report/marker binding, exact historical manifest, contract digest, active log binding, and resume file SHA/canonical tensor digest. It then checks all B2/final destinations absent. The attempt-2 contract binds every exact namespace path, effective pin set, immutable identity, active log, and full attempt-1 manifest. The old attempt-1 pilot names are retired from runnable catalogs; attempt 2 remains explicit and non-default. The catalog wrapper performs the non-mutating launch check, then `set -o noclobber; exec 3>"$LOG"; set +o noclobber`, runs with `--log-fd 3`, attests `stat(log_path)==fstat(log_fd)` before and after lock acquisition, and sends tee only to `/dev/fd/3`. No truncation, cleanup, overwrite, suffix allocation, retry, fallback, or attempt 3 is permitted. A2 and B2 each have one immutable namespace and one authorized attempt only. A2/B2 budgets remain `2/2700s`, `1/1500s`, total `4200s`; retry/fallback remain `none`.

Path A remains permanently frozen/STOP. Real A2 requires fresh explicit operator authorization and visible execution. B2 requires completed, verified A2 evidence plus separate explicit operator authorization; A2 authorization never carries forward. Synthetic closure proves resource-observer behavior, namespace bindings, collision/no-write semantics, and consumer oracles only. It does not claim convergence, quality, optimizer/RNG/dataset-cursor/scheduler/global-step continuity, full-training readiness, or behavior beyond later-observed adapter-weight equality/continuity, bounded readiness, throughput, or real execution evidence.

## Story 14.5b — canonical MLX version source and A2 reauthorization

The authorized A2 invocation at `c910d1ba33912236ee87f3f9bdfb5b31edece6e7` exited `1` before opening the phase log with `pilot launch check failed closed: MLX version mismatch: None`. It performed zero training/provider calls and zero optimizer updates, created none of the fixed A2/final destinations, and consumed its authorization. `agent-output/cmux-14-5-attempt-2/phase-a2-prelog-failure.md` remains immutable historical lineage.

`ds4_segmented_pilot.py::_runtime_preflight()` must import `mlx` as mandatory runtime-availability evidence but must never use `mlx.__version__` for admission or identity. The sole installed-version source is `importlib.metadata.version("mlx")`; admission requires an unmodified string exactly equal to `0.31.2`, and immutable identity `mlx_version` records that metadata value. Missing distribution metadata, metadata exceptions, empty or non-string results, nearby versions, local/dev suffixes, and newer versions all fail closed. No module-attribute, MLX-LM, package-map, environment, configuration, shell, `pip`, alternate-distribution, normalization, coercion, or compatible-range fallback is permitted.

The still-empty fixed `ds4-segmented-pilot-attempt-2` artifact namespace remains canonical; Story 14.5b does not allocate attempt 3 or change catalog paths, reports, markers, identity schema, digests, budgets, cardinality, logging, collision, retry, fallback, or training semantics. The failed invocation remains a distinct consumed authorization event. Any future A2 invocation requires the repaired exact revision, tracked synthetic tests, canonical regression GREEN, independent Reviewer PASS, Test Manager GREEN, immediate proof that every fixed A2/final destination remains absent and attempt-1 evidence remains immutable, then fresh explicit operator authorization bound to one exact command and one visible invocation. Any exit consumes that authorization. A collision or historical-manifest drift is STOP with no cleanup, deletion, rename, overwrite, fallback, suffix allocation, or automatic retry. B2 remains blocked until exact A2 success evidence is independently verified and separately authorized.
