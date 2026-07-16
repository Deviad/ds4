# Story 13.3b-5b — Coder resume (clean-resource 4096 LoRA smoke)

## Goal
Complete the interrupted 20-iteration real-model LoRA smoke after the user finished the unrelated GLM-5.2 quantization. Prove the backward-safe CSA fix on clean resources, finish evidence/notes, and stage this slice without committing.

## User model/budget override
Use the model already selected in this Coder pane: `opencode-go/kimi-k2.7`. The user is out of budget on the previously documented alternatives. Do not switch/relaunch to Codex, Neuralwatt, or Anthropic.

## Context
- Read the original task first: `agent-output/cmux-13-3b/task-coder-13-3b-5b.md`.
- Read Architect spec: `agent-output/cmux-13-3b/architecture-13-3b-5b-backward-safe-blockbias.md`.
- The implementation is already present locally:
  - `_csa_block_bias_mlx:902` wraps scatter indices in `mx.stop_gradient`.
  - vendor SHA256 is `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b`.
  - all three sha16 pins are `5e11a9c4d82aeb24`.
  - ADR 0026 Amendment 1 exists.
  - passthrough chat template exists in `model-4bit/tokenizer_config.json`.
- Earlier clean forward completed with finite `Val loss 19.116`.
- Earlier backward failed before the fix with `[scatter_axis] Cannot calculate VJP with respect to indices`.
- Post-fix 4096 run reached finite validation then OOMed only because unrelated GLM quantization was consuming RAM. The user stopped the 2048 fallback. Do not classify either interruption as code failure.
- GLM quantization is now complete at `/Volumes/Data NVME/GLM-5.2-GGUF/GLM-5.2-D5-IQ2S-Q2K-last6-IQ4NL-original-imatrix-expert` (247GB, 33 shards); no process uses it.
- Current preflight: no GLM/MLX heavy process; >300GiB free RAM; 182GiB free on `/Volumes/Data NVME`.
- Supervisor preflight tests just passed: 50 passed / 2 skipped across CSA parity + hash/reference tests; direct `_csa_block_bias_mlx` VJP probe returned finite non-zero gradients.

## Resource preflight — mandatory immediately before launch
1. Confirm no competing `mlx_lm`, quantization, conversion, DeepSeek, or GLM model process.
2. Confirm at least 250GiB free/reclaimable RAM and at least 20GiB free on `/Volumes/Data NVME`.
3. Do not load or run the GLM artifact while DS4 training runs.
4. Never bypass an instance lock. Never kill unrelated work without user authorization.
5. If contention reappears, pause and report `PENDING-RESOURCE`; do not call it a regression.

## Clean stale run artifacts
Preserve prior logs as historical evidence. Before the new run:
- Rename existing `coder-13-3b-5b-smoke-train.log` to `coder-13-3b-5b-smoke-train-resource-contention.log` if the archive does not already exist.
- Remove only incomplete adapter output directories:
  - `/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke/`
  - `/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke-2048/`
- Remove stale `.pid`/`.rc` files for 13.3b-5b.
- Do NOT touch any `model-4bit` shard or dataset file.

## Run
Launch the canonical 4096 command through the existing wrapper/orchestrator:

```bash
cd /Users/spotted/projects/ds4-finetuning
unset SSLKEYLOGFILE
nohup bash agent-output/cmux-13-3b/run-coder-13-3b-5b-smoke-train.sh \
  > agent-output/cmux-13-3b/coder-13-3b-5b-smoke-train.log 2>&1 &
echo $! > agent-output/cmux-13-3b/coder-13-3b-5b-smoke-train.pid
caffeinate -i -w "$(cat agent-output/cmux-13-3b/coder-13-3b-5b-smoke-train.pid)" &
```

Poll with short bounded checks. Long runtime is expected. Do not use context-mode or one multi-hour blocking tool call.

If the 4096 run OOMs under verified clean resources, record memory state and run the existing 2048 fallback once. If 2048 also OOMs, STOP-ESCALATE.

## Acceptance criteria
All must hold:
1. Focused/full regression suite remains green; report exact counts. Baseline before this resume: focused 50 passed / 2 skipped; prior full baseline 578 passed / 13 skipped / 0 failed.
2. Smoke training reports finite validation and training losses.
3. First backward/optimizer step completes without `[scatter_axis]` VJP failure.
4. Iteration 20 completes without crash/OOM.
5. Gradient tree is non-empty; all LoRA leaves finite; at least one LoRA gradient/update non-zero. Use training evidence plus a saved-adapter probe if trainer does not expose gradients.
6. `/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke/adapters.safetensors` exists, is loadable, all tensors finite, and at least one tensor non-zero.
7. Vendor hash/pins remain exactly as specified; old sha16 `96c39168c78e5fd9` has zero tracked code/test hits.
8. Every test file cited in the verdict is tracked (`git ls-files`).

## Scope and staging
Do not make additional production edits unless a new genuine blocker requires STOP-ESCALATE.

Stage ONLY slice-relevant files; do not stage unrelated dirty/untracked files:
- `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`
- `python-envs/mlx/src/ds4_ft_mlx/real_forward_intermediate_dump.py`
- `tests/test_numpy_real_forward_reference_composition.py`
- `tests/test_deepseek_v4_real_config_reference_forward.py`
- `docs/adr/0026-csa-hca-indexer-mlx-real-port.md`
- `scripts/apply_passthrough_chat_template.py`
- `agent-output/cmux-13-3b/coder-13-3b-5b-notes.md`
- canonical smoke log/RC evidence if required by existing project convention

Do NOT commit. Supervisor commits only after Reviewer PASS + Test Manager GREEN.

## STOP-ESCALATE
Write `agent-output/cmux-13-3b/coder-13-3b-5b-stop.md` and emit error JSON if:
- a different autograd/correctness failure appears after backward begins;
- finite-loss/gradient/adapter AC fails;
- both 4096 and 2048 OOM under clean resources;
- required staging would include an untracked baseline test that cannot legitimately be tracked;
- any change outside Architect-sanctioned scope is required.

## Deliverables
- `agent-output/cmux-13-3b/coder-13-3b-5b-smoke-train.log`
- successful adapter under `/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke/`
- `agent-output/cmux-13-3b/coder-13-3b-5b-notes.md` with per-AC evidence
- slice-relevant files staged, no commit
- `.cmux-status/coder.done`
- terminal JSON in this pane only: `{"status":"ok","role":"Coder"}` or error JSON
