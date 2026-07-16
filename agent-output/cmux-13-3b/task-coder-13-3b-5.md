# Story 13.3b-5 — Coder (smoke-train run + tokenizer chat_template wiring fix)

## Slice context
- **Story**: 13.3b-5 — MILESTONE: first actual `mlx_lm.lora --train` run on real 43-layer DeepSeek V4 with nn port + model-4bit.
- **Predecessor**: 13.3b-4b (`c1d3bf0`) produced on-disk `model-4bit/`. BA re-pin (`requirements-13-3b-5.md`) = UNBLOCKED-GO.
- **First supervisor run** (10:46-10:47, rc=1) hit a **tokenizer wiring gap** (NOT a model/nn-port bug). This Coder slice fixes the wiring + re-runs + verifies AC.

## Diagnosis (done by supervisor — carry forward, don't re-derive)

### The crash
```
ValueError: Cannot use chat template functions because tokenizer.chat_template is not set
  and no template argument was passed!
File ".../mlx_lm/tuner/datasets.py", line 113, in process
    tokens = self.tokenizer.apply_chat_template(messages, tools=tools, return_dict=False)
```
Model loaded fine (`Trainable parameters: 0.004% (5.571M/154430.069M)` — LoRA over 154B base ✓). Crash is in dataset `process()`.

### Root cause
1. **mlx_lm 0.31.3 `CommaSeparated.process()`** (datasets.py:103-114) ALWAYS builds `messages=[{user: prompt},{assistant: completion}]` + calls `tokenizer.apply_chat_template(messages)`. NO raw-concatenation branch.
2. **Dataset** (`/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096/train.jsonl`) has `prompt`+`completion` where `prompt` is **ALREADY pre-templated** with DeepSeek chat tokens: `<｜begin▁of▁sentence｜><｜User｜>"ababcba" ...<｜Assistant｜>`. `completion` is raw assistant response. Built by THIS project (Epic 2, `docs/backlog.md`) for a raw-concatenation pipeline.
3. **No DS4 tokenizer in the system has a `chat_template`** — verified: original HF snapshot (`/Volumes/Data NVME/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/553034d7...`), `hf-f8shim`, and `model-4bit` ALL have `chat_template=ABSENT`. DeepSeek ships without one (relies on pre-templated prompts).

### The fix — passthrough chat_template (semantically CORRECT, not a hack)
Add a passthrough/identity `chat_template` to `model-4bit/tokenizer_config.json`:
```jinja
{{ messages | map(attribute='content') | join('') }}
```

**Why correct**: the `prompt` is ALREADY chat-framed (has `<｜begin▁of▁sentence｜><｜User｜>...<｜Assistant｜>`). Passthrough concatenation of message contents produces the exact target token stream:
- **Full call** (`messages=[user,assistant]`): `prompt + completion` = `<｜begin▁of▁sentence｜><｜User｜>...<｜Assistant｜>` + `Let's think...` ✓ correct token stream (chat framing present, then completion).
- **mask_prompt offset call** (`messages[:-1]=[user]`, `add_generation_prompt=True`): passthrough emits just `prompt` (ignores `add_generation_prompt` because the prompt already ends with `<｜Assistant｜>`). `offset = len(prompt tokens)` ✓ correct loss masking (loss only on completion).

This is a **data patch** (config field on disk, like 13.3b-4b §2.0), NOT a FROZEN/nn-port/shim code edit. Model weights untouched. FROZEN `deepseek_v4.py` untouched.

## SCOPE

### 1. Reproducibility script (tracked) — NEW `scripts/apply_passthrough_chat_template.py`
Idempotent script that patches a `tokenizer_config.json` with the passthrough `chat_template` (so the fix is reproducible from a fresh clone / re-convert). Mirrors the §2.0 config-flip pattern from 13.3b-4b. Args: `--tokenizer-config <path>` (default: `$MLX_WORK/model-4bit/tokenizer_config.json`). Idempotent: if `chat_template` already present + matches, no-op.

### 2. Apply the patch to `model-4bit/tokenizer_config.json`
Run the script (or inline python) on `/Volumes/Data NVME/mlx-ft/ds4/model-4bit/tokenizer_config.json`. Verify `chat_template` field is now present.

### 3. Re-run smoke-train (AC §3 verification)
```bash
cd /Volumes/Data NVME/mlx-ft/ds4 && unset SSLKEYLOGFILE && . .venv/bin/activate
mlx_lm.lora --config lora-config.json --model model-4bit --train \
  --data /Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096 \
  --adapter-path adapters-smoke --fine-tune-type lora --iters 20 \
  --batch-size 1 --learning-rate 1e-5 --max-seq-length 4096 --mask-prompt --grad-checkpoint
```
(Or via orchestrator: `python3 scripts/finetune_ds4.py run-command smoke-train --execute --yes` from project root with env set.)

Capture the log to `agent-output/cmux-13-3b/coder-13-3b-5-smoke-train.log`.

## AC (backward GREEN LOCKED 13.3a-3 BA Q4 + 13.3b-5 additions)
Run is GREEN iff ALL hold:
- (a) loss finite (no NaN/Inf) across iters.
- (b) grad tree non-empty.
- (c) every LoRA leaf grad finite.
- (d) ≥1 LoRA grad non-zero.
- (e) iter 20 completes — no OOM, no crash.
- (f) adapter saves to disk: `/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke/adapters.safetensors`.

For AC (b)/(c)/(d): if `mlx_lm.lora` doesn't print grad-tree summary by default, write a tiny post-run probe that loads the saved adapter + asserts the LoRA leaves have finite non-zero values. OR inspect the mlx_lm training callback output (it prints loss per `--steps-per-report`; default 10 → 2 reports for 20 iters). Loss decreasing or stable + finite is the primary signal; adapter saved + loadable + LoRA params finite/non-zero is the concrete proof.

## Q5 RAM fallback (BA §7)
`--max-seq-length 4096` on 43 layers real dims on 512GB M3 Ultra. If OOM → fall back to `smoke-train-2048` (orchestrator command at `finetune_ds4.py:950`, identical but `--max-seq-length 2048` + `adapters-smoke-2048/`). If BOTH OOM → STOP, escalate Architect.

## MUST NOT
- Edit FROZEN `deepseek_v4.py` (ADR 0017/0024/0026).
- Edit nn-port `deepseek_v4_nn.py` or `deepseek_v4_nn_remap.py` (ADR 0025 — not needed for this fix).
- Edit `shim_ds4_safetensors.py` (BA Q1 from 13.3b-4 — the rename table is separate).
- Run `git commit` (commit-gating — supervisor commits post-double-green). `git add` (stage) the new script + notes only.
- Touch the model-4bit weight shards (only `tokenizer_config.json` gets the patch — it's on NVME, not git).
- Introduce C++.

## STOP-ESCALATE
Write `agent-output/cmux-13-3b/coder-13-3b-5-stop.md` if:
- **Smoke-train exposes a REAL model/nn-port bug** on real 43-layer bytes during `--train` (forward NaN on a real cr=4/cr=128 layer that 13.3b-4b's reload-forward seq_len=8 missed, OR backward dies on the real FP4 expert path, OR strict-load fails on model-4bit nn keys, OR loss explodes to inf). Record failing iter + layer + error. Genuine correctness issue → escalate Architect/Coder. Do NOT patch tests or relax AC.
- Both 4096 AND 2048 OOM (escalate Architect for further seq-len reduction or activation-memory analysis).
- The passthrough chat_template produces a WRONG token stream (e.g. double-templating detected — prompt gets `<<｜begin▁of▁sentence｜><｜User｜>` twice) → STOP, escalate (the dataset may need re-formatting instead, which is a bigger slice).

**Do NOT STOP on further simple wiring gaps** (e.g. another missing config field) — fix them inline like this one (they're data/config patches, not model bugs).

## Pre-flight read
1. `agent-output/cmux-13-3b/requirements-13-3b-5.md` (BA re-pin SPEC — AC §3, dataset wiring §4, lora-config §5, Q5 RAM §7).
2. `agent-output/cmux-13-3b/ba-13-3b-5-stop.md` (first BA Q1 STOP — now resolved).
3. This task file (diagnosis above).
4. `/Volumes/Data NVME/mlx-ft/ds4/smoke-train-13-3b-5.log` (the first crashed run — full traceback).
5. `scripts/finetune_ds4.py:949` (smoke-train command) + `:950` (smoke-train-2048 fallback) + `:4498-4505` (validate gates).
6. `/Volumes/Data NVME/mlx-ft/ds4/model-4bit/tokenizer_config.json` (the file to patch).
7. `/Volumes/Data NVME/mlx-ft/ds4/lora-config.json` (rank=8, keys verified by BA).
8. `docs/backlog.md` Epic 2 (dataset provenance — pre-templated prompts intentional).
9. ADR 0024 (FP4 dequant — live training-forward path), ADR 0025 (nn-port ownership), ADR 0027 (load-side remap).
10. `AGENTS.md` (tracking hygiene HARD RULE `58194a9` + commit-gating).

## Build
`source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first). The smoke-train run may take real time (20 iters × 43 layers × seq 4096 on real dims). Use a background launch (nohup + log) + short bounded bash polls (NOT `ctx_execute` for long polls — lesson learned). caffeinate -i to prevent sleep. Caveman ultra default; byte-exact exempt.

## Deliverables
1. NEW `scripts/apply_passthrough_chat_template.py` (tracked, idempotent reproducibility script).
2. Patched `/Volumes/Data NVME/mlx-ft/ds4/model-4bit/tokenizer_config.json` (on NVME — `chat_template` field present).
3. `/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke/adapters.safetensors` (the trained adapter — AC f).
4. `agent-output/cmux-13-3b/coder-13-3b-5-smoke-train.log` (full training log — loss trace, iters).
5. `agent-output/cmux-13-3b/coder-13-3b-5-notes.md` (diagnosis recap + fix applied + AC §3 (a)-(f) verdict with evidence: loss values, grad-tree summary, adapter saved confirmation, seq-len used 4096 or 2048).
6. `.cmux-status/coder.done` marker + in-pane JSON `{"status":"ok","role":"Coder"}`.
7. `git add` (stage, NOT commit): the new script + coder notes.

## Regression
- FROZEN `deepseek_v4.py` byte-intact (source-hash unchanged `96c39168...`).
- nn-port `deepseek_v4_nn.py` + `deepseek_v4_nn_remap.py` byte-intact (untouched by this fix — only tokenizer_config.json + new script).
- Full suite still 578 / 13 / 0 RED (the new script is additive; no test changes expected unless you add a tiny test for the passthrough template — optional, only if clean).
- model-4bit weight shards untouched (only tokenizer_config.json patched).

Echo `{"status":"ok","role":"Coder"}` in this pane only. BEGIN NOW.
