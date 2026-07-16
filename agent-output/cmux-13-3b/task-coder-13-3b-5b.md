# Story 13.3b-5b — Coder (apply backward-safe block_bias fix + re-run smoke-train)

## Slice context
- **Story**: 13.3b-5b — backward-safe CSA block_bias (stop_gradient; ADR 0026 amendment).
- **Predecessor**: 13.3b-5 STOPPED legit (forward PROVEN on real 43-layer bytes `Iter 1: Val loss 19.116`; first backward step crashed `[scatter_axis] Cannot calculate VJP with respect to indices`).
- **Architect SPEC** (read in full): `agent-output/cmux-13-3b/architecture-13-3b-5b-backward-safe-blockbias.md`.
  - Decision: **Option A** — `scatter_indices = mx.stop_gradient(mx.expand_dims(safe_indices, 1))` at FROZEN `deepseek_v4.py:902`.
  - Probe-verified VJP-safe + byte-identical forward (block_bias output unchanged).
  - ADR 0026 Amendment 1 sanctions this ONE FROZEN body edit.

## The fix (EXACTLY per Architect §2 — additive wrap, ONE FROZEN line changed)

File: `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`
Function: `_csa_block_bias_mlx` (line `902`).

```python
# OLD (line 902)
    scatter_indices = mx.expand_dims(safe_indices, 1)
# NEW (with the inline comment per Architect §2 / AGENTS.md comment policy)
    # block_bias is a DISCRETE top-k routing mask (indices from argsort over frozen
    # indexer scores). MLX cannot VJP put_along_axis w.r.t. indices; stop_gradient is
    # forward-identity (parity preserved) and semantically correct — grads flow through
    # attention values, not the discrete position selection. (13.3b-5b / ADR 0026 amend.)
    scatter_indices = mx.stop_gradient(mx.expand_dims(safe_indices, 1))
```

Lines 903-904 unchanged. Nothing else in the function changes.

## ADR 0026 Amendment 1 — append to `docs/adr/0026-csa-hca-indexer-mlx-real-port.md`

Append Architect §3's amendment text verbatim as a new section in ADR 0026.

## sha-pin cascade — ADVANCE ALL sites in the SAME slice (Architect §4, AGENTS.md SLICE-INVARIANT)

Old sha16: `96c39168c78e5fd9` → New full sha256: `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b` (sha16 `5e11a9c4d82aeb24`).

Sites that MUST advance (literal pin sites — verify with a grep for old hash first):
1. `tests/test_numpy_real_forward_reference_composition.py:397` — `EXPECTED_SHAS[...deepseek_v4.py] = "5e11a9c4d82aeb24"`
2. `tests/test_deepseek_v4_real_config_reference_forward.py:383` — `EXPECTED_SHAS[...deepseek_v4.py] = "5e11a9c4d82aeb24"`
3. `python-envs/mlx/src/ds4_ft_mlx/real_forward_intermediate_dump.py:61` — `VENDOR_SHA256_16 = "5e11a9c4d82aeb24"` (and update the `# ADR 0026 ... baseline` comment to note the amendment)

Auto-advances (reads from `rfid.VENDOR_SHA256_16` — no literal edit): `tests/test_real_forward_intermediate_dump.py:83`.

ALSO: after appending ADR 0026 Amendment 1 (which changes 0026's bytes), grep ALL `EXPECTED_SHAS` entries for the ADR 0026 path; advance any that pin 0026 to its new hash. Verify with grep before declaring done.

After the slice: `grep -rn "96c39168c78e5fd9" tests/ scripts/ python-envs/mlx/src/` MUST return ZERO hits (old hash fully retired from tracked code/tests).

## Pre-flight probe (Architect §6 — may re-run to confirm VJP-safety before touching FROZEN)

Architect ALREADY ran this; you MAY re-run to confirm:
```python
import mlx.core as mx
cl=4; dtype=mx.float32
def make(stop):
    def fn(w):
        scores = w * mx.array([3.0,1.0,2.0,0.0]); idx = mx.argsort(-scores)[:2].astype(mx.int32).reshape(1,1,2)
        safe = mx.where(idx>=0, idx, mx.full(idx.shape, cl, dtype=idx.dtype)).astype(mx.int32)
        bb = mx.full((1,1,1,cl+1), -float("inf"), dtype=dtype) + mx.sum(w)
        sc = mx.expand_dims(safe,1)
        if stop: sc = mx.stop_gradient(sc)
        out = mx.put_along_axis(bb, sc, mx.zeros(sc.shape,dtype=dtype), axis=-1)[...,:cl]
        return mx.sum(mx.where(mx.isinf(out), mx.zeros_like(out), out))
    return fn
for s,l in [(False,"BASELINE"),(True,"OPT_A")]:
    try: mx.eval(mx.grad(make(s))(mx.array([0.1,0.2,0.3,0.4]))); print(l,"OK")
    except Exception as e: print(l,"FAIL",e)
# expect: BASELINE FAIL [scatter_axis]... ; OPT_A OK
```

## Re-run smoke-train (the actual PROOF — AC §5)
```bash
cd /Users/spotted/projects/ds4-finetuning
unset SSLKEYLOGFILE
. python-envs/mlx/.venv/bin/activate
python3 scripts/finetune_ds4.py run-command smoke-train --execute --yes
```
Emitted training command (same as 13.3b-5):
```bash
mlx_lm.lora --config /Volumes/Data NVME/mlx-ft/ds4/lora-config.json --model /Volumes/Data NVME/mlx-ft/ds4/model-4bit --train --data /Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096 --adapter-path /Volumes/Data NVME/mlx-ft/ds4/adapters-smoke --fine-tune-type lora --iters 20 --batch-size 1 --learning-rate 1e-5 --max-seq-length 4096 --mask-prompt --grad-checkpoint
```
Capture log to `agent-output/cmux-13-3b/coder-13-3b-5b-smoke-train.log`.

The passthrough `chat_template` patch from 13.3b-5 is already on `model-4bit/tokenizer_config.json` — no data-config step needed here, just re-run training.

## AC (forward parity + backward completes + smoke-train GREEN)
- (1) Edit lands: `_csa_block_bias_mlx:902` wrapped in `mx.stop_gradient`; lines 903-904 unchanged.
- (2) **Forward parity preserved** — 13.3b-2b AC2 (CSA-vs-torch) STILL byte-identical (stop_gradient is forward-identity). The 13.3b-2b test `test_csa_attention_real_mlx_parity.py` MUST still pass. Full suite stays 578/13/0 RED.
- (3) **Backward completes** — smoke-train reaches iter 20 with NO `[scatter_axis]` VJP error; first training-loss reported (`Iter N: Train loss ...`).
- (4) BA AC §3 (a)-(f) for 13.3b-5 now ACHIEVABLE: (a) loss finite, (b) grad tree non-empty, (c) all LoRA leaf grads finite, (d) ≥1 LoRA grad non-zero, (e) iter 20 no OOM/crash, (f) `adapters-smoke/adapters.safetensors` saved.
- (5) FROZEN source-hash = `5e11a9c4d82aeb24...` (probe-confirmed by Architect).
- (6) sha-pin cascade ADVANCED — all §4 sites pinned to new hash in same commit; `grep "96c39168c78e5fd9"` returns ZERO hits in tracked code/tests.

For AC (b)/(c)/(d): if `mlx_lm.lora` doesn't print a grad-tree summary, write a tiny post-run probe that loads the saved `adapters-smoke/adapters.safetensors` + asserts LoRA leaves have finite non-zero values. OR inspect the training callback output (loss per `steps-per-report`; default 10 → 2 reports for 20 iters). Loss decreasing/stable + finite is the primary signal; adapter saved + loadable + LoRA params finite/non-zero is the concrete proof.

## Q5 RAM fallback (still active)
If `--max-seq-length 4096` OOMs → fall back to `smoke-train-2048` (`finetune_ds4.py:950`, identical but `--max-seq-length 2048` + `adapters-smoke-2048/`). If BOTH OOM → STOP, escalate.

## MUST NOT
- Edit any OTHER line in `_csa_block_bias_mlx` (only `:902`).
- Edit any OTHER FROZEN primitive (ADR 0017's 9 forbidden OUR-Python symbols, the FROZEN dequant, etc.). ADR 0026 Amendment 1 authorizes ONLY the `:902` stop_gradient wrap — nothing else.
- Touch the §4 sha-pin sites incorrectly — advance them ALL to the new hash, no stale old hash left behind.
- Run `git commit` (commit-gating — supervisor commits post-double-green). `git add` (stage) the FROZEN edit + ADR amendment + sha-pin advances + new notes/log only.
- Touch model-4bit weight shards.
- Introduce C++.

## STOP-ESCALATE
Write `agent-output/cmux-13-3b/coder-13-3b-5b-stop.md` if:
- Smoke-train STILL crashes backward after the fix (e.g. a different autograd-unsafe op surfaces in the grad path that the Architect's grep missed). Record the new failing op + line. Genuine correctness issue → escalate Architect r2.
- Smoke-train forward goes NaN on a real layer (would indicate the stop_gradient somehow corrupted forward — IMPOSSIBLE per probe, but if it happens STOP immediately). Record failing layer + iter.
- Both 4096 AND 2048 OOM (escalate Architect).
- sha-pin cascade has a phantom site or stale hash you can't resolve (escalate Architect).

Do NOT STOP on the smoke-train taking real time (20 iters × 154B is slow — that's expected, not a failure).

## Pre-flight read
1. `agent-output/cmux-13-3b/architecture-13-3b-5b-backward-safe-blockbias.md` (Architect SPEC — the decision, exact edit, ADR amendment text, sha-pin cascade, probe code).
2. `agent-output/cmux-13-3b/coder-13-3b-5-stop.md` (predecessor STOP — diagnosis + val loss 19.116).
3. `agent-output/cmux-13-3b/coder-13-3b-5-smoke-train.log` (the crashed run traceback).
4. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` lines around `:888-904` (`_csa_block_bias_mlx` body — verify the exact site before editing).
5. `/Volumes/Data NVME/mlx-ft/ds4/model-4bit/tokenizer_config.json` (passthrough `chat_template` already present from 13.3b-5 — no action needed).
6. `docs/adr/0026-csa-hca-indexer-mlx-real-port.md` (append Amendment 1 here).
7. ADR 0024 (FP4 frozen — not in grad path), ADR 0025 (nn-port).
8. AGENTS.md (HARD RULE `58194a9` tracking hygiene + commit-gating + SLICE-INVARIANT sha-pin cascade).

## Build
`source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first). Smoke-train may take real time (~40min for val + iter 20 × ~per-batch — could be 1-3 hours total). Use background launch (nohup + log) + short bounded bash polls (NOT `ctx_execute` for long polls). `caffeinate -i` to prevent sleep. Caveman ultra default; byte-exact exempt.

## Deliverables
1. Modified `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` (the `:902` stop_gradient wrap + inline comment).
2. Modified `docs/adr/0026-csa-hca-indexer-mlx-real-port.md` (Amendment 1 appended).
3. Modified `python-envs/mlx/src/ds4_ft_mlx/real_forward_intermediate_dump.py:61` + `tests/test_numpy_real_forward_reference_composition.py:397` + `tests/test_deepseek_v4_real_config_reference_forward.py:383` (sha-pin cascade advances; any additional pin sites you discover via grep).
4. `/Volumes/Data NVME/mlx-ft/ds4/adapters-smoke/adapters.safetensors` (the trained adapter — AC f).
5. `agent-output/cmux-13-3b/coder-13-3b-5b-smoke-train.log` (full training log — loss trace, iters).
6. `agent-output/cmux-13-3b/coder-13-3b-5b-notes.md` (decision recap + edit applied + ADR amendment appended + sha-pin cascade advanced + AC verdict evidence: loss values, adapter saved, seq-len used, full suite still 578/13/0, `grep "96c39168c78e5fd9"` returns ZERO).
7. `.cmux-status/coder.done` marker + in-pane JSON `{"status":"ok","role":"Coder"}`.
8. `git add` (stage, NOT commit): the FROZEN edit + ADR amendment + sha-pin advances + notes/log.

## Regression
- Full suite STILL 578/13/0 (stop_gradient is forward-identity — 13.3b-2b AC2 CSA-vs-torch parity MUST still pass).
- ADR 0026 amendment appended idempotently.
- `grep -rn "96c39168c78e5fd9" tests/ scripts/ python-envs/mlx/src/` returns ZERO hits post-edit.

Echo `{"status":"ok","role":"Coder"}` in this pane only. BEGIN NOW.
