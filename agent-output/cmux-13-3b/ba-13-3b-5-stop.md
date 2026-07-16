# BA STOP-ESCALATE — Story 13.3b-5 (smoke-train 43-layer real)

**Verdict:** STOP. Q1 trips the documented split condition: on-disk remapped
`model-4bit/` is REQUIRED for smoke-train, but no writer exists. Producing it is
Coder/Architect scope, NOT BA. BA cannot pin a runnable smoke-train AC until a
code slice produces `model-4bit/`.

Role: BA. Status: error/blocked. No `ba.done` written (STOP protocol).

---

## The blocker (Q1) — verified end-to-end

smoke-train command (`finetune_ds4.py:949`):
```
mlx_lm.lora --config lora-config.json --model <mlx_work>/model-4bit --train \
  --data <split_dir> --adapter-path <mlx_work>/adapters-smoke \
  --fine-tune-type lora --iters 20 --batch-size 1 --learning-rate 1e-5 \
  --max-seq-length 4096 --mask-prompt --grad-checkpoint
```
Needs on-disk `--model model-4bit`. Chain that should produce it is BROKEN:

1. `model-4bit/` ABSENT (`/Volumes/Data NVME/mlx-ft/ds4/model-4bit` not present).
2. Only producer in orchestrator = `convert-shimmed`:
   `mlx_lm.convert --model hf-f8shim -q --mlx-path model-4bit`.
3. `mlx_lm.convert` → `load_model` → `model.load_weights(list(weights.items()), strict=True)`
   (verified in installed `mlx_lm.utils.load_model`).
4. `hf-f8shim/` keys are CKPT-NATIVE (verified from index headers):
   - `embed.weight`, `head.weight`, `hc_head_{fn,base,scale}`
   - `layers.N.attn.wq_a.{weight,scale}`, `wq_b`, `wkv`, `wo_a`, `wo_b`
   - `layers.N.ffn.experts.M.w{1,2,3}.{weight,scale}` (per-expert, NOT stacked)
   - `layers.N.attn.compressor.ape` (untransposed), redundant `.scale` sidecars
   - `config.json` `model_type=deepseek_v4`, 43 layers, hidden=4096, head_dim=512,
     n_routed_experts=256, q_lora_rank=1024
5. nn `deepseek_v4_nn.Model.load_weights` = INHERITED nn.Module strict. Only
   `sanitize` runs = mtp-strip pass-through (`deepseek_v4_nn.py:547`,
   `sanitize_weights` `deepseek_v4.py:2162`). NO key rename.
6. Strict load of ckpt-native keys into nn param tree
   (`model.layers.N.self_attn.q_a_proj.weight`, stacked experts) → FAILS, every
   key mismatches.
7. The rename = `remap_weight_dict` in `scripts/remap_ds4_nn_weights.py:329`.
   It is IN-MEMORY only (returns a dict, no `save_file`/`mkdir`/`save_weights` —
   grep-confirmed) and is NOT wired into nn `Model.load_weights` nor into convert.
8. `remap_ds4_nn_weights.py` exposes ONLY the `plan` subcommand
   (`main:450`, `add_parser("plan")` `:454`). NO `apply`/`convert` subcommand that
   writes a remapped on-disk dir.

Predecessor 13.3b-4 delivered remap as IN-MEMORY load-verify only (coder r2 notes:
"single-shard / shape-only acceptable", `model.load_weights(list(...), strict=True)`
on a compacted 3-layer model). On-disk `model-4bit/` production was NOT in 13.3b-4
scope and is NOT BA scope for 13.3b-5.

**Conclusion:** there is NO in-memory path (nn Model has no remap hook) AND no
on-disk writer. model-4bit cannot be produced without a CODE change.

## Required split (Architect to arbitrate, Coder to implement) — pick ONE

- **Option A (convert-side, matches ADR 0025/0026 convert-only contract):** add
  an `apply`/`convert` subcommand to `scripts/remap_ds4_nn_weights.py` that reads
  `hf-f8shim`, runs `remap_weight_dict` over real payload, and WRITES a remapped
  on-disk dir (nn-port keys, `model_type=deepseek_v4_nn`, sidecars/index). Then
  `mlx_lm.convert -q --mlx-path model-4bit` quantizes it. NO FROZEN/nn-file edit.
- **Option B (load-side, nn-file edit, ADR 0025 territory):** wire
  `remap_weight_dict` into `deepseek_v4_nn.Model.load_weights` (mirror frozen
  `deepseek_v4.Model.load_weights:2367` which already calls
  `remap_shimmed_ckpt_keys`). Then `mlx_lm.convert` of hf-f8shim succeeds directly.

A is the convert-only contract (ADR 0026 §5, 13.3b-4 lineage); B is an nn-port edit
(ADR 0025). Architect chooses; this is a new coding slice (13.3b-4b or Coder
prerequisite folded into 13.3b-5). BA does NOT design it.

---

## Other open questions — RESOLVED by BA (carry forward; not blockers)

- **Q2 training data — RESOLVED, NOT a blocker.** Full dataset exists:
  `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096/`
  train=15170 / valid=819 / test=824, `manifest.json` (`split_counts` +
  `final_unique_records=16813`) + `meta/records-meta.jsonl` present.
  `validate_dataset(args)` (`finetune_ds4.py:813`, gate `:4500`) PASSES. NO
  synthesis needed. iters=20 smoke draws from the existing train split.
- **Q3 orchestrator vs direct — RESOLVED.** Both reference on-disk model-4bit;
  ties back to Q1. Once model-4bit exists, USE orchestrator `smoke-train` (it
  carries `validate_mlx_lora_target_gate` + `validate_dataset` + model-4bit-dir
  gates `:4498-4505`). No direct call.
- **Q4 lora keys ↔ nn leaf names — RESOLVED, NO divergence, NOT a blocker.**
  lora-config keys `self_attn.q_a_proj` / `q_b_proj` / `kv_proj` MATCH nn port
  leaf names exactly: `deepseek_v4_nn.py:237` `self.q_a_proj`, `:239` `self.q_b_proj`,
  `:240` `self.kv_proj`, exposed under `self.self_attn = AttentionNN` (`:485`).
  No remap of lora keys, no FROZEN/nn edit for targeting.
- **Q5 RAM at 4096 — DEFERRED to run, NOT a BA blocker now.** Cannot measure
  without model-4bit. `smoke-train-2048` fallback (`max-seq-length 2048`) exists
  in the orchestrator if 4096 OOMs. Run-slice STOP-rule (e) (OOM → escalate
  Architect for seq-length reduction) preserved.

## PINNED AC (for the eventual smoke-train slice, once model-4bit exists)

Backward AC GREEN (13.3a-3 BA Q4 LOCKED): (a) loss finite; (b) grad tree
non-empty; (c) every LoRA leaf grad finite; (d) ≥1 LoRA grad non-zero. PLUS
(e) iter 20 completes, no OOM, no crash; (f) adapter saves to disk
(`adapters-smoke/adapters.safetensors`).

STOP-RULE for the run (unchanged): real-bytes bug in nn port/remap (forward NaN
on a real cr=4/cr=128 layer, backward dies on real FP4 expert path, or strict-load
fails on full 43-layer ckpt) → STOP, record failing layer + error, escalate
Architect/Coder = genuine correctness issue, NOT a test fix.

## Recon evidence index (paths:lines)

- `scripts/finetune_ds4.py:949` smoke-train; `:4498-4505` gates; `:813` validate_dataset.
- `scripts/remap_ds4_nn_weights.py:329` remap_weight_dict (in-memory); `:450/:454` plan-only.
- `deepseek_v4_nn.py:237/239/240/485` leaf names; `:547` sanitize (mtp-only).
- `deepseek_v4.py:2162` sanitize_weights; `:2367` frozen remap_shimmed_ckpt_keys precedent.
- `mlx_lm.utils.load_model` → `model.load_weights(..., strict=True)` (installed, verified).
- `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json` + index (ckpt-native keys).
- `/Volumes/Data NVME/mlx-ft/ds4/lora-config.json` + `lora-targets.json`.
- dataset dir above; `model-4bit/` ABSENT.

## Ask of supervisor

Route a Coder/Architect slice to produce on-disk `model-4bit/` (Option A or B per
Architect). After it lands, re-run this BA to pin the runnable smoke-train AC and
emit `ba.done`. BA scope for 13.3b-5 is otherwise fully resolved (Q2/Q3/Q4 closed,
Q5 deferred to run).
