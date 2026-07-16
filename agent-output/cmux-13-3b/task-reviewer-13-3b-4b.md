# Story 13.3b-4b — Reviewer (8-axis review of Coder model-4bit production)

## Slice context
- **Story**: 13.3b-4b — model-4bit production (load-side remap into nn `Model.load_weights`, Architect Option B). Prerequisite before 13.3b-5 smoke-train.
- **Coder done**: 10:03. HEAD still `fecc337` (NO commit per commit-gating). Review staged work via `git diff --cached`.
- **Architect SPEC**: `agent-output/cmux-13-3b/architecture-13-3b-5-model-4bit-path.md` (§1-§6).

## What Coder staged (review via `git diff --cached`)
```
.cmake-status/coder.done
agent-output/cmux-13-3b/coder-13-3b-4b-notes.md          (131L)
docs/adr/0027-load-side-ckpt-native-to-nn-remap.md       (41L, NEW ADR)
python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_nn_remap.py    (228L, NEW extracted remap core)
python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py (58L added)
scripts/remap_ds4_nn_weights.py                           (222L rewired)
tests/test_deepseek_v4_attention_parity.py               (1647L, NEWLY-TRACKED)
tests/test_deepseek_v4_checkpoint.py                     (1179L, NEWLY-TRACKED)
tests/test_deepseek_v4_dequant_parity.py                 (1238L, NEWLY-TRACKED)
tests/test_deepseek_v4_moe_parity.py                     (508L, NEWLY-TRACKED)
tests/test_deepseek_v4_nn_load_weights_override.py       (108L, NEW AC3)
tests/test_deepseek_v4_nn_remap_module.py                (51L, NEW AC2)
tests/test_deepseek_v4_real_config_reference_forward.py  (2L, tiny update)
tests/test_ds4_gguf_base_smoke.py                        (238L, NEWLY-TRACKED)
```

## 8-axis review

### Axis 1 — Scope / blast radius
- ONLY nn-port + convert-side + tests + ADR staged. NO edits to FROZEN `deepseek_v4.py`.
- `git diff --cached -- python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` → MUST be empty.
- 14 files staged total. Confirm matches Architect §3 scope guard (nn-port-only edit = ADR 0025 sanctioned; NEW module + re-import + data patch; ADR 0027).

### Axis 2 — FROZEN byte-intactness (CRITICAL)
- `deepseek_v4.py` source-hash + AST UNCHANGED. Coder claims sha256 `96c39168c78e5fd9158039707334f92206e8bb39f7e718e0d9064a7356809a44`. Verify independently.
- `git diff --cached -- <FROZEN path>` empty (AGENTS.md: also source-hash + AST, not git diff alone).
- FROZEN `load_weights:2367`, `remap_shimmed_ckpt_keys`, `_attention_mlx`, FP4 dequant (`dequant:322/375/291`), 9 forbidden OUR-Python symbols, all 13.3b-1/2a/2b/3 helpers — all byte-intact.

### Axis 3 — Correctness (Architect §2)
- **§2.0 config flip**: `hf-f8shim/config.json` `model_type` is now `deepseek_v4_nn` (data patch, not code — verify via the file, not git).
- **§2.1 extract**: `deepseek_v4_nn_remap.py` contains `KeyRemap`, `_DROP_PATTERNS`, `_DIRECT_RULES`, `_EXPERT_RULE`, regex consts, `remap_key`, `_zero_bias_keys`, `_validate_expert_sources`, `_get_mx`, `_config_int`, `_convert_value`, `remap_weight_dict`. `scripts/remap_ds4_nn_weights.py` re-imports them. **Single source of truth preserved**: AC1 r2 1460-key match must still hold (`expected_params 1460, remapped_specs 1460, missing 0, extra 0`).
- **§2.2 load_weights override**: present in `deepseek_v4_nn.Model` (`:528`). Mirrors frozen `deepseek_v4.py:2367`. **IDEMPOTENT GATE CRITICAL**: verify `is_native = any(not k.startswith(("model.", "lm_head.")) for k in mapping)` is present + correct. Native ckpt keys (embed./head./norm./hc_head_./layers.) → remap fires. Already-nn keys (model./lm_head.) → bypass (idempotent model-4bit reload).
- **§2.3 convert consumption**: no convert-side code change. `mlx_lm.convert -q` does everything. Verify `DeepseekV4FP4Experts` (`:342`) still has NO `to_quantized` → experts SKIPPED by `quantize_model` → stay FP4 (ADR 0024).

### Axis 4 — AC4 convert result (CRITICAL — FP4 expert preservation)
- `model-4bit/config.json` `model_type=deepseek_v4_nn` + `quantization: {group_size:64, bits:4, mode:affine}`.
- **FP4 experts PRESERVED**: expert `w1_weight` U8 `(256,2048,2048)` + `w1_scale` BF16 `(256,2048,128)`. Expert leaves MUST NOT have `.scales`/`.biases` (those are 4-bit nn.Linear artifacts). Coder claims: expert keys 258, expert `.scales`/`.biases` = 0.
- attn `nn.Linear` quantized 4-bit: `q_a_proj.weight` U32 `(1024,512)` + `q_a_proj.scales`/`.biases` BF16 `(1024,64)`. Coder claims attn quant scales/biases = 762.
- **If experts re-quantized** (expert `.scales`/`.biases` present) → ADR 0024 violation → BLOCKED.

### Axis 5 — AC2/AC3 legitimacy / anti-tautology
- **AC2** `test_deepseek_v4_nn_remap_module.py`: import from package + `plan` subcommand still works + 1460-key match. Anti-circular: expected from probed `model.parameters()`, not hand-written.
- **AC3** `test_deepseek_v4_nn_load_weights_override.py`: native-key load remaps (1460 keys, 0 missing/extra) + nn-key load bypasses (idempotent). Verify the test asserts BOTH the remap path AND the bypass path (idempotency is the whole point of §2.2's gate).

### Axis 6 — WATCH-ITEM: 5 newly-tracked historical test files (HARD RULE `58194a9` scrutiny)
Coder staged 5 previously-untracked baseline test files (`test_deepseek_v4_attention_parity.py` 1647L, `test_deepseek_v4_checkpoint.py` 1179L, `test_deepseek_v4_dequant_parity.py` 1238L, `test_deepseek_v4_moe_parity.py` 508L, `test_ds4_gguf_base_smoke.py` 238L). Coder's notes: "Updated stale historical tests that asserted `/Volumes/Data NVME/mlx-ft/ds4/model-4bit` must be absent; they now accept absence OR a valid current `deepseek_v4_nn` 4-bit artifact."

**You MUST read the actual edits** (`git diff --cached` on each) and adjudicate: legitimate contract update (model-4bit now exists legitimately, so "must be absent" is genuinely stale) OR relaxation hiding a regression? This is the 13.3b-3 chain-of-custody gap that triggered HARD RULE `58194a9`. Untracked test files participating in the baseline MUST be tracked, AND any contract update MUST be diff-checked for legitimacy. If a "must be absent" assertion was relaxed to "valid OR absent" in a way that masks a real regression (e.g. accepts a corrupt model-4bit) → BLOCKED Axis 6.

### Axis 7 — WATCH-ITEM: AutoConfig registration (narrow tokenizer tolerance)
Coder added "a narrow optional `AutoConfig` registration in `deepseek_v4_nn.py` so `AutoTokenizer` tolerates MLX-only `model_type=deepseek_v4_nn` during `mlx_lm.convert`/`mlx_lm.load`" (part of the 58L nn-port addition). `git diff --cached python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py` and verify:
- The registration is NARROW (only config-level, discards DS4 layer/rope fields generic Transformers validation rejects).
- It does NOT touch FROZEN `deepseek_v4.py` (cross-check Axis 2).
- It does NOT alter the nn port's training behavior (it's tokenizer-config tolerance only).
- It does NOT silently bypass validation in a way that hides a real config bug.

If broad / touches FROZEN / alters training → BLOCKED Axis 7.

### Axis 8 — sha-pin cascade + regression
- sha-pin cascade: 13.3b-4b touches `deepseek_v4.py`? No (FROZEN byte-intact — see Axis 2). So cascade should be no-op. Verify no phantom/stale pin sites.
- Regression: full suite 578 / 13 / 0 RED (Coder claims; was 572 in 13.3b-4, +6 = 2 new tests + updated historical tests). No C++, no scope creep.

## Deliverables
1. `agent-output/cmux-13-3b/review-13-3b-4b.md` — 8-axis verdict with evidence per axis (Axis 4 + Axis 6 + Axis 7 are the focus — show FP4 preservation grep + the 5 historical-file diffs + AutoConfig diff).
2. `.cmux-status/reviewer.done` marker.
3. In-pane JSON `{"status":"ok","role":"Reviewer"}` in surface:84 ONLY.

## STOP-ESCALATE
Write `agent-output/cmux-13-3b/reviewer-13-3b-4b-blocked.md` if:
- Any axis RED (FROZEN edited, FP4 experts re-quantized, idempotency gate missing, AC tautology).
- The 5 historical test files have relaxations hiding regressions (Axis 6).
- The AutoConfig registration is broad / touches FROZEN / alters training (Axis 7).
- Coder committed (HEAD past `fecc337`).

## Pre-flight read
1. `agent-output/cmux-13-3b/coder-13-3b-4b-notes.md` (Coder summary).
2. `agent-output/cmux-13-3b/architecture-13-3b-5-model-4bit-path.md` (Architect §1-§6 SPEC).
3. `agent-output/cmux-13-3b/ba-13-3b-5-stop.md` (BA finding).
4. `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_nn_remap.py` (NEW module).
5. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py:528` (Model) + `load_weights` override + AutoConfig registration (the 58L).
6. `scripts/remap_ds4_nn_weights.py` (the re-import rewiring).
7. The 5 newly-tracked historical test files (full `git diff --cached` on each).
8. `AGENTS.md` (tracking hygiene HARD RULE `58194a9`).

## Build
`source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first). Use `ctx_execute`/`ctx_batch_execute` for AST/source-hash + git diff (bounded). Caveman ultra default; byte-exact exempt. BEGIN NOW. Echo `{"status":"ok","role":"Reviewer"}` in this pane only.
