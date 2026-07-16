# Story 13.3b-4 — Coder (real ckpt convert / key remap + load)

## Slice context
- **Story**: 13.3b-4 — real ckpt remap script + load. Convert side ONLY.
- **Predecessor**: 13.3b-3 complete (HEAD `d1f1488`, +`d29e3e3` test-tracking fix, +`58194a9` rule docs). All 41/43 real layers run at real dims through `_attention_mlx` dispatch. nn port `deepseek_v4_nn.py` is wired (AttentionNN guard lifted + §4.1 submodules `AttentionCompressorNN` + `AttentionIndexerNN`).
- **13.3b-5 owns**: smoke-train. Do NOT touch convert in 13.3b-3-style slices.
- **Estimated**: 3 dev-days (Architect §5 + breakdown).
- **HEAD before this slice**: `58194a9`.
- **BA verdict**: GO. BA requirements-13-3b-4.md is the SPEC. BA resolved Q1-Q4 (see below).

## BA Q resolutions (from requirements-13-3b-4.md)

- **Q1 (NEW script vs extend shim)**: NEW `scripts/remap_ds4_nn_weights.py`. The existing `shim_ds4_safetensors.py` is a byte-level FP8→BF16 shard converter with `copy_sidecars:361`; remap is a higher-level transform (per-expert stack + scale-drop + `model.` prefix) operating on the ALREADY-shimmed ckpt. Mixing concerns would muddy the shim. NEW script.
- **Q2 (validation budget)**: 16GB RAM constraint — grading does NOT require a full 162GB / 46-shard / 69187-key materialized run. Shape-only + sub-checkpoint validation is ACCEPTABLE (mirror 13.3a-3 Q9). Mechanism: read safetensors **headers** (dtype+shape, zero tensor bytes — proven in §0.2) for the full key-set check; per-instance load+forward on a sub-checkpoint (1 CSA + 1 HCA + 1 sliding layer).
- **Q3 (STOP-rule)**: did NOT fire. Every nn param target has a ckpt source. Only un-consumed ckpt templates are DELIBERATE drops: `mtp.*` (sanitize strips) + FP8 `.scale` sidecars on BF16 attn-core + `indexer.wq_b` (§5.3 drop). If a NEW ckpt tensor with no nn target shows up during implementation → STOP (write `coder-13-3b-4-stop.md` with the exact shape) and escalate.
- **Q4 (nn param tree TARGET)**: BA probed `model.parameters()` from `deepseek_v4_nn.py`. The nn leaf set is the GROUND TRUTH the remap must produce. §5.1 table is the spec; nn tree is ground truth. Coder follows the nn tree. CRITICAL DIVERGENCE (see §0.4 below).

## ⚠️ §0.4 ape orientation — BA caught §5.2 prose STALE

**Architect §5.2 says**: "ckpt CSA/indexer `ape` is `(rate, 2*out_dim)`; nn port expects `(2*out_dim, rate)` → remap TRANSPOSES CSA + indexer ape; HCA NO transpose." That premise cites FROZEN `_csa_windowed_compressor_mlx:368` (`expected = (2*out_dim, rate)`).

**But the WIRED real helpers (13.3b-1/2a/2b) consume ape token-major, NO transpose:**
- `_csa_compressor_real_mlx:819` → expected `ape = (rate, 2*out_dim)` token-major, NO transpose
- `_indexer_mlx:653` → expected `indexer ape = (rate, 2*out_dim)`, NO transpose
- `_hca_compressor_mlx:773` → expected `ape = (rate, out_dim)`, NO transpose

**BA probe confirms:**
- nn leaves: `compressor.ape (4,1024)`, `indexer.compressor.ape (4,256)`, HCA `compressor.ape (128,512)` — ALL `(rate, ...)` token-major
- ckpt headers: CSA `compressor.ape [4,1024]`, CSA `indexer.compressor.ape [4,256]`, HCA `compressor.ape [128,512]` — IDENTICAL to nn leaves

**CONCLUSION: ape needs ZERO transpose for ALL three (CSA / indexer / HCA).** ckpt layout == nn-leaf layout == wired-real-path expectation. §5.2's transpose is a relic of the superseded FROZEN tiny helper. **Coder MUST copy ape verbatim (no `.T`).** Flagged for Architect doc-fix (NOT escalated — code is right, §5.2 prose is stale).

## Scope (convert-side ONLY — NO FROZEN/nn-port code edits)

### File 1 (NEW): `scripts/remap_ds4_nn_weights.py`
Implements the key-remap transform on an ALREADY-shimmed ckpt (BF16 attn + FP4 experts). Per Architect §5.1 + BA §0.4 corrections:
- `embed.weight → model.embed_tokens.weight`
- `head.weight → lm_head.weight`
- `hc_head_{fn,base,scale} → model.<hyperhead leaf>` (use actual 13.3a leaf names from nn tree)
- `layers.N.attn_norm.weight → model.layers.N.input_layernorm.weight`
- `layers.N.ffn_norm.weight → model.layers.N.post_attention_layernorm.weight`
- `layers.N.attn.wq_a.{weight,scale} → model.layers.N.self_attn.q_a_proj.weight` (drop `.scale` — BF16 §5.3)
- `layers.N.attn.wq_b.* → ...self_attn.q_b_proj.weight`
- `layers.N.attn.wkv.* → ...self_attn.kv_proj.weight`
- `layers.N.attn.wo_a.* → ...self_attn.o_a_proj.weight`
- `layers.N.attn.wo_b.* → ...self_attn.o_b_proj.weight`
- `layers.N.attn.q_norm.weight → ...self_attn.q_norm.weight`
- `layers.N.attn.kv_norm.weight → ...self_attn.kv_norm.weight`
- `layers.N.attn.attn_sink → ...self_attn.sinks`
- `layers.N.attn.compressor.wkv.weight → ...self_attn.compressor.wkv.weight`
- `layers.N.attn.compressor.wgate.weight → ...self_attn.compressor.wgate.weight`
- `layers.N.attn.compressor.ape → ...self_attn.compressor.ape` (**verbatim — NO transpose, BA §0.4**)
- `layers.N.attn.compressor.norm.weight → ...self_attn.compressor.norm.weight`
- `layers.N.attn.indexer.compressor.* → ...self_attn.indexer.compressor.*` (CSA only; ape verbatim NO transpose)
- `layers.N.attn.indexer.weights_proj.weight → ...self_attn.indexer.weights_proj.weight`
- `layers.N.attn.indexer.wq_b.* → ...self_attn.indexer.wq_b.weight` (drop `.scale`)
- `layers.N.hc_attn_{fn,base,scale} → model.layers.N.<hc attn leaf>` (13.3a leaf names from nn tree)
- `layers.N.hc_ffn_{fn,base,scale} → model.layers.N.<hc ffn leaf>`
- `layers.N.ffn.gate.weight → model.layers.N.mlp.gate.weight`
- `layers.N.ffn.gate.tid2eid → model.layers.N.mlp.gate.tid2eid` (hash layers 0,1,2 only)
- `layers.N.ffn.gate.bias → model.layers.N.mlp.gate.bias` (moe layers only)
- `layers.N.ffn.shared_experts.w1.* → ...shared_experts.gate_proj.weight`
- `layers.N.ffn.shared_experts.w3.* → ...shared_experts.up_proj.weight`
- `layers.N.ffn.shared_experts.w2.* → ...shared_experts.down_proj.weight` (w1=gate, w3=up, w2=down)
- `layers.N.ffn.experts.M.w{1,2,3}.{weight,scale} → STACK over M → model.layers.N.mlp.experts.w{1,2,3}_{weight,scale}` (shape `(n_routed, ...)`; routed experts STAY FP4 — uint8 weight + BF16 scale; `_dequantize_fp4` at forward, ADR 0024)
- `mtp.* → DROP` (sanitize strips; remap should not emit)

### File 2 (NEW tests, tracked from the start per AGENTS.md hard rule)
- `tests/test_remap_ds4_nn_weights_keyset.py` (AC1): remapped key set == nn `model.parameters()` key set EXACTLY (no missing/extra). Shape-only acceptable (header scan, zero tensor bytes) per BA Q2. Anti-circularity: the expected key-set comes from `model.parameters()` actually probed on the nn module (NOT a hand-written list), so the test cannot pass on a wrong remap by accident.
- `tests/test_remap_ds4_nn_weights_load.py` (AC2): load succeeds strict on a sub-checkpoint (1 real CSA layer + 1 real HCA layer + 1 sliding layer). Forward finite. Shape-only on full ckpt is OK; the sub-checkpoint load+forward is the real assertion.
- `tests/test_remap_ds4_nn_weights_regression.py` (AC3): tiny byte-identity regression — tiny CSA 11.14/11.15 still byte-identical pre/post remap (the remap script must not touch the tiny path; remap is real-ckpt-only).

### Files NOT to touch
- NO FROZEN edits (`deepseek_v4.py` FROZEN bodies stay byte-identical — `_csa_*`, `_hca_*`, `_indexer_*`, `_attention_mlx`, `sanitize_weights`, parity `Model`, dequant, 9 forbidden OUR-Python symbols, `_apply_rope_*`, `_rope_*_tables`, `_compress_rope_yarn_tail_tables_mlx`, all 13.3b-1/2a/2b helper bodies).
- NO nn-port edits (`deepseek_v4_nn.py` — ADR 0025 says nn-port edits are separate slices; 13.3b-4 is convert-side ONLY). The nn param tree is the TARGET; if you find nn leaves diverge from §5.1 in a way the remap can't accommodate → STOP (escalate, this is the Q3 STOP materializing).
- NO convert-in-shim edits (`shim_ds4_safetensors.py` stays byte-identical — BA Q1 chose NEW script).
- NO smoke-train (13.3b-5).

## AC (BA pinned)
- **AC1**: remapped key set == nn `model.parameters()` key set exactly (no missing/extra). Shape-only acceptable.
- **AC2**: load succeeds strict on sub-checkpoint (1 CSA + 1 HCA + 1 sliding); forward finite; one real CSA + one real HCA + one sliding layer load + forward finite.
- **AC3**: tiny byte-identity regression (tiny CSA 11.14/11.15 still byte-identical).
- **NO full 162GB run required for grading** (BA Q2 — sub-checkpoint / shape-only acceptable).

## ⚠️ HARD RULE — test file tracking (AGENTS.md, new rule committed `58194a9`)

Story 13.3b-3 had a chain-of-custody break: Coder modified `tests/test_deepseek_v4_mlx_port.py` (untracked) to update stale expectations so the suite passed, but left it uncommitted → the "569 passed / 0 RED" verdict was non-reproducible from a fresh clone, and Reviewer's "legitimate, not tautology" verdict was reached without a diff. **This MUST NOT recur.**

- Every NEW test file you write for 13.3b-4 (`test_remap_ds4_nn_weights_*.py`) MUST be `git add`ed (staged) so it's ready for the slice commit. **Detect-and-stage is mandatory.**
- If you modify ANY existing test file, run `git ls-files -- <path>` first; if untracked, `git add` it (stage) OR STOP-escalate if you believe it should stay untracked. **NEVER leave "local expectations updated" as untracked/unstaged files in the working tree.**
- In `coder-13-3b-4-notes.md`, DECLARE every file you modified/created (tracked + newly-tracked) with `git ls-files` confirmation.

## ⚠️ COMMIT GATING — do NOT commit in this slice

**Coder MUST NOT run `git commit`.** The slice commit lands only AFTER both Test Manager AND Reviewer return GREEN, executed by the supervisor. Coder's job is `git add` (stage) + `git ls-files` declare + notes; the supervisor commits once both gates pass.

- If Coder discovers a blocker mid-work → STOP-escalate (write `coder-13-3b-4-stop.md`), leave changes staged-but-uncommitted; supervisor decides.
- If Coder finishes → stage everything, write notes + `coder.done` + in-pane JSON. **Do NOT commit.** The supervisor will commit post-green-gates.
- Staged-but-uncommitted is the correct end state for Coder. Reviewer and Test Manager validate the staged work (they can read the working tree + staged diff via `git diff --cached`).

## STOP-ESCALATE
Write `agent-output/cmux-13-3b/coder-13-3b-4-stop.md` if:
- **S-orphan-tensor**: a ckpt tensor has no nn param target (BA Q3 STOP materializes) — record the exact ckpt tensor name + shape + the nn leaves it should map to but doesn't. Escalate for Architect § clarification (unknown arch piece).
- **S-nn-tree-divergence**: the nn param tree diverges from §5.1 in a way the remap can't accommodate without editing FROZEN or nn-port code → STOP (escalate; 13.3b-4 is convert-side ONLY).
- **S-frozen-edit**: the remap requires editing a FROZEN body (it shouldn't — remap is a standalone script) → STOP.
- **S-tiny-regression**: tiny CSA 11.14/11.15 byte-identity breaks (AC3 RED) — STOP, the remap is touching the tiny path.

## Pre-flight read
1. `agent-output/cmux-13-3b/requirements-13-3b-4.md` — BA SPEC (all Q1-Q4 + §0.4 ape correction + AC + STOP).
2. `agent-output/cmux-13-3b/architecture.md` §5.1-§5.4 + §0 nn tree. NOTE §5.2 ape-transpose prose is STALE (BA §0.4); follow the WIRED real helpers + nn tree.
3. `agent-output/cmux-13-3b/subslice-breakdown.md` 13.3b-4 row.
4. `scripts/shim_ds4_safetensors.py` (existing convert infra; `copy_sidecars:361` writes `model_type="deepseek_v4_nn"`; DO NOT edit this file — BA Q1 chose NEW script).
5. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py` (the nn param tree — TARGET key set; `Model.sanitize` at `:547`; `model_type="deepseek_v4_nn"` resolves via `mlx_lm.convert`).
6. `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json` (real config: 43 layers, `num_hash_layers=3`, `n_routed_experts=256`, `expert_dtype="fp4"`, `compress_ratios` array: 3×cr=0, 21×cr=4, 20×cr=128).
7. `agent-output/cmux-13-3b/requirements-13-3b-3.md` (predecessor context).
8. `docs/adr/0025-*.md` (nn file ownership — DO NOT edit nn-port) + `docs/adr/0024-*.md` (FP4 dequant — routed experts STAY FP4, `_dequantize_fp4` at forward).
9. `AGENTS.md` new "Tracking hygiene for test files participating in the baseline (HARD RULE)" section (committed `58194a9`).

## Build
`source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first). Use `ctx_execute` for the param-tree probe + header scans (keep large outputs out of context).

## Deliverables
1. `scripts/remap_ds4_nn_weights.py` (NEW).
2. 3 NEW tests (`test_remap_ds4_nn_weights_keyset.py`, `_load.py`, `_regression.py`) — ALL `git add`ed (staged, NOT committed).
3. `agent-output/cmux-13-3b/coder-13-3b-4-notes.md` (incl ape-verbatim decision, key-remap table coverage, sub-checkpoint validation choice, file-tracking declaration with `git ls-files` confirmation).
4. `.cmux-status/coder.done` + in-pane JSON `{"status":"ok","role":"Coder"}` in surface:83 ONLY.
5. **All changes `git add`-staged but UNCOMMITTED.** The slice commit is the supervisor's job, executed only after Test Manager AND Reviewer both return GREEN. Coder does NOT commit.

## Regression target
Full suite: 569+new / 13 / 0 introduced RED. FROZEN bodies + nn-port + shim byte-intact (source-hash + AST, NOT git diff for untracked-baseline files — though `test_deepseek_v4_mlx_port.py` is NOW tracked as of `d29e3e3`). sha-pin cascade SLICE-INVARIANT (any pin advancement for `deepseek_v4.py` or nn-port must advance the right count — though 13.3b-4 should NOT touch those files, so the cascade should be a no-op; verify).

## MUST NOT
- Edit FROZEN bodies / nn-port (`deepseek_v4_nn.py`) / `shim_ds4_safetensors.py`.
- Touch convert in other slices / smoke-train (13.3b-5).
- Introduce C++.
- Transpose ape (BA §0.4 — copy verbatim).
- Leave modified test files uncommitted (AGENTS.md HARD RULE) — stage them, don't commit.
- Run `git commit` — that's the supervisor's job, post-green-gates.

Echo `{"status":"ok","role":"Coder"}` in this pane only.
