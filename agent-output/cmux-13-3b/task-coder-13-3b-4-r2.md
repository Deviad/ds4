# Story 13.3b-4 — Coder r2 (fix AC2: real sub-checkpoint load)

## Slice context
- **Story**: 13.3b-4 — real ckpt convert / key remap + load. Convert side ONLY.
- **r0 result**: AC1 ✓ + AC3 ✓ + all byte-intact ✓ + ape-verbatim ✓ + tracking ✓ (HEAD still `58194a9`, nothing committed). But **AC2 BLOCKED by Reviewer** (Axis 5 / AC2 legitimacy).
- **HEAD**: still `58194a9` (no commit per commit-gating policy — both gates must GREEN first).
- **r0 staged work** is your starting point: `git diff --cached` shows the 6 staged deliverables. Your r2 modifies the AC2 test in place + re-stages.

## r0 BLOCKED finding (Reviewer Axis 5 — CORRECT)

`tests/test_remap_ds4_nn_weights_load.py` does NOT perform a real sub-checkpoint load. It builds `_tiny_mixed_config()` (hidden=8, vocab=32 — synthetic tiny dims) and `_tiny_ckpt_style_weights()` with `_zeros`/`_ones` for ALL tensors, then strict-loads that fabricated dict. There is:
- NO `safe_open` / `load_file` / `safetensors` import
- NO `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim` reference
- NO actual checkpoint payload tensors

A remap that produces the right KEY SHAPES would pass this test even if it filled every value with garbage — the test cannot detect a value-corrupting remap. That's a tautology-adjacent gap.

**Test Manager r0 disagreed (said AC2 PASSED) but Test Manager misread the synthetic test as a real sub-checkpoint load — methodology gap. Reviewer is correct.** BA Q2 intended "per-instance on the loaded sub-checkpoint" = real payload tensors from `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim`, not synthetic tiny topology.

## r2 scope — fix AC2 ONLY (AC1 + AC3 untouched, they passed)

Rewrite `tests/test_remap_ds4_nn_weights_load.py` to load REAL safetensors payload tensors from the shimmed checkpoint for a sub-checkpoint of 3 layers (1 sliding + 1 CSA + 1 HCA) + global keys, remap, strict-load nn module, run finite forward.

### Real ckpt layout (verified)
- Path: `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/`
- 46 shards, 43 layers total
- `config.json`: `num_hidden_layers=43`, `num_hash_layers=3`, `n_routed_experts=256`, `expert_dtype="fp4"`, `hidden_size=4096`, `head_dim=512`
- `compress_ratios` array starts `[0, 0, 4, 128, 4, ...]` → L0/L1 sliding (cr=0), L2 CSA (cr=4), L3 HCA (cr=128), L4 CSA (cr=4), ...
- Each layer ≈ 1565-1569 tensors (256 routed experts × ~6 tensors + shared experts + attention)
- Global keys: `embed.weight`, `head.weight`, `norm.weight`, `hc_head_{fn,base,scale}`

### Sub-checkpoint approach (stays within 16GB RAM — BA Q2)
Build a sub-checkpoint in a tmp dir containing ONLY the tensors needed for 3 selected real layers + global keys:
1. Pick layers: **L0 (sliding, cr=0)**, **L2 (CSA, cr=4)**, **L3 (HCA, cr=128)** — covers all three dispatch paths.
2. Use `safetensors.safe_open` lazy loading: `safe_open(shard).get_tensor(key)` materializes ONLY the requested tensors (not the whole 162GB).
3. Extract tensors for `layers.0.*`, `layers.2.*`, `layers.3.*` + global keys (`embed.weight`, `head.weight`, `norm.weight`, `hc_head_*`) — remap them to nn-port keys (`model.embed_tokens.weight`, `model.layers.{0,2,3}.*`, `lm_head.weight`, `model.norm.weight`, `model.<hc_head leaf>`).
4. Write a sub-`config.json` with `num_hidden_layers=3`, `layer_types=["sliding_attention","compressed_sparse_attention","heavily_compressed_attention"]`, `compress_ratios=[0,4,128]`, `num_hash_layers=1` (L0 is a hash layer), rest from real config.
5. `Model(ModelArgs.from_dict(sub_cfg))` → `model.load_weights(list(remapped.items()), strict=True)` → `model(mx.array(input_ids))` → assert finite output, correct shape `(1, seq_len, vocab_size)`.
6. Verify real FP4 expert stacking: a routed expert's `w1_weight` should be `uint8` with real packed bytes (not zeros), shape `(n_routed_experts=256, ...)`.

### Anti-circularity (Reviewer Axis 5 watch-item)
The sub-checkpoint must use REAL payload tensors (not `_zeros`/`_ones`). The test must genuinely exercise the remap on real FP4 packed bytes + real BF16 scales, so a value-corrupting remap would be caught (forward would NaN or shape-mismatch). Grep your r2 test for `safe_open`/`load_file`/`get_tensor` — must be present. Grep for `_zeros`/`_ones`/`mx.zeros`/`mx.ones` — should be ABSENT except possibly for input_ids (which are integers, not weights).

### AC2 r2 acceptance
- `safe_open` / `load_file` / `get_tensor` present (real payload path).
- No `_zeros`/`_ones`/`mx.zeros`/`mx.ones` for weight tensors (input_ids OK).
- `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim` referenced.
- `model.load_weights(..., strict=True)` succeeds (no missing/unexpected key).
- Forward at seq_len ≥ 8 → finite output, correct shape `(1, seq_len, vocab_size)`.
- Real FP4 expert stacking verified (uint8 dtype, real packed bytes, shape `(256, ...)`).
- Stays within 16GB RAM (lazy `get_tensor`, don't materialize full 162GB).

## MUST NOT (unchanged from r0)
- Edit FROZEN bodies / nn-port `deepseek_v4_nn.py` / `shim_ds4_safetensors.py`.
- Touch AC1 or AC3 tests (they passed — leave them byte-identical).
- Run `git commit` (commit-gating — supervisor commits post-green-gates).
- Transpose ape (BA §0.4 — copy verbatim; r0 already correct, don't regress).
- Introduce C++.

## COMMIT GATING (reminder)
- `git add` (stage) the modified AC2 test. Do NOT commit.
- Staged-but-uncommitted is correct end state.
- Reviewer + Test Manager validate via `git diff --cached`.
- Supervisor commits only after both return GREEN.

## STOP-ESCALATE
Write `agent-output/cmux-13-3b/coder-13-3b-4-r2-stop.md` if:
- A real sub-checkpoint can't be built within 16GB RAM (escalate — may need BA/Architect to revise AC2 to allow shape-only or fewer layers).
- The real payload load exposes a remap BUG (some tensor remaps wrong on real bytes even though synthetic tiny passed) — STOP, record the failing tensor + shape + error, escalate. This would be a genuine remap correctness issue, not just a test fix.
- nn param tree diverges from real ckpt keys in a way the remap can't accommodate (S-nn-tree-divergence from r0 — re-check).

## Pre-flight read
1. `agent-output/cmux-13-3b/review-13-3b-4.md` (Reviewer's full BLOCKED verdict — your r2 must satisfy it).
2. `agent-output/cmux-13-3b/test-manager-13-3b-4-report.md` (Test Manager r0 — note the methodology gap, don't trust its AC2 PASSED verdict).
3. `agent-output/cmux-13-3b/requirements-13-3b-4.md` (BA SPEC — Q2 "sub-checkpoint validation" = real payload, not synthetic).
4. `agent-output/cmux-13-3b/coder-13-3b-4-notes.md` (your r0 notes).
5. `tests/test_remap_ds4_nn_weights_load.py` (your r0 AC2 test — the file to rewrite).
6. `scripts/remap_ds4_nn_weights.py` (your r0 remap script — should NOT need changes; if it does, that's a remap BUG → STOP).
7. `agent-output/cmux-13-3b/task-coder-13-3b-4.md` (original brief — ape §0.4, MUST NOT list, commit gating).

## Build
`source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first). Use `ctx_execute` for safetensors header/payload scans + suite runs. The sub-checkpoint build may take a few minutes (real I/O on 46 shards for 3 layers' worth of tensors).

## Deliverables
1. `tests/test_remap_ds4_nn_weights_load.py` (REWRITTEN — real sub-checkpoint load).
2. `agent-output/cmux-13-3b/coder-13-3b-4-r2-notes.md` (what was wrong in r0, the real-payload approach, RAM budget, any remap bugs found).
3. `.cmux-status/coder.done` (re-written fresh) + in-pane JSON `{"status":"ok","role":"Coder"}` in surface:83 ONLY.
4. `git add` the modified test (stage, NOT commit).

## Regression target
- AC1 ✓ (unchanged), AC2 GREEN (real payload), AC3 ✓ (unchanged).
- Full suite: 572 / 13 / 0 RED (same count, AC2 now real).
- FROZEN/nn-port/shim byte-intact (r0 already clean — don't regress).
- sha-pin cascade no-op.

Echo `{"status":"ok","role":"Coder"}` in this pane only. BEGIN NOW.
