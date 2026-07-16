# Story 13.3b-4 — Thin BA scope (real ckpt convert / key remap + load)

## Slice context
- **Story**: 13.3b-4 — real ckpt remap script + load. Convert side ONLY (NOT FROZEN, NOT nn-port code).
- **Predecessor**: 13.3b-3 complete (HEAD `d1f1488`). All 41/43 real layers run at real dims through `_attention_mlx` dispatch. nn port `deepseek_v4_nn.py` is wired (AttentionNN guard lifted + §4.1 submodules).
- **Estimated**: 3 dev-days (Architect §5 + breakdown).
- **13.3b-5 owns**: smoke-train.

## Architect §5 owns the rename table (READ `agent-output/cmux-13-3b/architecture.md` §5.1-§5.4)
- NEW script `scripts/remap_ds4_nn_weights.py` (or extend `scripts/shim_ds4_safetensors.py`).
- Rename table §5.1: `embed.weight→model.embed_tokens.weight`, `head.weight→lm_head.weight`, `layers.N.attn.*` → `model.layers.N.self_attn.*`, `layers.N.ffn.*` → `model.layers.N.mlp.*`, per-expert→stacked, `hc_attn_*`/`hc_ffn_*` → 13.3a leaf names, `mtp.*` DROP.
- `ape` TRANSPOSE §5.2: ckpt CSA/indexer `ape` is `(rate, 2*out_dim)`; nn port expects `(2*out_dim, rate)`. HCA ape → NO transpose (torch convention).
- dtype §5.3: attn core BF16 + drop redundant `.scale`; routed experts STAY FP4 (stacked, `_dequantize_fp4` at forward ADR 0024); ape/sinks/hc F32 keep.
- load §5.4: nn `sanitize` delegates FROZEN `sanitize_weights:1600` (mtp strip); `model.load_weights(list(...))`; `mlx_lm.convert` resolves nn module by `model_type="deepseek_v4_nn"` (13.3a-3 wiring).

## BA tasks (thin — not re-designing, Architect §5 owns design)
1. Confirm scope: NEW `scripts/remap_ds4_nn_weights.py` (extend shim acceptable). NO FROZEN/nn-port code edits (convert side only).
2. Resolve: (Q1) NEW script vs extend `shim_ds4_safetensors.py:361` — which is cleaner? (Q2) shape-only/sub-checkpoint validation acceptable (16GB constraint, mirror 13.3a-3 Q9) — confirm grading does NOT require full 162GB run. (Q3) STOP-rule: ckpt tensor with no nn param target → STOP; confirm exact shape.
3. Pin AC: remapped key set == nn `model.parameters()` key set exactly; load succeeds strict; one real CSA + one real HCA + one sliding layer load + forward finite; tiny byte-identity regression.
4. Verify `model.parameters()` tree (the nn port leaf set the remap must match) — list the actual leaf names from `deepseek_v4_nn.py` AttentionNN submodules + DeepseekV4FP4ExpertsNN + HyperConnectionNN + HyperHeadNN. This is the TARGET key set the remap must produce. CRITICAL: Coder must produce keys that match this exact tree, not just the §5.1 table — the §5.1 table is the spec, the nn tree is the ground truth; flag any mismatch for Architect § clarification.
5. Update `docs/backlog.md` Story 13.3b-4 row.

## Deliverables
1. `agent-output/cmux-13-3b/requirements-13-3b-4.md` — scope + Q1-Q3 resolutions + AC + STOP-rule + nn leaf-set target (Q4).
2. `.cmux-status/ba.done` (`{"status":"ok","role":"BA"}`).
3. In-pane JSON: `{"status":"ok","role":"BA"}` in surface:81 ONLY.
4. `docs/backlog.md` update.

## STOP-ESCALATE
Write `agent-output/cmux-13-3b/ba-13-3b-4-stop.md` + `{"status":"error","role":"BA",...}` if:
- Architect §5.1 rename table has a tensor with no nn param target (Q3 STOP-rule materializes → escalate for Architect § clarification on the unknown arch piece).
- nn param tree (Q4) diverges from §5.1 table in a way Coder can't resolve → escalate for Architect § clarification.

## Pre-flight read
1. `agent-output/cmux-13-3b/architecture.md` §5.1-§5.4 + §0 nn tree.
2. `agent-output/cmux-13-3b/subslice-breakdown.md` 13.3b-4 row.
3. `scripts/shim_ds4_safetensors.py` (existing convert infra — `copy_sidecars:361` writes `model_type="deepseek_v4_nn"`).
4. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py` (the nn param tree — AttentionNN submodules + FP4Experts + HyperConnection + HyperHead leaf names).
5. `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json` (real config: 43 layers, `num_hash_layers=3`, `n_routed_experts=256`, `expert_dtype="fp4"`, etc.).
6. `agent-output/cmux-13-3b/requirements-13-3b-3.md` (predecessor context).
7. `docs/adr/0025-*.md` (nn file ownership) + `docs/adr/0024-*.md` (FP4 dequant — experts STAY FP4).

## Build
`source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first). Use `ctx_execute` for the param-tree probe.

## Style
Caveman ultra default; byte-exact exempt. Cite §5/Q/AC numbers. BEGIN NOW.
