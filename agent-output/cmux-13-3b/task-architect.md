# Story 13.3b — Architect Task Brief (THICK: MLX CSA+HCA+Indexer port + ADR 0026 + sub-slice breakdown)

## Role
Architect (anthropic/claude-opus-4-8 · high, surface:82, fresh-context via /new). THICK — you own the design, ADR, and sub-slice breakdown.

## Context — what recon found (user picked Option B: full port)
The real DeepSeek V4 ckpt (`/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/`, 162GB, 46 shards) has 43 layers routed by `compress_ratios`:
- **cr=0 (sliding)**: 3 layers (0, 1, 42). FROZEN `_attention_mlx` cr=0 path IS real-dim-capable (only needs num_kv_heads==1 ✓, num_heads%o_groups==0 ✓, per-head sinks ✓).
- **cr=4 (CSA — Compressed Sparse Attention)**: 21 layers (even idx 2-40). FROZEN tiny CSA path hard-blocked at `_csa_config_error:317` (requires head_dim==hidden, hc_mult==1, o_groups==1, num_heads==1 — real violates ALL).
- **cr=128 (HCA — Heavily Compressed Attention)**: 20 layers (odd idx 3-41). FROZEN has NO HCA path at all (cr=128 not even in guards).

### Recon positives (de-risks the port)
1. **hc_mult=4 already supported** by FROZEN `_hyperconnection_mlx:550` + `_hyperhead_mlx:600` — they validate `stream.shape[-2]==hc_mult` and scale shapes with hc_mult. NO HC/HyperHead port work needed; 13.3a nn port (`HyperConnectionNN`/`HyperHeadNN` delegate at `:103`/`:131`) reuses directly at real dims.
2. **FP4 expert path already real-dim-capable** (proven 13.3a-2 AC5 parity + 13.3a-3 backward AC) — `DeepseekV4FP4Experts` + `_dequantize_fp4_block_scale_mlx` + `_moe_mlx` work at real `n_routed_experts=256`. NO MoE port work needed.
3. **torch impl is the reference**: `python-envs/mlx/.venv/lib/python3.13/site-packages/transformers/models/deepseek_v4/modeling_deepseek_v4.py` (1526 lines) has the full CSA/HCA/Indexer/GroupedLinear + Attention wiring.

### Port scope (~705 lines torch → MLX functional helpers)
| torch class | lines | MLX port |
|---|---|---|
| `DeepseekV4HCACompressor` | 362-445 | `_hca_compressor_mlx` (cache-less, multi-head, cr=128) |
| `DeepseekV4CSACompressor` | 589-754 | `_csa_compressor_real_mlx` (cr=4, multi-head, overlap windows) |
| `DeepseekV4IndexerScorer` | 446-461 | `_indexer_scorer_mlx` |
| `DeepseekV4Indexer` | 462-588 | `_indexer_mlx` (top-k compressed KV selection, position bias, causal threshold) |
| `DeepseekV4HCACache` | 171-254 | cache-less forward in MLX (cache is train-time-stateless; defer cache to generate path) |
| `DeepseekV4CSACache` | 255-302 | cache-less forward (same) |
| `DeepseekV4GroupedLinear` | 303-361 | `_grouped_linear_mlx` (o_groups=8 output projection) |
| `DeepseekV4Attention` forward | 755-875 | expand FROZEN `_attention_mlx:632` cr≠0 branch to dispatch cr=4→CSA, cr=128→HCA at real dims |

### Key remap (13.3b-4, NOT FROZEN — convert side)
- ckpt keys: `layers.N.ffn.experts.M.w{1,2,3}.{weight,scale}` (per-expert, NO `model.` prefix) → nn port `model.layers.N.mlp.experts.w{1,2,3}_{weight,scale}` (stacked `(n_routed, ...)`, `model.` prefix).
- field rename: `attn.wq_a`→`self_attn.q_a_proj`, `attn.wkv`→`self_attn.kv_proj`, `attn.wo_a/b`→`self_attn.o_a/b_proj`, `attn_norm`→`input_layernorm`, `ffn`→`mlp`, `post_attention_norm`→`post_attention_layernorm`.
- attn weights FP8 E4M3 + E8M0 scale → shim unpacks to BF16 (experts stay FP4 uint8+BF16 scale).
- CSA/HCA compressor/indexer weights: `compressor_wkv/wgate/ape/norm`, `indexer_compressor_*`, `indexer_*` — verify present in ckpt keys (BA/Architect 13.3b-3/4 to confirm).

## Your job (THICK Architect — design + ADR + breakdown)
Per AGENTS.md HARD STARTUP GATE + ADR 0017/0024 FROZEN contract. You OWN:

### 1. Design the MLX CSA/HCA/Indexer port (functional helpers, dict-based)
Mirror the FROZEN `_csa_compressor_mlx`/`_csa_indexer_mlx`/`_attention_mlx` pattern: stateless MLX functional helpers taking `args` + `weights: dict[str, mx.array]`. The port REPLACES the tiny-only `_require_csa_config` guards with real-dim multi-head implementations derived from torch `modeling_deepseek_v4.py:362-754`.

Design decisions to LOCK:
- **cache policy**: training is stateless per-forward (no KV cache); generate path cache is DEFERRED to a later slice (flag for 13.3c+). Confirm.
- **multi-head CSA/HCA**: torch uses per-head compressors/indexers. MLX port must handle `num_attention_heads=64`, `o_groups=8`, `head_dim=512`, `index_n_heads=64`, `index_head_dim=128`, `index_topk=512` at real dims.
- **GroupedLinear**: o_groups=8 grouped output projection — port as `_grouped_linear_mlx(x, weight, n_groups)` (torch `DeepseekV4GroupedLinear:303`).
- **`_attention_mlx` dispatch**: expand the cr≠0 branch (currently `_csa_attention_mlx` tiny-only) to: `if cr==4 → _csa_attention_real_mlx`, `elif cr==128 → _hca_attention_mlx`, `else raise`. The cr=0 path stays unchanged (real-dim-capable).
- **AttentionNN guard lift**: `deepseek_v4_nn.py:153` `raise NotImplementedError("13.3a-1 AttentionNN supports only compression_ratio=0")` → lift to dispatch cr=0/4/128. Verify the `linear_weight()` QuantizedLinear handling (13.3a-3) still composes with CSA/HCA weight dicts.

### 2. Write ADR 0026 — FROZEN contract expansion for real-ckpt attention
Justify expanding FROZEN `deepseek_v4.py` with real-dim CSA/HCA/Indexer helpers. Document: why the tiny-only guards are lifted, what the new contract is, how parity is preserved (torch as reference + tiny fixtures stay via the existing `_require_csa_config` path for tiny tests), blast-radius on FROZEN symbols. Place at `docs/adr/0026-csa-hca-indexer-mlx-real-port.md`.

### 3. Sub-slice breakdown (13.3b-1 .. 13.3b-5)
Propose the serial sub-slices (the 5-row table above is a STARTING point — refine):
- 13.3b-1: HCA (cr=128) — `_hca_compressor_mlx` + cache-less forward + tiny+real-dim tests vs torch reference.
- 13.3b-2: CSA (cr=4) compressor + Indexer + IndexerScorer — the hard part.
- 13.3b-3: GroupedLinear + `_attention_mlx` cr≠0 dispatch + AttentionNN guard lift + integration tests.
- 13.3b-4: real ckpt convert (key remap + FP8→BF16 + model. prefix + load_weights + sanitize) — NOT FROZEN, convert side.
- 13.3b-5: smoke-train on real 43-layer model (`mlx_lm.lora --train --iters 20`).

For EACH sub-slice: scope, blast-radius (which FROZEN files), AC (parity vs torch reference + tiny fixtures stay GREEN + backward AC stays GREEN), STOP-rules, dev-day estimate. Apply the STOP-rule: if total > 7 dev days for the design+first port slice, prescribe further split.

### 4. architecture.md update
Extend `agent-output/cmux-13-3a/architecture.md` (or new `agent-output/cmux-13-3b/architecture.md`) §0-§N with: real-ckpt layer breakdown (cr=0/4/128), the MLX port design, the FROZEN contract expansion, the key remap, the sub-slice sequence. Cite torch line numbers + FROZEN line numbers.

## Pre-flight read list (READ FIRST, in this order)
1. `agent-output/cmux-13-3a/architecture.md` §0-§15 (13.3a design — the nn port you're extending) + ADR 0025.
2. `agent-output/cmux-13-3a/requirements-13-3a-3.md` (13.3a-3 integration — the baseline you build on; HEAD `702199f`).
3. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` — FROZEN. Read: `_attention_mlx:632`, `_csa_attention_mlx:515`, `_csa_compressor_mlx:419`, `_csa_windowed_compressor_mlx:347`, `_csa_indexer_mlx` (find via grep), `_require_csa_config:335`, `_csa_config_error:317`, `_hyperconnection_mlx:550`, `_hyperhead_mlx:600`, `sanitize_weights:1600`, the FROZEN import list at top.
4. `python-envs/mlx/.venv/lib/python3.13/site-packages/transformers/models/deepseek_v4/modeling_deepseek_v4.py` — torch reference. Read lines 171-302 (caches), 303-361 (GroupedLinear), 362-754 (HCA/CSA/Indexer), 755-875 (Attention.forward), 876-973 (HyperConnection/HyperHead — confirm parity with FROZEN), 1085-1104 (SparseMoeBlock — confirm parity with 13.3a-2).
5. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py` — the nn port. Read `AttentionNN:135` (the guard at `:153` to lift), `linear_weight` helper (13.3a-3 QuantizedLinear handling), `Model.sanitize:421`, `ModelArgs:30`.
6. `/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim/config.json` — the real config (already probed: 43 layers, cr=0/4/128, hc_mult=4, o_groups=8, index_topk=512, n_routed_experts=256, expert_dtype=fp4).
7. `scripts/shim_ds4_safetensors.py:copy_sidecars:346` — the convert site (13.3a-3 wrote `model_type=deepseek_v4_nn` here; 13.3b-4 extends with key remap).
8. `agent-output/cmux-11-14/architecture.md` + `agent-output/cmux-11-15/architecture.md` — the original CSA recon (Story 11.14/11.15) that produced the tiny-only FROZEN CSA path. Understand WHY the tiny guards existed (parity-fixture scope) before lifting them.

## STOP-ESCALATE (binding)
Emit `{"status":"error","role":"Architect",...}` + write `agent-output/cmux-13-3b/architect-stop.md` if:
- The torch CSA/HCA/Indexer impl cannot be ported as MLX functional helpers without C++ or Metal kernels (AGENTS.md forbids C++; Metal kernels are a separate slice — flag, don't silently add).
- The port would break the FROZEN tiny-only CSA parity tests (Stories 11.14/11.15 fixtures) — the tiny path MUST stay GREEN via the existing `_require_csa_config` guards or a preserved tiny-codepath.
- The total port scope exceeds ~3 weeks dev effort even after sub-slicing — prescribe an Epic-level re-scope back to operator.
- A FROZEN symbol (`_attention_mlx`, `_csa_*`, `_hyperconnection_mlx`, parity `Model`, `sanitize_weights`) would need semantic change (not additive expansion) to existing tiny behavior — STOP, the contract is additive only.

## Deliverables
1. `agent-output/cmux-13-3b/architecture.md` §0-§N — design + ADR 0026 reference + sub-slice breakdown with per-slice scope/blast-radius/AC/STOP/dev-days.
2. `docs/adr/0026-csa-hca-indexer-mlx-real-port.md` — the FROZEN contract expansion ADR.
3. `agent-output/cmux-13-3b/subslice-breakdown.md` — the 13.3b-1..13.3b-5 table with per-slice dev-day estimates (apply STOP-rule: split if >7d).
4. `.cmux-status/architect.done` (`{"status":"ok","role":"Architect"}`).
5. In-pane JSON: `{"status":"ok","role":"Architect"}` in surface:82 ONLY.

MUST NOT edit production code (FROZEN or nn port or convert). Write ONLY architecture.md + ADR 0026 + subslice-breakdown.md + architect.done.

## Style
Caveman ultra default; byte-exact exempt (code/paths/line numbers/torch class:line/FROZEN helper:line/ADR cites verbatim). Use `ctx_execute`/`ctx_batch_execute` for any probe of torch/FROZEN files (keep large file bytes out of your context). Cite torch `modeling_deepseek_v4.py:line` + FROZEN `deepseek_v4.py:line` precisely. BEGIN DESIGN NOW.
