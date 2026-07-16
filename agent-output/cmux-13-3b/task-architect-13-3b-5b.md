# Story 13.3b-5b — Architect (backward-safe CSA block_bias scatter; ADR 0026 amendment)

## Context — smoke-train MILESTONE hit ONE real backward bug
Story 13.3b-5 Coder STOPPED (legitimate). The smoke-train on real 43-layer model:
- ✅ Model loaded (`Trainable parameters: 0.004% (5.571M/154430.069M)` — LoRA over 154B base).
- ✅ **Forward path PROVEN on real bytes**: `Iter 1: Val loss 19.116, Val took 2381.597s` (finite, no NaN, all 41/43 real layers including cr=4 CSA + cr=128 HCA + 256 FP4 experts dequantized in forward). The ENTIRE 13.3b arc (HCA + Indexer + CSA + Integration + remap + model-4bit) is FORWARD-correct on real bytes.
- ❌ First training BACKWARD step crashed: `ValueError: [scatter_axis] Cannot calculate VJP with respect to indices.` at `mlx_lm/tuner/trainer.py:250 loss_value_and_grad(model, *batch)`.

## Root cause (precise, supervisor-confirmed)
FROZEN `deepseek_v4.py:888-904` `_csa_block_bias_mlx` builds the CSA attention -inf mask via:
```python
scatter_indices = mx.expand_dims(safe_indices, 1)            # from top_k_indices (argsort)
zeros = mx.zeros(scatter_indices.shape, dtype=dtype)
return mx.put_along_axis(block_bias, scatter_indices, zeros, axis=-1)[..., :compressed_len]
```
MLX autograd traces through `put_along_axis` during `loss_value_and_grad` and tries to compute the VJP w.r.t. the `scatter_indices` argument. Indices are DISCRETE (from `argsort`) → non-differentiable → MLX raises `[scatter_axis] Cannot calculate VJP with respect to indices`.

## Scatter/argsort site inventory (FROZEN deepseek_v4.py)
| Line | Site | In LoRA grad path? | Risk |
|---|---|---|---|
| `:746` `_indexer_mlx` `top_k_indices = mx.argsort(-masked_scores)[...,:top_k]` | Feeds `_csa_block_bias_mlx` | Indexer weights NOT LoRA targets → argsort over frozen-weight scores | Covered by block_bias fix |
| `:902-904` `_csa_block_bias_mlx` `put_along_axis(block_bias, scatter_indices, zeros)` | **YES — the failing site** | YES (CSA attention is in the LoRA grad path via q_a/q_b/kv_proj) | **MUST FIX** |
| `:1282` MoE router `top_idx = mx.argsort(-selection_scores)[...,:num_experts_per_tok]` | FFN/MoE | NO — `DeepseekV4FP4Experts` is FROZEN FP4 (not LoRA target, ADR 0024); not in grad graph | Likely OK (verify) |

## Likely fix (Architect decides + sanctions)
**Option A — `mx.stop_gradient` on scatter_indices** (minimal, surgical):
```python
scatter_indices = mx.stop_gradient(mx.expand_dims(safe_indices, 1))
```
Or **Option C — `mx.stop_gradient` on the whole block_bias result**:
```python
return mx.stop_gradient(mx.put_along_axis(block_bias, scatter_indices, zeros, axis=-1)[..., :compressed_len])
```
Or **Option B — restructure as boolean-mask broadcast (no scatter)**:
```python
# valid_mask [B,1,S,compressed_len] via broadcast comparison, no put_along_axis
valid = (arange(compressed_len)[None,None,None,:] == safe_indices[...,None]).any(axis=-2, keepdims=False)
# ... build block_bias via mx.where(valid, 0.0, -inf)
```

**Architectural rationale for stop_gradient**: the CSA block_bias is a DISCRETE routing mask (top-k position selection), exactly analogous to MoE expert routing. MoE routers are non-differentiable by design (gradients flow through expert VALUES, not the discrete selection). The indexer weights are NOT LoRA targets → `top_k_indices` derive from frozen-weight scores → stop_gradient is doubly correct (no grad needed there anyway). Forward output unchanged (stop_gradient is identity in forward) → 13.3b-2b AC2 forward parity preserved.

## Architect job — thin decision + spec
1. **Confirm diagnosis** (scatter VJP in `_csa_block_bias_mlx` is the sole blocker; verify `:1282` MoE argsort is NOT in grad path because FP4 experts are frozen).
2. **Decide the fix** (A vs B vs C) + rationale. Prefer the MINIMAL change that preserves forward parity + makes backward complete. If Option A/C (stop_gradient), confirm it's sufficient (does MLX still trace through `put_along_axis` if the indices are stop_gradient'd? — verify in MLX 0.31.3 / mlx.core source, or spec a tiny probe Coder runs first).
3. **Sanction the FROZEN body edit** via ADR 0026 amendment: real backward bug justifies editing the FROZEN `_csa_block_bias_mlx` body (the additive-only rule yields to a real correctness bug; forward parity preserved). Draft the ADR amendment text.
4. **Check for OTHER backward risks** in the CSA/HCA/Indexer forward path (any other scatter/gather that's differentiable-risky). The HCA path (`_hca_compressor_mlx`, `_attention_real_mlx` HCA branch) doesn't use `put_along_axis` (confirmed by grep) — but verify there's no other autograd-unsafe op in the grad path.
5. **AC** for the Coder slice: (1) the edit lands on `_csa_block_bias_mlx` (or equivalent); (2) forward parity 13.3b-2b AC2 (CSA attention vs torch) STILL holds — stop_gradient is forward-identity so parity must be byte-identical; (3) smoke-train completes iter 20 backward without the scatter VJP error; (4) BA's AC §3 (a)-(f) for 13.3b-5 now achievable; (5) FROZEN source-hash CHANGES (sanctioned by ADR amendment — this is the one case where the hash changes, with ADR justification); (6) sha-pin cascade ADVANCES (the FROZEN body edit to `_csa_block_bias_mlx` shifts `deepseek_v4.py` source-hash → ALL sha-pin sites must advance to the new hash per AGENTS.md SLICE-INVARIANT).
6. **New slice name**: e.g. **13.3b-5b — backward-safe CSA block_bias (stop_gradient / ADR 0026 amendment)**.

## Deliverable — ONE micro-revision file
`agent-output/cmux-13-3b/architecture-13-3b-5b-backward-safe-blockbias.md` with:
- Decision (A/B/C) + rationale.
- Exact edit spec for Coder (which line, what code).
- ADR 0026 amendment text (sanctioning the FROZEN body edit for the real backward bug).
- AC for the Coder slice (forward parity preserved + backward completes + smoke-train GREEN + sha-pin cascade advances).
- Any pre-probe Coder should run (e.g. tiny MLX test that `mx.stop_gradient(indices)` + `put_along_axis` is VJP-safe before touching FROZEN).

## Pre-flight read
1. `agent-output/cmux-13-3b/coder-13-3b-5-stop.md` (full Coder STOP — diagnosis + val loss 19.116 evidence + scope guard).
2. `agent-output/cmux-13-3b/coder-13-3b-5-smoke-train.log` (the actual traceback + val loss).
3. `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py:888-904` (`_csa_block_bias_mlx` — the failing site) + `:746` (`_indexer_mlx` argsort) + `:1282` (MoE argsort — verify frozen).
4. `agent-output/cmux-13-3b/architecture.md` §3.4 (CSA block_bias spec) + §8 (FROZEN contract).
5. ADR 0026 (additive FROZEN — the amendment target) + ADR 0024 (FP4 dequant — experts frozen, not in grad path) + ADR 0025 (nn-port).
6. `agent-output/cmux-13-3b/requirements-13-3b-5.md` (BA AC §3 — the GREEN target).
7. MLX 0.31.3 `mx.stop_gradient` + `put_along_axis` VJP semantics (probe or docs).

## Build
`source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first). Can run a tiny MLX probe to verify `mx.stop_gradient(indices)` makes `put_along_axis` VJP-safe (recommended before sanctioning). Caveman ultra default; byte-exact exempt. NO full test runs — design decision + spec + ADR amendment only. Echo `{"status":"ok","role":"Architect"}` in surface:82 only when the micro-revision file is written. BEGIN NOW.
