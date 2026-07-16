# Architecture micro-revision — 13.3b-5b backward-safe CSA block_bias (stop_gradient / ADR 0026 amendment)

Slice: **13.3b-5b — backward-safe CSA block_bias (stop_gradient; ADR 0026 amendment)**
Architect deliverable. Design + spec + ADR amendment only. NO full test runs.

## §0 Diagnosis — CONFIRMED (probe-verified, MLX 0.31.2)

Sole backward blocker = FROZEN `deepseek_v4.py:902-904` `_csa_block_bias_mlx`:
```python
scatter_indices = mx.expand_dims(safe_indices, 1)
zeros = mx.zeros(scatter_indices.shape, dtype=dtype)
return mx.put_along_axis(block_bias, scatter_indices, zeros, axis=-1)[..., :compressed_len]
```
`safe_indices` derive from `_indexer_mlx:746` `mx.argsort(-masked_scores)[...,:top_k]`. Under
`loss_value_and_grad`, MLX traces `put_along_axis` VJP w.r.t. its `indices` arg → discrete (argsort) →
`ValueError: [scatter_axis] Cannot calculate VJP with respect to indices`.

Probe (indices derived from a differentiable leaf, mirroring real model) reproduced EXACTLY:
```
BASELINE             : FAIL ValueError: [scatter_axis] Cannot calculate VJP with respect to indices.
OPT_A_stopgrad_idx   : OK grad computed
OPT_C_stopgrad_result: OK grad computed
```

Scatter/argsort inventory verdict:
| Line | Site | In LoRA grad path? | Verdict |
|---|---|---|---|
| `:746` `_indexer_mlx` argsort | feeds block_bias | indexer weights NOT LoRA targets | covered by the fix; no grad needed |
| `:902-904` `_csa_block_bias_mlx` put_along_axis | **YES** (CSA q_a/q_b/kv in grad path) | **MUST FIX** |
| `:1282` MoE argsort | FFN/MoE | NO — FP4 experts FROZEN (ADR 0024) | **OK — probe-verified backward-safe**: argsort feeds equality `mx.any(top_idx==eid)`, NOT scatter; `mx.grad` succeeds |

Other CSA/HCA/Indexer grad-path ops: `put_along_axis` appears ONLY at `:904` (grep). HCA path
(`_hca_compressor_mlx`, `_attention_real_mlx` HCA branch) uses no scatter/gather-by-traced-index.
`mx.where`/`mx.argsort`/`mx.softmax`/`_linear_mlx` are all VJP-safe. **No other autograd-unsafe op in the grad path.**

## §1 Decision — **Option A** (`mx.stop_gradient` on `scatter_indices`)

```python
scatter_indices = mx.stop_gradient(mx.expand_dims(safe_indices, 1))
```

Rationale (minimal, forward-parity-preserving):
- **Forward-identity, probe-verified byte-identical**: `stop_gradient` is identity in forward; block_bias
  output (`[[[[0.0,-inf,0.0,-inf]]]]`) is bit-for-bit equal baseline vs optA. 13.3b-2b AC2 (CSA-vs-torch
  forward parity) holds by construction.
- **Doubly correct semantically**: block_bias is a DISCRETE routing mask (top-k position selection),
  analogous to MoE expert routing — non-differentiable by design; gradients flow through attention VALUES,
  not the discrete index selection. The indices also derive from indexer weights which are NOT LoRA targets,
  so no gradient is needed there regardless.
- **Smaller blast radius than Option C**: A stops grad only on the discrete `indices` arg (the exact thing
  MLX cannot differentiate). Option C `stop_gradient` on the whole result would ALSO sever grad w.r.t. the
  differentiable `block_bias`/`zeros` values — unnecessary and semantically over-broad (the `-inf`/`0.0`
  values are constants here so C happens to be equivalent today, but A is the precise statement of intent
  and is robust if the values path ever becomes differentiable). Option B (boolean-mask broadcast) is a
  larger rewrite of FROZEN body — rejected as non-minimal.

## §2 Exact edit spec for Coder

File: `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`
Function: `_csa_block_bias_mlx` (line `902`). ONE line, additive wrap:

```python
# OLD (line 902)
    scatter_indices = mx.expand_dims(safe_indices, 1)
# NEW
    scatter_indices = mx.stop_gradient(mx.expand_dims(safe_indices, 1))
```

Nothing else in the function changes. Lines `903-904` unchanged.

Suggested inline comment (beside the edit, per AGENTS.md comment policy):
```python
    # block_bias is a DISCRETE top-k routing mask (indices from argsort over frozen
    # indexer scores). MLX cannot VJP put_along_axis w.r.t. indices; stop_gradient is
    # forward-identity (parity preserved) and semantically correct — grads flow through
    # attention values, not the discrete position selection. (13.3b-5b / ADR 0026 amend.)
    scatter_indices = mx.stop_gradient(mx.expand_dims(safe_indices, 1))
```

## §3 ADR 0026 amendment text (sanctioning the FROZEN body edit)

Append to `docs/adr/0026-csa-hca-indexer-mlx-real-port.md` (new section before/after Consequences). The
additive-only rule yields to a REAL backward correctness bug; forward parity is preserved, so the contract's
intent (tiny parity protected, real path correct) is honored.

```markdown
## Amendment 1 — 2026-06-27 (Story 13.3b-5b): backward-safe block_bias

**Status:** Accepted. **Trigger:** first real 43-layer `mlx_lm.lora --train` run. Forward PROVEN on
real bytes (Iter 1 Val loss 19.116, finite, all cr=4 CSA + cr=128 HCA + 256 FP4 experts dequant
forward-correct); first BACKWARD step crashed `ValueError: [scatter_axis] Cannot calculate VJP with
respect to indices` at `mlx_lm/tuner/trainer.py:250 loss_value_and_grad`.

**Decision.** The additive-only FROZEN rule (Decision §1) is amended for ONE precise case: a REAL
backward correctness bug in a FROZEN helper BODY may be fixed with a forward-identity transform. The
sole sanctioned body edit is in `_csa_block_bias_mlx` (`deepseek_v4.py:902`):
`scatter_indices = mx.stop_gradient(mx.expand_dims(safe_indices, 1))`.

**Why this is contract-consistent, not a contract break.**
- `mx.stop_gradient` is IDENTITY in the forward pass → block_bias output is byte-identical (probe-verified).
  All tiny CSA parity fixtures (Story 11.14/11.15) and 13.3b-2b AC2 (CSA-vs-torch forward parity) stay GREEN.
- block_bias is a DISCRETE top-k routing mask whose indices come from argsort over indexer-weight scores;
  those weights are NOT LoRA targets → no gradient is required there. The mask is non-differentiable by
  design (like MoE expert routing); gradients flow through attention values, not the discrete selection.
- The fix is the minimal statement of that fact: stop grad ONLY on the `indices` argument MLX cannot
  differentiate. No algorithm, shape, dtype, or numeric value changes.

**Hash consequence (the single sanctioned source-hash change).** This edit shifts `deepseek_v4.py`
sha256 from `96c39168c78e5fd9...` to `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b`
(sha16 `5e11a9c4d82aeb24`). Per AGENTS.md SLICE-INVARIANT, ALL sha-pin sites MUST advance to the new hash
in the SAME slice (cascade enumerated in 13.3b-5b architecture §4). This is the documented case where the
FROZEN hash legitimately changes, with this ADR amendment as the justification.

**Scope limit.** This amendment authorizes ONLY the `_csa_block_bias_mlx:902` stop_gradient wrap. No other
FROZEN body edit is sanctioned. Any further FROZEN body change still STOP-ESCALATES.
```

## §4 sha-pin cascade — sites that MUST advance (SLICE-INVARIANT)

Old: `96c39168c78e5fd9...` → New full: `5e11a9c4d82aeb24ea595383dfe1a339cc43aa6877e9978cb5d303534eb6d98b`
(sha16 `5e11a9c4d82aeb24`).

Coder advances ALL of these in the SAME slice as the body edit:
1. `tests/test_numpy_real_forward_reference_composition.py:397` — `EXPECTED_SHAS[...deepseek_v4.py] = "5e11a9c4d82aeb24"`
2. `tests/test_deepseek_v4_real_config_reference_forward.py:383` — `EXPECTED_SHAS[...deepseek_v4.py] = "5e11a9c4d82aeb24"`
3. `python-envs/mlx/src/ds4_ft_mlx/real_forward_intermediate_dump.py:61` — `VENDOR_SHA256_16 = "5e11a9c4d82aeb24"` (and update its `# ADR 0026 ... baseline` comment to note the amendment)

Auto-advances (no literal, reads from `rfid.VENDOR_SHA256_16`): `tests/test_real_forward_intermediate_dump.py:83`.

Also re-pin ADR 0026's OWN hash where pinned, AFTER the amendment text is appended (the amendment changes
0026's bytes). `tests/test_deepseek_v4_real_config_reference_forward.py` and
`test_numpy_real_forward_reference_composition.py` do NOT pin 0026; verify with a grep for the 0026 path in
all `EXPECTED_SHAS` and advance any that do before declaring done.

## §5 AC for the Coder slice (13.3b-5b)

- (1) Edit lands: `_csa_block_bias_mlx:902` wrapped in `mx.stop_gradient`; lines 903-904 unchanged; no other body change.
- (2) **Forward parity preserved** — 13.3b-2b AC2 (CSA-vs-torch) STILL byte-identical (stop_gradient is forward-identity; probe-confirmed).
- (3) **Backward completes** — smoke-train reaches iter 20 with NO `[scatter_axis]` VJP error; first training-loss reported.
- (4) BA AC §3 (a)-(f) for 13.3b-5 now ACHIEVABLE: (a) loss finite, (b) grad tree non-empty, (c) all LoRA leaf grads finite, (d) ≥1 LoRA grad non-zero, (e) iter 20 no OOM/crash, (f) `adapters-smoke/adapters.safetensors` saved.
- (5) FROZEN source-hash CHANGES to `5e11a9c4d82aeb24...` — sanctioned by ADR 0026 Amendment 1 (the ONE case).
- (6) sha-pin cascade ADVANCES — all §4 sites pinned to new hash in the same commit; `git ls-files` confirms every baseline-contributing test file is TRACKED before declaring done (AGENTS.md tracking-hygiene HARD RULE).

## §6 Pre-probe (RUN BY ARCHITECT — Coder may re-run to confirm before touching FROZEN)

Already executed (MLX 0.31.2, `python-envs/mlx/.venv`). Results above. Coder MAY re-run this minimal probe
first to re-confirm VJP-safety on its own environment before the FROZEN edit:

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

## §7 Notes for downstream

- Pure design/spec slice → Coder edit is ONE FROZEN line + 4 sha-pin advances + ADR amendment + ADR re-pin. Keep file-mutating work serial.
- Reviewer (`xhigh-reviewer`, gpt-5.5) MUST verify forward parity unchanged (not just backward passes) and that the cascade advanced every pin (grep old hash `96c39168c78e5fd9` → expect ZERO hits in tracked code/tests after the slice).
- Test Manager runs smoke-train to iter 20 + AC §3 (a)-(f); independent verdict.
