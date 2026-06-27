# ADR 0027 — Load-side ckpt-native → `deepseek_v4_nn` remap

- **Date:** 2026-06-27 (Story 13.3b-4b)
- **Status:** Accepted
- **Amends:** ADR 0025. Consumes ADR 0024 and ADR 0026 unchanged. Does not amend ADR 0017.

## Context

`mlx_lm.convert --model hf-f8shim -q --mlx-path model-4bit` must load the Jun 17
shimmed checkpoint directly into the trainable `deepseek_v4_nn` model. The
checkpoint keys are ckpt-native (`embed.*`, `layers.N.attn.*`, per-expert FP4
leaves), while the nn module expects `model.*` / `lm_head.*` leaves with stacked
routed experts.

Writing an intermediate remapped directory is disk-infeasible on the target NVME;
the source shim is ~163G and the output `model-4bit/` is ~149G.

## Decision

Use load-side remapping in `deepseek_v4_nn.Model.load_weights`:

1. Extract the ckpt-native → nn remap core into
   `ds4_ft_mlx.deepseek_v4_nn_remap` so the planner script and loader share one
   rename table.
2. In `Model.load_weights`, detect native keys by the absence of `model.` /
   `lm_head.` prefixes, remap with `remap_weight_dict`, then call
   `nn.Module.load_weights` strictly.
3. If all keys are already nn/save-model keys, bypass remap. This preserves
   idempotent reload of `model-4bit/` including quantized `.scales` / `.biases`.
4. Keep FP4 routed experts as `DeepseekV4FP4Experts` leaves with no
   `to_quantized`; `mlx_lm.convert -q` skips them and quantizes only regular
   `nn.Linear` leaves.

## Consequences

- One-step conversion succeeds without a 163G intermediate directory.
- The on-disk `model-4bit/` config keeps `model_type=deepseek_v4_nn` and a
  quantization block.
- Historical tests that asserted `model-4bit/` absence now accept either absence
  or a valid current `deepseek_v4_nn` 4-bit artifact.
- FROZEN `deepseek_v4.py` remains byte-intact.
