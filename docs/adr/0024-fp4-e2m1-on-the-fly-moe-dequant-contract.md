# ADR 0024 — FP4 (OCP MXFP4 E2M1) on-the-fly MoE dequant contract

- **Date:** 2026-06-26 (Story 13.2, Blocker 2)
- **Status:** Accepted
- **Supersedes / amends:** Supersedes ADR 0022-key-remap §(b) (the FP4 expert
  packing contract documented there); those technical findings stand and are
  promoted here to a durable dequant contract. Does **NOT** amend ADR 0017
  (consumes its boundary-X / 9-forbidden-symbols rule unchanged).
- **Related:** ADR 0007 §4 (anti-circularity), ADR 0015 (routed-dequant trusted-
  reference readiness), ADR 0016/0017 (derivation boundary X + forbidden
  symbols), ADR 0022 (strategic pivot to local MLX QLoRA).

## Context

Epic 13 (Path A, local MLX QLoRA) requires the vendor `mlx_lm` DeepSeek-V4 port
to decode FP4 byte-packed routed-expert tensors on-the-fly during the MoE
forward pass, so `mlx_lm.load()` + forward over the shimmed checkpoint completes
in-RAM on M3 Ultra without a bulk-dequant OOM. The shimmed checkpoint declares
`expert_dtype="fp4"`; routed-expert `wN.weight` ship as I8 containers holding 2
FP4 nibbles/byte, paired with BF16 `wN.scale`.

Before Story 13.2 the fp4 path was fail-closed (`NotImplementedError`) because
no trusted reference existed (ADR 0002 / ADR 0015). Story 13.2 establishes the
reference under derivation boundary X.

## LIVE-confirmed format facts (Story 13.2 BA, LOCKED)

- `expert_dtype="fp4"`; e.g. w1.weight I8 `[2048,2048]` + w1.scale BF16
  `[2048,128]`. Logical in-features = `in_bytes * 2` (4096 for w1).
- Decoded value range EXACTLY `[-6.0, +6.0]`; all 16 nibble codes present; first
  BF16 scales are clean powers-of-two (`2^-7, 2^-6, ...`).
- ⟹ format is **OCP MXFP4 E2M1 with BF16 per-32-logical block scales**. NOT
  NVFP4 (no E4M3 micro-scale, no tensor global scale), NOT a custom format.

## Decision (the durable contract)

1. **E2M1 LUT** (1 sign, 2 exp, 1 mantissa, bias 1), indexed by raw 4-bit code
   0..15:
   `(+0.0,+0.5,+1.0,+1.5,+2.0,+3.0,+4.0,+6.0, -0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0)`.
   Cite: `transformers/integrations/mxfp4.py:28-45`; cross-cite
   `transformers/integrations/finegrained_fp8.py:925` `_FP4_E2M1_LUT`.
2. **LSB-first nibble unpack**: low nibble (`byte & 0x0F`) = first/even logical
   element; high nibble (`byte >> 4 & 0x0F`) = second/odd. Cite:
   `transformers/integrations/mxfp4.py:292-298`.
3. **block_size = 32 LOGICAL** features per scale (NOT 16 packed bytes). axis=1
   (in-features). `n_blocks = in_logical / 32 = scale.shape[1]`.
4. **Scale = BF16, linear domain, direct multiply** (NOT E8M0 ldexp). Each scale
   spans 32 consecutive logical features (`mx.repeat(scale, 32, axis=1)`).
5. **in_logical = in_bytes * 2**; reinterpret the I8 byte as uint8 BEFORE bit
   ops (logical shift, no sign extension).
6. **Derivation provenance (boundary X)**: math derives ONLY from the OCP MXFP4
   spec + HF mxfp4 reference (math/convention authorities under ADR 0016/0017
   boundary X). The new primitives MUST NOT reuse, import, or wrap any of the 9
   forbidden OUR-Python symbols (`_apply_i8_block_scales`,
   `dequantize_i8_block_scale`, `dequantize_i8_e8m0_block_scale`,
   `_dequantize_i8_block_scale_mlx`, `_moe_mlx` math, `decode_f8_e8m0_scales`,
   `dequantize_f8_e4m3fn_with_e8m0_scales`, the f8 e4m3 decode, `f8_e8m0_to_bf16`).
   They MAY reuse pure structural plumbing (`_as_shape`, `_product`,
   `_normalize_axis`, `_expected_scale_shape`, `_row_major_coords`,
   `_decode_bf16`) — none of which carry dequant math.

## Primitives introduced

- `_dequantize_fp4_block_scale_mlx(weight, scale, *, block_size=32, axis=1)` in
  vendor `deepseek_v4.py` — forward path; returns fp32.
- `dequantize_fp4_block_scale(payload, scales, *, shape, block_size=32,
  scale_axis=1)` in `deepseek_v4_dequant.py` — pure-Python parity authority.
- `dequantize_expert_packed("fp4", …)` fp4 stub becomes a real dispatch to the
  Python ref. i8 branch + final `raise` unchanged.
- `_moe_mlx` gains an `elif expert_dtype == "fp4":` branch; i8 / else branches
  unchanged.

## Correctness / epsilon

- E2M1 LUT values are exact in fp32; LUT lookup + fp32 multiply is
  deterministic. fp32 path parity: `max_abs_error == 0`.
- bf16 output (if ever cast) is within `2^-7 · max|scale|` (one bf16 ULP).
- The MLX primitive and the Python ref are independently derived from the same
  spec and cross-checked on identical synthetic input (AC1↔AC2).

## Anti-circularity caveat (load-bearing)

Synthetic parity passes for ANY consistent nibble order. Order is established
ONLY by (a) the cited external OCP/HF authority (LSB-first), and (b) real-ckpt
forward finiteness (AC4) + Story 13.3 generation coherence. If AC4/13.3 fail
under LSB-first, the order is wrong → STOP and re-LOCK. Byte histograms cannot
disambiguate. Test expected tables are literals from an in-test numpy reference,
never derived from production primitives (ADR 0007 §4).

## Consequences

- The fp4 forward path is unblocked for Story 13.3 (convert-shimmed + smoke
  train) and downstream fusion (ADR 0019).
- FROZEN i8/e8m0/f8 primitives and the Metal production path are untouched;
  boundary X is preserved.
- If the real checkpoint ever ships a different FP4 packing (e.g. NVFP4 or
  MSB-first), this contract is falsified and must be revised — the AC4/13.3
  gates are the detectors.
