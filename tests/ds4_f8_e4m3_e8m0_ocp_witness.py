#!/usr/bin/env python3
"""Standalone OCP/DS4 F8_E4M3 x F8_E8M0 witness for attention weights.

Math authority: OCP Microscaling Formats (MX) Specification v1.0 plus the
DS4 production importer in ``deepseek4-quantize.c`` L631-L647 and L683-L710.
This witness intentionally does not import the fuse helper.  It mirrors the C
reader's block-128 attention dequant path, including the DS4 convention that
E4M3 max-magnitude bytes 0x7F and 0xFF decode to 0.0 sentinels rather than
finite +/-448 values.
"""

from __future__ import annotations

import math
import struct
from array import array


def e8m0_to_f32(e: int) -> float:
    """Decode C ``e8m0_to_f32``: bits = e==0 ? 0x00400000 : e << 23."""

    e = int(e) & 0xFF
    if e == 0:
        return math.ldexp(1.0, -127)
    if e == 255:
        return float("inf")
    return math.ldexp(1.0, e - 127)


def e4m3fn_to_f32(x: int) -> float:
    """Decode DS4 F8_E4M3FN bytes exactly like ``deepseek4-quantize.c``.

    Critical DS4 quirk: abs==0x7f returns +0.0 for both 0x7F and 0xFF.
    """

    x = int(x) & 0xFF
    abs_x = x & 0x7F
    sign = bool(x & 0x80)
    if abs_x == 0:
        return -0.0 if sign else 0.0
    if abs_x == 0x7F:
        return 0.0
    exp = (x >> 3) & 0x0F
    man = x & 0x07
    if exp == 0:
        value = math.ldexp(float(man), -9)
    else:
        value = math.ldexp(1.0 + float(man) / 8.0, exp - 7)
    return -value if sign else value


def decode_f8_e4m3_e8m0_to_float32(
    weight_bytes: bytes | bytearray | memoryview,
    scale_bytes: bytes | bytearray | memoryview,
    *,
    out_dim: int,
    in_dim: int,
    block: int = 128,
) -> array:
    """Decode row-major F8_E4M3 weights with row-major E8M0 block scales.

    ``weight`` shape is ``(out_dim, in_dim)``. ``scale`` shape is
    ``(out_dim/128, in_dim/128)``.  The block geometry matches the C importer
    ``dequant_fp8_weight`` path for DS4 attention tensors.
    """

    if block != 128:
        raise ValueError("DeepSeek-V4 F8_E4M3 witness requires block=128")
    if out_dim <= 0 or in_dim <= 0 or out_dim % block or in_dim % block:
        raise ValueError(f"invalid F8 shape out_dim={out_dim} in_dim={in_dim} block={block}")

    weights = memoryview(weight_bytes).cast("B")
    scales = memoryview(scale_bytes).cast("B")
    expected_weights = out_dim * in_dim
    scale_rows = out_dim // block
    scale_cols = in_dim // block
    expected_scales = scale_rows * scale_cols
    if len(weights) != expected_weights:
        raise ValueError(f"weight byte length {len(weights)} != out_dim*in_dim {expected_weights}")
    if len(scales) != expected_scales:
        raise ValueError(f"scale byte length {len(scales)} != (out_dim/128)*(in_dim/128) {expected_scales}")

    out = array("f", [0.0]) * expected_weights
    for ob in range(scale_rows):
        for ib in range(scale_cols):
            scale = e8m0_to_f32(scales[ob * scale_cols + ib])
            for r in range(block):
                row = ob * block + r
                base = row * in_dim + ib * block
                for c in range(block):
                    idx = base + c
                    out[idx] = float(e4m3fn_to_f32(weights[idx]) * scale)
    return out


def f32_to_bf16_bits(f: float) -> int:
    """Round float32 to BF16 bits using round-to-nearest-even."""

    u = struct.unpack("<I", struct.pack("<f", float(f)))[0]
    return int((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


def bf16_bits_to_f32(h: int) -> float:
    """Decode BF16 bits by zero-extending into an IEEE float32."""

    return struct.unpack("<f", struct.pack("<I", (int(h) & 0xFFFF) << 16))[0]


__all__ = [
    "decode_f8_e4m3_e8m0_to_float32",
    "e4m3fn_to_f32",
    "e8m0_to_f32",
    "f32_to_bf16_bits",
    "bf16_bits_to_f32",
]
