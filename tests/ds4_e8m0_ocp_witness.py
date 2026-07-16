#!/usr/bin/env python3
"""Standalone OCP MX v1.0 E8M0 witness for DS4 routed I8 weights.

Math authority: OCP Microscaling Formats (MX) Specification v1.0 (Final),
E8M0 shared scale semantics: scale = 2^(e - 127), e=255 is reserved for NaN,
e=0 denotes the subnormal scale 2^(-127) rather than zero, and there is no
infinity scale encoding.
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
Peer-reviewed backing: Darvish Rouhani et al., arXiv:2310.10537.

DeepSeek-V4 routed experts use the real checkpoint header geometry
I8 [rows, cols] + F8_E8M0 [rows, cols/16], so block_size=16 on axis=1. They do
not apply the MXINT8 implicit 2^(-6) element factor; the decoded value here is
exactly signed_int8 * 2^(e - 127).
"""

from __future__ import annotations

import math
from array import array


def _signed_i8(value: int) -> int:
    return value - 256 if value >= 128 else value


def decode_i8_e8m0_to_float32(
    weight_bytes: bytes | bytearray | memoryview,
    scale_bytes: bytes | bytearray | memoryview,
    *,
    rows: int,
    cols: int,
    block_size: int = 16,
) -> array:
    """Decode row-major I8 weights using row-major E8M0 block scales.

    The scale tensor is indexed as scale[row, col // 16], matching the
    header-derived axis=1/block_size=16 routed-expert layout. The computation is
    intentionally tiny and auditable: stdlib math.ldexp is the spec expression
    for multiplying by an exact power of two.
    """

    if block_size != 16:
        raise ValueError("DeepSeek-V4 routed E8M0 witness requires block_size=16")
    if rows <= 0 or cols <= 0 or cols % block_size != 0:
        raise ValueError(f"invalid routed shape rows={rows} cols={cols} block_size={block_size}")

    weights = memoryview(weight_bytes).cast("B")
    scales = memoryview(scale_bytes).cast("B")
    expected_weights = rows * cols
    scale_cols = cols // block_size
    expected_scales = rows * scale_cols
    if len(weights) != expected_weights:
        raise ValueError(f"weight byte length {len(weights)} != rows*cols {expected_weights}")
    if len(scales) != expected_scales:
        raise ValueError(f"scale byte length {len(scales)} != rows*(cols/16) {expected_scales}")

    out = array("f")
    append = out.append
    for row in range(rows):
        weight_row = row * cols
        scale_row = row * scale_cols
        for block in range(scale_cols):
            e = int(scales[scale_row + block])
            start = weight_row + block * block_size
            if e == 255:
                for _ in range(block_size):
                    append(float("nan"))
                continue
            scale = math.ldexp(1.0, e - 127)
            for col in range(block_size):
                append(float(_signed_i8(int(weights[start + col])) * scale))
    return out


__all__ = ["decode_i8_e8m0_to_float32"]
