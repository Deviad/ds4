"""Pure-Python DeepSeek V4 dequantization parity helpers.

This module is deliberately tiny and dependency-free.  It mirrors the existing
`scripts/shim_ds4_safetensors.py` FP8 byte semantics for tests and cheap gates;
it does not read safetensors shards and implements packed expert formats only
when their semantics are proven.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


Shape = tuple[int, ...]

# OCP MXFP4 E2M1 (1 sign, 2 exp, 1 mantissa, bias=1). Index = 4-bit code 0..15.
# Codes 8..15 are the negative half (-0.0 .. -6.0).
_E2M1_FP4_LUT = (
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _as_shape(shape: Sequence[int]) -> Shape:
    out = tuple(int(dim) for dim in shape)
    if any(dim < 0 for dim in out):
        raise ValueError(f"shape dimensions must be non-negative, got {shape!r}")
    return out


def _product(shape: Shape) -> int:
    n = 1
    for dim in shape:
        n *= dim
    return n


def f8_e4m3fn_to_float(byte: int) -> float:
    """Decode one F8_E4M3FN byte using the existing shim semantics.

    The DeepSeek shim treats only 0x7f and 0xff as NaN and maps exponent 0b1111
    finite values through the normal E4M3 formula, matching safetensors/PyTorch
    float8_e4m3fn behavior used by the current compatibility path.
    """

    b = int(byte) & 0xFF
    if b in (0x7F, 0xFF):
        return float("nan")
    sign = -1.0 if (b & 0x80) else 1.0
    exp = (b >> 3) & 0x0F
    mant = b & 0x07
    bias = 7
    if exp == 0:
        if mant == 0:
            return -0.0 if sign < 0 else 0.0
        return sign * (mant / 8.0) * (2.0 ** (1 - bias))
    return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - bias))


def f8_e8m0_scale_to_float(byte: int) -> float:
    """Decode one F8_E8M0 scale byte using the existing shim semantics."""

    b = int(byte) & 0xFF
    if b == 0:
        return 2.0 ** -127
    if b == 255:
        return float("nan")
    return 2.0 ** (b - 127)


def decode_f8_e4m3fn(raw: bytes | bytearray | memoryview) -> list[float]:
    return [f8_e4m3fn_to_float(byte) for byte in bytes(raw)]


def decode_f8_e8m0_scales(raw: bytes | bytearray | memoryview) -> list[float]:
    return [f8_e8m0_scale_to_float(byte) for byte in bytes(raw)]


def _row_major_coords(index: int, shape: Shape) -> tuple[int, ...]:
    if not shape:
        return ()
    coords = [0] * len(shape)
    remaining = index
    for axis in range(len(shape) - 1, -1, -1):
        dim = shape[axis]
        if dim == 0:
            coords[axis] = 0
        else:
            coords[axis] = remaining % dim
            remaining //= dim
    return tuple(coords)


def _row_major_index(coords: tuple[int, ...], shape: Shape) -> int:
    index = 0
    for coord, dim in zip(coords, shape, strict=True):
        index = index * dim + coord
    return index


def _broadcast_scale_index(value_index: int, value_shape: Shape, scale_shape: Shape) -> int:
    if len(value_shape) != len(scale_shape):
        raise ValueError(f"scale_shape rank {len(scale_shape)} must match value_shape rank {len(value_shape)} for explicit broadcast")
    value_coords = _row_major_coords(value_index, value_shape)
    scale_coords: list[int] = []
    for axis, (coord, value_dim, scale_dim) in enumerate(zip(value_coords, value_shape, scale_shape, strict=True)):
        if scale_dim == value_dim:
            scale_coords.append(coord)
        elif scale_dim == 1:
            scale_coords.append(0)
        else:
            raise ValueError(f"scale_shape is not broadcast-compatible at axis {axis}: value_dim={value_dim}, scale_dim={scale_dim}")
    return _row_major_index(tuple(scale_coords), scale_shape)


def dequantize_f8_e4m3fn_with_e8m0_scales(
    values: bytes | bytearray | memoryview,
    scales: bytes | bytearray | memoryview,
    *,
    value_shape: Sequence[int],
    scale_shape: Sequence[int],
) -> list[float]:
    """Decode F8_E4M3FN values and apply explicit F8_E8M0 scales.

    Broadcasting is intentionally narrow: `scale_shape` must have the same rank
    as `value_shape`, and each scale dimension must be either 1 or exactly the
    corresponding value dimension.  This avoids silently guessing DeepSeek V4
    block layouts before architecture parity is proven.
    """

    value_shape_t = _as_shape(value_shape)
    scale_shape_t = _as_shape(scale_shape)
    value_raw = bytes(values)
    scale_raw = bytes(scales)
    expected_values = _product(value_shape_t)
    expected_scales = _product(scale_shape_t)
    if len(value_raw) != expected_values:
        raise ValueError(f"values byte length {len(value_raw)} does not match value_shape product {expected_values}")
    if len(scale_raw) != expected_scales:
        raise ValueError(f"scales byte length {len(scale_raw)} does not match scale_shape product {expected_scales}")

    decoded_values = decode_f8_e4m3fn(value_raw)
    decoded_scales = decode_f8_e8m0_scales(scale_raw)
    out: list[float] = []
    for i, value in enumerate(decoded_values):
        scale = decoded_scales[_broadcast_scale_index(i, value_shape_t, scale_shape_t)]
        if math.isnan(value) or math.isnan(scale):
            out.append(float("nan"))
        else:
            out.append(value * scale)
    return out


def _derive_nd_block_shape(value_shape: Shape, scale_shape: Shape) -> Shape:
    if len(value_shape) != len(scale_shape):
        raise ValueError(f"scale_shape rank {len(scale_shape)} must match value_shape rank {len(value_shape)}")
    block_shape: list[int] = []
    for axis, (value_dim, scale_dim) in enumerate(zip(value_shape, scale_shape, strict=True)):
        if scale_dim <= 0:
            raise ValueError(f"scale_dim at axis {axis} must be positive, got {scale_dim}")
        if value_dim % scale_dim != 0:
            raise ValueError(
                f"value_shape axis {axis} dimension {value_dim} is not divisible by scale_shape dimension {scale_dim}"
            )
        block_shape.append(value_dim // scale_dim)
    return tuple(block_shape)


def dequantize_f8_e4m3_e8m0_2d_block_scale(
    values: bytes | bytearray | memoryview,
    scales: bytes | bytearray | memoryview,
    *,
    value_shape: Sequence[int],
    scale_shape: Sequence[int],
) -> list[float]:
    """Decode F8_E4M3FN values with F8_E8M0 N-D sub-block scales.

    The block geometry is derived only from metadata shapes: each
    `scale_shape` axis must divide the corresponding `value_shape` axis, and
    `scale_index = coord // block_shape` in row-major order.  This models the
    real shared expert 128x128 two-dimensional sub-block layout without
    changing the older explicit-broadcast helper. Story 11.24 proves this path
    against the closed form `e4m3 * 2^(scale-127)` on the real
    `[2048,4096]+[16,32]` geometry plus a NaN-aware sweep.
    """

    value_shape_t = _as_shape(value_shape)
    scale_shape_t = _as_shape(scale_shape)
    block_shape = _derive_nd_block_shape(value_shape_t, scale_shape_t)
    value_raw = bytes(values)
    scale_raw = bytes(scales)
    expected_values = _product(value_shape_t)
    expected_scales = _product(scale_shape_t)
    if len(value_raw) != expected_values:
        raise ValueError(f"values byte length {len(value_raw)} does not match value_shape product {expected_values}")
    if len(scale_raw) != expected_scales:
        raise ValueError(f"scales byte length {len(scale_raw)} does not match scale_shape product {expected_scales}")

    decoded_values = decode_f8_e4m3fn(value_raw)
    decoded_scales = decode_f8_e8m0_scales(scale_raw)
    out: list[float] = []
    for i, value in enumerate(decoded_values):
        coords = _row_major_coords(i, value_shape_t)
        scale_coords = tuple(coord // block for coord, block in zip(coords, block_shape, strict=True))
        scale = decoded_scales[_row_major_index(scale_coords, scale_shape_t)]
        if math.isnan(value) or math.isnan(scale):
            out.append(float("nan"))
        else:
            out.append(value * scale)
    return out


def _signed_i8(byte: int) -> int:
    value = int(byte) & 0xFF
    return value - 256 if value >= 128 else value


def _bf16_to_float(raw: bytes | bytearray | memoryview) -> float:
    """Decode one little-endian BF16 value to Python float."""

    import struct

    raw_bytes = bytes(raw)
    if len(raw_bytes) != 2:
        raise ValueError("BF16 value must be exactly 2 bytes")
    u16 = struct.unpack("<H", raw_bytes)[0]
    sign = u16 >> 15
    exp = (u16 >> 7) & 0xFF
    mant = u16 & 0x7F
    if exp == 0xFF:
        if mant:
            return float("nan")
        return float("-inf") if sign else float("inf")
    if exp == 0:
        value = (mant / 128.0) * (2.0 ** -126)
    else:
        value = (1.0 + mant / 128.0) * (2.0 ** (exp - 127))
    return -value if sign else value


def _decode_bf16(raw: bytes | bytearray | memoryview) -> list[float]:
    raw_bytes = bytes(raw)
    if len(raw_bytes) % 2 != 0:
        raise ValueError("BF16 byte length must be a multiple of 2")
    return [_bf16_to_float(raw_bytes[i : i + 2]) for i in range(0, len(raw_bytes), 2)]


def dequantize_i8_affine(
    payload: bytes | bytearray | memoryview,
    *,
    scales: Sequence[float],
    zero_points: Sequence[int],
    shape: Sequence[int],
    scale_shape: Sequence[int],
) -> list[float]:
    """Decode an explicit signed-I8 affine fixture.

    Formula: `(signed_i8 - zero_point) * scale` with the same narrow explicit
    broadcast rules used by the FP8 scale helper.  This is a tiny primitive
    fixture, not proof of the real DeepSeek V4 packed expert layout.
    """

    value_shape_t = _as_shape(shape)
    scale_shape_t = _as_shape(scale_shape)
    raw = bytes(payload)
    expected_values = _product(value_shape_t)
    expected_scales = _product(scale_shape_t)
    if len(raw) != expected_values:
        raise ValueError(f"payload byte length {len(raw)} does not match shape product {expected_values}")
    if len(scales) != expected_scales:
        raise ValueError(f"scale count {len(scales)} does not match scale_shape product {expected_scales}")
    if len(zero_points) != expected_scales:
        raise ValueError(f"zero point count {len(zero_points)} does not match scale_shape product {expected_scales}")
    out: list[float] = []
    for i, byte in enumerate(raw):
        scale_index = _broadcast_scale_index(i, value_shape_t, scale_shape_t)
        out.append((_signed_i8(byte) - int(zero_points[scale_index])) * float(scales[scale_index]))
    return out


def run_tiny_i8_affine_fixture() -> dict[str, object]:
    got = dequantize_i8_affine(bytes([0, 127, 128, 255]), scales=[0.5], zero_points=[0], shape=(4,), scale_shape=(1,))
    expected = [0.0, 63.5, -64.0, -0.5]
    max_abs_error = max(abs(a - b) for a, b in zip(got, expected))
    return {
        "fixture": "i8-affine-explicit",
        "status": "ok" if max_abs_error <= 1e-12 else "failed",
        "max_abs_error": max_abs_error,
        "covered": ["signed I8 decode", "explicit affine scale", "explicit zero point", "narrow broadcast"],
        "not_covered": ["not a packed expert layout claim", "FP4", "blockwise expert packing"],
    }


def _apply_i8_block_scales(
    raw_payload: bytes,
    scale_values: list[float],
    *,
    value_shape: Shape,
    block_size: int,
    axis: int,
) -> list[float]:
    expected_payload = _product(value_shape)
    if len(raw_payload) != expected_payload:
        raise ValueError(
            f"payload byte length {len(raw_payload)} does not match shape product {expected_payload}"
        )
    scale_shape_t = _expected_scale_shape(value_shape, block_size=block_size, scale_axis=axis)
    expected_scale_count = _product(scale_shape_t)
    if len(scale_values) != expected_scale_count:
        raise ValueError(
            f"scale count {len(scale_values)} does not match expected {expected_scale_count} "
            f"for scale_shape {scale_shape_t}"
        )

    out: list[float] = []
    for i, byte in enumerate(raw_payload):
        coords = _row_major_coords(i, value_shape)
        block_coords = list(coords)
        block_coords[axis] = coords[axis] // block_size
        scale_index = _row_major_index(tuple(block_coords), scale_shape_t)
        out.append(_signed_i8(byte) * scale_values[scale_index])
    return out


def dequantize_i8_block_scale(
    payload: bytes | bytearray | memoryview,
    scales: bytes | bytearray | memoryview,
    *,
    shape: Sequence[int],
    block_size: int,
    scale_axis: int,
) -> list[float]:
    """Decode signed I8 values and apply per-block BF16 scales.

    The layout is inferred from `shape`, `block_size`, and `scale_axis`: each
    contiguous block of `block_size` elements along `scale_axis` shares one
    scale.  The axis dimension must be divisible by `block_size`; partial
    blocks are rejected because their padding semantics have not been verified.

    This helper is intended for the DeepSeek V4 routed-expert layout where the
    header scan found I8 weights paired with BF16 scales, block size 16, axis 1.
    """

    value_shape = _as_shape(shape)
    axis = _normalize_axis(scale_axis, len(value_shape))
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    axis_dim = value_shape[axis]
    if axis_dim % block_size != 0:
        raise ValueError(
            f"shape axis {axis} dimension {axis_dim} is not divisible by block_size {block_size}"
        )
    scale_shape_t = _expected_scale_shape(value_shape, block_size=block_size, scale_axis=axis)
    expected_payload = _product(value_shape)
    expected_scale_bytes = _product(scale_shape_t) * 2
    raw_payload = bytes(payload)
    raw_scales = bytes(scales)
    if len(raw_payload) != expected_payload:
        raise ValueError(
            f"payload byte length {len(raw_payload)} does not match shape product {expected_payload}"
        )
    if len(raw_scales) != expected_scale_bytes:
        raise ValueError(
            f"scales byte length {len(raw_scales)} does not match expected {expected_scale_bytes} "
            f"for scale_shape {scale_shape_t}"
        )

    scale_values = _decode_bf16(raw_scales)
    return _apply_i8_block_scales(
        raw_payload,
        scale_values,
        value_shape=value_shape,
        block_size=block_size,
        axis=axis,
    )


def dequantize_i8_e8m0_block_scale(
    payload: bytes | bytearray | memoryview,
    scales: bytes | bytearray | memoryview,
    *,
    shape: Sequence[int],
    block_size: int,
    scale_axis: int,
) -> list[float]:
    """Decode signed I8 values with per-block F8_E8M0 (UE8M0) scales.

    Real DeepSeek V4 routed experts are stored as I8 weights paired with
    1-byte exponent-only scales. Partial blocks are rejected because padding
    semantics remain unproven.
    """

    value_shape = _as_shape(shape)
    axis = _normalize_axis(scale_axis, len(value_shape))
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    axis_dim = value_shape[axis]
    if axis_dim % block_size != 0:
        raise ValueError(
            f"shape axis {axis} dimension {axis_dim} is not divisible by block_size {block_size}"
        )
    scale_shape_t = _expected_scale_shape(value_shape, block_size=block_size, scale_axis=axis)
    expected_payload = _product(value_shape)
    expected_scale_bytes = _product(scale_shape_t)
    raw_payload = bytes(payload)
    raw_scales = bytes(scales)
    if len(raw_payload) != expected_payload:
        raise ValueError(
            f"payload byte length {len(raw_payload)} does not match shape product {expected_payload}"
        )
    if len(raw_scales) != expected_scale_bytes:
        raise ValueError(
            f"scales byte length {len(raw_scales)} does not match expected {expected_scale_bytes} "
            f"for scale_shape {scale_shape_t}"
        )
    return _apply_i8_block_scales(
        raw_payload,
        decode_f8_e8m0_scales(raw_scales),
        value_shape=value_shape,
        block_size=block_size,
        axis=axis,
    )


def dequantize_fp4_block_scale(
    payload: bytes | bytearray | memoryview,
    scales: bytes | bytearray | memoryview,
    *,
    shape: Sequence[int],
    block_size: int = 32,
    scale_axis: int = 1,
) -> list[float]:
    """Pure-Python OCP MXFP4 E2M1 → float dequant reference (parity authority).

    `shape` is the LOGICAL value shape [out, in_logical]; `payload` holds
    in_logical//2 packed bytes per row (2 E2M1 nibbles/byte, LSB-first).
    `scales` are raw BF16 bytes, one per 32-logical block, linear domain
    (direct multiply, NOT E8M0). Math derives from OCP MXFP4 spec + HF mxfp4
    reference; NOT from any FROZEN i8/e8m0/f8 primitive (ADR 0017).
    """

    value_shape = _as_shape(shape)
    axis = _normalize_axis(scale_axis, len(value_shape))
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if block_size != 32:
        raise ValueError("fp4 block_size must be 32 (OCP MXFP4 default; BA Q2 LOCKED)")
    in_logical = value_shape[axis]
    if in_logical % block_size != 0:
        raise ValueError(f"logical axis {axis} dim {in_logical} not divisible by block_size {block_size}")
    expected_values = _product(value_shape)
    expected_payload = expected_values // 2
    raw_payload = bytes(payload)
    raw_scales = bytes(scales)
    if expected_values % 2 != 0:
        raise ValueError(f"logical value count {expected_values} must be even for packed FP4")
    if len(raw_payload) != expected_payload:
        raise ValueError(f"payload byte length {len(raw_payload)} does not match packed product {expected_payload}")
    scale_shape_t = _expected_scale_shape(value_shape, block_size=block_size, scale_axis=axis)
    expected_scale_count = _product(scale_shape_t)
    expected_scale_bytes = 2 * expected_scale_count
    if len(raw_scales) != expected_scale_bytes:
        raise ValueError(f"scales byte length {len(raw_scales)} does not match expected {expected_scale_bytes} (BF16)")
    scale_values = _decode_bf16(raw_scales)

    decoded: list[float] = []
    for byte in raw_payload:
        decoded.append(_E2M1_FP4_LUT[byte & 0x0F])
        decoded.append(_E2M1_FP4_LUT[(byte >> 4) & 0x0F])

    out: list[float] = []
    for i, value in enumerate(decoded):
        coords = _row_major_coords(i, value_shape)
        block_coords = list(coords)
        block_coords[axis] = coords[axis] // block_size
        scale_index = _row_major_index(tuple(block_coords), scale_shape_t)
        out.append(value * scale_values[scale_index])
    return out


def run_tiny_i8_block_scale_fixture() -> dict[str, object]:
    """Deterministic synthetic I8 block-scale dequant parity fixture.

    Compares the pure-Python helper against a PyTorch reference on a tiny
    matrix.  This proves the helper's own semantics, not full DS4/Transformers
    checkpoint decode semantics.
    """

    try:
        import torch
    except ImportError:
        return {
            "fixture": "i8-block-scale-synthetic",
            "status": "skipped",
            "max_abs_error": None,
            "reason": "torch not available",
        }

    shape = (2, 16)
    block_size = 16
    scale_axis = 1
    payload = bytes((i * 13 + 5) % 256 for i in range(_product(_as_shape(shape))))
    scale_shape = _expected_scale_shape(shape, block_size=block_size, scale_axis=scale_axis)
    scales_f32 = torch.tensor([[0.5], [1.5]], dtype=torch.float32)
    scales_bf16_bytes = scales_f32.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()

    weight = torch.frombuffer(bytearray(payload), dtype=torch.int8).reshape(shape)
    expected = (weight.float() * scales_f32).flatten().tolist()
    got = dequantize_i8_block_scale(
        payload,
        scales_bf16_bytes,
        shape=shape,
        block_size=block_size,
        scale_axis=scale_axis,
    )
    max_abs_error = max(abs(a - b) for a, b in zip(got, expected))
    return {
        "fixture": "i8-block-scale-synthetic",
        "status": "ok" if max_abs_error <= 1e-3 else "failed",
        "max_abs_error": max_abs_error,
        "covered": [
            "signed I8 decode",
            "BF16 scale decode",
            "block size 16 along axis 1",
            "row-major block scale lookup",
            "PyTorch reference parity",
        ],
        "not_covered": [
            "real checkpoint payload decode",
            "non-divisible block boundaries",
            "packed expert dispatch",
            "DS4 runtime verification",
        ],
    }


def _normalize_axis(axis: int, rank: int) -> int:
    normalized = int(axis)
    if normalized < 0:
        normalized += rank
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"scale_axis {axis} is out of range for rank {rank}")
    return normalized


def _packed_expected_payload_bytes(kind: str, value_count: int) -> int | None:
    if kind == "fp4":
        return (value_count + 1) // 2
    if kind == "i8":
        return value_count
    return None


def _expected_scale_shape(value_shape: Shape, *, block_size: int, scale_axis: int) -> Shape:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    axis = _normalize_axis(scale_axis, len(value_shape))
    out = list(value_shape)
    axis_dim = out[axis]
    out[axis] = (axis_dim + block_size - 1) // block_size
    return tuple(out)


_PACKED_UNKNOWN_FACTS = {
    "fp4": (
        "trusted_reference",
        "fp4_encoding",
        "fp4_nibble_order",
        "fp4_signedness_or_zero_point",
        "scale_application_order",
        "block_padding_behavior",
        "scale_dtype",
        "scale_axis",
        "block_size",
        "axis_layout",
    ),
    "i8": (
        "trusted_reference",
        "i8_signedness_or_zero_point",
        "scale_application_order",
        "block_padding_behavior",
        "scale_dtype",
        "scale_axis",
        "block_size",
        "axis_layout",
    ),
}


def _packed_unknown_facts(kind: str) -> tuple[str, ...]:
    return _PACKED_UNKNOWN_FACTS.get(str(kind).casefold(), ("supported_packing_kind", "trusted_reference"))


def describe_packed_expert_risk(
    kind: str,
    *,
    payload: bytes | bytearray | memoryview,
    scales: bytes | bytearray | memoryview | None,
    shape: Sequence[int],
    block_size: int | None = None,
    scale_axis: int | None = None,
    scale_shape: Sequence[int] | None = None,
) -> dict[str, object]:
    """Return deterministic fail-closed evidence for packed expert payloads.

    This helper intentionally does not decode payload bytes.  It records the
    minimum metadata still needed before `dequantize_expert_packed` may safely
    support DeepSeek V4 FP4/I8 expert weights.
    """

    normalized = str(kind).casefold()
    shape_t = _as_shape(shape)
    value_count = _product(shape_t)
    payload_len = len(payload)
    scale_len = None if scales is None else len(scales)
    plausible_payload_bytes = _packed_expected_payload_bytes(normalized, value_count)
    payload_plausible = plausible_payload_bytes is not None and payload_len == plausible_payload_bytes
    payload_size = {
        "expected_bytes": plausible_payload_bytes,
        "actual_bytes": payload_len,
        "plausible": payload_plausible,
        "delta_bytes": None if plausible_payload_bytes is None else payload_len - plausible_payload_bytes,
    }
    block_report: dict[str, object] | None = None
    scale_relationship: dict[str, object]
    if block_size is not None and scale_axis is not None and scale_shape is not None:
        axis = _normalize_axis(scale_axis, len(shape_t))
        expected_shape_t = _expected_scale_shape(shape_t, block_size=int(block_size), scale_axis=axis)
        scale_shape_t = _as_shape(scale_shape)
        expected_scale_count = _product(expected_shape_t)
        actual_scale_count = _product(scale_shape_t)
        issues: list[str] = []
        if scale_shape_t != expected_shape_t:
            issues.append("scale_shape_mismatch")
        if scales is None:
            issues.append("scale_payload_missing")
        block_report = {
            "block_size": int(block_size),
            "axis": axis,
            "axis_dim": shape_t[axis],
            "blocks_per_axis": expected_shape_t[axis],
            "expected_scale_shape": list(expected_shape_t),
        }
        scale_relationship = {
            "status": "plausible" if not issues else "mismatch",
            "scale_shape": list(scale_shape_t),
            "expected_scale_shape": list(expected_shape_t),
            "actual_scale_count": actual_scale_count,
            "expected_scale_count": expected_scale_count,
            "scale_bytes": scale_len,
            "issues": issues,
        }
    else:
        issues = []
        if block_size is None:
            issues.append("block_size_missing")
        if scale_axis is None:
            issues.append("scale_axis_missing")
        if scale_shape is None:
            issues.append("scale_shape_missing")
        scale_relationship = {
            "status": "unknown",
            "scale_shape": None if scale_shape is None else list(_as_shape(scale_shape)),
            "expected_scale_shape": None,
            "actual_scale_count": None if scale_shape is None else _product(_as_shape(scale_shape)),
            "expected_scale_count": None,
            "scale_bytes": scale_len,
            "issues": issues,
        }
    missing = list(_packed_unknown_facts(normalized))
    if block_size is not None and "block_size" in missing:
        missing.remove("block_size")
    if scale_axis is not None and "scale_axis" in missing:
        missing.remove("scale_axis")
    if scale_shape is not None and "scale_shape" in missing:
        missing.remove("scale_shape")
    if not payload_plausible:
        missing.append("payload_size_mismatch")
    if scale_relationship["status"] == "mismatch":
        missing.append("scale_relationship_mismatch")
    return {
        "fixture": "packed-expert-risk-report",
        "status": "blocked",
        "packing": normalized,
        "shape": list(shape_t),
        "value_count": value_count,
        "payload_bytes": payload_len,
        "scale_bytes": scale_len,
        "plausible_payload_bytes": plausible_payload_bytes,
        "payload_size": payload_size,
        "block": block_report,
        "scale_relationship": scale_relationship,
        "missing": missing,
        "covered": ["metadata accounting", "payload size plausibility", "scale relationship accounting", "fail-closed risk classification"],
        "not_covered": ["does not decode payload bytes", "no packed expert parity claim"],
    }


def _infer_scale_block_layout(
    weight_shape: Sequence[int],
    scale_shape: Sequence[int],
) -> dict[str, object]:
    """Infer block size and axis from a paired weight/scale shape.

    Returns the unique integer quotient weight_dim / scale_dim for any axis
    where the division is exact and scale_dim > 0.  If more than one axis
    reduces exactly, the layout is ambiguous and must be verified against a
    trusted reference.
    """

    weight_t = _as_shape(weight_shape)
    scale_t = _as_shape(scale_shape)
    if len(weight_t) != len(scale_t):
        return {"status": "rank_mismatch", "block_size": None, "axis": None}
    candidates: list[tuple[int, int]] = []
    for axis, (wd, sd) in enumerate(zip(weight_t, scale_t)):
        if sd <= 0:
            continue
        if wd % sd == 0 and wd != sd:
            candidates.append((axis, wd // sd))
    if not candidates:
        return {"status": "no_block_axis", "block_size": None, "axis": None}
    if len(candidates) != 1:
        return {"status": "ambiguous", "block_size": None, "axis": None, "candidates": candidates}
    axis, block_size = candidates[0]
    return {"status": "inferred", "block_size": block_size, "axis": axis}


# Common regex for routed/shared expert tensor names.
_ROUTED_EXPERT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\.w(?P<wkind>[123])\.(?P<suffix>weight|scale)$"
)
_SHARED_EXPERT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.shared_experts\.w(?P<wkind>[123])\.(?P<suffix>weight|scale)$"
)


def read_safetensors_header(
    path: str | Path,
    *,
    max_bytes: int = 2 * 1024 * 1024,
) -> dict[str, dict[str, object]]:
    """Read only the JSON header of a safetensors file.

    Does not read or decode any tensor payload bytes.  Opens the file and reads
    exactly the header-length prefix; `max_bytes` is a safety cap on the header
    read size.
    """

    import struct

    p = Path(path)
    with p.open("rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError(f"{p}: file too short for safetensors header")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        if 8 + header_len > max_bytes:
            raise ValueError(f"{p}: safetensors header length {header_len} exceeds max_bytes {max_bytes}")
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"{p}: truncated safetensors header: expected {header_len} bytes, got {len(header_bytes)}")
        header = json.loads(header_bytes)
        if not isinstance(header, dict):
            raise ValueError(f"{p}: safetensors header JSON must be an object")
        return header


def _header_meta_dtype_shape(meta: Mapping[str, object]) -> dict[str, object]:
    """Copy only safetensors header fields needed by metadata classifiers."""

    copied: dict[str, object] = {"dtype": meta.get("dtype", "unknown")}
    shape = meta.get("shape", None)
    if shape is not None:
        copied["shape"] = shape
    return copied


def _single_dtype_from_counts(counts: Mapping[str, int]) -> str | None:
    if len(counts) != 1:
        return None
    return next(iter(counts))


def _classify_packed_weight_dtype(weight_dtype: str | None, has_scale: bool, scale_dtype: str | None) -> str | None:
    if weight_dtype is None:
        return None
    folded = weight_dtype.casefold()
    if folded in {"i8", "int8"}:
        return "i8"
    if has_scale and (folded in {"u8", "uint8"} or _looks_fp4_dtype(weight_dtype)):
        _ = scale_dtype
        return "fp4"
    return None


def _packed_decode_family_report(
    weights: Mapping[str, Mapping[str, object]],
    scales: Mapping[str, Mapping[str, object]],
) -> tuple[str | None, str | None, str | None, list[str], list[str]]:
    weight_counts: dict[str, int] = {}
    scale_counts: dict[str, int] = {}
    kinds: set[str] = set()
    unrecognized: set[str] = set()
    known_nonpacked = {"BF16", "F16", "F32", "F64", "F8_E4M3", "F8_E8M0"}
    for wname in sorted(weights):
        weight_dtype = str(weights[wname].get("dtype", "unknown"))
        weight_counts[weight_dtype] = weight_counts.get(weight_dtype, 0) + 1
        sname = wname.replace(".weight", ".scale")
        scale_dtype = None
        if sname in scales:
            scale_dtype = str(scales[sname].get("dtype", "unknown"))
            scale_counts[scale_dtype] = scale_counts.get(scale_dtype, 0) + 1
        kind = _classify_packed_weight_dtype(weight_dtype, sname in scales, scale_dtype)
        if kind is not None:
            kinds.add(kind)
        elif weight_dtype.upper() not in known_nonpacked:
            unrecognized.add(weight_dtype)
    packing = next(iter(kinds)) if len(kinds) == 1 else ("mixed" if kinds else None)
    return packing, _single_dtype_from_counts(weight_counts), _single_dtype_from_counts(scale_counts), sorted(kinds), sorted(unrecognized)


def classify_expert_metadata_from_header(
    header: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Classify packed expert tensor metadata from a safetensors header.

    The input `header` maps tensor names to their safetensors metadata
    dictionaries (`dtype`, `shape`).  No payload bytes are decoded.  The
    function counts routed/shared expert dtypes, infers scale block layouts,
    checks weight/scale pairing, and records everything still unknown before
    I8/FP4 expert dequantization can be implemented safely.
    """

    routed_weights: dict[str, dict[str, object]] = {}
    routed_scales: dict[str, dict[str, object]] = {}
    shared_weights: dict[str, dict[str, object]] = {}
    shared_scales: dict[str, dict[str, object]] = {}

    for name, meta in header.items():
        m = _ROUTED_EXPERT_RE.match(str(name))
        if m:
            if m.group("suffix") == "weight":
                routed_weights[name] = _header_meta_dtype_shape(meta)
            else:
                routed_scales[name] = _header_meta_dtype_shape(meta)
            continue
        m = _SHARED_EXPERT_RE.match(str(name))
        if m:
            if m.group("suffix") == "weight":
                shared_weights[name] = _header_meta_dtype_shape(meta)
            else:
                shared_scales[name] = _header_meta_dtype_shape(meta)
            continue

    def _dtype_counts(tensors: Mapping[str, Mapping[str, object]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for meta in tensors.values():
            dtype = str(meta.get("dtype", "unknown"))
            counts[dtype] = counts.get(dtype, 0) + 1
        return counts

    def _first_shape(tensors: Mapping[str, Mapping[str, object]]) -> Shape | None:
        for meta in tensors.values():
            shape = meta.get("shape")
            if shape is not None:
                return _as_shape(shape)
        return None

    def _pairing_report(weights: Mapping[str, Mapping[str, object]], scales: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
        unpaired_weights = []
        for wname in weights:
            sname = wname.replace(".weight", ".scale")
            if sname not in scales:
                unpaired_weights.append(wname)
        unpaired_scales = []
        for sname in scales:
            wname = sname.replace(".scale", ".weight")
            if wname not in weights:
                unpaired_scales.append(sname)
        return {
            "weight_count": len(weights),
            "scale_count": len(scales),
            "paired_count": len(weights) - len(unpaired_weights),
            "unpaired_weight_examples": unpaired_weights[:10],
            "unpaired_scale_examples": unpaired_scales[:10],
        }

    routed_report = _pairing_report(routed_weights, routed_scales)
    shared_report = _pairing_report(shared_weights, shared_scales)

    def _aggregate_block_layout(weights: Mapping[str, Mapping[str, object]], scales: Mapping[str, Mapping[str, object]]) -> dict[str, object] | None:
        layouts: list[dict[str, object]] = []
        seen: set[tuple[int, int]] = set()
        all_inferred = True
        for wname in sorted(weights):
            sname = wname.replace(".weight", ".scale")
            if sname not in scales:
                continue
            layout = _infer_scale_block_layout(
                weights[wname]["shape"],  # type: ignore[arg-type]
                scales[sname]["shape"],  # type: ignore[arg-type]
            )
            layouts.append({"name": wname, **layout})
            if layout.get("status") == "inferred":
                seen.add((int(layout["axis"]), int(layout["block_size"])))
            else:
                all_inferred = False
        if not layouts:
            return None
        if all_inferred and len(seen) == 1:
            axis, block_size = next(iter(seen))
            return {"status": "inferred", "axis": axis, "block_size": block_size, "sample_count": len(layouts), "examples": layouts[:5]}
        return {"status": "mixed_or_ambiguous", "sample_count": len(layouts), "examples": layouts[:10]}

    routed_block_layout = _aggregate_block_layout(routed_weights, routed_scales)
    shared_block_layout = _aggregate_block_layout(shared_weights, shared_scales)
    routed_packing, routed_weight_dtype, routed_scale_dtype, routed_kinds, routed_unrecognized = _packed_decode_family_report(routed_weights, routed_scales)
    shared_packing, shared_weight_dtype, shared_scale_dtype, shared_kinds, shared_unrecognized = _packed_decode_family_report(shared_weights, shared_scales)
    packing_kinds_present = sorted(set(routed_kinds + shared_kinds))
    unrecognized_weight_dtypes = sorted(set(routed_unrecognized + shared_unrecognized))
    packed_unknowns = sorted({fact for kind in packing_kinds_present for fact in _packed_unknown_facts(kind)})
    if unrecognized_weight_dtypes:
        packed_unknowns = sorted(set(packed_unknowns).union(_packed_unknown_facts("unknown")))

    unknown = [
        "I8 signedness / affine formula",
        "scale application order (weight-first vs block-first)",
        "block padding behavior at axis boundaries",
        "axis layout broadcast semantics vs Transformers/DS4",
        "FP4 expert packing (if present elsewhere)",
        "reference verification of inferred scale block layout",
    ]
    unknown.extend(packed_unknowns)
    if not (routed_block_layout and routed_block_layout.get("status") == "inferred"):
        unknown.append("scale block layout")
    unknown = sorted(dict.fromkeys(unknown))

    packed_expert_decode = {
        "routed_packing": routed_packing,
        "shared_packing": shared_packing,
        "packing_kinds_present": packing_kinds_present,
        "routed_weight_dtype": routed_weight_dtype,
        "routed_scale_dtype": routed_scale_dtype,
        "shared_weight_dtype": shared_weight_dtype,
        "shared_scale_dtype": shared_scale_dtype,
        "fp4_candidate_basis": (
            "packed U8/UINT8 weight paired with scale tensor; nibble encoding/order unverified"
            if "fp4" in packing_kinds_present
            else None
        ),
        "block_layout_verified": False,
        "unrecognized_weight_dtypes": unrecognized_weight_dtypes,
        "unknown_facts": packed_unknowns,
    }

    return {
        "schema": 2,
        "status": "classified",
        "routed_experts": {
            "weight_count": len(routed_weights),
            "scale_count": len(routed_scales),
            "weight_dtype_counts": _dtype_counts(routed_weights),
            "scale_dtype_counts": _dtype_counts(routed_scales),
            "weight_shape_example": _first_shape(routed_weights),
            "scale_shape_example": _first_shape(routed_scales),
            "pairing": routed_report,
            "inferred_block_layout": routed_block_layout,
        },
        "shared_experts": {
            "weight_count": len(shared_weights),
            "scale_count": len(shared_scales),
            "weight_dtype_counts": _dtype_counts(shared_weights),
            "scale_dtype_counts": _dtype_counts(shared_scales),
            "weight_shape_example": _first_shape(shared_weights),
            "scale_shape_example": _first_shape(shared_scales),
            "pairing": shared_report,
            "inferred_block_layout": shared_block_layout,
        },
        "packed_expert_decode": packed_expert_decode,
        # Fail-closed until Story 11.15b proves packed expert decode parity.
        "can_decode_payload": False,
        "unknown_required_for_decode": unknown,
        "not_covered": ["does not read safetensors payload bytes", "does not decode I8/FP4 values"],
    }


_FP4_LIKE_DTYPE_TOKENS = frozenset({"fp4", "e2m1", "u4", "uint4", "int4", "nf4", "q4", "u8", "uint8"})
_KNOWN_EXPERT_DTYPES = frozenset({"I8", "INT8", "U8", "UINT8", "F8_E4M3", "F8_E8M0", "BF16", "F32"})


def _sorted_counts(counts: Mapping[str, object]) -> dict[str, int]:
    return {str(key): int(counts[key]) for key in sorted(counts)}


def _expert_dtype_strings(header: Mapping[str, Mapping[str, object]]) -> list[str]:
    dtypes: list[str] = []
    for name, meta in header.items():
        if _ROUTED_EXPERT_RE.match(str(name)) or _SHARED_EXPERT_RE.match(str(name)):
            dtypes.append(str(meta.get("dtype", "unknown")))
    return dtypes


def _looks_fp4_dtype(dtype: str) -> bool:
    folded = dtype.casefold()
    return any(token in folded for token in _FP4_LIKE_DTYPE_TOKENS)


def _declared_quant_report(declared_quant: Mapping[str, object] | None) -> dict[str, object]:
    if declared_quant is None:
        return {"present": False}
    report: dict[str, object] = {"present": True}
    preferred = ("quant_method", "fmt", "scale_fmt", "weight_block_size", "activation_scheme")
    for key in preferred:
        if key in declared_quant:
            value = declared_quant[key]
            if key == "weight_block_size" and isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                report[key] = [int(v) for v in value]
            else:
                report[key] = value
    for key in sorted(str(k) for k in declared_quant):
        if key not in report:
            report[key] = declared_quant[key]
    return report


def _declared_block_sizes(declared_quant: Mapping[str, object] | None) -> list[int] | None:
    if declared_quant is None or "weight_block_size" not in declared_quant:
        return None
    value = declared_quant["weight_block_size"]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [int(v) for v in value]
    return [int(value)]


def _layout_has_declared_2d_ambiguity(layout: Mapping[str, object] | None, declared_blocks: list[int]) -> bool:
    if not layout or layout.get("status") != "mixed_or_ambiguous":
        return False
    if len(declared_blocks) != 2:
        return False
    examples = layout.get("examples")
    if not isinstance(examples, Sequence):
        return False
    for example in examples:
        if not isinstance(example, Mapping) or example.get("status") != "ambiguous":
            return False
        candidates = example.get("candidates")
        if not isinstance(candidates, Sequence):
            return False
        candidate_blocks = sorted(int(candidate[1]) for candidate in candidates)  # type: ignore[index]
        if candidate_blocks != sorted(declared_blocks):
            return False
    return True


class ExpertBlockLayoutError(ValueError):
    """Routed/shared block geometry could not be resolved from metadata."""


def _mapping_at(value: object, key: str) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        child = value.get(key)
        if isinstance(child, Mapping):
            return child
    return None


def _shape_list(value: object) -> list[int] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [int(dim) for dim in value]
    return None


def _count_keys_upper(value: object) -> set[str]:
    if not isinstance(value, Mapping):
        return set()
    return {str(key).upper() for key in value}


def _shared_has_observed_clean_2d_tiling(shared: Mapping[str, object], declared_blocks: list[int] | None) -> bool:
    """Whether observed shared metadata is the proven clean 2-D E4M3/E8M0 tiling.

    The 2-D scale tensor determines the tiling: rank-2 shared weight/scale
    shapes with a 128x128 per-axis block, declared `[128,128]`, and the proven
    F8_E4M3 + E8M0 dtype pair are decoded by
    `dequantize_f8_e4m3_e8m0_2d_block_scale`. The older 1-D axis ambiguity does
    not apply once this observed 2-D scale tensor is present.
    """

    if declared_blocks != [128, 128]:
        return False
    weight_shape = _shape_list(shared.get("weight_shape_example"))
    scale_shape = _shape_list(shared.get("scale_shape_example"))
    if weight_shape is None or scale_shape is None or len(weight_shape) != 2 or len(scale_shape) != 2:
        return False
    if weight_shape[0] % 128 != 0 or weight_shape[1] % 128 != 0:
        return False
    if scale_shape != [weight_shape[0] // 128, weight_shape[1] // 128]:
        return False
    weight_dtypes = _count_keys_upper(shared.get("weight_dtype_counts"))
    scale_dtypes = _count_keys_upper(shared.get("scale_dtype_counts"))
    if weight_dtypes.isdisjoint({"F8_E4M3", "F8_E4M3FN", "E4M3"}):
        return False
    if scale_dtypes.isdisjoint({"F8_E8M0", "E8M0", "UE8M0"}):
        return False
    return True


def reconcile_routed_block_layout(report: Mapping[str, object]) -> dict[str, object]:
    """Explain routed expert block geometry using observed shapes as authority."""

    observed_packing = _mapping_at(report, "observed_packing")
    routed_observed = _mapping_at(observed_packing, "routed") if observed_packing is not None else None
    observed_layout = _mapping_at(routed_observed, "inferred_block_layout") if routed_observed is not None else None
    declared_quant = _mapping_at(report, "declared_quant")
    declared_blocks = None if declared_quant is None else _shape_list(declared_quant.get("weight_block_size"))

    if observed_layout is None:
        return {
            "status": "unresolved",
            "reason": "routed block layout unavailable in classifier metadata",
            "declared_weight_block_size": declared_blocks,
        }
    if observed_layout.get("status") != "inferred":
        layout_status = observed_layout.get("status", "unknown")
        return {
            "status": "unresolved",
            "reason": f"routed block layout is {layout_status}; shape ratio is ambiguous or unverified",
            "declared_weight_block_size": declared_blocks,
        }

    axis = int(observed_layout["axis"])
    block_size = int(observed_layout["block_size"])
    base = _mapping_at(report, "base")
    routed_base = _mapping_at(base, "routed_experts") if base is not None else None
    weight_shape = None if routed_base is None else _shape_list(routed_base.get("weight_shape_example"))
    scale_shape = None if routed_base is None else _shape_list(routed_base.get("scale_shape_example"))
    ratio = {"axis": axis, "weight_dim": None, "scale_dim": None, "block_size": block_size}
    if weight_shape is not None and scale_shape is not None and axis < len(weight_shape) and axis < len(scale_shape):
        ratio = {
            "axis": axis,
            "weight_dim": int(weight_shape[axis]),
            "scale_dim": int(scale_shape[axis]),
            "block_size": block_size,
        }
    declared_text = "not declared" if declared_blocks is None else str(declared_blocks)
    return {
        "status": "shape_authoritative",
        "geometry": {"axis": axis, "block_size": block_size},
        "observed_scale_ratio": ratio,
        "declared_weight_block_size": declared_blocks,
        "declared_role": "governs fmt=e4m3 fp8 path (shared/non-expert); advisory for the I8 routed micro-block",
        "discrepancy_explained": (
            f"declared {declared_text} != observed routed block_size {block_size}; "
            "observed weight/scale shapes are authoritative for routed expert geometry"
        ),
    }


def _config_consistency_for_layout(
    family: str,
    layout: Mapping[str, object] | None,
    declared_blocks: list[int] | None,
) -> dict[str, object]:
    if declared_blocks is None:
        return {"status": "not_declared", "reason": "declared weight_block_size absent"}
    if layout is None:
        return {"status": "no_observed_layout", "reason": f"no paired {family} expert weight/scale shapes", "declared": declared_blocks}
    if layout.get("status") == "inferred":
        observed = {"axis": int(layout["axis"]), "block_size": int(layout["block_size"])}
        if observed["block_size"] in declared_blocks:
            return {"status": "consistent", "reason": "observed block size is declared", "observed": observed, "declared": declared_blocks}
        return {
            "status": "discrepancy",
            "reason": f"observed_block_axis{observed['axis']}={observed['block_size']} != declared_weight_block_size={declared_blocks}",
            "observed": observed,
            "declared": declared_blocks,
        }
    if family == "shared" and _layout_has_declared_2d_ambiguity(layout, declared_blocks):
        return {
            "status": "consistent_but_ambiguous",
            "reason": f"{declared_blocks[0]}x{declared_blocks[1]} 2-D block matches declared but 1-D inference cannot disambiguate axis",
            "declared": declared_blocks,
        }
    return {
        "status": "ambiguous_or_unverified",
        "reason": f"observed {family} block layout is {layout.get('status')!r}; trusted reference required",
        "declared": declared_blocks,
    }


def resolve_routed_block_layout(report: Mapping[str, object]) -> dict[str, int]:
    """Resolve routed block geometry from classifier metadata only.

    Observed routed weight/scale shapes are authoritative.  Declared
    `weight_block_size` is still reported by the classifier, but it is advisory
    for the I8 routed micro-block and must not veto an unambiguous shape ratio.
    """

    config = _mapping_at(report, "config_consistency")
    routed_consistency = _mapping_at(config, "routed") if config is not None else None
    observed_packing = _mapping_at(report, "observed_packing")
    routed_observed = _mapping_at(observed_packing, "routed") if observed_packing is not None else None
    observed_layout = _mapping_at(routed_observed, "inferred_block_layout") if routed_observed is not None else None

    if observed_layout is not None:
        layout_status = str(observed_layout.get("status", ""))
        if layout_status == "inferred" and "axis" in observed_layout and "block_size" in observed_layout:
            return {"axis": int(observed_layout["axis"]), "block_size": int(observed_layout["block_size"])}
        if layout_status in {"mixed_or_ambiguous", "ambiguous"}:
            raise ExpertBlockLayoutError("routed block layout is ambiguous; refusing to guess")

    if routed_consistency is not None and str(routed_consistency.get("status", "")) == "consistent":
        observed = _mapping_at(routed_consistency, "observed")
        if observed is not None and "axis" in observed and "block_size" in observed:
            return {"axis": int(observed["axis"]), "block_size": int(observed["block_size"])}
    if routed_consistency is not None and str(routed_consistency.get("status", "")) == "ambiguous_or_unverified":
        raise ExpertBlockLayoutError("routed block layout is ambiguous; refusing to guess")
    raise ExpertBlockLayoutError("routed block layout unavailable in classifier metadata")


def classify_checkpoint_expert_packing(
    header: Mapping[str, Mapping[str, object]],
    *,
    declared_quant: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Classify real-checkpoint expert packing metadata, header-only.

    The checkpoint is FP8/UE8M0-family today (not FP4): routed experts are I8
    weights paired with F8_E8M0 scales. Routed-I8 E8M0 raw-decode is proven
    (Story 11.22: closed-form + shim-pipeline parity) and dispatched through
    `dequantize_expert_packed`. Shared-expert F8_E4M3 + E8M0 clean observed
    2-D `[rows/128, cols/128]` tiling is proven by
    `dequantize_f8_e4m3_e8m0_2d_block_scale` (Story 11.24); it stays
    fail-closed only when the 2-D scale tensor is not observed and 1-D inference
    is genuinely ambiguous. FP4/unrecognized dtypes also remain fail-closed. It
    reconciles observed metadata with the declared quantization config.
    """

    base = classify_expert_metadata_from_header(header)
    routed = base["routed_experts"]  # type: ignore[index]
    shared = base["shared_experts"]  # type: ignore[index]
    routed_layout = routed["inferred_block_layout"]  # type: ignore[index]
    shared_layout = shared["inferred_block_layout"]  # type: ignore[index]
    declared_blocks = _declared_block_sizes(declared_quant)
    config_consistency = {
        "routed": _config_consistency_for_layout("routed", routed_layout, declared_blocks),
        "shared": _config_consistency_for_layout("shared", shared_layout, declared_blocks),
    }

    dtypes = _expert_dtype_strings(header)
    fp4_like = sorted({dtype for dtype in dtypes if _looks_fp4_dtype(dtype)})
    unrecognized = sorted({dtype for dtype in dtypes if dtype not in _KNOWN_EXPERT_DTYPES})

    unknowns = {str(item) for item in base.get("unknown_required_for_decode", [])}
    routed_weights = _sorted_counts(routed["weight_dtype_counts"])  # type: ignore[index]
    routed_scales = _sorted_counts(routed["scale_dtype_counts"])  # type: ignore[index]
    shared_weights = _sorted_counts(shared["weight_dtype_counts"])  # type: ignore[index]
    shared_scales = _sorted_counts(shared["scale_dtype_counts"])  # type: ignore[index]

    shared_clean_2d = _shared_has_observed_clean_2d_tiling(shared, declared_blocks)
    # For a clean observed 2-D shared tensor, the proven closed form is exactly
    # decoded E4M3 value times decoded E8M0 scale; application order is no
    # longer a separate unknown. Missing/non-clean shared metadata stays closed.
    if (shared_weights or shared_scales) and not shared_clean_2d:
        unknowns.add("F8_E8M0 (UE8M0) scale application order for F8_E4M3 shared weights")
    if config_consistency["routed"].get("status") == "discrepancy":
        unknowns.add("routed block layout differs declared weight_block_size; needs trusted reference")
    if not shared_clean_2d and (
        config_consistency["shared"].get("status") == "consistent_but_ambiguous"
        or (isinstance(shared_layout, Mapping) and shared_layout.get("status") == "mixed_or_ambiguous")
    ):
        unknowns.add("shared 128x128 2-D block axis assignment (1-D inference ambiguous)")
    if fp4_like:
        unknowns.add("fp4 decode (no trusted reference)")
    if unrecognized:
        unknowns.add("unrecognized expert dtype(s): " + ", ".join(unrecognized))
    unknowns.add("trusted reference (DS4-CPU / Transformers) end-to-end expert decode parity")

    not_covered = [
        "does not read safetensors payload bytes",
        "does not decode packed expert payloads",
    ]
    if not shared_clean_2d:
        not_covered.append("does not prove shared F8_E4M3 scale application order (2-D axis unresolved)")
    not_covered.append("does not write forward parity markers")

    return {
        "schema": 2,
        "status": "classified",
        "base": base,
        "fp4_present": bool(fp4_like),
        "fp4_like_dtypes": fp4_like,
        "observed_packing": {
            "routed": {
                "weight_count": routed["weight_count"],  # type: ignore[index]
                "scale_count": routed["scale_count"],  # type: ignore[index]
                "weight_dtypes": routed_weights,
                "scale_dtypes": routed_scales,
                "inferred_block_layout": routed_layout,
                "pairing": routed["pairing"],  # type: ignore[index]
            },
            "shared": {
                "weight_count": shared["weight_count"],  # type: ignore[index]
                "scale_count": shared["scale_count"],  # type: ignore[index]
                "weight_dtypes": shared_weights,
                "scale_dtypes": shared_scales,
                "inferred_block_layout": shared_layout,
                "pairing": shared["pairing"],  # type: ignore[index]
            },
        },
        "declared_quant": _declared_quant_report(declared_quant),
        "config_consistency": config_consistency,
        "unrecognized_dtypes": unrecognized,
        "can_decode_payload": False,
        "unknown_required_for_decode": sorted(unknowns),
        "not_covered": not_covered,
    }


def dequantize_expert_packed(
    kind: str,
    payload: bytes,
    *,
    scales: bytes | None,
    shape: Sequence[int],
    block_size: int | None = None,
    scale_axis: int | None = None,
) -> list[float]:
    """Decode the proven packed-expert wrapper cases.

    The `i8` branch is a metadata-gated router: BF16 (2 bytes/scale) to the
    proven `dequantize_i8_block_scale`; E8M0/UE8M0 (1 byte/scale) to the proven
    `dequantize_i8_e8m0_block_scale` (Story 11.22; routed-I8 E8M0 raw-decode
    proven via closed-form + shim-pipeline parity). `fp4` routes to the OCP
    MXFP4 E2M1 parity reference (BF16 per-32-logical block scales, LSB-first).
    `i8` without `block_size`+`scale_axis` and ambiguous i8 scale byte lengths
    remain fail-closed per ADR 0002.
    """

    normalized = str(kind).casefold()
    if normalized == "fp4":
        if block_size is None:
            block_size = 32
        if scale_axis is None:
            scale_axis = 1
        if scales is None:
            raise ValueError("fp4 dispatch requires non-None scales payload (BF16 per-32-logical block)")
        return dequantize_fp4_block_scale(payload, scales, shape=shape, block_size=block_size, scale_axis=scale_axis)
    if normalized == "i8":
        if block_size is None or scale_axis is None:
            raise NotImplementedError(
                f"unsupported DeepSeek V4 expert packing {kind!r}: i8 BF16 dispatch requires block_size and scale_axis metadata "
                "(block_size_missing/scale_axis_missing); parity not proven without it"
            )
        if scales is None:
            raise ValueError("i8 BF16 dispatch requires a non-None scales payload when block_size and scale_axis are supplied")
        shape_t = _as_shape(shape)
        axis = _normalize_axis(scale_axis, len(shape_t))
        scale_count = _product(_expected_scale_shape(shape_t, block_size=block_size, scale_axis=axis))
        scale_bytes = len(scales)
        if scale_bytes == 2 * scale_count:
            return dequantize_i8_block_scale(payload, scales, shape=shape, block_size=block_size, scale_axis=scale_axis)
        if scale_bytes == scale_count:
            return dequantize_i8_e8m0_block_scale(
                payload, scales, shape=shape, block_size=block_size, scale_axis=scale_axis
            )
        raise ValueError(
            f"ambiguous i8 scale byte length {scale_bytes}; expected {scale_count} bytes for E8M0 or "
            f"{2 * scale_count} bytes for BF16"
        )
    raise NotImplementedError(f"unsupported DeepSeek V4 expert packing {kind!r}")
