"""Story 13.2 FP4 Python reference tests.

Expected values are produced by in-test numpy code from the OCP MXFP4 E2M1
literal LUT, LSB-first nibble order (transformers/integrations/mxfp4.py:292-298),
and BF16 direct multiply.  The expected table is not derived from production
helpers (ADR 0007 §4 / ADR 0017 boundary X).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MLX_SRC = _ROOT / "python-envs" / "mlx" / "src"
if str(_MLX_SRC) not in sys.path:
    sys.path.insert(0, str(_MLX_SRC))

E2M1_FP4_LUT = np.array(
    [
        +0.0,
        +0.5,
        +1.0,
        +1.5,
        +2.0,
        +3.0,
        +4.0,
        +6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


def _bf16_le_bytes(values: np.ndarray) -> bytes:
    u32 = np.asarray(values, dtype=np.float32).view(np.uint32)
    return (u32 >> 16).astype("<u2").tobytes()


def _decode_bf16_le(raw: bytes, shape: tuple[int, ...]) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (u16 << 16).astype(np.uint32).view(np.float32).reshape(shape)


def _fixture() -> tuple[np.ndarray, bytes, bytes, np.ndarray]:
    rng = np.random.default_rng(42)
    packed_u8 = rng.integers(0, 256, size=(8, 16), dtype=np.uint8)
    scale_choices = np.array([0.015625, 0.03125, 0.0625, 0.125], dtype=np.float32)
    scales = rng.choice(scale_choices, size=(8, 1)).astype(np.float32)
    scale_bytes = _bf16_le_bytes(scales)
    return packed_u8, packed_u8.tobytes(), scale_bytes, _numpy_fp4_dequant(packed_u8, scale_bytes)


def _numpy_fp4_dequant(packed_u8: np.ndarray, scale_bytes: bytes, *, block_size: int = 32) -> np.ndarray:
    # Independent reference: LSB-first unpack + literal E2M1 LUT + BF16 multiply.
    lo = packed_u8 & np.uint8(0x0F)
    hi = (packed_u8 >> np.uint8(4)) & np.uint8(0x0F)
    vals = np.stack([E2M1_FP4_LUT[lo], E2M1_FP4_LUT[hi]], axis=-1).reshape(packed_u8.shape[0], packed_u8.shape[1] * 2)
    scales = _decode_bf16_le(scale_bytes, (packed_u8.shape[0], vals.shape[1] // block_size))
    return vals.astype(np.float32) * np.repeat(scales.astype(np.float32), block_size, axis=1)


def test_dequantize_expert_packed_fp4_dispatch_matches_numpy_reference():
    from ds4_ft_mlx import deepseek_v4_dequant as dequant

    _packed_u8, payload, scales, expected = _fixture()
    got = dequant.dequantize_expert_packed("fp4", payload, scales=scales, shape=(8, 32), block_size=32, scale_axis=1)

    got_np = np.asarray(got, dtype=np.float32).reshape(8, 32)
    assert got_np.shape == (8, 32)
    assert float(np.max(np.abs(got_np - expected))) == 0.0


def test_dequantize_fp4_block_scale_rejects_wrong_metadata():
    from ds4_ft_mlx import deepseek_v4_dequant as dequant

    _packed_u8, payload, scales, _expected = _fixture()
    with pytest.raises(ValueError, match="block_size"):
        dequant.dequantize_fp4_block_scale(payload, scales, shape=(8, 32), block_size=16, scale_axis=1)
    with pytest.raises(ValueError, match="scales byte length"):
        dequant.dequantize_fp4_block_scale(payload, scales + b"\x00\x00", shape=(8, 32), block_size=32, scale_axis=1)
    with pytest.raises(ValueError, match="requires non-None scales"):
        dequant.dequantize_expert_packed("fp4", payload, scales=None, shape=(8, 32), block_size=32, scale_axis=1)


def test_python_ref_matches_mlx_primitive_on_same_bytes_and_scales():
    mx = pytest.importorskip("mlx.core")
    from ds4_ft_mlx import deepseek_v4_dequant as dequant
    from ds4_ft_mlx.vendor.mlx_lm_models import deepseek_v4 as vendor

    packed_u8, payload, scales, _expected = _fixture()
    py = np.asarray(
        dequant.dequantize_expert_packed("fp4", payload, scales=scales, shape=(8, 32), block_size=32, scale_axis=1),
        dtype=np.float32,
    ).reshape(8, 32)
    scale_values = _decode_bf16_le(scales, (8, 1))
    mx_out = vendor._dequantize_fp4_block_scale_mlx(
        mx.array(packed_u8.view(np.int8)),
        mx.array(scale_values.tolist(), dtype=mx.bfloat16),
    )
    mx_np = np.asarray(mx_out.tolist(), dtype=np.float32)
    assert float(np.max(np.abs(py - mx_np))) == 0.0


def test_fp4_reference_expected_table_is_in_test_literal_not_production_derived():
    source = inspect.getsource(_numpy_fp4_dequant)
    assert "dequantize_expert_packed" not in source
    assert "dequantize_fp4_block_scale" not in source
    assert "_dequantize_fp4_block_scale_mlx" not in source
    assert E2M1_FP4_LUT.tolist() == [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
