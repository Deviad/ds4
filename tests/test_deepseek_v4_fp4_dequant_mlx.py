"""Story 13.2 FP4 MLX dequant tests.

Expected values are computed by in-test numpy code from the OCP MXFP4 E2M1
contract: LUT literal, LSB-first nibble order per transformers/integrations/
mxfp4.py:292-298, and BF16 direct multiply.  They are not derived from any
production dequant primitive (ADR 0007 §4 / ADR 0017 boundary X).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MLX_SRC = _ROOT / "python-envs" / "mlx" / "src"
if str(_MLX_SRC) not in sys.path:
    sys.path.insert(0, str(_MLX_SRC))

mx = pytest.importorskip("mlx.core")

# OCP MXFP4 E2M1 literal table.  Low nibble is the first/even logical value
# (HF mxfp4.py:292-298); parity alone cannot prove this ordering.
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


def _bf16_roundtrip(values: np.ndarray) -> np.ndarray:
    raw = _bf16_le_bytes(values)
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (u16 << 16).astype(np.uint32).view(np.float32).reshape(values.shape)


def _fixture(*, out: int = 8, in_bytes: int = 16, seed: int = 42) -> tuple[np.ndarray, np.ndarray, bytes, np.ndarray]:
    rng = np.random.default_rng(seed)
    packed_u8 = rng.integers(0, 256, size=(out, in_bytes), dtype=np.uint8)
    scale_choices = np.array([0.015625, 0.03125, 0.0625, 0.125], dtype=np.float32)
    scales = rng.choice(scale_choices, size=(out, (in_bytes * 2) // 32)).astype(np.float32)
    scale_bytes = _bf16_le_bytes(scales)
    return packed_u8, scales, scale_bytes, _numpy_fp4_dequant(packed_u8, scale_bytes)


def _numpy_fp4_dequant(packed_u8: np.ndarray, scale_bytes: bytes, *, block_size: int = 32) -> np.ndarray:
    # Independent from production: literal LUT + LSB-first unpack + BF16 decode.
    lo = packed_u8 & np.uint8(0x0F)
    hi = (packed_u8 >> np.uint8(4)) & np.uint8(0x0F)
    vals = np.stack([E2M1_FP4_LUT[lo], E2M1_FP4_LUT[hi]], axis=-1).reshape(packed_u8.shape[0], packed_u8.shape[1] * 2)
    scales_u16 = np.frombuffer(scale_bytes, dtype="<u2").astype(np.uint32)
    scales = (scales_u16 << 16).astype(np.uint32).view(np.float32).reshape(packed_u8.shape[0], vals.shape[1] // block_size)
    return vals.astype(np.float32) * np.repeat(scales.astype(np.float32), block_size, axis=1)


def _mx_fp4_inputs(packed_u8: np.ndarray, scales: np.ndarray) -> tuple[mx.array, mx.array]:
    # Real checkpoint stores I8 containers; view random bytes as signed I8 to
    # cover the required uint8 reinterpret-before-bit-ops path.
    return mx.array(packed_u8.view(np.int8)), mx.array(_bf16_roundtrip(scales).tolist(), dtype=mx.bfloat16)


def _to_numpy(value: mx.array) -> np.ndarray:
    return np.asarray(value.tolist(), dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _moe_numpy_reference(args, hidden_states: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    router = weights["mlp.gate.weight"]
    logits = hidden_states @ router.T
    scores = np.sqrt(_softplus(logits))
    correction = weights.get("mlp.gate.e_score_correction_bias")
    selection_scores = scores if correction is None else scores + correction
    top_idx = np.argsort(-selection_scores, axis=-1)[..., : args.num_experts_per_tok]

    denom = np.zeros(scores.shape[:-1], dtype=np.float32)
    for eid in range(args.n_routed_experts):
        selected = np.any(top_idx == eid, axis=-1).astype(np.float32)
        denom += scores[..., eid] * selected

    routed = np.zeros_like(hidden_states, dtype=np.float32)
    for eid in range(args.n_routed_experts):
        w1 = weights[f"mlp.experts.{eid}.w1.weight"]
        w2 = weights[f"mlp.experts.{eid}.w2.weight"]
        w3 = weights[f"mlp.experts.{eid}.w3.weight"]
        gate = np.clip(hidden_states @ w1.T, None, args.swiglu_limit)
        up = np.clip(hidden_states @ w3.T, -args.swiglu_limit, args.swiglu_limit)
        expert_out = (_sigmoid(gate) * gate * up) @ w2.T
        selected = np.any(top_idx == eid, axis=-1).astype(np.float32)
        factor = (scores[..., eid] / (denom + 1e-20)) * selected * args.routed_scaling_factor
        routed += expert_out * factor[..., None]

    sw1 = weights["mlp.shared_experts.w1.weight"]
    sw2 = weights["mlp.shared_experts.w2.weight"]
    sw3 = weights["mlp.shared_experts.w3.weight"]
    shared_gate = np.clip(hidden_states @ sw1.T, None, args.swiglu_limit)
    shared_up = np.clip(hidden_states @ sw3.T, -args.swiglu_limit, args.swiglu_limit)
    return routed + (_sigmoid(shared_gate) * shared_gate * shared_up) @ sw2.T


def _args(*, hidden_size: int, intermediate: int, expert_dtype: str):
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import ModelArgs

    return ModelArgs.from_dict(
        {
            "model_type": "deepseek_v4",
            "vocab_size": 8,
            "hidden_size": hidden_size,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": hidden_size,
            "q_lora_rank": hidden_size,
            "o_lora_rank": hidden_size,
            "qk_rope_head_dim": 2,
            "n_routed_experts": 1,
            "num_experts_per_tok": 1,
            "n_shared_experts": 1,
            "moe_intermediate_size": intermediate,
            "expert_dtype": expert_dtype,
            "layer_types": ["sliding_attention"],
            "mlp_layer_types": ["moe"],
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.0,
            "swiglu_limit": 10.0,
            "o_groups": 1,
        }
    )


def test_dequantize_fp4_block_scale_mlx_matches_numpy_reference_exactly():
    from ds4_ft_mlx.vendor.mlx_lm_models import deepseek_v4 as vendor

    packed_u8, scales, _scale_bytes, expected = _fixture()
    got = vendor._dequantize_fp4_block_scale_mlx(*_mx_fp4_inputs(packed_u8, scales))

    got_np = _to_numpy(got)
    assert got_np.shape == (8, 32)
    assert float(np.max(np.abs(got_np - expected))) == 0.0


def test_dequantize_fp4_block_scale_mlx_shape_guards():
    from ds4_ft_mlx.vendor.mlx_lm_models import deepseek_v4 as vendor

    packed_u8, scales, _scale_bytes, _expected = _fixture()
    weight_mx, scale_mx = _mx_fp4_inputs(packed_u8, scales)
    with pytest.raises(ValueError, match="block_size"):
        vendor._dequantize_fp4_block_scale_mlx(weight_mx, scale_mx, block_size=16)
    with pytest.raises(ValueError, match="scale cols"):
        vendor._dequantize_fp4_block_scale_mlx(weight_mx, mx.array([[1.0, 1.0]] * 8, dtype=mx.bfloat16))


def test_moe_mlx_fp4_branch_matches_numpy_reference():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _moe_mlx

    args = _args(hidden_size=32, intermediate=32, expert_dtype="fp4")
    rng = np.random.default_rng(42)
    hidden = rng.normal(0.0, 0.05, size=(1, 2, 32)).astype(np.float32)
    weights_np: dict[str, np.ndarray] = {
        "mlp.gate.weight": np.zeros((1, 32), dtype=np.float32),
        "mlp.shared_experts.w1.weight": np.zeros((32, 32), dtype=np.float32),
        "mlp.shared_experts.w2.weight": np.zeros((32, 32), dtype=np.float32),
        "mlp.shared_experts.w3.weight": np.zeros((32, 32), dtype=np.float32),
    }
    weights_mx: dict[str, mx.array] = {key: mx.array(value) for key, value in weights_np.items()}
    for offset, proj in enumerate(("w1", "w2", "w3")):
        packed_u8, scales, scale_bytes, dequant = _fixture(out=32, in_bytes=16, seed=42 + offset)
        weights_np[f"mlp.experts.0.{proj}.weight"] = dequant
        weights_mx[f"mlp.experts.0.{proj}.weight"] = mx.array(packed_u8.view(np.int8))
        weights_mx[f"mlp.experts.0.{proj}.scale"] = mx.array(_bf16_roundtrip(scales).tolist(), dtype=mx.bfloat16)
        assert np.isfinite(_numpy_fp4_dequant(packed_u8, scale_bytes)).all()

    expected = _moe_numpy_reference(args, hidden, weights_np)
    got = _to_numpy(_moe_mlx(args, mx.array(hidden), weights_mx))
    assert np.isfinite(got).all()
    assert float(np.max(np.abs(got - expected))) <= 1e-6


def test_moe_mlx_i8_branch_regression_unchanged():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _moe_mlx

    args = _args(hidden_size=16, intermediate=16, expert_dtype="i8")
    hidden = np.linspace(-0.08, 0.08, 32, dtype=np.float32).reshape(1, 2, 16)
    scale = np.full((16, 1), 0.03125, dtype=np.float32)
    i8 = np.eye(16, dtype=np.int8)
    dequant = i8.astype(np.float32) * np.repeat(scale, 16, axis=1)
    weights_np: dict[str, np.ndarray] = {
        "mlp.gate.weight": np.zeros((1, 16), dtype=np.float32),
        "mlp.shared_experts.w1.weight": np.zeros((16, 16), dtype=np.float32),
        "mlp.shared_experts.w2.weight": np.zeros((16, 16), dtype=np.float32),
        "mlp.shared_experts.w3.weight": np.zeros((16, 16), dtype=np.float32),
    }
    weights_mx: dict[str, mx.array] = {key: mx.array(value) for key, value in weights_np.items()}
    for proj in ("w1", "w2", "w3"):
        weights_np[f"mlp.experts.0.{proj}.weight"] = dequant
        weights_mx[f"mlp.experts.0.{proj}.weight"] = mx.array(i8)
        weights_mx[f"mlp.experts.0.{proj}.scale"] = mx.array(scale.tolist(), dtype=mx.bfloat16)

    expected = _moe_numpy_reference(args, hidden, weights_np)
    got = _to_numpy(_moe_mlx(args, mx.array(hidden), weights_mx))
    assert float(np.max(np.abs(got - expected))) <= 1e-7


def test_moe_mlx_raw_else_branch_regression_unchanged():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _moe_mlx

    args = _args(hidden_size=4, intermediate=2, expert_dtype="fp4")
    args.expert_dtype = "raw"  # direct helper regression for the non-i8/non-fp4 else branch.
    hidden = np.array([[[0.1, -0.2, 0.3, -0.4]]], dtype=np.float32)
    weights_np: dict[str, np.ndarray] = {
        "mlp.gate.weight": np.zeros((1, 4), dtype=np.float32),
        "mlp.experts.0.w1.weight": np.array([[0.2, -0.1, 0.3, 0.0], [0.0, 0.1, -0.2, 0.4]], dtype=np.float32),
        "mlp.experts.0.w2.weight": np.array([[0.1, -0.2], [0.0, 0.3], [-0.1, 0.2], [0.4, 0.1]], dtype=np.float32),
        "mlp.experts.0.w3.weight": np.array([[0.0, 0.2, 0.1, -0.1], [0.3, -0.2, 0.0, 0.1]], dtype=np.float32),
        "mlp.shared_experts.w1.weight": np.zeros((2, 4), dtype=np.float32),
        "mlp.shared_experts.w2.weight": np.zeros((4, 2), dtype=np.float32),
        "mlp.shared_experts.w3.weight": np.zeros((2, 4), dtype=np.float32),
    }
    expected = _moe_numpy_reference(args, hidden, weights_np)
    got = _to_numpy(_moe_mlx(args, mx.array(hidden), {key: mx.array(value) for key, value in weights_np.items()}))
    assert float(np.max(np.abs(got - expected))) <= 1e-7
