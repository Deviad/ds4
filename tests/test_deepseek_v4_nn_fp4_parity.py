"""Story 13.3a-2 parity: SparseMoeBlockNN FP4 path equals frozen _moe_mlx."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MLX_SRC = _ROOT / "python-envs" / "mlx" / "src"
if str(_MLX_SRC) not in sys.path:
    sys.path.insert(0, str(_MLX_SRC))

mx = pytest.importorskip("mlx.core")


def _fp4_config() -> dict[str, object]:
    # FP4 dequant is defined on 32-logical-element blocks; use the minimal block-sized sibling.
    return dict(
        model_type="deepseek_v4_nn",
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_hash_layers=1,
        mlp_layer_types=["hash_moe", "moe"],
        hc_mult=2,
        hc_sinkhorn_iters=3,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=32,
        expert_dtype="fp4",
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        qk_rope_head_dim=4,
        compression_ratio=0,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
    )


def _stacked_packed(n: int, rows: int, packed_cols: int, base: int) -> mx.array:
    return mx.stack([mx.full((rows, packed_cols), base + eid, dtype=mx.uint8) for eid in range(n)])


def _stacked_scale(n: int, rows: int, scale_cols: int, value: float) -> mx.array:
    return mx.stack([mx.full((rows, scale_cols), value, dtype=mx.bfloat16) for _ in range(n)])


def test_sparse_moe_fp4_forward_matches_frozen_moe_mlx_with_same_weights():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _moe_mlx
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    args = ModelArgs.from_dict(_fp4_config())
    block = SparseMoeBlockNN(args, layer_idx=1)

    gate = (mx.arange(4 * 32, dtype=mx.float32).reshape(4, 32) - 64.0) / 500.0
    bias = mx.array([0.02, -0.03, 0.01, 0.0], dtype=mx.float32)
    w1 = _stacked_packed(4, 32, 16, 0x11)
    w3 = _stacked_packed(4, 32, 16, 0x22)
    w2 = _stacked_packed(4, 32, 16, 0x33)
    s1 = _stacked_scale(4, 32, 1, 0.05)
    s3 = _stacked_scale(4, 32, 1, 0.04)
    s2 = _stacked_scale(4, 32, 1, 0.03)
    sw1 = mx.ones((32, 32), dtype=mx.float32) * 0.02
    sw3 = mx.ones((32, 32), dtype=mx.float32) * -0.015
    sw2 = mx.ones((32, 32), dtype=mx.float32) * 0.01

    block.gate_weight = gate
    block.e_score_correction_bias = bias
    block.experts.w1_weight = w1
    block.experts.w3_weight = w3
    block.experts.w2_weight = w2
    block.experts.w1_scale = s1
    block.experts.w3_scale = s3
    block.experts.w2_scale = s2
    block.shared_experts.gate_proj.weight = sw1
    block.shared_experts.up_proj.weight = sw3
    block.shared_experts.down_proj.weight = sw2

    weights = {
        "mlp.gate.weight": gate,
        "mlp.gate.e_score_correction_bias": bias,
        "mlp.shared_experts.w1.weight": sw1,
        "mlp.shared_experts.w3.weight": sw3,
        "mlp.shared_experts.w2.weight": sw2,
    }
    for eid in range(4):
        weights[f"mlp.experts.{eid}.w1.weight"] = w1[eid]
        weights[f"mlp.experts.{eid}.w3.weight"] = w3[eid]
        weights[f"mlp.experts.{eid}.w2.weight"] = w2[eid]
        weights[f"mlp.experts.{eid}.w1.scale"] = s1[eid]
        weights[f"mlp.experts.{eid}.w3.scale"] = s3[eid]
        weights[f"mlp.experts.{eid}.w2.scale"] = s2[eid]

    x = mx.arange(2 * 3 * 32, dtype=mx.float32).reshape(2, 3, 32) / 100.0
    got = block(x)
    expected = _moe_mlx(args, x, weights)
    mx.eval(got, expected)

    abs_err = mx.abs(got - expected)
    bound = 2e-6 + 1e-6 * mx.abs(expected)
    nrmse = mx.sqrt(mx.mean(mx.square(abs_err))) / mx.maximum(
        mx.sqrt(mx.mean(mx.square(expected))), mx.array(1e-12, dtype=mx.float32)
    )
    denom = mx.maximum(mx.maximum(mx.abs(got), mx.abs(expected)), mx.array(1e-6, dtype=mx.float32))
    max_abs = float(mx.max(abs_err).item())
    max_rel = float(mx.max(abs_err / denom).item())
    nrmse_value = float(nrmse.item())
    assert bool(mx.all(abs_err <= bound).item()) and nrmse_value <= 1e-6, (
        f"fp4 whole forward max_abs={max_abs} max_rel={max_rel} nrmse={nrmse_value}"
    )
