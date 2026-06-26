"""Story 13.3a-2 forward coverage for learned top-k moe layers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MLX_SRC = _ROOT / "python-envs" / "mlx" / "src"
if str(_MLX_SRC) not in sys.path:
    sys.path.insert(0, str(_MLX_SRC))

mx = pytest.importorskip("mlx.core")


def _tiny_config() -> dict[str, object]:
    return dict(
        model_type="deepseek_v4_nn",
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_hash_layers=1,
        mlp_layer_types=["hash_moe", "moe"],
        hc_mult=2,
        hc_sinkhorn_iters=3,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=16,
        expert_dtype="fp4",
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=8,
        o_lora_rank=8,
        o_groups=2,
        qk_rope_head_dim=4,
        compression_ratio=0,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
    )


def _fill_fp4(experts, nibble_byte: int = 0x11, scale: float = 0.05) -> None:
    experts.w1_weight = mx.full(experts.w1_weight.shape, nibble_byte, dtype=mx.uint8)
    experts.w2_weight = mx.full(experts.w2_weight.shape, nibble_byte, dtype=mx.uint8)
    experts.w3_weight = mx.full(experts.w3_weight.shape, nibble_byte, dtype=mx.uint8)
    experts.w1_scale = mx.full(experts.w1_scale.shape, scale, dtype=mx.bfloat16)
    experts.w2_scale = mx.full(experts.w2_scale.shape, scale, dtype=mx.bfloat16)
    experts.w3_scale = mx.full(experts.w3_scale.shape, scale, dtype=mx.bfloat16)


def _zero_shared(block) -> None:
    block.shared_experts.gate_proj.weight = mx.zeros_like(block.shared_experts.gate_proj.weight)
    block.shared_experts.up_proj.weight = mx.zeros_like(block.shared_experts.up_proj.weight)
    block.shared_experts.down_proj.weight = mx.zeros_like(block.shared_experts.down_proj.weight)


def test_moe_layer_forward_returns_finite_routed_plus_shared_tensor():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    args = ModelArgs.from_dict(_tiny_config())
    block = SparseMoeBlockNN(args, layer_idx=1)
    _fill_fp4(block.experts)
    _zero_shared(block)
    block.gate_weight = mx.stack([mx.ones((16,)) * scale for scale in (0.01, 0.02, 0.03, 0.04)])
    block.e_score_correction_bias = mx.array([0.0, 0.01, 0.02, 0.03], dtype=mx.float32)

    x = mx.ones((2, 3, 16), dtype=mx.float32) * 0.25
    indices = block.routing_indices(x)
    out = block(x, input_ids=None)
    mx.eval(indices, out)

    assert isinstance(out, mx.array)
    assert out.shape == (2, 3, 16)
    assert bool(mx.all(mx.isfinite(out)).item())
    assert indices.shape == (2, 3, 2)
    assert indices[0, 0].tolist() == [3, 2]
    assert float(mx.max(mx.abs(out)).item()) > 0.0
    assert not isinstance(out, list)
