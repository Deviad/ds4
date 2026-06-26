"""Story 13.3a-2 quantize guard for frozen FP4 routed experts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MLX_SRC = _ROOT / "python-envs" / "mlx" / "src"
if str(_MLX_SRC) not in sys.path:
    sys.path.insert(0, str(_MLX_SRC))

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
from mlx.utils import tree_flatten


def _tiny_quantizable_config() -> dict[str, object]:
    # nn.quantize group_size=32 requires last weight dimensions divisible by 32.
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


def test_fp4_experts_are_frozen_and_invisible_to_nn_quantize():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import (
        DeepseekV4FP4Experts,
        Model,
        ModelArgs,
    )

    model = Model(ModelArgs.from_dict(_tiny_quantizable_config()))
    expert = model.model.layers[0].mlp.experts

    assert isinstance(expert, DeepseekV4FP4Experts)
    assert not hasattr(expert, "to_quantized")
    assert tree_flatten(expert.trainable_parameters()) == []
    assert expert.w1_weight.dtype == mx.uint8
    assert expert.w2_weight.dtype == mx.uint8
    assert expert.w3_weight.dtype == mx.uint8

    nn.quantize(model, group_size=32, bits=4)

    quantized_expert = model.model.layers[0].mlp.experts
    assert isinstance(quantized_expert, DeepseekV4FP4Experts)
    assert type(quantized_expert).__name__ != "QuantizedLinear"
    assert quantized_expert.w1_weight.dtype == mx.uint8
    assert quantized_expert.w2_weight.dtype == mx.uint8
    assert quantized_expert.w3_weight.dtype == mx.uint8
    assert isinstance(model.model.layers[0].self_attn.q_a_proj, nn.QuantizedLinear)
