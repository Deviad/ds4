"""Story 13.3a-1 nn.quantize coverage for DeepSeek V4 nn.Module leaves."""

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


def _tiny_quantizable_config() -> dict[str, object]:
    # MLX weight quantization requires the last weight dimension to be divisible
    # by group_size=32.  This is the minimal 32-wide sibling of the §13 fixture;
    # construct/forward tests keep the exact 16-wide fixture.
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


def test_deepseek_v4_nn_quantize_converts_attention_and_stub_mlp_linear_leaves():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import HyperConnectionNN, Model, ModelArgs

    model = Model(ModelArgs.from_dict(_tiny_quantizable_config()))
    assert isinstance(model.model.layers[0].self_attn.q_a_proj, nn.Linear)
    assert isinstance(model.model.layers[0].mlp.gate_proj, nn.Linear)

    nn.quantize(model, group_size=32, bits=4)

    layer = model.model.layers[0]
    assert isinstance(layer.self_attn.q_a_proj, nn.QuantizedLinear)
    assert isinstance(layer.self_attn.q_b_proj, nn.QuantizedLinear)
    assert isinstance(layer.self_attn.kv_proj, nn.QuantizedLinear)
    assert isinstance(layer.self_attn.o_b_proj, nn.QuantizedLinear)
    assert isinstance(layer.mlp.gate_proj, nn.QuantizedLinear)
    assert isinstance(layer.mlp.up_proj, nn.QuantizedLinear)
    assert isinstance(layer.mlp.down_proj, nn.QuantizedLinear)
    assert isinstance(layer.attn_hc, HyperConnectionNN)
    assert isinstance(model.model.hc_head.fn, mx.array)
