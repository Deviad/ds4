"""Story 13.3a-1 LoRA attachment tests for trainable attention leaves."""

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


def test_deepseek_v4_nn_lora_attaches_to_attention_only_keys():
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import Model, ModelArgs

    model = Model(ModelArgs.from_dict(_tiny_config()))
    linear_to_lora_layers(
        model,
        num_layers=2,
        config={
            "rank": 2,
            "scale": 4.0,
            "dropout": 0.0,
            "keys": {"self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_proj"},
        },
    )

    for layer in model.model.layers:
        assert type(layer.self_attn.q_a_proj).__name__ == "LoRALinear"
        assert type(layer.self_attn.q_b_proj).__name__ == "LoRALinear"
        assert type(layer.self_attn.kv_proj).__name__ == "LoRALinear"
        assert isinstance(layer.self_attn.o_b_proj, nn.Linear)
        assert isinstance(layer.mlp.gate_proj, nn.Linear)
        assert type(layer.mlp.gate_proj).__name__ != "LoRALinear"

    out = model(mx.array([[1, 2, 3]], dtype=mx.int32))
    assert out.shape == (1, 3, 64)
