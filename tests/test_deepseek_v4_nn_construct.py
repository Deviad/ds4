"""Story 13.3a-1 construct tests for the trainable DeepSeek V4 nn.Module port."""

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


def test_deepseek_v4_nn_constructs_trainable_module_tree():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import Model, ModelArgs

    args = ModelArgs.from_dict(_tiny_config())
    model = Model(args)

    assert args.model_type == "deepseek_v4_nn"
    assert isinstance(model, nn.Module)
    assert len(model.layers) == 2
    assert model.model.layers[0].mlp.layer_type == "hash_moe"
    assert model.model.layers[1].mlp.layer_type == "moe"

    module_names = {name for name, _ in model.named_modules()}
    assert "model.layers.0.self_attn.q_a_proj" in module_names
    assert "model.layers.0.attn_hc" in module_names
    assert "model.hc_head" in module_names

    trainable = tree_flatten(model.trainable_parameters())
    assert trainable
    assert any(name.endswith("self_attn.q_a_proj.weight") for name, _ in trainable)
    assert any(name.endswith("attn_hc.fn") for name, _ in trainable)
