"""Story 13.3a-2 construct tests for SparseMoeBlockNN + FP4 experts."""

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


def test_sparse_moe_block_constructs_moe_and_hash_variants():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import (
        DeepseekV4FP4Experts,
        ModelArgs,
        SparseMoeBlockNN,
    )

    args = ModelArgs.from_dict(_tiny_config())
    hash_block = SparseMoeBlockNN(args, layer_idx=0)
    moe_block = SparseMoeBlockNN(args, layer_idx=1)

    assert isinstance(hash_block, nn.Module)
    assert isinstance(moe_block, nn.Module)
    assert hash_block.is_hash is True
    assert moe_block.is_hash is False
    assert hash_block.layer_type == "hash_moe"
    assert moe_block.layer_type == "moe"

    assert hash_block.gate_weight.shape == (4, 16)
    assert moe_block.gate_weight.shape == (4, 16)
    assert moe_block.e_score_correction_bias.shape == (4,)
    assert hash_block.tid2eid.shape == (64, 2)
    assert hash_block.tid2eid.dtype == mx.int32

    assert isinstance(hash_block.experts, DeepseekV4FP4Experts)
    assert isinstance(moe_block.experts, DeepseekV4FP4Experts)
    assert hasattr(hash_block, "shared_experts")
    assert hasattr(moe_block, "shared_experts")

    module_names = {name for name, _ in hash_block.named_modules()}
    assert "experts" in module_names
    assert "shared_experts" in module_names

    trainable_names = {name for name, _ in tree_flatten(hash_block.trainable_parameters())}
    assert "gate_weight" in trainable_names
    assert "tid2eid" not in trainable_names
