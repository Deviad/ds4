"""Story 13.3a-2 forward coverage for tid2eid hash_moe routing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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


def _zero_fp4(experts) -> None:
    experts.w1_weight = mx.zeros_like(experts.w1_weight)
    experts.w2_weight = mx.zeros_like(experts.w2_weight)
    experts.w3_weight = mx.zeros_like(experts.w3_weight)
    experts.w1_scale = mx.ones_like(experts.w1_scale)
    experts.w2_scale = mx.ones_like(experts.w2_scale)
    experts.w3_scale = mx.ones_like(experts.w3_scale)


def _set_shared(block, value: float) -> None:
    block.shared_experts.gate_proj.weight = mx.ones_like(block.shared_experts.gate_proj.weight) * value
    block.shared_experts.up_proj.weight = mx.ones_like(block.shared_experts.up_proj.weight) * value
    block.shared_experts.down_proj.weight = mx.ones_like(block.shared_experts.down_proj.weight) * value


def test_hash_moe_forward_uses_tid2eid_not_learned_topk_indices():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs, SparseMoeBlockNN

    args = ModelArgs.from_dict(_tiny_config())
    block = SparseMoeBlockNN(args, layer_idx=0)
    _zero_fp4(block.experts)
    _set_shared(block, 0.05)

    # Learned top-k would pick [0, 1]; hash_moe must ignore that for index choice.
    block.gate_weight = mx.zeros((4, 16), dtype=mx.float32)
    block.e_score_correction_bias = mx.array([10.0, 9.0, 0.0, -1.0], dtype=mx.float32)
    table = np.zeros((64, 2), dtype=np.int32)
    table[7] = [2, 3]
    block.tid2eid = mx.array(table, dtype=mx.int32)

    x = mx.ones((1, 1, 16), dtype=mx.float32)
    input_ids = mx.array([[7]], dtype=mx.int32)
    indices = block.routing_indices(x, input_ids=input_ids)
    out_with_shared = block(x, input_ids=input_ids)
    _set_shared(block, 0.0)
    out_without_shared = block(x, input_ids=input_ids)
    mx.eval(indices, out_with_shared, out_without_shared)

    assert indices.shape == (1, 1, 2)
    assert indices[0, 0].tolist() == [2, 3]
    assert isinstance(out_with_shared, mx.array)
    assert out_with_shared.shape == (1, 1, 16)
    assert bool(mx.all(mx.isfinite(out_with_shared)).item())
    assert float(mx.max(mx.abs(out_with_shared - out_without_shared)).item()) > 0.0
