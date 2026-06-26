"""Story 13.3a-3 sanitize/load self-consistency for deepseek_v4_nn."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MLX_SRC = _ROOT / "python-envs" / "mlx" / "src"
if str(_MLX_SRC) not in sys.path:
    sys.path.insert(0, str(_MLX_SRC))

mx = pytest.importorskip("mlx.core")
from mlx.utils import tree_flatten, tree_unflatten


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


def _replacement_like(name: str, value: mx.array) -> mx.array:
    if value.dtype == mx.uint8:
        return mx.full(value.shape, 1, dtype=value.dtype)
    if value.dtype == mx.int32:
        return mx.zeros(value.shape, dtype=value.dtype)
    if mx.issubdtype(value.dtype, mx.floating):
        fill = (len(name) % 7 + 1) / 100.0
        return mx.full(value.shape, fill, dtype=value.dtype)
    return mx.zeros(value.shape, dtype=value.dtype)


def test_model_sanitize_strips_mtp_and_nn_tree_round_trips_own_synthesized_weights():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import Model, ModelArgs

    model = Model(ModelArgs.from_dict(_tiny_config()))
    nn_weights = {name: _replacement_like(name, value) for name, value in tree_flatten(model.parameters())}
    weights = dict(nn_weights)
    weights["mtp.0.proj.weight"] = mx.ones((1,), dtype=mx.float32)
    weights["mtp.extra"] = mx.ones((1,), dtype=mx.float32)

    sanitized = model.sanitize(weights)

    assert "mtp.0.proj.weight" not in sanitized
    assert "mtp.extra" not in sanitized
    assert set(sanitized) == set(nn_weights)
    for name, expected in nn_weights.items():
        assert sanitized[name].shape == expected.shape
        assert sanitized[name].dtype == expected.dtype
        assert bool(mx.all(sanitized[name] == expected).item()), name

    # 13.3a-3 scope is self-consistency for the nn tree's own keys.  Real ckpt
    # stacked-vs-per-expert plus model.-prefix remapping remains 13.3b.
    model.update(tree_unflatten(list(sanitized.items())))

    round_tripped = dict(tree_flatten(model.parameters()))
    assert set(round_tripped) == set(nn_weights)
    for name, expected_value in nn_weights.items():
        got = round_tripped[name]
        assert got.shape == expected_value.shape
        assert got.dtype == expected_value.dtype
        assert bool(mx.all(got == expected_value).item()), name
