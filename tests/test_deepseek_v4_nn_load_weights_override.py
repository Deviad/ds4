"""Story 13.3b-4b: nn Model.load_weights remaps native ckpt keys once."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MLX_SRC = _ROOT / "python-envs" / "mlx" / "src"
for _path in (_ROOT, _MLX_SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

mx = pytest.importorskip("mlx.core")


def _tiny_args():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs

    args = ModelArgs(
        vocab_size=8,
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        q_lora_rank=2,
        o_lora_rank=2,
        qk_rope_head_dim=2,
        index_head_dim=2,
        index_n_heads=2,
        n_routed_experts=2,
        num_experts_per_tok=1,
        n_shared_experts=1,
        moe_intermediate_size=4,
        num_hash_layers=0,
        hc_mult=1,
        layer_types=["sliding_attention"],
        mlp_layer_types=["moe"],
        o_groups=1,
        compression_ratio=0,
        compress_ratios=[0],
    )
    args._normalize_layer_types()
    args.validate()
    return args


def test_model_defines_load_weights_override():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import Model

    assert "load_weights" in Model.__dict__


def test_transformers_autoconfig_accepts_deepseek_v4_nn_after_model_import():
    transformers = pytest.importorskip("transformers")
    from ds4_ft_mlx.vendor.mlx_lm_models import deepseek_v4_nn  # noqa: F401

    cfg = transformers.AutoConfig.for_model(
        "deepseek_v4_nn",
        rope_scaling={"type": "yarn", "factor": 16, "original_max_position_embeddings": 65536},
        rope_theta=10000.0,
        layer_types=["sliding_attention"],
        mlp_layer_types=["hash_moe"],
        eos_token_id=1,
    )

    assert cfg.model_type == "deepseek_v4_nn"
    assert cfg.eos_token_id == 1


def test_native_checkpoint_keys_remap_before_strict_module_load(monkeypatch):
    from ds4_ft_mlx import deepseek_v4_nn_remap
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import Model

    calls = []
    original = deepseek_v4_nn_remap.remap_weight_dict

    def spy(weights, *, config):
        calls.append({"keys": set(dict(weights)), "config": dict(config)})
        return original(weights, config=config)

    monkeypatch.setattr(deepseek_v4_nn_remap, "remap_weight_dict", spy)
    model = Model(_tiny_args())
    value = mx.ones((8, 4)) * 7

    model.load_weights([("embed.weight", value)], strict=False)

    assert calls == [{"keys": {"embed.weight"}, "config": {"n_routed_experts": 2, "num_hash_layers": 0}}]
    assert bool(mx.all(model.model.embed_tokens.weight == value).item())


def test_already_nn_keys_bypass_remap_for_idempotent_reload(monkeypatch):
    from ds4_ft_mlx import deepseek_v4_nn_remap
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import Model

    def fail_remap(*_args, **_kwargs):  # pragma: no cover - should never be called
        raise AssertionError("nn-key reload must not call remap_weight_dict")

    monkeypatch.setattr(deepseek_v4_nn_remap, "remap_weight_dict", fail_remap)
    model = Model(_tiny_args())
    value = mx.ones((8, 4)) * 5

    model.load_weights([("model.embed_tokens.weight", value)], strict=False)

    assert bool(mx.all(model.model.embed_tokens.weight == value).item())
