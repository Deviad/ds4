"""AC3 additive dispatch contract for Story 13.3b-1.

Tiny CSA inputs still satisfy `_csa_config_error(args) is None` and route to the
pre-existing `_csa_attention_mlx` result byte-identically. Real-dim HCA inputs
fail the tiny guard and route to `_attention_real_mlx` instead of raising the old
CSA `NotImplementedError`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MLX_SRC = _ROOT / "python-envs" / "mlx" / "src"
if str(_MLX_SRC) not in sys.path:
    sys.path.insert(0, str(_MLX_SRC))

mx = pytest.importorskip("mlx.core")

from test_compress_rope_oracle import real_hca_mlx_args


def _tiny_csa_args():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import ModelArgs

    return ModelArgs.from_dict(
        {
            "model_type": "deepseek_v4",
            "vocab_size": 8,
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "q_lora_rank": 4,
            "o_lora_rank": 4,
            "qk_rope_head_dim": 4,
            "index_head_dim": 2,
            "index_n_heads": 2,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "n_shared_experts": 1,
            "moe_intermediate_size": 2,
            "expert_dtype": "fp4",
            "rms_norm_eps": 1e-6,
            "hc_mult": 1,
            "layer_types": ["sliding_attention"],
            "mlp_layer_types": ["moe"],
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.0,
            "swiglu_limit": 10.0,
            "rope_theta": 10000.0,
            "compress_rope_theta": 160000.0,
            "sliding_window": 128,
            "o_groups": 1,
            "compression_ratio": 4,
        }
    )


def _tiny_hidden():
    return mx.array(
        [[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5, 0.0],
            [0.25, 0.25, 0.25, 0.25],
        ]]
    )


def _tiny_weights():
    def dense(rows, cols, scale):
        return [[round(scale * (i + j + 1) * (1.0 if (i + j) % 2 == 0 else -1.0), 4) for j in range(cols)] for i in range(rows)]

    return {key: mx.array(value) for key, value in {
        "compressor_wkv": dense(4, 8, 0.5),
        "compressor_wgate": dense(4, 8, 0.25),
        "compressor_ape": dense(8, 4, 0.05),
        "compressor_norm": [1.0, 1.0, 1.0, 1.0],
        "indexer_wq_b": dense(4, 4, 0.5),
        "indexer_proj": dense(4, 2, 0.3),
        "indexer_compressor_wkv": dense(4, 4, 0.4),
        "indexer_compressor_wgate": dense(4, 4, 0.2),
        "indexer_compressor_ape": dense(4, 4, 0.05),
        "indexer_compressor_norm": [1.0, 1.0],
    }.items()}


def test_dispatch_tiny_csa_byte_identical_and_real_hca_routes_new(monkeypatch):
    from ds4_ft_mlx.vendor.mlx_lm_models import deepseek_v4

    tiny_args = _tiny_csa_args()
    assert deepseek_v4._csa_config_error(tiny_args) is None
    direct = deepseek_v4._csa_attention_mlx(tiny_args, _tiny_hidden(), _tiny_weights(), index_topk=2)
    routed = deepseek_v4._attention_mlx(tiny_args, _tiny_hidden(), _tiny_weights(), index_topk=2)
    mx.eval(direct, routed)
    assert routed.tolist() == direct.tolist()

    real_args = real_hca_mlx_args()
    assert deepseek_v4._csa_config_error(real_args) is not None
    sentinel = mx.array([[[13.3]]])
    called = {}

    def fake_attention_real(fake_args, fake_x, fake_weights, *, index_topk=None):
        called["args"] = fake_args
        called["shape"] = tuple(fake_x.shape)
        called["index_topk"] = index_topk
        return sentinel

    monkeypatch.setattr(deepseek_v4, "_attention_real_mlx", fake_attention_real)
    got = deepseek_v4._attention_mlx(real_args, mx.zeros((1, 1, real_args.hidden_size)), {}, index_topk=None)
    assert got is sentinel
    assert called == {"args": real_args, "shape": (1, 1, real_args.hidden_size), "index_topk": None}
