"""AC1 IndexerScorer MLX parity for Story 13.3b-2a.

Ports torch DeepseekV4IndexerScorer.forward at real indexer dimensions:
index_n_heads=64, index_head_dim=128, hidden=4096. The scorer uses fp32
accumulation and the real checkpoint weights_proj layout `(64, 4096)`.
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
torch = pytest.importorskip("torch")


def _synth_scorer_fixture(batch: int = 1, seq_len: int = 17, compressed_len: int = 9):
    g = torch.Generator(device="cpu").manual_seed(133201)
    q = torch.randn((batch, seq_len, 64, 128), generator=g, dtype=torch.float32) * 0.03
    compressed_kv = torch.randn((batch, compressed_len, 128), generator=g, dtype=torch.float32) * 0.04
    hidden = torch.randn((batch, seq_len, 4096), generator=g, dtype=torch.float32) * 0.02
    weights_proj = torch.randn((64, 4096), generator=g, dtype=torch.float32) * 0.01
    return q, compressed_kv, hidden, weights_proj


def _load_torch_scorer(weights_proj):
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4IndexerScorer

    config = DeepseekV4Config(
        model_type="deepseek_v4",
        vocab_size=129280,
        hidden_size=4096,
        num_hidden_layers=1,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        q_lora_rank=1024,
        o_lora_rank=1024,
        qk_rope_head_dim=64,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=512,
        n_routed_experts=256,
        num_experts_per_tok=6,
        n_shared_experts=1,
        moe_intermediate_size=2048,
        expert_dtype="fp4",
        compress_rope_theta=160000,
        rope_theta=10000,
        rope_scaling={
            "type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 65536,
        },
        compress_ratios=[4],
        layer_types=["compressed_sparse_attention"],
        mlp_layer_types=["moe"],
        rms_norm_eps=1e-6,
        sliding_window=128,
        o_groups=8,
        hc_mult=4,
        attention_dropout=0.0,
    )
    module = DeepseekV4IndexerScorer(config).eval()
    with torch.no_grad():
        module.weights_proj.weight.copy_(weights_proj)
    return module


def test_indexer_scorer_mlx_matches_torch_forward_real_dims_fp32_accum():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _indexer_scorer_mlx

    q, compressed_kv, hidden, weights_proj = _synth_scorer_fixture()
    assert tuple(weights_proj.shape) == (64, 4096)
    torch_module = _load_torch_scorer(weights_proj)
    with torch.no_grad():
        expected = torch_module(q, compressed_kv, hidden)

    got = _indexer_scorer_mlx(
        mx.array(q.numpy()),
        mx.array(compressed_kv.numpy()),
        mx.array(hidden.numpy()),
        index_n_heads=64,
        index_head_dim=128,
        weights_proj=mx.array(weights_proj.numpy()),
    )
    mx.eval(got)
    got_t = torch.tensor(got.tolist(), dtype=torch.float32)

    assert tuple(got.shape) == tuple(expected.shape) == (1, 17, 9)
    assert got.dtype == mx.float32
    torch.testing.assert_close(got_t, expected.float(), atol=1e-3, rtol=0.0)
