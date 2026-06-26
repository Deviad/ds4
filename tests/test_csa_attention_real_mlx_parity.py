"""AC2 CSA real attention parity for Story 13.3b-2b.

Compares the additive MLX `_csa_attention_real_mlx` helper against torch
DeepseekV4Attention.forward for a compressed_sparse_attention layer. This pins
CSA compressor real keys, block_bias scatter from `_indexer_mlx` top-k
sentinels, KV append, multi-head broadcast from one KV head, per-head sinks, and
grouped-o output projection.
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

from test_attention_real_mlx_hca_parity import _sliding_mask
from test_indexer_mlx_parity import _real_header_shape, real_csa_mlx_args, real_csa_torch_config


def _synth_csa_attention_fixture(seq_len: int = 129):
    g = torch.Generator(device="cpu").manual_seed(133221)
    hidden = torch.randn((1, seq_len, 4096), generator=g, dtype=torch.float32) * 0.02
    weights = {
        "q_a_proj.weight": torch.randn((1024, 4096), generator=g, dtype=torch.float32) * 0.010,
        "q_norm.weight": torch.linspace(0.8, 1.2, 1024, dtype=torch.float32),
        "q_b_proj.weight": torch.randn((64 * 512, 1024), generator=g, dtype=torch.float32) * 0.006,
        "kv_proj.weight": torch.randn((512, 4096), generator=g, dtype=torch.float32) * 0.010,
        "kv_norm.weight": torch.linspace(0.9, 1.1, 512, dtype=torch.float32),
        "o_a_proj.weight": torch.randn((8 * 1024, 8 * 512), generator=g, dtype=torch.float32) * 0.004,
        "o_b_proj.weight": torch.randn((4096, 8 * 1024), generator=g, dtype=torch.float32) * 0.004,
        "sinks": torch.linspace(-0.4, 0.4, 64, dtype=torch.float32),
        "compressor_wkv": torch.randn((2 * 512, 4096), generator=g, dtype=torch.float32) * 0.012,
        "compressor_wgate": torch.randn((2 * 512, 4096), generator=g, dtype=torch.float32) * 0.010,
        "compressor_ape": torch.randn((4, 2 * 512), generator=g, dtype=torch.float32) * 0.01,
        "compressor_norm": torch.linspace(0.85, 1.15, 512, dtype=torch.float32),
        "indexer_compressor_wkv": torch.randn((2 * 128, 4096), generator=g, dtype=torch.float32) * 0.012,
        "indexer_compressor_wgate": torch.randn((2 * 128, 4096), generator=g, dtype=torch.float32) * 0.010,
        "indexer_compressor_ape": torch.randn((4, 2 * 128), generator=g, dtype=torch.float32) * 0.01,
        "indexer_compressor_norm": torch.linspace(0.85, 1.15, 128, dtype=torch.float32),
        "indexer_wq_b": torch.randn((64 * 128, 1024), generator=g, dtype=torch.float32) * 0.006,
        "indexer_proj": torch.randn((64, 4096), generator=g, dtype=torch.float32) * 0.008,
    }
    return hidden, weights


def _load_torch_attention(config, weights):
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Attention

    module = DeepseekV4Attention(config, layer_idx=0).eval()
    with torch.no_grad():
        module.q_a_proj.weight.copy_(weights["q_a_proj.weight"])
        module.q_a_norm.weight.copy_(weights["q_norm.weight"])
        module.q_b_proj.weight.copy_(weights["q_b_proj.weight"])
        module.kv_proj.weight.copy_(weights["kv_proj.weight"])
        module.kv_norm.weight.copy_(weights["kv_norm.weight"])
        module.o_a_proj.weight.copy_(weights["o_a_proj.weight"])
        module.o_b_proj.weight.copy_(weights["o_b_proj.weight"])
        module.sinks.copy_(weights["sinks"])
        module.compressor.kv_proj.weight.copy_(weights["compressor_wkv"])
        module.compressor.gate_proj.weight.copy_(weights["compressor_wgate"])
        module.compressor.position_bias.copy_(weights["compressor_ape"])
        module.compressor.kv_norm.weight.copy_(weights["compressor_norm"])
        module.compressor.indexer.kv_proj.weight.copy_(weights["indexer_compressor_wkv"])
        module.compressor.indexer.gate_proj.weight.copy_(weights["indexer_compressor_wgate"])
        module.compressor.indexer.position_bias.copy_(weights["indexer_compressor_ape"])
        module.compressor.indexer.kv_norm.weight.copy_(weights["indexer_compressor_norm"])
        module.compressor.indexer.q_b_proj.weight.copy_(weights["indexer_wq_b"])
        module.compressor.indexer.scorer.weights_proj.weight.copy_(weights["indexer_proj"])
    return module


def test_csa_attention_real_mlx_matches_torch_attention_forward_real_dims():
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import (
        _attention_mlx,
        _csa_attention_real_mlx,
        _csa_block_bias_mlx,
        _csa_compressor_real_mlx,
        _indexer_mlx,
    )

    assert _real_header_shape("layers.2.attn.compressor.wkv.weight") == (1024, 4096)
    assert _real_header_shape("layers.2.attn.indexer.weights_proj.weight") == (64, 4096)

    args = real_csa_mlx_args()
    config = real_csa_torch_config()
    hidden, weights = _synth_csa_attention_fixture()
    position_ids = torch.arange(hidden.shape[1], dtype=torch.long).unsqueeze(0)
    rotary = DeepseekV4RotaryEmbedding(config)
    position_embeddings = {"compress": rotary(hidden, position_ids=position_ids, layer_type="compress")}
    attention_mask = _sliding_mask(hidden.shape[1], args.sliding_window)
    torch_module = _load_torch_attention(config, weights)
    with torch.no_grad():
        expected, _ = torch_module(
            hidden,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
        )
        expected_compressed, expected_bias = torch_module.compressor(
            hidden,
            torch_module.q_a_norm(torch_module.q_a_proj(hidden)),
            position_ids,
            past_key_values=None,
            layer_idx=0,
        )

    mlx_hidden = mx.array(hidden.numpy())
    mlx_weights = {key: mx.array(value.numpy()) for key, value in weights.items()}
    mlx_position_ids = mx.array(position_ids.numpy())
    q_residual = mx.array(torch_module.q_a_norm(torch_module.q_a_proj(hidden)).detach().numpy())
    got_compressed = _csa_compressor_real_mlx(args, mlx_hidden, mlx_weights)
    top_k_indices = _indexer_mlx(
        args,
        mlx_hidden,
        q_residual,
        mlx_weights,
        position_ids=mlx_position_ids,
        index_topk=512,
    )
    got_bias = _csa_block_bias_mlx(top_k_indices, int(got_compressed.shape[2]), dtype=mlx_hidden.dtype)
    got = _csa_attention_real_mlx(args, mlx_hidden, mlx_weights, index_topk=512)
    routed = _attention_mlx(args, mlx_hidden, mlx_weights, index_topk=512)
    mx.eval(got_compressed, got_bias, got, routed)

    got_compressed_t = torch.tensor(got_compressed.tolist(), dtype=torch.float32)
    got_bias_t = torch.tensor(got_bias.tolist(), dtype=torch.float32)
    got_t = torch.tensor(got.tolist(), dtype=torch.float32)
    routed_t = torch.tensor(routed.tolist(), dtype=torch.float32)

    assert tuple(got_compressed.shape) == tuple(expected_compressed.shape) == (1, 1, hidden.shape[1] // 4, 512)
    assert tuple(got_bias.shape) == tuple(expected_bias.shape) == (1, 1, hidden.shape[1], hidden.shape[1] // 4)
    torch.testing.assert_close(got_compressed_t, expected_compressed.float(), atol=1e-2, rtol=0.0)
    torch.testing.assert_close(torch.isneginf(got_bias_t), torch.isneginf(expected_bias))
    torch.testing.assert_close(
        torch.nan_to_num(got_bias_t, neginf=-1e9),
        torch.nan_to_num(expected_bias.float(), neginf=-1e9),
        atol=0.0,
        rtol=0.0,
    )
    assert tuple(got.shape) == (1, hidden.shape[1], args.hidden_size)
    torch.testing.assert_close(got_t, expected.float(), atol=1e-2, rtol=0.0)
    torch.testing.assert_close(routed_t, got_t, atol=0.0, rtol=0.0)
