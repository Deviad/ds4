"""AC1 CSA real compressor parity for Story 13.3b-2b.

Compares the additive MLX real CSA compressor against torch
DeepseekV4CSACompressor.forward at real dimensions. This pins the single-KV-head
Ca/Cb overlap pool, token-major APE `(rate=4, 2*head_dim=1024)` with no
transpose, and compress-YaRN RoPE on the trailing qk_rope_head_dim=64 slice.
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

from test_indexer_mlx_parity import _real_header_shape, real_csa_mlx_args, real_csa_torch_config


def _synth_csa_compressor_fixture(seq_len: int = 257):
    g = torch.Generator(device="cpu").manual_seed(133220)
    hidden = torch.randn((1, seq_len, 4096), generator=g, dtype=torch.float32) * 0.02
    q_residual = torch.randn((1, seq_len, 1024), generator=g, dtype=torch.float32) * 0.025
    weights = {
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
    return hidden, q_residual, weights


def _load_torch_csa_compressor(config, weights):
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4CSACompressor

    module = DeepseekV4CSACompressor(config).eval()
    with torch.no_grad():
        module.kv_proj.weight.copy_(weights["compressor_wkv"])
        module.gate_proj.weight.copy_(weights["compressor_wgate"])
        module.position_bias.copy_(weights["compressor_ape"])
        module.kv_norm.weight.copy_(weights["compressor_norm"])
        module.indexer.kv_proj.weight.copy_(weights["indexer_compressor_wkv"])
        module.indexer.gate_proj.weight.copy_(weights["indexer_compressor_wgate"])
        module.indexer.position_bias.copy_(weights["indexer_compressor_ape"])
        module.indexer.kv_norm.weight.copy_(weights["indexer_compressor_norm"])
        module.indexer.q_b_proj.weight.copy_(weights["indexer_wq_b"])
        module.indexer.scorer.weights_proj.weight.copy_(weights["indexer_proj"])
    return module


def test_csa_compressor_real_mlx_matches_torch_stateless_forward_real_dims():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _csa_compressor_real_mlx

    assert _real_header_shape("layers.2.attn.compressor.wkv.weight") == (1024, 4096)
    assert _real_header_shape("layers.2.attn.compressor.wgate.weight") == (1024, 4096)
    assert _real_header_shape("layers.2.attn.compressor.ape") == (4, 1024)
    assert _real_header_shape("layers.2.attn.compressor.norm.weight") == (512,)

    args = real_csa_mlx_args()
    hidden, q_residual, weights = _synth_csa_compressor_fixture()
    position_ids = torch.arange(hidden.shape[1], dtype=torch.long).unsqueeze(0)
    torch_module = _load_torch_csa_compressor(real_csa_torch_config(), weights)
    with torch.no_grad():
        expected_kv, _expected_bias = torch_module(
            hidden,
            q_residual,
            position_ids,
            past_key_values=None,
            layer_idx=0,
        )

    got_kv = _csa_compressor_real_mlx(
        args,
        mx.array(hidden.numpy()),
        {key: mx.array(value.numpy()) for key, value in weights.items()},
    )
    mx.eval(got_kv)
    got_kv_t = torch.tensor(got_kv.tolist(), dtype=torch.float32)

    assert tuple(got_kv.shape) == (1, 1, hidden.shape[1] // 4, 512)
    torch.testing.assert_close(got_kv_t, expected_kv.float(), atol=1e-2, rtol=0.0)
