"""AC2 HCA real attention parity for Story 13.3b-1.

Compares the additive MLX `_attention_real_mlx` HCA branch against torch
DeepseekV4Attention.forward at real dimensions. Core projections are shared with
cr=0 (`wq_a/wq_b/wkv/wo_a/wo_b`, q/kv norms, per-head sinks); only the HCA
compressor weights are branch-specific. The path also verifies compress-RoPE
undo and grouped-o projection in the HCA branch.
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

from test_compress_rope_oracle import real_hca_mlx_args, real_hca_torch_config


def _sliding_mask(seq_len: int, window: int) -> torch.Tensor:
    pos = torch.arange(seq_len)
    mask = torch.where(pos[:, None] >= pos[None, :], torch.tensor(0.0), torch.tensor(float("-inf")))
    if 0 < window < seq_len:
        mask = torch.where((pos[:, None] - pos[None, :]) < window, mask, torch.tensor(float("-inf")))
    return mask.reshape(1, 1, seq_len, seq_len)


def _synth_attention_fixture(seq_len: int = 129):
    g = torch.Generator(device="cpu").manual_seed(13031)
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
        "compressor_wkv": torch.randn((512, 4096), generator=g, dtype=torch.float32) * 0.015,
        "compressor_wgate": torch.randn((512, 4096), generator=g, dtype=torch.float32) * 0.012,
        "compressor_ape": torch.randn((128, 512), generator=g, dtype=torch.float32) * 0.01,
        "compressor_norm": torch.linspace(0.85, 1.15, 512, dtype=torch.float32),
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
    return module


def test_attention_real_mlx_hca_matches_torch_attention_forward_real_dims():
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _attention_real_mlx

    args = real_hca_mlx_args()
    config = real_hca_torch_config()
    hidden, weights = _synth_attention_fixture()
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

    got = _attention_real_mlx(
        args,
        mx.array(hidden.numpy()),
        {key: mx.array(value.numpy()) for key, value in weights.items()},
    )
    mx.eval(got)
    got_t = torch.tensor(got.tolist(), dtype=torch.float32)

    assert tuple(got.shape) == (1, hidden.shape[1], args.hidden_size)
    torch.testing.assert_close(got_t, expected.float(), atol=5e-3, rtol=0.0)
