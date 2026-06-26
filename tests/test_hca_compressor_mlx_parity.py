"""AC1 HCA compressor parity for Story 13.3b-1.

Ports torch DeepseekV4HCACompressor.forward stateless branch at real dimensions:
head_dim=512, rate=128, hidden=4096, q_lora_rank=1024, o_groups=8,
num_heads=64, qk_rope_head_dim=64. The compressor APE is synthesized directly
as (128, 512): no transpose, no CSA Ca/Cb overlap.
"""

from __future__ import annotations

import json
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


def _real_header_shape(key: str):
    from safetensors import safe_open

    root = Path("/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim")
    if not root.exists():
        pytest.skip(f"real hf-f8shim checkpoint absent: {root}")
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shard = root / index["weight_map"][key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return tuple(handle.get_slice(key).get_shape())


def _synth_hca_fixture(seq_len: int = 257):
    g = torch.Generator(device="cpu").manual_seed(1331)
    hidden = torch.randn((1, seq_len, 4096), generator=g, dtype=torch.float32) * 0.02
    weights = {
        "compressor_wkv": torch.randn((512, 4096), generator=g, dtype=torch.float32) * 0.015,
        "compressor_wgate": torch.randn((512, 4096), generator=g, dtype=torch.float32) * 0.012,
        "compressor_ape": torch.randn((128, 512), generator=g, dtype=torch.float32) * 0.01,
        "compressor_norm": torch.linspace(0.85, 1.15, 512, dtype=torch.float32),
    }
    return hidden, weights


def _load_torch_hca_compressor(config, weights):
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HCACompressor

    module = DeepseekV4HCACompressor(config).eval()
    with torch.no_grad():
        module.kv_proj.weight.copy_(weights["compressor_wkv"])
        module.gate_proj.weight.copy_(weights["compressor_wgate"])
        module.position_bias.copy_(weights["compressor_ape"])
        module.kv_norm.weight.copy_(weights["compressor_norm"])
    return module


def test_hca_compressor_mlx_matches_torch_stateless_forward_real_dims():
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _hca_compressor_mlx

    assert _real_header_shape("layers.3.attn.compressor.wkv.weight") == (512, 4096)
    assert _real_header_shape("layers.3.attn.compressor.wgate.weight") == (512, 4096)
    assert _real_header_shape("layers.3.attn.compressor.ape") == (128, 512)
    assert _real_header_shape("layers.3.attn.compressor.norm.weight") == (512,)

    args = real_hca_mlx_args()
    hidden, weights = _synth_hca_fixture()
    position_ids = torch.arange(hidden.shape[1], dtype=torch.long).unsqueeze(0)
    torch_module = _load_torch_hca_compressor(real_hca_torch_config(), weights)
    with torch.no_grad():
        expected_kv, expected_bias = torch_module(
            hidden,
            torch.zeros((1, hidden.shape[1], args.q_lora_rank), dtype=torch.float32),
            position_ids,
            past_key_values=None,
            layer_idx=0,
        )

    got_kv, got_bias = _hca_compressor_mlx(
        args,
        mx.array(hidden.numpy()),
        {key: mx.array(value.numpy()) for key, value in weights.items()},
        position_ids=mx.array(position_ids.numpy()),
    )
    mx.eval(got_kv, got_bias)
    got_kv_t = torch.tensor(got_kv.tolist(), dtype=torch.float32)
    got_bias_t = torch.tensor(got_bias.tolist(), dtype=torch.float32)

    assert tuple(got_kv.shape) == (1, 1, 2, 512)
    assert tuple(got_bias.shape) == (1, 1, hidden.shape[1], 2)
    torch.testing.assert_close(got_kv_t, expected_kv.float(), atol=2e-3, rtol=0.0)
    torch.testing.assert_close(torch.isneginf(got_bias_t), torch.isneginf(expected_bias))
    torch.testing.assert_close(torch.nan_to_num(got_bias_t, neginf=-1e9), torch.nan_to_num(expected_bias.float(), neginf=-1e9), atol=0.0, rtol=0.0)
