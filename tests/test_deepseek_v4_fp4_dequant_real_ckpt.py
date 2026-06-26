"""Story 13.2 AC4 slow-gated real-checkpoint FP4 MoE smoke.

Skipped unless DS4_AC4_REAL_CKPT_ACK=1 because the operator must explicitly
acknowledge the shimmed checkpoint mmap/RSS budget before touching real shards.
"""

from __future__ import annotations

import json
import os
import resource
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MLX_SRC = _ROOT / "python-envs" / "mlx" / "src"
if str(_MLX_SRC) not in sys.path:
    sys.path.insert(0, str(_MLX_SRC))

CKPT = Path(os.environ.get("DS4_AC4_REAL_CKPT", "/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim"))
RSS_LIMIT_BYTES = 460 * 1024**3


def _peak_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024


@pytest.mark.slow
def test_real_shimmed_ckpt_one_expert_fp4_moe_forward_finite_and_under_rss_cap():
    if os.environ.get("DS4_AC4_REAL_CKPT_ACK") != "1":
        pytest.skip("DS4_AC4_REAL_CKPT_ACK=1 required; ~162GB mmap/RSS operator ack deferred to Story 13.3")
    if not CKPT.exists():
        pytest.skip(f"shimmed checkpoint not present: {CKPT}")

    mx = pytest.importorskip("mlx.core")
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors")
    from safetensors import safe_open
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import ModelArgs, _moe_mlx

    index_path = CKPT / "model.safetensors.index.json"
    config_path = CKPT / "config.json"
    index = json.loads(index_path.read_text())
    config = json.loads(config_path.read_text())
    weight_map = index["weight_map"]

    def load_tensor(name: str):
        shard = CKPT / weight_map[name]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name).contiguous()

    args = ModelArgs.from_dict(
        {
            **config,
            "num_hidden_layers": 1,
            "n_routed_experts": 1,
            "num_experts_per_tok": 1,
            "layer_types": ["sliding_attention"],
            "mlp_layer_types": ["moe"],
        }
    )

    # Story 13.1 key-remap smoke in miniature: real ffn.* keys become the
    # vendor _moe_mlx internal mlp.* keys for one routed expert.
    weights = {
        "mlp.gate.weight": mx.array(load_tensor("layers.0.ffn.gate.weight")[:1, :].float().numpy()),
        "mlp.shared_experts.w1.weight": mx.zeros((args.moe_intermediate_size, args.hidden_size), dtype=mx.float32),
        "mlp.shared_experts.w2.weight": mx.zeros((args.hidden_size, args.moe_intermediate_size), dtype=mx.float32),
        "mlp.shared_experts.w3.weight": mx.zeros((args.moe_intermediate_size, args.hidden_size), dtype=mx.float32),
    }
    for proj in ("w1", "w2", "w3"):
        prefix = f"layers.0.ffn.experts.0.{proj}"
        weights[f"mlp.experts.0.{proj}.weight"] = mx.array(load_tensor(f"{prefix}.weight").numpy())
        weights[f"mlp.experts.0.{proj}.scale"] = mx.array(load_tensor(f"{prefix}.scale").float().numpy(), dtype=mx.bfloat16)

    hidden = np.linspace(-0.001, 0.001, args.hidden_size, dtype=np.float32).reshape(1, 1, args.hidden_size)
    out = _moe_mlx(args, mx.array(hidden), weights)
    assert bool(mx.all(mx.isfinite(out)).item())
    assert _peak_rss_bytes() < RSS_LIMIT_BYTES
    assert safetensors.__name__ == "safetensors"
    assert torch.__name__ == "torch"
