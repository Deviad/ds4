"""Story 13.3a-3 model_type wiring for shimmed DeepSeek V4 nn checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MLX_SRC = _ROOT / "python-envs" / "mlx" / "src"
if str(_MLX_SRC) not in sys.path:
    sys.path.insert(0, str(_MLX_SRC))


def _tiny_config(model_type: str) -> dict[str, object]:
    return dict(
        model_type=model_type,
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_hash_layers=1,
        mlp_layer_types=["hash_moe", "moe"],
        hc_mult=2,
        hc_sinkhorn_iters=3,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=32,
        expert_dtype="fp4",
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        qk_rope_head_dim=4,
        compression_ratio=0,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
    )


def test_model_type_wiring_patches_shim_config_and_preserves_parity_route(tmp_path: Path):
    from scripts import shim_ds4_safetensors as shim
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model as ParityModel
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import ModelArgs as ParityModelArgs
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import ModelArgs as NNModelArgs

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    source_config = _tiny_config("deepseek_v4")
    (src / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
    (src / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "dummy"}), encoding="utf-8")

    shim.copy_sidecars(src, dst, force=True)

    copied_config = json.loads((dst / "config.json").read_text(encoding="utf-8"))
    assert copied_config["model_type"] == "deepseek_v4_nn"
    assert json.loads((src / "config.json").read_text(encoding="utf-8"))["model_type"] == "deepseek_v4"
    assert json.loads((dst / "tokenizer_config.json").read_text(encoding="utf-8")) == {"tokenizer_class": "dummy"}
    assert NNModelArgs.from_dict(copied_config).model_type == "deepseek_v4_nn"

    nn_args = NNModelArgs.from_dict(_tiny_config("deepseek_v4_nn"))
    assert nn_args.model_type == "deepseek_v4_nn"

    parity_args = ParityModelArgs.from_dict(_tiny_config("deepseek_v4"))
    assert parity_args.model_type == "deepseek_v4"
    assert ParityModel.__module__.endswith(".deepseek_v4")

    with pytest.raises(ValueError, match="expected model_type deepseek_v4_nn"):
        NNModelArgs.from_dict(_tiny_config("deepseek_v4"))
