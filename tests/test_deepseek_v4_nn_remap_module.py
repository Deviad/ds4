"""Story 13.3b-4b: shared deepseek_v4_nn remap module is importable.

The CLI planner and nn ``Model.load_weights`` must consume one package-level
source of truth for ckpt-native -> ``deepseek_v4_nn`` key conversion.
"""

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


def test_remap_core_is_importable_from_package_and_reexported_by_plan_script():
    from ds4_ft_mlx import deepseek_v4_nn_remap as package_remap
    import scripts.remap_ds4_nn_weights as script_remap

    assert package_remap.remap_key("embed.weight").target == "model.embed_tokens.weight"
    assert package_remap.remap_key("layers.0.attn.compressor.ape").transform == "identity"
    assert script_remap.remap_key is package_remap.remap_key
    assert script_remap.remap_weight_dict is package_remap.remap_weight_dict


def test_package_remap_weight_dict_preserves_values_and_synthesizes_hash_bias():
    from ds4_ft_mlx.deepseek_v4_nn_remap import remap_weight_dict

    weights = {
        "embed.weight": mx.ones((3, 2)),
        "head.weight": mx.ones((3, 2)) * 2,
        "layers.0.ffn.gate.tid2eid": mx.array([[0], [1]], dtype=mx.uint8),
    }
    remapped, report = remap_weight_dict(weights, config={"n_routed_experts": 2, "num_hash_layers": 1})

    assert set(remapped) == {
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.0.mlp.tid2eid",
        "model.layers.0.mlp.e_score_correction_bias",
    }
    assert remapped["model.layers.0.mlp.tid2eid"].dtype == mx.int32
    assert tuple(remapped["model.layers.0.mlp.e_score_correction_bias"].shape) == (2,)
    assert report.synthetic_zero_keys == {"model.layers.0.mlp.e_score_correction_bias"}
