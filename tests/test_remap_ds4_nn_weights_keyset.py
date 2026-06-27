"""Story 13.3b-4 AC1: real ckpt remap key-set equals nn param tree.

This is a header/index-only gate: it probes the real ``deepseek_v4_nn``
``model.parameters()`` tree for the expected keys and shapes, then remaps the
shimmed checkpoint index/header metadata without reading tensor payload bytes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MLX_SRC = _ROOT / "python-envs" / "mlx" / "src"
for _path in (_ROOT, _MLX_SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

mx = pytest.importorskip("mlx.core")

CKPT = Path("/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim")


def _real_nn_config() -> dict[str, object]:
    cfg = json.loads((CKPT / "config.json").read_text(encoding="utf-8"))
    cfg["model_type"] = "deepseek_v4_nn"
    return cfg


def _expected_nn_params(cfg: dict[str, object]):
    from mlx.utils import tree_flatten
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import Model, ModelArgs

    model = Model(ModelArgs.from_dict(cfg))
    return dict(tree_flatten(model.parameters()))


@pytest.mark.skipif(not (CKPT / "model.safetensors.index.json").exists(), reason="real shimmed DS4 checkpoint unavailable")
def test_real_ckpt_header_remap_keyset_matches_probed_nn_parameters_exactly():
    import scripts.remap_ds4_nn_weights as remap

    cfg = _real_nn_config()
    expected = _expected_nn_params(cfg)
    specs, report = remap.remap_header_specs_from_checkpoint(CKPT, config=cfg)

    assert set(specs) == set(expected), {
        "missing": sorted(set(expected) - set(specs))[:20],
        "extra": sorted(set(specs) - set(expected))[:20],
    }

    shape_mismatches = {
        key: {"got": tuple(specs[key].shape), "expected": tuple(value.shape)}
        for key, value in expected.items()
        if tuple(specs[key].shape) != tuple(value.shape)
    }
    assert shape_mismatches == {}

    # BA §0.4 correction: all ape tensors copy checkpoint token-major layout
    # verbatim.  No CSA/indexer/HCA transpose is allowed in the remap plan.
    assert tuple(specs["model.layers.2.self_attn.compressor.ape"].shape) == (4, 1024)
    assert tuple(specs["model.layers.2.self_attn.indexer.compressor.ape"].shape) == (4, 256)
    assert tuple(specs["model.layers.3.self_attn.compressor.ape"].shape) == (128, 512)
    assert report.transforms["model.layers.2.self_attn.compressor.ape"] == "identity"
    assert report.transforms["model.layers.2.self_attn.indexer.compressor.ape"] == "identity"
    assert report.transforms["model.layers.3.self_attn.compressor.ape"] == "identity"

    assert all(".scale" not in key for key in specs if ".self_attn." in key)
    assert report.synthetic_zero_keys == {
        "model.layers.0.mlp.e_score_correction_bias",
        "model.layers.1.mlp.e_score_correction_bias",
        "model.layers.2.mlp.e_score_correction_bias",
    }
    assert report.payload_bytes_read == 0
