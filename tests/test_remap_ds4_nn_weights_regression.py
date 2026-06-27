"""Story 13.3b-4 AC3: remap script does not perturb tiny CSA path."""

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

from test_13_3b_1_dispatch_additive import _tiny_csa_args, _tiny_hidden, _tiny_weights


def test_tiny_csa_dispatch_remains_byte_identical_after_remap_module_import():
    import scripts.remap_ds4_nn_weights as remap
    from ds4_ft_mlx.vendor.mlx_lm_models import deepseek_v4

    tiny_args = _tiny_csa_args()
    direct = deepseek_v4._csa_attention_mlx(tiny_args, _tiny_hidden(), _tiny_weights(), index_topk=2)
    routed = deepseek_v4._attention_mlx(tiny_args, _tiny_hidden(), _tiny_weights(), index_topk=2)
    mx.eval(direct, routed)

    assert routed.tolist() == direct.tolist()
    assert remap.remap_key("layers.2.attn.compressor.ape").target == "model.layers.2.self_attn.compressor.ape"
    assert remap.remap_key("layers.2.attn.compressor.ape").transform == "identity"
    assert remap.remap_key("layers.2.attn.indexer.compressor.ape").transform == "identity"
