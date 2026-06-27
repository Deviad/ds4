"""Story 13.3b-4 AC2: real sub-checkpoint strict-loads and forwards.

This test materializes only a real three-layer slice from the shimmed DS4
checkpoint: original L0 sliding, original L2 CSA, and original L3 HCA plus the
real global embedding/head/norm/hyper-head tensors.  It remaps those real
payload tensors into a compact three-layer ``deepseek_v4_nn`` model, strict-loads
that model, verifies routed FP4 expert bytes are real packed checkpoint bytes,
and runs a finite forward.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MLX_SRC = _ROOT / "python-envs" / "mlx" / "src"
for _path in (_ROOT, _MLX_SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

mx = pytest.importorskip("mlx.core")

CKPT = Path("/Volumes/Data NVME/mlx-ft/ds4/hf-f8shim")
GLOBAL_KEYS = {
    "embed.weight",
    "head.weight",
    "norm.weight",
    "hc_head_fn",
    "hc_head_base",
    "hc_head_scale",
}
# Compact real layers L0/L2/L3 into local layers L0/L1/L2 for a 3-layer model.
REAL_TO_LOCAL_LAYER = {0: 0, 2: 1, 3: 2}
SUBCHECKPOINT_PAYLOAD_BUDGET = 16 * 1024**3
_DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _real_config() -> dict[str, object]:
    cfg = json.loads((CKPT / "config.json").read_text(encoding="utf-8"))
    cfg["model_type"] = "deepseek_v4_nn"
    return cfg


def _real_three_layer_config() -> dict[str, object]:
    cfg = _real_config()
    cfg["num_hidden_layers"] = 3
    cfg["layer_types"] = [
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ]
    cfg["compress_ratios"] = [0, 4, 128]
    cfg["compress_rates"] = {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}
    # Original L2 is still inside the real checkpoint's hash-routing prefix
    # (num_hash_layers=3).  The compact model therefore has two hash MLP layers:
    # local L0 from real L0 and local L1 from real L2.  This keeps the real
    # tid2eid payload truthful instead of fabricating a CSA MoE gate bias.
    cfg["num_hash_layers"] = 2
    cfg["mlp_layer_types"] = ["hash_moe", "hash_moe", "moe"]
    return cfg


def _weight_map() -> dict[str, str]:
    index = json.loads((CKPT / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in index["weight_map"].items()}


def _header_with_data_start(path: Path) -> tuple[dict[str, object], int]:
    with path.open("rb") as fp:
        raw_len = fp.read(8)
        header_len = int.from_bytes(raw_len, "little")
        header = json.loads(fp.read(header_len))
    return header, 8 + header_len


def _tensor_nbytes(meta: dict[str, object]) -> int:
    total = 1
    for dim in meta["shape"]:
        total *= int(dim)
    return total * _DTYPE_SIZES[str(meta["dtype"])]


def _selected_pairs(weight_map: dict[str, str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    layer_re = re.compile(r"^layers\.(\d+)\.(.+)$")
    for raw_key in sorted(weight_map):
        if raw_key in GLOBAL_KEYS:
            pairs.append((raw_key, raw_key))
            continue
        match = layer_re.match(raw_key)
        if not match:
            continue
        real_layer = int(match.group(1))
        if real_layer in REAL_TO_LOCAL_LAYER:
            pairs.append((raw_key, f"layers.{REAL_TO_LOCAL_LAYER[real_layer]}.{match.group(2)}"))
    return pairs


def _pairs_for_global() -> list[tuple[str, str]]:
    return [(key, key) for key in sorted(GLOBAL_KEYS)]


def _pairs_for_real_layer(weight_map: dict[str, str], real_layer: int) -> list[tuple[str, str]]:
    local_layer = REAL_TO_LOCAL_LAYER[real_layer]
    prefix = f"layers.{real_layer}."
    return [
        (key, f"layers.{local_layer}.{key[len(prefix):]}")
        for key in sorted(weight_map)
        if key.startswith(prefix)
    ]


def _load_real_tensors(
    root: Path,
    weight_map: dict[str, str],
    pairs: list[tuple[str, str]],
    header_cache: dict[str, dict[str, object]],
    shard_cache: dict[str, dict[str, object]],
) -> dict[str, object]:
    from safetensors import safe_open
    from safetensors.mlx import load_file

    assert callable(load_file)

    by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for raw_key, local_key in pairs:
        by_shard[weight_map[raw_key]].append((raw_key, local_key))

    out: dict[str, object] = {}
    for shard, shard_pairs in sorted(by_shard.items()):
        if shard not in header_cache:
            header, _data_start = _header_with_data_start(root / shard)
            header_cache[shard] = header
        header = header_cache[shard]
        # safetensors.safe_open(...).get_tensor is the primary lazy payload path.
        # The local safetensors MLX reader cannot get_tensor(BF16) in this venv
        # (TypeError: data type 'bfloat16' not understood), so BF16 tensors use
        # mx.load shard mmap fallback while retaining only requested keys.
        with safe_open(root / shard, framework="mlx", device="cpu") as handle:
            for raw_key, local_key in shard_pairs:
                meta = header[raw_key]
                if meta["dtype"] == "BF16":
                    if shard not in shard_cache:
                        shard_cache[shard] = mx.load(str(root / shard))
                    out[local_key] = shard_cache[shard][raw_key]
                else:
                    out[local_key] = handle.get_tensor(raw_key)
    return out


def _remapped_real_subcheckpoint(cfg: dict[str, object]) -> tuple[dict[str, object], object, int]:
    import scripts.remap_ds4_nn_weights as remap

    weight_map = _weight_map()
    header_cache: dict[str, dict[str, object]] = {}
    shard_cache: dict[str, dict[str, object]] = {}
    selected = _selected_pairs(weight_map)
    remapped: dict[str, object] = {}

    payload_bytes = 0
    for raw_key, _local_key in selected:
        shard = weight_map[raw_key]
        if shard not in header_cache:
            header, _data_start = _header_with_data_start(CKPT / shard)
            header_cache[shard] = header
        payload_bytes += _tensor_nbytes(header_cache[shard][raw_key])
    assert payload_bytes < SUBCHECKPOINT_PAYLOAD_BUDGET

    raw_global = _load_real_tensors(CKPT, weight_map, _pairs_for_global(), header_cache, shard_cache)
    global_remapped, report = remap.remap_weight_dict(raw_global, config=cfg)
    remapped.update(global_remapped)

    for real_layer in sorted(REAL_TO_LOCAL_LAYER):
        raw_layer = _load_real_tensors(
            CKPT,
            weight_map,
            _pairs_for_real_layer(weight_map, real_layer),
            header_cache,
            shard_cache,
        )
        layer_cfg = dict(cfg)
        layer_cfg["num_hash_layers"] = 0
        layer_remapped, layer_report = remap.remap_weight_dict(raw_layer, config=layer_cfg)
        remapped.update(layer_remapped)
        report.dropped_keys.extend(layer_report.dropped_keys)
        report.stacked_targets.update(layer_report.stacked_targets)
        report.transforms.update(layer_report.transforms)

    return remapped, report, payload_bytes


def _roundtrip_real_probe_with_load_file(tmp_path: Path, tensor: object) -> None:
    from safetensors.mlx import load_file, save_file

    probe = tmp_path / "real-global-probe.safetensors"
    save_file({"hc_head_base": tensor}, str(probe))
    loaded = load_file(str(probe))
    assert tuple(loaded["hc_head_base"].shape) == tuple(tensor.shape)


@pytest.mark.skipif(not (CKPT / "model.safetensors.index.json").exists(), reason="real shimmed DS4 checkpoint unavailable")
def test_real_sliding_csa_hca_subcheckpoint_strict_loads_and_forward_is_finite(tmp_path):
    from mlx.utils import tree_flatten
    from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import Model, ModelArgs

    cfg = _real_three_layer_config()
    (tmp_path / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")

    args = ModelArgs.from_dict(cfg)
    model = Model(args)
    remapped, report, payload_bytes = _remapped_real_subcheckpoint(cfg)

    expected_keys = {key for key, _ in tree_flatten(model.parameters())}
    assert set(remapped) == expected_keys
    assert payload_bytes < SUBCHECKPOINT_PAYLOAD_BUDGET
    assert report.synthetic_zero_keys == {
        "model.layers.0.mlp.e_score_correction_bias",
        "model.layers.1.mlp.e_score_correction_bias",
    }
    assert report.transforms["model.layers.1.self_attn.compressor.ape"] == "identity"
    assert report.transforms["model.layers.1.self_attn.indexer.compressor.ape"] == "identity"
    assert report.transforms["model.layers.2.self_attn.compressor.ape"] == "identity"

    _roundtrip_real_probe_with_load_file(tmp_path, remapped["model.hc_head.base"])

    stacked_w1 = remapped["model.layers.1.mlp.experts.w1_weight"]
    assert tuple(stacked_w1.shape) == (
        int(cfg["n_routed_experts"]),
        int(cfg["moe_intermediate_size"]),
        int(cfg["hidden_size"]) // 2,
    )
    assert stacked_w1.dtype == mx.uint8
    assert bool(mx.any(stacked_w1[0, :8, :8] != 0).item())

    from safetensors import safe_open

    with safe_open(CKPT / _weight_map()["layers.2.ffn.experts.0.w1.weight"], framework="mlx", device="cpu") as handle:
        original_sample = handle.get_tensor("layers.2.ffn.experts.0.w1.weight")[:8, :8].astype(mx.uint8)
    assert bool(mx.all(stacked_w1[0, :8, :8] == original_sample).item())

    model.load_weights(list(remapped.items()), strict=True)
    seq_len = 8
    tokens = mx.array([list(range(seq_len))], dtype=mx.int32)
    out = model(tokens)
    mx.eval(out)

    assert tuple(out.shape) == (1, seq_len, int(cfg["vocab_size"]))
    assert bool(mx.all(mx.isfinite(out)).item())
