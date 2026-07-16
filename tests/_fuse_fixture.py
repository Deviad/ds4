from __future__ import annotations

import json
import pathlib
import struct
from collections import OrderedDict

import numpy as np

SHARD = "model-00001-of-00001.safetensors"
ALIASES = ("q_a", "q_b", "kv")
BASE_SUFFIX = {"q_a": "wq_a", "q_b": "wq_b", "kv": "wkv"}
MODULE = {"q_a": "self_attn.q_a_proj", "q_b": "self_attn.q_b_proj", "kv": "self_attn.kv_proj"}


def _f32_bytes(values: np.ndarray) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def _bf16_zero_bytes(count: int) -> bytes:
    return (np.zeros(count, dtype="<u2")).tobytes()


def write_safetensors(path: pathlib.Path, tensors: list[tuple[str, str, list[int], bytes]]) -> None:
    header: OrderedDict[str, dict[str, object]] = OrderedDict()
    offset = 0
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        payload.extend(data)
        offset += len(data)
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)


def read_safetensors(path: pathlib.Path) -> tuple[dict[str, dict[str, object]], bytes]:
    raw = path.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_len])
    return header, raw[8 + header_len :]


def read_tensor(root: pathlib.Path, name: str) -> tuple[dict[str, object], bytes]:
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shard = index["weight_map"][name]
    header, data = read_safetensors(root / shard)
    meta = header[name]
    start, end = meta["data_offsets"]
    return meta, data[start:end]


def tensor_bytes(root: pathlib.Path, name: str) -> bytes:
    return read_tensor(root, name)[1]


def tensor_dtype(root: pathlib.Path, name: str) -> str:
    return str(read_tensor(root, name)[0]["dtype"])


def build_synthetic_f8_base(
    tmp_path: pathlib.Path,
    *,
    n_layers: int = 3,
    out_dim: int = 256,
    in_dim: int = 256,
    targets: tuple[str, ...] = ALIASES,
    expert_dtype: str = "I8",
) -> pathlib.Path:
    base = tmp_path / "base"
    base.mkdir()
    tensors: list[tuple[str, str, list[int], bytes]] = []
    target_set = set(targets)
    all_attn = list(target_set) + ["wo_a", "wo_b"]
    for layer in range(n_layers):
        for idx, alias in enumerate(all_attn):
            suffix = BASE_SUFFIX.get(alias, alias)
            key = f"layers.{layer}.attn.{suffix}.weight"
            weight = ((np.arange(out_dim * in_dim, dtype=np.uint32) * (7 + layer + idx) + 0x3C + idx) & 0xFF).astype(np.uint8)
            if alias in target_set and weight.size >= 4:
                weight[0] = 0x7F
                weight[1] = 0xFF
                weight[2] = 0x38
                weight[3] = 0xB8
            scale_rows = out_dim // 128 if out_dim % 128 == 0 else 1
            scale_cols = in_dim // 128 if in_dim % 128 == 0 else 1
            scale = (np.array([126, 127, 128, 129], dtype=np.uint8)[: scale_rows * scale_cols]).copy()
            if scale.size < scale_rows * scale_cols:
                scale = np.resize(scale, scale_rows * scale_cols).astype(np.uint8)
            tensors.append((key, "F8_E4M3", [out_dim, in_dim], weight.tobytes()))
            tensors.append((key.replace(".weight", ".scale"), "F8_E8M0", [scale_rows, scale_cols], scale.tobytes()))
        expert_key = f"layers.{layer}.ffn.experts.0.weight"
        if expert_dtype == "I8":
            expert_data = ((np.arange(128 * 128, dtype=np.uint32) * 3 + layer) & 0xFF).astype(np.uint8).tobytes()
        elif expert_dtype == "BF16":
            expert_data = _bf16_zero_bytes(128 * 128)
        else:
            raise AssertionError(expert_dtype)
        tensors.append((expert_key, expert_dtype, [128, 128], expert_data))
        tensors.append((expert_key.replace(".weight", ".scale"), "F8_E8M0", [128, 1], bytes([127]) * 128))
        tensors.append((f"layers.{layer}.attn_norm.weight", "BF16", [out_dim], _bf16_zero_bytes(out_dim)))
    write_safetensors(base / SHARD, tensors)
    weight_map = {name: SHARD for name, _, _, _ in tensors}
    total_size = sum(len(data) for _, _, _, data in tensors)
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2),
        encoding="utf-8",
    )
    (base / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "num_hidden_layers": n_layers,
                "quantization_config": {"attention": "F8_E4M3/F8_E8M0", "experts": "I8/F8_E8M0"},
            }
        ),
        encoding="utf-8",
    )
    (base / "tokenizer.json").write_text("{}", encoding="utf-8")
    (base / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return base


def _lora_matrix(shape: tuple[int, int], seed: int, *, zero: bool = False, perturb: bool = False) -> np.ndarray:
    if zero:
        return np.zeros(shape, dtype=np.float32)
    values = ((np.arange(shape[0] * shape[1], dtype=np.float32) % 17.0) - 8.0).reshape(shape)
    scale = np.float32(0.004 * (seed + 1))
    if perturb:
        scale = np.float32(scale * 7.0)
    return (values * scale).astype(np.float32)


def build_mlx_adapter(
    tmp_path: pathlib.Path,
    *,
    n_layers: int = 3,
    in_dim: int = 256,
    out_dim: int = 256,
    rank: int = 8,
    scale: float = 20.0,
    zero: bool = False,
    perturb_q_b: bool = False,
    extra: bool = False,
) -> tuple[pathlib.Path, pathlib.Path]:
    adapter_dir = tmp_path / f"adapter_mlx_{len(list(tmp_path.glob('adapter_mlx_*')))}"
    adapter_dir.mkdir()
    tensors: list[tuple[str, str, list[int], bytes]] = []
    for layer in range(n_layers):
        for target_index, alias in enumerate(ALIASES):
            module = MODULE[alias]
            perturb = perturb_q_b and alias == "q_b"
            a = _lora_matrix((in_dim, rank), layer * 3 + target_index, zero=zero, perturb=perturb)
            b = _lora_matrix((rank, out_dim), layer * 5 + target_index + 11, zero=zero, perturb=perturb)
            tensors.append((f"layers.{layer}.{module}.lora_a", "F32", [in_dim, rank], _f32_bytes(a)))
            tensors.append((f"layers.{layer}.{module}.lora_b", "F32", [rank, out_dim], _f32_bytes(b)))
    if extra:
        tensors.append(("layers.0.self_attn.extra_proj.lora_a", "F32", [1, 1], _f32_bytes(np.ones((1, 1), dtype=np.float32))))
    adapter = adapter_dir / "adapters.safetensors"
    config = adapter_dir / "adapter_config.json"
    write_safetensors(adapter, tensors)
    config.write_text(
        json.dumps({"lora_parameters": {"rank": rank, "scale": scale, "dropout": 0.0, "keys": [MODULE[a] for a in ALIASES]}}),
        encoding="utf-8",
    )
    return adapter, config
