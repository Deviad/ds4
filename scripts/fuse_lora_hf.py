#!/usr/bin/env python3
"""Fuse DS4 LoRA adapter deltas into an HF F8 safetensors checkpoint.

This helper is deliberately byte-oriented: it reads HF safetensors headers and
raw payload bytes, reproduces the DS4 C importer F8_E4M3 x F8_E8M0 block-128
attention dequant convention, emits fused attention weights as BF16, and copies
all non-target tensors byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import pathlib
import re
import shutil
import struct
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

TensorData = bytes | memoryview

DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
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

ALLOWED_TARGETS = ("q_a", "q_b", "kv")
TARGET_TO_BASE = {"q_a": "wq_a", "q_b": "wq_b", "kv": "wkv"}
TARGET_TO_MODULE = {"q_a": "self_attn.q_a_proj", "q_b": "self_attn.q_b_proj", "kv": "self_attn.kv_proj"}
FORMAT_ERROR = "adapter_config.json: must contain exactly one of {lora_parameters} (MLX) or {lora_alpha, target_modules} (HF/PEFT)"
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
BASE_TARGET_RE = re.compile(r"^layers\.(\d+)\.attn\.(wq_a|wq_b|wkv)\.weight$")
EXPERT_WEIGHT_RE = re.compile(r"(?:^|\.)ffn\.experts\.\d+\..*\.weight$|(?:^|\.)ffn\.experts\.\d+\.weight$")
EXPERT_SCALE_RE = re.compile(r"(?:^|\.)ffn\.experts\.\d+\..*\.scale$|(?:^|\.)ffn\.experts\.\d+\.scale$")


class FusionError(RuntimeError):
    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = int(code)


@dataclass(frozen=True)
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: str
    data: TensorData

    @property
    def nbytes(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class AdapterTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: TensorData


def _product(shape: Iterable[int]) -> int:
    n = 1
    for value in shape:
        n *= int(value)
    return n


def _read_safetensors_raw(path: pathlib.Path) -> tuple[OrderedDict[str, dict[str, Any]], memoryview, mmap.mmap]:
    mm: mmap.mmap | None = None
    try:
        with path.open("rb") as fh:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if len(mm) < 8:
                raise FusionError(f"{path}: invalid safetensors file", 1)
            header_len = struct.unpack("<Q", mm[:8])[0]
            header_start = 8
            header_end = header_start + header_len
            if header_end > len(mm):
                raise FusionError(f"{path}: safetensors header exceeds file length", 1)
            header_obj = json.loads(mm[header_start:header_end].decode("utf-8"), object_pairs_hook=OrderedDict)
            return header_obj, memoryview(mm)[header_end:], mm
        except Exception:
            mm.close()
            raise
    except OSError as exc:
        raise FusionError(f"{path}: cannot read safetensors: {exc}", 1) from exc
    except json.JSONDecodeError as exc:
        raise FusionError(f"{path}: invalid safetensors header JSON: {exc}", 1) from exc


def _tensor_items(header: OrderedDict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            raise FusionError(f"{name}: invalid tensor metadata", 1)
        yield name, meta


def _validate_meta(path: pathlib.Path, name: str, meta: dict[str, Any], data_len: int) -> tuple[str, tuple[int, ...], int, int]:
    dtype = meta.get("dtype")
    shape = meta.get("shape")
    offsets = meta.get("data_offsets")
    if not isinstance(dtype, str) or dtype not in DTYPE_SIZES:
        raise FusionError(f"{path}:{name}: unsupported safetensors dtype {dtype!r}", 1)
    if not isinstance(shape, list) or not all(isinstance(x, int) and x >= 0 for x in shape):
        raise FusionError(f"{path}:{name}: invalid shape", 1)
    if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(x, int) and x >= 0 for x in offsets):
        raise FusionError(f"{path}:{name}: invalid data_offsets", 1)
    start, end = int(offsets[0]), int(offsets[1])
    expected = _product(shape) * DTYPE_SIZES[dtype]
    if end < start or end > data_len or end - start != expected:
        raise FusionError(f"{path}:{name}: invalid byte length for dtype/shape", 1)
    return dtype, tuple(int(x) for x in shape), start, end


def _load_base(base: pathlib.Path) -> tuple[OrderedDict[str, TensorEntry], dict[str, str], dict[str, str], dict[str, mmap.mmap]]:
    if not base.is_dir():
        raise FusionError(f"{base}: base HF checkpoint path is not a directory", 1)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        if not (base / name).is_file():
            raise FusionError(f"{base / name}: missing required HF file", 1)
    index_path = base / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FusionError(f"{index_path}: invalid JSON: {exc}", 1) from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise FusionError(f"{index_path}: expected non-empty weight_map", 1)

    shard_hashes: dict[str, str] = {}
    entries: OrderedDict[str, TensorEntry] = OrderedDict()
    shard_headers: dict[str, tuple[OrderedDict[str, dict[str, Any]], memoryview]] = {}
    shard_mmaps: dict[str, mmap.mmap] = {}
    for shard in sorted({str(s) for s in weight_map.values()}):
        shard_path = base / shard
        if not shard_path.is_file():
            raise FusionError(f"{shard_path}: missing referenced safetensors shard", 1)
        h = hashlib.sha256()
        with shard_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(64 * 1024 * 1024), b""):
                h.update(chunk)
        shard_hashes[shard] = h.hexdigest()
        header, data, mm = _read_safetensors_raw(shard_path)
        shard_headers[shard] = (header, data)
        shard_mmaps[shard] = mm

    for name, shard in weight_map.items():
        shard = str(shard)
        if shard not in shard_headers:
            raise FusionError(f"{index_path}:{name}: unknown shard {shard}", 1)
        header, data = shard_headers[shard]
        if name not in header:
            raise FusionError(f"{base / shard}:{name}: missing tensor header", 1)
        dtype, shape, start, end = _validate_meta(base / shard, name, header[name], len(data))
        entries[name] = TensorEntry(name=name, dtype=dtype, shape=shape, shard=shard, data=data[start:end])
    return entries, {str(k): str(v) for k, v in weight_map.items()}, shard_hashes, shard_mmaps


def _load_adapter(path: pathlib.Path, *, return_mmap: bool = False):
    if not path.is_file():
        raise FusionError(f"{path}: adapter safetensors file is missing", 1)
    header, data, mm = _read_safetensors_raw(path)
    out: OrderedDict[str, AdapterTensor] = OrderedDict()
    for name, meta in _tensor_items(header):
        dtype, shape, start, end = _validate_meta(path, name, meta, len(data))
        out[name] = AdapterTensor(name=name, dtype=dtype, shape=shape, data=data[start:end])
    if not out:
        raise FusionError(f"{path}: adapter contains no tensors", 5)
    if return_mmap:
        return out, mm
    return out


def _load_config(path: pathlib.Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FusionError(f"{path}: missing adapter_config.json", 2) from exc
    except json.JSONDecodeError as exc:
        raise FusionError(f"{path}: invalid adapter_config.json: {exc}", 2) from exc


def _detect_format(config: dict[str, Any]) -> str:
    is_mlx = isinstance(config.get("lora_parameters"), dict)
    is_peft = "lora_alpha" in config and "target_modules" in config
    if is_mlx == is_peft:
        raise FusionError(FORMAT_ERROR, 2)
    return "mlx" if is_mlx else "peft"


def _parse_targets(value: str | None) -> tuple[str, ...]:
    raw = "q_a,q_b,kv" if value is None else value
    aliases = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not aliases:
        raise FusionError("--targets: empty target list; allowed: q_a,q_b,kv", 3)
    for alias in aliases:
        if alias not in ALLOWED_TARGETS:
            raise FusionError(f"--targets: forbidden alias '{alias}'; allowed: q_a,q_b,kv", 3)
    deduped: list[str] = []
    for alias in aliases:
        if alias not in deduped:
            deduped.append(alias)
    return tuple(deduped)


def _layer_of(name: str) -> int | None:
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _adapter_kind(name: str, fmt: str) -> tuple[int | None, str, str] | None:
    layer = _layer_of(name)
    for alias, module in TARGET_TO_MODULE.items():
        if fmt == "mlx":
            if name.endswith(f"{module}.lora_a"):
                return layer, alias, "a"
            if name.endswith(f"{module}.lora_b"):
                return layer, alias, "b"
        else:
            if name.endswith(f"{module}.lora_A.weight"):
                return layer, alias, "a"
            if name.endswith(f"{module}.lora_B.weight"):
                return layer, alias, "b"
    return None


def _collect_adapter_pairs(
    tensors: OrderedDict[str, AdapterTensor],
    *,
    fmt: str,
    requested_targets: tuple[str, ...],
    ignore_unknown: bool,
) -> dict[tuple[int | None, str, str], AdapterTensor]:
    pairs: dict[tuple[int | None, str, str], AdapterTensor] = {}
    unknown: list[str] = []
    for name, tensor in tensors.items():
        kind = _adapter_kind(name, fmt)
        if kind is None:
            unknown.append(name)
            continue
        layer, alias, which = kind
        if alias not in requested_targets:
            unknown.append(name)
            continue
        key = (layer, alias, which)
        if key in pairs:
            raise FusionError(f"{name}: duplicate adapter tensor for layer={layer} target={alias} lora_{which}", 5)
        pairs[key] = tensor
    if unknown and not ignore_unknown:
        raise FusionError(f"unmapped adapter tensor(s): {', '.join(unknown[:20])}", 5)
    return pairs


def _base_layers(entries: OrderedDict[str, TensorEntry], requested_targets: tuple[str, ...]) -> list[int]:
    wanted_suffixes = {TARGET_TO_BASE[t] for t in requested_targets}
    layers = sorted({int(m.group(1)) for name in entries for m in [BASE_TARGET_RE.match(name)] if m and m.group(2) in wanted_suffixes})
    if not layers:
        raise FusionError(f"base checkpoint has no requested attention target weights for {','.join(requested_targets)}", 4)
    return layers


def _scale_key(weight_key: str) -> str:
    if not weight_key.endswith(".weight"):
        raise FusionError(f"{weight_key}: FP8 tensor without .weight suffix", 4)
    return weight_key[: -len(".weight")] + ".scale"


def _require_tensor(entries: dict[str, TensorEntry], name: str, code: int) -> TensorEntry:
    try:
        return entries[name]
    except KeyError as exc:
        raise FusionError(f"missing tensor {name}", code) from exc


def _require_adapter_pair(
    pairs: dict[tuple[int | None, str, str], AdapterTensor], layer: int, alias: str
) -> tuple[AdapterTensor, AdapterTensor]:
    a = pairs.get((layer, alias, "a")) or pairs.get((None, alias, "a"))
    b = pairs.get((layer, alias, "b")) or pairs.get((None, alias, "b"))
    if a is None or b is None:
        raise FusionError(f"layers.{layer}.{TARGET_TO_MODULE[alias]}: missing requested adapter A/B pair", 5)
    return a, b


def _tensor_to_f32(tensor: AdapterTensor) -> np.ndarray:
    if tensor.dtype == "F32":
        return np.frombuffer(tensor.data, dtype="<f4").reshape(tensor.shape).astype(np.float32)
    if tensor.dtype == "F16":
        return np.frombuffer(tensor.data, dtype="<f2").reshape(tensor.shape).astype(np.float32)
    raise FusionError(f"{tensor.name}: adapter dtype {tensor.dtype} is unsupported; expected F32 or F16", 5)


def _e4m3_table() -> np.ndarray:
    values = np.zeros(256, dtype=np.float32)
    for x in range(256):
        abs_x = x & 0x7F
        sign = bool(x & 0x80)
        if abs_x == 0:
            value = -0.0 if sign else 0.0
        elif abs_x == 0x7F:
            value = 0.0
        else:
            exp = (x >> 3) & 0x0F
            man = x & 0x07
            if exp == 0:
                value = np.ldexp(np.float32(man), -9)
            else:
                value = np.ldexp(np.float32(1.0 + man / 8.0), exp - 7)
            if sign:
                value = -value
        values[x] = np.float32(value)
    return values


_E4M3_TABLE = _e4m3_table()


def _e8m0_values(scale_bytes: bytes, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.frombuffer(scale_bytes, dtype=np.uint8).reshape(shape)
    exponents = raw.astype(np.int32) - 127
    return np.ldexp(np.ones(raw.shape, dtype=np.float32), exponents).astype(np.float32)


def _dequant_fp8_weight(weight: TensorEntry, scale: TensorEntry) -> np.ndarray:
    if weight.dtype != "F8_E4M3" or scale.dtype != "F8_E8M0":
        raise FusionError(f"bad FP8 weight/scale dtype for {weight.name}: {weight.dtype}/{scale.dtype}", 6)
    if len(weight.shape) != 2 or len(scale.shape) != 2:
        raise FusionError(f"{weight.name}: FP8 tensor must be 2D", 6)
    out_dim, in_dim = weight.shape
    if out_dim % 128 or in_dim % 128:
        raise FusionError(f"{weight.name}: FP8 dims are not divisible by 128", 6)
    scale_rows, scale_cols = out_dim // 128, in_dim // 128
    if scale.shape != (scale_rows, scale_cols):
        raise FusionError(f"{scale.name}: FP8 scale shape mismatch", 6)
    weight_f32 = _E4M3_TABLE[np.frombuffer(weight.data, dtype=np.uint8)].reshape(out_dim, in_dim)
    scale_f32 = _e8m0_values(scale.data, scale.shape)
    out = np.empty((out_dim, in_dim), dtype=np.float32)
    for ob in range(scale_rows):
        rows = slice(ob * 128, (ob + 1) * 128)
        for ib in range(scale_cols):
            cols = slice(ib * 128, (ib + 1) * 128)
            out[rows, cols] = weight_f32[rows, cols] * scale_f32[ob, ib]
    return out


def _f32_to_bf16_bytes(values: np.ndarray) -> bytes:
    f32 = np.asarray(values, dtype="<f4")
    u32 = f32.view(np.uint32)
    rounded = ((u32 + np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype("<u2")
    return rounded.tobytes()


def _compute_delta(fmt: str, config: dict[str, Any], a: AdapterTensor, b: AdapterTensor, alpha_override: float | None) -> tuple[np.ndarray, dict[str, float | int | None]]:
    a_arr = _tensor_to_f32(a)
    b_arr = _tensor_to_f32(b)
    if fmt == "mlx":
        params = config["lora_parameters"]
        scale = float(params.get("scale", 20.0))
        rank = int(params.get("rank", a_arr.shape[-1]))
        delta = np.float32(scale) * (b_arr.T @ a_arr.T)
        return delta.astype(np.float32), {"rank": rank, "scale": scale, "alpha": None}
    rank = int(config.get("r", config.get("rank", a_arr.shape[0])))
    alpha = float(config.get("lora_alpha")) if alpha_override is None else float(alpha_override)
    delta = np.float32(alpha / rank) * (b_arr @ a_arr)
    return delta.astype(np.float32), {"rank": rank, "scale": alpha / rank, "alpha": alpha}


def _validate_adapter_matrix(tensor: AdapterTensor) -> None:
    if tensor.dtype not in ("F32", "F16"):
        raise FusionError(f"{tensor.name}: adapter dtype {tensor.dtype} is unsupported; expected F32 or F16", 5)
    if len(tensor.shape) != 2:
        raise FusionError(f"{tensor.name}: adapter tensor must be 2D", 5)


def _planned_delta_shape(fmt: str, a: AdapterTensor, b: AdapterTensor) -> tuple[int, int]:
    _validate_adapter_matrix(a)
    _validate_adapter_matrix(b)
    if fmt == "mlx":
        if a.shape[1] != b.shape[0]:
            raise FusionError(f"{a.name}/{b.name}: incompatible LoRA rank dimensions", 5)
        return int(b.shape[1]), int(a.shape[0])
    if b.shape[1] != a.shape[0]:
        raise FusionError(f"{a.name}/{b.name}: incompatible LoRA rank dimensions", 5)
    return int(b.shape[0]), int(a.shape[1])


def _adapter_params_for_manifest(fmt: str, config: dict[str, Any], a: AdapterTensor, alpha_override: float | None) -> dict[str, float | int | None]:
    if fmt == "mlx":
        params = config["lora_parameters"]
        scale = float(params.get("scale", 20.0))
        rank = int(params.get("rank", a.shape[-1]))
        return {"rank": rank, "scale": scale, "alpha": None}
    rank = int(config.get("r", config.get("rank", a.shape[0])))
    alpha = float(config.get("lora_alpha")) if alpha_override is None else float(alpha_override)
    return {"rank": rank, "scale": alpha / rank, "alpha": alpha}


def _validate_experts(entries: OrderedDict[str, TensorEntry]) -> None:
    for name, tensor in entries.items():
        if EXPERT_WEIGHT_RE.search(name):
            if tensor.dtype != "I8":
                raise FusionError(f"{name}: expert weight must remain I8 (got {tensor.dtype})", 7)
            scale_name = _scale_key(name)
            if scale_name in entries and entries[scale_name].dtype != "F8_E8M0":
                raise FusionError(f"{scale_name}: expert scale must remain F8_E8M0 (got {entries[scale_name].dtype})", 7)
        if EXPERT_SCALE_RE.search(name) and tensor.dtype != "F8_E8M0":
            raise FusionError(f"{name}: expert scale must remain F8_E8M0 (got {tensor.dtype})", 7)


def _write_safetensors(path: pathlib.Path, tensors: OrderedDict[str, tuple[str, tuple[int, ...], int]], payload_for: Callable[[str], TensorData]) -> int:
    header: OrderedDict[str, dict[str, Any]] = OrderedDict()
    offset = 0
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as out_file:
        out_file.write(struct.pack("<Q", len(raw_header)))
        out_file.write(raw_header)
        for name, (_dtype, _shape, nbytes) in tensors.items():
            data = payload_for(name)
            if len(data) != nbytes:
                raise FusionError(f"{name}: payload length {len(data)} != header length {nbytes}", 1)
            out_file.write(data)
    return offset


def _copy_side_files(base: pathlib.Path, out: pathlib.Path) -> None:
    for child in base.iterdir():
        if child.is_file() and not child.name.endswith(".safetensors") and child.name != "model.safetensors.index.json":
            shutil.copy2(child, out / child.name)


def fuse(args: argparse.Namespace) -> int:
    base = pathlib.Path(args.base)
    adapter = pathlib.Path(args.adapter)
    adapter_config = pathlib.Path(args.adapter_config) if args.adapter_config else adapter.parent / "adapter_config.json"
    out = pathlib.Path(args.out)
    targets = _parse_targets(args.targets)

    entries, _weight_map, shard_hashes, base_mmaps = _load_base(base)
    _validate_experts(entries)
    adapter_tensors, adapter_mmap = _load_adapter(adapter, return_mmap=True)
    mmap_lifetimes = list(base_mmaps.values()) + [adapter_mmap]
    config = _load_config(adapter_config)
    fmt = _detect_format(config)
    pairs = _collect_adapter_pairs(adapter_tensors, fmt=fmt, requested_targets=targets, ignore_unknown=bool(args.ignore_unknown))
    layers = _base_layers(entries, targets)

    fuse_plan: dict[str, tuple[TensorEntry, TensorEntry, AdapterTensor, AdapterTensor]] = {}
    dropped_scales: set[str] = set()
    manifest_layers: list[dict[str, Any]] = []
    manifest_rank: int | None = None
    manifest_scale: float | None = None
    manifest_alpha: float | None = None

    for layer in layers:
        for alias in targets:
            suffix = TARGET_TO_BASE[alias]
            weight_key = f"layers.{layer}.attn.{suffix}.weight"
            if weight_key not in entries:
                continue
            scale_key = _scale_key(weight_key)
            weight = entries[weight_key]
            scale = _require_tensor(entries, scale_key, 4)
            a, b = _require_adapter_pair(pairs, layer, alias)
            delta_shape = _planned_delta_shape(fmt, a, b)
            if delta_shape != weight.shape:
                raise FusionError(f"{weight_key}: LoRA delta shape {delta_shape} != base weight shape {weight.shape}", 5)
            params = _adapter_params_for_manifest(fmt, config, a, args.alpha_override)
            fuse_plan[weight_key] = (weight, scale, a, b)
            dropped_scales.add(scale_key)
            manifest_layers.append({"layer": layer, "target": alias, "weight": weight_key})
            manifest_rank = int(params["rank"]) if params["rank"] is not None else manifest_rank
            manifest_scale = float(params["scale"]) if params["scale"] is not None else manifest_scale
            manifest_alpha = None if params["alpha"] is None else float(params["alpha"])

    if not fuse_plan:
        raise FusionError("no attention targets were fused", 5)

    if out.exists():
        if out.is_dir():
            shutil.rmtree(out)
        else:
            raise FusionError(f"{out}: output path exists and is not a directory", 1)
    out.mkdir(parents=True, exist_ok=True)
    _copy_side_files(base, out)

    out_tensors: OrderedDict[str, tuple[str, tuple[int, ...], int]] = OrderedDict()
    for name, tensor in entries.items():
        if name in dropped_scales:
            continue
        if name in fuse_plan:
            out_tensors[name] = ("BF16", tensor.shape, _product(tensor.shape) * DTYPE_SIZES["BF16"])
        else:
            out_tensors[name] = (tensor.dtype, tensor.shape, tensor.nbytes)

    def payload_for(name: str) -> TensorData:
        plan = fuse_plan.get(name)
        if plan is None:
            return entries[name].data
        weight, scale, a, b = plan
        base_f32 = _dequant_fp8_weight(weight, scale)
        delta, _params = _compute_delta(fmt, config, a, b, args.alpha_override)
        if delta.shape != base_f32.shape:
            raise FusionError(f"{name}: LoRA delta shape {delta.shape} != base weight shape {base_f32.shape}", 5)
        return _f32_to_bf16_bytes(base_f32 + delta)

    total_size = _write_safetensors(out / "model-00001-of-00001.safetensors", out_tensors, payload_for)
    weight_map = {name: "model-00001-of-00001.safetensors" for name in out_tensors}
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": 1,
        "format": fmt,
        "targets": list(targets),
        "rank": manifest_rank,
        "scale": manifest_scale,
        "alpha": manifest_alpha,
        "layers": manifest_layers,
        "base_dir": str(base),
        "base_sha256": shard_hashes,
    }
    (out / "fuse-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="HF F8 checkpoint directory")
    parser.add_argument("--adapter", required=True, help="adapter.safetensors path")
    parser.add_argument("--adapter-config", help="adapter_config.json path; defaults beside --adapter")
    parser.add_argument("--out", required=True, help="output fused HF safetensors directory")
    parser.add_argument("--targets", default="q_a,q_b,kv", help="comma-separated target aliases: q_a,q_b,kv")
    parser.add_argument("--ignore-unknown", action="store_true", help="permit extra unmapped adapter tensors")
    parser.add_argument("--alpha-override", type=float, help="override PEFT lora_alpha; ignored for MLX adapters")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return fuse(args)
    except FusionError as exc:
        print(str(exc), file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
