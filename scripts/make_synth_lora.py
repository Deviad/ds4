#!/usr/bin/env python3
"""Create synthetic MLX-format LoRA adapters for DS4 fusion smoke tests."""

import argparse
import json
import pathlib
import struct
import sys

import numpy as np

TARGET_TO_BASE = {"q_a": "wq_a", "q_b": "wq_b", "kv": "wkv"}
TARGET_TO_MODULE = {"q_a": "q_a_proj", "q_b": "q_b_proj", "kv": "kv_proj"}
ALLOWED_TARGETS = ("q_a", "q_b", "kv")
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


class CliParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


class SynthError(RuntimeError):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = int(code)


def _product(shape):
    n = 1
    for value in shape:
        n *= int(value)
    return n


def _parse_targets(raw):
    targets = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not targets:
        raise SynthError("--targets: empty target list; allowed: q_a,q_b,kv", 1)
    for target in targets:
        if target not in ALLOWED_TARGETS:
            raise SynthError(f"--targets: forbidden alias '{target}'; allowed: q_a,q_b,kv", 1)
    deduped = []
    for target in targets:
        if target not in deduped:
            deduped.append(target)
    return tuple(deduped)


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SynthError(f"{path}: missing required file", 2) from exc
    except json.JSONDecodeError as exc:
        raise SynthError(f"{path}: invalid JSON: {exc}", 2) from exc
    except OSError as exc:
        raise SynthError(f"{path}: cannot read: {exc}", 2) from exc


def _read_header(path):
    try:
        with path.open("rb") as fh:
            raw_len = fh.read(8)
            if len(raw_len) != 8:
                raise SynthError(f"{path}: invalid safetensors header", 2)
            header_len = struct.unpack("<Q", raw_len)[0]
            header = fh.read(header_len)
            if len(header) != header_len:
                raise SynthError(f"{path}: truncated safetensors header", 2)
            return json.loads(header.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise SynthError(f"{path}: invalid safetensors header JSON: {exc}", 2) from exc
    except OSError as exc:
        raise SynthError(f"{path}: cannot read safetensors: {exc}", 2) from exc


def _load_base_shapes(base, num_layers, targets):
    base = pathlib.Path(base)
    config = _read_json(base / "config.json")
    index = _read_json(base / "model.safetensors.index.json")
    config_layers = config.get("num_hidden_layers")
    if not isinstance(config_layers, int) or config_layers <= 0:
        raise SynthError(f"{base / 'config.json'}: num_hidden_layers must be positive int", 2)
    if num_layers is None:
        num_layers = config_layers
    if num_layers <= 0:
        raise SynthError("--num-layers must be positive", 1)
    if num_layers > config_layers:
        raise SynthError(f"--num-layers {num_layers} exceeds config num_hidden_layers {config_layers}", 2)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SynthError(f"{base / 'model.safetensors.index.json'}: expected non-empty weight_map", 2)

    headers = {}
    shapes = {}
    for layer in range(num_layers):
        for target in targets:
            weight_key = f"layers.{layer}.attn.{TARGET_TO_BASE[target]}.weight"
            shard = weight_map.get(weight_key)
            if not isinstance(shard, str):
                raise SynthError(f"{weight_key}: missing from weight_map", 2)
            if shard not in headers:
                headers[shard] = _read_header(base / shard)
            meta = headers[shard].get(weight_key)
            if not isinstance(meta, dict):
                raise SynthError(f"{base / shard}:{weight_key}: missing tensor header", 2)
            dtype = meta.get("dtype")
            shape = meta.get("shape")
            offsets = meta.get("data_offsets")
            if dtype not in DTYPE_SIZES:
                raise SynthError(f"{weight_key}: unsupported dtype {dtype!r}", 2)
            if not isinstance(shape, list) or len(shape) != 2 or not all(isinstance(x, int) and x > 0 for x in shape):
                raise SynthError(f"{weight_key}: expected 2D positive shape", 2)
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise SynthError(f"{weight_key}: invalid data_offsets", 2)
            if int(offsets[1]) - int(offsets[0]) != _product(shape) * DTYPE_SIZES[dtype]:
                raise SynthError(f"{weight_key}: invalid byte length for dtype/shape", 2)
            shapes[(layer, target)] = (int(shape[0]), int(shape[1]))
    return int(num_layers), shapes


def _f32_bytes(values):
    return np.asarray(values, dtype="<f4").tobytes()


def _make_tensor(shape, variant, rng):
    if variant == "alpha0":
        return np.zeros(shape, dtype=np.float32)
    return rng.normal(0.0, 1e-2, size=shape).astype(np.float32)


def _build_tensors(num_layers, targets, shapes, rank, variant, seed):
    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed + 1)
    tensors = []
    for layer in range(num_layers):
        for target in targets:
            out_dim, in_dim = shapes[(layer, target)]
            module = TARGET_TO_MODULE[target]
            a_name = f"layers.{layer}.self_attn.{module}.lora_a"
            b_name = f"layers.{layer}.self_attn.{module}.lora_b"
            a = _make_tensor((in_dim, rank), variant, rng_a)
            b = _make_tensor((rank, out_dim), variant, rng_b)
            tensors.append((a_name, "F32", (in_dim, rank), _f32_bytes(a)))
            tensors.append((b_name, "F32", (rank, out_dim), _f32_bytes(b)))
    return tensors


def _write_safetensors(path, tensors):
    header = {}
    payload = bytearray()
    offset = 0
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        payload.extend(data)
        offset += len(data)
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)


def create_adapter(args):
    targets = _parse_targets(args.targets)
    rank = int(args.rank)
    if rank <= 0:
        raise SynthError("--rank must be positive", 1)
    num_layers, shapes = _load_base_shapes(pathlib.Path(args.base), args.num_layers, targets)
    tensors = _build_tensors(num_layers, targets, shapes, rank, args.variant, int(args.seed))
    scale = 0.0 if args.variant == "alpha0" else float(args.alpha_scale)
    out = pathlib.Path(args.out)
    try:
        out.mkdir(parents=True, exist_ok=True)
        _write_safetensors(out / "adapter.safetensors", tensors)
        config = {"lora_parameters": {"rank": rank, "scale": scale, "dropout": 0.0}}
        (out / "adapter_config.json").write_text(json.dumps(config, separators=(",", ":")) + "\n", encoding="utf-8")
    except OSError as exc:
        raise SynthError(f"{out}: write failed: {exc}", 3) from exc
    print(f"wrote {len(tensors)} tensors to {out / 'adapter.safetensors'}")
    print(f"wrote adapter_config.json scale={scale}")
    return 0


def build_parser():
    parser = CliParser(description=__doc__)
    parser.add_argument("--base", required=True, help="HF F8 checkpoint directory")
    parser.add_argument("--variant", required=True, choices=("alpha0", "nonzero"), help="synthetic adapter variant")
    parser.add_argument("--out", required=True, help="output adapter directory")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--num-layers", type=int, default=None, help="number of layers; defaults to base config num_hidden_layers")
    parser.add_argument("--alpha-scale", type=float, default=20.0, help="MLX lora_parameters.scale for nonzero variant")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for nonzero variant")
    parser.add_argument("--targets", default="q_a,q_b,kv", help="comma-separated target aliases: q_a,q_b,kv")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return create_adapter(args)
    except SynthError as exc:
        print(str(exc), file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
