#!/usr/bin/env python3
"""Rewrite DeepSeek V4 safetensors FP8 dtypes that MLX cannot load yet.

This is a compatibility shim for the DeepSeek V4 Flash HF snapshot whose
safetensors shards contain FP8 tensors.  Current mlx.core.load fails before
mlx-lm can inspect the model on unsupported FP8 dtypes such as:

    RuntimeError: [safetensor] unsupported dtypeF8_E8M0

The shim copies a HF snapshot directory and emulates FP8 support by decoding
F8_E4M3 weights and F8_E8M0 scale tensors to a standard safetensors dtype (BF16
by default, F32 optionally). It preserves tensor names, shapes, shard names,
configs/tokenizer files, and model.safetensors.index.json. It does not mutate
the source snapshot.

F8_E8M0/UE8M0 encoding assumption: each byte is an exponent-only positive scale
with IEEE-like exponent bias 127, so byte e maps to 2**(e - 127). BF16 storage
for exact powers of two is therefore the 16-bit pattern (e << 7). This is meant
for scale tensors, not general model weights; by default the script refuses to
rewrite non-*.scale F8_E8M0 tensors. F8_E4M3 is decoded as E4M3FN-style
sign/exponent/mantissa values for model weights.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import struct
import sys
from collections import Counter, defaultdict
from typing import Any

SUPPORTED_COPY_EXTS = {
    ".json", ".txt", ".model", ".py", ".md", ".tiktoken",
    ".sentencepiece", ".spm",
}

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


def die(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def read_header(path: pathlib.Path) -> tuple[int, dict[str, Any], bytes]:
    with path.open("rb") as fp:
        raw_len = fp.read(8)
        if len(raw_len) != 8:
            die(f"{path}: shorter than safetensors header length")
        header_len = int.from_bytes(raw_len, "little")
        header_bytes = fp.read(header_len)
        if len(header_bytes) != header_len:
            die(f"{path}: truncated safetensors header")
    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as exc:
        die(f"{path}: invalid safetensors JSON header: {exc}")
    if not isinstance(header, dict):
        die(f"{path}: expected safetensors header object")
    return header_len, header, header_bytes


def tensor_items(header: dict[str, Any]):
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            die(f"tensor {name}: expected metadata object")
        yield name, meta


def dtype_size(dtype: str) -> int:
    try:
        return DTYPE_SIZES[dtype]
    except KeyError:
        die(f"unsupported dtype in shim metadata: {dtype}")


def product(xs: list[int]) -> int:
    n = 1
    for x in xs:
        n *= int(x)
    return n


def validate_header(path: pathlib.Path, header: dict[str, Any], data_len: int) -> None:
    ranges: list[tuple[int, int, str]] = []
    for name, meta in tensor_items(header):
        dtype = meta.get("dtype")
        shape = meta.get("shape")
        offsets = meta.get("data_offsets")
        if not isinstance(dtype, str):
            die(f"{path}:{name}: missing string dtype")
        if not isinstance(shape, list) or not all(isinstance(x, int) and x >= 0 for x in shape):
            die(f"{path}:{name}: invalid shape")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(x, int) and x >= 0 for x in offsets):
            die(f"{path}:{name}: invalid data_offsets")
        start, end = offsets
        if end < start or end > data_len:
            die(f"{path}:{name}: data_offsets outside data section")
        expected = product(shape) * dtype_size(dtype)
        if end - start != expected:
            die(f"{path}:{name}: byte length {end - start} != shape*dtype bytes {expected}")
        ranges.append((start, end, name))
    ranges.sort()
    pos = 0
    for start, end, name in ranges:
        if start != pos:
            die(f"{path}:{name}: non-contiguous or overlapping data_offsets at {start}, expected {pos}")
        pos = end
    if pos != data_len:
        die(f"{path}: tensor data ends at {pos}, file data length is {data_len}")


def scan_snapshot(src: pathlib.Path) -> dict[str, Any]:
    if not src.is_dir():
        die(f"{src}: source snapshot directory not found")
    shards = sorted(src.glob("model*.safetensors"))
    if not shards:
        die(f"{src}: no model*.safetensors shards found")
    dtype_counts: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    f8_bad_names: list[str] = []
    f8_by_suffix: Counter[str] = Counter()
    fp8_dtype_counts: Counter[str] = Counter()
    shard_f8: dict[str, int] = {}
    tensor_count = 0
    for shard in shards:
        header_len, header, _ = read_header(shard)
        data_len = shard.stat().st_size - 8 - header_len
        validate_header(shard, header, data_len)
        f8_count = 0
        for name, meta in tensor_items(header):
            dtype = meta["dtype"]
            start, end = meta["data_offsets"]
            dtype_counts[dtype] += 1
            dtype_bytes[dtype] += end - start
            tensor_count += 1
            if dtype.startswith("F8_"):
                fp8_dtype_counts[dtype] += 1
            if dtype == "F8_E8M0":
                f8_count += 1
                suffix = name.rsplit(".", 1)[-1]
                f8_by_suffix[suffix] += 1
                if not name.endswith(".scale"):
                    f8_bad_names.append(name)
        if f8_count:
            shard_f8[shard.name] = f8_count
    return {
        "source": str(src),
        "shards": len(shards),
        "tensors": tensor_count,
        "dtype_counts": dict(dtype_counts),
        "dtype_bytes": dict(dtype_bytes),
        "f8_e8m0_by_suffix": dict(f8_by_suffix),
        "fp8_dtype_counts": dict(fp8_dtype_counts),
        "f8_e8m0_shards": shard_f8,
        "f8_e8m0_non_scale_examples": f8_bad_names[:20],
        "f8_e8m0_non_scale_count": len(f8_bad_names),
    }


def f8_e8m0_to_bf16(raw: bytes) -> bytes:
    # E8M0 encodes unsigned powers of two with bias 127, with 255 as NaN.
    # Byte 0 is 2^-127, which is subnormal in BF16 (0x0040), not zero.
    out = bytearray(len(raw) * 2)
    j = 0
    for b in raw:
        if b == 0:
            bits = 0x0040
        elif b == 255:
            bits = 0x7FC0
        else:
            bits = int(b) << 7
        out[j] = bits & 0xFF
        out[j + 1] = (bits >> 8) & 0xFF
        j += 2
    return bytes(out)


def f8_e8m0_to_f32(raw: bytes) -> bytes:
    out = bytearray(len(raw) * 4)
    j = 0
    for b in raw:
        if b == 0:
            bits = 0x00400000
        elif b == 255:
            bits = 0x7FC00000
        else:
            bits = int(b) << 23
        struct.pack_into("<I", out, j, bits)
        j += 4
    return bytes(out)


def f32_to_bf16_bytes(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    # Round-to-nearest-even when truncating FP32 mantissa to BF16.
    lsb = (bits >> 16) & 1
    rounded = bits + 0x7FFF + lsb
    bf16 = (rounded >> 16) & 0xFFFF
    return struct.pack("<H", bf16)


def f8_e4m3_to_float(byte: int) -> float:
    b = int(byte) & 0xFF
    if b in (0x7F, 0xFF):
        return float("nan")
    sign = -1.0 if (b & 0x80) else 1.0
    exp = (b >> 3) & 0x0F
    mant = b & 0x07
    bias = 7
    if exp == 0:
        if mant == 0:
            return -0.0 if sign < 0 else 0.0
        return sign * (mant / 8.0) * (2.0 ** (1 - bias))
    # E4M3FN treats exponent 0b1111 as finite in common safetensors/PyTorch
    # float8_e4m3fn usage; this decoder maps it through the same formula.
    return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - bias))


def f8_e4m3_to_f32(raw: bytes) -> bytes:
    out = bytearray(len(raw) * 4)
    for i, b in enumerate(raw):
        struct.pack_into("<f", out, i * 4, f8_e4m3_to_float(b))
    return bytes(out)


def f8_e4m3_to_bf16(raw: bytes) -> bytes:
    return b"".join(f32_to_bf16_bytes(f8_e4m3_to_float(b)) for b in raw)


def convert_payload(dtype: str, raw: bytes, out_dtype: str) -> bytes:
    if dtype == "F8_E8M0":
        if out_dtype == "BF16":
            return f8_e8m0_to_bf16(raw)
        if out_dtype == "F32":
            return f8_e8m0_to_f32(raw)
        die(f"unsupported output dtype for F8_E8M0 conversion: {out_dtype}")
    if dtype == "F8_E4M3":
        if out_dtype == "BF16":
            return f8_e4m3_to_bf16(raw)
        if out_dtype == "F32":
            return f8_e4m3_to_f32(raw)
        die(f"unsupported output dtype for F8_E4M3 conversion: {out_dtype}")
    return raw


def rewrite_shard(src: pathlib.Path, dst: pathlib.Path, out_dtype: str, require_scale: bool = True) -> dict[str, Any]:
    header_len, header, _ = read_header(src)
    data_start = 8 + header_len
    file_size = src.stat().st_size
    data_len = file_size - data_start
    validate_header(src, header, data_len)

    new_header: dict[str, Any] = {}
    if "__metadata__" in header:
        new_header["__metadata__"] = header["__metadata__"]

    entries: list[tuple[str, dict[str, Any], int, int]] = []
    f8_count = 0
    fp8_count = 0
    fp8_dtype_counts: Counter[str] = Counter()
    fp8_old_bytes = 0
    f8_e8m0_old_bytes = 0
    new_bytes = 0
    cur = 0
    for name, meta in tensor_items(header):
        dtype = str(meta["dtype"])
        start, end = meta["data_offsets"]
        shape = list(meta["shape"])
        new_dtype = dtype
        byte_len = end - start
        if dtype.startswith("F8_"):
            if dtype == "F8_E8M0" and require_scale and not name.endswith(".scale"):
                die(f"{src}:{name}: refusing to rewrite non-*.scale F8_E8M0 tensor")
            if dtype not in {"F8_E8M0", "F8_E4M3"}:
                die(f"{src}:{name}: unsupported FP8 emulation dtype {dtype}")
            new_dtype = out_dtype
            fp8_count += 1
            fp8_dtype_counts[dtype] += 1
            if dtype == "F8_E8M0":
                f8_count += 1
            fp8_old_bytes += byte_len
            if dtype == "F8_E8M0":
                f8_e8m0_old_bytes += byte_len
            new_len = product(shape) * dtype_size(new_dtype)
        else:
            new_len = byte_len
        new_meta = dict(meta)
        new_meta["dtype"] = new_dtype
        new_meta["data_offsets"] = [cur, cur + new_len]
        new_header[name] = new_meta
        entries.append((name, meta, start, end))
        cur += new_len
        new_bytes += new_len

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    encoded = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    with src.open("rb") as inp, tmp.open("wb") as out:
        out.write(len(encoded).to_bytes(8, "little"))
        out.write(encoded)
        for name, meta, start, end in entries:
            inp.seek(data_start + start)
            raw = inp.read(end - start)
            if len(raw) != end - start:
                die(f"{src}:{name}: truncated tensor payload")
            out.write(convert_payload(str(meta["dtype"]), raw, out_dtype))
    os.replace(tmp, dst)
    # Validate destination after writing.
    new_header_len, written_header, _ = read_header(dst)
    validate_header(dst, written_header, dst.stat().st_size - 8 - new_header_len)
    return {
        "source": str(src),
        "dest": str(dst),
        "f8_e8m0_converted": f8_count,
        "fp8_converted": fp8_count,
        "fp8_dtype_counts": dict(fp8_dtype_counts),
        "fp8_old_bytes": fp8_old_bytes,
        "f8_e8m0_old_bytes": f8_e8m0_old_bytes,
        "data_bytes_written": new_bytes,
    }


def copy_sidecars(src: pathlib.Path, dst: pathlib.Path, force: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name.startswith("model-") and item.suffix == ".safetensors":
            continue
        if item.is_dir():
            continue
        if item.suffix in SUPPORTED_COPY_EXTS or item.name == "model.safetensors.index.json":
            target = dst / item.name
            if target.exists() and not force:
                continue
            if item.name == "config.json":
                config = json.loads(item.read_text(encoding="utf-8"))
                if not isinstance(config, dict):
                    die(f"{item}: expected config.json object")
                config["model_type"] = "deepseek_v4_nn"
                target.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                continue
            shutil.copy2(item, target)


def update_index_total_size(dst: pathlib.Path, total_data_bytes: int) -> None:
    index_path = dst / "model.safetensors.index.json"
    if not index_path.is_file():
        return
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(index, dict):
        return
    metadata = index.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        index["metadata"] = metadata
    metadata["total_size"] = total_data_bytes
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def convert_snapshot(src: pathlib.Path, dst: pathlib.Path, out_dtype: str, shard_limit: int | None, force: bool, require_scale: bool) -> list[dict[str, Any]]:
    if not src.is_dir():
        die(f"{src}: source snapshot directory not found")
    if dst.exists() and any(dst.iterdir()) and not force:
        die(f"{dst}: destination exists and is not empty; pass --force to overwrite shard outputs")
    dst.mkdir(parents=True, exist_ok=True)
    copy_sidecars(src, dst, force=True)
    shards = sorted(src.glob("model*.safetensors"))
    if shard_limit is not None:
        shards = shards[:shard_limit]
    results = []
    for i, shard in enumerate(shards, 1):
        target = dst / shard.name
        print(f"[{i}/{len(shards)}] rewrite {shard.name} -> {target}", flush=True)
        results.append(rewrite_shard(shard, target, out_dtype=out_dtype, require_scale=require_scale))
    update_index_total_size(dst, sum(r["data_bytes_written"] for r in results))
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan", help="scan snapshot safetensors headers")
    scan.add_argument("--src", required=True, type=pathlib.Path)

    conv = sub.add_parser("convert", help="copy snapshot and rewrite FP8 tensors to MLX-readable BF16/F32")
    conv.add_argument("--src", required=True, type=pathlib.Path)
    conv.add_argument("--dst", required=True, type=pathlib.Path)
    conv.add_argument("--out-dtype", choices=("BF16", "F32"), default="BF16")
    conv.add_argument("--shard-limit", type=int, default=None, help="rewrite only first N shards for probe runs")
    conv.add_argument("--force", action="store_true")
    conv.add_argument("--allow-non-scale-f8", action="store_true")
    conv.add_argument("--execute", action="store_true", help="required for writes")
    conv.add_argument("--yes", action="store_true", help="required with --execute")

    args = ap.parse_args(argv)
    if args.cmd == "scan":
        print(json.dumps(scan_snapshot(args.src), indent=2, sort_keys=True))
        return 0

    if args.cmd == "convert":
        if not (args.execute and args.yes):
            print("DRY RUN: no files written. Add --execute --yes to rewrite shards.")
            print(json.dumps(scan_snapshot(args.src), indent=2, sort_keys=True))
            return 0
        results = convert_snapshot(
            args.src,
            args.dst,
            args.out_dtype,
            args.shard_limit,
            args.force,
            require_scale=not args.allow_non_scale_f8,
        )
        print(json.dumps({"dest": str(args.dst), "results": results}, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    main()
