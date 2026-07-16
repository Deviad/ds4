#!/usr/bin/env python3
"""Convert HF/PEFT LoRA adapter tensor names to DS4 canonical names.

The converter rewrites only the safetensors header tensor keys.  Tensor
metadata (dtype, shape, data_offsets) and the tensor data section are copied
byte-for-byte when every tensor is mapped.  With --ignore-unknown, only mapped
tensor payload bytes are repacked contiguously so the output remains valid
safetensors; LoRA values are not changed.

Default suffix mappings cover the current DS4 runtime allowlist:

  output:   lm_head, output
  attn_q_a: self_attn.q_a_proj, attn.wq_a, attn_q_a
  attn_q_b: self_attn.q_b_proj, attn.wq_b, attn_q_b
  attn_kv:  self_attn.kv_proj, self_attn.kv_a_proj_with_mqa, attn.wkv, attn_kv

Add or replace suffixes with --map TARGET=SUFFIX[,SUFFIX...] and
--clear-default-maps.  Unknown/non-LoRA tensors fail closed by default; pass
--ignore-unknown only when intentionally extracting the supported DS4 subset.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable

SUPPORTED_TARGETS = ("output", "attn_q_a", "attn_q_b", "attn_kv")

DEFAULT_SUFFIX_MAP: dict[str, list[str]] = {
    "output": ["lm_head", "output"],
    "attn_q_a": ["self_attn.q_a_proj", "attn.wq_a", "attn_q_a"],
    "attn_q_b": ["self_attn.q_b_proj", "attn.wq_b", "attn_q_b"],
    "attn_kv": ["self_attn.kv_proj", "self_attn.kv_a_proj_with_mqa", "attn.wkv", "attn_kv"],
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

# DS4 runtime LoRA loader currently supports only F16 and F32 tensors.
RUNTIME_DTYPES = {"F16", "F32"}

LORA_RE = re.compile(r"^(?P<base>.+)\.lora_(?P<which>[AB])(?:\.(?P<adapter>[^.]+))?\.weight$")
LAYER_PATTERNS = (
    re.compile(r"(?:^|\.)blk\.(\d+)\."),
    re.compile(r"(?:^|\.)layers\.(\d+)\."),
    re.compile(r"(?:^|\.)layer\.(\d+)\."),
    re.compile(r"(?:^|\.)blocks\.(\d+)\."),
    re.compile(r"(?:^|\.)h\.(\d+)\."),
)


class ConversionError(ValueError):
    """Raised when a safetensors adapter cannot be converted safely."""


def die(msg: str) -> None:
    raise ConversionError(msg)


@dataclass(frozen=True)
class ParsedLoraName:
    tensor_name: str
    module_base: str
    which: str
    adapter: str | None

    @property
    def group_key(self) -> tuple[str, str | None]:
        return (self.module_base, self.adapter)


@dataclass
class TensorGroup:
    module_base: str
    adapter: str | None
    a_name: str | None = None
    b_name: str | None = None
    canonical: str | None = None


@dataclass
class ConversionPlan:
    output_header: OrderedDict[str, Any]
    mappings: list[dict[str, str]]
    ignored: list[str]
    input_tensors: int
    metadata: dict[str, Any]


def read_safetensors(path: pathlib.Path) -> tuple[dict[str, Any], bytes, bytes]:
    with path.open("rb") as fp:
        raw_len = fp.read(8)
        if len(raw_len) != 8:
            die(f"{path}: shorter than safetensors header length")
        header_len = int.from_bytes(raw_len, "little")
        header_bytes = fp.read(header_len)
        if len(header_bytes) != header_len:
            die(f"{path}: truncated safetensors header")
        data = fp.read()
    try:
        header = json.loads(header_bytes, object_pairs_hook=OrderedDict)
    except json.JSONDecodeError as exc:
        die(f"{path}: invalid safetensors JSON header: {exc}")
    if not isinstance(header, dict):
        die(f"{path}: safetensors header must be a JSON object")
    return header, header_bytes, data


def tensor_items(header: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            die(f"tensor {name}: metadata must be an object")
        yield name, meta


def product(shape: list[int]) -> int:
    n = 1
    for dim in shape:
        n *= dim
    return n


def read_adapter_config(src: pathlib.Path) -> dict[str, Any] | None:
    """Read optional PEFT adapter_config.json next to the input safetensors."""
    config_path = src.with_name("adapter_config.json")
    if not config_path.is_file():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def alpha_from_config(config: dict[str, Any] | None) -> float | None:
    if not config:
        return None
    alpha = config.get("lora_alpha")
    if isinstance(alpha, (int, float)) and alpha > 0:
        return float(alpha)
    return None


def validate_tensor_metadata(path: pathlib.Path, header: dict[str, Any], data_len: int) -> None:
    ranges: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for name, meta in tensor_items(header):
        if name in seen:
            die(f"{path}: duplicate tensor {name}")
        seen.add(name)
        dtype = meta.get("dtype")
        shape = meta.get("shape")
        offsets = meta.get("data_offsets")
        if not isinstance(dtype, str):
            die(f"{path}:{name}: missing string dtype")
        if dtype not in DTYPE_SIZES:
            die(f"{path}:{name}: unsupported safetensors dtype {dtype}")
        if not isinstance(shape, list) or not all(isinstance(x, int) and x >= 0 for x in shape):
            die(f"{path}:{name}: invalid shape")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(x, int) and x >= 0 for x in offsets):
            die(f"{path}:{name}: invalid data_offsets")
        start, end = offsets
        if end < start or end > data_len:
            die(f"{path}:{name}: data_offsets outside data section")
        expected = product(shape) * DTYPE_SIZES[dtype]
        if end - start != expected:
            die(f"{path}:{name}: data_offsets byte length {end - start} != shape*dtype bytes {expected}")
        if name == "__metadata__":
            continue
        ranges.append((start, end, name))
    ranges.sort()
    last_end = 0
    last_name = None
    for start, end, name in ranges:
        if start < last_end:
            die(f"{path}:{name}: data_offsets overlap previous tensor {last_name}")
        last_end = end
        last_name = name


def require_runtime_dtype(name: str, dtype: str) -> None:
    if dtype not in RUNTIME_DTYPES:
        die(f"{name}: dtype {dtype} is not supported by the DS4 runtime loader (supports {', '.join(sorted(RUNTIME_DTYPES))}); cast the adapter to F16/F32 before conversion")


def require_positive_rank(name: str, shape: list[int]) -> None:
    if not shape or shape[0] == 0:
        die(f"{name}: LoRA rank must be > 0")


def require_2d(name: str, shape: list[int]) -> None:
    if len(shape) != 2:
        die(f"{name}: LoRA weight must be 2D, got shape {shape}")


def validate_lora_pair(a_name: str, a_meta: dict[str, Any], b_name: str, b_meta: dict[str, Any]) -> None:
    a_shape = a_meta.get("shape", [])
    b_shape = b_meta.get("shape", [])
    require_2d(a_name, a_shape)
    require_2d(b_name, b_shape)
    require_positive_rank(a_name, a_shape)
    require_positive_rank(b_name, b_shape)
    if a_shape[0] != b_shape[1]:
        die(f"LoRA rank mismatch for {a_name} / {b_name}: A.shape[0]={a_shape[0]} != B.shape[1]={b_shape[1]}")


def parse_lora_name(name: str) -> ParsedLoraName | None:
    match = LORA_RE.match(name)
    if not match:
        return None
    base = match.group("base")
    if base.endswith(".weight"):
        base = base[: -len(".weight")]
    return ParsedLoraName(
        tensor_name=name,
        module_base=base,
        which=match.group("which"),
        adapter=match.group("adapter"),
    )


def suffix_matches(module_base: str, suffix: str) -> bool:
    suffix = suffix.strip(".")
    return bool(suffix) and (module_base == suffix or module_base.endswith("." + suffix))


def find_layer(module_base: str) -> int | None:
    for pattern in LAYER_PATTERNS:
        match = pattern.search(module_base)
        if match:
            return int(match.group(1))
    return None


def canonical_for(module_base: str, suffix_map: dict[str, list[str]]) -> str | None:
    matches: list[str] = []
    for target, suffixes in suffix_map.items():
        if target not in SUPPORTED_TARGETS:
            die(f"unsupported map target {target!r}; supported targets: {', '.join(SUPPORTED_TARGETS)}")
        if any(suffix_matches(module_base, suffix) for suffix in suffixes):
            matches.append(target)
    if not matches:
        return None
    if len(matches) > 1:
        die(f"{module_base}: suffix mapping is ambiguous: {', '.join(matches)}")
    target = matches[0]
    if target == "output":
        return "output"
    layer = find_layer(module_base)
    if layer is None:
        die(f"{module_base}: matched {target} but no layer index was found")
    return f"blk.{layer}.{target}"


def make_suffix_map(clear_defaults: bool, mappings: list[str] | None) -> dict[str, list[str]]:
    suffix_map = {k: list(v) for k, v in ({} if clear_defaults else DEFAULT_SUFFIX_MAP).items()}
    for raw in mappings or []:
        if "=" not in raw:
            die(f"invalid --map {raw!r}; expected TARGET=SUFFIX[,SUFFIX...]")
        target, suffixes_raw = raw.split("=", 1)
        target = target.strip()
        if target not in SUPPORTED_TARGETS:
            die(f"invalid --map target {target!r}; supported targets: {', '.join(SUPPORTED_TARGETS)}")
        suffixes = [s.strip().strip(".") for s in suffixes_raw.split(",") if s.strip()]
        if not suffixes:
            die(f"invalid --map {raw!r}; no suffixes provided")
        suffix_map.setdefault(target, [])
        for suffix in suffixes:
            if suffix not in suffix_map[target]:
                suffix_map[target].append(suffix)
    return suffix_map


def build_conversion_plan(
    header: dict[str, Any],
    *,
    suffix_map: dict[str, list[str]] | None = None,
    ignore_unknown: bool = False,
    lora_alpha: float | None = None,
) -> ConversionPlan:
    if suffix_map is None:
        suffix_map = make_suffix_map(False, None)
    groups: OrderedDict[tuple[str, str | None], TensorGroup] = OrderedDict()
    ignored: list[str] = []
    input_tensors = 0

    for name, _meta in tensor_items(header):
        input_tensors += 1
        parsed = parse_lora_name(name)
        if not parsed:
            if ignore_unknown:
                ignored.append(name)
                continue
            die(f"{name}: not a LoRA A/B weight tensor")
        key = parsed.group_key
        group = groups.setdefault(key, TensorGroup(parsed.module_base, parsed.adapter))
        if parsed.which == "A":
            if group.a_name is not None:
                die(f"{name}: duplicate lora_A tensor for {parsed.module_base}")
            group.a_name = name
        else:
            if group.b_name is not None:
                die(f"{name}: duplicate lora_B tensor for {parsed.module_base}")
            group.b_name = name

    ds4_metadata: dict[str, Any] = {}
    if "__metadata__" in header and isinstance(header["__metadata__"], dict):
        ds4_metadata.update(header["__metadata__"])
    if lora_alpha is not None:
        ds4_metadata["ds4_lora_alpha"] = str(float(lora_alpha))

    mappings_out: list[dict[str, str]] = []
    output_entries: OrderedDict[str, Any] = OrderedDict()

    for group in groups.values():
        canonical = canonical_for(group.module_base, suffix_map)
        if canonical is None:
            names = [n for n in (group.a_name, group.b_name) if n]
            if ignore_unknown:
                ignored.extend(names)
                continue
            die(f"{group.module_base}: unknown module suffix for LoRA tensor(s) {', '.join(names)}")
        group.canonical = canonical
        if group.a_name is None:
            die(f"{group.b_name}: missing matching lora_A pair")
        if group.b_name is None:
            die(f"{group.a_name}: missing matching lora_B pair")

        validate_lora_pair(group.a_name, header[group.a_name], group.b_name, header[group.b_name])

        for which, source in (("A", group.a_name), ("B", group.b_name)):
            assert source is not None
            target_name = f"{canonical}.lora_{which}.weight"
            if target_name in output_entries:
                die(f"{source}: target collision for {target_name}")
            source_meta = header[source]
            require_runtime_dtype(source, source_meta.get("dtype", ""))
            output_entries[target_name] = dict(source_meta)
            mappings_out.append({"source": source, "target": target_name})

    if not mappings_out:
        die("no supported DS4 LoRA tensor pairs were found")

    output_header: OrderedDict[str, Any] = OrderedDict()
    if ds4_metadata:
        output_header["__metadata__"] = ds4_metadata
    output_header.update(output_entries)

    return ConversionPlan(
        output_header=output_header,
        mappings=mappings_out,
        ignored=ignored,
        input_tensors=input_tensors,
        metadata=ds4_metadata,
    )


def write_safetensors(path: pathlib.Path, header: dict[str, Any], data: bytes) -> None:
    raw_header = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(raw_header).to_bytes(8, "little") + raw_header + data)


def repack_mapped_tensors(
    header: dict[str, Any],
    mappings: list[dict[str, str]],
    data: bytes,
    metadata: dict[str, Any] | None = None,
) -> tuple[OrderedDict[str, Any], bytes]:
    output_header: OrderedDict[str, Any] = OrderedDict()
    merged_metadata: dict[str, Any] = {}
    if "__metadata__" in header and isinstance(header["__metadata__"], dict):
        merged_metadata.update(header["__metadata__"])
    if metadata:
        merged_metadata.update(metadata)
    if merged_metadata:
        output_header["__metadata__"] = merged_metadata
    chunks: list[bytes] = []
    offset = 0
    for mapping in mappings:
        source = mapping["source"]
        target = mapping["target"]
        meta = dict(header[source])
        start, end = meta["data_offsets"]
        payload = data[start:end]
        meta["data_offsets"] = [offset, offset + len(payload)]
        output_header[target] = meta
        chunks.append(payload)
        offset += len(payload)
    return output_header, b"".join(chunks)


def convert_file(
    src: pathlib.Path,
    dst: pathlib.Path,
    *,
    suffix_map: dict[str, list[str]] | None = None,
    ignore_unknown: bool = False,
    dry_run: bool = False,
    lora_alpha: float | None = None,
) -> dict[str, Any]:
    header, _header_bytes, data = read_safetensors(src)
    validate_tensor_metadata(src, header, len(data))
    if lora_alpha is None:
        lora_alpha = alpha_from_config(read_adapter_config(src))
    plan = build_conversion_plan(header, suffix_map=suffix_map, ignore_unknown=ignore_unknown, lora_alpha=lora_alpha)
    output_header = plan.output_header
    output_data = data
    if plan.ignored:
        output_header, output_data = repack_mapped_tensors(header, plan.mappings, data, plan.metadata)
    if not dry_run:
        if src.resolve() == dst.resolve():
            die("input and output paths must be different")
        write_safetensors(dst, output_header, output_data)
    return {
        "source": str(src),
        "destination": str(dst),
        "dry_run": dry_run,
        "input_tensors": plan.input_tensors,
        "mapped_tensors": len(plan.mappings),
        "mapped_pairs": len(plan.mappings) // 2,
        "ignored_tensors": plan.ignored,
        "mappings": plan.mappings,
    }


def print_suffix_map(suffix_map: dict[str, list[str]]) -> None:
    print(json.dumps({"suffix_mappings": suffix_map}, indent=2, sort_keys=True))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=pathlib.Path, help="HF/PEFT LoRA adapter safetensors")
    parser.add_argument("output", nargs="?", type=pathlib.Path, help="DS4 canonical adapter safetensors to write")
    parser.add_argument("--dry-run", action="store_true", help="validate and print tensor mappings without writing output")
    parser.add_argument("--list-mappings", action="store_true", help="print configured HF suffix -> DS4 target mappings")
    parser.add_argument(
        "--ignore-unknown",
        action="store_true",
        help="skip non-LoRA tensors and LoRA module suffixes not mapped to DS4; missing/colliding supported pairs still fail",
    )
    parser.add_argument(
        "--clear-default-maps",
        action="store_true",
        help="start with no suffix mappings; use with --map to define an explicit training-stack map",
    )
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="TARGET=SUFFIX[,SUFFIX...]",
        help="add module-name suffixes for output, attn_q_a, attn_q_b, or attn_kv",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        suffix_map = make_suffix_map(args.clear_default_maps, args.map)
        if args.list_mappings:
            print_suffix_map(suffix_map)
            if args.input is None:
                return 0
        if args.input is None:
            die("missing input safetensors path")
        if args.output is None:
            if args.dry_run:
                args.output = pathlib.Path("<dry-run>")
            else:
                die("missing output safetensors path")
        adapter_config = read_adapter_config(args.input)
        lora_alpha = alpha_from_config(adapter_config)
        summary = convert_file(
            args.input,
            args.output,
            suffix_map=suffix_map,
            ignore_unknown=args.ignore_unknown,
            dry_run=args.dry_run,
            lora_alpha=lora_alpha,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
