from __future__ import annotations

import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
from collections import OrderedDict

import numpy as np

from tests.ds4_f8_e4m3_e8m0_ocp_witness import (
    bf16_bits_to_f32,
    decode_f8_e4m3_e8m0_to_float32,
    e4m3fn_to_f32,
    f32_to_bf16_bits,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fuse_lora_hf.py"
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
        json.dumps({"model_type": "deepseek_v4", "num_hidden_layers": n_layers, "quantization_config": {"attention": "F8_E4M3/F8_E8M0", "experts": "I8/F8_E8M0"}}),
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


def build_peft_adapter(
    tmp_path: pathlib.Path,
    *,
    n_layers: int = 3,
    in_dim: int = 256,
    out_dim: int = 256,
    rank: int = 8,
    lora_alpha: int = 16,
) -> tuple[pathlib.Path, pathlib.Path]:
    adapter_dir = tmp_path / f"adapter_peft_{len(list(tmp_path.glob('adapter_peft_*')))}"
    adapter_dir.mkdir()
    tensors: list[tuple[str, str, list[int], bytes]] = []
    for layer in range(n_layers):
        for target_index, alias in enumerate(ALIASES):
            module = MODULE[alias]
            a = _lora_matrix((rank, in_dim), layer * 7 + target_index + 21)
            b = _lora_matrix((out_dim, rank), layer * 11 + target_index + 31)
            tensors.append((f"layers.{layer}.{module}.lora_A.weight", "F32", [rank, in_dim], _f32_bytes(a)))
            tensors.append((f"layers.{layer}.{module}.lora_B.weight", "F32", [out_dim, rank], _f32_bytes(b)))
    adapter = adapter_dir / "adapters.safetensors"
    config = adapter_dir / "adapter_config.json"
    write_safetensors(adapter, tensors)
    config.write_text(json.dumps({"lora_alpha": lora_alpha, "r": rank, "target_modules": [MODULE[a] for a in ALIASES]}), encoding="utf-8")
    return adapter, config


def run_fuse(base: pathlib.Path, adapter: pathlib.Path, config: pathlib.Path, out: pathlib.Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--base", str(base), "--adapter", str(adapter), "--adapter-config", str(config), "--out", str(out), *extra],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _adapter_array(adapter: pathlib.Path, name: str) -> np.ndarray:
    meta, data = read_tensor_from_file(adapter, name)
    return np.frombuffer(data, dtype="<f4").reshape(tuple(meta["shape"])).astype(np.float32)


def read_tensor_from_file(path: pathlib.Path, name: str) -> tuple[dict[str, object], bytes]:
    header, data = read_safetensors(path)
    meta = header[name]
    start, end = meta["data_offsets"]
    return meta, data[start:end]


def _bf16_values(raw: bytes) -> np.ndarray:
    bits = np.frombuffer(raw, dtype="<u2")
    return np.array([bf16_bits_to_f32(int(x)) for x in bits], dtype=np.float32)


def _roundtrip_bf16(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1)
    return np.array([bf16_bits_to_f32(f32_to_bf16_bits(float(x))) for x in flat], dtype=np.float32).reshape(values.shape)


def _reference_mlx(base: pathlib.Path, adapter: pathlib.Path, *, layer: int, alias: str, scale: float) -> np.ndarray:
    suffix = BASE_SUFFIX[alias]
    weight_key = f"layers.{layer}.attn.{suffix}.weight"
    scale_key = weight_key.replace(".weight", ".scale")
    weight_meta, weight_bytes = read_tensor(base, weight_key)
    _, scale_bytes = read_tensor(base, scale_key)
    out_dim, in_dim = weight_meta["shape"]
    base_f32 = np.array(decode_f8_e4m3_e8m0_to_float32(weight_bytes, scale_bytes, out_dim=out_dim, in_dim=in_dim), dtype=np.float32).reshape(out_dim, in_dim)
    module = MODULE[alias]
    a = _adapter_array(adapter, f"layers.{layer}.{module}.lora_a")
    b = _adapter_array(adapter, f"layers.{layer}.{module}.lora_b")
    return _roundtrip_bf16(base_f32 + np.float32(scale) * (b.T @ a.T))


def _reference_peft(base: pathlib.Path, adapter: pathlib.Path, *, layer: int, alias: str, alpha: float, rank: int) -> np.ndarray:
    suffix = BASE_SUFFIX[alias]
    weight_key = f"layers.{layer}.attn.{suffix}.weight"
    scale_key = weight_key.replace(".weight", ".scale")
    weight_meta, weight_bytes = read_tensor(base, weight_key)
    _, scale_bytes = read_tensor(base, scale_key)
    out_dim, in_dim = weight_meta["shape"]
    base_f32 = np.array(decode_f8_e4m3_e8m0_to_float32(weight_bytes, scale_bytes, out_dim=out_dim, in_dim=in_dim), dtype=np.float32).reshape(out_dim, in_dim)
    module = MODULE[alias]
    a = _adapter_array(adapter, f"layers.{layer}.{module}.lora_A.weight")
    b = _adapter_array(adapter, f"layers.{layer}.{module}.lora_B.weight")
    return _roundtrip_bf16(base_f32 + np.float32(alpha / rank) * (b @ a))


def test_mlx_format_detection_runs_and_exits_0(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    adapter, config = build_mlx_adapter(tmp_path)
    result = run_fuse(base, adapter, config, tmp_path / "out")
    assert result.returncode == 0, result.stderr


def test_peft_format_detection_runs_and_exits_0(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    adapter, config = build_peft_adapter(tmp_path)
    result = run_fuse(base, adapter, config, tmp_path / "out")
    assert result.returncode == 0, result.stderr


def test_format_detection_fail_closed_neither(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    adapter, config = build_mlx_adapter(tmp_path)
    bad = config.parent / "bad_config.json"
    bad.write_text(json.dumps({"rank": 8}), encoding="utf-8")
    result = run_fuse(base, adapter, bad, tmp_path / "out")
    assert result.returncode != 0
    assert "exactly one" in result.stderr


def test_format_detection_fail_closed_both(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    adapter, config = build_mlx_adapter(tmp_path)
    bad = json.loads(config.read_text(encoding="utf-8"))
    bad.update({"lora_alpha": 16, "target_modules": [MODULE[a] for a in ALIASES]})
    both = config.parent / "both_config.json"
    both.write_text(json.dumps(bad), encoding="utf-8")
    result = run_fuse(base, adapter, both, tmp_path / "out")
    assert result.returncode != 0
    assert "exactly one" in result.stderr


def test_mlx_delta_math_within_tolerance(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    adapter, config = build_mlx_adapter(tmp_path, scale=20.0)
    out = tmp_path / "out"
    result = run_fuse(base, adapter, config, out)
    assert result.returncode == 0, result.stderr
    got = _bf16_values(tensor_bytes(out, "layers.0.attn.wq_a.weight")).reshape(256, 256)
    expected = _reference_mlx(base, adapter, layer=0, alias="q_a", scale=20.0)
    np.testing.assert_allclose(got, expected, atol=1e-3, rtol=0)


def test_peft_delta_math_within_tolerance(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    adapter, config = build_peft_adapter(tmp_path, lora_alpha=16, rank=8)
    out = tmp_path / "out"
    result = run_fuse(base, adapter, config, out)
    assert result.returncode == 0, result.stderr
    got = _bf16_values(tensor_bytes(out, "layers.0.attn.wq_a.weight")).reshape(256, 256)
    expected = _reference_peft(base, adapter, layer=0, alias="q_a", alpha=16, rank=8)
    np.testing.assert_allclose(got, expected, atol=1e-3, rtol=0)


def test_q_b_perturbation_does_not_cross_pair_q_a(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    adapter_a, config_a = build_mlx_adapter(tmp_path, perturb_q_b=False)
    adapter_b, config_b = build_mlx_adapter(tmp_path, perturb_q_b=True)
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    result_a = run_fuse(base, adapter_a, config_a, out_a)
    result_b = run_fuse(base, adapter_b, config_b, out_b)
    assert result_a.returncode == 0, result_a.stderr
    assert result_b.returncode == 0, result_b.stderr
    assert tensor_bytes(out_a, "layers.0.attn.wq_a.weight") == tensor_bytes(out_b, "layers.0.attn.wq_a.weight")


def test_fused_attn_emitted_bf16_and_scale_dropped(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    adapter, config = build_mlx_adapter(tmp_path)
    out = tmp_path / "out"
    result = run_fuse(base, adapter, config, out)
    assert result.returncode == 0, result.stderr
    index = json.loads((out / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert tensor_dtype(out, "layers.0.attn.wq_a.weight") == "BF16"
    assert "layers.0.attn.wq_a.scale" not in index["weight_map"]


def test_experts_byte_untouched_after_fuse(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    adapter, config = build_mlx_adapter(tmp_path)
    out = tmp_path / "out"
    result = run_fuse(base, adapter, config, out)
    assert result.returncode == 0, result.stderr
    for key in ("layers.0.ffn.experts.0.weight", "layers.0.ffn.experts.0.scale"):
        assert tensor_bytes(out, key) == tensor_bytes(base, key)
        assert tensor_dtype(out, key) == tensor_dtype(base, key)


def test_base_read_only_mtime_and_sha_unchanged(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    adapter, config = build_mlx_adapter(tmp_path)
    shard = base / SHARD
    before_mtime = base.stat().st_mtime_ns
    before_bytes = shard.read_bytes()
    result = run_fuse(base, adapter, config, tmp_path / "out")
    assert result.returncode == 0, result.stderr
    assert base.stat().st_mtime_ns == before_mtime
    assert shard.read_bytes() == before_bytes


def test_e4m3fn_max_magnitude_byte_decodes_to_zero(tmp_path: pathlib.Path) -> None:
    assert e4m3fn_to_f32(0x7F) == 0.0
    assert e4m3fn_to_f32(0xFF) == 0.0
    base = build_synthetic_f8_base(tmp_path)
    adapter, config = build_mlx_adapter(tmp_path, zero=True)
    out = tmp_path / "out"
    result = run_fuse(base, adapter, config, out)
    assert result.returncode == 0, result.stderr
    got = _bf16_values(tensor_bytes(out, "layers.0.attn.wq_a.weight"))
    assert got[0] == 0.0
    assert got[1] == 0.0


def test_forbidden_target_lm_head_fail_closed(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    adapter, config = build_mlx_adapter(tmp_path)
    result = run_fuse(base, adapter, config, tmp_path / "out", "--targets", "lm_head")
    assert result.returncode != 0
    assert "lm_head" in result.stderr


def test_forbidden_target_output_fail_closed(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    adapter, config = build_mlx_adapter(tmp_path)
    result = run_fuse(base, adapter, config, tmp_path / "out", "--targets", "output")
    assert result.returncode != 0
    assert "output" in result.stderr


def test_dims_not_divisible_128_fail_closed(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path, n_layers=1, out_dim=130, in_dim=130, targets=("q_a",))
    adapter, config = build_mlx_adapter(tmp_path, n_layers=1, in_dim=130, out_dim=130)
    result = run_fuse(base, adapter, config, tmp_path / "out", "--targets", "q_a", "--ignore-unknown")
    assert result.returncode != 0
    assert "divisible by 128" in result.stderr


def test_expert_dtype_wrong_bf16_fail_closed(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path, expert_dtype="BF16")
    adapter, config = build_mlx_adapter(tmp_path)
    result = run_fuse(base, adapter, config, tmp_path / "out")
    assert result.returncode != 0
    assert "expert" in result.stderr.lower()
    assert "I8" in result.stderr


def test_ignore_unknown_permits_extra_adapter_tensors(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    adapter, config = build_mlx_adapter(tmp_path, extra=True)
    result = run_fuse(base, adapter, config, tmp_path / "out", "--ignore-unknown")
    assert result.returncode == 0, result.stderr


def test_alpha_override_ignored_for_mlx_format(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    adapter, config = build_mlx_adapter(tmp_path, scale=20.0)
    out = tmp_path / "out"
    result = run_fuse(base, adapter, config, out, "--alpha-override", "99.9")
    assert result.returncode == 0, result.stderr
    got = _bf16_values(tensor_bytes(out, "layers.0.attn.wq_a.weight")).reshape(256, 256)
    expected = _reference_mlx(base, adapter, layer=0, alias="q_a", scale=20.0)
    np.testing.assert_allclose(got, expected, atol=1e-3, rtol=0)
