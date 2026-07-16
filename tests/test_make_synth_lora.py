from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

import scripts.fuse_lora_hf as fuse_helper
from tests._fuse_fixture import ALIASES, BASE_SUFFIX, MODULE, build_synthetic_f8_base, read_safetensors

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_synth_lora.py"


def run_make(base: pathlib.Path, out: pathlib.Path, variant: str = "alpha0", *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--base", str(base), "--variant", variant, "--out", str(out), *extra],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def adapter_header(out: pathlib.Path) -> dict[str, dict[str, object]]:
    header, _ = read_safetensors(out / "adapter.safetensors")
    return header


def tensor_array(out: pathlib.Path, name: str) -> np.ndarray:
    header, data = read_safetensors(out / "adapter.safetensors")
    meta = header[name]
    start, end = meta["data_offsets"]
    assert meta["dtype"] == "F32"
    return np.frombuffer(data[start:end], dtype="<f4").reshape(tuple(meta["shape"])).astype(np.float32)


def expected_names(n_layers: int, targets: tuple[str, ...] = ALIASES) -> list[str]:
    names: list[str] = []
    for layer in range(n_layers):
        for alias in targets:
            module = MODULE[alias]
            names.append(f"layers.{layer}.{module}.lora_a")
            names.append(f"layers.{layer}.{module}.lora_b")
    return names


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_missing_required_args_exits_nonzero() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "base" in result.stderr
    assert "variant" in result.stderr
    assert "out" in result.stderr


def test_cli_bad_variant_exits_nonzero(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    result = run_make(base, tmp_path / "adapter", "bogus")
    assert result.returncode != 0
    assert "alpha0" in result.stderr
    assert "nonzero" in result.stderr


def test_alpha0_creates_adapter_and_config_exits_0(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    out = tmp_path / "adapter"
    result = run_make(base, out, "alpha0")
    assert result.returncode == 0, result.stderr
    assert (out / "adapter.safetensors").is_file()
    assert (out / "adapter_config.json").is_file()


def test_alpha0_config_scale_is_zero(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    out = tmp_path / "adapter"
    result = run_make(base, out, "alpha0")
    assert result.returncode == 0, result.stderr
    config = json.loads((out / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["lora_parameters"]["scale"] == 0.0


def test_alpha0_lora_a_and_lora_b_are_zeros(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path, n_layers=2)
    out = tmp_path / "adapter"
    result = run_make(base, out, "alpha0")
    assert result.returncode == 0, result.stderr
    for name in expected_names(2):
        assert np.count_nonzero(tensor_array(out, name)) == 0


def test_alpha0_lora_shapes_match_base(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path, n_layers=2, out_dim=384, in_dim=256)
    out = tmp_path / "adapter"
    result = run_make(base, out, "alpha0", "--rank", "8")
    assert result.returncode == 0, result.stderr
    header = adapter_header(out)
    for layer in range(2):
        for alias in ALIASES:
            base_key = f"layers.{layer}.attn.{BASE_SUFFIX[alias]}.weight"
            base_shape = read_safetensors(base / "model-00001-of-00001.safetensors")[0][base_key]["shape"]
            out_dim, in_dim = base_shape
            module = MODULE[alias]
            assert header[f"layers.{layer}.{module}.lora_a"]["shape"] == [in_dim, 8]
            assert header[f"layers.{layer}.{module}.lora_b"]["shape"] == [8, out_dim]


def test_alpha0_dtype_float32(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    out = tmp_path / "adapter"
    result = run_make(base, out, "alpha0")
    assert result.returncode == 0, result.stderr
    assert {meta["dtype"] for meta in adapter_header(out).values()} == {"F32"}


def test_alpha0_emits_all_43_layers_default(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path, n_layers=43, out_dim=128, in_dim=128)
    out = tmp_path / "adapter"
    result = run_make(base, out, "alpha0")
    assert result.returncode == 0, result.stderr
    assert sorted(adapter_header(out)) == sorted(expected_names(43))


def test_nonzero_creates_adapter_exits_0(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    out = tmp_path / "adapter"
    result = run_make(base, out, "nonzero")
    assert result.returncode == 0, result.stderr
    assert (out / "adapter.safetensors").is_file()
    assert (out / "adapter_config.json").is_file()


def test_nonzero_config_scale_default_20(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    out = tmp_path / "adapter"
    result = run_make(base, out, "nonzero")
    assert result.returncode == 0, result.stderr
    config = json.loads((out / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["lora_parameters"]["scale"] == 20.0


def test_nonzero_lora_a_and_lora_b_nonzero(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path, n_layers=2)
    out = tmp_path / "adapter"
    result = run_make(base, out, "nonzero", "--seed", "42")
    assert result.returncode == 0, result.stderr
    for name in expected_names(2):
        assert np.count_nonzero(tensor_array(out, name)) > 0


def test_nonzero_alpha_scale_override(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    out = tmp_path / "adapter"
    result = run_make(base, out, "nonzero", "--alpha-scale", "3.5")
    assert result.returncode == 0, result.stderr
    config = json.loads((out / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["lora_parameters"]["scale"] == 3.5


def test_nonzero_reproducible_same_seed(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    out_a = tmp_path / "adapter_a"
    out_b = tmp_path / "adapter_b"
    result_a = run_make(base, out_a, "nonzero", "--seed", "123")
    result_b = run_make(base, out_b, "nonzero", "--seed", "123")
    assert result_a.returncode == 0, result_a.stderr
    assert result_b.returncode == 0, result_b.stderr
    assert sha256(out_a / "adapter.safetensors") == sha256(out_b / "adapter.safetensors")


def test_nonzero_different_seed_differs(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path)
    out_a = tmp_path / "adapter_a"
    out_b = tmp_path / "adapter_b"
    result_a = run_make(base, out_a, "nonzero", "--seed", "42")
    result_b = run_make(base, out_b, "nonzero", "--seed", "99")
    assert result_a.returncode == 0, result_a.stderr
    assert result_b.returncode == 0, result_b.stderr
    assert sha256(out_a / "adapter.safetensors") != sha256(out_b / "adapter.safetensors")


def test_default_targets_q_a_q_b_kv(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path, n_layers=2)
    default_out = tmp_path / "default_adapter"
    q_a_out = tmp_path / "q_a_adapter"
    default_result = run_make(base, default_out, "alpha0")
    q_a_result = run_make(base, q_a_out, "alpha0", "--targets", "q_a")
    assert default_result.returncode == 0, default_result.stderr
    assert q_a_result.returncode == 0, q_a_result.stderr
    assert sorted(adapter_header(default_out)) == sorted(expected_names(2))
    assert sorted(adapter_header(q_a_out)) == sorted(expected_names(2, ("q_a",)))


def test_reads_real_shapes_from_base(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path, out_dim=256, in_dim=256)
    out = tmp_path / "adapter"
    result = run_make(base, out, "alpha0", "--rank", "8")
    assert result.returncode == 0, result.stderr
    header = adapter_header(out)
    assert header["layers.0.self_attn.q_a_proj.lora_a"]["shape"] == [256, 8]
    assert header["layers.0.self_attn.q_a_proj.lora_b"]["shape"] == [8, 256]


def test_adapter_keys_match_fuse_helper_mlx_naming(tmp_path: pathlib.Path) -> None:
    base = build_synthetic_f8_base(tmp_path, n_layers=2)
    out = tmp_path / "adapter"
    result = run_make(base, out, "alpha0")
    assert result.returncode == 0, result.stderr
    config = json.loads((out / "adapter_config.json").read_text(encoding="utf-8"))
    fmt = fuse_helper._detect_format(config)
    tensors = fuse_helper._load_adapter(out / "adapter.safetensors")
    pairs = fuse_helper._collect_adapter_pairs(tensors, fmt=fmt, requested_targets=ALIASES, ignore_unknown=False)
    assert len(pairs) == 2 * len(ALIASES) * 2
