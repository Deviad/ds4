from __future__ import annotations

import argparse
import json
import pathlib
import struct
from collections import OrderedDict

import pytest

from scripts import finetune_ds4

SHARD = "model-00001-of-00001.safetensors"


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


def write_fused_dir(
    root: pathlib.Path,
    *,
    include_fused: bool = True,
    expert_dtype: str = "I8",
    include_manifest: bool = True,
    include_scale_companion: bool = False,
) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    tensors: list[tuple[str, str, list[int], bytes]] = []
    if include_fused:
        tensors.append(("layers.0.attn.wq_a.weight", "BF16", [1], b"\x00\x00"))
        if include_scale_companion:
            tensors.append(("layers.0.attn.wq_a.scale", "F8_E8M0", [1, 1], b"\x7f"))
    tensors.append(("layers.0.ffn.experts.0.weight", expert_dtype, [1], b"\x00\x00" if expert_dtype == "BF16" else b"\x00"))
    tensors.append(("layers.0.ffn.experts.0.scale", "F8_E8M0", [1], b"\x7f"))
    write_safetensors(root / SHARD, tensors)
    total_size = sum(len(data) for _, _, _, data in tensors)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": {name: SHARD for name, _, _, _ in tensors}}, indent=2),
        encoding="utf-8",
    )
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    if include_manifest:
        (root / "fuse-manifest.json").write_text(json.dumps({"schema": 1, "layers": [0], "targets": ["q_a"]}), encoding="utf-8")
    return root


def args_for(tmp_path: pathlib.Path, *, step: str = "fuse-hf") -> argparse.Namespace:
    return argparse.Namespace(
        step=step,
        backend="local-mlx",
        hf_model=str(tmp_path / "hf"),
        dataset_root=str(tmp_path / "dataset"),
        mlx_work=str(tmp_path / "work"),
        ds4_root=str(tmp_path / "ds4"),
        ds4_gguf=None,
        split_dir="mlx-4096",
        fused_hf_model=None,
        ds4_imatrix=None,
        adapter_ds4=None,
    )


def test_fuse_hf_catalog_entry_emits_helper_command(tmp_path: pathlib.Path) -> None:
    args = args_for(tmp_path)
    catalog = finetune_ds4.command_catalog(args)
    command = catalog["fuse-hf"][0]
    assert "fuse_lora_hf.py" in command
    assert "--base" in command
    assert "--adapter" in command
    assert "--out" in command


def test_fuse_hf_step_runs_own_gate_when_fused_dir_valid(tmp_path: pathlib.Path) -> None:
    fused = write_fused_dir(tmp_path / "fused-hf")
    assert finetune_ds4.validate_fused_hf_safetensors_dir(fused) == 0


def test_fuse_hf_not_model_4bit_gated(tmp_path: pathlib.Path) -> None:
    args = args_for(tmp_path)
    mlx_work = pathlib.Path(args.mlx_work)
    write_fused_dir(mlx_work / "fused-hf")
    finetune_ds4.check_execute_prerequisites(args)


def test_gate_fail_closed_missing_bf16_fused_tensor(tmp_path: pathlib.Path) -> None:
    fused = write_fused_dir(tmp_path / "fused-hf", include_fused=False)
    with pytest.raises(finetune_ds4.PlanError, match="missing fused BF16 attn tensor"):
        finetune_ds4.validate_fused_hf_safetensors_dir(fused)


def test_gate_fail_closed_bf16_experts(tmp_path: pathlib.Path) -> None:
    fused = write_fused_dir(tmp_path / "fused-hf", expert_dtype="BF16")
    with pytest.raises(finetune_ds4.PlanError, match="expert weight must remain I8"):
        finetune_ds4.validate_fused_hf_safetensors_dir(fused)


def test_gate_fail_closed_missing_manifest(tmp_path: pathlib.Path) -> None:
    fused = write_fused_dir(tmp_path / "fused-hf", include_manifest=False)
    with pytest.raises(finetune_ds4.PlanError, match="fuse-manifest.json"):
        finetune_ds4.validate_fused_hf_safetensors_dir(fused)


def test_gate_fail_closed_scale_companion_present(tmp_path: pathlib.Path) -> None:
    fused = write_fused_dir(tmp_path / "fused-hf", include_scale_companion=True)
    with pytest.raises(finetune_ds4.PlanError, match="scale companion"):
        finetune_ds4.validate_fused_hf_safetensors_dir(fused)
