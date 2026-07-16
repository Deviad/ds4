from __future__ import annotations

import json
import pathlib
import time

import pytest

from scripts import finetune_ds4
from tests._fuse_fixture import SHARD, write_safetensors


def _write_required_hf_files(root: pathlib.Path, tensors: list[tuple[str, str, list[int], bytes]], *, manifest: bool = True) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    write_safetensors(root / SHARD, tensors)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(len(data) for _, _, _, data in tensors)},
                "weight_map": {name: SHARD for name, _, _, _ in tensors},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    if manifest:
        (root / "fuse-manifest.json").write_text(json.dumps({"schema": 1}), encoding="utf-8")
    return root


def _valid_tensors(*, expert_count: int = 1) -> list[tuple[str, str, list[int], bytes]]:
    tensors: list[tuple[str, str, list[int], bytes]] = [
        ("layers.0.attn.wq_a.weight", "BF16", [1], b"\x00\x00"),
    ]
    for expert in range(expert_count):
        tensors.append((f"layers.0.ffn.experts.{expert}.weight", "I8", [1], b"\x00"))
        tensors.append((f"layers.0.ffn.experts.{expert}.scale", "F8_E8M0", [1], b"\x7f"))
    return tensors


def test_fused_gate_is_linear_for_10k_tensor_single_shard(tmp_path: pathlib.Path) -> None:
    tensors = _valid_tensors(expert_count=5_000)
    assert len(tensors) >= 10_000
    fused = _write_required_hf_files(tmp_path / "fused-large", tensors)

    start = time.perf_counter()
    assert finetune_ds4.validate_fused_hf_safetensors_dir(fused) == 0
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0, f"fused-HF gate took {elapsed:.3f}s for {len(tensors)} tensors; expected O(N) parse-once-per-shard lookup"


@pytest.mark.parametrize(
    ("tensors", "match"),
    [
        (
            [
                ("layers.0.attn.wq_a.weight", "F8_E4M3", [1], b"\x00"),
                ("layers.0.ffn.experts.0.weight", "I8", [1], b"\x00"),
                ("layers.0.ffn.experts.0.scale", "F8_E8M0", [1], b"\x7f"),
            ],
            "fused attn target must be BF16",
        ),
        (
            [
                ("layers.0.attn.wq_a.weight", "BF16", [1], b"\x00\x00"),
                ("layers.0.ffn.experts.0.weight", "BF16", [1], b"\x00\x00"),
                ("layers.0.ffn.experts.0.scale", "F8_E8M0", [1], b"\x7f"),
            ],
            "expert weight must remain I8",
        ),
        (
            [
                ("layers.0.attn.wq_a.weight", "BF16", [1], b"\x00\x00"),
                ("layers.0.ffn.experts.0.weight", "I8", [1], b"\x00"),
                ("layers.0.ffn.experts.0.scale", "BF16", [1], b"\x00\x00"),
            ],
            "expert scale must remain F8_E8M0",
        ),
        (
            [
                ("layers.0.attn.wq_a.weight", "BF16", [1], b"\x00\x00"),
                ("layers.0.attn.wq_a.scale", "F8_E8M0", [1], b"\x7f"),
                ("layers.0.ffn.experts.0.weight", "I8", [1], b"\x00"),
                ("layers.0.ffn.experts.0.scale", "F8_E8M0", [1], b"\x7f"),
            ],
            "scale companion .* must be dropped after fuse",
        ),
        (
            [
                ("layers.0.attn.wq_b.weight", "BF16", [1], b"\x00\x00"),
                ("layers.0.ffn.experts.0.weight", "I8", [1], b"\x00"),
                ("layers.0.ffn.experts.0.scale", "F8_E8M0", [1], b"\x7f"),
            ],
            "missing fused BF16 attn tensor",
        ),
    ],
)
def test_fused_gate_dtype_assertions_remain_fail_closed(
    tmp_path: pathlib.Path,
    tensors: list[tuple[str, str, list[int], bytes]],
    match: str,
) -> None:
    fused = _write_required_hf_files(tmp_path / "fused-invalid", tensors)
    with pytest.raises(finetune_ds4.PlanError, match=match):
        finetune_ds4.validate_fused_hf_safetensors_dir(fused)


def test_fused_gate_requires_manifest(tmp_path: pathlib.Path) -> None:
    fused = _write_required_hf_files(tmp_path / "fused-missing-manifest", _valid_tensors(), manifest=False)
    with pytest.raises(finetune_ds4.PlanError, match="missing fuse manifest"):
        finetune_ds4.validate_fused_hf_safetensors_dir(fused)
