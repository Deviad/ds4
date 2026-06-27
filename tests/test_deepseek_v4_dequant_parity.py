#!/usr/bin/env python3
"""Tiny pure-Python dequant parity tests for DeepSeek V4 FP8 helpers.

These tests intentionally avoid loading real checkpoint shards.  They use the
existing `scripts/shim_ds4_safetensors.py` byte-level decoders as the reference
for FP8 special values and scale application semantics.
"""

from __future__ import annotations

import json
import math
import os
import random
import struct
import sys
import unittest
from pathlib import Path

from scripts import shim_ds4_safetensors as shim

REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_SRC = REPO_ROOT / "python-envs" / "mlx" / "src"
if str(MLX_SRC) not in sys.path:
    sys.path.insert(0, str(MLX_SRC))

from ds4_ft_mlx import deepseek_v4_dequant as dq  # noqa: E402
from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import ModelArgs  # noqa: E402


DECLARED_FP8_QUANT = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128],
    "activation_scheme": "dynamic",
}


def real_packing_header_fixture() -> dict[str, dict[str, object]]:
    """Tiny expert-count fixture using real DeepSeek V4 Flash expert dims."""

    return {
        "layers.0.ffn.experts.0.w1.weight": {"dtype": "I8", "shape": [2048, 2048]},
        "layers.0.ffn.experts.0.w1.scale": {"dtype": "F8_E8M0", "shape": [2048, 128]},
        "layers.0.ffn.experts.0.w2.weight": {"dtype": "I8", "shape": [4096, 1024]},
        "layers.0.ffn.experts.0.w2.scale": {"dtype": "F8_E8M0", "shape": [4096, 64]},
        "layers.0.ffn.experts.0.w3.weight": {"dtype": "I8", "shape": [2048, 2048]},
        "layers.0.ffn.experts.0.w3.scale": {"dtype": "F8_E8M0", "shape": [2048, 128]},
        "layers.0.ffn.shared_experts.w1.weight": {"dtype": "F8_E4M3", "shape": [2048, 4096]},
        "layers.0.ffn.shared_experts.w1.scale": {"dtype": "F8_E8M0", "shape": [16, 32]},
    }


def genuinely_ambiguous_shared_header_fixture() -> dict[str, dict[str, object]]:
    """Shared expert fixture where no clean observed 2-D scale tensor exists."""

    header = {name: dict(meta) for name, meta in real_packing_header_fixture().items()}
    header["layers.0.ffn.shared_experts.w1.scale"] = {"dtype": "F8_E8M0", "shape": [512]}
    return header


def fp4_candidate_header_fixture() -> dict[str, dict[str, object]]:
    """Real-shaped packed-byte FP4-candidate fixture, header metadata only."""

    args = ModelArgs()
    hidden = args.hidden_size
    intermediate = args.moe_intermediate_size
    return {
        "layers.0.ffn.experts.0.w1.weight": {"dtype": "U8", "shape": [intermediate, hidden // 2]},
        "layers.0.ffn.experts.0.w1.scale": {"dtype": "F8_E8M0", "shape": [intermediate, hidden // 32]},
        "layers.0.ffn.experts.0.w2.weight": {"dtype": "U8", "shape": [hidden, intermediate // 2]},
        "layers.0.ffn.experts.0.w2.scale": {"dtype": "F8_E8M0", "shape": [hidden, intermediate // 32]},
        "layers.0.ffn.experts.0.w3.weight": {"dtype": "U8", "shape": [intermediate, hidden // 2]},
        "layers.0.ffn.experts.0.w3.scale": {"dtype": "F8_E8M0", "shape": [intermediate, hidden // 32]},
        "layers.0.ffn.experts.1.w1.weight": {"dtype": "U8", "shape": [intermediate, hidden // 2]},
        "layers.0.ffn.experts.1.w1.scale": {"dtype": "F8_E8M0", "shape": [intermediate, hidden // 32]},
        "layers.0.ffn.experts.1.w2.weight": {"dtype": "U8", "shape": [hidden, intermediate // 2]},
        "layers.0.ffn.experts.1.w2.scale": {"dtype": "F8_E8M0", "shape": [hidden, intermediate // 32]},
        "layers.0.ffn.experts.1.w3.weight": {"dtype": "U8", "shape": [intermediate, hidden // 2]},
        "layers.0.ffn.experts.1.w3.scale": {"dtype": "F8_E8M0", "shape": [intermediate, hidden // 32]},
        "layers.0.ffn.shared_experts.w1.weight": {"dtype": "BF16", "shape": [intermediate, hidden]},
        "layers.0.ffn.shared_experts.w1.scale": {"dtype": "BF16", "shape": [intermediate, hidden // 16]},
    }


class StrictHeaderMeta:
    """Mapping-like metadata that fails on any payload/data_offsets access."""

    def __init__(self, dtype: str, shape: list[int]):
        self._allowed = {"dtype": dtype, "shape": shape}

    def __getitem__(self, key: str) -> object:
        if key not in self._allowed:
            raise AssertionError(f"classifier must not access metadata key {key!r}")
        return self._allowed[key]

    def __iter__(self):
        yield "dtype"
        yield "shape"
        yield "data_offsets"

    def __len__(self) -> int:
        return 3

    def get(self, key: str, default: object | None = None) -> object:
        if key not in self._allowed:
            raise AssertionError(f"classifier must not access metadata key {key!r}")
        return self._allowed.get(key, default)


def default_real_checkpoint_dir() -> Path:
    return Path(
        os.environ.get(
            "DS4_HF_MODEL",
            "/Volumes/Data NVME/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/"
            "snapshots/553034d7dd9e06c2eeaee68cf85a17d6d4754cf0",
        )
    )


def f32_bytes_to_floats(raw: bytes) -> list[float]:
    return [x[0] for x in struct.iter_unpack("<f", raw)]


def bf16_bytes(*values: float) -> bytes:
    out = bytearray()
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        out.extend(struct.pack("<H", bits >> 16))
    return bytes(out)


class DeepSeekV4DequantParityTests(unittest.TestCase):
    def _assert_model_4bit_absent_or_valid_current_artifact(self):
        model_4bit = Path("/Volumes/Data NVME/mlx-ft/ds4/model-4bit")
        if not model_4bit.exists():
            return
        cfg = json.loads((model_4bit / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg.get("model_type"), "deepseek_v4_nn")
        self.assertEqual(cfg.get("quantization", {}).get("bits"), 4)

    def assertFloatsEqualWithNan(self, actual: list[float], expected: list[float]) -> None:
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected, strict=True):
            if math.isnan(want):
                self.assertTrue(math.isnan(got), f"expected NaN, got {got!r}")
            else:
                self.assertEqual(got, want)

    def test_f8_e4m3fn_scalar_special_values_match_existing_shim(self):
        raw = bytes([0x00, 0x80, 0x01, 0x38, 0x7E, 0xFE, 0x7F, 0xFF])
        expected = f32_bytes_to_floats(shim.f8_e4m3_to_f32(raw))
        actual = [dq.f8_e4m3fn_to_float(byte) for byte in raw]
        self.assertFloatsEqualWithNan(actual, expected)
        self.assertEqual(math.copysign(1.0, actual[1]), -1.0)
        self.assertTrue(math.isnan(actual[-1]))

    def test_f8_e8m0_scale_scalar_special_values_match_existing_shim(self):
        raw = bytes([0, 1, 126, 127, 128, 254, 255])
        expected = f32_bytes_to_floats(shim.f8_e8m0_to_f32(raw))
        actual = [dq.f8_e8m0_scale_to_float(byte) for byte in raw]
        self.assertFloatsEqualWithNan(actual, expected)
        self.assertEqual(actual[0], 2.0 ** -127)
        self.assertEqual(actual[3], 1.0)
        self.assertTrue(math.isnan(actual[-1]))

    def test_decode_vectors_match_existing_shim_f32_values(self):
        weights = bytes(range(0, 16)) + bytes([0x7F, 0xFF])
        scales = bytes([0, 1, 127, 128, 255])
        self.assertFloatsEqualWithNan(
            dq.decode_f8_e4m3fn(weights),
            f32_bytes_to_floats(shim.f8_e4m3_to_f32(weights)),
        )
        self.assertFloatsEqualWithNan(
            dq.decode_f8_e8m0_scales(scales),
            f32_bytes_to_floats(shim.f8_e8m0_to_f32(scales)),
        )

    def test_apply_f8_e8m0_scales_uses_explicit_broadcast_blocks(self):
        # Two rows x four columns, one scale per row.
        encoded_values = bytes([0x38, 0x40, 0x48, 0x50, 0xB8, 0xC0, 0xC8, 0xD0])
        encoded_scales = bytes([127, 128])  # 1.0 then 2.0
        actual = dq.dequantize_f8_e4m3fn_with_e8m0_scales(
            encoded_values,
            encoded_scales,
            value_shape=(2, 4),
            scale_shape=(2, 1),
        )
        base = dq.decode_f8_e4m3fn(encoded_values)
        expected = base[:4] + [x * 2.0 for x in base[4:]]
        self.assertFloatsEqualWithNan(actual, expected)

    def test_apply_f8_e8m0_scales_propagates_nan_scale(self):
        actual = dq.dequantize_f8_e4m3fn_with_e8m0_scales(
            bytes([0x38, 0x38]),
            bytes([127, 255]),
            value_shape=(2,),
            scale_shape=(2,),
        )
        self.assertEqual(actual[0], 1.0)
        self.assertTrue(math.isnan(actual[1]))

    def test_apply_scales_rejects_implicit_or_incompatible_broadcasting(self):
        with self.assertRaisesRegex(ValueError, "rank"):
            dq.dequantize_f8_e4m3fn_with_e8m0_scales(
                bytes([0x38, 0x38, 0x38, 0x38]),
                bytes([127, 127]),
                value_shape=(2, 2),
                scale_shape=(2,),
            )
        with self.assertRaisesRegex(ValueError, "broadcast"):
            dq.dequantize_f8_e4m3fn_with_e8m0_scales(
                bytes([0x38, 0x38, 0x38, 0x38]),
                bytes([127, 127, 127]),
                value_shape=(2, 2),
                scale_shape=(3, 1),
            )

    def test_i8_affine_tiny_fixture_is_explicit_not_packed_expert_claim(self):
        actual = dq.dequantize_i8_affine(bytes([0, 127, 128, 255]), scales=[0.5], zero_points=[0], shape=(4,), scale_shape=(1,))
        self.assertEqual(actual, [0.0, 63.5, -64.0, -0.5])
        report = dq.run_tiny_i8_affine_fixture()
        self.assertEqual(report["fixture"], "i8-affine-explicit")
        self.assertEqual(report["status"], "ok")
        self.assertIn("not a packed expert layout claim", report["not_covered"])

    def test_packed_expert_risk_report_counts_lengths_without_materializing_payload(self):
        class LengthOnly:
            def __init__(self, n: int):
                self.n = n

            def __len__(self) -> int:
                return self.n

            def __bytes__(self) -> bytes:
                raise AssertionError("payload bytes must not be materialized for risk accounting")

        report = dq.describe_packed_expert_risk("fp4", payload=LengthOnly(2), scales=LengthOnly(1), shape=(4,))  # type: ignore[arg-type]
        self.assertEqual(report["payload_bytes"], 2)
        self.assertEqual(report["scale_bytes"], 1)

    def test_packed_expert_risk_report_is_blocked_without_reference_metadata(self):
        fp4_report = dq.describe_packed_expert_risk("fp4", payload=b"\x12", scales=b"\x7f", shape=(2,))
        self.assertEqual(fp4_report["status"], "blocked")
        self.assertEqual(fp4_report["packing"], "fp4")
        self.assertIn("trusted_reference", fp4_report["missing"])
        self.assertIn("fp4_nibble_order", fp4_report["missing"])
        self.assertIn("axis_layout", fp4_report["missing"])
        self.assertEqual(fp4_report["payload_size"], {
            "expected_bytes": 1,
            "actual_bytes": 1,
            "plausible": True,
            "delta_bytes": 0,
        })
        self.assertEqual(fp4_report["scale_relationship"]["status"], "unknown")
        self.assertEqual(fp4_report["scale_relationship"]["issues"], ["block_size_missing", "scale_axis_missing", "scale_shape_missing"])
        self.assertIn("does not decode payload bytes", fp4_report["not_covered"])

        i8_report = dq.describe_packed_expert_risk("i8", payload=b"\x00\xff", scales=b"\x7f", shape=(2,))
        self.assertEqual(i8_report["status"], "blocked")
        self.assertEqual(i8_report["packing"], "i8")
        self.assertIn("i8_signedness_or_zero_point", i8_report["missing"])
        self.assertIn("block_size", i8_report["missing"])
        self.assertEqual(i8_report["payload_size"]["expected_bytes"], 2)
        self.assertTrue(i8_report["payload_size"]["plausible"])

    def test_packed_expert_risk_report_accounts_for_block_axis_and_scale_relationship(self):
        report = dq.describe_packed_expert_risk(
            "fp4",
            payload=bytes(8),
            scales=bytes(4),
            shape=(2, 8),
            block_size=4,
            scale_axis=-1,
            scale_shape=(2, 2),
        )
        self.assertEqual(report["payload_size"], {
            "expected_bytes": 8,
            "actual_bytes": 8,
            "plausible": True,
            "delta_bytes": 0,
        })
        self.assertEqual(report["block"], {
            "block_size": 4,
            "axis": 1,
            "axis_dim": 8,
            "blocks_per_axis": 2,
            "expected_scale_shape": [2, 2],
        })
        self.assertEqual(report["scale_relationship"], {
            "status": "plausible",
            "scale_shape": [2, 2],
            "expected_scale_shape": [2, 2],
            "actual_scale_count": 4,
            "expected_scale_count": 4,
            "scale_bytes": 4,
            "issues": [],
        })
        self.assertNotIn("block_size", report["missing"])
        self.assertNotIn("scale_axis", report["missing"])
        self.assertIn("trusted_reference", report["missing"])

    def test_classify_expert_metadata_reports_ambiguous_block_layout(self):
        header = {
            "layers.0.ffn.experts.0.w1.weight": {"dtype": "I8", "shape": [2048, 4096]},
            "layers.0.ffn.experts.0.w1.scale": {"dtype": "BF16", "shape": [16, 32]},
        }
        report = dq.classify_expert_metadata_from_header(header)
        self.assertEqual(report["routed_experts"]["inferred_block_layout"]["status"], "mixed_or_ambiguous")
        self.assertIn("scale block layout", report["unknown_required_for_decode"])

    def test_classify_expert_metadata_reports_mixed_inferred_and_ambiguous_as_mixed(self):
        header = {
            "layers.0.ffn.experts.0.w1.weight": {"dtype": "I8", "shape": [2048, 2048]},
            "layers.0.ffn.experts.0.w1.scale": {"dtype": "BF16", "shape": [2048, 128]},
            "layers.0.ffn.experts.0.w2.weight": {"dtype": "I8", "shape": [2048, 4096]},
            "layers.0.ffn.experts.0.w2.scale": {"dtype": "BF16", "shape": [16, 32]},
        }
        report = dq.classify_expert_metadata_from_header(header)
        self.assertEqual(report["routed_experts"]["inferred_block_layout"]["status"], "mixed_or_ambiguous")
        self.assertIn("scale block layout", report["unknown_required_for_decode"])

    def test_packed_expert_risk_report_flags_implausible_payload_and_scales(self):
        report = dq.describe_packed_expert_risk(
            "i8",
            payload=bytes(3),
            scales=bytes(1),
            shape=(2, 8),
            block_size=4,
            scale_axis=1,
            scale_shape=(2, 1),
        )
        self.assertEqual(report["payload_size"]["expected_bytes"], 16)
        self.assertFalse(report["payload_size"]["plausible"])
        self.assertEqual(report["payload_size"]["delta_bytes"], -13)
        self.assertEqual(report["scale_relationship"]["status"], "mismatch")
        self.assertIn("scale_shape_mismatch", report["scale_relationship"]["issues"])
        self.assertIn("payload_size_mismatch", report["missing"])

    def test_packed_expert_risk_report_rejects_invalid_block_metadata(self):
        with self.assertRaisesRegex(ValueError, "block_size"):
            dq.describe_packed_expert_risk("fp4", payload=b"\x00", scales=b"\x00", shape=(2,), block_size=0, scale_axis=0, scale_shape=(1,))
        with self.assertRaisesRegex(ValueError, "scale_axis"):
            dq.describe_packed_expert_risk("fp4", payload=b"\x00", scales=b"\x00", shape=(2,), block_size=2, scale_axis=2, scale_shape=(1,))

    def test_fp4_expert_header_classification_reports_packing_scales_block_axis(self):
        report = dq.classify_expert_metadata_from_header(fp4_candidate_header_fixture())

        self.assertEqual(report["schema"], 2)
        packed = report["packed_expert_decode"]
        self.assertEqual(packed["routed_packing"], "fp4")
        self.assertIsNone(packed["shared_packing"])
        self.assertEqual(packed["packing_kinds_present"], ["fp4"])
        self.assertEqual(packed["routed_weight_dtype"], "U8")
        self.assertEqual(packed["routed_scale_dtype"], "F8_E8M0")
        self.assertEqual(report["routed_experts"]["weight_dtype_counts"], {"U8": 6})
        self.assertEqual(report["routed_experts"]["scale_dtype_counts"], {"F8_E8M0": 6})
        self.assertEqual(report["routed_experts"]["pairing"]["paired_count"], 6)
        self.assertEqual(report["shared_experts"]["weight_dtype_counts"], {"BF16": 1})
        self.assertFalse(packed["block_layout_verified"])
        layout = report["routed_experts"]["inferred_block_layout"]
        self.assertEqual(layout["status"], "inferred")
        self.assertEqual(layout["axis"], 1)
        self.assertEqual(layout["block_size"], 16)

    def test_fp4_classification_lists_unknowns_and_blocks_decode(self):
        report = dq.classify_expert_metadata_from_header(fp4_candidate_header_fixture())

        self.assertFalse(report["can_decode_payload"])
        packed = report["packed_expert_decode"]
        for missing in (
            "trusted_reference",
            "fp4_encoding",
            "fp4_nibble_order",
            "scale_dtype",
            "scale_axis",
            "block_size",
            "axis_layout",
        ):
            self.assertIn(missing, packed["unknown_facts"])
            self.assertIn(missing, report["unknown_required_for_decode"])

    def test_fp4_classifier_does_not_read_payload_bytes(self):
        header = {
            "layers.0.ffn.experts.0.w1.weight": StrictHeaderMeta("U8", [2048, 2048]),
            "layers.0.ffn.experts.0.w1.scale": StrictHeaderMeta("F8_E8M0", [2048, 128]),
        }

        report = dq.classify_expert_metadata_from_header(header)  # type: ignore[arg-type]
        self.assertEqual(report["packed_expert_decode"]["routed_packing"], "fp4")
        self.assertFalse(report["can_decode_payload"])

    def test_fp4_classification_is_deterministic(self):
        first = dq.classify_expert_metadata_from_header(fp4_candidate_header_fixture())
        second = dq.classify_expert_metadata_from_header(fp4_candidate_header_fixture())

        self.assertEqual(first, second)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

    def test_unknown_expert_weight_dtype_reports_supported_kind_unknown(self):
        header = {
            "layers.0.ffn.experts.0.w1.weight": {"dtype": "Q5", "shape": [2048, 2048]},
            "layers.0.ffn.experts.0.w1.scale": {"dtype": "BF16", "shape": [2048, 128]},
        }

        report = dq.classify_expert_metadata_from_header(header)
        self.assertIsNone(report["packed_expert_decode"]["routed_packing"])
        self.assertEqual(report["packed_expert_decode"]["unrecognized_weight_dtypes"], ["Q5"])
        self.assertIn("supported_packing_kind", report["unknown_required_for_decode"])
        self.assertIn("trusted_reference", report["unknown_required_for_decode"])

    def test_checkpoint_packing_classifier_reports_real_i8_and_f8_families(self):
        report = dq.classify_checkpoint_expert_packing(real_packing_header_fixture(), declared_quant=DECLARED_FP8_QUANT)

        self.assertEqual(report["schema"], 2)
        self.assertEqual(report["status"], "classified")
        self.assertEqual(report["observed_packing"]["routed"]["weight_dtypes"], {"I8": 3})
        self.assertEqual(report["observed_packing"]["routed"]["scale_dtypes"], {"F8_E8M0": 3})
        self.assertEqual(report["observed_packing"]["shared"]["weight_dtypes"], {"F8_E4M3": 1})
        self.assertEqual(report["observed_packing"]["shared"]["scale_dtypes"], {"F8_E8M0": 1})
        routed_layout = report["observed_packing"]["routed"]["inferred_block_layout"]
        self.assertEqual(routed_layout["status"], "inferred")
        self.assertEqual(routed_layout["axis"], 1)
        self.assertEqual(routed_layout["block_size"], 16)
        self.assertEqual(report["base"]["routed_experts"]["pairing"]["paired_count"], 3)

    def test_checkpoint_packing_classifier_verifies_fp4_absent(self):
        report = dq.classify_checkpoint_expert_packing(real_packing_header_fixture(), declared_quant=DECLARED_FP8_QUANT)

        self.assertFalse(report["fp4_present"])
        self.assertEqual(report["fp4_like_dtypes"], [])
        self.assertNotIn("fp4 decode (no trusted reference)", report["unknown_required_for_decode"])
        self.assertFalse(report["can_decode_payload"])

    def test_checkpoint_packing_classifier_clears_clean_shared_2d_tiling(self):
        report = dq.classify_checkpoint_expert_packing(real_packing_header_fixture(), declared_quant=DECLARED_FP8_QUANT)

        self.assertFalse(report["can_decode_payload"])
        unknowns = report["unknown_required_for_decode"]
        self.assertNotIn("F8_E8M0 (UE8M0) scale decode + application order for I8 routed weights", unknowns)
        self.assertNotIn("F8_E8M0 (UE8M0) scale application order for F8_E4M3 shared weights", unknowns)
        self.assertIn("trusted reference (DS4-CPU / Transformers) end-to-end expert decode parity", unknowns)
        self.assertIn("routed block layout differs declared weight_block_size; needs trusted reference", unknowns)
        self.assertNotIn("shared 128x128 2-D block axis assignment (1-D inference ambiguous)", unknowns)
        self.assertIn("does not read safetensors payload bytes", report["not_covered"])

    def test_checkpoint_packing_classifier_keeps_shared_unknowns_when_2d_tiling_not_observed(self):
        report = dq.classify_checkpoint_expert_packing(genuinely_ambiguous_shared_header_fixture(), declared_quant=DECLARED_FP8_QUANT)

        self.assertFalse(report["can_decode_payload"])
        unknowns = report["unknown_required_for_decode"]
        self.assertIn("F8_E8M0 (UE8M0) scale application order for F8_E4M3 shared weights", unknowns)
        self.assertIn("shared 128x128 2-D block axis assignment (1-D inference ambiguous)", unknowns)
        self.assertIn("routed block layout differs declared weight_block_size; needs trusted reference", unknowns)
        self.assertIn("trusted reference (DS4-CPU / Transformers) end-to-end expert decode parity", unknowns)

    def test_checkpoint_packing_classifier_flags_declared_quant_discrepancy(self):
        report = dq.classify_checkpoint_expert_packing(real_packing_header_fixture(), declared_quant=DECLARED_FP8_QUANT)

        self.assertEqual(report["declared_quant"]["weight_block_size"], [128, 128])
        routed = report["config_consistency"]["routed"]
        shared = report["config_consistency"]["shared"]
        self.assertEqual(routed["status"], "discrepancy")
        self.assertEqual(routed["observed"], {"axis": 1, "block_size": 16})
        self.assertEqual(routed["declared"], [128, 128])
        self.assertEqual(shared["status"], "consistent_but_ambiguous")
        self.assertIn("128x128", shared["reason"])

    def test_reconcile_routed_block_layout_is_shape_authoritative(self):
        header = {
            "layers.0.ffn.experts.0.w1.weight": {"dtype": "I8", "shape": [8, 32]},
            "layers.0.ffn.experts.0.w1.scale": {"dtype": "F8_E8M0", "shape": [8, 2]},
        }
        report = dq.classify_checkpoint_expert_packing(header, declared_quant=DECLARED_FP8_QUANT)

        first = dq.reconcile_routed_block_layout(report)
        second = dq.reconcile_routed_block_layout(report)

        self.assertEqual(first["status"], "shape_authoritative")
        self.assertEqual(first["geometry"], {"axis": 1, "block_size": 16})
        self.assertEqual(first["observed_scale_ratio"], {"axis": 1, "weight_dim": 32, "scale_dim": 2, "block_size": 16})
        self.assertEqual(first["declared_weight_block_size"], [128, 128])
        self.assertIn("advisory", first["declared_role"])
        self.assertIn("shape", first["discrepancy_explained"])
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

    def test_reconcile_routed_block_layout_reports_unresolved_for_ambiguous_shapes(self):
        header = {
            "layers.0.ffn.experts.0.w1.weight": {"dtype": "I8", "shape": [2048, 4096]},
            "layers.0.ffn.experts.0.w1.scale": {"dtype": "F8_E8M0", "shape": [16, 32]},
        }
        report = dq.classify_checkpoint_expert_packing(header, declared_quant=DECLARED_FP8_QUANT)
        reconciled = dq.reconcile_routed_block_layout(report)

        self.assertEqual(reconciled["status"], "unresolved")
        self.assertIn("ambiguous", reconciled["reason"])
        self.assertNotIn("geometry", reconciled)

    def test_checkpoint_packing_classifier_does_not_read_payload_bytes(self):
        class ExplodingOffsets:
            def __iter__(self):
                raise AssertionError("payload offsets must not be iterated")

            def __len__(self):
                raise AssertionError("payload offsets must not be sized")

        header = real_packing_header_fixture()
        header["layers.0.ffn.experts.0.w1.weight"] = {
            "dtype": "I8",
            "shape": [2048, 2048],
            "data_offsets": ExplodingOffsets(),
        }
        report = dq.classify_checkpoint_expert_packing(header, declared_quant=DECLARED_FP8_QUANT)
        self.assertEqual(report["observed_packing"]["routed"]["weight_dtypes"], {"I8": 3})
        self.assertFalse(report["can_decode_payload"])

    def test_checkpoint_packing_classifier_is_deterministic(self):
        first = dq.classify_checkpoint_expert_packing(real_packing_header_fixture(), declared_quant=DECLARED_FP8_QUANT)
        second = dq.classify_checkpoint_expert_packing(real_packing_header_fixture(), declared_quant=DECLARED_FP8_QUANT)

        self.assertEqual(first, second)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

    def test_real_checkpoint_header_classification_or_skip(self):
        model_dir = default_real_checkpoint_dir()
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            self.skipTest(f"real DeepSeek V4 Flash checkpoint not present at {model_dir}")
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        expert_names = sorted(name for name in weight_map if ".ffn.experts." in name or ".ffn.shared_experts." in name)
        if not expert_names:
            self.skipTest("real checkpoint index contains no expert tensors")
        shard = model_dir / weight_map[expert_names[0]]
        header = dq.read_safetensors_header(shard, max_bytes=128 * 1024 * 1024)
        report = dq.classify_expert_metadata_from_header(header)
        self.assertEqual(report["status"], "classified")
        self.assertEqual(report["schema"], 2)
        self.assertIn("packed_expert_decode", report)
        self.assertFalse(report["can_decode_payload"])

    def test_forward_parity_marker_absent_after_slice(self):
        marker = Path("/Volumes/Data NVME/mlx-ft/ds4/.deepseek-v4-forward-parity-ok")
        self.assertFalse(marker.exists(), f"{marker} must not be written by Story 11.15a/11.15b/11.15c")
        self._assert_model_4bit_absent_or_valid_current_artifact()
        self.assertNotIn(".deepseek-v4-forward-parity-ok", (REPO_ROOT / "python-envs" / "mlx" / "src" / "ds4_ft_mlx" / "deepseek_v4_dequant.py").read_text())

    def test_classify_expert_metadata_from_header_detects_i8_routed_and_bf16_shared(self):
        header = {
            "layers.0.ffn.experts.0.w1.weight": {"dtype": "I8", "shape": [2048, 2048]},
            "layers.0.ffn.experts.0.w1.scale": {"dtype": "BF16", "shape": [2048, 128]},
            "layers.0.ffn.experts.0.w2.weight": {"dtype": "I8", "shape": [4096, 1024]},
            "layers.0.ffn.experts.0.w2.scale": {"dtype": "BF16", "shape": [4096, 64]},
            "layers.0.ffn.experts.1.w1.weight": {"dtype": "I8", "shape": [2048, 2048]},
            "layers.0.ffn.experts.1.w1.scale": {"dtype": "BF16", "shape": [2048, 128]},
            "layers.0.ffn.experts.1.w2.weight": {"dtype": "I8", "shape": [4096, 1024]},
            "layers.0.ffn.experts.1.w2.scale": {"dtype": "BF16", "shape": [4096, 64]},
            "layers.0.ffn.shared_experts.w1.weight": {"dtype": "BF16", "shape": [2048, 2048]},
            "layers.0.ffn.shared_experts.w1.scale": {"dtype": "BF16", "shape": [2048, 128]},
            "layers.0.ffn.gate.weight": {"dtype": "BF16", "shape": [256, 2048]},
        }
        report = dq.classify_expert_metadata_from_header(header)
        self.assertEqual(report["status"], "classified")
        self.assertEqual(report["routed_experts"]["weight_dtype_counts"], {"I8": 4})
        self.assertEqual(report["routed_experts"]["scale_dtype_counts"], {"BF16": 4})
        self.assertEqual(report["routed_experts"]["pairing"]["paired_count"], 4)
        self.assertEqual(report["shared_experts"]["weight_dtype_counts"], {"BF16": 1})
        layout = report["routed_experts"]["inferred_block_layout"]
        self.assertEqual(layout["status"], "inferred")
        self.assertEqual(layout["block_size"], 16)
        self.assertEqual(layout["axis"], 1)
        self.assertFalse(report["can_decode_payload"])
        self.assertIn("does not decode I8/FP4 values", report["not_covered"])

    def test_read_safetensors_header_reads_only_json_prefix(self):
        import json
        import os
        import struct
        import tempfile

        data = struct.pack("<f", 1.0)
        header = {"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
        hbytes = json.dumps(header).encode()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(struct.pack("<Q", len(hbytes)) + hbytes + data)
            path = f.name
        try:
            got = dq.read_safetensors_header(path)
            self.assertEqual(got["x"]["dtype"], "F32")
            self.assertEqual(got["x"]["shape"], [1])
        finally:
            os.unlink(path)

    def test_unsupported_expert_packing_fails_closed(self):
        for kind in ("i8", "I8"):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(NotImplementedError, "unsupported.*expert"):
                    dq.dequantize_expert_packed(kind, b"\x00", scales=b"\x7f", shape=(1,))
        for kind in ("fp4", "FP4"):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, "requires non-None scales"):
                    dq.dequantize_expert_packed(kind, b"\x00" * 16, scales=None, shape=(1, 32))

    def test_dequantize_expert_packed_i8_bf16_dispatch_matches_primitive(self):
        shape = (2, 16)
        block_size = 16
        scale_axis = 1
        payload = bytes((i * 13 + 5) % 256 for i in range(32))
        scales = bf16_bytes(0.5, 1.5)

        wrapped = dq.dequantize_expert_packed(
            "i8",
            payload,
            scales=scales,
            shape=shape,
            block_size=block_size,
            scale_axis=scale_axis,
        )
        direct = dq.dequantize_i8_block_scale(
            payload,
            scales,
            shape=shape,
            block_size=block_size,
            scale_axis=scale_axis,
        )
        self.assertEqual(wrapped, direct)

    def test_dequantize_expert_packed_i8_e8m0_dispatch_matches_primitive(self):
        shape = (2, 16)
        block_size = 16
        scale_axis = 1
        payload = bytes((i * 13 + 5) % 256 for i in range(32))
        scales = bytes([130, 120])

        wrapped = dq.dequantize_expert_packed(
            "i8",
            payload,
            scales=scales,
            shape=shape,
            block_size=block_size,
            scale_axis=scale_axis,
        )
        direct = dq.dequantize_i8_e8m0_block_scale(
            payload,
            scales,
            shape=shape,
            block_size=block_size,
            scale_axis=scale_axis,
        )
        self.assertEqual(wrapped, direct)

    def test_dequantize_expert_packed_i8_e8m0_byte_ratio_dispatches(self):
        payload = bytes(range(32))
        scales = bytes([127, 128])
        shape = (2, 16)
        block_size = 16
        scale_axis = 1

        wrapped = dq.dequantize_expert_packed(
            "i8",
            payload,
            scales=scales,
            shape=shape,
            block_size=block_size,
            scale_axis=scale_axis,
        )
        direct = dq.dequantize_i8_e8m0_block_scale(
            payload,
            scales,
            shape=shape,
            block_size=block_size,
            scale_axis=scale_axis,
        )
        self.assertEqual(wrapped, direct)

    def test_dequantize_expert_packed_i8_ambiguous_scale_bytes_raises(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            dq.dequantize_expert_packed(
                "i8",
                bytes(range(32)),
                scales=b"\x01\x02\x03",
                shape=(2, 16),
                block_size=16,
                scale_axis=1,
            )

    def test_dequantize_expert_packed_i8_metadata_missing_raises(self):
        with self.assertRaisesRegex(NotImplementedError, r"unsupported.*expert.*block_size"):
            dq.dequantize_expert_packed("i8", b"\x00" * 32, scales=b"\x00" * 4, shape=(2, 16))

    def test_dequantize_expert_packed_fp4_rejects_non_ocp_block_size(self):
        with self.assertRaisesRegex(ValueError, "block_size must be 32"):
            dq.dequantize_expert_packed(
                "fp4",
                b"\x00" * 16,
                scales=b"\x00" * 4,
                shape=(2, 16),
                block_size=16,
                scale_axis=1,
            )

    def test_i8_e8m0_block_scale_matches_independent_closed_form_sweep(self):
        def ref_signed_i8(b: int) -> int:
            return b - 256 if b >= 128 else b

        def ref_e8m0_scale(s: int) -> float:
            if s == 0:
                return 2.0 ** -127
            if s == 255:
                return float("nan")
            return 2.0 ** (s - 127)

        def product(shape: tuple[int, ...]) -> int:
            out = 1
            for dim in shape:
                out *= dim
            return out

        def row_major_coords(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
            coords = [0] * len(shape)
            remaining = index
            for axis in range(len(shape) - 1, -1, -1):
                coords[axis] = remaining % shape[axis]
                remaining //= shape[axis]
            return tuple(coords)

        def row_major_index(coords: tuple[int, ...], shape: tuple[int, ...]) -> int:
            index = 0
            for coord, dim in zip(coords, shape, strict=True):
                index = index * dim + coord
            return index

        cases = [
            ((1, 16), 8, 1, 1101),
            ((2, 16), 16, 1, 2202),
            ((4, 8), 2, 0, 3303),
            ((4, 8), 4, 0, 4404),
        ]
        for shape, block_size, scale_axis, seed in cases:
            with self.subTest(shape=shape, block_size=block_size, scale_axis=scale_axis):
                rng = random.Random(seed)
                payload = bytes(rng.randrange(256) for _ in range(product(shape)))
                axis = scale_axis if scale_axis >= 0 else len(shape) + scale_axis
                scale_shape = list(shape)
                scale_shape[axis] //= block_size
                scale_count = product(tuple(scale_shape))
                scale_bytes = [rng.randrange(255) for _ in range(scale_count)]
                scale_bytes[0] = 0
                scale_bytes[-1] = 255
                if scale_count > 2:
                    scale_bytes[scale_count // 2] = 127
                scales = bytes(scale_bytes)

                actual = dq.dequantize_i8_e8m0_block_scale(
                    payload,
                    scales,
                    shape=shape,
                    block_size=block_size,
                    scale_axis=scale_axis,
                )
                for value_index, payload_byte in enumerate(payload):
                    coords = row_major_coords(value_index, shape)
                    block_coords = list(coords)
                    block_coords[axis] = coords[axis] // block_size
                    scale_index = row_major_index(tuple(block_coords), tuple(scale_shape))
                    expected = ref_signed_i8(payload_byte) * ref_e8m0_scale(scales[scale_index])
                    got = actual[value_index]
                    if math.isnan(expected):
                        self.assertTrue(math.isnan(got), (value_index, scale_index, got))
                    else:
                        self.assertEqual(got, expected, (value_index, scale_index))

    def test_i8_e8m0_block_scale_matches_bf16_shim_pipeline(self):
        shape = (2, 16)
        block_size = 16
        scale_axis = 1
        payload = bytes((i * 7 + 3) % 256 for i in range(32))
        for scales in (bytes([0, 1]), bytes([64, 127]), bytes([128, 200]), bytes([254, 127])):
            with self.subTest(scales=list(scales)):
                e8m0 = dq.dequantize_i8_e8m0_block_scale(
                    payload,
                    scales,
                    shape=shape,
                    block_size=block_size,
                    scale_axis=scale_axis,
                )
                bf16 = dq.dequantize_i8_block_scale(
                    payload,
                    shim.f8_e8m0_to_bf16(scales),
                    shape=shape,
                    block_size=block_size,
                    scale_axis=scale_axis,
                )
                self.assertEqual(e8m0, bf16)

        nan_scales = bytes([127, 255])
        e8m0 = dq.dequantize_i8_e8m0_block_scale(
            payload,
            nan_scales,
            shape=shape,
            block_size=block_size,
            scale_axis=scale_axis,
        )
        bf16 = dq.dequantize_i8_block_scale(
            payload,
            shim.f8_e8m0_to_bf16(nan_scales),
            shape=shape,
            block_size=block_size,
            scale_axis=scale_axis,
        )
        for got, expected in zip(e8m0, bf16, strict=True):
            if math.isnan(expected):
                self.assertTrue(math.isnan(got))
            else:
                self.assertEqual(got, expected)

    def test_i8_e8m0_block_scale_matches_closed_form_reference(self):
        # E8M0 byte 127 is 1.0 and 128 is 2.0. With block_size=2 along
        # axis 1, the first two signed-I8 values use scale 1.0 and the next
        # two use scale 2.0.
        actual = dq.dequantize_i8_e8m0_block_scale(
            bytes([1, 2, 3, 4]),
            bytes([127, 128]),
            shape=(1, 4),
            block_size=2,
            scale_axis=1,
        )
        self.assertEqual(actual, [1.0, 2.0, 6.0, 8.0])

    def test_i8_e8m0_block_scale_propagates_e8m0_special_values(self):
        actual = dq.dequantize_i8_e8m0_block_scale(
            bytes([1, 2]),
            bytes([0, 255]),
            shape=(1, 2),
            block_size=1,
            scale_axis=1,
        )
        self.assertEqual(actual[0], 2.0 ** -127)
        self.assertTrue(math.isnan(actual[1]))

    def test_i8_e8m0_block_scale_rejects_partial_blocks_and_bad_lengths(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            dq.dequantize_i8_e8m0_block_scale(bytes([1, 2, 3]), bytes([127, 128]), shape=(1, 3), block_size=2, scale_axis=1)
        with self.assertRaisesRegex(ValueError, "scales byte length"):
            dq.dequantize_i8_e8m0_block_scale(bytes([1, 2, 3, 4]), bytes([127]), shape=(1, 4), block_size=2, scale_axis=1)
        with self.assertRaisesRegex(ValueError, "payload byte length"):
            dq.dequantize_i8_e8m0_block_scale(bytes([1, 2, 3]), bytes([127, 128]), shape=(1, 4), block_size=2, scale_axis=1)

    def test_i8_bf16_block_scale_byte_identical_after_refactor(self):
        actual = dq.dequantize_i8_block_scale(
            bytes([1, 2, 3, 4]),
            bf16_bytes(0.5, 1.5),
            shape=(1, 4),
            block_size=2,
            scale_axis=1,
        )
        self.assertEqual(actual, [0.5, 1.0, 4.5, 6.0])

    def test_shared_f8_e4m3_e8m0_block_decode_via_existing_primitive(self):
        # This proves shared expert F8_E4M3 value decode + E8M0 scale
        # application order for an explicit broadcast layout. The sibling
        # 2-D sub-block primitive below covers the real 128x128 shape ratio.
        actual = dq.dequantize_f8_e4m3fn_with_e8m0_scales(
            bytes([0x38, 0x40, 0xB8, 0x7F]),
            bytes([127, 128]),
            value_shape=(2, 2),
            scale_shape=(2, 1),
        )
        self.assertEqual(actual[:3], [1.0, 2.0, -2.0])
        self.assertTrue(math.isnan(actual[3]))

    def _ref_f8_e4m3fn_to_float(self, byte: int) -> float:
        b = int(byte) & 0xFF
        if b in (0x7F, 0xFF):
            return float("nan")
        sign = -1.0 if (b & 0x80) else 1.0
        exp = (b >> 3) & 0x0F
        mant = b & 0x07
        if exp == 0:
            if mant == 0:
                return -0.0 if sign < 0 else 0.0
            return sign * (mant / 8.0) * (2.0 ** -6)
        return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))

    def _ref_e8m0_to_float(self, byte: int) -> float:
        b = int(byte) & 0xFF
        if b == 0:
            return 2.0 ** -127
        if b == 0xFF:
            return float("nan")
        return 2.0 ** (b - 127)

    def _shared_2d_sample_indices(self, value_shape: tuple[int, int], scale_shape: tuple[int, int]) -> list[int]:
        rows, cols = value_shape
        scale_rows, scale_cols = scale_shape
        block_rows = rows // scale_rows
        block_cols = cols // scale_cols
        seen: set[int] = set()
        out: list[int] = []
        for scale_row in range(scale_rows):
            for scale_col in range(scale_cols):
                row0 = scale_row * block_rows
                col0 = scale_col * block_cols
                candidates = [
                    (row0, col0),
                    (row0 + block_rows // 2, col0 + block_cols // 2),
                    (row0 + block_rows - 1, col0 + block_cols - 1),
                    (row0 + (scale_row * 17 + scale_col * 5) % block_rows, col0 + (scale_row * 7 + scale_col * 19) % block_cols),
                ]
                for row, col in candidates:
                    idx = row * cols + col
                    if idx not in seen:
                        seen.add(idx)
                        out.append(idx)
        return out

    def _assert_shared_2d_samples_match_closed_form(
        self,
        actual: list[float],
        values: bytes | bytearray,
        scales: bytes | bytearray,
        *,
        value_shape: tuple[int, int],
        scale_shape: tuple[int, int],
        sample_indices: list[int],
    ) -> None:
        rows, cols = value_shape
        scale_rows, scale_cols = scale_shape
        block_rows = rows // scale_rows
        block_cols = cols // scale_cols
        for idx in sample_indices:
            row, col = divmod(idx, cols)
            scale_idx = (row // block_rows) * scale_cols + (col // block_cols)
            value = self._ref_f8_e4m3fn_to_float(values[idx])
            scale = self._ref_e8m0_to_float(scales[scale_idx])
            expected = float("nan") if math.isnan(value) or math.isnan(scale) else value * scale
            got = actual[idx]
            if math.isnan(expected):
                self.assertTrue(math.isnan(got), (idx, scale_idx, got))
            else:
                self.assertEqual(got, expected, (idx, scale_idx))

    def test_shared_2d_block_decode_matches_closed_form_real_geometry(self):
        value_shape = (2048, 4096)
        scale_shape = (16, 32)
        rng = random.Random(20240619)
        values = bytearray(rng.randbytes(value_shape[0] * value_shape[1]))
        scales = bytearray(rng.randbytes(scale_shape[0] * scale_shape[1]))
        scales[0] = 0x00
        scales[1] = 0x7F
        scales[-1] = 0xFF
        sample_indices = self._shared_2d_sample_indices(value_shape, scale_shape)
        for n, idx in enumerate(sample_indices):
            if n % 257 == 0:
                values[idx] = 0x7F
            elif n % 263 == 0:
                values[idx] = 0xFF
            elif n % 269 == 0:
                values[idx] = 0x00

        actual = dq.dequantize_f8_e4m3_e8m0_2d_block_scale(
            values,
            scales,
            value_shape=value_shape,
            scale_shape=scale_shape,
        )

        self._assert_shared_2d_samples_match_closed_form(
            actual,
            values,
            scales,
            value_shape=value_shape,
            scale_shape=scale_shape,
            sample_indices=sample_indices,
        )

    def test_shared_2d_block_decode_nan_aware_sweep_across_geometries(self):
        cases = [
            ((4, 4), (2, 2), 1144),
            ((8, 16), (2, 2), 2816),
            ((256, 256), (2, 2), 2256),
            ((2048, 4096), (16, 32), 2416),
        ]
        for value_shape, scale_shape, seed in cases:
            with self.subTest(value_shape=value_shape, scale_shape=scale_shape):
                rng = random.Random(seed)
                values = bytearray(rng.randbytes(value_shape[0] * value_shape[1]))
                scales = bytearray(rng.randbytes(scale_shape[0] * scale_shape[1]))
                scales[0] = 0x00
                if len(scales) > 1:
                    scales[1] = 0x7F
                scales[-1] = 0xFF
                sample_indices = self._shared_2d_sample_indices(value_shape, scale_shape)
                for n, idx in enumerate(sample_indices):
                    if n % 11 == 0:
                        values[idx] = 0x7F
                    elif n % 13 == 0:
                        values[idx] = 0xFF
                    elif n % 17 == 0:
                        values[idx] = 0x00

                actual = dq.dequantize_f8_e4m3_e8m0_2d_block_scale(
                    values,
                    scales,
                    value_shape=value_shape,
                    scale_shape=scale_shape,
                )

                self._assert_shared_2d_samples_match_closed_form(
                    actual,
                    values,
                    scales,
                    value_shape=value_shape,
                    scale_shape=scale_shape,
                    sample_indices=sample_indices,
                )

    def test_shared_2d_block_decode_matches_closed_form(self):
        # value_shape=(4,4), scale_shape=(2,2) gives 2x2 sub-blocks.
        # Scale bytes are [1.0, 2.0, 0.5, 4.0] in row-major scale order.
        values = bytes([
            0x38, 0x40, 0x48, 0x50,  # 1, 2, 4, 8
            0xB8, 0xC0, 0x7F, 0x00,  # -1, -2, NaN, 0
            0x38, 0x40, 0x48, 0x50,  # 1, 2, 4, 8
            0xB8, 0xC0, 0x48, 0x50,  # -1, -2, 4, 8
        ])
        actual = dq.dequantize_f8_e4m3_e8m0_2d_block_scale(
            values,
            bytes([127, 128, 126, 129]),
            value_shape=(4, 4),
            scale_shape=(2, 2),
        )
        self.assertFloatsEqualWithNan(
            actual,
            [
                1.0, 2.0, 8.0, 16.0,
                -1.0, -2.0, float("nan"), 0.0,
                0.5, 1.0, 16.0, 32.0,
                -0.5, -1.0, 16.0, 32.0,
            ],
        )

    def test_shared_2d_block_decode_rejects_bad_geometry(self):
        with self.assertRaisesRegex(ValueError, "rank"):
            dq.dequantize_f8_e4m3_e8m0_2d_block_scale(bytes([0x38] * 4), bytes([127, 128]), value_shape=(2, 2), scale_shape=(2,))
        with self.assertRaisesRegex(ValueError, "axis 1.*not divisible"):
            dq.dequantize_f8_e4m3_e8m0_2d_block_scale(bytes([0x38] * 6), bytes([127, 128]), value_shape=(2, 3), scale_shape=(2, 2))
        with self.assertRaisesRegex(ValueError, "scale_dim"):
            dq.dequantize_f8_e4m3_e8m0_2d_block_scale(bytes(), bytes(), value_shape=(2, 2), scale_shape=(2, 0))
        with self.assertRaisesRegex(ValueError, "values byte length"):
            dq.dequantize_f8_e4m3_e8m0_2d_block_scale(bytes([0x38] * 3), bytes([127]), value_shape=(2, 2), scale_shape=(1, 1))
        with self.assertRaisesRegex(ValueError, "scales byte length"):
            dq.dequantize_f8_e4m3_e8m0_2d_block_scale(bytes([0x38] * 4), bytes([127]), value_shape=(2, 2), scale_shape=(2, 2))

    def test_per_axis_broadcast_helper_unchanged_after_2d_primitive(self):
        actual = dq.dequantize_f8_e4m3fn_with_e8m0_scales(
            bytes([0x38, 0x40, 0xB8, 0x7F]),
            bytes([127, 128]),
            value_shape=(2, 2),
            scale_shape=(1, 2),
        )
        self.assertEqual(actual[:3], [1.0, 4.0, -1.0])
        self.assertTrue(math.isnan(actual[3]))

    def test_reconcile_and_2d_decode_are_deterministic(self):
        header = {
            "layers.0.ffn.experts.0.w1.weight": {"dtype": "I8", "shape": [8, 32]},
            "layers.0.ffn.experts.0.w1.scale": {"dtype": "F8_E8M0", "shape": [8, 2]},
        }
        report = dq.classify_checkpoint_expert_packing(header, declared_quant=DECLARED_FP8_QUANT)
        reconciled_a = dq.reconcile_routed_block_layout(report)
        reconciled_b = dq.reconcile_routed_block_layout(report)
        self.assertEqual(json.dumps(reconciled_a, sort_keys=True), json.dumps(reconciled_b, sort_keys=True))

        decoded_a = dq.dequantize_f8_e4m3_e8m0_2d_block_scale(bytes([0x38, 0x40, 0x48, 0x50]), bytes([127]), value_shape=(2, 2), scale_shape=(1, 1))
        decoded_b = dq.dequantize_f8_e4m3_e8m0_2d_block_scale(bytes([0x38, 0x40, 0x48, 0x50]), bytes([127]), value_shape=(2, 2), scale_shape=(1, 1))
        self.assertEqual(decoded_a, decoded_b)

    def test_resolve_routed_block_layout_from_consistent_classifier(self):
        header = {
            "layers.0.ffn.experts.0.w1.weight": {"dtype": "I8", "shape": [1, 32]},
            "layers.0.ffn.experts.0.w1.scale": {"dtype": "F8_E8M0", "shape": [1, 2]},
        }
        report = dq.classify_checkpoint_expert_packing(header, declared_quant={"weight_block_size": [16, 16]})
        layout = dq.resolve_routed_block_layout(report)
        self.assertEqual(layout, {"axis": 1, "block_size": 16})
        decoded = dq.dequantize_i8_e8m0_block_scale(
            bytes([1] * 32),
            bytes([127, 128]),
            shape=(1, 32),
            block_size=layout["block_size"],
            scale_axis=layout["axis"],
        )
        self.assertEqual(decoded, [1.0] * 16 + [2.0] * 16)

    def test_resolve_routed_block_layout_raises_on_ambiguous(self):
        header = {
            "layers.0.ffn.experts.0.w1.weight": {"dtype": "I8", "shape": [2048, 4096]},
            "layers.0.ffn.experts.0.w1.scale": {"dtype": "F8_E8M0", "shape": [16, 32]},
        }
        report = dq.classify_checkpoint_expert_packing(header, declared_quant={"weight_block_size": [128, 128]})
        with self.assertRaisesRegex(dq.ExpertBlockLayoutError, "ambiguous"):
            dq.resolve_routed_block_layout(report)

    def test_resolve_routed_block_layout_returns_on_shape_inferred_despite_declared_discrepancy(self):
        """11.15c supersedes the 11.15b discrepancy-raise contract by shape authority."""
        report = dq.classify_checkpoint_expert_packing(real_packing_header_fixture(), declared_quant=DECLARED_FP8_QUANT)
        self.assertEqual(report["config_consistency"]["routed"]["status"], "discrepancy")
        self.assertEqual(dq.resolve_routed_block_layout(report), {"axis": 1, "block_size": 16})

    def test_real_shard_routed_ratio_is_16_axis1_or_skip(self):
        model_dir = default_real_checkpoint_dir()
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            self.skipTest(f"real DeepSeek V4 Flash checkpoint not present at {model_dir}")
        with index_path.open("r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        weight_name = "layers.0.ffn.experts.0.w1.weight"
        scale_name = "layers.0.ffn.experts.0.w1.scale"
        if weight_name not in weight_map or scale_name not in weight_map:
            self.skipTest("real checkpoint index missing routed w1 weight/scale pair")
        header: dict[str, dict[str, object]] = {}
        for name in (weight_name, scale_name):
            shard_header = dq.read_safetensors_header(model_dir / weight_map[name], max_bytes=128 * 1024 * 1024)
            meta = shard_header[name]
            header[name] = {"dtype": meta["dtype"], "shape": meta["shape"]}
        report = dq.classify_checkpoint_expert_packing(header, declared_quant=DECLARED_FP8_QUANT)
        reconciled = dq.reconcile_routed_block_layout(report)
        self.assertEqual(reconciled["status"], "shape_authoritative")
        self.assertEqual(reconciled["geometry"], {"axis": 1, "block_size": 16})
        self.assertEqual(reconciled["observed_scale_ratio"], {"axis": 1, "weight_dim": 2048, "scale_dim": 128, "block_size": 16})

    def test_real_shard_shared_ratio_is_128x128_or_skip(self):
        model_dir = default_real_checkpoint_dir()
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            self.skipTest(f"real DeepSeek V4 Flash checkpoint not present at {model_dir}")
        with index_path.open("r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        weight_name = "layers.0.ffn.shared_experts.w1.weight"
        scale_name = "layers.0.ffn.shared_experts.w1.scale"
        if weight_name not in weight_map or scale_name not in weight_map:
            self.skipTest("real checkpoint index missing shared w1 weight/scale pair")
        shard_header = dq.read_safetensors_header(model_dir / weight_map[weight_name], max_bytes=128 * 1024 * 1024)
        weight_shape = tuple(shard_header[weight_name]["shape"])
        scale_shape = tuple(shard_header[scale_name]["shape"])
        self.assertEqual(weight_shape, (2048, 4096))
        self.assertEqual(scale_shape, (16, 32))
        self.assertEqual([w // s for w, s in zip(weight_shape, scale_shape, strict=True)], [128, 128])

    def test_i8_e8m0_decode_and_resolver_are_deterministic(self):
        first = dq.dequantize_i8_e8m0_block_scale(bytes([1, 2, 3, 4]), bytes([127, 128]), shape=(1, 4), block_size=2, scale_axis=1)
        second = dq.dequantize_i8_e8m0_block_scale(bytes([1, 2, 3, 4]), bytes([127, 128]), shape=(1, 4), block_size=2, scale_axis=1)
        self.assertEqual(first, second)
        report = dq.classify_checkpoint_expert_packing(
            {
                "layers.0.ffn.experts.0.w1.weight": {"dtype": "I8", "shape": [1, 32]},
                "layers.0.ffn.experts.0.w1.scale": {"dtype": "F8_E8M0", "shape": [1, 2]},
            },
            declared_quant={"weight_block_size": [16, 16]},
        )
        resolved_a = dq.resolve_routed_block_layout(report)
        resolved_b = dq.resolve_routed_block_layout(report)
        self.assertEqual(json.dumps(resolved_a, sort_keys=True), json.dumps(resolved_b, sort_keys=True))

    def test_i8_e8m0_block_scale_matches_pytorch_reference(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        shape = (2, 4)
        block_size = 2
        scale_axis = 1
        payload = bytes([1, 2, 3, 4, 253, 254, 255, 0])
        scales = bytes([127, 128, 126, 129])
        scale_shape = dq._expected_scale_shape(shape, block_size=block_size, scale_axis=scale_axis)
        weight = torch.frombuffer(bytearray(payload), dtype=torch.int8).reshape(shape)
        scale_values = torch.tensor(dq.decode_f8_e8m0_scales(scales), dtype=torch.float32).reshape(scale_shape)
        expected = (weight.float() * scale_values.repeat_interleave(block_size, dim=scale_axis)).flatten().tolist()
        actual = dq.dequantize_i8_e8m0_block_scale(payload, scales, shape=shape, block_size=block_size, scale_axis=scale_axis)
        self.assertEqual(actual, expected)

    def test_i8_block_scale_matches_pytorch_reference(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        shape = (3, 32)
        block_size = 16
        scale_axis = 1
        payload = bytes((i * 7 + 3) % 256 for i in range(dq._product(dq._as_shape(shape))))
        scale_shape = dq._expected_scale_shape(shape, block_size=block_size, scale_axis=scale_axis)
        scales_f32 = torch.linspace(0.1, 2.0, dq._product(dq._as_shape(scale_shape))).reshape(scale_shape).float()
        scales_bf16_bytes = scales_f32.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()

        weight = torch.frombuffer(bytearray(payload), dtype=torch.int8).reshape(shape)
        scales_decoded = torch.tensor(dq._decode_bf16(scales_bf16_bytes), dtype=torch.float32).reshape(scale_shape)
        scales_expanded = scales_decoded.repeat_interleave(block_size, dim=scale_axis)
        expected = (weight.float() * scales_expanded).flatten().tolist()

        actual = dq.dequantize_i8_block_scale(
            payload,
            scales_bf16_bytes,
            shape=shape,
            block_size=block_size,
            scale_axis=scale_axis,
        )
        self.assertEqual(len(actual), len(expected))
        max_abs_error = max(abs(a - b) for a, b in zip(actual, expected))
        self.assertLessEqual(max_abs_error, 1e-3)

    def test_i8_block_scale_fixture_reports_ok_or_skipped(self):
        report = dq.run_tiny_i8_block_scale_fixture()
        self.assertEqual(report["fixture"], "i8-block-scale-synthetic")
        if report["status"] == "skipped":
            self.assertIn("torch", report.get("reason", "").lower())
        else:
            self.assertEqual(report["status"], "ok")

    def test_i8_block_scale_rejects_non_divisible_partial_blocks(self):
        payload = bytes(range(18))
        scales = bytes(4)  # two BF16 scales
        with self.assertRaisesRegex(ValueError, "divisible"):
            dq.dequantize_i8_block_scale(
                payload,
                scales,
                shape=(1, 18),
                block_size=16,
                scale_axis=1,
            )


if __name__ == "__main__":
    unittest.main()
