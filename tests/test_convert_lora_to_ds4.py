#!/usr/bin/env python3
import json
import pathlib
import tempfile
import unittest

from scripts import convert_lora_to_ds4 as conv


def write_st(path: pathlib.Path, header: dict, data: bytes) -> None:
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + data)


def read_st(path: pathlib.Path):
    raw = path.read_bytes()
    header_len = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + header_len])
    data = raw[8 + header_len :]
    return header, data


class ConvertLoraToDs4Tests(unittest.TestCase):
    def test_success_maps_hf_peft_names_to_all_supported_ds4_targets(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            src = tmp / "adapter.safetensors"
            dst = tmp / "adapter-ds4.safetensors"
            data = bytes(range(48))
            header = {
                "__metadata__": {"format": "pt"},
                "base_model.model.lm_head.lora_A.weight": {"dtype": "F16", "shape": [2, 3], "data_offsets": [0, 12]},
                "base_model.model.lm_head.lora_B.weight": {"dtype": "F16", "shape": [4, 2], "data_offsets": [12, 28]},
                "base_model.model.model.layers.7.self_attn.q_a_proj.lora_A.weight": {"dtype": "F16", "shape": [1, 2], "data_offsets": [28, 32]},
                "base_model.model.model.layers.7.self_attn.q_a_proj.lora_B.weight": {"dtype": "F16", "shape": [2, 1], "data_offsets": [32, 36]},
                "base_model.model.model.layers.7.self_attn.q_b_proj.lora_A.default.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [36, 38]},
                "base_model.model.model.layers.7.self_attn.q_b_proj.lora_B.default.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [38, 40]},
                "base_model.model.model.layers.9.self_attn.kv_proj.lora_A.weight": {"dtype": "F16", "shape": [2, 1], "data_offsets": [40, 44]},
                "base_model.model.model.layers.9.self_attn.kv_proj.lora_B.weight": {"dtype": "F16", "shape": [1, 2], "data_offsets": [44, 48]},
            }
            write_st(src, header, data)

            summary = conv.convert_file(src, dst)

            out_header, out_data = read_st(dst)
            expected_keys = [
                "__metadata__",
                "output.lora_A.weight",
                "output.lora_B.weight",
                "blk.7.attn_q_a.lora_A.weight",
                "blk.7.attn_q_a.lora_B.weight",
                "blk.7.attn_q_b.lora_A.weight",
                "blk.7.attn_q_b.lora_B.weight",
                "blk.9.attn_kv.lora_A.weight",
                "blk.9.attn_kv.lora_B.weight",
            ]
            self.assertEqual(list(out_header), expected_keys)
            self.assertEqual(out_header["__metadata__"], {"format": "pt"})
            self.assertEqual(out_header["output.lora_A.weight"], header["base_model.model.lm_head.lora_A.weight"])
            self.assertEqual(out_header["blk.7.attn_q_b.lora_A.weight"], header["base_model.model.model.layers.7.self_attn.q_b_proj.lora_A.default.weight"])
            self.assertEqual(out_header["blk.9.attn_kv.lora_B.weight"], header["base_model.model.model.layers.9.self_attn.kv_proj.lora_B.weight"])
            self.assertEqual(out_data, data)
            self.assertEqual(summary["mapped_pairs"], 4)
            self.assertEqual(summary["ignored_tensors"], [])

    def test_dry_run_validates_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            src = tmp / "adapter.safetensors"
            dst = tmp / "dry-run-out.safetensors"
            header = {
                "base_model.model.lm_head.lora_A.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [0, 2]},
                "base_model.model.lm_head.lora_B.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [2, 4]},
            }
            write_st(src, header, b"abcd")

            summary = conv.convert_file(src, dst, dry_run=True)

            self.assertFalse(dst.exists())
            self.assertEqual(summary["dry_run"], True)
            self.assertEqual(
                summary["mappings"],
                [
                    {"source": "base_model.model.lm_head.lora_A.weight", "target": "output.lora_A.weight"},
                    {"source": "base_model.model.lm_head.lora_B.weight", "target": "output.lora_B.weight"},
                ],
            )

    def test_unknown_target_fails_closed_unless_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            src = tmp / "adapter.safetensors"
            dst = tmp / "out.safetensors"
            header = {
                "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [0, 2]},
                "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [2, 4]},
                "base_model.model.lm_head.lora_A.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [4, 6]},
                "base_model.model.lm_head.lora_B.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [6, 8]},
            }
            write_st(src, header, b"abcdefgh")

            with self.assertRaisesRegex(conv.ConversionError, "unknown module suffix"):
                conv.convert_file(src, dst)

            summary = conv.convert_file(src, dst, ignore_unknown=True)
            out_header, out_data = read_st(dst)
            self.assertEqual(set(out_header), {"output.lora_A.weight", "output.lora_B.weight"})
            self.assertEqual(out_header["output.lora_A.weight"]["data_offsets"], [0, 2])
            self.assertEqual(out_header["output.lora_B.weight"]["data_offsets"], [2, 4])
            self.assertEqual(out_data, b"efgh")
            self.assertEqual(summary["mapped_pairs"], 1)
            self.assertEqual(len(summary["ignored_tensors"]), 2)

    def test_collision_failure(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            src = tmp / "adapter.safetensors"
            dst = tmp / "out.safetensors"
            header = {
                "base_model.model.model.layers.3.self_attn.q_a_proj.lora_A.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [0, 2]},
                "base_model.model.model.layers.3.self_attn.q_a_proj.lora_B.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [2, 4]},
                "base_model.model.model.layers.3.attn.wq_a.lora_A.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [4, 6]},
                "base_model.model.model.layers.3.attn.wq_a.lora_B.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [6, 8]},
            }
            write_st(src, header, b"abcdefgh")

            with self.assertRaisesRegex(conv.ConversionError, "target collision"):
                conv.convert_file(src, dst)
            self.assertFalse(dst.exists())

    def test_unsupported_dtype_and_zero_rank_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            bf16 = tmp / "bf16.safetensors"
            zero_rank = tmp / "zero.safetensors"
            dst = tmp / "out.safetensors"
            write_st(
                bf16,
                {"base_model.model.lm_head.lora_A.weight": {"dtype": "BF16", "shape": [1, 1], "data_offsets": [0, 2]},
                 "base_model.model.lm_head.lora_B.weight": {"dtype": "BF16", "shape": [1, 1], "data_offsets": [2, 4]}},
                b"abcd",
            )
            write_st(
                zero_rank,
                {"base_model.model.lm_head.lora_A.weight": {"dtype": "F16", "shape": [0, 1], "data_offsets": [0, 0]},
                 "base_model.model.lm_head.lora_B.weight": {"dtype": "F16", "shape": [1, 0], "data_offsets": [0, 0]}},
                b"",
            )
            with self.assertRaisesRegex(conv.ConversionError, "not supported by the DS4 runtime"):
                conv.convert_file(bf16, dst)
            with self.assertRaisesRegex(conv.ConversionError, "rank must be"):
                conv.convert_file(zero_rank, dst)

    def test_alpha_metadata_embedded_from_adapter_config(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            src = tmp / "adapter.safetensors"
            cfg = tmp / "adapter_config.json"
            dst = tmp / "adapter-ds4.safetensors"
            write_st(
                src,
                {"base_model.model.lm_head.lora_A.weight": {"dtype": "F16", "shape": [2, 3], "data_offsets": [0, 12]},
                 "base_model.model.lm_head.lora_B.weight": {"dtype": "F16", "shape": [4, 2], "data_offsets": [12, 28]}},
                bytes(range(28)),
            )
            cfg.write_text(json.dumps({"lora_alpha": 32, "r": 8}), encoding="utf-8")
            conv.convert_file(src, dst)
            out_header, _out_data = read_st(dst)
            self.assertEqual(out_header["__metadata__"]["ds4_lora_alpha"], "32.0")

    def test_ignore_unknown_preserves_embedded_alpha(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            src = tmp / "adapter.safetensors"
            cfg = tmp / "adapter_config.json"
            dst = tmp / "adapter-ds4.safetensors"
            header = {
                "base_model.model.lm_head.lora_A.weight": {"dtype": "F16", "shape": [2, 3], "data_offsets": [0, 12]},
                "base_model.model.lm_head.lora_B.weight": {"dtype": "F16", "shape": [4, 2], "data_offsets": [12, 28]},
                "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [28, 30]},
                "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [30, 32]},
            }
            write_st(src, header, bytes(range(32)))
            cfg.write_text(json.dumps({"lora_alpha": 64, "r": 2}), encoding="utf-8")
            conv.convert_file(src, dst, ignore_unknown=True)
            out_header, _out_data = read_st(dst)
            self.assertEqual(out_header["__metadata__"]["ds4_lora_alpha"], "64.0")
            self.assertEqual(set(out_header) - {"__metadata__"}, {"output.lora_A.weight", "output.lora_B.weight"})

    def test_pair_rank_and_shape_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            src = tmp / "adapter.safetensors"
            dst = tmp / "out.safetensors"
            rank_mismatch = {
                "base_model.model.lm_head.lora_A.weight": {"dtype": "F16", "shape": [2, 3], "data_offsets": [0, 12]},
                "base_model.model.lm_head.lora_B.weight": {"dtype": "F16", "shape": [4, 1], "data_offsets": [12, 20]},
            }
            non_2d = {
                "base_model.model.lm_head.lora_A.weight": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
                "base_model.model.lm_head.lora_B.weight": {"dtype": "F16", "shape": [4, 2], "data_offsets": [4, 20]},
            }
            write_st(src, rank_mismatch, bytes(range(20)))
            with self.assertRaisesRegex(conv.ConversionError, "rank mismatch"):
                conv.convert_file(src, dst)
            write_st(src, non_2d, bytes(range(20)))
            with self.assertRaisesRegex(conv.ConversionError, "must be 2D"):
                conv.convert_file(src, dst)

    def test_non_lora_tensor_and_missing_pair_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            non_lora = tmp / "non-lora.safetensors"
            missing = tmp / "missing.safetensors"
            dst = tmp / "out.safetensors"
            write_st(
                non_lora,
                {"base_model.model.lm_head.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [0, 2]}},
                b"ab",
            )
            write_st(
                missing,
                {"base_model.model.lm_head.lora_A.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [0, 2]}},
                b"ab",
            )

            with self.assertRaisesRegex(conv.ConversionError, "not a LoRA A/B"):
                conv.convert_file(non_lora, dst)
            with self.assertRaisesRegex(conv.ConversionError, "missing matching lora_B"):
                conv.convert_file(missing, dst)


if __name__ == "__main__":
    unittest.main()
