import json
import pathlib
import tempfile
import unittest

from scripts import shim_ds4_safetensors as shim


def write_st(path: pathlib.Path, header: dict, data: bytes) -> None:
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + data)


class ShimDs4SafetensorsTests(unittest.TestCase):
    def test_scan_counts_f8_scale(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            header = {
                "x.scale": {"dtype": "F8_E8M0", "shape": [3], "data_offsets": [0, 3]},
                "x.weight": {"dtype": "I8", "shape": [2], "data_offsets": [3, 5]},
            }
            write_st(root / "model-00001-of-00001.safetensors", header, bytes([127, 128, 129, 1, 2]))
            summary = shim.scan_snapshot(root)
            self.assertEqual(summary["dtype_counts"]["F8_E8M0"], 1)
            self.assertEqual(summary["f8_e8m0_non_scale_count"], 0)

    def test_f8_special_values_decode_or_preserve_nan(self):
        # F8_E8M0: 0 is 2^-127, 255 is NaN.
        self.assertEqual(shim.f8_e8m0_to_bf16(bytes([0, 255])), bytes([0x40, 0x00, 0xC0, 0x7F]))
        self.assertEqual(shim.f8_e8m0_to_f32(bytes([0, 255])), bytes([0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0xC0, 0x7F]))
        # F8_E4M3FN: 0x7f and 0xff are NaN sentinels, not finite +/-480.
        self.assertEqual(shim.f8_e4m3_to_bf16(bytes([0x7F, 0xFF])), bytes([0xC0, 0x7F, 0xC0, 0x7F]))
        self.assertEqual(shim.f8_e4m3_to_f32(bytes([0x7F, 0xFF])), bytes([0x00, 0x00, 0xC0, 0x7F, 0x00, 0x00, 0xC0, 0x7F]))

    def test_rewrite_f8_e4m3_weights_to_bf16(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td) / "src"
            out = pathlib.Path(td) / "dst"
            root.mkdir()
            header = {
                "x.weight": {"dtype": "F8_E4M3", "shape": [4], "data_offsets": [0, 4]},
            }
            # E4M3: 0x38=1.0, 0x40=2.0, 0x30=0.5, 0xb8=-1.0.
            write_st(root / "model-00001-of-00001.safetensors", header, bytes([0x38, 0x40, 0x30, 0xB8]))
            results = shim.convert_snapshot(root, out, "BF16", None, True, True)
            self.assertEqual(results[0]["fp8_converted"], 1)
            self.assertEqual(results[0]["fp8_old_bytes"], 4)
            self.assertEqual(results[0]["f8_e8m0_old_bytes"], 0)
            header_len, new_header, _ = shim.read_header(out / "model-00001-of-00001.safetensors")
            self.assertEqual(new_header["x.weight"]["dtype"], "BF16")
            self.assertEqual(new_header["x.weight"]["data_offsets"], [0, 8])
            data = (out / "model-00001-of-00001.safetensors").read_bytes()[8 + header_len:]
            self.assertEqual(data, bytes([0x80, 0x3F, 0x00, 0x40, 0x00, 0x3F, 0x80, 0xBF]))

    def test_rewrite_f8_e4m3_weights_to_f32(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td) / "src"
            out = pathlib.Path(td) / "dst"
            root.mkdir()
            header = {
                "x.weight": {"dtype": "F8_E4M3", "shape": [2], "data_offsets": [0, 2]},
            }
            write_st(root / "model-00001-of-00001.safetensors", header, bytes([0x38, 0xB8]))
            shim.convert_snapshot(root, out, "F32", None, True, True)
            header_len, new_header, _ = shim.read_header(out / "model-00001-of-00001.safetensors")
            self.assertEqual(new_header["x.weight"]["dtype"], "F32")
            data = (out / "model-00001-of-00001.safetensors").read_bytes()[8 + header_len:]
            self.assertEqual(data, bytes([0, 0, 0x80, 0x3F, 0, 0, 0x80, 0xBF]))

    def test_rewrite_f8_e8m0_to_bf16(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td) / "src"
            out = pathlib.Path(td) / "dst"
            root.mkdir()
            header = {
                "x.scale": {"dtype": "F8_E8M0", "shape": [3], "data_offsets": [0, 3]},
                "x.weight": {"dtype": "I8", "shape": [2], "data_offsets": [3, 5]},
            }
            write_st(root / "model-00001-of-00001.safetensors", header, bytes([127, 128, 129, 1, 2]))
            (root / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
            results = shim.convert_snapshot(root, out, "BF16", None, True, True)
            self.assertEqual(results[0]["f8_e8m0_converted"], 1)
            self.assertEqual(results[0]["fp8_converted"], 1)
            self.assertEqual(results[0]["fp8_old_bytes"], 3)
            self.assertEqual(results[0]["f8_e8m0_old_bytes"], 3)
            header_len, new_header, _ = shim.read_header(out / "model-00001-of-00001.safetensors")
            self.assertEqual(new_header["x.scale"]["dtype"], "BF16")
            self.assertEqual(new_header["x.scale"]["data_offsets"], [0, 6])
            self.assertEqual(new_header["x.weight"]["data_offsets"], [6, 8])
            data = (out / "model-00001-of-00001.safetensors").read_bytes()[8 + header_len:]
            # BF16 powers of two for exponent bytes 127, 128, 129: 1.0, 2.0, 4.0.
            self.assertEqual(data[:6], bytes([0x80, 0x3F, 0x00, 0x40, 0x80, 0x40]))
            self.assertEqual(data[6:], bytes([1, 2]))

    def test_refuses_non_scale_f8_e8m0_by_default_but_allows_e4m3_weights(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td) / "src"
            out = pathlib.Path(td) / "dst"
            root.mkdir()
            header = {
                "x.weight": {"dtype": "F8_E8M0", "shape": [1], "data_offsets": [0, 1]},
            }
            write_st(root / "model-00001-of-00001.safetensors", header, bytes([127]))
            with self.assertRaises(SystemExit):
                shim.convert_snapshot(root, out, "BF16", None, True, True)
            header["x.weight"]["dtype"] = "F8_E4M3"
            write_st(root / "model-00001-of-00001.safetensors", header, bytes([0x38]))
            shim.convert_snapshot(root, out, "BF16", None, True, True)

    def test_updates_index_total_size(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td) / "src"
            out = pathlib.Path(td) / "dst"
            root.mkdir()
            header = {
                "x.scale": {"dtype": "F8_E8M0", "shape": [1], "data_offsets": [0, 1]},
                "x.weight": {"dtype": "I8", "shape": [1], "data_offsets": [1, 2]},
            }
            write_st(root / "model-00001-of-00001.safetensors", header, bytes([127, 1]))
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": 2}, "weight_map": {"x.scale": "model-00001-of-00001.safetensors", "x.weight": "model-00001-of-00001.safetensors"}}),
                encoding="utf-8",
            )
            shim.convert_snapshot(root, out, "BF16", None, True, True)
            index = json.loads((out / "model.safetensors.index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["metadata"]["total_size"], 3)

    def test_scan_reports_all_fp8_dtypes(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            header = {
                "x.scale": {"dtype": "F8_E8M0", "shape": [1], "data_offsets": [0, 1]},
                "x.weight": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [1, 2]},
            }
            write_st(root / "model-00001-of-00001.safetensors", header, bytes([127, 0x38]))
            summary = shim.scan_snapshot(root)
            self.assertEqual(summary["fp8_dtype_counts"], {"F8_E4M3": 1, "F8_E8M0": 1})


if __name__ == "__main__":
    unittest.main()
