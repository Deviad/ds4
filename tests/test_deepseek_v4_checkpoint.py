#!/usr/bin/env python3
import json
import struct
import tempfile
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_SRC = REPO_ROOT / "python-envs" / "mlx" / "src"
if str(MLX_SRC) not in sys.path:
    sys.path.insert(0, str(MLX_SRC))


class DeepSeekV4CheckpointHeaderTests(unittest.TestCase):
    def _assert_model_4bit_absent_or_valid_current_artifact(self):
        model_4bit = Path("/Volumes/Data NVME/mlx-ft/ds4/model-4bit")
        if not model_4bit.exists():
            return
        cfg = json.loads((model_4bit / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg.get("model_type"), "deepseek_v4_nn")
        self.assertEqual(cfg.get("quantization", {}).get("bits"), 4)

    def _write_header_only_shard(self, path: Path, header: dict[str, dict[str, object]]) -> None:
        raw = json.dumps(header, sort_keys=True).encode("utf-8")
        path.write_bytes(struct.pack("<Q", len(raw)) + raw)

    def _canonical_to_checkpoint_source(self) -> dict[str, str]:
        reverse = {
            "embed.weight": "embed.weight",
            "input_layernorm.weight": "layers.0.attn_norm.weight",
            "post_attention_layernorm.weight": "layers.0.ffn_norm.weight",
            "q_a_proj.weight": "layers.0.attn.wq_a.weight",
            "q_norm.weight": "layers.0.attn.q_norm.weight",
            "q_b_proj.weight": "layers.0.attn.wq_b.weight",
            "kv_proj.weight": "layers.0.attn.wkv.weight",
            "kv_norm.weight": "layers.0.attn.kv_norm.weight",
            "o_a_proj.weight": "layers.0.attn.wo_a.weight",
            "o_b_proj.weight": "layers.0.attn.wo_b.weight",
            "sinks": "layers.0.attn.attn_sink",
            "attn_hc.fn": "layers.0.hc_attn_fn",
            "attn_hc.base": "layers.0.hc_attn_base",
            "attn_hc.scale": "layers.0.hc_attn_scale",
            "ffn_hc.fn": "layers.0.hc_ffn_fn",
            "ffn_hc.base": "layers.0.hc_ffn_base",
            "ffn_hc.scale": "layers.0.hc_ffn_scale",
            "mlp.gate.weight": "layers.0.ffn.gate.weight",
            "mlp.gate.e_score_correction_bias": "layers.0.ffn.gate.e_score_correction_bias",
            "mlp.shared_experts.w1.weight": "layers.0.ffn.shared_experts.w1.weight",
            "mlp.shared_experts.w2.weight": "layers.0.ffn.shared_experts.w2.weight",
            "mlp.shared_experts.w3.weight": "layers.0.ffn.shared_experts.w3.weight",
        }
        for eid in range(2):
            for proj in ("w1", "w2", "w3"):
                reverse[f"mlp.experts.{eid}.{proj}.weight"] = f"layers.0.ffn.experts.{eid}.{proj}.weight"
                reverse[f"mlp.experts.{eid}.{proj}.scale"] = f"layers.0.ffn.experts.{eid}.{proj}.scale"
        return reverse

    def _complete_tiny_checkpoint(self, root: Path) -> Path:
        from ds4_ft_mlx.deepseek_v4_checkpoint import bounded_real_required_keys

        reverse = self._canonical_to_checkpoint_source()

        weight_map: dict[str, str] = {}
        header: dict[str, dict[str, object]] = {}
        for key in bounded_real_required_keys(expert_dtype="i8", n_routed_experts=2):
            source = reverse[key]
            weight_map[source] = "model-00001-of-00001.safetensors"
            if source.endswith(".scale"):
                dtype = "BF16"
                shape = [16, 1]
            elif ".experts." in source and source.endswith(".weight"):
                dtype = "I8"
                shape = [16, 16]
            elif source.endswith("attn_sink"):
                dtype = "F32"
                shape = [1]
            else:
                dtype = "BF16"
                shape = [16, 16] if source.endswith(("weight", "_fn")) else [3]
            header[source] = {"dtype": dtype, "shape": shape, "data_offsets": [10_000, 10_001]}

        # Extra real-checkpoint-like tensors: present in the index/header but not
        # part of the bounded loader, and no payload bytes are written.
        weight_map["layers.0.attn.wq_a.scale"] = "model-00001-of-00001.safetensors"
        header["layers.0.attn.wq_a.scale"] = {"dtype": "F8_E8M0", "shape": [16, 1], "data_offsets": [20_000, 20_001]}
        weight_map["layers.1.attn.wq_a.weight"] = "model-00001-of-00001.safetensors"
        header["layers.1.attn.wq_a.weight"] = {"dtype": "F8_E4M3", "shape": [16, 16], "data_offsets": [30_000, 30_001]}
        weight_map["mtp.0.hc_head_fn"] = "model-00001-of-00001.safetensors"
        header["mtp.0.hc_head_fn"] = {"dtype": "BF16", "shape": [2, 16], "data_offsets": [40_000, 40_001]}

        self._write_header_only_shard(root / "model-00001-of-00001.safetensors", header)
        index = root / "model.safetensors.index.json"
        index.write_text(json.dumps({"metadata": {"total_size": 999999}, "weight_map": weight_map}, sort_keys=True), encoding="utf-8")
        return index

    def test_checkpoint_key_canonicalization_for_real_names(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import canonicalize_checkpoint_key

        self.assertEqual(canonicalize_checkpoint_key("embed.weight"), "embed.weight")
        self.assertEqual(canonicalize_checkpoint_key("layers.0.attn_norm.weight"), "input_layernorm.weight")
        self.assertEqual(canonicalize_checkpoint_key("layers.0.ffn_norm.weight"), "post_attention_layernorm.weight")
        self.assertEqual(canonicalize_checkpoint_key("layers.0.attn.wq_a.weight"), "q_a_proj.weight")
        self.assertEqual(canonicalize_checkpoint_key("layers.0.attn.q_norm.weight"), "q_norm.weight")
        self.assertEqual(canonicalize_checkpoint_key("layers.0.attn.attn_sink"), "sinks")
        self.assertEqual(canonicalize_checkpoint_key("layers.0.hc_ffn_scale"), "ffn_hc.scale")
        self.assertEqual(canonicalize_checkpoint_key("layers.0.ffn.experts.1.w3.scale"), "mlp.experts.1.w3.scale")
        self.assertIsNone(canonicalize_checkpoint_key("layers.1.attn.wq_a.weight"))
        self.assertIsNone(canonicalize_checkpoint_key("mtp.0.hc_head_fn"))

    def test_validate_checkpoint_header_reads_no_payload_and_reports_coverage(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import validate_bounded_real_checkpoint_headers

        with tempfile.TemporaryDirectory() as td:
            index = self._complete_tiny_checkpoint(Path(td))
            report = validate_bounded_real_checkpoint_headers(index, expected_metadata={})
        self.assertEqual(report["kind"], "deepseek-v4-checkpoint-header-report")
        self.assertEqual(len(report["index_sha256"]), 64)
        self.assertTrue(report["coverage_ok"])
        self.assertEqual(report["missing"], [])
        self.assertEqual(report["matched"]["q_a_proj.weight"]["source_name"], "layers.0.attn.wq_a.weight")
        self.assertEqual(report["matched"]["mlp.experts.0.w1.scale"]["dtype"], "BF16")
        self.assertIn("q_a_proj.scale", report["unexpected_families"])
        self.assertIn("other_layer_or_unmapped_layer_tensor", report["ignored_family_counts"])
        self.assertIn("mtp_tensor", report["ignored_family_counts"])
        self.assertTrue(any("header-only validation" in blocker for blocker in report["blockers"]))

    def test_missing_zero_buffer_bias_is_synthesized_with_proof(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import validate_bounded_real_checkpoint_headers

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index = self._complete_tiny_checkpoint(root)
            data = json.loads(index.read_text(encoding="utf-8"))
            del data["weight_map"]["layers.0.ffn.gate.e_score_correction_bias"]
            index.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
            report = validate_bounded_real_checkpoint_headers(index, expected_metadata={})
        self.assertTrue(report["coverage_ok"])
        default = report["synthetic_defaults"]["mlp.gate.e_score_correction_bias"]
        self.assertEqual(default["value"], "zeros")
        self.assertIn("persistent zero buffer", default["proof"])

    def test_missing_tensor_diagnostics(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import validate_bounded_real_checkpoint_headers

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index = self._complete_tiny_checkpoint(root)
            data = json.loads(index.read_text(encoding="utf-8"))
            del data["weight_map"]["layers.0.attn.wq_a.weight"]
            index.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
            report = validate_bounded_real_checkpoint_headers(index, expected_metadata={})
        self.assertFalse(report["coverage_ok"])
        self.assertIn("q_a_proj.weight", report["missing"])
        self.assertTrue(any("missing" in blocker for blocker in report["blockers"]))

    def test_dtype_and_shape_mismatches_are_reported(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import validate_bounded_real_checkpoint_headers

        with tempfile.TemporaryDirectory() as td:
            index = self._complete_tiny_checkpoint(Path(td))
            expected = {
                "q_a_proj.weight": {"dtype": "BF16", "shape": [99, 99]},
                "mlp.experts.0.w1.weight": {"dtype": "BF16", "shape": [16, 16]},
            }
            report = validate_bounded_real_checkpoint_headers(index, expected_metadata=expected)
        self.assertFalse(report["coverage_ok"])
        self.assertEqual(report["shape_mismatches"][0]["key"], "q_a_proj.weight")
        self.assertEqual(report["dtype_mismatches"][0]["key"], "mlp.experts.0.w1.weight")
        self.assertTrue(any("shape mismatches" in blocker for blocker in report["blockers"]))
        self.assertTrue(any("dtype mismatches" in blocker for blocker in report["blockers"]))

    def test_required_keys_fail_closed_for_unproven_expert_count(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import bounded_real_required_keys

        with self.assertRaisesRegex(ValueError, "n_routed_experts=2"):
            bounded_real_required_keys(expert_dtype="i8", n_routed_experts=3)

    def test_malformed_index_is_rejected(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import validate_bounded_real_checkpoint_headers

        with tempfile.TemporaryDirectory() as td:
            index = Path(td) / "model.safetensors.index.json"
            index.write_text("{not-json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "malformed safetensors index JSON"):
                validate_bounded_real_checkpoint_headers(index)
            index.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "weight_map"):
                validate_bounded_real_checkpoint_headers(index)

    def test_stale_index_missing_shard_is_rejected(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import validate_bounded_real_checkpoint_headers

        with tempfile.TemporaryDirectory() as td:
            index = Path(td) / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"embed.weight": "missing.safetensors"}}), encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "missing safetensors shard"):
                validate_bounded_real_checkpoint_headers(index)

    def test_truncated_safetensors_header_is_rejected(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import validate_bounded_real_checkpoint_headers

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shard = root / "model-00001-of-00001.safetensors"
            shard.write_bytes(struct.pack("<Q", 64) + b"{}")
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"embed.weight": shard.name}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "truncated safetensors header"):
                validate_bounded_real_checkpoint_headers(index)

    def _write_tiny_payload_checkpoint(self, root: Path) -> Path:
        try:
            import numpy as np
            from safetensors.numpy import save_file
        except Exception as exc:
            self.skipTest(f"safetensors/numpy unavailable: {exc}")
        tensors = {
            "layers.0.attn.wq_a.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
            "layers.0.attn.q_norm.weight": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "layers.0.ffn.gate.weight": np.arange(12, dtype=np.float32).reshape(4, 3),
            "layers.0.unselected.weight": np.array([99.0], dtype=np.float32),
        }
        shard = root / "model-00001-of-00001.safetensors"
        save_file(tensors, shard)
        index = root / "model.safetensors.index.json"
        index.write_text(
            json.dumps({"weight_map": {name: shard.name for name in tensors}}, sort_keys=True),
            encoding="utf-8",
        )
        return index

    def test_selected_tensor_dry_run_reads_headers_only_and_synthesizes_zero_bias(self):
        from ds4_ft_mlx import deepseek_v4_checkpoint as ckpt

        with tempfile.TemporaryDirectory() as td:
            index = self._write_tiny_payload_checkpoint(Path(td))
            old_loader = ckpt._load_selected_tensors_as_mlx
            try:
                def _boom(*_args, **_kwargs):
                    raise AssertionError("payload loader must not run during dry-run")

                ckpt._load_selected_tensors_as_mlx = _boom
                result = ckpt.plan_selected_tensor_load(
                    index,
                    {"q_norm.weight", "mlp.gate.e_score_correction_bias"},
                    expected_metadata={
                        "q_norm.weight": {"dtype": "F32", "shape": [3]},
                        "mlp.gate.e_score_correction_bias": {"dtype": "F32", "shape": [4]},
                    },
                    max_bytes=1024,
                    execute=False,
                )
            finally:
                ckpt._load_selected_tensors_as_mlx = old_loader
        plan = result["plan"]
        self.assertFalse(plan["execute"])
        self.assertEqual(result["arrays"], {})
        self.assertEqual(plan["tensor_count"], 1)
        self.assertEqual(plan["tensors"][0]["source_name"], "layers.0.attn.q_norm.weight")
        self.assertIn("mlp.gate.e_score_correction_bias", plan["synthetic_defaults"])
        self.assertTrue(plan["headers_read"])

    def test_selected_tensor_execute_loads_only_selected_tensors_with_budget(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx unavailable: {exc}")
        from ds4_ft_mlx.deepseek_v4_checkpoint import plan_selected_tensor_load

        with tempfile.TemporaryDirectory() as td:
            index = self._write_tiny_payload_checkpoint(Path(td))
            result = plan_selected_tensor_load(
                index,
                {"q_norm.weight", "mlp.gate.e_score_correction_bias"},
                expected_metadata={"q_norm.weight": {"dtype": "F32", "shape": [3]}},
                max_bytes=1024,
                execute=True,
            )
        arrays = result["arrays"]
        self.assertEqual(sorted(arrays), ["mlp.gate.e_score_correction_bias", "q_norm.weight"])
        self.assertEqual(arrays["q_norm.weight"].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(arrays["mlp.gate.e_score_correction_bias"].tolist(), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(result["plan"]["payload_reader"], "bounded direct safetensors selected data_offsets reader")

    def test_selected_tensor_byte_budget_fail_closes_before_payload_load(self):
        from ds4_ft_mlx import deepseek_v4_checkpoint as ckpt

        with tempfile.TemporaryDirectory() as td:
            index = self._write_tiny_payload_checkpoint(Path(td))
            old_loader = ckpt._load_selected_tensors_as_mlx
            try:
                def _boom(*_args, **_kwargs):
                    raise AssertionError("payload loader must not run after budget failure")

                ckpt._load_selected_tensors_as_mlx = _boom
                with self.assertRaisesRegex(ckpt.CheckpointTensorLoadError, "byte budget exceeded"):
                    ckpt.plan_selected_tensor_load(
                        index,
                        {"q_a_proj.weight"},
                        expected_metadata={"q_a_proj.weight": {"dtype": "F32", "shape": [2, 3]}},
                        max_bytes=1,
                        execute=True,
                    )
            finally:
                ckpt._load_selected_tensors_as_mlx = old_loader

    def test_selected_tensor_execute_requires_explicit_byte_budget(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import CheckpointTensorLoadError, plan_selected_tensor_load

        with tempfile.TemporaryDirectory() as td:
            index = self._write_tiny_payload_checkpoint(Path(td))
            with self.assertRaisesRegex(CheckpointTensorLoadError, "requires an explicit max_bytes"):
                plan_selected_tensor_load(index, {"q_norm.weight"}, expected_metadata={}, execute=True)

    def test_selected_tensor_malformed_data_offsets_fail_closed(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import CheckpointTensorLoadError, plan_selected_tensor_load

        bad_offsets = [
            "not-a-list",
            [0],
            [0, 1, 2],
            [False, True],
            [0.0, 1.0],
            ["0", "1"],
            [-1, 1],
            [100, 10],
        ]
        for offsets in bad_offsets:
            with self.subTest(offsets=offsets):
                with tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    shard = root / "model-00001-of-00001.safetensors"
                    header = {"layers.0.attn.q_norm.weight": {"dtype": "F32", "shape": [3], "data_offsets": offsets}}
                    self._write_header_only_shard(shard, header)
                    index = root / "model.safetensors.index.json"
                    index.write_text(json.dumps({"weight_map": {"layers.0.attn.q_norm.weight": shard.name}}), encoding="utf-8")
                    with self.assertRaisesRegex(CheckpointTensorLoadError, "invalid safetensors data_offsets"):
                        plan_selected_tensor_load(index, {"q_norm.weight"}, expected_metadata={}, max_bytes=0, execute=True)

    def test_selected_tensor_malformed_real_shape_fail_closed_before_budget(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import CheckpointTensorLoadError, plan_selected_tensor_load

        bad_shapes = [[-1], [False], [1.5], ["3"], []]
        for shape in bad_shapes:
            with self.subTest(shape=shape):
                with tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    shard = root / "model-00001-of-00001.safetensors"
                    header = {"layers.0.attn.q_norm.weight": {"dtype": "F32", "shape": shape, "data_offsets": [0, 12]}}
                    self._write_header_only_shard(shard, header)
                    index = root / "model.safetensors.index.json"
                    index.write_text(json.dumps({"weight_map": {"layers.0.attn.q_norm.weight": shard.name}}), encoding="utf-8")
                    with self.assertRaisesRegex(CheckpointTensorLoadError, "invalid tensor shape"):
                        plan_selected_tensor_load(index, {"q_norm.weight"}, expected_metadata={}, max_bytes=0, execute=True)

    def test_selected_tensor_synthetic_default_rejects_malformed_gate_shape(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import CheckpointTensorLoadError, plan_selected_tensor_load

        bad_shapes = [[-5, 3], [False, 3], [1.5, 3], ["4", 3], []]
        for shape in bad_shapes:
            with self.subTest(shape=shape):
                with tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    shard = root / "model-00001-of-00001.safetensors"
                    header = {"layers.0.ffn.gate.weight": {"dtype": "F32", "shape": shape, "data_offsets": [0, 12]}}
                    self._write_header_only_shard(shard, header)
                    index = root / "model.safetensors.index.json"
                    index.write_text(json.dumps({"weight_map": {"layers.0.ffn.gate.weight": shard.name}}), encoding="utf-8")
                    with self.assertRaisesRegex(CheckpointTensorLoadError, "invalid gate shape"):
                        plan_selected_tensor_load(
                            index,
                            {"mlp.gate.e_score_correction_bias"},
                            expected_metadata={},
                            max_bytes=0,
                            execute=True,
                        )

    def test_selected_tensor_missing_dtype_and_shape_mismatches_fail(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import CheckpointTensorLoadError, plan_selected_tensor_load

        with tempfile.TemporaryDirectory() as td:
            index = self._write_tiny_payload_checkpoint(Path(td))
            with self.assertRaisesRegex(CheckpointTensorLoadError, "validation failed"):
                plan_selected_tensor_load(index, {"kv_proj.weight"}, expected_metadata={}, execute=False)
            with self.assertRaisesRegex(CheckpointTensorLoadError, "validation failed"):
                plan_selected_tensor_load(
                    index,
                    {"q_norm.weight"},
                    expected_metadata={"q_norm.weight": {"dtype": "I8", "shape": [2]}},
                    execute=False,
                )
            with self.assertRaisesRegex(CheckpointTensorLoadError, "validation failed"):
                plan_selected_tensor_load(
                    index,
                    {"mlp.gate.e_score_correction_bias"},
                    expected_metadata={"mlp.gate.e_score_correction_bias": {"dtype": "F32", "shape": [5]}},
                    execute=False,
                )

    def _tiny_complete_payload_array(self, key: str, ordinal: int):
        import numpy as np
        from ds4_ft_mlx.deepseek_v4_checkpoint import _default_bounded_model_args, bounded_model_expected_shapes

        shape = tuple(bounded_model_expected_shapes(_default_bounded_model_args())[key])
        if ".experts." in key and key.endswith(".weight"):
            values = (np.arange(int(np.prod(shape)), dtype=np.int16) + ordinal) % 127
            return values.astype(np.int8).reshape(shape)
        return np.arange(ordinal, ordinal + int(np.prod(shape)), dtype=np.float32).reshape(shape)

    def _complete_tiny_payload_checkpoint(self, root: Path, *, omit_bias: bool = True) -> tuple[Path, dict[str, dict[str, object]]]:
        try:
            import numpy as np  # noqa: F401
            from safetensors.numpy import save_file
        except Exception as exc:
            self.skipTest(f"safetensors/numpy unavailable: {exc}")
        from ds4_ft_mlx.deepseek_v4_checkpoint import bounded_real_required_keys

        reverse = self._canonical_to_checkpoint_source()
        tensors = {}
        expected = {}
        ordinal = 1
        for key in bounded_real_required_keys(expert_dtype="i8", n_routed_experts=2):
            if omit_bias and key == "mlp.gate.e_score_correction_bias":
                continue
            arr = self._tiny_complete_payload_array(key, ordinal)
            ordinal += arr.size + 3
            tensors[reverse[key]] = arr
            expected[key] = {"dtype": "I8" if arr.dtype.name == "int8" else "F32", "shape": list(arr.shape)}
        if omit_bias:
            expected["mlp.gate.e_score_correction_bias"] = {"dtype": "F32", "shape": [2]}
        shard = root / "model-00001-of-00001.safetensors"
        save_file(tensors, shard)
        index = root / "model.safetensors.index.json"
        index.write_text(
            json.dumps({"weight_map": {name: shard.name for name in tensors}}, sort_keys=True),
            encoding="utf-8",
        )
        return index, expected

    def test_bounded_model_expected_shapes_support_hc_mult4_metadata(self):
        from ds4_ft_mlx import deepseek_v4_checkpoint as ckpt

        args = ckpt._default_bounded_model_args()
        args.hc_mult = 4
        shapes = ckpt.bounded_model_expected_shapes(args)
        self.assertEqual(shapes["attn_hc.fn"], [24, 64])
        self.assertEqual(shapes["attn_hc.base"], [24])
        self.assertEqual(shapes["attn_hc.scale"], [3])
        self.assertEqual(shapes["ffn_hc.fn"], [24, 64])
        self.assertEqual(shapes["ffn_hc.base"], [24])
        self.assertEqual(shapes["ffn_hc.scale"], [3])
        self.assertEqual(shapes["hc_head.fn"], [4, 64])
        self.assertEqual(shapes["hc_head.base"], [4])
        self.assertEqual(shapes["hc_head.scale"], [1])
        self.assertEqual(shapes["norm.weight"], [16])
        self.assertEqual(shapes["lm_head.weight"], [4, 16])

        real_sized_args = ckpt._default_bounded_model_args()
        real_sized_args.hidden_size = 4096
        real_sized_args.head_dim = 512
        real_sized_args.q_lora_rank = 1024
        real_sized_args.o_lora_rank = 1024
        real_sized_args.moe_intermediate_size = 2048
        real_sized_args.vocab_size = 129280
        real_sized_args.hc_mult = 4
        real_shapes = ckpt.bounded_model_expected_shapes(real_sized_args)
        self.assertEqual(real_shapes["attn_hc.fn"], [24, 16384])
        self.assertEqual(real_shapes["ffn_hc.fn"], [24, 16384])
        self.assertEqual(real_shapes["hc_head.fn"], [4, 16384])

    def test_bounded_model_expected_shapes_support_multihead_grouped_attention_metadata(self):
        from ds4_ft_mlx import deepseek_v4_checkpoint as ckpt

        args = ckpt._default_bounded_model_args()
        args.hidden_size = 16
        args.num_attention_heads = 4
        args.head_dim = 2
        args.q_lora_rank = 3
        args.o_lora_rank = 5
        args.qk_rope_head_dim = 2
        args.o_groups = 2
        args.hc_mult = 2
        shapes = ckpt.bounded_model_expected_shapes(args)
        self.assertEqual(shapes["q_a_proj.weight"], [3, 16])
        self.assertEqual(shapes["q_norm.weight"], [3])
        self.assertEqual(shapes["q_b_proj.weight"], [8, 3])
        self.assertEqual(shapes["kv_proj.weight"], [2, 16])
        self.assertEqual(shapes["kv_norm.weight"], [2])
        self.assertEqual(shapes["sinks"], [4])
        self.assertEqual(shapes["o_a_proj.weight"], [10, 4])
        self.assertEqual(shapes["o_b_proj.weight"], [16, 10])
        self.assertEqual(shapes["attn_hc.fn"], [8, 32])
        self.assertEqual(shapes["ffn_hc.fn"], [8, 32])

        plan = {
            "selected_keys": sorted(shapes),
            "tensors": [
                {"canonical_key": key, "shape": value}
                for key, value in shapes.items()
                if key != "mlp.gate.e_score_correction_bias"
            ],
            "synthetic_defaults": {"mlp.gate.e_score_correction_bias": {"shape": shapes["mlp.gate.e_score_correction_bias"]}},
        }
        compatible = ckpt.validate_selected_plan_model_shapes(plan, args)
        self.assertTrue(compatible["model_shape_compatible"])
        self.assertEqual(compatible["model_shape_mismatches"], [])

    def test_bounded_model_expected_shapes_multihead_grouped_attention_mismatches(self):
        from ds4_ft_mlx import deepseek_v4_checkpoint as ckpt

        args = ckpt._default_bounded_model_args()
        args.hidden_size = 16
        args.num_attention_heads = 4
        args.head_dim = 2
        args.q_lora_rank = 3
        args.o_lora_rank = 5
        args.qk_rope_head_dim = 2
        args.o_groups = 2
        shapes = ckpt.bounded_model_expected_shapes(args)
        bad_shapes = dict(shapes)
        bad_shapes["q_b_proj.weight"] = [2, 3]
        bad_shapes["o_a_proj.weight"] = [5, 8]
        bad_shapes["o_b_proj.weight"] = [16, 8]
        bad_shapes["sinks"] = [1]
        plan = {
            "selected_keys": sorted(shapes),
            "tensors": [{"canonical_key": key, "shape": value} for key, value in bad_shapes.items()],
            "synthetic_defaults": {},
        }
        result = ckpt.validate_selected_plan_model_shapes(plan, args)
        self.assertFalse(result["model_shape_compatible"])
        mismatches = {item["key"]: item for item in result["model_shape_mismatches"]}
        self.assertEqual(mismatches["q_b_proj.weight"]["expected"], [8, 3])
        self.assertEqual(mismatches["o_a_proj.weight"]["expected"], [10, 4])
        self.assertEqual(mismatches["o_b_proj.weight"]["expected"], [16, 10])
        self.assertEqual(mismatches["sinks"]["expected"], [4])

    def test_bounded_model_expected_shapes_support_synthetic_topk_moe_unquantized_metadata(self):
        from ds4_ft_mlx import deepseek_v4_checkpoint as ckpt

        args = ckpt._default_bounded_model_args(expert_dtype="fp4", n_routed_experts=3, num_experts_per_tok=2)
        args.hidden_size = 8
        args.moe_intermediate_size = 6
        shapes = ckpt.bounded_model_expected_shapes(args)
        self.assertEqual(shapes["mlp.gate.weight"], [3, 8])
        self.assertEqual(shapes["mlp.gate.e_score_correction_bias"], [3])
        self.assertEqual(shapes["mlp.shared_experts.w1.weight"], [6, 8])
        self.assertEqual(shapes["mlp.shared_experts.w2.weight"], [8, 6])
        self.assertEqual(shapes["mlp.shared_experts.w3.weight"], [6, 8])
        for eid in range(3):
            self.assertEqual(shapes[f"mlp.experts.{eid}.w1.weight"], [6, 8])
            self.assertEqual(shapes[f"mlp.experts.{eid}.w2.weight"], [8, 6])
            self.assertEqual(shapes[f"mlp.experts.{eid}.w3.weight"], [6, 8])
            self.assertNotIn(f"mlp.experts.{eid}.w1.scale", shapes)
        self.assertNotIn("mlp.experts.3.w1.weight", shapes)

        plan = {
            "selected_keys": sorted(shapes),
            "tensors": [
                {"canonical_key": key, "shape": value}
                for key, value in shapes.items()
                if key != "mlp.gate.e_score_correction_bias"
            ],
            "synthetic_defaults": {"mlp.gate.e_score_correction_bias": {"shape": [3]}},
        }
        compatible = ckpt.validate_selected_plan_model_shapes(plan, args)
        self.assertTrue(compatible["model_shape_compatible"])
        self.assertEqual(compatible["model_shape_mismatches"], [])

    def test_bounded_model_expected_shapes_support_synthetic_topk_moe_i8_metadata(self):
        from ds4_ft_mlx import deepseek_v4_checkpoint as ckpt

        args = ckpt._default_bounded_model_args(expert_dtype="i8", n_routed_experts=4, num_experts_per_tok=2)
        args.hidden_size = 32
        args.moe_intermediate_size = 16
        shapes = ckpt.bounded_model_expected_shapes(args)
        self.assertEqual(shapes["mlp.gate.weight"], [4, 32])
        self.assertEqual(shapes["mlp.gate.e_score_correction_bias"], [4])
        for eid in range(4):
            self.assertEqual(shapes[f"mlp.experts.{eid}.w1.weight"], [16, 32])
            self.assertEqual(shapes[f"mlp.experts.{eid}.w1.scale"], [16, 2])
            self.assertEqual(shapes[f"mlp.experts.{eid}.w2.weight"], [32, 16])
            self.assertEqual(shapes[f"mlp.experts.{eid}.w2.scale"], [32, 1])
            self.assertEqual(shapes[f"mlp.experts.{eid}.w3.weight"], [16, 32])
            self.assertEqual(shapes[f"mlp.experts.{eid}.w3.scale"], [16, 2])

    def test_bounded_model_expected_shapes_synthetic_topk_moe_mismatches(self):
        from ds4_ft_mlx import deepseek_v4_checkpoint as ckpt

        args = ckpt._default_bounded_model_args(expert_dtype="fp4", n_routed_experts=3, num_experts_per_tok=2)
        args.hidden_size = 12
        args.moe_intermediate_size = 6
        shapes = ckpt.bounded_model_expected_shapes(args)
        bad_shapes = dict(shapes)
        bad_shapes["mlp.gate.weight"] = [2, args.hidden_size]
        bad_shapes["mlp.gate.e_score_correction_bias"] = [2]
        bad_shapes["mlp.experts.2.w1.weight"] = [args.hidden_size, args.moe_intermediate_size]
        bad_shapes["mlp.shared_experts.w2.weight"] = [args.moe_intermediate_size, args.hidden_size]
        plan = {
            "selected_keys": sorted(shapes),
            "tensors": [{"canonical_key": key, "shape": value} for key, value in bad_shapes.items()],
            "synthetic_defaults": {},
        }
        result = ckpt.validate_selected_plan_model_shapes(plan, args)
        self.assertFalse(result["model_shape_compatible"])
        mismatches = {item["key"]: item for item in result["model_shape_mismatches"]}
        self.assertEqual(mismatches["mlp.gate.weight"]["expected"], [3, args.hidden_size])
        self.assertEqual(mismatches["mlp.gate.e_score_correction_bias"]["expected"], [3])
        self.assertEqual(mismatches["mlp.experts.2.w1.weight"]["expected"], [args.moe_intermediate_size, args.hidden_size])
        self.assertEqual(mismatches["mlp.shared_experts.w2.weight"]["expected"], [args.hidden_size, args.moe_intermediate_size])

    def test_bounded_model_expected_shapes_reject_invalid_multihead_grouped_boundaries(self):
        from ds4_ft_mlx import deepseek_v4_checkpoint as ckpt

        cases = [
            ("num_key_value_heads", 2, "num_key_value_heads=1"),
            ("num_attention_heads", 0, "positive num_attention_heads"),
            ("o_groups", 0, "positive o_groups"),
            ("o_groups", 3, "divisible by o_groups"),
        ]
        for attr, value, pattern in cases:
            with self.subTest(attr=attr, value=value):
                args = ckpt._default_bounded_model_args()
                args.num_attention_heads = 4
                args.o_groups = 2
                setattr(args, attr, value)
                with self.assertRaisesRegex(ckpt.CheckpointTensorLoadError, pattern):
                    ckpt.bounded_model_expected_shapes(args)

    def test_bounded_model_expected_shapes_reject_invalid_synthetic_topk_moe_boundaries(self):
        from ds4_ft_mlx import deepseek_v4_checkpoint as ckpt

        cases = [
            ({"n_routed_experts": 0}, "positive n_routed_experts"),
            ({"n_routed_experts": 5}, "synthetic n_routed_experts<=4"),
            ({"num_experts_per_tok": 0}, "positive num_experts_per_tok"),
            ({"n_routed_experts": 3, "num_experts_per_tok": 4}, "num_experts_per_tok<=n_routed_experts"),
            ({"n_routed_experts": 256, "num_experts_per_tok": 6, "expert_dtype": "fp4"}, "synthetic n_routed_experts<=4"),
            ({"scoring_func": "softmax"}, "scoring_func='sqrtsoftplus'"),
            ({"expert_dtype": "mystery"}, "expert_dtype.*fp4.*i8"),
        ]
        for updates, pattern in cases:
            with self.subTest(updates=updates):
                args = ckpt._default_bounded_model_args()
                for attr, value in updates.items():
                    setattr(args, attr, value)
                with self.assertRaisesRegex(ckpt.CheckpointTensorLoadError, pattern):
                    ckpt.bounded_model_expected_shapes(args)

    def test_bounded_model_expected_shapes_reject_i8_block_shape_boundaries(self):
        from ds4_ft_mlx import deepseek_v4_checkpoint as ckpt

        for attr in ("hidden_size", "moe_intermediate_size"):
            with self.subTest(attr=attr):
                args = ckpt._default_bounded_model_args(expert_dtype="i8", n_routed_experts=3, num_experts_per_tok=2)
                setattr(args, attr, 18)
                with self.assertRaisesRegex(ckpt.CheckpointTensorLoadError, "I8.*divisible by 16"):
                    ckpt.bounded_model_expected_shapes(args)

    def test_bounded_model_expected_shapes_support_multilayer_per_layer_keys(self):
        from ds4_ft_mlx import deepseek_v4_checkpoint as ckpt

        args = ckpt._default_bounded_model_args()
        args.num_hidden_layers = 2
        args.layer_types = ["sliding_attention", "sliding_attention"]
        args.mlp_layer_types = ["moe", "moe"]
        shapes = ckpt.bounded_model_expected_shapes(args)
        # Single global embed/norm/lm_head.
        self.assertIn("embed.weight", shapes)
        self.assertIn("norm.weight", shapes)
        self.assertIn("lm_head.weight", shapes)
        # Per-layer keys use layers.{i}. prefix for multi-layer.
        for i in range(2):
            self.assertIn(f"layers.{i}.input_layernorm.weight", shapes)
            self.assertIn(f"layers.{i}.q_a_proj.weight", shapes)
            self.assertIn(f"layers.{i}.sinks", shapes)
            self.assertIn(f"layers.{i}.mlp.gate.weight", shapes)
            self.assertIn(f"layers.{i}.mlp.experts.0.w1.weight", shapes)
        # No flat single-layer keys for multi-layer.
        self.assertNotIn("input_layernorm.weight", shapes)
        self.assertNotIn("q_a_proj.weight", shapes)

        plan = {
            "selected_keys": sorted(shapes),
            "tensors": [
                {"canonical_key": key, "shape": value}
                for key, value in shapes.items()
                if key != "mlp.gate.e_score_correction_bias"
            ],
            "synthetic_defaults": {
                f"layers.{i}.mlp.gate.e_score_correction_bias": {"shape": shapes[f"layers.{i}.mlp.gate.e_score_correction_bias"]}
                for i in range(2)
            },
        }
        compatible = ckpt.validate_selected_plan_model_shapes(plan, args)
        self.assertTrue(compatible["model_shape_compatible"])
        self.assertEqual(compatible["model_shape_mismatches"], [])

    def test_bounded_model_expected_shapes_reject_multilayer_above_three_layers(self):
        from ds4_ft_mlx import deepseek_v4_checkpoint as ckpt

        args = ckpt._default_bounded_model_args()
        args.num_hidden_layers = 4
        args.layer_types = ["sliding_attention"] * 4
        args.mlp_layer_types = ["moe"] * 4
        with self.assertRaisesRegex(ckpt.CheckpointTensorLoadError, r"num_hidden_layers in \{1,2,3\}"):
            ckpt.bounded_model_expected_shapes(args)

    def test_bounded_model_expected_shapes_reject_multilayer_wrong_layer_types_length(self):
        from ds4_ft_mlx import deepseek_v4_checkpoint as ckpt

        args = ckpt._default_bounded_model_args()
        args.num_hidden_layers = 2
        args.layer_types = ["sliding_attention"]
        args.mlp_layer_types = ["moe", "moe"]
        with self.assertRaisesRegex(ckpt.CheckpointTensorLoadError, "layer_types='sliding_attention' with length"):
            ckpt.bounded_model_expected_shapes(args)

    def test_bounded_model_expected_shapes_reject_multilayer_hc_mult_gt1(self):
        from ds4_ft_mlx import deepseek_v4_checkpoint as ckpt

        args = ckpt._default_bounded_model_args()
        args.num_hidden_layers = 2
        args.layer_types = ["sliding_attention"] * 2
        args.mlp_layer_types = ["moe"] * 2
        args.hc_mult = 2
        with self.assertRaisesRegex(ckpt.CheckpointTensorLoadError, "multi-layer only for hc_mult=1"):
            ckpt.bounded_model_expected_shapes(args)

    def test_bounded_model_expected_shapes_allow_single_layer_hc_mult_gt1(self):
        from ds4_ft_mlx import deepseek_v4_checkpoint as ckpt

        args = ckpt._default_bounded_model_args()
        args.num_hidden_layers = 1
        args.hc_mult = 2
        shapes = ckpt.bounded_model_expected_shapes(args)
        self.assertEqual(shapes["hc_head.fn"], [2, 2 * args.hidden_size])

    def test_bounded_model_checkpoint_dry_run_is_header_only_for_complete_key_set(self):
        from ds4_ft_mlx import deepseek_v4_checkpoint as ckpt

        with tempfile.TemporaryDirectory() as td:
            index, expected = self._complete_tiny_payload_checkpoint(Path(td))
            old_loader = ckpt._load_selected_tensors_as_mlx
            old_model = ckpt.Model
            try:
                def _payload_boom(*_args, **_kwargs):
                    raise AssertionError("payload loader must not run during bounded model dry-run")

                class BoomModel:
                    _INTEGRATED_WEIGHTS = old_model._INTEGRATED_WEIGHTS

                    def __init__(self, *_args, **_kwargs):
                        raise AssertionError("model must not be constructed during dry-run")

                ckpt._load_selected_tensors_as_mlx = _payload_boom
                ckpt.Model = BoomModel
                result = ckpt.load_bounded_real_model_from_checkpoint(
                    index,
                    expected_metadata=expected,
                    max_bytes=20_000,
                    execute=False,
                )
            finally:
                ckpt._load_selected_tensors_as_mlx = old_loader
                ckpt.Model = old_model
        self.assertFalse(result["plan"]["execute"])
        self.assertEqual(result["arrays"], {})
        self.assertIsNone(result["model"])
        self.assertEqual(result["plan"]["selected_keys"], sorted(ckpt.bounded_real_required_keys()))
        self.assertIn("mlp.gate.e_score_correction_bias", result["plan"]["synthetic_defaults"])
        self.assertTrue(result["plan"]["model_shape_compatible"])
        self.assertEqual(result["plan"]["model_shape_mismatches"], [])

    def test_bounded_model_checkpoint_execute_model_shape_mismatch_precedes_payload_and_model(self):
        from ds4_ft_mlx import deepseek_v4_checkpoint as ckpt

        with tempfile.TemporaryDirectory() as td:
            index, expected = self._complete_tiny_payload_checkpoint(Path(td))
            bad_config = ckpt._default_bounded_model_args().__dict__.copy()
            bad_config["hidden_size"] = 32
            old_loader = ckpt._load_selected_tensors_as_mlx
            old_model = ckpt.Model
            try:
                def _payload_boom(*_args, **_kwargs):
                    raise AssertionError("payload loader must not run after model-shape mismatch")

                class BoomModel:
                    _INTEGRATED_WEIGHTS = old_model._INTEGRATED_WEIGHTS

                    def __init__(self, *_args, **_kwargs):
                        raise AssertionError("model must not be constructed after model-shape mismatch")

                ckpt._load_selected_tensors_as_mlx = _payload_boom
                ckpt.Model = BoomModel
                with self.assertRaisesRegex(ckpt.CheckpointTensorLoadError, "not compatible with ModelArgs") as cm:
                    ckpt.load_bounded_real_model_from_checkpoint(
                        index,
                        model_config=bad_config,
                        expected_metadata=expected,
                        max_bytes=1_000_000,
                        execute=True,
                    )
            finally:
                ckpt._load_selected_tensors_as_mlx = old_loader
                ckpt.Model = old_model
        payload = json.loads(str(cm.exception))
        self.assertFalse(payload["plan"]["model_shape_compatible"])
        self.assertTrue(any(item["key"] == "embed.weight" for item in payload["plan"]["model_shape_mismatches"]))

    def test_bounded_model_checkpoint_execute_loads_complete_canonical_set_into_model(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx unavailable: {exc}")
        from ds4_ft_mlx.deepseek_v4_checkpoint import bounded_real_required_keys, load_bounded_real_model_from_checkpoint

        class CaptureModel:
            def __init__(self):
                self.received = None
                self.strict = None

            def load_weights(self, weights, strict=True):
                self.received = dict(weights)
                self.strict = strict

        with tempfile.TemporaryDirectory() as td:
            index, expected = self._complete_tiny_payload_checkpoint(Path(td))
            model = CaptureModel()
            result = load_bounded_real_model_from_checkpoint(
                index,
                model=model,
                expected_metadata=expected,
                max_bytes=1_000_000,
                execute=True,
            )
        required = set(bounded_real_required_keys())
        self.assertIs(result["model"], model)
        self.assertTrue(result["model_loaded"])
        self.assertTrue(model.strict)
        self.assertEqual(set(model.received), required)
        self.assertEqual(set(result["arrays"]), required)
        self.assertEqual(model.received["mlp.gate.e_score_correction_bias"].tolist(), [0.0] * 2)
        self.assertEqual(list(model.received["q_norm.weight"].shape), [16])

    def test_bounded_model_checkpoint_execute_constructs_default_bounded_model(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx unavailable: {exc}")
        from ds4_ft_mlx.deepseek_v4_checkpoint import bounded_real_required_keys, load_bounded_real_model_from_checkpoint

        with tempfile.TemporaryDirectory() as td:
            index, expected = self._complete_tiny_payload_checkpoint(Path(td))
            result = load_bounded_real_model_from_checkpoint(
                index,
                expected_metadata=expected,
                max_bytes=1_000_000,
                execute=True,
            )
        self.assertTrue(result["model_loaded"])
        self.assertEqual(set(result["model"].parameters()), set(bounded_real_required_keys()))
        self.assertEqual(result["model"].parameters()["mlp.gate.e_score_correction_bias"].tolist(), [0.0] * 2)

    def test_bounded_model_checkpoint_execute_requires_explicit_max_bytes(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import CheckpointTensorLoadError, load_bounded_real_model_from_checkpoint

        with tempfile.TemporaryDirectory() as td:
            index, expected = self._complete_tiny_payload_checkpoint(Path(td))
            with self.assertRaisesRegex(CheckpointTensorLoadError, "requires an explicit max_bytes"):
                load_bounded_real_model_from_checkpoint(index, expected_metadata=expected, execute=True)

    def test_bounded_model_checkpoint_validation_failures_precede_payload_and_model(self):
        from ds4_ft_mlx import deepseek_v4_checkpoint as ckpt

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index, expected = self._complete_tiny_payload_checkpoint(root)
            missing_data = json.loads(index.read_text(encoding="utf-8"))
            del missing_data["weight_map"]["layers.0.attn.q_norm.weight"]
            missing_index = root / "missing-index.json"
            missing_index.write_text(json.dumps(missing_data, sort_keys=True), encoding="utf-8")
            error_cases = [
                (index, expected, 1, "byte budget exceeded"),
                (missing_index, expected, 1_000_000, "validation failed"),
                (index, {**expected, "q_norm.weight": {"dtype": "I8", "shape": [2]}}, 1_000_000, "validation failed"),
                (index, {**expected, "q_norm.weight": {"dtype": "F32", "shape": [99]}}, 1_000_000, "validation failed"),
            ]
            old_loader = ckpt._load_selected_tensors_as_mlx
            old_model = ckpt.Model
            try:
                def _payload_boom(*_args, **_kwargs):
                    raise AssertionError("payload loader must not run after validation/budget failure")

                class BoomModel:
                    _INTEGRATED_WEIGHTS = old_model._INTEGRATED_WEIGHTS

                    def __init__(self, *_args, **_kwargs):
                        raise AssertionError("model must not be constructed after validation/budget failure")

                ckpt._load_selected_tensors_as_mlx = _payload_boom
                ckpt.Model = BoomModel
                for case_index, case_expected, max_bytes, pattern in error_cases:
                    with self.subTest(pattern=pattern):
                        with self.assertRaisesRegex(ckpt.CheckpointTensorLoadError, pattern):
                            ckpt.load_bounded_real_model_from_checkpoint(
                                case_index,
                                expected_metadata=case_expected,
                                max_bytes=max_bytes,
                                execute=True,
                            )
            finally:
                ckpt._load_selected_tensors_as_mlx = old_loader
                ckpt.Model = old_model

    def test_non_object_safetensors_header_is_rejected(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import validate_bounded_real_checkpoint_headers

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shard = root / "model-00001-of-00001.safetensors"
            raw = json.dumps([]).encode("utf-8")
            shard.write_bytes(struct.pack("<Q", len(raw)) + raw)
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"embed.weight": shard.name}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "header JSON must be an object"):
                validate_bounded_real_checkpoint_headers(index)

    def _stacked_real_shape_meta(self, *, num_layers: int = 43, n_routed_experts: int = 2) -> dict[str, dict[str, object]]:
        meta: dict[str, dict[str, object]] = {
            "embed.weight": {"dtype": "BF16", "shape": [129280, 4096]},
            "head.weight": {"dtype": "BF16", "shape": [129280, 4096]},
            "norm.weight": {"dtype": "BF16", "shape": [4096]},
            "hc_head_fn": {"dtype": "F32", "shape": [4, 16384]},
            "hc_head_base": {"dtype": "F32", "shape": [4]},
            "hc_head_scale": {"dtype": "F32", "shape": [1]},
        }
        base = {
            "attn.attn_sink": ("F32", [64]),
            "attn.kv_norm.weight": ("BF16", [512]),
            "attn.q_norm.weight": ("BF16", [1024]),
            "attn.wkv.scale": ("F8_E8M0", [4, 32]),
            "attn.wkv.weight": ("F8_E4M3", [512, 4096]),
            "attn.wo_a.scale": ("F8_E8M0", [64, 32]),
            "attn.wo_a.weight": ("F8_E4M3", [8192, 4096]),
            "attn.wo_b.scale": ("F8_E8M0", [32, 64]),
            "attn.wo_b.weight": ("F8_E4M3", [4096, 8192]),
            "attn.wq_a.scale": ("F8_E8M0", [8, 32]),
            "attn.wq_a.weight": ("F8_E4M3", [1024, 4096]),
            "attn.wq_b.scale": ("F8_E8M0", [256, 8]),
            "attn.wq_b.weight": ("F8_E4M3", [32768, 1024]),
            "attn_norm.weight": ("BF16", [4096]),
            "ffn.gate.weight": ("BF16", [256, 4096]),
            "ffn.shared_experts.w1.scale": ("F8_E8M0", [16, 32]),
            "ffn.shared_experts.w1.weight": ("F8_E4M3", [2048, 4096]),
            "ffn.shared_experts.w2.scale": ("F8_E8M0", [32, 16]),
            "ffn.shared_experts.w2.weight": ("F8_E4M3", [4096, 2048]),
            "ffn.shared_experts.w3.scale": ("F8_E8M0", [16, 32]),
            "ffn.shared_experts.w3.weight": ("F8_E4M3", [2048, 4096]),
            "ffn_norm.weight": ("BF16", [4096]),
            "hc_attn_base": ("F32", [24]),
            "hc_attn_fn": ("F32", [24, 16384]),
            "hc_attn_scale": ("F32", [3]),
            "hc_ffn_base": ("F32", [24]),
            "hc_ffn_fn": ("F32", [24, 16384]),
            "hc_ffn_scale": ("F32", [3]),
        }
        compressor = {
            "attn.compressor.ape": ("F32", [4, 1024]),
            "attn.compressor.norm.weight": ("BF16", [512]),
            "attn.compressor.wgate.weight": ("BF16", [1024, 4096]),
            "attn.compressor.wkv.weight": ("BF16", [1024, 4096]),
        }
        indexer = {
            "attn.indexer.compressor.ape": ("F32", [4, 256]),
            "attn.indexer.compressor.norm.weight": ("BF16", [128]),
            "attn.indexer.compressor.wgate.weight": ("BF16", [256, 4096]),
            "attn.indexer.compressor.wkv.weight": ("BF16", [256, 4096]),
            "attn.indexer.weights_proj.weight": ("BF16", [64, 4096]),
            "attn.indexer.wq_b.scale": ("F8_E8M0", [64, 8]),
            "attn.indexer.wq_b.weight": ("F8_E4M3", [8192, 1024]),
        }
        for layer in range(num_layers):
            layer_families = dict(base)
            if layer < 3:
                layer_families["ffn.gate.tid2eid"] = ("I64", [129280, 6])
            else:
                layer_families["ffn.gate.bias"] = ("F32", [256])
            if layer >= 2:
                layer_families.update(compressor)
            if layer == 2 or (layer >= 4 and layer % 2 == 0):
                layer_families.update(indexer)
            for rest, (dtype, shape) in layer_families.items():
                meta[f"layers.{layer}.{rest}"] = {"dtype": dtype, "shape": list(shape)}
            for eid in range(n_routed_experts):
                for proj, shape in (("w1", [2048, 2048]), ("w2", [4096, 1024]), ("w3", [2048, 2048])):
                    meta[f"layers.{layer}.ffn.experts.{eid}.{proj}.weight"] = {"dtype": "I8", "shape": list(shape)}
                for proj, shape in (("w1", [2048, 128]), ("w2", [4096, 64]), ("w3", [2048, 128])):
                    meta[f"layers.{layer}.ffn.experts.{eid}.{proj}.scale"] = {"dtype": "F8_E8M0", "shape": list(shape)}
        return meta

    def test_stacked_enumeration_covers_all_43_layers(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import stacked_shape_compatibility_report

        report = stacked_shape_compatibility_report(self._stacked_real_shape_meta(), expected_num_layers=43, n_routed_experts=2)

        self.assertEqual(report["num_layers_observed"], 43)
        self.assertEqual(report["missing_layers"], [])
        self.assertEqual([template["layers"] for template in report["templates"]], [[0, 1], [2], list(range(3, 42, 2)), list(range(4, 43, 2))])
        self.assertEqual([len(template["signature"]) for template in report["templates"]], [35, 46, 39, 46])

    def test_stacked_shape_compat_passes_for_consistent_real_shaped_fixture(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import assert_stacked_shape_compatible, stacked_shape_compatibility_report

        meta = self._stacked_real_shape_meta()
        report = stacked_shape_compatibility_report(meta, expected_num_layers=43, n_routed_experts=2)

        self.assertTrue(report["ok"], report["blockers"])
        self.assertEqual(report["blockers"], [])
        self.assertEqual(assert_stacked_shape_compatible(meta, expected_num_layers=43, n_routed_experts=2), report)

    def test_stacked_shape_compat_fails_closed_on_missing_core_key(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import StackedShapeCompatError, assert_stacked_shape_compatible, stacked_shape_compatibility_report

        meta = self._stacked_real_shape_meta()
        del meta["layers.17.attn.wkv.weight"]
        report = stacked_shape_compatibility_report(meta, expected_num_layers=43, n_routed_experts=2)

        self.assertFalse(report["ok"])
        self.assertTrue(any(issue["key"] == "layers.17.attn.wkv.weight" for issue in report["core_family_issues"]))
        with self.assertRaisesRegex(StackedShapeCompatError, "wkv"):
            assert_stacked_shape_compatible(meta, expected_num_layers=43, n_routed_experts=2)

    def test_stacked_shape_compat_fails_closed_on_shape_mismatch(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import stacked_shape_compatibility_report

        meta = self._stacked_real_shape_meta()
        meta["layers.5.attn.wq_b.weight"] = {"dtype": "F8_E4M3", "shape": [1, 1024]}
        report = stacked_shape_compatibility_report(meta, expected_num_layers=43, n_routed_experts=2)

        self.assertFalse(report["ok"])
        issue = next(item for item in report["core_family_issues"] if item["key"] == "layers.5.attn.wq_b.weight")
        self.assertEqual(issue["expected_shape"], [32768, 1024])
        self.assertEqual(issue["observed_shape"], [1, 1024])

    def test_stacked_shape_compat_fails_closed_on_expert_index_gap(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import stacked_shape_compatibility_report

        meta = self._stacked_real_shape_meta(n_routed_experts=3)
        for proj in ("w1", "w2", "w3"):
            del meta[f"layers.9.ffn.experts.1.{proj}.weight"]
            del meta[f"layers.9.ffn.experts.1.{proj}.scale"]
        report = stacked_shape_compatibility_report(meta, expected_num_layers=43, n_routed_experts=3)

        self.assertFalse(report["ok"])
        self.assertTrue(any("expert index gap" in blocker and "layers.9" in blocker for blocker in report["blockers"]))

    def test_stacked_shape_compat_fails_closed_on_unclassified_or_inconsistent_family(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import stacked_shape_compatibility_report

        meta = self._stacked_real_shape_meta()
        meta["layers.3.attn.indexer.mystery.weight"] = {"dtype": "BF16", "shape": [1]}
        meta["layers.5.attn.compressor.wkv.weight"] = {"dtype": "BF16", "shape": [2048, 4096]}
        report = stacked_shape_compatibility_report(meta, expected_num_layers=43, n_routed_experts=2)

        self.assertFalse(report["ok"])
        self.assertIn("attn.indexer.mystery.weight", report["unclassified_families"])
        self.assertTrue(any(conflict["family"] == "attn.compressor.wkv.weight" for conflict in report["intra_template_shape_conflicts"]))

    def test_stacked_shape_compat_fails_closed_when_structural_family_disappears(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import StackedShapeCompatError, assert_stacked_shape_compatible, stacked_shape_compatibility_report

        meta = self._stacked_real_shape_meta()
        del meta["layers.5.attn.compressor.wkv.weight"]
        report = stacked_shape_compatibility_report(meta, expected_num_layers=43, n_routed_experts=2)

        self.assertFalse(report["ok"])
        self.assertTrue(any(issue["key"] == "layers.5.attn.compressor.wkv.weight" for issue in report["structural_family_issues"]))
        self.assertTrue(any("structural family" in blocker and "attn.compressor.wkv.weight" in blocker for blocker in report["blockers"]))
        with self.assertRaisesRegex(StackedShapeCompatError, "attn\\.compressor\\.wkv\\.weight"):
            assert_stacked_shape_compatible(meta, expected_num_layers=43, n_routed_experts=2)

    def test_stacked_shape_compat_fails_closed_on_extra_layer(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import StackedShapeCompatError, assert_stacked_shape_compatible, stacked_shape_compatibility_report

        meta = self._stacked_real_shape_meta()
        extra = self._stacked_real_shape_meta(num_layers=44)
        for key, value in extra.items():
            if key.startswith("layers.43."):
                meta[key] = value
        report = stacked_shape_compatibility_report(meta, expected_num_layers=43, n_routed_experts=2)

        self.assertFalse(report["ok"])
        self.assertEqual(report["extra_layers"], [43])
        self.assertTrue(any("extra layer" in blocker and "43" in blocker for blocker in report["blockers"]))
        with self.assertRaisesRegex(StackedShapeCompatError, "extra layer"):
            assert_stacked_shape_compatible(meta, expected_num_layers=43, n_routed_experts=2)

    def test_stacked_enumeration_validates_256_expert_indices_on_a_layer(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import stacked_shape_compatibility_report

        meta = self._stacked_real_shape_meta(num_layers=1, n_routed_experts=256)
        report = stacked_shape_compatibility_report(meta, expected_num_layers=1, n_routed_experts=256)
        self.assertTrue(report["ok"], report["blockers"])

        for proj in ("w1", "w2", "w3"):
            del meta[f"layers.0.ffn.experts.255.{proj}.weight"]
            del meta[f"layers.0.ffn.experts.255.{proj}.scale"]
        report = stacked_shape_compatibility_report(meta, expected_num_layers=1, n_routed_experts=256)
        self.assertFalse(report["ok"])
        self.assertTrue(any("255" in blocker for blocker in report["blockers"]))

    def test_stacked_shape_compat_report_is_deterministic(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import stacked_shape_compatibility_report

        first = stacked_shape_compatibility_report(self._stacked_real_shape_meta(), expected_num_layers=43, n_routed_experts=2)
        second = stacked_shape_compatibility_report(self._stacked_real_shape_meta(), expected_num_layers=43, n_routed_experts=2)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

    def test_real_config_default_modelargs_fails_while_relaxed_configs_construct(self):
        # Story 11.15g R2: block 1 still fails closed (bare ModelArgs() has
        # layer_types=None so gate #2 length guard raises, matching
        # "num_hidden_layers"). Blocks 2 & 3 were fail-closed on gates #11
        # and #10 respectively; both relaxed, so they now construct.
        # Paired parity: gate #11 -> ADR 0017 dequant + _moe_mlx top-k stub
        # (11.15d) + Q3 probe L2_REL=0.000e+00 at 256; gate #10 -> 11.11
        # hyperconnection synthetic. Witnesses in
        # tests/test_deepseek_v4_validate_real_mode_relaxation.py.
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, ModelArgs

        with self.assertRaisesRegex(NotImplementedError, "num_hidden_layers"):
            Model(ModelArgs())
        one_layer_many_experts = ModelArgs(num_hidden_layers=1, layer_types=["sliding_attention"], mlp_layer_types=["moe"], n_routed_experts=256, num_experts_per_tok=6, compression_ratio=0, num_key_value_heads=1, hc_mult=1)
        model = Model(one_layer_many_experts)
        self.assertEqual(model.args.n_routed_experts, 256)
        hcmult_multi = ModelArgs(num_hidden_layers=2, layer_types=["sliding_attention"] * 2, mlp_layer_types=["moe"] * 2, n_routed_experts=2, num_experts_per_tok=1, compression_ratio=0, num_key_value_heads=1, hc_mult=2)
        model = Model(hcmult_multi)
        self.assertEqual(model.args.num_hidden_layers, 2)
        self.assertEqual(model.args.hc_mult, 2)

    def test_shape_compat_never_decodes_and_markers_absent(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import stacked_shape_compatibility_report
        from ds4_ft_mlx.deepseek_v4_dequant import dequantize_expert_packed

        report = stacked_shape_compatibility_report(self._stacked_real_shape_meta(), expected_num_layers=43, n_routed_experts=2)
        self.assertTrue(report["ok"])
        with self.assertRaises(NotImplementedError):
            dequantize_expert_packed("i8", b"\x00", scales=b"\x7f", shape=(1,))
        with self.assertRaisesRegex(ValueError, "requires non-None scales"):
            dequantize_expert_packed("fp4", b"\x00" * 16, scales=None, shape=(1, 32))
        self.assertFalse(Path("/Volumes/Data NVME/mlx-ft/ds4/.deepseek-v4-forward-parity-ok").exists())
        self._assert_model_4bit_absent_or_valid_current_artifact()

    def test_bounded_real_expected_metadata_byte_identical_after_extract(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import bounded_real_expected_metadata

        expected = bounded_real_expected_metadata(expert_dtype="i8", n_routed_experts=2)
        self.assertEqual(expected["q_a_proj.weight"], {"dtype": "BF16", "shape": [1024, 4096]})
        self.assertEqual(expected["mlp.experts.1.w3.scale"], {"dtype": "BF16", "shape": [2048, 128]})
        self.assertEqual(len(expected), 34)

    def test_real_index_stacked_shape_compat_or_skip(self):
        from ds4_ft_mlx.deepseek_v4_checkpoint import stacked_shape_compatibility_from_index

        model_dir = Path("/Volumes/Data NVME/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/553034d7dd9e06c2eeaee68cf85a17d6d4754cf0")
        index = model_dir / "model.safetensors.index.json"
        if not index.exists():
            self.skipTest(f"real DeepSeek V4 Flash index not present at {index}")
        report = stacked_shape_compatibility_from_index(index, expected_num_layers=43, n_routed_experts=256)
        self.assertEqual(report["num_layers_observed"], 43)
        self.assertEqual(report["missing_layers"], [])
        self.assertEqual([len(template["signature"]) for template in report["templates"]], [35, 46, 39, 46])
        if not report["ok"]:
            self.assertTrue(report["blockers"])


if __name__ == "__main__":
    unittest.main()
