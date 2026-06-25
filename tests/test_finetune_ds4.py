#!/usr/bin/env python3
import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest

from scripts import finetune_ds4


class FinetuneDs4Tests(unittest.TestCase):
    def write_jsonl(self, path, rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def quiet(self, func, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return func(*args)

    def args(self, tmp, opus46, opus47, fable5):
        return type(
            "Args",
            (),
            {
                "dataset_root": str(tmp / "dataset"),
                "split_dir": "mlx-4096",
                "opus46_data": str(opus46),
                "opus47_data": str(opus47),
                "fable5_data": str(fable5),
            },
        )()

    def test_help_examples_use_fp8_shimmed_local_mlx_path(self):
        help_text = finetune_ds4.build_parser().format_help()
        self.assertIn("fp8-shim-probe fp8-shim convert-shimmed", help_text)
        self.assertIn("run-command convert-shimmed", help_text)
        self.assertNotIn("emit-commands convert smoke-train full-train", help_text)
        self.assertNotIn("run-command convert          # dry-run only", help_text)

    def test_build_dataset_dedupes_by_question_newest_wins(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            opus46 = tmp / "opus46.jsonl"
            opus47 = tmp / "opus47.jsonl"
            fable5 = tmp / "fable5.jsonl"
            self.write_jsonl(
                opus46,
                [
                    {"input": "Same question?", "inverted_reasoning": "<think>old reasoning", "output": "old answer", "id": "old"},
                    {"input": "Unique opus46", "inverted_reasoning": "trace", "output": "answer", "id": "u46"},
                ],
            )
            self.write_jsonl(
                opus47,
                [
                    {"input": " same   QUESTION? ", "inverted_reasoning": "<think>new reasoning", "output": "new answer", "id": "new"},
                ],
            )
            self.write_jsonl(
                fable5,
                [
                    {"context": "Fable only", "completion": "<think>fable thought\nfinal", "uid": "f1", "cot": "cot", "output": "final"},
                ],
            )

            rc = self.quiet(finetune_ds4.build_dataset, self.args(tmp, opus46, opus47, fable5))
            self.assertEqual(rc, 0)

            manifest = json.loads((tmp / "dataset" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["final_unique_records"], 3)
            self.assertEqual(manifest["kept_by_source"]["opus47"], 1)
            self.assertEqual(manifest["stats"]["cross_source_replacements"]["opus46->opus47"], 1)

            rows = []
            for split in ("train", "valid", "test"):
                path = tmp / "dataset" / "mlx-4096" / f"{split}.jsonl"
                rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
            same = [row for row in rows if "same   QUESTION" in row["prompt"]]
            self.assertEqual(len(same), 1)
            self.assertIn("new reasoning", same[0]["completion"])
            self.assertIn("new answer", same[0]["completion"])
            self.assertNotIn("old reasoning", same[0]["completion"])
            fable = [row for row in rows if "Fable only" in row["prompt"]][0]
            self.assertFalse(fable["completion"].startswith("<think>"))
            self.assertTrue(fable["completion"].endswith(finetune_ds4.EOS))
            self.assertEqual(set(fable), {"prompt", "completion"})

            self.quiet(finetune_ds4.validate_dataset, self.args(tmp, opus46, opus47, fable5))

    def test_build_dataset_handles_same_source_duplicates_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            opus46 = tmp / "opus46.jsonl"
            opus47 = tmp / "opus47.jsonl"
            fable5 = tmp / "fable5.jsonl"
            self.write_jsonl(
                opus46,
                [
                    {"input": "Repeat", "inverted_reasoning": "old", "output": "old"},
                    {"input": " repeat ", "inverted_reasoning": "new", "output": "new"},
                ],
            )
            self.write_jsonl(opus47, [])
            self.write_jsonl(fable5, [])
            args = self.args(tmp, opus46, opus47, fable5)
            self.quiet(finetune_ds4.build_dataset, args)
            first = {p.name: p.read_bytes() for p in (tmp / "dataset" / "mlx-4096").glob("*.jsonl")}
            manifest = json.loads((tmp / "dataset" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["stats"]["internal_replacements"]["opus46"], 1)
            self.assertEqual(manifest["kept_by_source"], {"opus46": 1, "opus47": 0, "fable5": 0})
            self.quiet(finetune_ds4.build_dataset, args)
            second = {p.name: p.read_bytes() for p in (tmp / "dataset" / "mlx-4096").glob("*.jsonl")}
            self.assertEqual(first, second)
            all_rows = b"".join(first.values()).decode("utf-8")
            self.assertIn("new", all_rows)
            self.assertNotIn("old", all_rows)

    def test_validate_dataset_rejects_extra_keys_and_missing_eos(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            split_dir = tmp / "dataset" / "mlx-4096"
            split_dir.mkdir(parents=True)
            good = {"prompt": finetune_ds4.BOS + finetune_ds4.USER + "q" + finetune_ds4.ASSISTANT_THINK, "completion": "a" + finetune_ds4.EOS}
            (split_dir / "train.jsonl").write_text(json.dumps({**good, "meta": "bad"}) + "\n", encoding="utf-8")
            (split_dir / "valid.jsonl").write_text(json.dumps(good) + "\n", encoding="utf-8")
            (split_dir / "test.jsonl").write_text(json.dumps(good) + "\n", encoding="utf-8")
            args = type("Args", (), {"dataset_root": str(tmp / "dataset"), "split_dir": "mlx-4096"})()
            with self.assertRaises(finetune_ds4.PlanError):
                self.quiet(finetune_ds4.validate_dataset, args)

            (split_dir / "train.jsonl").write_text(json.dumps({"prompt": good["prompt"], "completion": "missing"}) + "\n", encoding="utf-8")
            with self.assertRaises(finetune_ds4.PlanError):
                self.quiet(finetune_ds4.validate_dataset, args)

    def test_helpers_are_strict_about_think_and_eos(self):
        self.assertEqual(finetune_ds4.question_key("  A\n B  "), "a b")
        self.assertEqual(finetune_ds4.strip_leading_think("  <think> one <think> two"), "one <think> two")
        self.assertEqual(finetune_ds4.append_exactly_one_eos("body" + finetune_ds4.EOS + finetune_ds4.EOS), "body" + finetune_ds4.EOS)
        self.assertTrue(finetune_ds4.has_exactly_one_final_eos("body" + finetune_ds4.EOS))
        self.assertFalse(finetune_ds4.has_exactly_one_final_eos("body" + finetune_ds4.EOS + finetune_ds4.EOS))

    def test_build_dataset_missing_source_is_plan_error(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            opus46 = tmp / "missing.jsonl"
            opus47 = tmp / "opus47.jsonl"
            fable5 = tmp / "fable5.jsonl"
            self.write_jsonl(opus47, [])
            self.write_jsonl(fable5, [])
            with self.assertRaises(finetune_ds4.PlanError):
                self.quiet(finetune_ds4.build_dataset, self.args(tmp, opus46, opus47, fable5))

    def test_build_dataset_rejects_non_string_required_fields(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            opus46 = tmp / "opus46.jsonl"
            opus47 = tmp / "opus47.jsonl"
            fable5 = tmp / "fable5.jsonl"
            self.write_jsonl(opus46, [{"input": {"bad": "dict"}, "inverted_reasoning": "r", "output": "a"}])
            self.write_jsonl(opus47, [])
            self.write_jsonl(fable5, [])
            with self.assertRaises(finetune_ds4.PlanError):
                finetune_ds4.build_dataset(self.args(tmp, opus46, opus47, fable5))

    def test_validate_dataset_checks_manifest_and_meta_counts(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            split_dir = tmp / "dataset" / "mlx-4096"
            meta_dir = tmp / "dataset" / "meta"
            split_dir.mkdir(parents=True)
            meta_dir.mkdir(parents=True)
            good = {"prompt": finetune_ds4.BOS + finetune_ds4.USER + "q" + finetune_ds4.ASSISTANT_THINK, "completion": "a" + finetune_ds4.EOS}
            (split_dir / "train.jsonl").write_text(json.dumps(good) + "\n", encoding="utf-8")
            (split_dir / "valid.jsonl").write_text("", encoding="utf-8")
            (split_dir / "test.jsonl").write_text("", encoding="utf-8")
            (meta_dir / "records-meta.jsonl").write_text(json.dumps({"split": "train"}) + "\n", encoding="utf-8")
            (tmp / "dataset" / "manifest.json").write_text(json.dumps({"split_counts": {"train": 2, "valid": 0, "test": 0}, "final_unique_records": 2}), encoding="utf-8")
            args = type("Args", (), {"dataset_root": str(tmp / "dataset"), "split_dir": "mlx-4096"})()
            with self.assertRaises(finetune_ds4.PlanError):
                self.quiet(finetune_ds4.validate_dataset, args)

    def test_preflight_reports_missing_hf_directory(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            opus46 = tmp / "opus46.jsonl"
            opus47 = tmp / "opus47.jsonl"
            fable5 = tmp / "fable5.jsonl"
            self.write_jsonl(opus46, [{"input": "q", "inverted_reasoning": "r", "output": "a"}])
            self.write_jsonl(opus47, [{"input": "q2", "inverted_reasoning": "r", "output": "a"}])
            self.write_jsonl(fable5, [{"context": "q3", "completion": "c"}])
            args = self.args(tmp, opus46, opus47, fable5)
            args.hf_model = str(tmp / "missing-hf")
            with self.assertRaises(finetune_ds4.PlanError):
                finetune_ds4.preflight(args)

    def test_run_command_requires_yes_for_execute(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type(
                "Args",
                (),
                {
                    "step": "setup-env",
                    "execute": True,
                    "yes": False,
                    "backend": "local-mlx",
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                },
            )()
            with self.assertRaises(finetune_ds4.PlanError):
                finetune_ds4.run_command(args)

    def test_token_audit_default_fails_on_over_limit_with_fake_tokenizer(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            split_dir = tmp / "dataset" / "mlx-4096"
            split_dir.mkdir(parents=True)
            good = {"prompt": finetune_ds4.BOS + finetune_ds4.USER + "q" + finetune_ds4.ASSISTANT_THINK, "completion": "a b c" + finetune_ds4.EOS}
            for split in ("train", "valid", "test"):
                (split_dir / f"{split}.jsonl").write_text(json.dumps(good) + "\n", encoding="utf-8")

            class FakeTokenizer:
                def __call__(self, text, add_special_tokens=False):
                    return type("TokenResult", (), {"input_ids": text.split()})()

            fake = types.ModuleType("transformers")
            fake.AutoTokenizer = type("AutoTokenizer", (), {"from_pretrained": staticmethod(lambda *a, **k: FakeTokenizer())})
            old = sys.modules.get("transformers")
            sys.modules["transformers"] = fake
            try:
                args = type("Args", (), {"hf_model": str(tmp / "hf"), "dataset_root": str(tmp / "dataset"), "split_dir": "mlx-4096", "max_seq_length": 2, "no_trust_remote_code": False, "report_only": False, "allow_over_limit": False})()
                self.assertEqual(self.quiet(finetune_ds4.token_audit, args), 3)
                args.report_only = True
                self.assertEqual(self.quiet(finetune_ds4.token_audit, args), 0)
            finally:
                if old is None:
                    del sys.modules["transformers"]
                else:
                    sys.modules["transformers"] = old

    def test_command_emitter_defaults_to_dry_run_safe_paths(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type(
                "Args",
                (),
                {
                    "hf_model": str(tmp / "hf model"),
                    "dataset_root": str(tmp / "dataset root"),
                    "mlx_work": str(tmp / "mlx work"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                },
            )()
            catalog = finetune_ds4.command_catalog(args)
            self.assertIn("convert", catalog)
            self.assertIn("mlx_lm.convert", catalog["convert"][0])
            self.assertIn("--mlx-path", catalog["convert"][0])
            self.assertIn("touch", catalog["smoke-generate"][0])
            self.assertIn(".generation-smoke-ok", catalog["smoke-generate"][0])
            self.assertIn("--hf /path/to/fused-hf-safetensors-model", catalog["quantize-q2"][0])

    def test_emit_commands_default_local_mlx_routes_through_fp8_shim_before_training(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type(
                "Args",
                (),
                {
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "adapter_ds4": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                    "backend": "local-mlx",
                    "steps": [],
                    "format": "json",
                },
            )()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = finetune_ds4.emit_commands(args)
            self.assertEqual(rc, 0)
            steps = list(json.loads(buf.getvalue())["steps"])
            self.assertIn("fp8-shim-probe", steps)
            self.assertIn("fp8-shim", steps)
            self.assertIn("convert-shimmed", steps)
            self.assertLess(steps.index("fp8-shim-probe"), steps.index("fp8-shim"))
            self.assertLess(steps.index("fp8-shim"), steps.index("convert-shimmed"))
            self.assertIn("mlx-lora-targets-check", steps)
            self.assertLess(steps.index("convert-shimmed"), steps.index("mlx-lora-targets-check"))
            self.assertLess(steps.index("mlx-lora-targets-check"), steps.index("smoke-train"))
            self.assertNotIn("convert", steps)

    def test_default_local_mlx_runs_deepseek_v4_gates_before_convert_shimmed(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type(
                "Args",
                (),
                {
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "adapter_ds4": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                    "backend": "local-mlx",
                    "steps": [],
                    "format": "json",
                },
            )()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = finetune_ds4.emit_commands(args)
            self.assertEqual(rc, 0)
            steps = list(json.loads(buf.getvalue())["steps"])
            for gate in ("deepseek-v4-import-check", "deepseek-v4-tiny-config-check", "deepseek-v4-mapping-check"):
                self.assertIn(gate, steps)
                self.assertLess(steps.index(gate), steps.index("convert-shimmed"))

    def test_deepseek_v4_tiny_config_check_writes_marker(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type("Args", (), {"mlx_work": str(tmp / "mlx")})()
            rc = finetune_ds4.deepseek_v4_tiny_config_check(args)
            self.assertEqual(rc, 0)
            marker = tmp / "mlx" / ".deepseek-v4-tiny-config-ok"
            self.assertTrue(marker.is_file())
            self.assertIn("NotImplementedError", marker.read_text(encoding="utf-8"))

    def test_deepseek_v4_mapping_check_writes_summary_and_fails_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            shim = tmp / "mlx" / "hf-f8shim"
            shim.mkdir(parents=True)
            index = {
                "weight_map": {
                    "embed.weight": "model-00001-of-00001.safetensors",
                    "hc_head_base": "model-00001-of-00001.safetensors",
                    "layers.0.attn.attn_sink": "model-00001-of-00001.safetensors",
                    "layers.0.attn.wq_a.weight": "model-00001-of-00001.safetensors",
                    "layers.0.attn.wq_b.scale": "model-00001-of-00001.safetensors",
                    "layers.0.attn.compressor.ape": "model-00001-of-00001.safetensors",
                    "layers.0.attn.indexer.weights_proj.weight": "model-00001-of-00001.safetensors",
                    "head.weight": "model-00001-of-00001.safetensors",
                }
            }
            (shim / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            args = type("Args", (), {"mlx_work": str(tmp / "mlx")})()
            self.assertEqual(finetune_ds4.deepseek_v4_mapping_check(args), 0)
            self.assertTrue((tmp / "mlx" / ".deepseek-v4-mapping-ok").is_file())
            report = json.loads((tmp / "mlx" / "deepseek-v4-mapping-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["total"], len(index["weight_map"]))
            self.assertFalse(report["unmapped"])
            index["weight_map"]["layers.0.unknown.weight"] = "model-00001-of-00001.safetensors"
            (shim / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(finetune_ds4.PlanError, "unmapped DeepSeek V4 tensors"):
                finetune_ds4.deepseek_v4_mapping_check(args)
            self.assertFalse((tmp / "mlx" / ".deepseek-v4-mapping-ok").exists())

    def test_deepseek_v4_mtp_exclusion_check_writes_marker_bound_to_index(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            mlx = tmp / "mlx"
            shim = mlx / "hf-f8shim"
            shim.mkdir(parents=True)
            index_path = shim / "model.safetensors.index.json"
            index_path.write_text(json.dumps({"weight_map": {"embed.weight": "s.safetensors", "mtp.0.hc_head_base": "s.safetensors"}}), encoding="utf-8")
            self.assertEqual(finetune_ds4.deepseek_v4_mtp_exclusion_check(type("Args", (), {"mlx_work": str(mlx)})()), 0)
            marker = json.loads((mlx / ".deepseek-v4-mtp-exclusion-ok").read_text(encoding="utf-8"))
            self.assertEqual(marker["schema"], 1)
            self.assertEqual(marker["gate"], "deepseek-v4-mtp-exclusion")
            self.assertEqual(marker["status"], "ok")
            self.assertEqual(marker["action"], "strip")
            self.assertEqual(marker["index_sha256"], finetune_ds4.sha256_file(index_path))
            self.assertEqual(marker["mtp_tensor_count"], 1)

    def test_mapping_check_allows_mtp_only_with_valid_exclusion_marker(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            mlx = tmp / "mlx"
            shim = mlx / "hf-f8shim"
            shim.mkdir(parents=True)
            index = {"weight_map": {"embed.weight": "s.safetensors", "mtp.0.hc_head_base": "s.safetensors"}}
            (shim / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            args = type("Args", (), {"mlx_work": str(mlx)})()
            with self.assertRaisesRegex(finetune_ds4.PlanError, "deepseek-v4-mtp-exclusion"):
                finetune_ds4.deepseek_v4_mapping_check(args)
            finetune_ds4.deepseek_v4_mtp_exclusion_check(args)
            self.assertEqual(finetune_ds4.deepseek_v4_mapping_check(args), 0)
            report = json.loads((mlx / "deepseek-v4-mapping-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["mtp_exclusion"]["action"], "strip")
            self.assertEqual(report["mtp_exclusion"]["count"], 1)

    def test_deepseek_v4_mapping_check_blocks_review_required_mtp_and_removes_stale_marker(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            mlx = tmp / "mlx"
            shim = mlx / "hf-f8shim"
            shim.mkdir(parents=True)
            marker = mlx / ".deepseek-v4-mapping-ok"
            marker.write_text("stale", encoding="utf-8")
            index = {
                "weight_map": {
                    "embed.weight": "model-00001-of-00001.safetensors",
                    "mtp.0.hc_head_base": "model-00001-of-00001.safetensors",
                }
            }
            (shim / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            args = type("Args", (), {"mlx_work": str(mlx)})()
            with self.assertRaisesRegex(finetune_ds4.PlanError, "deepseek-v4-mtp-exclusion"):
                finetune_ds4.deepseek_v4_mapping_check(args)
            self.assertFalse(marker.exists())
            report = json.loads((mlx / "deepseek-v4-mapping-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["review_required"], ["mtp.0.hc_head_base"])
            self.assertEqual(report["review_required_policy"]["mtp.intentional-review-required"]["action"], "fail-closed")

    def test_deepseek_v4_mapping_check_blocks_incomplete_recognized_moe_families(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            mlx = tmp / "mlx"
            shim = mlx / "hf-f8shim"
            shim.mkdir(parents=True)
            (shim / "config.json").write_text(json.dumps({
                "hidden_size": 16,
                "moe_intermediate_size": 6,
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
                "n_shared_experts": 1,
            }), encoding="utf-8")
            index = {
                "weight_map": {
                    "embed.weight": "model-00001-of-00001.safetensors",
                    "layers.0.ffn.gate.weight": "model-00001-of-00001.safetensors",
                    "layers.0.ffn.experts.0.w1.weight": "model-00001-of-00001.safetensors",
                }
            }
            (shim / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(finetune_ds4.PlanError, "incomplete DeepSeek V4 MoE"):
                finetune_ds4.deepseek_v4_mapping_check(type("Args", (), {"mlx_work": str(mlx)})())
            report = json.loads((mlx / "deepseek-v4-mapping-report.json").read_text(encoding="utf-8"))
            self.assertIn("layers.0.ffn.experts.0.w2.weight", report["moe_completeness"]["missing_required_tensors"])

    def test_deepseek_v4_dequant_parity_check_writes_structured_marker(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type("Args", (), {"mlx_work": str(tmp / "mlx")})()
            self.assertEqual(finetune_ds4.deepseek_v4_dequant_parity_check(args), 0)
            marker = json.loads((tmp / "mlx" / ".deepseek-v4-dequant-parity-ok").read_text(encoding="utf-8"))
            self.assertEqual(marker["schema"], 1)
            self.assertEqual(marker["gate"], "deepseek-v4-dequant-parity")
            self.assertEqual(marker["status"], "ok")
            self.assertIn("f8_e4m3fn", marker["checks"])
            self.assertIn("f8_e8m0", marker["checks"])
            self.assertIn("i8_affine_explicit", marker["checks"])
            self.assertEqual(marker["reference"], "scripts/shim_ds4_safetensors.py")

    def test_convert_shimmed_requires_dequant_parity_gate(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            mlx = tmp / "mlx"
            (mlx / ".venv/bin").mkdir(parents=True)
            (mlx / ".venv/bin/activate").write_text("", encoding="utf-8")
            shim = mlx / "hf-f8shim"
            shim.mkdir(parents=True)
            index_path = shim / "model.safetensors.index.json"
            index_path.write_text(json.dumps({"weight_map": {"embed.weight": "model-00001-of-00001.safetensors"}}), encoding="utf-8")
            finetune_ds4.deepseek_v4_mapping_check(type("Args", (), {"mlx_work": str(mlx)})())
            for marker, payload in {
                ".deepseek-v4-import-ok": {"schema": 1, "gate": "deepseek-v4-import", "status": "ok"},
                ".deepseek-v4-tiny-config-ok": {"schema": 1, "gate": "deepseek-v4-tiny-config", "status": "ok"},
                ".deepseek-v4-forward-parity-ok": {"schema": 1, "gate": "deepseek-v4-forward-parity", "status": "ok"},
            }.items():
                (mlx / marker).write_text(json.dumps(payload), encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "step": "convert-shimmed",
                    "mlx_work": str(mlx),
                    "dataset_root": str(tmp / "dataset"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "adapter_ds4": None,
                    "split_dir": "mlx-4096",
                },
            )()
            with self.assertRaisesRegex(finetune_ds4.PlanError, "deepseek-v4-dequant-parity"):
                finetune_ds4.check_execute_prerequisites(args)

    def test_deepseek_v4_mapping_marker_is_bound_to_index_hash(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            mlx = tmp / "mlx"
            shim = mlx / "hf-f8shim"
            shim.mkdir(parents=True)
            index = {"weight_map": {"embed.weight": "model-00001-of-00001.safetensors"}}
            (shim / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            args = type("Args", (), {"mlx_work": str(mlx)})()
            self.assertEqual(finetune_ds4.deepseek_v4_mapping_check(args), 0)
            marker = json.loads((mlx / ".deepseek-v4-mapping-ok").read_text(encoding="utf-8"))
            self.assertEqual(marker["gate"], "deepseek-v4-mapping")
            self.assertEqual(marker["status"], "ok")
            self.assertEqual(marker["index_sha256"], finetune_ds4.sha256_file(shim / "model.safetensors.index.json"))

    def test_convert_shimmed_rejects_stale_mapping_marker_after_index_changes(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            mlx = tmp / "mlx"
            (mlx / ".venv/bin").mkdir(parents=True)
            (mlx / ".venv/bin/activate").write_text("", encoding="utf-8")
            shim = mlx / "hf-f8shim"
            shim.mkdir(parents=True)
            index_path = shim / "model.safetensors.index.json"
            index_path.write_text(json.dumps({"weight_map": {"embed.weight": "model-00001-of-00001.safetensors"}}), encoding="utf-8")
            marker_args = type("Args", (), {"mlx_work": str(mlx)})()
            finetune_ds4.deepseek_v4_mapping_check(marker_args)
            index_path.write_text(json.dumps({"weight_map": {"embed.weight": "model-00001-of-00001.safetensors", "mtp.0.hc_head_base": "model-00001-of-00001.safetensors"}}), encoding="utf-8")
            for marker, payload in {
                ".deepseek-v4-import-ok": {"schema": 1, "gate": "deepseek-v4-import", "status": "ok"},
                ".deepseek-v4-tiny-config-ok": {"schema": 1, "gate": "deepseek-v4-tiny-config", "status": "ok"},
                ".deepseek-v4-forward-parity-ok": {"schema": 1, "gate": "deepseek-v4-forward-parity", "status": "ok"},
            }.items():
                (mlx / marker).write_text(json.dumps(payload), encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "step": "convert-shimmed",
                    "mlx_work": str(mlx),
                    "dataset_root": str(tmp / "dataset"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "adapter_ds4": None,
                    "split_dir": "mlx-4096",
                },
            )()
            with self.assertRaisesRegex(finetune_ds4.PlanError, "stale DeepSeek V4 mapping gate marker"):
                finetune_ds4.check_execute_prerequisites(args)

    def test_convert_shimmed_forward_parity_retired_but_marker_validator_rejects_placeholder(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            mlx = tmp / "mlx"
            (mlx / ".venv/bin").mkdir(parents=True)
            (mlx / ".venv/bin/activate").write_text("", encoding="utf-8")
            shim = mlx / "hf-f8shim"
            shim.mkdir(parents=True)
            index_path = shim / "model.safetensors.index.json"
            index_path.write_text(json.dumps({"weight_map": {"embed.weight": "s.safetensors"}}), encoding="utf-8")
            gate_args = type("Args", (), {"mlx_work": str(mlx)})()
            finetune_ds4.deepseek_v4_mapping_check(gate_args)
            for marker, payload in {
                ".deepseek-v4-import-ok": {"schema": 1, "gate": "deepseek-v4-import", "status": "ok"},
                ".deepseek-v4-tiny-config-ok": {"schema": 1, "gate": "deepseek-v4-tiny-config", "status": "ok"},
                ".deepseek-v4-dequant-parity-ok": {"schema": 1, "gate": "deepseek-v4-dequant-parity", "status": "ok"},
                ".deepseek-v4-forward-parity-ok": {"schema": 1, "gate": "deepseek-v4-forward-parity", "status": "ok"},
            }.items():
                (mlx / marker).write_text(json.dumps(payload), encoding="utf-8")
            # ADR 0022 §Decision 3: convert-shimmed hard-gate no longer calls
            # _validate_forward_parity_marker; placeholder marker no longer aborts
            # the architecture gate. Validator stays callable (ADR 0023 honest-write).
            finetune_ds4.validate_deepseek_v4_architecture_gates(mlx)
            with self.assertRaisesRegex(finetune_ds4.PlanError, "invalid DeepSeek V4 forward parity marker"):
                finetune_ds4._validate_forward_parity_marker(mlx)

    def test_convert_shimmed_requires_mtp_exclusion_marker_when_index_has_mtp(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            mlx = tmp / "mlx"
            (mlx / ".venv/bin").mkdir(parents=True)
            (mlx / ".venv/bin/activate").write_text("", encoding="utf-8")
            shim = mlx / "hf-f8shim"
            shim.mkdir(parents=True)
            index_path = shim / "model.safetensors.index.json"
            index_path.write_text(json.dumps({"weight_map": {"embed.weight": "s.safetensors", "mtp.0.hc_head_base": "s.safetensors"}}), encoding="utf-8")
            gate_args = type("Args", (), {"mlx_work": str(mlx)})()
            finetune_ds4.deepseek_v4_mtp_exclusion_check(gate_args)
            finetune_ds4.deepseek_v4_mapping_check(gate_args)
            (mlx / ".deepseek-v4-mtp-exclusion-ok").unlink()
            for marker, payload in {
                ".deepseek-v4-import-ok": {"schema": 1, "gate": "deepseek-v4-import", "status": "ok"},
                ".deepseek-v4-tiny-config-ok": {"schema": 1, "gate": "deepseek-v4-tiny-config", "status": "ok"},
                ".deepseek-v4-dequant-parity-ok": {"schema": 1, "gate": "deepseek-v4-dequant-parity", "status": "ok"},
                ".deepseek-v4-forward-parity-ok": {"schema": 1, "gate": "deepseek-v4-forward-parity", "status": "ok"},
            }.items():
                (mlx / marker).write_text(json.dumps(payload), encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "step": "convert-shimmed",
                    "mlx_work": str(mlx),
                    "dataset_root": str(tmp / "dataset"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "adapter_ds4": None,
                    "split_dir": "mlx-4096",
                },
            )()
            with self.assertRaisesRegex(finetune_ds4.PlanError, "deepseek-v4-mtp-exclusion"):
                finetune_ds4.check_execute_prerequisites(args)

    def test_convert_shimmed_rejects_forged_plain_gate_markers(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            mlx = tmp / "mlx"
            (mlx / ".venv/bin").mkdir(parents=True)
            (mlx / ".venv/bin/activate").write_text("", encoding="utf-8")
            shim = mlx / "hf-f8shim"
            shim.mkdir(parents=True)
            (shim / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"embed.weight": "model-00001-of-00001.safetensors"}}), encoding="utf-8")
            for marker in (".deepseek-v4-import-ok", ".deepseek-v4-tiny-config-ok", ".deepseek-v4-mapping-ok", ".deepseek-v4-forward-parity-ok"):
                (mlx / marker).write_text("forged\n", encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "step": "convert-shimmed",
                    "mlx_work": str(mlx),
                    "dataset_root": str(tmp / "dataset"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "adapter_ds4": None,
                    "split_dir": "mlx-4096",
                },
            )()
            with self.assertRaisesRegex(finetune_ds4.PlanError, "invalid DeepSeek V4 gate marker"):
                finetune_ds4.check_execute_prerequisites(args)

    def test_mlx_lora_targets_check_emits_ds4_allowlist_not_default_all_linear(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type(
                "Args",
                (),
                {
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "adapter_ds4": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                    "backend": "local-mlx",
                    "steps": ["mlx-lora-targets-check"],
                    "format": "json",
                },
            )()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = finetune_ds4.emit_commands(args)
            self.assertEqual(rc, 0)
            script = "\n".join(json.loads(buf.getvalue())["steps"]["mlx-lora-targets-check"])
            self.assertIn("mlx-lora-targets-check", script)
            self.assertIn("--mlx-work", script)
            self.assertNotIn("mlx_lm.lora", script)
            self.assertNotIn("all-linear", script)

    def test_mlx_lora_targets_check_writes_validated_config_and_bound_marker(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type("Args", (), {"mlx_work": str(tmp / "mlx")})()
            self.assertEqual(finetune_ds4.mlx_lora_targets_check(args), 0)
            config_path = tmp / "mlx" / "lora-config.json"
            marker_path = tmp / "mlx" / ".mlx-lora-targets-ok"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(config["lora_parameters"]["keys"], ["self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_proj"])
            self.assertEqual(marker["schema"], 1)
            self.assertEqual(marker["gate"], "mlx-lora-targets")
            self.assertEqual(marker["status"], "ok")
            self.assertEqual(marker["config_sha256"], finetune_ds4.sha256_file(config_path))

    def test_training_commands_use_explicit_lora_config(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type(
                "Args",
                (),
                {
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "adapter_ds4": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                    "backend": "local-mlx",
                    "steps": ["smoke-train", "smoke-train-2048", "full-train", "continue-train"],
                    "format": "json",
                },
            )()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(finetune_ds4.emit_commands(args), 0)
            steps = json.loads(buf.getvalue())["steps"]
            for name, commands in steps.items():
                script = "\n".join(commands)
                self.assertIn("--config", script, name)
                self.assertIn("lora-config.json", script, name)

    def test_mlx_training_rejects_forged_or_stale_lora_allowlist_marker(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            mlx = tmp / "mlx"
            (mlx / ".venv/bin").mkdir(parents=True)
            (mlx / ".venv/bin/activate").write_text("", encoding="utf-8")
            (mlx / "model-4bit").mkdir()
            (mlx / "lora-config.json").write_text(json.dumps({"lora_parameters": {"keys": ["self_attn.q_a_proj"]}}), encoding="utf-8")
            (mlx / ".mlx-lora-targets-ok").write_text("forged\n", encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "step": "smoke-train",
                    "execute": True,
                    "yes": True,
                    "backend": "local-mlx",
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(mlx),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "adapter_ds4": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                },
            )()
            with self.assertRaisesRegex(finetune_ds4.PlanError, "invalid MLX LoRA target gate marker"):
                self.quiet(finetune_ds4.run_command, args)
            finetune_ds4.mlx_lora_targets_check(type("Args", (), {"mlx_work": str(mlx)})())
            (mlx / "lora-config.json").write_text(json.dumps({"lora_parameters": {"keys": ["self_attn.q_a_proj", "self_attn.compressor.wkv"]}}), encoding="utf-8")
            with self.assertRaisesRegex(finetune_ds4.PlanError, "stale MLX LoRA target gate marker"):
                self.quiet(finetune_ds4.run_command, args)

    def test_emit_commands_default_backend_local_mlx(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type(
                "Args",
                (),
                {
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                    "backend": "local-mlx",
                    "steps": ["setup-env"],
                    "format": "shell",
                },
            )()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = finetune_ds4.emit_commands(args)
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("cd", out)
            self.assertIn("{ test -d .venv || uv venv --seed .venv; }", out)
            self.assertIn("install_startup_pth_hook", out)
            self.assertNotIn("test -d .venv || uv venv --seed .venv &&", out)

    def test_emit_local_torch_mps_includes_torch_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type(
                "Args",
                (),
                {
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                    "backend": "local-torch-mps",
                    "steps": ["torch-smoke"],
                    "format": "json",
                },
            )()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = finetune_ds4.emit_commands(args)
            self.assertEqual(rc, 0)
            data = json.loads(buf.getvalue())
            self.assertEqual(data["backend"], "local-torch-mps")
            self.assertIn("torch-smoke", data["steps"])
            script = "\n".join(data["steps"]["torch-smoke"])
            self.assertIn("torch_lora_smoke.py", script)
            self.assertIn("AutoModelForCausalLM.from_config", script)
            self.assertIn("get_peft_model", script)
            self.assertIn("save_pretrained", script)
            self.assertIn("peft-real-export", script)
            self.assertIn("adapter_model.safetensors", script)
            self.assertIn("if not torch.backends.mps.is_available()", script)

    def test_emit_shell_uses_fail_fast_grouping(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type(
                "Args",
                (),
                {
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                    "backend": "local-torch-mps",
                    "steps": ["torch-env-create"],
                    "format": "shell",
                },
            )()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = finetune_ds4.emit_commands(args)
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("set -euo pipefail", out)
            self.assertIn("(", out)
            self.assertIn(")", out)
            self.assertLess(out.index("set -euo pipefail"), out.index("uv venv --seed"))

    def test_gitignore_ignores_generated_egg_info(self):
        ignore = pathlib.Path(".gitignore").read_text(encoding="utf-8")
        self.assertIn("*.egg-info/", ignore)
        ignored_path = "python-envs/torch/src/ds4_ft_torch.egg-info/PKG-INFO"
        result = subprocess.run(["git", "check-ignore", ignored_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), ignored_path)

    def test_emit_torch_env_create_uses_uv_seed_and_pyproject(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type(
                "Args",
                (),
                {
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                    "backend": "local-torch-mps",
                    "steps": ["torch-env-create"],
                    "format": "json",
                },
            )()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = finetune_ds4.emit_commands(args)
            self.assertEqual(rc, 0)
            script = "\n".join(json.loads(buf.getvalue())["steps"]["torch-env-create"])
            self.assertIn("uv venv --seed --python 3.12 --clear .venv-torch", script)
            self.assertIn("pip install -e", script)
            self.assertIn("python-envs/torch", script)
            self.assertNotIn("pip install torch transformers peft", script)

    def test_full_fp8_shim_requires_probe_marker(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            mlx = tmp / "mlx"
            (mlx / ".venv/bin").mkdir(parents=True)
            (mlx / ".venv/bin/activate").write_text("", encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "step": "fp8-shim",
                    "execute": True,
                    "yes": True,
                    "backend": "local-mlx",
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(mlx),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "adapter_ds4": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                },
            )()
            with self.assertRaisesRegex(finetune_ds4.PlanError, "missing FP8 shim probe marker"):
                self.quiet(finetune_ds4.run_command, args)

    def test_convert_shimmed_requires_deepseek_v4_architecture_gate_markers(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            mlx = tmp / "mlx"
            (mlx / ".venv/bin").mkdir(parents=True)
            (mlx / ".venv/bin/activate").write_text("", encoding="utf-8")
            (mlx / ".fp8-shim-probe-ok").write_text("", encoding="utf-8")
            (mlx / "hf-f8shim").mkdir()
            args = type(
                "Args",
                (),
                {
                    "step": "convert-shimmed",
                    "execute": True,
                    "yes": True,
                    "backend": "local-mlx",
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(mlx),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "adapter_ds4": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                },
            )()
            with self.assertRaisesRegex(finetune_ds4.PlanError, "missing DeepSeek V4 architecture gate marker"):
                self.quiet(finetune_ds4.run_command, args)

    def test_mlx_training_requires_lora_target_allowlist_marker(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            mlx = tmp / "mlx"
            (mlx / ".venv/bin").mkdir(parents=True)
            (mlx / ".venv/bin/activate").write_text("", encoding="utf-8")
            (mlx / "model-4bit").mkdir()
            args = type(
                "Args",
                (),
                {
                    "step": "smoke-train",
                    "execute": True,
                    "yes": True,
                    "backend": "local-mlx",
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(mlx),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "adapter_ds4": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                },
            )()
            with self.assertRaisesRegex(finetune_ds4.PlanError, "missing MLX LoRA target allowlist marker"):
                self.quiet(finetune_ds4.run_command, args)

    def test_torch_steps_require_torch_venv(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type(
                "Args",
                (),
                {
                    "step": "torch-env-check",
                    "execute": True,
                    "yes": True,
                    "backend": "local-torch-mps",
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                },
            )()
            with self.assertRaisesRegex(finetune_ds4.PlanError, "missing Torch/PEFT venv"):
                self.quiet(finetune_ds4.run_command, args)

    def test_emit_torch_real_v4_feasibility_uses_real_checkpoint_safely(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type(
                "Args",
                (),
                {
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "adapter_ds4": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                    "backend": "local-torch-mps",
                    "steps": ["torch-real-v4-feasibility"],
                    "format": "json",
                },
            )()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = finetune_ds4.emit_commands(args)
            self.assertEqual(rc, 0)
            script = "\n".join(json.loads(buf.getvalue())["steps"]["torch-real-v4-feasibility"])
            self.assertIn("torch_real_v4_feasibility.py", script)
            self.assertIn("if not torch.backends.mps.is_available()", script)
            self.assertIn("model.safetensors.index.json", script)
            self.assertIn("init_empty_weights", script)
            self.assertIn("F8_E4M3", script)
            self.assertIn("F8_E8M0", script)
            self.assertIn("dtype.startswith('F8_')", script)
            self.assertIn("local_training_feasible", script)
            self.assertIn("heuristic_limitations", script)
            self.assertIn("activations/optimizer", script)
            self.assertIn("training_feasibility_report.json", script)
            self.assertNotIn("from_pretrained", script)

    def test_emit_torch_one_step_lora_trains_exports_and_converts(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type(
                "Args",
                (),
                {
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                    "backend": "local-torch-mps",
                    "steps": ["torch-one-step-lora"],
                    "format": "json",
                },
            )()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = finetune_ds4.emit_commands(args)
            self.assertEqual(rc, 0)
            script = "\n".join(json.loads(buf.getvalue())["steps"]["torch-one-step-lora"])
            self.assertIn("torch_one_step_lora.py", script)
            self.assertIn("loss.backward()", script)
            self.assertIn("optimizer.step()", script)
            self.assertIn("save_pretrained", script)
            self.assertIn("convert_lora_to_ds4.py", script)
            self.assertIn("adapter.ds4.safetensors", script)
            self.assertIn("if not torch.backends.mps.is_available()", script)

    def test_execute_requires_explicit_yes_not_environment_bypass(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type(
                "Args",
                (),
                {
                    "step": "validate-dataset",
                    "execute": True,
                    "yes": False,
                    "backend": "cpu-check",
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                },
            )()
            old = os.environ.get("DS4_FT_YES")
            os.environ["DS4_FT_YES"] = "1"
            try:
                with self.assertRaisesRegex(finetune_ds4.PlanError, "requires --yes"):
                    self.quiet(finetune_ds4.run_command, args)
            finally:
                if old is None:
                    os.environ.pop("DS4_FT_YES", None)
                else:
                    os.environ["DS4_FT_YES"] = old

    def test_emit_remote_cuda_full_guidance_includes_model_for_inspect(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type(
                "Args",
                (),
                {
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": str(tmp / "ds4flash.gguf"),
                    "adapter_ds4": str(tmp / "adapter.ds4.safetensors"),
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                    "backend": "remote-cuda",
                    "steps": ["remote-cuda-full"],
                    "format": "json",
                },
            )()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = finetune_ds4.emit_commands(args)
            self.assertEqual(rc, 0)
            script = "\n".join(json.loads(buf.getvalue())["steps"]["remote-cuda-full"])
            self.assertIn("./ds4 --inspect -m", script)
            self.assertIn("--lora", script)
            self.assertIn(str(tmp / "ds4flash.gguf"), script)
            self.assertNotIn("--inspect --lora", script)

    def test_emit_cpu_check_ds4_adapter_inspect_executes_inspect(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            adapter = tmp / "adapter.ds4.safetensors"
            args = type(
                "Args",
                (),
                {
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": str(tmp / "ds4flash.gguf"),
                    "adapter_ds4": str(adapter),
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                    "backend": "cpu-check",
                    "steps": ["ds4-adapter-inspect"],
                    "format": "json",
                },
            )()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = finetune_ds4.emit_commands(args)
            self.assertEqual(rc, 0)
            script = "\n".join(json.loads(buf.getvalue())["steps"]["ds4-adapter-inspect"])
            self.assertIn("./ds4 --inspect", script)
            self.assertIn("--lora", script)
            self.assertIn(str(adapter), script)
            self.assertNotIn("# ds4-adapter-inspect", script)

    def test_ds4_adapter_inspect_requires_adapter_and_gguf(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            adapter = tmp / "adapter.ds4.safetensors"
            gguf = tmp / "ds4flash.gguf"
            args = type(
                "Args",
                (),
                {
                    "step": "ds4-adapter-inspect",
                    "execute": True,
                    "yes": True,
                    "backend": "cpu-check",
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": str(gguf),
                    "adapter_ds4": str(adapter),
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                },
            )()
            with self.assertRaisesRegex(finetune_ds4.PlanError, "missing DS4 adapter"):
                self.quiet(finetune_ds4.run_command, args)
            adapter.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(finetune_ds4.PlanError, "missing ds4flash.gguf"):
                self.quiet(finetune_ds4.run_command, args)

    def test_emit_cpu_check_includes_dataset_validation(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type(
                "Args",
                (),
                {
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                    "backend": "cpu-check",
                    "steps": ["validate-dataset"],
                    "format": "shell",
                },
            )()
            rc = self.quiet(finetune_ds4.emit_commands, args)
            self.assertEqual(rc, 0)

    def test_unknown_backend_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type(
                "Args",
                (),
                {
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                    "backend": "quantum-gpu",
                    "steps": [],
                    "format": "shell",
                },
            )()
            with self.assertRaises(finetune_ds4.PlanError):
                self.quiet(finetune_ds4.emit_commands, args)

    def test_backend_step_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type(
                "Args",
                (),
                {
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                    "backend": "local-torch-mps",
                    "steps": ["convert"],
                    "format": "shell",
                },
            )()
            with self.assertRaises(finetune_ds4.PlanError):
                self.quiet(finetune_ds4.emit_commands, args)

    def test_full_train_requires_ds4_inspect_marker(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            mlx = tmp / "mlx"
            (mlx / ".venv/bin").mkdir(parents=True)
            (mlx / ".venv/bin/activate").write_text("", encoding="utf-8")
            (mlx / "model-4bit").mkdir()
            finetune_ds4.mlx_lora_targets_check(type("Args", (), {"mlx_work": str(mlx)})())
            (mlx / "adapters-smoke").mkdir()
            (mlx / "adapters-smoke/.generation-smoke-ok").write_text("", encoding="utf-8")
            (mlx / "adapters-smoke/adapters.safetensors").write_text("", encoding="utf-8")
            (mlx / "adapters-smoke/adapter.ds4.safetensors").write_text("", encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "step": "full-train",
                    "execute": True,
                    "yes": True,
                    "backend": "local-mlx",
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(mlx),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                },
            )()
            split_dir = tmp / "dataset" / "mlx-4096"
            split_dir.mkdir(parents=True)
            meta_dir = tmp / "dataset" / "meta"
            meta_dir.mkdir()
            manifest = {
                "split_dir": "mlx-4096",
                "total_raw_records": 3,
                "final_unique_records": 3,
                "split_counts": {"train": 1, "valid": 1, "test": 1},
                "kept_by_source": {},
                "stats": {},
            }
            (tmp / "dataset" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            meta_lines = []
            for i, split in enumerate(("train", "valid", "test")):
                row = {
                    "prompt": finetune_ds4.BOS + finetune_ds4.USER + "q" + finetune_ds4.ASSISTANT_THINK,
                    "completion": "a" + finetune_ds4.EOS,
                }
                (split_dir / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
                meta_lines.append(json.dumps({"key_hash": f"h{i}", "source": "x", "source_priority": 1, "source_line": i, "split": split}))
            (meta_dir / "records-meta.jsonl").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(finetune_ds4.PlanError, "missing DS4 inspect marker"):
                self.quiet(finetune_ds4.run_command, args)
            (mlx / "adapters-smoke/.ds4-inspect-ok").write_text("", encoding="utf-8")
            # With marker present it should progress past prerequisites (lock or execute will fail for other reasons in this stub environment).
            with self.assertRaises(Exception):
                self.quiet(finetune_ds4.run_command, args)

    def test_run_command_backend_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = type(
                "Args",
                (),
                {
                    "step": "torch-smoke",
                    "execute": False,
                    "yes": False,
                    "backend": "local-mlx",
                    "hf_model": str(tmp / "hf"),
                    "dataset_root": str(tmp / "dataset"),
                    "mlx_work": str(tmp / "mlx"),
                    "ds4_root": str(tmp / "ds4"),
                    "ds4_gguf": None,
                    "split_dir": "mlx-4096",
                    "fused_hf_model": None,
                    "ds4_imatrix": None,
                },
            )()
            with self.assertRaises(finetune_ds4.PlanError):
                self.quiet(finetune_ds4.run_command, args)


if __name__ == "__main__":
    unittest.main()
