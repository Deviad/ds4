#!/usr/bin/env python3
import contextlib
import io
import json
import pathlib
import shutil
import tempfile
import unittest

from scripts import finetune_ds4


class Model4BitConversionPlanTests(unittest.TestCase):
    def write_marker(self, mlx: pathlib.Path, marker: str, gate: str, **extra: object) -> None:
        payload = {"schema": 1, "gate": gate, "status": "ok", **extra}
        (mlx / marker).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def make_args(self, tmp: pathlib.Path, mlx: pathlib.Path, *, out: pathlib.Path | None = None, report_only: bool = False):
        return type(
            "Args",
            (),
            {
                "hf_model": str(tmp / "hf"),
                "dataset_root": str(tmp / "dataset"),
                "mlx_work": str(mlx),
                "ds4_root": str(tmp / "ds4"),
                "ds4_gguf": None,
                "adapter_ds4": None,
                "split_dir": "mlx-4096",
                "fused_hf_model": None,
                "ds4_imatrix": None,
                "out": str(out) if out is not None else None,
                "report_only": report_only,
            },
        )()

    def build_workspace(self, tmp: pathlib.Path, *, mtp: bool = False) -> tuple[pathlib.Path, pathlib.Path]:
        mlx = tmp / "mlx"
        shim = mlx / "hf-f8shim"
        shim.mkdir(parents=True)
        weight_map = {"embed.weight": "s.safetensors"}
        if mtp:
            weight_map["mtp.0.hc_head_base"] = "s.safetensors"
        index_path = shim / "model.safetensors.index.json"
        index_path.write_text(json.dumps({"weight_map": weight_map}, sort_keys=True) + "\n", encoding="utf-8")
        index_sha = finetune_ds4.sha256_file(index_path)
        self.write_marker(mlx, ".deepseek-v4-import-ok", "deepseek-v4-import")
        self.write_marker(mlx, ".deepseek-v4-tiny-config-ok", "deepseek-v4-tiny-config")
        self.write_marker(mlx, ".deepseek-v4-mapping-ok", "deepseek-v4-mapping", index_sha256=index_sha)
        if mtp:
            self.write_marker(
                mlx,
                ".deepseek-v4-mtp-exclusion-ok",
                "deepseek-v4-mtp-exclusion",
                action="strip",
                index_sha256=index_sha,
                mtp_tensor_count=1,
            )
        self.write_marker(mlx, ".deepseek-v4-dequant-parity-ok", "deepseek-v4-dequant-parity")
        return mlx, index_path

    def load_plan(self, path: pathlib.Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    def without_generated_at(self, plan: dict[str, object]) -> dict[str, object]:
        return {key: value for key, value in plan.items() if key != "generated_at"}

    def test_success_plan_records_forward_absent_and_does_not_create_conversion_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            mlx, index_path = self.build_workspace(tmp)
            args = self.make_args(tmp, mlx)

            parsed = finetune_ds4.build_parser().parse_args(["model-4bit-conversion-plan", "--mlx-work", str(mlx)])
            self.assertIn("convert-shimmed", finetune_ds4.command_catalog(parsed))

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = finetune_ds4.model_4bit_conversion_plan(args)
            self.assertEqual(rc, 0)

            plan_path = mlx / "model-4bit-conversion-plan.json"
            self.assertTrue(plan_path.is_file())
            out = stdout.getvalue()
            self.assertIn(str(plan_path), out)
            self.assertIn("execution_allowed=false", out)
            # ADR 0022: forward-parity marker retired as gate; stdout blockers
            # no longer carry the gate-closed text. Plan still honest-reports
            # marker absent (marker_status below) and stays read-only.
            self.assertNotIn("forward-parity marker", out)
            self.assertNotIn("convert-shimmed execution gate intentionally closed", out)

            plan = self.load_plan(plan_path)
            catalog = finetune_ds4.command_catalog(args)
            self.assertEqual(plan["schema"], 1)
            self.assertEqual(plan["plan"], "model-4bit-conversion-plan")
            self.assertEqual(plan["source"], str((mlx / "hf-f8shim").resolve()))
            self.assertEqual(plan["destination"], str((mlx / "model-4bit").resolve()))
            self.assertEqual(plan["index_path"], str(index_path.resolve()))
            self.assertEqual(plan["index_sha256"], finetune_ds4.sha256_file(index_path))
            self.assertEqual(plan["mapping_index_sha256"], plan["index_sha256"])
            self.assertEqual(plan["mtp_count"], 0)
            self.assertIs(plan["destination_exists"], False)
            self.assertIs(plan["execution_allowed"], False)
            self.assertEqual(plan["convert_shimmed_command"], catalog["convert-shimmed"][0])
            self.assertEqual(plan["convert_shimmed_postcheck"], catalog["convert-shimmed"][1])
            self.assertEqual(plan["marker_status"], {
                ".deepseek-v4-import-ok": "present",
                ".deepseek-v4-tiny-config-ok": "present",
                ".deepseek-v4-mapping-ok": "present",
                ".deepseek-v4-mtp-exclusion-ok": "not-required",
                ".deepseek-v4-dequant-parity-ok": "present",
                ".deepseek-v4-forward-parity-ok": "absent",
            })
            self.assertIn("no mlx_lm.convert executed", plan["non_claims"])
            self.assertFalse((mlx / ".deepseek-v4-forward-parity-ok").exists())
            self.assertFalse((mlx / "model-4bit").exists())
            with self.assertRaises(finetune_ds4.PlanError):
                finetune_ds4._load_gate_marker(mlx, "model-4bit-conversion-plan.json", "deepseek-v4-forward-parity")

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(finetune_ds4.model_4bit_conversion_plan(args), 0)
            plan2 = self.load_plan(plan_path)
            self.assertEqual(self.without_generated_at(plan), self.without_generated_at(plan2))

    def test_missing_or_stale_inputs_fail_closed_without_plan(self):
        def run_case(mutator, pattern: str, *, mtp: bool = False) -> None:
            with tempfile.TemporaryDirectory() as td:
                tmp = pathlib.Path(td)
                mlx, index_path = self.build_workspace(tmp, mtp=mtp)
                mutator(mlx, index_path)
                args = self.make_args(tmp, mlx)
                with self.assertRaisesRegex(finetune_ds4.PlanError, pattern):
                    finetune_ds4.model_4bit_conversion_plan(args)
                self.assertFalse((mlx / "model-4bit-conversion-plan.json").exists())

        run_case(lambda mlx, index: shutil.rmtree(mlx / "hf-f8shim"), "hf-f8shim.*missing shimmed checkpoint directory")
        run_case(lambda mlx, index: index.unlink(), "missing shimmed checkpoint index")
        run_case(
            lambda mlx, index: index.write_text(json.dumps({"weight_map": {"embed.weight": "s.safetensors", "layers.0.extra": "s.safetensors"}}, sort_keys=True), encoding="utf-8"),
            "stale DeepSeek V4 mapping gate marker",
        )
        for marker in (
            ".deepseek-v4-import-ok",
            ".deepseek-v4-tiny-config-ok",
            ".deepseek-v4-mapping-ok",
            ".deepseek-v4-dequant-parity-ok",
        ):
            run_case(lambda mlx, index, marker=marker: (mlx / marker).unlink(), marker.replace(".", "\\."))
        run_case(lambda mlx, index: (mlx / ".deepseek-v4-mtp-exclusion-ok").unlink(), "deepseek-v4-mtp-exclusion", mtp=True)
        run_case(lambda mlx, index: (mlx / ".deepseek-v4-import-ok").write_text(json.dumps({"schema": 1, "gate": "wrong", "status": "ok"}), encoding="utf-8"), "invalid DeepSeek V4 gate marker")

    def test_stale_destination_fails_by_default_and_report_only_records_blocker(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            mlx, _index_path = self.build_workspace(tmp)
            (mlx / "model-4bit").mkdir()
            args = self.make_args(tmp, mlx)
            with self.assertRaisesRegex(finetune_ds4.PlanError, "stale conversion output present"):
                finetune_ds4.model_4bit_conversion_plan(args)
            plan_path = mlx / "model-4bit-conversion-plan.json"
            self.assertFalse(plan_path.exists())

            report_args = self.make_args(tmp, mlx, report_only=True)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(finetune_ds4.model_4bit_conversion_plan(report_args), 0)
            plan = self.load_plan(plan_path)
            self.assertIs(plan["destination_exists"], True)
            self.assertIs(plan["execution_allowed"], False)
            self.assertTrue(any("stale conversion output present" in blocker for blocker in plan["blockers"]))

    def test_out_path_rejects_protected_marker_paths(self):
        protected = (
            ".deepseek-v4-forward-parity-ok",
            ".deepseek-test-ok",
        )
        for name in protected:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                tmp = pathlib.Path(td)
                mlx, _index_path = self.build_workspace(tmp)
                out_path = mlx / name
                args = self.make_args(tmp, mlx, out=out_path)
                with self.assertRaisesRegex(finetune_ds4.PlanError, "protected"):
                    finetune_ds4.model_4bit_conversion_plan(args)
                self.assertFalse(out_path.exists())
                self.assertFalse((mlx / ".deepseek-v4-forward-parity-ok").exists())
                self.assertFalse((mlx / "model-4bit").exists())
                self.assertFalse((mlx / "model-4bit-conversion-plan.json").exists())

    def test_out_path_rejects_protected_model_4bit_path_when_absent_file_or_dir(self):
        for existing in ("absent", "file", "dir"):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as td:
                tmp = pathlib.Path(td)
                mlx, _index_path = self.build_workspace(tmp)
                out_path = mlx / "model-4bit"
                if existing == "file":
                    out_path.write_text("stale", encoding="utf-8")
                elif existing == "dir":
                    out_path.mkdir()
                args = self.make_args(tmp, mlx, out=out_path, report_only=True)
                with self.assertRaisesRegex(finetune_ds4.PlanError, "protected"):
                    finetune_ds4.model_4bit_conversion_plan(args)
                if existing == "absent":
                    self.assertFalse(out_path.exists())
                elif existing == "file":
                    self.assertTrue(out_path.is_file())
                    self.assertEqual(out_path.read_text(encoding="utf-8"), "stale")
                else:
                    self.assertTrue(out_path.is_dir())
                self.assertFalse((mlx / ".deepseek-v4-forward-parity-ok").exists())
                self.assertFalse((mlx / "model-4bit-conversion-plan.json").exists())

    def test_convert_shimmed_forward_parity_retired_but_marker_validator_intact(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            mlx, index_path = self.build_workspace(tmp)
            with self.assertRaisesRegex(finetune_ds4.PlanError, "deepseek-v4-forward-parity"):
                finetune_ds4._validate_forward_parity_marker(mlx)
            # ADR 0022 §Decision 3: architecture-gate no longer raises on absent
            # forward-parity marker (retired); marker validator STILL raises (ADR 0023).
            finetune_ds4.validate_deepseek_v4_architecture_gates(mlx)
            core = finetune_ds4.validate_deepseek_v4_architecture_gates_no_forward_parity(mlx)
            self.assertEqual(core["index_path"], index_path)
            self.assertEqual(core["index_sha256"], finetune_ds4.sha256_file(index_path))
            self.assertEqual(core["mtp_count"], 0)


if __name__ == "__main__":
    unittest.main()
