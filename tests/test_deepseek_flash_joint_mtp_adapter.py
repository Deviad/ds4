#!/usr/bin/env python3
"""Safe adapter contract and real-component dry-run rehearsal."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from deepseek_flash_joint_mtp.adapters import (  # noqa: E402
    AdapterUnavailable,
    RunContext,
    TrainJointAdapter,
)
from deepseek_flash_joint_mtp.canonical import identity_dict, sha256_file  # noqa: E402
from deepseek_flash_joint_mtp.config import load_config  # noqa: E402
from deepseek_flash_joint_mtp.orchestrator import dry_run, _preview_context  # noqa: E402
from deepseek_flash_joint_mtp.runner import run_process  # noqa: E402
from deepseek_flash_joint_mtp.stages import STAGE_BY_NAME  # noqa: E402


COMPONENT = Path("/Users/spotted/projects/ds4-finetuning")
PINNED_INTERPRETER = Path("/Volumes/Data NVME/mlx-ft/ds4/.venv/bin/python")
PINNED_PYTHON = PINNED_INTERPRETER.resolve()
SOURCE_IDENTITY = {
    "source_mode": "release",
    "mlx": "0.31.2",
    "mlx_lm": "0.31.3",
    "mlx_lm_module": "/Volumes/Data NVME/mlx-ft/ds4/.venv/lib/python3.13/site-packages/mlx_lm/__init__.py",
}
STAGED_HASH = "30635b80ae2632ec6baa421b4ce58914342c5e43c5d12130d1ae699334b8e072"


class AdapterFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="deepseek-flash-mlx-adapter-")
        self.root = Path(self.temp.name).resolve()
        self.fixture = self.root / "fixture"
        self.fixture.mkdir()
        self.writable = self.fixture / "writable"
        self.writable.mkdir()
        self.protected = self.fixture / "protected"
        self.protected.mkdir()
        self.reference = self.protected / "reference.gguf"
        self.reference.write_text("metadata-only\n")
        self.source = self.fixture / "source.json"
        self.source.write_text("{}\n")
        self.dataset = self.fixture / "dataset.json"
        self.dataset.write_text('{"train":"train.jsonl"}\n')
        self.model = self.fixture / "joint-base"
        (self.model / "inference").mkdir(parents=True)
        (self.model / "inference/config.json").write_text(json.dumps({
            "n_mtp_layers": 1,
            "dspark_block_size": 5,
            "dspark_target_layer_ids": [0],
        }) + "\n")
        (self.model / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {
                "layers.0.attn.wq_a.weight": "fake-main.safetensors",
                "layers.0.attn.wq_b.weight": "fake-main.safetensors",
                "layers.0.attn.wkv.weight": "fake-main.safetensors",
                "mtp.0.attn.wq_a.weight": "fake-mtp.safetensors",
                "mtp.0.attn.wq_b.weight": "fake-mtp.safetensors",
                "mtp.0.attn.wkv.weight": "fake-mtp.safetensors",
            }
        }) + "\n")
        self.component_config = self.fixture / "joint-config.json"
        self.component_output = self.writable / "component-output"
        self.component_report = self.component_output / "joint-report.json"
        self.config_path = self.fixture / "pipeline.json"
        self._write_pipeline()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_pipeline(self, **trainer_changes: object) -> None:
        self.component_config.write_text(json.dumps({
            "model": str(self.model),
            "data": str(self.dataset),
            "main_targets": None,
            "mtp_targets": None,
            "lambda_mtp": 0.5,
            "rank": 2,
            "scale": 5.0,
            "dropout": 0.0,
            "seed": 7,
            "batch_size": 1,
            "seq_len": 8,
            "grad_accumulation_steps": 1,
            "learning_rate": 0.00005,
            "iters": 400,
            "checkpoint_interval": 100,
            "resume": None,
            "source_mode": "release",
            "output_dir": str(self.component_output),
        }, sort_keys=True) + "\n")
        tools_dir = self.fixture / "tools"
        tools_dir.mkdir(exist_ok=True)
        fake = REPO / "tests/fixtures/deepseek_flash_joint_mtp/fake_stage.py"
        sentinel = REPO / "tests/fixtures/deepseek_flash_joint_mtp/must_not_run.py"
        trainer = {
            "executable": str(COMPONENT / "python-envs/mlx/src/ds4_ft_mlx/joint_train.py"),
            "version": "joint_train-v1",
            "component_repo": str(COMPONENT),
            "interpreter": str(PINNED_INTERPRETER),
            "interpreter_sha256": identity_dict(PINNED_PYTHON)["sha256"],
            "config": str(self.component_config),
            "config_sha256": identity_dict(self.component_config)["sha256"],
            "output_dir": str(self.component_output),
            "report_path": str(self.component_report),
            "report_schema_version": 1,
            "expected_branch": "ds4-finetuning",
            "expected_staged_diff_sha256": STAGED_HASH,
            "source_mode": "release",
            "expected_source_identity": SOURCE_IDENTITY,
            "environment": {},
            "resume": None,
            "rehearsal": True,
        }
        trainer.update(trainer_changes)
        values = {
            "schema_version": 1,
            "pipeline_version": "deepseek-flash-joint-mtp-slice1",
            "execution": {"adapter": "production-v1", "fixture": None},
            "paths": {
                "state_root": str(self.writable / "state"),
                "output_root": str(self.writable / "output"),
                "disposable_root": str(self.writable / "output/disposable"),
                "promotion_destination": str(self.writable / "promotion/final.gguf"),
                "approved_read_roots": [str(self.fixture), str(COMPONENT)],
                "approved_write_roots": [str(self.writable)],
                "protected_roots": [str(self.protected)],
            },
            "protected_artifacts": {key: str(self.reference) for key in (
                "final_quantized_reference", "frozen_finetuned_reference",
                "old_support_reference", "retained_clean_imatrix",
            )},
            "source": {
                "hf_source": str(self.source), "mlx_source": str(self.source),
                "dataset_manifest": str(self.dataset), "template_gguf": str(self.source),
                "source_tensor_index": str(self.source), "expected_sha256": {},
            },
            "model": {
                "identity": "rehearsal", "layer_count": 1, "hidden_size": 2,
                "vocabulary_size": 3, "mtp_layer_count": 1,
                "expected_mtp_tensors": [{"name": "mtp.0.fixture", "shape": [1], "dtype": "F32"}],
            },
            "training": {
                "seed": 7, "lambda_mtp": 0.5, "checkpoint_interval": 1,
                "dataset_split_manifest": str(self.dataset),
                "main_lora_allowlist": ["main.fixture"],
                "mtp_lora_allowlist": ["mtp.fixture"],
                "hyperparameters": {"learning_rate": 1, "batch_size": 1, "epochs": 1, "weight_decay": 0, "warmup_steps": 0},
            },
            "fusion": {"precision": "F32", "output_format": "fixture"},
            "imatrix": {"dataset": str(self.dataset), "collector_args": ["--tiny"], "repair_value": 0, "strict_consumption": True},
            "quantization": {
                "layers37_42_experts": "Q4_K", "lower_gate_up": "IQ2_XXS", "lower_down": "Q2_K",
                "attention": "Q8_0", "shared_experts": "Q8_0", "output_head": "Q8_0",
                "passthrough_rules": [{"name": "fixture", "dtype": "F32"}],
                "mtp_policy": [{"name": "mtp.0.fixture", "dtype": "F32"}],
            },
            "verification": {
                "correctness_prompts": ["tiny"], "performance_prompts": ["tiny"],
                "seed": 7, "max_tokens": 1, "acceptance_threshold": 0.6,
                "speed_threshold": 1.2, "warmup": 0, "repetitions": 1,
            },
            "tools": {
                "fixture-v1": {
                    "python": str(Path(sys.executable).resolve()),
                    "python_sha256": sha256_file(Path(sys.executable).resolve()),
                    "fake_stage": str(fake), "fake_stage_sha256": sha256_file(fake),
                    "must_not_run": str(sentinel), "must_not_run_sha256": sha256_file(sentinel),
                    "version": "fixture-v1",
                },
                "production-v1": {
                    "trainer": trainer,
                    **{name: {"executable": "deferred", "version": "deferred"} for name in (
                        "fusion", "converter", "collector", "repair", "quantizer", "inventory",
                        "engine", "correctness", "benchmark",
                    )},
                },
            },
            "environment": {stage: {} for stage in (
                "preflight", "prepare-joint-base", "train-joint", "fuse-joint",
                "export-fused-imatrix-source", "collect-imatrix", "repair-imatrix",
                "quantize-and-embed", "inventory-verify", "no-sidecar-load-verify",
                "correctness-verify", "performance-verify", "release-candidate",
            )},
        }
        self.config_path.write_text(json.dumps(values, sort_keys=True, indent=2) + "\n")

    def _adapter(self) -> tuple[TrainJointAdapter, object]:
        config = load_config(self.config_path)
        return TrainJointAdapter(config), config

    def test_dry_run_renders_direct_argv_and_is_read_only(self) -> None:
        result = dry_run(self.config_path)
        nodes = {node["name"]: node for node in result["nodes"]}
        node = nodes["train-joint"]
        self.assertTrue(node["launchable"])
        self.assertEqual(node["argv"][1:5], ["-m", "ds4_ft_mlx.joint_train", "--config", str(self.component_config)])
        self.assertEqual(node["argv"][-1], "--dry-run")
        self.assertNotIn("bash", node["argv"])
        self.assertNotIn("-c", node["argv"])
        self.assertFalse(self.component_report.exists())
        self.assertFalse((self.writable / "state").exists())

    def test_real_component_dry_run_rehearsal(self) -> None:
        adapter, config = self._adapter()
        context = _preview_context(config)
        attempt = self.writable / "attempt" / "0001"
        temporary = attempt / "temp" / "artifact.partial"
        final = self.writable / "output" / "artifact"
        launch = adapter.render(STAGE_BY_NAME["train-joint"], context, {}, attempt, temporary, final)
        self.assertEqual(launch.argv[-1], "--dry-run")
        result = run_process(launch, attempt, manifest_event=lambda *_args, **_kwargs: None)
        validation = adapter.validate(STAGE_BY_NAME["train-joint"], context, launch, result, temporary)
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["mode"], "dry-run")
        self.assertFalse(self.component_report.exists())
        self.assertEqual(list(self.model.rglob("*.safetensors")), [])
        self.assertFalse((self.component_output / "checkpoint-400").exists())
        events = [name for name in ("launch.json", "process-started.json", "stdout.log", "result.json") if (attempt / name).exists()]
        self.assertEqual(events, ["launch.json", "process-started.json", "stdout.log", "result.json"])

    def test_identity_drift_rejected_before_launch(self) -> None:
        mutations = {
            "expected_branch": "wrong-branch",
            "expected_staged_diff_sha256": "0" * 64,
            "interpreter_sha256": "sha256:" + "0" * 64,
            "config_sha256": "sha256:" + "0" * 64,
            "source_mode": "fork",
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                self._write_pipeline(**{field: value})
                adapter, config = self._adapter()
                with self.assertRaises(AdapterUnavailable):
                    adapter.render(STAGE_BY_NAME["train-joint"], _preview_context(config), {}, self.writable / "a", self.writable / "a/temp/x", self.writable / "out")

    def test_relative_unexpanded_and_stale_report_rejected(self) -> None:
        for field, value in (("config", "relative.json"), ("report_path", "$REPORT/joint-report.json"), ("component_repo", "~/repo")):
            with self.subTest(field=field):
                self._write_pipeline(**{field: value})
                with self.assertRaises(Exception):
                    self._adapter()[0].tool_identity()
        self._write_pipeline()
        self.component_report.parent.mkdir(parents=True, exist_ok=True)
        self.component_report.write_text("{}\n")
        adapter, config = self._adapter()
        with self.assertRaises(AdapterUnavailable):
            adapter.render(STAGE_BY_NAME["train-joint"], _preview_context(config), {}, self.writable / "a", self.writable / "a/temp/x", self.writable / "out")

    def test_production_report_validation_is_fail_closed(self) -> None:
        adapter, config = self._adapter()
        context = _preview_context(config)
        attempt = self.writable / "attempt"
        temporary = attempt / "temp" / "artifact.partial"
        launch = adapter.render(STAGE_BY_NAME["train-joint"], context, {}, attempt, temporary, self.writable / "out")
        result = type("Result", (), {"status": {"kind": "exit", "code": 0}})()
        launch = launch.__class__(
            stage=launch.stage, attempt_id=launch.attempt_id,
            argv=tuple(value for value in launch.argv if value != "--dry-run"),
            cwd=launch.cwd, environment=launch.environment,
            tool_identity=launch.tool_identity, input_identities=launch.input_identities,
            temporary_outputs=launch.temporary_outputs, final_outputs=launch.final_outputs,
        )
        for report in (
            {},
            {"report_schema_version": 1, "iters": 399},
            {"report_schema_version": 1, "iters": 400, "trainable_groups": {"main": {"selected_count": 1}, "mtp": {"selected_count": 0}}},
        ):
            with self.subTest(report=report):
                self.component_report.parent.mkdir(parents=True, exist_ok=True)
                self.component_report.write_text(json.dumps(report) + "\n")
                with self.assertRaises(ValueError):
                    adapter.validate(STAGE_BY_NAME["train-joint"], context, launch, result, temporary)
                self.component_report.unlink()

    def test_child_exit_propagates_without_report_acceptance(self) -> None:
        adapter, config = self._adapter()
        context = _preview_context(config)
        attempt = self.writable / "failed"
        temporary = attempt / "temp" / "artifact.partial"
        launch = adapter.render(STAGE_BY_NAME["train-joint"], context, {}, attempt, temporary, self.writable / "out")
        bad = launch.__class__(
            stage=launch.stage, attempt_id=launch.attempt_id,
            argv=(str(PINNED_INTERPRETER), "-c", "import sys; sys.exit(17)"), cwd=launch.cwd,
            environment=launch.environment, tool_identity=launch.tool_identity,
            input_identities=launch.input_identities, temporary_outputs=launch.temporary_outputs,
            final_outputs=launch.final_outputs,
        )
        result = run_process(bad, attempt, manifest_event=lambda *_args, **_kwargs: None)
        self.assertEqual(result.return_code, 17)
        with self.assertRaises(ValueError):
            adapter.validate(STAGE_BY_NAME["train-joint"], context, bad, result, temporary)


if __name__ == "__main__":
    unittest.main()
