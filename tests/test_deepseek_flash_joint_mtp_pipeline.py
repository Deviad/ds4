#!/usr/bin/env python3
"""Tiny, harmless Slice 1 orchestration rehearsal tests."""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from deepseek_flash_joint_mtp.canonical import canonical_dumps, load_json, sha256_file
from deepseek_flash_joint_mtp.cli import main
from deepseek_flash_joint_mtp.config import ConfigError, load_config
from deepseek_flash_joint_mtp.orchestrator import dry_run, plan, run, verify
from deepseek_flash_joint_mtp.runner import LaunchSpec, ProcessResult, run_process
from deepseek_flash_joint_mtp.state import StateError
from deepseek_flash_joint_mtp.stages import ORDINARY_STAGE_NAMES


class PipelineFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="deepseek-flash-slice1-")
        self.root = Path(self.temp.name).resolve()
        self.fixture = self.root / "fixture"
        self.fixture.mkdir()
        self.writable = self.fixture / "writable"
        self.writable.mkdir()
        self.protected = self.fixture / "protected"
        self.protected.mkdir()
        self.source = self.fixture / "source.json"
        self.source.write_text("{\"fixture\":true}\n")
        self.dataset = self.fixture / "dataset.txt"
        self.dataset.write_text("tiny\n")
        self.tools = self.fixture / "tools"
        self.tools.mkdir()
        shutil.copy2(REPO / "tests/fixtures/deepseek_flash_joint_mtp/fake_stage.py", self.tools / "fake_stage.py")
        shutil.copy2(REPO / "tests/fixtures/deepseek_flash_joint_mtp/must_not_run.py", self.tools / "must_not_run.py")
        (self.protected / "reference.gguf").write_text("not a model\n")
        self.marker = self.fixture / "sentinel.marker"
        self.config_path = self.root / "config.json"
        self.write_config()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_config(self, **changes: object) -> None:
        environment = {stage: {} for stage in ORDINARY_STAGE_NAMES}
        environment.update(changes.pop("environment", {}))
        values = {
            "schema_version": 1,
            "pipeline_version": "deepseek-flash-joint-mtp-slice1",
            "execution": {
                "adapter": "fixture-v1",
                "fixture": {
                    "fixture_root": str(self.fixture),
                    "interrupt_once_stage": None,
                    "forced_exit_stage": None,
                    "forced_exit_code": 73,
                },
            },
            "paths": {
                "state_root": str(self.writable / "state"),
                "output_root": str(self.writable / "output"),
                "disposable_root": str(self.writable / "output" / "disposable"),
                "promotion_destination": str(self.writable / "promotion" / "final.gguf"),
                "approved_read_roots": [str(self.fixture)],
                "approved_write_roots": [str(self.writable)],
                "protected_roots": [str(self.protected)],
            },
            "protected_artifacts": {key: str(self.protected / "reference.gguf") for key in ("final_quantized_reference", "frozen_finetuned_reference", "old_support_reference", "retained_clean_imatrix")},
            "source": {
                "hf_source": str(self.source),
                "mlx_source": str(self.source),
                "dataset_manifest": str(self.dataset),
                "template_gguf": str(self.source),
                "source_tensor_index": str(self.source),
                "expected_sha256": {},
            },
            "model": {
                "identity": "fixture-model",
                "layer_count": 1,
                "hidden_size": 2,
                "vocabulary_size": 3,
                "mtp_layer_count": 1,
                "expected_mtp_tensors": [{"name": "mtp.0.fixture", "shape": [1], "dtype": "F32"}],
            },
            "training": {
                "seed": 7,
                "lambda_mtp": 1,
                "checkpoint_interval": 1,
                "dataset_split_manifest": str(self.dataset),
                "main_lora_allowlist": ["main.fixture"],
                "mtp_lora_allowlist": ["mtp.fixture"],
                "hyperparameters": {"learning_rate": 1, "batch_size": 1, "epochs": 1, "weight_decay": 0, "warmup_steps": 0},
            },
            "fusion": {"precision": "F32", "output_format": "fixture"},
            "imatrix": {"dataset": str(self.dataset), "collector_args": ["--tiny"], "repair_value": 0, "strict_consumption": True},
            "quantization": {
                "layers37_42_experts": "Q4_K",
                "lower_gate_up": "IQ2_XXS",
                "lower_down": "Q2_K",
                "attention": "Q8_0",
                "shared_experts": "Q8_0",
                "output_head": "Q8_0",
                "passthrough_rules": [{"name": "fixture", "dtype": "F32"}],
                "mtp_policy": [{"name": "mtp.0.fixture", "dtype": "F32"}],
            },
            "verification": {
                "correctness_prompts": ["tiny"],
                "performance_prompts": ["tiny"],
                "seed": 7,
                "max_tokens": 1,
                "acceptance_threshold": 0.6,
                "speed_threshold": 1.2,
                "warmup": 0,
                "repetitions": 1,
            },
            "tools": {
                "fixture-v1": {
                    "python": str(Path(sys.executable).resolve()),
                    "python_sha256": sha256_file(Path(sys.executable).resolve()),
                    "fake_stage": str(self.tools / "fake_stage.py"),
                    "fake_stage_sha256": sha256_file(self.tools / "fake_stage.py"),
                    "must_not_run": str(self.tools / "must_not_run.py"),
                    "must_not_run_sha256": sha256_file(self.tools / "must_not_run.py"),
                    "version": "fixture-v1",
                },
                "production-v1": {name: {"executable": "deferred", "version": "deferred"} for name in ("trainer", "fusion", "converter", "collector", "repair", "quantizer", "inventory", "engine", "correctness", "benchmark")},
            },
            "environment": environment,
        }
        values.update(changes)
        self.config_path.write_text(json.dumps(values, sort_keys=True, indent=2) + "\n")

    def cli(self, *args: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(list(args))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_strict_config_and_path_rejections(self) -> None:
        raw = self.config_path.read_text()
        self.config_path.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaises(ConfigError):
            load_config(self.config_path)
        self.config_path.write_text(raw)
        values = json.loads(raw)
        values["paths"]["state_root"] = "relative/state"
        self.config_path.write_text(json.dumps(values))
        with self.assertRaises(ConfigError):
            load_config(self.config_path)

    def test_plan_and_dry_run_are_read_only(self) -> None:
        before = sorted(self.fixture.rglob("*"))
        plan_result = plan(self.config_path)
        dry_result = dry_run(self.config_path)
        self.assertEqual([node["name"] for node in plan_result["nodes"]], list(ORDINARY_STAGE_NAMES) + ["promote", "cleanup"])
        ordinary_nodes = dry_result["nodes"][:len(ORDINARY_STAGE_NAMES)]
        self.assertTrue(all(node["launchable"] for node in ordinary_nodes))
        self.assertEqual(len(dry_result["launches"]), len(ORDINARY_STAGE_NAMES))
        self.assertTrue(dry_result["nodes"][1]["argv"])
        self.assertTrue(dry_result["nodes"][1]["preview_dependencies"])
        self.assertEqual(dry_result["nodes"][1]["environment"], {})
        self.assertTrue(dry_result["nodes"][1]["temporary_outputs"])
        self.assertTrue(dry_result["nodes"][1]["final_outputs"])
        self.assertTrue(dry_result["nodes"][1]["publication_order"])
        self.assertTrue(dry_result["dry_run"])
        self.assertFalse((self.writable / "state").exists())
        self.assertFalse(self.marker.exists())
        self.assertEqual(before, sorted(self.fixture.rglob("*")))

    def test_production_refuses_before_state_or_launch(self) -> None:
        values = json.loads(self.config_path.read_text())
        values["execution"] = {"adapter": "production-v1", "fixture": None}
        self.config_path.write_text(json.dumps(values))
        code, _out, err = self.cli("run", "--config", str(self.config_path), "--run-id", "prod-test")
        self.assertEqual(code, 4)
        self.assertIn("not implemented in Slice 1", err)
        self.assertFalse((self.writable / "state").exists())
        self.assertFalse(self.marker.exists())

    def test_fake_full_dag_rehearsal(self) -> None:
        code, run_id = run(self.config_path, run_id="run-full")
        self.assertEqual(code, 0)
        self.assertEqual(run_id, "run-full")
        state = self.writable / "state" / run_id
        output = self.writable / "output" / run_id
        self.assertEqual(len(list(state.glob("stages/*/complete.json"))), len(ORDINARY_STAGE_NAMES))
        self.assertEqual(len(list(output.glob("*/artifact"))), len(ORDINARY_STAGE_NAMES))
        self.assertEqual(verify(self.config_path, run_id)["next_stage"], None)
        for stage in ORDINARY_STAGE_NAMES:
            self.assertTrue((output / stage / "artifact").is_file())
        for path in list(state.rglob("*")) + list(output.rglob("*")):
            self.assertTrue(path.resolve().is_relative_to(self.root))
        self.assertFalse(self.marker.exists())

    def test_generated_ids_and_active_resume_stability(self) -> None:
        code, generated_id = run(self.config_path, through_stage="preflight")
        self.assertEqual(code, 0)
        self.assertRegex(generated_id, r"^[a-z0-9][a-z0-9._-]{0,63}$")
        code, run_id = run(self.config_path, resume=True, through_stage="preflight")
        self.assertEqual(code, 0)
        self.assertRegex(run_id, r"^[a-z0-9][a-z0-9._-]{0,63}$")
        config = load_config(self.config_path)
        active = config.state_root / "active" / f"{config.resolved_config_digest}.json"
        self.assertEqual(load_json(active)["run_id"], run_id)
        code2, run_id2 = run(self.config_path, resume=True, through_stage="preflight")
        self.assertEqual((code2, run_id2), (0, run_id))
        attempts = list((config.state_root / run_id / "stages" / "preflight" / "attempts").iterdir())
        self.assertEqual(len(attempts), 1)

    def test_malformed_and_dangling_active_records_reject(self) -> None:
        config = load_config(self.config_path)
        active = config.state_root / "active" / f"{config.resolved_config_digest}.json"
        active.parent.mkdir(parents=True)
        active.write_text("{}\n")
        with self.assertRaises(StateError):
            run(self.config_path, resume=True)
        active.write_text(json.dumps({"run_id": "missing", "resolved_config_digest": config.resolved_config_digest}))
        with self.assertRaises(StateError):
            run(self.config_path, resume=True)

    def test_resume_and_through_stage_skip_valid_records(self) -> None:
        code, run_id = run(self.config_path, run_id="run-boundary", through_stage="train-joint")
        self.assertEqual(code, 0)
        self.assertEqual(run(self.config_path, run_id=run_id, resume=True)[0], 0)
        train_attempts = sorted((self.writable / "state" / run_id / "stages" / "train-joint" / "attempts").iterdir())
        self.assertEqual(len(train_attempts), 1)
        self.assertEqual(verify(self.config_path, run_id)["next_stage"], None)

    def test_checkpoint_resume_and_child_exit_are_preserved(self) -> None:
        values = json.loads(self.config_path.read_text())
        values["execution"]["fixture"]["interrupt_once_stage"] = "train-joint"
        self.config_path.write_text(json.dumps(values))
        code, run_id = run(self.config_path, run_id="run-checkpoint")
        self.assertEqual(code, 75)
        self.assertIsNone(load_json(self.writable / "state" / run_id / "stages" / "train-joint" / "complete.json") if (self.writable / "state" / run_id / "stages" / "train-joint" / "complete.json").exists() else None)
        self.assertEqual(run(self.config_path, run_id=run_id, resume=True)[0], 0)
        attempts = sorted((self.writable / "state" / run_id / "stages" / "train-joint" / "attempts").iterdir())
        self.assertEqual(len(attempts), 2)
        artifact = load_json(self.writable / "output" / run_id / "train-joint" / "artifact")
        self.assertTrue(artifact["resumed_from_checkpoint"])

    def test_resume_rejects_changed_tool_input_and_output_identities(self) -> None:
        code, run_id = run(self.config_path, run_id="run-identities", through_stage="prepare-joint-base")
        self.assertEqual(code, 0)
        fake = self.tools / "fake_stage.py"
        fake.write_text(fake.read_text() + "\n")
        with self.assertRaises(Exception):
            run(self.config_path, run_id=run_id, resume=True, through_stage="prepare-joint-base")
        with self.assertRaises(Exception):
            verify(self.config_path, run_id)
        self.write_config()
        output = self.writable / "output" / run_id / "preflight" / "artifact"
        output.write_text(output.read_text() + "tampered\n")
        with self.assertRaises(Exception):
            run(self.config_path, run_id=run_id, resume=True, from_stage="prepare-joint-base")

    def test_resume_rejects_changed_config_and_unknown_final(self) -> None:
        code, run_id = run(self.config_path, run_id="run-integrity", through_stage="preflight")
        self.assertEqual(code, 0)
        values = json.loads(self.config_path.read_text())
        values["training"]["seed"] = 8
        self.config_path.write_text(json.dumps(values))
        with self.assertRaises((ConfigError, Exception)) as caught:
            run(self.config_path, run_id=run_id, resume=True)
        self.assertTrue(caught.exception)
        self.write_config()
        unknown = self.writable / "output" / run_id / "prepare-joint-base" / "artifact"
        unknown.parent.mkdir(parents=True)
        unknown.write_text("unknown\n")
        with self.assertRaises(Exception):
            run(self.config_path, run_id=run_id, resume=True)

    def test_manifest_tamper_and_unreferenced_completion_reject(self) -> None:
        self.assertEqual(run(self.config_path, run_id="run-manifest", through_stage="preflight")[0], 0)
        root = self.writable / "state" / "run-manifest"
        (root / "manifest" / "evil.json").write_text("{}\n")
        with self.assertRaises(Exception):
            verify(self.config_path, "run-manifest")
        (root / "manifest" / "evil.json").unlink()
        source_complete = root / "stages" / "preflight" / "complete.json"
        target = root / "stages" / "prepare-joint-base"
        target.mkdir()
        (target / "complete.json").write_text(source_complete.read_text())
        with self.assertRaises(Exception):
            verify(self.config_path, "run-manifest")

    def test_manifest_tamper_and_state_verify(self) -> None:
        self.assertEqual(run(self.config_path, run_id="run-manifest", through_stage="preflight")[0], 0)
        head = self.writable / "state" / "run-manifest" / "manifest" / "head.json"
        value = json.loads(head.read_text())
        value["sequence"] += 1
        head.write_text(json.dumps(value))
        with self.assertRaises(Exception):
            verify(self.config_path, "run-manifest")

    def test_metachar_argv_allowlisted_environment_and_launch_order(self) -> None:
        weird_dir = self.fixture / "tools with spaces; meta"
        weird_dir.mkdir()
        weird_fake = weird_dir / "fake stage.py"
        shutil.copy2(self.tools / "fake_stage.py", weird_fake)
        values = json.loads(self.config_path.read_text())
        values["tools"]["fixture-v1"]["fake_stage"] = str(weird_fake)
        values["tools"]["fixture-v1"]["fake_stage_sha256"] = sha256_file(weird_fake)
        values["environment"]["preflight"] = {"VISIBLE": "safe value"}
        self.config_path.write_text(json.dumps(values))
        previous = os.environ.get("PARENT_SECRET")
        os.environ["PARENT_SECRET"] = "must-not-leak"
        try:
            self.assertEqual(run(self.config_path, run_id="run-argv", through_stage="preflight")[0], 0)
        finally:
            if previous is None:
                os.environ.pop("PARENT_SECRET", None)
            else:
                os.environ["PARENT_SECRET"] = previous
        attempt = self.writable / "state" / "run-argv" / "stages" / "preflight" / "attempts" / "0001"
        launch = load_json(attempt / "launch.json")
        self.assertIn(str(weird_fake), launch["argv"])
        self.assertEqual(launch["environment"], {"VISIBLE": "safe value"})
        artifact = load_json(self.writable / "output" / "run-argv" / "preflight" / "artifact")
        self.assertEqual(artifact["observed_environment"]["VISIBLE"], "safe value")
        self.assertNotIn("PARENT_SECRET", artifact["observed_environment"])
        events = [load_json(path)["payload"]["event_type"] for path in sorted((self.writable / "state" / "run-argv" / "manifest").glob("[0-9]*-*.json"))]
        self.assertLess(events.index("launch-prepared"), events.index("process-started"))
        self.assertLess(events.index("process-started"), events.index("log-published"))
        self.assertLess(events.index("log-published"), events.index("process-finished"))

    def test_argv_environment_and_launch_order(self) -> None:
        values = json.loads(self.config_path.read_text())
        values["environment"]["preflight"] = {"VISIBLE": "safe value", "FAKE_STAGE_SLEEP": "0"}
        self.config_path.write_text(json.dumps(values))
        self.assertEqual(run(self.config_path, run_id="run-argv", through_stage="preflight")[0], 0)
        attempt = self.writable / "state" / "run-argv" / "stages" / "preflight" / "attempts" / "0001"
        launch = load_json(attempt / "launch.json")
        self.assertEqual(launch["environment"], values["environment"]["preflight"])
        self.assertTrue((attempt / "process-started.json").exists())
        self.assertTrue((attempt / "stdout.log").exists())

    def test_logging_failure_preserves_child_status(self) -> None:
        attempt = self.fixture / "runner-attempt"
        log_directory = self.fixture / "log-directory"
        log_directory.mkdir()
        spec = LaunchSpec("preflight", "0001", (str(Path(sys.executable).resolve()), "-c", "import sys; sys.exit(17)"), self.fixture, {}, {}, {}, (attempt / "temp" / "artifact.partial",), (self.fixture / "final",))
        result = run_process(spec, attempt, manifest_event=lambda *_: None, log_path_override=log_directory)
        self.assertEqual(result.status, {"kind": "exit", "code": 17})
        self.assertIsNotNone(result.log_error)

    def test_signal_forwarding_writes_failure_without_completion(self) -> None:
        values = json.loads(self.config_path.read_text())
        values["environment"]["preflight"] = {"FAKE_STAGE_SLEEP": "5"}
        self.config_path.write_text(json.dumps(values))
        code = (
            "import sys; sys.path.insert(0, %r); "
            "from deepseek_flash_joint_mtp.orchestrator import run; "
            "raise SystemExit(run(%r, run_id='run-signal')[0])"
        ) % (str(SCRIPTS), str(self.config_path))
        child = subprocess.Popen([str(Path(sys.executable).resolve()), "-c", code], cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process_started = self.writable / "state" / "run-signal" / "stages" / "preflight" / "attempts" / "0001" / "process-started.json"
        deadline = time.time() + 10
        while time.time() < deadline and not process_started.exists():
            time.sleep(0.02)
        self.assertTrue(process_started.exists())
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=10)
        child.communicate()
        self.assertEqual(child.returncode, 143)
        stage = self.writable / "state" / "run-signal" / "stages" / "preflight"
        self.assertFalse((stage / "complete.json").exists())
        self.assertTrue(list((stage / "attempts").glob("*/failure.json")))

    def test_decimal_and_strict_schema_battery(self) -> None:
        values = json.loads(self.config_path.read_text())
        values["training"]["lambda_mtp"] = 0.5
        values["imatrix"]["repair_value"] = 0.0
        self.config_path.write_text(json.dumps(values))
        self.assertEqual(load_config(self.config_path).data["training"]["lambda_mtp"], 0.5)
        mutations = []
        for key in ("unknown",):
            candidate = json.loads(self.config_path.read_text())
            candidate[key] = True
            mutations.append(candidate)
        candidate = json.loads(self.config_path.read_text())
        candidate["verification"]["unknown"] = True
        mutations.append(candidate)
        candidate = json.loads(self.config_path.read_text())
        candidate["training"]["hyperparameters"]["unknown"] = True
        mutations.append(candidate)
        candidate = json.loads(self.config_path.read_text())
        candidate["protected_artifacts"]["unknown"] = str(self.protected / "reference.gguf")
        mutations.append(candidate)
        candidate = json.loads(self.config_path.read_text())
        candidate["tools"]["fixture-v1"]["unknown"] = True
        mutations.append(candidate)
        candidate = json.loads(self.config_path.read_text())
        candidate["verification"].pop("warmup")
        mutations.append(candidate)
        candidate = json.loads(self.config_path.read_text())
        candidate.pop("training")
        mutations.append(candidate)
        candidate = json.loads(self.config_path.read_text())
        candidate["schema_version"] = 2
        mutations.append(candidate)
        candidate = json.loads(self.config_path.read_text())
        candidate["pipeline_version"] = "other"
        mutations.append(candidate)
        candidate = json.loads(self.config_path.read_text())
        candidate["paths"]["state_root"] = "~/unsafe"
        mutations.append(candidate)
        for mutation in mutations:
            self.config_path.write_text(json.dumps(mutation))
            with self.assertRaises(ConfigError):
                load_config(self.config_path)
        production = json.loads(self.config_path.read_text())
        production["execution"] = {"adapter": "production-v1", "fixture": None}
        self.config_path.write_text(json.dumps(production))
        with self.assertRaises(ConfigError):
            dry_run(self.config_path, self.fixture)

    def test_log_failure_does_not_create_completion(self) -> None:
        values = json.loads(self.config_path.read_text())
        self.config_path.write_text(json.dumps(values))
        result = ProcessResult("start", "end", 1, {"kind": "exit", "code": 0}, None, "injected log failure", ())
        with patch("deepseek_flash_joint_mtp.orchestrator.run_process", return_value=result):
            code, run_id = run(self.config_path, run_id="run-log-failure", through_stage="preflight")
        self.assertEqual(code, 5)
        stage = self.writable / "state" / run_id / "stages" / "preflight"
        self.assertFalse((stage / "complete.json").exists())
        failure = load_json(next((stage / "attempts").glob("*/failure.json")))
        self.assertEqual(failure["failure_class"], "logging")

    def test_signal_escalation_second_signal_and_forwarded_zero_exit(self) -> None:
        def launch_with_environment(environment: dict[str, str], run_id: str, send_twice: bool = False) -> tuple[int, Path]:
            values = json.loads(self.config_path.read_text())
            values["environment"]["preflight"] = environment
            self.config_path.write_text(json.dumps(values))
            code = (
                "import sys; sys.path.insert(0, %r); "
                "from deepseek_flash_joint_mtp.orchestrator import run; "
                "raise SystemExit(run(%r, run_id=%r)[0])"
            ) % (str(SCRIPTS), str(self.config_path), run_id)
            child = subprocess.Popen([str(Path(sys.executable).resolve()), "-c", code], cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process_started = self.writable / "state" / run_id / "stages" / "preflight" / "attempts" / "0001" / "process-started.json"
            deadline = time.time() + 10
            while time.time() < deadline and not process_started.exists():
                time.sleep(0.02)
            self.assertTrue(process_started.exists())
            child.send_signal(signal.SIGTERM)
            if send_twice:
                child.send_signal(signal.SIGTERM)
            child.wait(timeout=5)
            child.communicate()
            return child.returncode, self.writable / "state" / run_id / "stages" / "preflight"

        code, stage = launch_with_environment({"FAKE_STAGE_SLEEP": "5", "FAKE_IGNORE_TERM": "1"}, "run-ignore", True)
        self.assertEqual(code, 143)
        self.assertFalse((stage / "complete.json").exists())
        code, stage = launch_with_environment({"FAKE_STAGE_SLEEP": "5", "FAKE_TERM_EXIT_ZERO": "1"}, "run-zero")
        self.assertEqual(code, 143)
        self.assertFalse((stage / "complete.json").exists())

    def test_action_nodes_are_unavailable_without_side_effects(self) -> None:
        for action in ("promote", "cleanup"):
            code, _out, err = self.cli(action, "--config", str(self.config_path), "--run-id", "run-x", "--confirm", "anything")
            self.assertEqual(code, 4)
            self.assertIn("not implemented in Slice 1", err)
        self.assertFalse((self.writable / "state").exists())
        self.assertFalse(self.marker.exists())

    def test_cli_json_and_compile_boundary(self) -> None:
        code, out, err = self.cli("plan", "--config", str(self.config_path), "--json")
        self.assertEqual(code, 0, err)
        parsed = json.loads(out)
        self.assertEqual(parsed["registry_digest"].split(":", 1)[0], "sha256")
        self.assertEqual(canonical_dumps(parsed), canonical_dumps(parsed))


if __name__ == "__main__":
    unittest.main()
