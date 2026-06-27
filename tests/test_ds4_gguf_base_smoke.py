#!/usr/bin/env python3
"""Gated DS4-GGUF base smoke test.

This module is serial and skip-by-default: the real test loads the tens-of-GB
`ds4flash.gguf` through the production Metal runtime and must be opted into with
`DS4_GGUF_BASE_SMOKE=1`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_SRC = REPO_ROOT / "python-envs" / "mlx" / "src"
PINNED_DS4_ROOT = Path("/Users/spotted/projects/ds4")
TRACK_A_MARKER = ".ds4-gguf-generate-ok"
TRACK_A_REPORT = "ds4-gguf-base-smoke.json"
TRACK_B_MARKER = Path("/Volumes/Data NVME/mlx-ft/ds4/.deepseek-v4-forward-parity-ok")
TRACK_B_MODEL_4BIT = Path("/Volumes/Data NVME/mlx-ft/ds4/model-4bit")


class DS4GGUFBaseSmokeTests(unittest.TestCase):
    def setUp(self):
        self._old_path = list(sys.path)
        sys.path.insert(0, str(REPO_ROOT))
        sys.path.insert(0, str(MLX_SRC))
        self._old_env = dict(os.environ)

    def tearDown(self):
        sys.path[:] = self._old_path
        os.environ.clear()
        os.environ.update(self._old_env)

    @staticmethod
    def _args(root: Path, *, gguf: Path | None = None, mlx_work: Path | None = None):
        return type(
            "Args",
            (),
            {
                "ds4_root": str(root),
                "ds4_gguf": str(gguf) if gguf is not None else None,
                "mlx_work": str(mlx_work or (root / "mlx-work")),
                "ds4_smoke_prompt": "Answer in one word: ready?",
                "ds4_smoke_tokens": 12,
                "ds4_smoke_timeout": 30,
            },
        )()

    @staticmethod
    def _write_fake_artifacts(root: Path) -> tuple[Path, Path]:
        binary = root / "ds4"
        binary.write_text("#!/bin/sh\necho fake ds4\n", encoding="utf-8")
        binary.chmod(0o755)
        gguf = root / "ds4flash.gguf"
        gguf.write_bytes(b"GGUF" + bytes(range(64)))
        return binary, gguf

    def _enable_gate(self):
        os.environ["DS4_GGUF_BASE_SMOKE"] = "1"

    def _assert_track_b_model_4bit_absent_or_valid_current_artifact(self):
        if not TRACK_B_MODEL_4BIT.exists():
            return
        cfg = json.loads((TRACK_B_MODEL_4BIT / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg.get("model_type"), "deepseek_v4_nn")
        self.assertEqual(cfg.get("quantization", {}).get("bits"), 4)

    def test_skips_without_env_gate(self):
        if os.environ.get("DS4_GGUF_BASE_SMOKE") == "1":
            self.skipTest("env gate is set; default skip behavior is only meaningful without it")
        self.skipTest("DS4_GGUF_BASE_SMOKE=1 not set; skip is not proof and writes no marker")

    def test_skips_when_binary_or_gguf_missing(self):
        self._enable_gate()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            marker = root / TRACK_A_MARKER
            self.assertFalse(marker.exists())
            if not (root / "ds4").is_file() or not (root / "ds4flash.gguf").is_file():
                self.skipTest(f"real ds4 artifacts not present at {root}; skip is not proof and writes no marker")
            self.fail("temporary artifact directory unexpectedly contained ds4 and ds4flash.gguf")

    def test_helper_clears_stale_marker_then_fails_closed(self):
        from scripts import finetune_ds4

        self._enable_gate()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            marker = root / TRACK_A_MARKER
            marker.write_text("stale\n", encoding="utf-8")
            args = self._args(root)
            with self.assertRaises(finetune_ds4.PlanError):
                finetune_ds4.ds4_gguf_base_smoke_check(args)
            self.assertFalse(marker.exists())
            self.assertFalse((root / TRACK_A_REPORT).exists())

    def test_helper_raises_on_nonzero_or_empty_output(self):
        from scripts import finetune_ds4

        for label, completed in (
            ("nonzero", subprocess.CompletedProcess(["ds4"], 17, stdout="", stderr="boom")),
            ("empty", subprocess.CompletedProcess(["ds4"], 0, stdout="Answer in one word: ready?\n   ", stderr="")),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as td:
                self._enable_gate()
                root = Path(td)
                self._write_fake_artifacts(root)
                calls: list[list[str]] = []

                def fake_run(cmd, *unused_args, **unused_kwargs):
                    calls.append([str(part) for part in cmd])
                    return completed

                with patch.object(finetune_ds4.subprocess, "run", side_effect=fake_run):
                    with self.assertRaises(finetune_ds4.PlanError):
                        finetune_ds4.ds4_gguf_base_smoke_check(self._args(root))
                self.assertFalse((root / TRACK_A_MARKER).exists())
                self.assertFalse((root / TRACK_A_REPORT).exists())
                generate_calls = [call for call in calls if "-p" in call]
                self.assertEqual(len(generate_calls), 1)
                self.assertIn("--metal", generate_calls[0])
                self.assertIn("--temp", generate_calls[0])
                self.assertLessEqual(int(generate_calls[0][generate_calls[0].index("-n") + 1]), 32)

    def test_helper_writes_evidence_bound_marker_with_stubbed_success(self):
        from scripts import finetune_ds4

        self._enable_gate()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary, gguf = self._write_fake_artifacts(root)
            before = gguf.resolve().stat()
            calls: list[list[str]] = []

            def fake_run(cmd, *unused_args, **unused_kwargs):
                calls.append([str(part) for part in cmd])
                return subprocess.CompletedProcess(cmd, 0, stdout="Answer in one word: ready?\nready\n", stderr="")

            with patch.object(finetune_ds4.subprocess, "run", side_effect=fake_run):
                rc = finetune_ds4.ds4_gguf_base_smoke_check(self._args(root))
            self.assertEqual(rc, 0)
            after = gguf.resolve().stat()
            self.assertEqual((before.st_size, before.st_mtime_ns), (after.st_size, after.st_mtime_ns))

            report_path = root / TRACK_A_REPORT
            marker_path = root / TRACK_A_MARKER
            self.assertTrue(report_path.is_file())
            self.assertTrue(marker_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(report["gate"], "ds4-gguf-base-smoke")
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["prompt"], "Answer in one word: ready?")
            self.assertEqual(report["tokens"], 12)
            self.assertEqual(report["binary"]["path"], str(binary))
            self.assertIsNone(report["binary"].get("version"))
            self.assertEqual(report["gguf"]["realpath"], str(gguf.resolve()))
            self.assertTrue(report["gguf"]["unchanged"])
            self.assertIn("ready", report["stdout_excerpt"])
            self.assertEqual(marker["gate"], "ds4-gguf-base-smoke")
            self.assertEqual(marker["status"], "ok")
            self.assertEqual(marker["report_path"], str(report_path))
            self.assertEqual(marker["report_sha256"], finetune_ds4.sha256_file(report_path))

            generate_calls = [call for call in calls if "-p" in call]
            self.assertEqual(generate_calls, [[
                str(binary),
                "-m", str(gguf.resolve()),
                "-p", "Answer in one word: ready?",
                "-n", "12",
                "--temp", "0",
                "--metal",
            ]])

    def test_track_a_helper_does_not_touch_mlx_track_artifacts(self):
        from scripts import finetune_ds4

        self._enable_gate()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ds4-root"
            root.mkdir()
            mlx_work = Path(td) / "mlx-work"
            mlx_work.mkdir()
            self._write_fake_artifacts(root)
            with patch.object(
                finetune_ds4.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(["ds4"], 0, stdout="Answer in one word: ready?\nready\n", stderr=""),
            ):
                finetune_ds4.ds4_gguf_base_smoke_check(self._args(root, mlx_work=mlx_work))
            self.assertFalse((mlx_work / ".deepseek-v4-forward-parity-ok").exists())
            self.assertFalse((mlx_work / "model-4bit").exists())
            self.assertFalse(TRACK_B_MARKER.exists())
            self._assert_track_b_model_4bit_absent_or_valid_current_artifact()

    def test_forward_parity_blockers_track_b_scoped(self):
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import forward_parity_blockers

        blockers = forward_parity_blockers()
        self.assertIn("full attention parity with RoPE/cache/sinks/compressor/indexer", blockers)
        self.assertIn("full MLX shimmed-checkpoint load/forward and MLX generation smoke (Track B only; DS4-GGUF base generation is its own Track-A gate, .ds4-gguf-generate-ok, not a forward-parity blocker)", blockers)
        self.assertIn("full MoE parity with packed FP4/I8 expert dequant and expert kernels", blockers)
        self.assertNotIn("over the shimmed checkpoint and generation smoke", "\n".join(blockers))

    def test_real_gated_base_smoke_writes_marker_when_opted_in(self):
        if os.environ.get("DS4_GGUF_BASE_SMOKE") != "1":
            self.skipTest("DS4_GGUF_BASE_SMOKE=1 not set; real base smoke skipped by default")
        from scripts import finetune_ds4

        root = Path(os.environ.get("DS4_ROOT", str(PINNED_DS4_ROOT))).expanduser()
        binary = root / "ds4"
        gguf = Path(os.environ.get("DS4_GGUF", str(root / "ds4flash.gguf"))).expanduser()
        if not binary.is_file():
            self.skipTest(f"real ds4 binary not present at {binary}")
        if not gguf.is_file():
            self.skipTest(f"real ds4flash.gguf not present at {gguf}")
        before = gguf.resolve().stat()
        rc = finetune_ds4.ds4_gguf_base_smoke_check(self._args(root, gguf=gguf))
        self.assertEqual(rc, 0)
        after = gguf.resolve().stat()
        self.assertEqual((before.st_size, before.st_mtime_ns), (after.st_size, after.st_mtime_ns))
        marker = json.loads((root / TRACK_A_MARKER).read_text(encoding="utf-8"))
        self.assertEqual(marker["gate"], "ds4-gguf-base-smoke")
        self.assertEqual(marker["status"], "ok")
        self.assertFalse(TRACK_B_MARKER.exists())
        self._assert_track_b_model_4bit_absent_or_valid_current_artifact()


if __name__ == "__main__":
    unittest.main()
