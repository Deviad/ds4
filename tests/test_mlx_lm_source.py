#!/usr/bin/env python3
import contextlib
import hashlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SEMANTIC_MLX_SRC_COUNT = 20
SEMANTIC_MLX_SRC_SHA256 = "58c588df0f744a4f483252b44da253a4a2f7d9274223dbe91125f95b59d3c551"
SEMANTIC_MLX_SRC_PRE_PROVIDER_COUNT = 19
SEMANTIC_MLX_SRC_PRE_PROVIDER_SHA256 = "8881561e55b5734ed47676b0baf03da577f202697ab1b9ebe50efff92b3128bc"
SEMANTIC_MLX_SRC_PROVIDER_PATH = "python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py"


def semantic_source_manifest(root, source_root):
    files = []
    for path in source_root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if path.suffix == ".pyc":
            continue
        if any(part.endswith(".egg-info") for part in relative.parts):
            continue
        files.append(path)
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    rows = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}"
        for path in files
    ]
    manifest = ("\n".join(rows) + "\n").encode("utf-8")
    return rows, len(files), hashlib.sha256(manifest).hexdigest()

from scripts import finetune_ds4


FORK_URL = "git@github.com:Deviad/mlx-lm.git"
FORK_SHA = "15b522f593b7ca5fbc0cac6f7572d40859d2d8fe"
class MlxLmSourceTests(unittest.TestCase):
    def args(self, tmp, *, step="mlx-lm-source", source="release", execute=False, yes=False, fmt="shell"):
        return type(
            "Args",
            (),
            {
                "hf_model": str(tmp / "hf"),
                "dataset_root": str(tmp / "dataset"),
                "mlx_work": str(tmp / "mlx work"),
                "ds4_root": str(tmp / "ds4"),
                "ds4_gguf": None,
                "adapter_ds4": None,
                "split_dir": "mlx-4096",
                "fused_hf_model": None,
                "ds4_imatrix": None,
                "backend": "local-mlx",
                "steps": [step] if step else [],
                "step": step,
                "format": fmt,
                "execute": execute,
                "yes": yes,
                "mlx_lm_source": source,
            },
        )()

    def quiet(self, func, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return func(*args)

    def test_constants_and_parser_default_release(self):
        self.assertEqual(finetune_ds4.MLX_VERSION, "0.31.2")
        self.assertEqual(finetune_ds4.MLX_LM_RELEASE_VERSION, "0.31.3")
        self.assertEqual(finetune_ds4.MLX_LM_FORK_URL, FORK_URL)
        self.assertEqual(finetune_ds4.MLX_LM_FORK_SHA, FORK_SHA)
        self.assertEqual(finetune_ds4.MLX_LM_FORK_PATH, pathlib.Path("vendor/mlx-lm"))
        self.assertIn("mlx-lm-source", finetune_ds4.MLX_STEPS)
        self.assertEqual(finetune_ds4.MLX_STEPS.index("mlx-lm-source"), finetune_ds4.MLX_STEPS.index("setup-env") + 1)
        args = finetune_ds4.build_parser().parse_args(["run-command", "mlx-lm-source"])
        self.assertEqual(args.mlx_lm_source, "release")

    def test_dry_run_shell_and_json_record_selected_source(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            shell_args = self.args(tmp, source="fork", fmt="shell")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(finetune_ds4.emit_commands(shell_args), 0)
            text = out.getvalue()
            self.assertIn("# mlx-lm-source=fork", text)
            self.assertIn("mlx-lm-source-verify --mode fork", text)

            json_args = self.args(tmp, source="release", fmt="json")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(finetune_ds4.emit_commands(json_args), 0)
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["mlx_lm_source"], "release")
            self.assertIn("mlx-lm-source", payload["steps"])

    def test_execute_requires_yes_without_environment_bypass(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            args = self.args(tmp, source="fork", execute=True, yes=False)
            with self.assertRaises(finetune_ds4.PlanError):
                finetune_ds4.run_command(args)

    def test_fork_command_order_and_no_dependency_resolution(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            commands = finetune_ds4.command_catalog(self.args(tmp, source="fork"))["mlx-lm-source"]
            joined = "\n".join(commands)
            self.assertIn("# mlx-lm-source=fork", joined)
            self.assertLess(joined.index("--mode fork --scope all"), joined.index("--mode fork --scope checkout"))
            self.assertLess(joined.index("--mode fork --scope checkout"), joined.index("-m pip --isolated --require-virtualenv"))
            self.assertLess(joined.index("-m pip --isolated --require-virtualenv"), joined.rindex("--mode fork --scope all"))
            self.assertIn("install --no-deps --no-build-isolation --force-reinstall -e", joined)
            self.assertIn(str(PROJECT_ROOT / "vendor/mlx-lm"), joined)
            self.assertNotIn(" pip install", joined)
            self.assertNotIn("--user", joined)
            self.assertNotIn("mlx==", joined)
            self.assertNotIn("mlx_lm.lora", joined)
            self.assertNotIn("--data", joined)

    def test_release_command_order_and_idempotent_skip(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            commands = finetune_ds4.command_catalog(self.args(tmp, source="release"))["mlx-lm-source"]
            joined = "\n".join(commands)
            self.assertIn("# mlx-lm-source=release", joined)
            self.assertIn("already selected", joined)
            self.assertLess(joined.index("--mode release --scope all"), joined.index("--mode release --scope environment"))
            self.assertLess(joined.index("--mode release --scope environment"), joined.index("-m pip --isolated --require-virtualenv"))
            self.assertLess(joined.index("-m pip --isolated --require-virtualenv"), joined.rindex("--mode release --scope all"))
            self.assertIn("install --no-deps --force-reinstall 'mlx-lm==0.31.3'", joined)
            self.assertNotIn("--no-build-isolation", joined)
            self.assertNotIn("-e ", joined)

    def test_verifier_environment_and_mlx_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            expected = tmp / "mlx" / ".venv"
            bad_probe = {
                "prefix": str(tmp / "global"),
                "base_prefix": str(tmp / "global"),
                "executable": str(tmp / "global" / "bin" / "python"),
                "user_site_enabled": True,
                "mlx_runtime_version": "0.31.1",
                "mlx_distribution_version": "0.31.1",
            }
            report = finetune_ds4.collect_mlx_lm_source_verification(
                mode="release",
                scope="environment",
                mlx_work=tmp / "mlx",
                project_root=PROJECT_ROOT,
                env_probe=bad_probe,
            )
            self.assertFalse(report["pass"])
            self.assertIn("not-isolated-venv", report["failures"])
            self.assertIn("wrong-venv-prefix", report["failures"])
            self.assertIn("wrong-venv-python", report["failures"])
            self.assertIn("user-site-enabled", report["failures"])
            self.assertIn("mlx-version-mismatch", report["failures"])
            self.assertEqual(pathlib.Path(report["environment"]["expected_prefix"]), expected.resolve(strict=False))

    def test_verifier_release_and_fork_identity_use_path_and_pep610_not_version_only(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            venv = tmp / "mlx" / ".venv"
            fork = tmp / "repo" / "vendor" / "mlx-lm"
            site_pkg = venv / "lib" / "python3.12" / "site-packages" / "mlx_lm" / "__init__.py"
            env_probe = {
                "prefix": str(venv),
                "base_prefix": str(tmp / "base"),
                "executable": str(venv / "bin" / "python"),
                "user_site_enabled": False,
                "mlx_runtime_version": "0.31.2",
                "mlx_distribution_version": "0.31.2",
            }
            release_probe = {
                "distribution_version": "0.31.3",
                "runtime_version": "0.31.3",
                "module_path": str(site_pkg),
                "packages_distributions": {"mlx_lm": ["mlx-lm"]},
                "direct_url": None,
            }
            release = finetune_ds4.collect_mlx_lm_source_verification(
                mode="release",
                scope="installed",
                mlx_work=tmp / "mlx",
                project_root=tmp / "repo",
                env_probe=env_probe,
                installed_probe=release_probe,
            )
            self.assertTrue(release["pass"], release)

            wrong_release = dict(release_probe, module_path=str(fork / "mlx_lm" / "__init__.py"))
            report = finetune_ds4.collect_mlx_lm_source_verification(
                mode="release",
                scope="installed",
                mlx_work=tmp / "mlx",
                project_root=tmp / "repo",
                env_probe=env_probe,
                installed_probe=wrong_release,
            )
            self.assertFalse(report["pass"])
            self.assertIn("source-mode-mismatch", report["failures"])

            fork_probe = dict(
                release_probe,
                module_path=str(fork / "mlx_lm" / "__init__.py"),
                direct_url={"url": fork.as_uri(), "dir_info": {"editable": True}},
            )
            fork_report = finetune_ds4.collect_mlx_lm_source_verification(
                mode="fork",
                scope="installed",
                mlx_work=tmp / "mlx",
                project_root=tmp / "repo",
                env_probe=env_probe,
                installed_probe=fork_probe,
                checkout_report={"pass": True, "failures": []},
            )
            self.assertTrue(fork_report["pass"], fork_report)
            self.assertEqual(fork_report["mlx_lm"]["distribution_version"], "0.31.3")

    def test_checkout_verifier_records_exact_mismatches(self):
        def fake_git(args, cwd=None):
            cmd = " ".join(args)
            if args[:3] == ["config", "-f", ".gitmodules"]:
                return "submodule.mlx-lm.path vendor/mlx-lm\nsubmodule.mlx-lm.url git@github.com:wrong/mlx-lm.git\nsubmodule.mlx-lm.branch main\n"
            if args[:2] == ["ls-files", "--stage"]:
                return "100644 deadbeef 0\tvendor/mlx-lm\n"
            if args[:2] == ["rev-parse", "HEAD"]:
                return "deadbeef\n"
            if args[:3] == ["remote", "get-url", "origin"]:
                return "git@github.com:wrong/mlx-lm.git\n"
            if args[:2] == ["cat-file", "-e"]:
                raise subprocess.CalledProcessError(1, args)
            if args[:2] == ["show-ref", "--verify"]:
                return ""
            if args[:2] == ["merge-base", "--is-ancestor"]:
                raise subprocess.CalledProcessError(1, args)
            if args[:1] == ["status"]:
                return " M mlx_lm/__init__.py\n"
            if args[:2] == ["submodule", "status"]:
                return "+deadbeef vendor/mlx-lm (heads/main)\n"
            return ""

        report = finetune_ds4.check_mlx_lm_submodule(PROJECT_ROOT, run_git=fake_git)
        self.assertFalse(report["pass"])
        for code in (
            "gitmodules-mismatch",
            "gitlink-mismatch",
            "submodule-head-mismatch",
            "origin-mismatch",
            "commit-object-missing",
            "main-provenance-mismatch",
            "recursive-status-mismatch",
            "submodule-dirty",
        ):
            self.assertIn(code, report["failures"])

    def test_current_repository_submodule_metadata_and_gitlink(self):
        gitmodules = PROJECT_ROOT / ".gitmodules"
        self.assertTrue(gitmodules.is_file())
        text = gitmodules.read_text(encoding="utf-8")
        self.assertIn("path = vendor/mlx-lm", text)
        self.assertIn(f"url = {FORK_URL}", text)
        self.assertNotIn("branch =", text)
        stage = subprocess.check_output(["git", "ls-files", "--stage", "--", "vendor/mlx-lm"], cwd=PROJECT_ROOT, text=True).strip()
        self.assertEqual(stage, f"160000 {FORK_SHA} 0\tvendor/mlx-lm")
        head = subprocess.check_output(["git", "-C", "vendor/mlx-lm", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        origin = subprocess.check_output(["git", "-C", "vendor/mlx-lm", "remote", "get-url", "origin"], cwd=PROJECT_ROOT, text=True).strip()
        self.assertEqual(head, FORK_SHA)
        self.assertEqual(origin, FORK_URL)

    def test_pyproject_release_pin_and_no_vcs_source(self):
        text = (PROJECT_ROOT / "python-envs/mlx/pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"mlx==0.31.2"', text)
        self.assertIn('"mlx-lm==0.31.3"', text)
        self.assertNotIn('"mlx-lm>=', text)
        self.assertNotIn("git+", text)
        self.assertNotIn("vendor/mlx-lm", text)

    def test_path_a_evidence_exact_hash_and_tracked_paths(self):
        backlog = (PROJECT_ROOT / "docs/backlog.md").read_text(encoding="utf-8")
        start = backlog.index("#### Story 13.3b-5i — permanent STOP closure")
        end = backlog.index("**EOF Epic 13**") + len("**EOF Epic 13**")
        section = (backlog[start:end] + "\n").encode("utf-8")
        self.assertEqual(len(section), 9958)
        self.assertEqual(hashlib.sha256(section).hexdigest(), "7c87e03eb74b316390025baeebc19dc47cae7db029dfad2d4cff026d9f6b8d47")
        self.assertIn("## Epic 14 — Deviad/mlx-lm fork successor", backlog[end:])
        for needle in (
            "Path A permanently stopped; no successor, redesign, diagnostic, smoke, fallback, or training authorized",
            "Story 13.3c — local QLoRA smoke-train",
            "Story 13.4 — numpy-delta fuse → fused GGUF",
            "Story 13.5 — Post-fuse generation coherence cross-check",
            "Story 13.6 — Full LoRA training",
            "Historical Path B exit — superseded by Story 13.3b-5i",
        ):
            self.assertIn(needle, section.decode("utf-8"))
        for rel in (
            "agent-output/cmux-13-3b/requirements-13-3b-5i-permanent-stop-closure.md",
            "agent-output/cmux-13-3b/architecture-13-3b-5i-final-interaction-classification.md",
            "agent-output/cmux-13-3b/multilayer-peak-report.json",
            "agent-output/cmux-13-3b/interaction-ablation-report.json",
            "agent-output/cmux-13-3b/review-13-3b-5h-r2.md",
            "agent-output/cmux-13-3b/test-report-13-3b-5h-r2.md",
        ):
            subprocess.check_call(["git", "ls-files", "--error-unmatch", rel], cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL)

    def test_protected_manifest_helpers_match_architect_baselines(self):
        report = finetune_ds4.compute_story_14_protected_manifest_report(PROJECT_ROOT)
        self.assertEqual(report["production_runtime"]["count"], 31)
        self.assertEqual(report["production_runtime"]["sha256"], "60e8a764a62b56edc1f170a381008cea51f1507ef7fc7738746e87db27945540")
        _, count, digest = semantic_source_manifest(
            PROJECT_ROOT, PROJECT_ROOT / "python-envs/mlx/src"
        )
        self.assertEqual(count, SEMANTIC_MLX_SRC_COUNT)
        self.assertEqual(digest, SEMANTIC_MLX_SRC_SHA256)
        self.assertEqual(report["path_a_evidence"]["count"], 365)
        self.assertEqual(report["path_a_evidence"]["sha256"], "7241924d6f9be2eb0df974fdcb1717728c71b6ee77d41dea3fe8a8af55ebb2af")

    def test_semantic_source_manifest_repins_only_segmented_provider(self):
        rows, count, digest = semantic_source_manifest(
            PROJECT_ROOT, PROJECT_ROOT / "python-envs/mlx/src"
        )
        old_rows = [row for row in rows if not row.endswith("  " + SEMANTIC_MLX_SRC_PROVIDER_PATH)]
        old_payload = ("\n".join(old_rows) + "\n").encode("utf-8")
        self.assertEqual((len(old_rows), hashlib.sha256(old_payload).hexdigest()),
                         (SEMANTIC_MLX_SRC_PRE_PROVIDER_COUNT, SEMANTIC_MLX_SRC_PRE_PROVIDER_SHA256))
        self.assertEqual((count, digest), (SEMANTIC_MLX_SRC_COUNT, SEMANTIC_MLX_SRC_SHA256))
        self.assertEqual(len(rows) - len(old_rows), 1)
        self.assertEqual(
            [row for row in rows if row not in old_rows],
            [next(row for row in rows if row.endswith("  " + SEMANTIC_MLX_SRC_PROVIDER_PATH))],
        )

    def test_semantic_source_manifest_excludes_generated_state_and_tracks_source_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            source = root / "python-envs/mlx/src"
            included = [source / "pkg/a.py", source / "pkg/nested/b.txt"]
            for path, content in zip(included, (b"a\n", b"b\n")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

            original = semantic_source_manifest(root, source)
            expected_rows = [
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}"
                for path in sorted(included, key=lambda item: item.relative_to(root).as_posix())
            ]
            self.assertEqual(original[0], expected_rows)
            self.assertEqual(original[1], 2)

            excluded = (
                source / "pkg/__pycache__/module.cpython-314.pyc",
                source / "pkg/module.pyc",
                source / "top.egg-info/PKG-INFO",
                source / "pkg/nested/build.egg-info/SOURCES.txt",
            )
            for path in excluded:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"generated\n")
            self.assertEqual(semantic_source_manifest(root, source), original)

            added = source / "pkg/c.py"
            added.write_bytes(b"c\n")
            changed = semantic_source_manifest(root, source)
            self.assertEqual(changed[1], 3)
            self.assertNotEqual(changed[2], original[2])
            added.unlink()
            included[0].write_bytes(b"changed\n")
            modified = semantic_source_manifest(root, source)
            self.assertEqual(modified[1], original[1])
            self.assertNotEqual(modified[2], original[2])
            included[0].write_bytes(b"a\n")
            included[1].unlink()
            deleted = semantic_source_manifest(root, source)
            self.assertEqual(deleted[1], 1)
            self.assertNotEqual(deleted[2], original[2])
            included[1].write_bytes(b"b\n")
            self.assertEqual(semantic_source_manifest(root, source), original)


if __name__ == "__main__":
    unittest.main()
