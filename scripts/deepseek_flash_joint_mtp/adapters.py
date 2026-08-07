"""Stage adapters for fixture rehearsal and the real MLX joint trainer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
from typing import Any, Mapping, Protocol

from .canonical import (
    atomic_write_json,
    digest_payload,
    identity_dict,
    load_json,
    load_json_bytes,
)
from .config import ResolvedConfig
from .runner import LaunchSpec, ProcessResult
from .stages import StageSpec


class AdapterUnavailable(RuntimeError):
    """Raised when a stage adapter cannot prove its immutable contract."""


@dataclass(frozen=True)
class RunContext:
    config: ResolvedConfig
    run_id: str
    registry_digest: str
    pipeline_identity: list[dict[str, Any]]
    state_root: Path
    output_root: Path


class StageAdapter(Protocol):
    def render(self, spec: StageSpec, context: RunContext, dependencies: Mapping[str, Mapping[str, Any]], attempt_dir: Path, temporary_output: Path, final_output: Path) -> LaunchSpec:
        ...

    def validate(self, spec: StageSpec, context: RunContext, launch: LaunchSpec, result: ProcessResult, temporary_output: Path) -> dict[str, Any]:
        ...

    def tool_identity(self) -> dict[str, Any]:
        ...


class FixtureAdapter:
    def __init__(self, config: ResolvedConfig):
        self.config = config
        tools = config.data["tools"]
        self.tools = tools.get("fixture-v1", tools) if isinstance(tools, Mapping) else {}

    def _tool(self, key: str) -> Path:
        try:
            return Path(self.tools[key])
        except KeyError as exc:
            raise AdapterUnavailable(f"fixture tool is missing: {key}") from exc

    def tool_identity(self) -> dict[str, Any]:
        identities = {
            "python": identity_dict(self._tool("python")),
            "fake_stage": identity_dict(self._tool("fake_stage")),
            "must_not_run": identity_dict(self._tool("must_not_run")),
            "version": self.tools["version"],
        }
        for key in ("python", "fake_stage", "must_not_run"):
            expected = self.tools[f"{key}_sha256"]
            if identities[key]["sha256"] != expected:
                raise AdapterUnavailable(f"fixture tool digest mismatch: {key}")
        if identities["version"] != "fixture-v1":
            raise AdapterUnavailable("unsupported fixture tool version")
        return identities

    def render(self, spec: StageSpec, context: RunContext, dependencies: Mapping[str, Mapping[str, Any]], attempt_dir: Path, temporary_output: Path, final_output: Path) -> LaunchSpec:
        python = self._tool("python")
        fake = self._tool("fake_stage")
        checkpoint = attempt_dir.parent.parent / "checkpoint.json"
        argv = [
            str(python), str(fake), "--stage", spec.name, "--run-id", context.run_id,
            "--config-digest", context.config.resolved_config_digest, "--output", str(temporary_output),
        ]
        environment = dict(context.config.environment[spec.name])
        argv.extend(("--argv0", str(python)))
        for name in spec.depends_on:
            argv.extend(("--dependency", f"{name}={dependencies[name]['digest']}"))
        fixture = context.config.data["execution"]["fixture"]
        if fixture.get("interrupt_once_stage") == spec.name:
            argv.extend(("--checkpoint", str(checkpoint)))
        if fixture.get("forced_exit_stage") == spec.name:
            argv.extend(("--force-exit", str(fixture["forced_exit_code"])))
        tool_identity = self.tool_identity()
        return LaunchSpec(
            stage=spec.name,
            attempt_id=attempt_dir.name,
            argv=tuple(argv),
            cwd=context.config.fixture_root or python.parent,
            environment=environment,
            tool_identity=tool_identity,
            input_identities={name: dependencies[name]["payload"]["output_identities"] for name in spec.depends_on},
            temporary_outputs=(temporary_output,),
            final_outputs=(final_output,),
        )

    def validate(self, spec: StageSpec, context: RunContext, launch: LaunchSpec, result: ProcessResult, temporary_output: Path) -> dict[str, Any]:
        if result.status != {"kind": "exit", "code": 0}:
            raise ValueError(f"fixture stage exited unsuccessfully: {result.status}")
        try:
            artifact = load_json(temporary_output)
        except Exception as exc:
            raise ValueError(f"invalid fixture artifact: {exc}") from exc
        if artifact.get("stage") != spec.name or artifact.get("run_id") != context.run_id:
            raise ValueError("fixture artifact identity mismatch")
        if artifact.get("config_digest") != context.config.resolved_config_digest:
            raise ValueError("fixture artifact config mismatch")
        if artifact.get("argv") != list(launch.argv):
            raise ValueError("fixture artifact argv mismatch")
        observed_environment = artifact.get("observed_environment")
        if not isinstance(observed_environment, dict):
            raise ValueError("fixture artifact lacks observed environment")
        if any(observed_environment.get(key) != value for key, value in launch.environment.items()):
            raise ValueError("fixture artifact environment mismatch")
        if artifact.get("payload_digest") != digest_payload({
            "stage": artifact.get("stage"),
            "run_id": artifact.get("run_id"),
            "config_digest": artifact.get("config_digest"),
            "dependencies": artifact.get("dependencies"),
            "argv": artifact.get("argv"),
            "observed_environment": artifact.get("observed_environment"),
            "resumed_from_checkpoint": artifact.get("resumed_from_checkpoint", False),
        }):
            raise ValueError("fixture artifact payload digest mismatch")
        return {
            "stage": spec.name,
            "contract_version": spec.contract_version,
            "valid": True,
            "resumed_from_checkpoint": bool(artifact.get("resumed_from_checkpoint", False)),
            "artifact_digest": identity_dict(temporary_output),
        }


_SOURCE_IDENTITY_FIELDS = {"source_mode", "mlx", "mlx_lm", "mlx_lm_module"}


def _required_path(settings: Mapping[str, Any], key: str, *, directory: bool = False, preserve_symlink: bool = False) -> Path:
    value = settings.get(key)
    if not isinstance(value, str) or not value or "$" in value or value.startswith("~"):
        raise AdapterUnavailable(f"joint trainer {key} must be an absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise AdapterUnavailable(f"joint trainer {key} must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise AdapterUnavailable(f"joint trainer {key} does not exist: {path}") from exc
    if directory and not resolved.is_dir():
        raise AdapterUnavailable(f"joint trainer {key} must be a directory: {resolved}")
    if not directory and not resolved.is_file():
        raise AdapterUnavailable(f"joint trainer {key} must be a file: {resolved}")
    return path if preserve_symlink else resolved


def _executable_identity(path: Path) -> dict[str, Any]:
    # Venv launchers are commonly symlinks; hash the target while preserving
    # the configured launcher path in argv so the venv remains active.
    return identity_dict(path.resolve(strict=True))


def _optional_path(settings: Mapping[str, Any], key: str) -> Path | None:
    value = settings.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value or "$" in value or value.startswith("~"):
        raise AdapterUnavailable(f"joint trainer {key} must be an absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise AdapterUnavailable(f"joint trainer {key} must be absolute")
    return path.resolve(strict=False)


def _git_text(repo: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=str(repo), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
    except OSError as exc:
        raise AdapterUnavailable(f"cannot inspect component repository: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise AdapterUnavailable(f"component repository identity failed: {detail or completed.returncode}")
    return completed.stdout.decode("utf-8").strip()


def _staged_diff_sha256(repo: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "diff", "--cached", "--binary"], cwd=str(repo),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
    except OSError as exc:
        raise AdapterUnavailable(f"cannot inspect component staged diff: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise AdapterUnavailable(f"component staged diff identity failed: {detail or completed.returncode}")
    return hashlib.sha256(completed.stdout).hexdigest()


def _digest_value(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise AdapterUnavailable(f"joint trainer {field} must be a sha256 digest")
    if value.startswith("sha256:"):
        value = value[7:]
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise AdapterUnavailable(f"joint trainer {field} must be a sha256 digest")
    return value


def _report_from_log(temporary_output: Path) -> dict[str, Any]:
    attempt_dir = temporary_output.parent.parent
    log_path = attempt_dir / "stdout.log"
    try:
        value = load_json_bytes(log_path.read_bytes())
    except Exception as exc:
        raise ValueError(f"joint trainer dry-run report is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("joint trainer report must be an object")
    return value


class TrainJointAdapter:
    """Thin, fail-closed adapter for ``ds4_ft_mlx.joint_train``.

    No shell is involved.  The component config is immutable input; the spine
    records its identity and validates the component report after exit.
    """

    def __init__(self, config: ResolvedConfig):
        self.config = config
        self.settings = config.data["tools"]["production-v1"]["trainer"]

    @property
    def rehearsal(self) -> bool:
        return bool(self.settings.get("rehearsal", False))

    def _paths(self) -> tuple[Path, Path, Path, Path, Path | None]:
        repo = _required_path(self.settings, "component_repo", directory=True)
        interpreter = _required_path(self.settings, "interpreter", preserve_symlink=True)
        component_config = _required_path(self.settings, "config")
        output_dir = _optional_path(self.settings, "output_dir")
        report_path = _optional_path(self.settings, "report_path")
        if output_dir is None or report_path is None:
            raise AdapterUnavailable("joint trainer output_dir and report_path are required")
        if report_path.parent != output_dir:
            raise AdapterUnavailable("joint trainer report_path must be directly below output_dir")
        return repo, interpreter, component_config, output_dir, report_path

    def tool_identity(self) -> dict[str, Any]:
        repo, interpreter, component_config, output_dir, report_path = self._paths()
        expected_interpreter = _digest_value(self.settings.get("interpreter_sha256"), "interpreter_sha256")
        expected_config = _digest_value(self.settings.get("config_sha256"), "config_sha256")
        interpreter_identity = _executable_identity(interpreter)
        config_identity = identity_dict(component_config)
        if interpreter_identity["sha256"].removeprefix("sha256:") != expected_interpreter:
            raise AdapterUnavailable("joint trainer interpreter identity mismatch")
        if config_identity["sha256"].removeprefix("sha256:") != expected_config:
            raise AdapterUnavailable("joint trainer config identity mismatch")
        expected_branch = self.settings.get("expected_branch")
        if not isinstance(expected_branch, str) or not expected_branch:
            raise AdapterUnavailable("joint trainer expected_branch is required")
        actual_branch = _git_text(repo, "branch", "--show-current")
        if actual_branch != expected_branch:
            raise AdapterUnavailable(f"joint trainer branch mismatch: expected {expected_branch}, got {actual_branch}")
        expected_staged = _digest_value(self.settings.get("expected_staged_diff_sha256"), "expected_staged_diff_sha256")
        actual_staged = _staged_diff_sha256(repo)
        if actual_staged != expected_staged:
            raise AdapterUnavailable("joint trainer staged-diff hash mismatch")
        source_mode = self.settings.get("source_mode")
        if source_mode not in {"release", "fork"}:
            raise AdapterUnavailable("joint trainer source_mode must be release or fork")
        expected_source = self.settings.get("expected_source_identity")
        if not isinstance(expected_source, Mapping) or set(expected_source) != _SOURCE_IDENTITY_FIELDS:
            raise AdapterUnavailable("joint trainer expected_source_identity is incomplete")
        if expected_source.get("source_mode") != source_mode:
            raise AdapterUnavailable("joint trainer source identity mode mismatch")
        if not all(isinstance(expected_source[key], str) and expected_source[key] for key in _SOURCE_IDENTITY_FIELDS):
            raise AdapterUnavailable("joint trainer expected_source_identity contains invalid values")
        report_schema = self.settings.get("report_schema_version")
        if isinstance(report_schema, bool) or not isinstance(report_schema, int) or report_schema < 1:
            raise AdapterUnavailable("joint trainer report_schema_version must be positive")
        report_descriptor = {
            "path": str(report_path),
            "schema_version": report_schema,
        }
        return {
            "component_repo": str(repo),
            "component_branch": actual_branch,
            "component_staged_diff_sha256": actual_staged,
            "interpreter": interpreter_identity,
            "component_config": config_identity,
            "output_dir": str(output_dir),
            "source_mode": source_mode,
            "expected_source_identity": dict(expected_source),
            "report": report_descriptor,
            "report_descriptor_digest": digest_payload(report_descriptor),
            "version": self.settings.get("version"),
        }

    def render(self, spec: StageSpec, context: RunContext, dependencies: Mapping[str, Mapping[str, Any]], attempt_dir: Path, temporary_output: Path, final_output: Path) -> LaunchSpec:
        if spec.name != "train-joint":
            raise AdapterUnavailable(f"joint trainer adapter cannot render {spec.name}")
        del dependencies
        repo, interpreter, component_config, _output_dir, report_path = self._paths()
        assert report_path is not None
        if report_path.exists():
            raise AdapterUnavailable(f"stale joint trainer report exists: {report_path}")
        identity = self.tool_identity()
        environment = {
            "PYTHONPATH": str(repo / "python-envs" / "mlx" / "src") + ":" + str(repo),
        }
        configured_environment = self.settings.get("environment", {})
        if not isinstance(configured_environment, Mapping):
            raise AdapterUnavailable("joint trainer environment must be an object")
        for key, value in configured_environment.items():
            if not isinstance(key, str) or not key or not key.replace("_", "A").isalnum() or not key.upper() == key:
                raise AdapterUnavailable(f"invalid joint trainer environment key: {key}")
            if not isinstance(value, str) or "\x00" in value:
                raise AdapterUnavailable(f"invalid joint trainer environment value: {key}")
            environment[key] = value
        argv = [str(interpreter), "-m", "ds4_ft_mlx.joint_train", "--config", str(component_config)]
        resume = _optional_path(self.settings, "resume")
        if resume is not None:
            if not resume.is_dir():
                raise AdapterUnavailable(f"joint trainer resume checkpoint does not exist: {resume}")
            argv.extend(("--resume", str(resume)))
        if context.run_id == "preview-run" or self.rehearsal:
            argv.append("--dry-run")
        return LaunchSpec(
            stage=spec.name,
            attempt_id=attempt_dir.name,
            argv=tuple(argv),
            cwd=repo,
            environment=environment,
            tool_identity=identity,
            input_identities={
                "component_config": identity["component_config"],
                "component_repo": {
                    "canonical_path": identity["component_repo"],
                    "branch": identity["component_branch"],
                    "staged_diff_sha256": identity["component_staged_diff_sha256"],
                },
                "expected_report": identity["report"],
            },
            temporary_outputs=(temporary_output,),
            final_outputs=(final_output,),
        )

    @staticmethod
    def _same_interpreter(left: Any, right: Any) -> bool:
        if not isinstance(left, str) or not isinstance(right, str):
            return False
        try:
            return Path(left).resolve(strict=True) == Path(right).resolve(strict=True)
        except OSError:
            return left == right

    def _validate_identity(self, report: Mapping[str, Any], launch: LaunchSpec, *, require_production_identity: bool = False) -> None:
        expected = launch.tool_identity
        source_identity = report.get("source_identity")
        if source_identity != expected["expected_source_identity"]:
            raise ValueError("joint trainer source identity mismatch")
        config = report.get("config")
        if isinstance(config, Mapping):
            if config.get("source_mode") != expected["source_mode"]:
                raise ValueError("joint trainer config source_mode mismatch")
            if config.get("output_dir") != expected["output_dir"]:
                raise ValueError("joint trainer config output_dir mismatch")
        report_path = report.get("report_path")
        if require_production_identity and report_path != expected["report"]["path"]:
            raise ValueError("joint trainer report path mismatch")
        if report_path is not None and report_path != expected["report"]["path"]:
            raise ValueError("joint trainer report path mismatch")
        reported_argv = report.get("argv")
        if require_production_identity and not isinstance(reported_argv, list):
            raise ValueError("joint trainer report lacks argv identity")
        if isinstance(reported_argv, list) and len(reported_argv) >= 5:
            expected_argv = [value for value in launch.argv if value != "--dry-run"]
            if not self._same_interpreter(reported_argv[0], expected_argv[0]) or reported_argv[1:] != expected_argv[1:]:
                raise ValueError("joint trainer argv identity mismatch")
        if require_production_identity:
            adapter_identity = report.get("adapter_identity")
            if not isinstance(adapter_identity, Mapping):
                raise ValueError("joint trainer report lacks adapter identity")
            expected_identity = {
                "component_branch": expected["component_branch"],
                "component_staged_diff_sha256": expected["component_staged_diff_sha256"],
                "config_sha256": expected["component_config"]["sha256"],
                "source_identity": expected["expected_source_identity"],
                "report_path": expected["report"]["path"],
            }
            if dict(adapter_identity) != expected_identity:
                raise ValueError("joint trainer adapter identity mismatch")

    def _write_artifact(self, temporary_output: Path, report: Mapping[str, Any], launch: LaunchSpec, *, mode: str) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "mode": mode,
            "report": dict(report),
            "report_digest": digest_payload(report),
            "report_path": launch.tool_identity["report"]["path"],
            "source_identity": dict(report["source_identity"]),
            "component_identity": {
                "branch": launch.tool_identity["component_branch"],
                "staged_diff_sha256": launch.tool_identity["component_staged_diff_sha256"],
                "config_sha256": launch.tool_identity["component_config"]["sha256"],
            },
            "valid": True,
        }
        atomic_write_json(temporary_output, payload)
        return {
            "schema_version": 1,
            "valid": True,
            "mode": mode,
            "report_digest": payload["report_digest"],
            "report_path": payload["report_path"],
            "source_identity": payload["source_identity"],
            "artifact_digest": identity_dict(temporary_output),
        }

    def validate(self, spec: StageSpec, context: RunContext, launch: LaunchSpec, result: ProcessResult, temporary_output: Path) -> dict[str, Any]:
        if spec.name != "train-joint":
            raise ValueError(f"joint trainer adapter cannot validate {spec.name}")
        if result.status != {"kind": "exit", "code": 0}:
            raise ValueError(f"joint trainer exited unsuccessfully: {result.status}")
        if "--dry-run" in launch.argv:
            report = _report_from_log(temporary_output)
            for field in ("execution_allowed", "payload_loaded", "training_started"):
                if report.get(field) is not False:
                    raise ValueError(f"joint trainer dry-run requires {field}=false")
            self._validate_identity(report, launch)
            return self._write_artifact(temporary_output, report, launch, mode="dry-run")

        _repo, _interpreter, _component_config, _output_dir, report_path = self._paths()
        assert report_path is not None
        try:
            report = load_json(report_path)
        except Exception as exc:
            raise ValueError(f"joint trainer report is unreadable: {exc}") from exc
        if not isinstance(report, Mapping):
            raise ValueError("joint trainer report must be an object")
        if report.get("report_schema_version") != launch.tool_identity["report"]["schema_version"]:
            raise ValueError("joint trainer report schema mismatch")
        if report.get("iters") != 400:
            raise ValueError("joint trainer report must prove iters=400")
        groups = report.get("trainable_groups")
        if not isinstance(groups, Mapping) or not all(isinstance(groups.get(key), Mapping) and groups[key].get("selected_count", 0) > 0 for key in ("main", "mtp")):
            raise ValueError("joint trainer report lacks both trainable groups")
        full_tuned = report.get("mtp_full_tuned_tensors")
        if not isinstance(full_tuned, list) or not full_tuned:
            raise ValueError("joint trainer report lacks full-tuned MTP tensors")
        evidence = report.get("completion_evidence")
        if not isinstance(evidence, Mapping) or evidence.get("main_update") is not True or evidence.get("mtp_update") is not True or evidence.get("checkpoint") is not True:
            raise ValueError("joint trainer report lacks completion evidence")
        if report.get("next_stage_eligible") is not True:
            raise ValueError("joint trainer report is not next-stage eligible")
        self._validate_identity(report, launch, require_production_identity=True)
        return self._write_artifact(temporary_output, report, launch, mode="production")


class NotImplementedAdapter:
    def tool_identity(self) -> dict[str, Any]:
        raise AdapterUnavailable("not implemented in Slice 1")

    def render(self, *args: object, **kwargs: object) -> LaunchSpec:
        raise AdapterUnavailable("not implemented in Slice 1")

    def validate(self, *args: object, **kwargs: object) -> dict[str, Any]:
        raise AdapterUnavailable("not implemented in Slice 1")


def adapter_for(config: ResolvedConfig, stage_name: str | None = None) -> StageAdapter:
    if config.adapter == "fixture-v1":
        return FixtureAdapter(config)
    if stage_name == "train-joint":
        return TrainJointAdapter(config)
    return NotImplementedAdapter()
