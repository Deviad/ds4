"""Strict configuration loading and canonical fixture/production path checks."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from .canonical import CanonicalError, digest_payload, load_json, sha256_bytes
from .stages import ORDINARY_STAGE_NAMES


class ConfigError(ValueError):
    """Raised for invalid or unsafe pipeline configuration."""


TOP_KEYS = {
    "schema_version", "pipeline_version", "execution", "paths",
    "protected_artifacts", "source", "model", "training", "fusion",
    "imatrix", "quantization", "verification", "tools", "environment",
}
PATH_SECTION_KEYS = {
    "state_root", "output_root", "disposable_root", "promotion_destination",
    "approved_read_roots", "approved_write_roots", "protected_roots",
}
ORDINARY = set(ORDINARY_STAGE_NAMES)
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
ENV_KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
PIPELINE_VERSION = "deepseek-flash-joint-mtp-slice1"
FIXTURE_TOOL_KEYS = {"python", "python_sha256", "fake_stage", "fake_stage_sha256", "must_not_run", "must_not_run_sha256", "version"}
PRODUCTION_TOOL_KEYS = {"trainer", "fusion", "converter", "collector", "repair", "quantizer", "inventory", "engine", "correctness", "benchmark"}
TRAINER_TOOL_KEYS = {
    "executable", "version", "component_repo", "interpreter", "interpreter_sha256",
    "config", "config_sha256", "output_dir", "report_path", "report_schema_version",
    "expected_branch", "expected_staged_diff_sha256", "source_mode",
    "expected_source_identity", "environment", "resume", "rehearsal",
}
SOURCE_IDENTITY_KEYS = {"source_mode", "mlx", "mlx_lm", "mlx_lm_module"}
HYPERPARAMETER_KEYS = {"learning_rate", "batch_size", "epochs", "weight_decay", "warmup_steps"}
VERIFICATION_KEYS = {"correctness_prompts", "performance_prompts", "seed", "max_tokens", "acceptance_threshold", "speed_threshold", "warmup", "repetitions"}
PROTECTED_ARTIFACT_KEYS = {"final_quantized_reference", "frozen_finetuned_reference", "old_support_reference", "retained_clean_imatrix"}
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(sorted(missing)))
        if extra:
            detail.append("unknown " + ", ".join(sorted(extra)))
        raise ConfigError(f"{label}: " + "; ".join(detail))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be an object")
    return value


def _string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise ConfigError(f"{label} must be a non-empty string")
    if unicodedata.normalize("NFC", value) != value:
        raise ConfigError(f"{label} must be NFC-normalized")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{label} must be >= {minimum}")
    return value


def _finite_decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise ConfigError(f"{label} must be a finite decimal")
    decimal = Decimal(value)
    if not decimal.is_finite():
        raise ConfigError(f"{label} must be a finite decimal")
    return decimal


def _absolute_raw(value: Any, label: str) -> Path:
    raw = _string(value, label)
    if "$" in raw or raw.startswith("~"):
        raise ConfigError(f"{label} contains an unexpanded path expression")
    path = Path(raw)
    if not path.is_absolute():
        raise ConfigError(f"{label} must be absolute")
    return path


def _canonical_path(path: Path, label: str, *, must_exist: bool) -> Path:
    if must_exist:
        current = path
        parts: list[Path] = []
        while current != current.parent:
            parts.append(current)
            current = current.parent
        for component in reversed(parts):
            if component.exists() and component.is_symlink():
                raise ConfigError(f"{label} traverses a symlink: {component}")
        try:
            return path.resolve(strict=True)
        except OSError as exc:
            raise ConfigError(f"{label} does not exist: {path}") from exc
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    if current.is_symlink():
        raise ConfigError(f"{label} nearest ancestor is a symlink: {current}")
    return path.resolve(strict=False)


def _under(path: Path, roots: list[Path], label: str) -> None:
    if not any(path == root or root in path.parents for root in roots):
        raise ConfigError(f"{label} is outside approved roots: {path}")


def _path_list(value: Any, label: str, *, must_exist: bool) -> list[Path]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{label} must be a non-empty array")
    return [_canonical_path(_absolute_raw(item, f"{label}[{i}]"), f"{label}[{i}]", must_exist=must_exist)
            for i, item in enumerate(value)]


def _path_value(value: Any, label: str, *, must_exist: bool) -> Path:
    return _canonical_path(_absolute_raw(value, label), label, must_exist=must_exist)


def _validate_model(model: Any) -> None:
    section = _mapping(model, "model")
    expected = {"identity", "layer_count", "hidden_size", "vocabulary_size", "mtp_layer_count", "expected_mtp_tensors"}
    _keys(section, expected, "model")
    _string(section["identity"], "model.identity")
    for key in ("layer_count", "hidden_size", "vocabulary_size", "mtp_layer_count"):
        _integer(section[key], f"model.{key}", minimum=0)
    tensors = section["expected_mtp_tensors"]
    if not isinstance(tensors, list):
        raise ConfigError("model.expected_mtp_tensors must be an array")
    names: set[str] = set()
    for index, tensor in enumerate(tensors):
        item = _mapping(tensor, f"model.expected_mtp_tensors[{index}]")
        _keys(item, {"name", "shape", "dtype"}, f"model.expected_mtp_tensors[{index}]")
        name = _string(item["name"], f"model.expected_mtp_tensors[{index}].name")
        if name in names:
            raise ConfigError(f"duplicate MTP tensor: {name}")
        names.add(name)
        if not name.startswith("mtp."):
            raise ConfigError(f"MTP tensor must start with mtp.: {name}")
        if not isinstance(item["shape"], list) or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in item["shape"]
        ):
            raise ConfigError(f"invalid shape for MTP tensor: {name}")
        _string(item["dtype"], f"model.expected_mtp_tensors[{index}].dtype")


def _validate_training(training: Any) -> None:
    section = _mapping(training, "training")
    expected = {"seed", "lambda_mtp", "checkpoint_interval", "dataset_split_manifest", "main_lora_allowlist", "mtp_lora_allowlist", "hyperparameters"}
    _keys(section, expected, "training")
    _integer(section["seed"], "training.seed")
    _finite_decimal(section["lambda_mtp"], "training.lambda_mtp")
    _integer(section["checkpoint_interval"], "training.checkpoint_interval", minimum=1)
    for key in ("main_lora_allowlist", "mtp_lora_allowlist"):
        values = section[key]
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v for v in values):
            raise ConfigError(f"training.{key} must be a non-empty string array")
    hyperparameters = _mapping(section["hyperparameters"], "training.hyperparameters")
    _keys(hyperparameters, HYPERPARAMETER_KEYS, "training.hyperparameters")
    _finite_decimal(hyperparameters["learning_rate"], "training.hyperparameters.learning_rate")
    _integer(hyperparameters["batch_size"], "training.hyperparameters.batch_size", minimum=1)
    _integer(hyperparameters["epochs"], "training.hyperparameters.epochs", minimum=1)
    _finite_decimal(hyperparameters["weight_decay"], "training.hyperparameters.weight_decay")
    _integer(hyperparameters["warmup_steps"], "training.hyperparameters.warmup_steps", minimum=0)


def _validate_quantization(value: Any) -> None:
    section = _mapping(value, "quantization")
    required = {"layers37_42_experts", "lower_gate_up", "lower_down", "attention", "shared_experts", "output_head", "passthrough_rules", "mtp_policy"}
    _keys(section, required, "quantization")
    for key in required - {"passthrough_rules", "mtp_policy"}:
        _string(section[key], f"quantization.{key}")
    if not isinstance(section["passthrough_rules"], list) or not isinstance(section["mtp_policy"], list):
        raise ConfigError("quantization rules must be arrays")
    seen: set[str] = set()
    for index, rule in enumerate(section["mtp_policy"]):
        item = _mapping(rule, f"quantization.mtp_policy[{index}]")
        _keys(item, {"name", "dtype"}, f"quantization.mtp_policy[{index}]")
        name = _string(item["name"], f"quantization.mtp_policy[{index}].name")
        if name in seen:
            raise ConfigError(f"duplicate MTP policy: {name}")
        seen.add(name)
        _string(item["dtype"], f"quantization.mtp_policy[{index}].dtype")


def _validate_verification(value: Any) -> None:
    section = _mapping(value, "verification")
    _keys(section, VERIFICATION_KEYS, "verification")
    for key in ("correctness_prompts", "performance_prompts"):
        if not isinstance(section[key], list) or any(not isinstance(item, str) or not item for item in section[key]):
            raise ConfigError(f"verification.{key} must be a non-empty-string array")
    _integer(section["seed"], "verification.seed")
    _integer(section["max_tokens"], "verification.max_tokens", minimum=1)
    _finite_decimal(section["acceptance_threshold"], "verification.acceptance_threshold")
    _finite_decimal(section["speed_threshold"], "verification.speed_threshold")
    _integer(section["warmup"], "verification.warmup", minimum=0)
    _integer(section["repetitions"], "verification.repetitions", minimum=1)


def _validate_tools(value: Any) -> None:
    tools = _mapping(value, "tools")
    _keys(tools, {"fixture-v1", "production-v1"}, "tools")
    fixture = _mapping(tools["fixture-v1"], "tools.fixture-v1")
    _keys(fixture, FIXTURE_TOOL_KEYS, "tools.fixture-v1")
    for key in ("python", "fake_stage", "must_not_run"):
        _string(fixture[key], f"tools.fixture-v1.{key}")
    for key in ("python_sha256", "fake_stage_sha256", "must_not_run_sha256"):
        value = _string(fixture[key], f"tools.fixture-v1.{key}")
        if not SHA256_RE.fullmatch(value):
            raise ConfigError(f"{key} must be a sha256 digest")
    _string(fixture["version"], "tools.fixture-v1.version")
    production = _mapping(tools["production-v1"], "tools.production-v1")
    _keys(production, PRODUCTION_TOOL_KEYS, "tools.production-v1")
    for name, declaration in production.items():
        item = _mapping(declaration, f"tools.production-v1.{name}")
        allowed = TRAINER_TOOL_KEYS if name == "trainer" else {"executable", "version"}
        extra = set(item) - allowed
        if extra:
            raise ConfigError(f"tools.production-v1.{name}: unknown " + ", ".join(sorted(extra)))
        if name != "trainer":
            _keys(item, {"executable", "version"}, f"tools.production-v1.{name}")
        elif not {"executable", "version"}.issubset(item):
            raise ConfigError("tools.production-v1.trainer: missing executable or version")
        _string(item["executable"], f"tools.production-v1.{name}.executable")
        _string(item["version"], f"tools.production-v1.{name}.version")
        if name != "trainer":
            continue
        for key in ("component_repo", "interpreter", "config", "output_dir", "report_path", "resume"):
            if key in item and item[key] is not None:
                _absolute_raw(item[key], f"tools.production-v1.trainer.{key}")
        for key in ("interpreter_sha256", "config_sha256", "expected_staged_diff_sha256"):
            if key in item:
                digest = _string(item[key], f"tools.production-v1.trainer.{key}")
                if key == "expected_staged_diff_sha256":
                    if not re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", digest):
                        raise ConfigError(f"tools.production-v1.trainer.{key} must be a sha256 digest")
                elif not SHA256_RE.fullmatch(digest):
                    raise ConfigError(f"tools.production-v1.trainer.{key} must be a sha256 digest")
        if "expected_branch" in item:
            _string(item["expected_branch"], "tools.production-v1.trainer.expected_branch")
        if "source_mode" in item and item["source_mode"] not in {"release", "fork"}:
            raise ConfigError("tools.production-v1.trainer.source_mode must be release or fork")
        if "expected_source_identity" in item:
            source_identity = _mapping(item["expected_source_identity"], "tools.production-v1.trainer.expected_source_identity")
            _keys(source_identity, SOURCE_IDENTITY_KEYS, "tools.production-v1.trainer.expected_source_identity")
            for key in SOURCE_IDENTITY_KEYS:
                _string(source_identity[key], f"tools.production-v1.trainer.expected_source_identity.{key}")
        if "report_schema_version" in item:
            _integer(item["report_schema_version"], "tools.production-v1.trainer.report_schema_version", minimum=1)
        if "environment" in item:
            env = _mapping(item["environment"], "tools.production-v1.trainer.environment")
            for key, env_value in env.items():
                if not isinstance(key, str) or not ENV_KEY_RE.fullmatch(key):
                    raise ConfigError(f"invalid joint trainer environment key: {key}")
                _string(env_value, f"tools.production-v1.trainer.environment.{key}", nonempty=False)
        if "rehearsal" in item and not isinstance(item["rehearsal"], bool):
            raise ConfigError("tools.production-v1.trainer.rehearsal must be boolean")


def _validate_environment(value: Any) -> None:
    section = _mapping(value, "environment")
    if set(section) != ORDINARY:
        raise ConfigError("environment must contain exactly every ordinary stage")
    for stage, values in section.items():
        env = _mapping(values, f"environment.{stage}")
        for key, env_value in env.items():
            if not isinstance(key, str) or not ENV_KEY_RE.fullmatch(key):
                raise ConfigError(f"invalid environment key: {key}")
            _string(env_value, f"environment.{stage}.{key}", nonempty=False)


def _validate_execution(value: Any) -> str:
    section = _mapping(value, "execution")
    _keys(section, {"adapter", "fixture"}, "execution")
    adapter = _string(section["adapter"], "execution.adapter")
    if adapter not in {"fixture-v1", "production-v1"}:
        raise ConfigError(f"unsupported execution.adapter: {adapter}")
    fixture = section["fixture"]
    if adapter == "fixture-v1":
        item = _mapping(fixture, "execution.fixture")
        _keys(item, {"fixture_root", "interrupt_once_stage", "forced_exit_stage", "forced_exit_code"}, "execution.fixture")
        _absolute_raw(item["fixture_root"], "execution.fixture.fixture_root")
        for key in ("interrupt_once_stage", "forced_exit_stage"):
            stage = item[key]
            if stage is not None and stage not in ORDINARY:
                raise ConfigError(f"invalid fixture stage: {stage}")
        _integer(item["forced_exit_code"], "execution.fixture.forced_exit_code")
    elif fixture is not None:
        raise ConfigError("production-v1 requires execution.fixture=null")
    return adapter


@dataclass(frozen=True)
class ResolvedConfig:
    source_path: Path
    source_config_identity: dict[str, Any]
    data: dict[str, Any]
    resolved_config_digest: str
    adapter: str
    fixture_root: Path | None

    @property
    def paths(self) -> dict[str, Any]:
        return self.data["paths"]

    @property
    def state_root(self) -> Path:
        return Path(self.paths["state_root"])

    @property
    def output_root(self) -> Path:
        return Path(self.paths["output_root"])

    @property
    def disposable_root(self) -> Path:
        return Path(self.paths["disposable_root"])

    @property
    def environment(self) -> dict[str, dict[str, str]]:
        return self.data["environment"]

    def as_snapshot(self, run_id: str, registry_digest: str, pipeline_identity: list[dict[str, Any]], created_at: str) -> dict[str, Any]:
        payload = {
            "schema_version": self.data["schema_version"],
            "run_id": run_id,
            "source_config_identity": self.source_config_identity,
            "resolved_config": self.data,
            "resolved_config_digest": self.resolved_config_digest,
            "registry_digest": registry_digest,
            "pipeline_identity": pipeline_identity,
            "approved_roots": {
                "read": self.data["paths"]["approved_read_roots"],
                "write": self.data["paths"]["approved_write_roots"],
            },
            "created_at": created_at,
        }
        return payload


def load_config(path: str | Path, *, preview_fixture_root: str | Path | None = None) -> ResolvedConfig:
    source = Path(path)
    raw = source.read_bytes()
    try:
        data = load_json(source)
    except (OSError, CanonicalError) as exc:
        raise ConfigError(f"cannot load config: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigError("config must be an object")
    _keys(data, TOP_KEYS, "config")
    if data["schema_version"] != 1:
        raise ConfigError("schema_version must be 1")
    if data["pipeline_version"] != PIPELINE_VERSION:
        raise ConfigError(f"unsupported pipeline_version: {data['pipeline_version']}")
    _string(data["pipeline_version"], "pipeline_version")
    adapter = _validate_execution(data["execution"])
    paths = _mapping(data["paths"], "paths")
    _keys(paths, PATH_SECTION_KEYS, "paths")
    must_exist_roots = True
    read_roots = _path_list(paths["approved_read_roots"], "paths.approved_read_roots", must_exist=must_exist_roots)
    write_roots = _path_list(paths["approved_write_roots"], "paths.approved_write_roots", must_exist=must_exist_roots)
    protected_roots = _path_list(paths["protected_roots"], "paths.protected_roots", must_exist=must_exist_roots)
    state_root = _path_value(paths["state_root"], "paths.state_root", must_exist=False)
    output_root = _path_value(paths["output_root"], "paths.output_root", must_exist=False)
    disposable_root = _path_value(paths["disposable_root"], "paths.disposable_root", must_exist=False)
    promotion = _path_value(paths["promotion_destination"], "paths.promotion_destination", must_exist=False)
    if len({state_root, output_root, disposable_root}) != 3:
        raise ConfigError("state, output, and disposable roots must be distinct")
    if not (disposable_root == output_root or output_root in disposable_root.parents):
        raise ConfigError("disposable_root must be below output_root")
    for root, label in ((state_root, "state_root"), (output_root, "output_root"), (disposable_root, "disposable_root"), (promotion, "promotion_destination")):
        _under(root, write_roots, label)
    for root in write_roots:
        for protected in protected_roots:
            if root == protected or protected in root.parents or root in protected.parents:
                raise ConfigError("write root overlaps protected root")
    protected = _mapping(data["protected_artifacts"], "protected_artifacts")
    _keys(protected, PROTECTED_ARTIFACT_KEYS, "protected_artifacts")
    protected_paths = [_path_value(protected[key], f"protected_artifacts.{key}", must_exist=(adapter == "fixture-v1")) for key in sorted(PROTECTED_ARTIFACT_KEYS)]
    for item in protected_paths:
        _under(item, read_roots, "protected artifact")
        if any(item == root or root in item.parents for root in write_roots):
            raise ConfigError("protected artifact overlaps write root")
    source_section = _mapping(data["source"], "source")
    source_keys = {"hf_source", "mlx_source", "dataset_manifest", "template_gguf", "source_tensor_index", "expected_sha256"}
    _keys(source_section, source_keys, "source")
    source_paths = []
    for key in ("hf_source", "mlx_source", "dataset_manifest", "template_gguf", "source_tensor_index"):
        source_paths.append(_path_value(source_section[key], f"source.{key}", must_exist=(adapter == "fixture-v1")))
        _under(source_paths[-1], read_roots, f"source.{key}")
    expected_sha = _mapping(source_section["expected_sha256"], "source.expected_sha256")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in expected_sha.items()):
        raise ConfigError("source.expected_sha256 must map strings to strings")
    _validate_model(data["model"])
    training = _mapping(data["training"], "training")
    _validate_training(training)
    _path_value(training["dataset_split_manifest"], "training.dataset_split_manifest", must_exist=(adapter == "fixture-v1"))
    fusion = _mapping(data["fusion"], "fusion")
    _keys(fusion, {"precision", "output_format"}, "fusion")
    _string(fusion["precision"], "fusion.precision")
    _string(fusion["output_format"], "fusion.output_format")
    imatrix = _mapping(data["imatrix"], "imatrix")
    _keys(imatrix, {"dataset", "collector_args", "repair_value", "strict_consumption"}, "imatrix")
    _path_value(imatrix["dataset"], "imatrix.dataset", must_exist=(adapter == "fixture-v1"))
    if not isinstance(imatrix["collector_args"], list) or any(not isinstance(item, str) for item in imatrix["collector_args"]):
        raise ConfigError("imatrix.collector_args must be a string array")
    _finite_decimal(imatrix["repair_value"], "imatrix.repair_value")
    if imatrix["strict_consumption"] is not True:
        raise ConfigError("imatrix.strict_consumption must be true")
    _validate_quantization(data["quantization"])
    _validate_verification(data["verification"])
    _validate_tools(data["tools"])
    _validate_environment(data["environment"])
    fixture_root: Path | None = None
    if adapter == "fixture-v1":
        fixture_root = _canonical_path(_absolute_raw(data["execution"]["fixture"]["fixture_root"], "fixture_root"), "fixture_root", must_exist=True)
        if preview_fixture_root is not None:
            preview = _canonical_path(_absolute_raw(preview_fixture_root, "preview fixture root"), "preview fixture root", must_exist=True)
            if preview != fixture_root and fixture_root not in preview.parents:
                raise ConfigError("preview fixture root must be the configured fixture root or below it")
        all_paths = [state_root, output_root, disposable_root, promotion] + read_roots + write_roots + protected_roots + protected_paths + source_paths
        all_paths += [_path_value(training["dataset_split_manifest"], "training.dataset_split_manifest", must_exist=True), _path_value(imatrix["dataset"], "imatrix.dataset", must_exist=True)]
        for item in all_paths:
            _under(item, [fixture_root], "fixture path")
        tools = _mapping(data["tools"], "tools")
        tools = _mapping(tools.get("fixture-v1", tools), "tools.fixture-v1")
        for key in ("python", "fake_stage", "must_not_run"):
            if key not in tools:
                raise ConfigError(f"tools.fixture-v1 missing {key}")
            tool_path = _path_value(tools[key], f"tools.fixture-v1.{key}", must_exist=True)
            if key != "python":
                _under(tool_path, [fixture_root], f"tools.fixture-v1.{key}")
    else:
        if preview_fixture_root is not None:
            raise ConfigError("--fixture-root is valid only for fixture-v1")
    resolved = dict(data)
    resolved["paths"] = dict(paths)
    for key, path_value in (("state_root", state_root), ("output_root", output_root), ("disposable_root", disposable_root), ("promotion_destination", promotion)):
        resolved["paths"][key] = str(path_value)
    resolved["paths"]["approved_read_roots"] = [str(item) for item in read_roots]
    resolved["paths"]["approved_write_roots"] = [str(item) for item in write_roots]
    resolved["paths"]["protected_roots"] = [str(item) for item in protected_roots]
    source_identity = {
        "canonical_path": str(source.resolve(strict=True)),
        "size": len(raw),
        "sha256": sha256_bytes(raw),
    }
    return ResolvedConfig(
        source_path=source.resolve(strict=True),
        source_config_identity=source_identity,
        data=resolved,
        resolved_config_digest=digest_payload(resolved),
        adapter=adapter,
        fixture_root=fixture_root,
    )


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or run_id in {".", ".."} or not RUN_ID_RE.fullmatch(run_id):
        raise ConfigError(f"invalid run id: {run_id}")
    return run_id
