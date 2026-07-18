from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import textwrap
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ds4_segmented_pilot.py"
_METADATA_OMITTED = object()

ATTEMPT1_EFFECTIVE_ORACLE = {
    "phase-a": json.loads(r'''{"adapter_path":"/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-a","batch_size":1,"clear_cache_threshold":0,"command":["/Volumes/Data NVME/mlx-ft/ds4/.venv/bin/python","/Users/spotted/projects/ds4-finetuning/scripts/ds4_segmented_pilot.py","--attempt","1","--phase","phase-a","--train"],"config":"/Volumes/Data NVME/mlx-ft/ds4/lora-config.json","data":"/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke","fine_tune_type":"lora","grad_accumulation_steps":1,"grad_checkpoint":true,"hf_dataset":false,"iters":2,"learning_rate":1e-05,"lr_schedule":null,"mask_prompt":true,"max_seq_length":4096,"model":"/Volumes/Data NVME/mlx-ft/ds4/model-4bit","num_layers":16,"optimizer":"adam","phase":"phase-a","project_name":null,"report_to":null,"resume_adapter_file":null,"save_every":1,"seed":0,"segment_size":1,"steps_per_eval":2,"steps_per_report":1,"test":false,"train":true,"trust_remote_code":false,"val_batches":25}'''),
    "phase-b": json.loads(r'''{"adapter_path":"/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-b","batch_size":1,"clear_cache_threshold":0,"command":["/Volumes/Data NVME/mlx-ft/ds4/.venv/bin/python","/Users/spotted/projects/ds4-finetuning/scripts/ds4_segmented_pilot.py","--attempt","1","--phase","phase-b","--train"],"config":"/Volumes/Data NVME/mlx-ft/ds4/lora-config.json","data":"/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke","fine_tune_type":"lora","grad_accumulation_steps":1,"grad_checkpoint":true,"hf_dataset":false,"iters":1,"learning_rate":1e-05,"lr_schedule":null,"mask_prompt":true,"max_seq_length":4096,"model":"/Volumes/Data NVME/mlx-ft/ds4/model-4bit","num_layers":16,"optimizer":"adam","phase":"phase-b","project_name":null,"report_to":null,"resume_adapter_file":"/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-a/0000002_adapters.safetensors","save_every":1,"seed":0,"segment_size":1,"steps_per_eval":1,"steps_per_report":1,"test":false,"train":true,"trust_remote_code":false,"val_batches":25}'''),
}
ATTEMPT1_CONTRACT_ORACLE = {
    "phase-a": {"effective_sha256": "6dd980d5aa78536a4d576811d98b56a39a77be9f75a74909c45c2fa0683cc958", "effective_bytes": 944, "contract_digest": "882243376bc65c79bef731bc6254ba532d0f7a545979ddde667355b21382d112", "contract_bytes": 1614},
    "phase-b": {"effective_sha256": "a0532bf02722824c4458f4d4763fc9ee6bb86e921d072311f0ab62445183984f", "effective_bytes": 1033, "contract_digest": "fa09b656e0a92209f61c90a020159b232356e5dae6c76dce1c2b099b144493e6", "contract_bytes": 1792},
}


def load_pilot():
    spec = importlib.util.spec_from_file_location("ds4_segmented_pilot_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _r10_wrapper_authorization(finetune, project_root, pilot_work, phase):
    paths = finetune._central_attempt3_namespace(
        repo_root=project_root, workspace=pilot_work)
    phase_spec = dict(finetune._central_attempt3_phase_specs(
        repo_root=project_root, workspace=pilot_work)[phase])
    phase_spec.setdefault("attempt", 3)
    phase_spec.setdefault("phase", phase)
    phase_spec.setdefault("namespace", "ds4-segmented-pilot-attempt-3")
    phase_spec.setdefault("namespace_paths", {key: str(value) for key, value in paths.items()})
    try:
        identity = finetune._central_attempt3_launch_identity(
            repo_root=project_root, workspace=pilot_work)
    except FileNotFoundError:
        identity = {"interpreter": pathlib.Path("/python"),
                    "script": pathlib.Path("/pilot.py"), "model": pathlib.Path("/model"),
                    "data": pathlib.Path("/data"), "config": pathlib.Path("/config")}
    command = [str(identity["interpreter"]), str(identity["script"]),
               "--attempt", "3", "--phase", phase,
               "--log-path", str(paths[f"{phase}-log"]),
               "--model", str(identity["model"]), "--data", str(identity["data"]),
               "--adapter-path", str(phase_spec["adapter_path"]),
               "--config", str(identity["config"])]
    if phase == "phase-b":
        command.extend(["--resume-adapter-file", str(phase_spec["resume_adapter_file"])])
    command.extend(["--train", "--fine-tune-type", "lora", "--num-layers", "16",
                    "--iters", str(phase_spec["iters"]), "--batch-size", "1",
                    "--learning-rate", "1e-5", "--max-seq-length", "4096",
                    "--mask-prompt", "--grad-checkpoint", "--grad-accumulation-steps", "1",
                    "--seed", "0", "--optimizer", "adam", "--val-batches", "25",
                    "--steps-per-report", "1", "--steps-per-eval", str(phase_spec["steps_per_eval"]),
                    "--save-every", "1", "--segment-size", "1"])
    value = {
        "revision": "0" * 40,
        "canonical_command_sha256": hashlib.sha256(
            json.dumps(command, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "pilot_source_sha256": "0" * 64,
        "catalog_source_sha256": "0" * 64,
        "protected_files_manifest_sha256": "0" * 64,
        "attempt2_runtime_manifest_sha256": "0" * 64,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _r10_wrapper_kwargs(finetune, project_root, pilot_work, phase):
    return {
        "authorization_json": _r10_wrapper_authorization(
            finetune, project_root, pilot_work, phase),
        "phase_a_authorization_json": (
            _r10_wrapper_authorization(finetune, project_root, pilot_work, "phase-a")
            if phase == "phase-b" else None),
    }


def test_attempt1_effective_pins_and_contract_oracle_are_stable():
    p = load_pilot()
    for phase_name, phase_spec in p.PHASES.items():
        values = {"phase": phase_name, "model": p.PILOT_MODEL, "data": p.PILOT_DATA,
                  "adapter_path": phase_spec["adapter_path"], "config": p.PILOT_CONFIG,
                  "resume_adapter_file": phase_spec["resume_adapter_file"], "train": True,
                  "test": False, "hf_dataset": False,
                  "_command_argv": [p.PILOT_INTERPRETER, str(p.REPO_ROOT / "scripts/ds4_segmented_pilot.py"),
                                    "--attempt", "1", "--phase", phase_name, "--train"]}
        values.update(p.COMMON_VALUES)
        values.update({"iters": phase_spec["iters"], "steps_per_eval": phase_spec["steps_per_eval"]})
        args = type("Args", (), values)()
        effective = p.validate_pins(args, phase_spec, {})
        oracle = ATTEMPT1_EFFECTIVE_ORACLE[phase_name]
        assert effective == oracle
        effective_bytes = json.dumps(effective, sort_keys=True, separators=(",", ":")).encode()
        assert len(effective_bytes) == ATTEMPT1_CONTRACT_ORACLE[phase_name]["effective_bytes"]
        assert hashlib.sha256(effective_bytes).hexdigest() == ATTEMPT1_CONTRACT_ORACLE[phase_name]["effective_sha256"]
        assert p.contract_digest(phase_name, effective, phase_spec=phase_spec) == ATTEMPT1_CONTRACT_ORACLE[phase_name]["contract_digest"]
        contract_payload = {"phase": phase_name, "common": p.COMMON_VALUES, "phase_spec": phase_spec,
                            "effective": effective, "identity": {}}
        contract_bytes = json.dumps(contract_payload, sort_keys=True, separators=(",", ":")).encode()
        assert len(contract_bytes) == ATTEMPT1_CONTRACT_ORACLE[phase_name]["contract_bytes"]
        assert hashlib.sha256(contract_bytes).hexdigest() == ATTEMPT1_CONTRACT_ORACLE[phase_name]["contract_digest"]


def install_tmp_fs_guard(monkeypatch, tmp_path):
    root = tmp_path.resolve()
    original_resolve = pathlib.Path.resolve
    original_path_methods = {
        name: getattr(pathlib.Path, name)
        for name in ("open", "read_text", "write_text", "read_bytes", "write_bytes", "unlink", "replace", "rename", "resolve")
    }
    original_os = {name: getattr(os, name) for name in ("open", "unlink", "replace", "rename")}
    def guard(value):
        path = pathlib.Path(value)
        resolved = original_resolve(path)
        assert resolved == root or root in resolved.parents, f"filesystem escape: {resolved}"
        return path
    for name, original in original_path_methods.items():
        if name == "resolve":
            monkeypatch.setattr(pathlib.Path, name, lambda self, *args, _original=original: (guard(self), _original(self, *args))[1])
        elif name in ("replace", "rename"):
            monkeypatch.setattr(pathlib.Path, name, lambda self, target, *args, _original=original: (guard(self), guard(target), _original(self, target, *args))[2])
        else:
            monkeypatch.setattr(pathlib.Path, name, lambda self, *args, _original=original, **kwargs: (guard(self), _original(self, *args, **kwargs))[1])
    monkeypatch.setattr(os, "open", lambda value, *args, **kwargs: (guard(value), original_os["open"](value, *args, **kwargs))[1])
    monkeypatch.setattr(os, "unlink", lambda value, *args, **kwargs: (guard(value), original_os["unlink"](value, *args, **kwargs))[1])
    monkeypatch.setattr(os, "replace", lambda source, target, *args, **kwargs: (guard(source), guard(target), original_os["replace"](source, target, *args, **kwargs))[2])
    monkeypatch.setattr(os, "rename", lambda source, target, *args, **kwargs: (guard(source), guard(target), original_os["rename"](source, target, *args, **kwargs))[2])
    return root


_MISSING_MODULE_VERSION = object()


def configure_runtime_preflight_fixture(monkeypatch, tmp_path, metadata_value, module_version=_MISSING_MODULE_VERSION):
    p = load_pilot()
    if not hasattr(p, "importlib"):
        import importlib
        p.importlib = importlib
    p.REPO_ROOT = tmp_path
    p.PILOT_WORKSPACE = str(tmp_path / "workspace")
    p.PILOT_INTERPRETER = sys.executable
    p.PILOT_MODEL = str(tmp_path / "model")
    p.PILOT_DATA = str(tmp_path / "data")
    p.PILOT_CONFIG = str(tmp_path / "config.json")
    p.PILOT_PROVENANCE = "provenance.json"
    p.PILOT_VENDOR_SHA = "vendor-sha"
    pathlib.Path(p.PILOT_MODEL).mkdir()
    pathlib.Path(p.PILOT_DATA).mkdir()
    pathlib.Path(p.PILOT_CONFIG).write_text("{}", encoding="utf-8")
    (tmp_path / p.PILOT_PROVENANCE).write_text("{}", encoding="utf-8")
    vendor_module = tmp_path / "vendor" / "mlx-lm" / "mlx_lm" / "__init__.py"
    vendor_module.parent.mkdir(parents=True)
    vendor_module.write_text("", encoding="utf-8")

    fake_mlx = types.ModuleType("mlx")
    if module_version is not _MISSING_MODULE_VERSION:
        fake_mlx.__version__ = module_version
    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_mlx_lm.__file__ = str(vendor_module)
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)

    def metadata_version(_name):
        if isinstance(metadata_value, BaseException):
            raise metadata_value
        return metadata_value

    monkeypatch.setattr(p.importlib.metadata, "version", metadata_version)
    monkeypatch.setattr(p, "_validate_lora_config", lambda _config: {
        "rank": 8, "scale": 20.0, "dropout": 0.0, "keys": ["synthetic"]})
    monkeypatch.setattr(p, "_resource_gate", lambda: {"available_memory": 1, "disk_free": 1,
                                                        "competing_processes": []})
    monkeypatch.setattr(p, "_validate_provenance_splits",
                        lambda _path, _manifest: {name: "hash" for name in ("train.jsonl", "valid.jsonl", "test.jsonl")})
    monkeypatch.setattr(p, "_file_manifest", lambda path: [{"path": pathlib.Path(path).name,
                                                               "size": 0, "sha256": "0" * 64}])

    def fake_file_sha256(path):
        path = pathlib.Path(path)
        if path.name == "segmented_loss_and_grad.py":
            return p.PILOT_PROVIDER_SHA256
        if path.name == "ds4_segmented_smoke.py":
            return p.PILOT_SMOKE_SHA256
        return hashlib.sha256(str(path).encode()).hexdigest()

    monkeypatch.setattr(p, "file_sha256", fake_file_sha256)

    def fake_git(*args):
        if "rev-parse" in args and "vendor/mlx-lm" in args:
            return p.PILOT_VENDOR_SHA
        if "--stage" in args:
            return f"160000 {p.PILOT_VENDOR_SHA} 0\\tvendor/mlx-lm"
        return ""

    monkeypatch.setattr(p, "_git_output", fake_git)
    return p


def write_safetensors(path: pathlib.Path, tensors: dict[str, tuple[str, list[int], bytes]], metadata=_METADATA_OMITTED):
    offset = 0
    header = {}
    payload = bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        payload.extend(raw)
        offset += len(raw)
    if metadata is not _METADATA_OMITTED:
        header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


class FakeArray:
    def __init__(self, shape=(1,), dtype="float32", value=1.0):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.value = value

    def item(self):
        return self.value

    def tolist(self):
        return [[0, 1]]


class FakeRandom:
    def __init__(self, calls):
        self.calls = calls

    def seed(self, value):
        self.calls.append(("mx.random.seed", value))


class FakeMX:
    float32 = "float32"
    int32 = "int32"

    def __init__(self, calls=None):
        self.calls = calls if calls is not None else []
        self.random = FakeRandom(self.calls)

    def eval(self, *values):
        self.calls.append(("mx.eval", len(values)))

    def save_safetensors(self, path, values):
        assert isinstance(values, dict)
        write_safetensors(pathlib.Path(path), {name: ("U8", [1], b"x") for name in values})


def fake_tree_flatten(tree):
    result = []
    def visit(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{prefix}.{index}" if prefix else str(index))
        else:
            result.append((prefix, value))
    visit(tree)
    return result


def test_exact_phase_specs_and_common_pins():
    p = load_pilot()
    assert p.PILOT_TOTAL_BUDGET == 4200
    assert p.COMMON_VALUES == {
        "max_seq_length": 4096,
        "batch_size": 1,
        "learning_rate": 1e-5,
        "mask_prompt": True,
        "grad_checkpoint": True,
        "segment_size": 1,
        "grad_accumulation_steps": 1,
        "seed": 0,
        "fine_tune_type": "lora",
        "optimizer": "adam",
        "num_layers": 16,
        "val_batches": 25,
        "steps_per_report": 1,
        "save_every": 1,
        "report_to": None,
        "project_name": None,
        "trust_remote_code": False,
        "lr_schedule": None,
        "clear_cache_threshold": 0,
    }
    assert p.PHASES["phase-a"]["iters"] == 2
    assert p.PHASES["phase-a"]["steps_per_eval"] == 2
    assert p.PHASES["phase-a"]["timeout"] == 2700
    assert p.PHASES["phase-a"]["global_offset"] == 0
    assert p.PHASES["phase-b"]["iters"] == 1
    assert p.PHASES["phase-b"]["steps_per_eval"] == 1
    assert p.PHASES["phase-b"]["timeout"] == 1500
    assert p.PHASES["phase-b"]["global_offset"] == 2
    assert p.PILOT_VENDOR_SHA == "80fab4e419a57f9465bb9e2f4e90010d645e124c"


def test_parser_rejects_contract_mismatch_before_execution():
    p = load_pilot()
    parser = p.build_parser()
    args = parser.parse_args(["--phase", "phase-a", "--iters", "1"])
    with pytest.raises(p.PilotError, match="pinned-args"):
        p.validate_pins(args, p.PHASES["phase-a"], {})


def test_parser_exposes_pinned_load_dataset_namespace():
    p = load_pilot()
    args = p.build_parser().parse_args(["--phase", "phase-a"])
    assert args.test is False
    assert args.hf_dataset is False
    assert args.train is None
    assert hasattr(args, "data")


def test_canonical_digest_ignores_metadata_and_header_order(tmp_path):
    p = load_pilot()
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    write_safetensors(first, {"b": ("F32", [1], b"bbbb"), "a": ("F32", [1], b"aaaa")}, {"run": "one"})
    write_safetensors(second, {"a": ("F32", [1], b"aaaa"), "b": ("F32", [1], b"bbbb")}, {"run": "two"})
    assert p.file_sha256(first) != p.file_sha256(second)
    assert p.canonical_tensor_digest(first)["canonical_tensor_digest_v1"] == p.canonical_tensor_digest(second)["canonical_tensor_digest_v1"]


def test_canonical_digest_detects_data_schema_and_malformed_files(tmp_path):
    p = load_pilot()
    good = tmp_path / "good.safetensors"
    changed = tmp_path / "changed.safetensors"
    write_safetensors(good, {"x": ("F32", [1], b"aaaa")})
    write_safetensors(changed, {"x": ("F32", [1], b"aaab")})
    assert p.canonical_tensor_digest(good)["canonical_tensor_digest_v1"] != p.canonical_tensor_digest(changed)["canonical_tensor_digest_v1"]
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes((2).to_bytes(8, "little") + b'{}')
    with pytest.raises(p.PilotError):
        p.canonical_tensor_digest(bad)


def test_null_metadata_matrix_preserves_manifest_digest_and_file_hash_sensitivity(tmp_path):
    p = load_pilot()
    tensors = {"x": ("F32", [1], b"aaaa")}
    variants = {
        "absent": _METADATA_OMITTED,
        "null": None,
        "empty": {},
        "object": {"run": "synthetic"},
    }
    parsed = {}
    for name, metadata in variants.items():
        path = tmp_path / f"{name}.safetensors"
        write_safetensors(path, tensors, metadata)
        parsed[name] = p.canonical_tensor_digest(path)
    assert all(item["tensors"] == [{"name": "x", "dtype": "F32", "shape": [1], "nbytes": 4}] for item in parsed.values())
    assert len({item["canonical_tensor_digest_v1"] for item in parsed.values()}) == 1
    assert len({item["file_sha256"] for item in parsed.values()}) == 4

    for value in ("", "metadata", [], [1], 0, 1.5, False, True):
        path = tmp_path / f"invalid-{type(value).__name__}.safetensors"
        write_safetensors(path, tensors, value)
        with pytest.raises(p.PilotError, match="object or null"):
            p.canonical_tensor_digest(path)

    duplicate = tmp_path / "duplicate-metadata.safetensors"
    header = b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"__metadata__":null,"__metadata__":{}}'
    duplicate.write_bytes(len(header).to_bytes(8, "little") + header + b"aaaa")
    with pytest.raises(p.PilotError, match="duplicate"):
        p.canonical_tensor_digest(duplicate)


def test_attempt3_is_only_live_namespace_and_attempt2_is_not_launchable():
    p = load_pilot()
    namespace = p.attempt3_namespace(repo_root=ROOT, workspace=pathlib.Path("/tmp/synthetic-ds4"))
    assert namespace["phase-a-output"].name == "adapters-segmented-pilot-attempt-3-phase-a"
    assert namespace["phase-b-resume"].name == "0000002_adapters.safetensors"
    assert p.attempt3_phase_specs()["phase-a"]["attempt"] == 3
    assert p.canonical_attempt3_command("phase-a", p.attempt3_phase_specs()["phase-a"])[2:4] == ["--attempt", "3"]
    with pytest.raises(SystemExit):
        p.build_parser().parse_args(["--attempt", "2", "--phase", "phase-a"])


def test_resume_proof_requires_equal_source_and_different_start(tmp_path):
    p = load_pilot()
    source = tmp_path / "source.safetensors"
    start = tmp_path / "start.safetensors"
    phase_start = tmp_path / "phase-start.safetensors"
    write_safetensors(source, {"lora": ("F32", [1], b"stp2")})
    write_safetensors(start, {"lora": ("F32", [1], b"stp2")})
    write_safetensors(phase_start, {"lora": ("F32", [1], b"stp0")})
    assert p.verify_resume_continuity(source, start, phase_start)["equal_source"]
    assert p.verify_resume_continuity(source, start, phase_start)["changed_from_phase_start"]
    write_safetensors(start, {"lora": ("F32", [1], b"stp0")})
    with pytest.raises(p.PilotError, match="resume-start"):
        p.verify_resume_continuity(source, start, phase_start)


def test_step_evidence_and_global_mapping_are_fail_closed():
    p = load_pilot()
    evidence = {"loss": 1.0, "n_tokens": 1, "gradients_finite": True,
                "gradient_schema": [{"path": "lora.a", "shape": [1], "dtype": "float32"}],
                "gradient_leaf_count": 1, "expected_mask_tokens": 1, "mask_tokens_match": True}
    phase_a = [{"local_step": 1, **evidence}, {"local_step": 2, **evidence}]
    phase_b = [{"local_step": 1, **evidence}]
    assert p.validate_step_evidence("phase-a", phase_a, 2) == [1, 2]
    assert p.validate_step_evidence("phase-b", phase_b, 1) == [1]
    assert p.global_steps("phase-a", [1, 2]) == [1, 2]
    assert p.global_steps("phase-b", [1]) == [3]
    with pytest.raises(p.PilotError):
        p.validate_step_evidence("phase-a", [{"local_step": 1, **evidence}], 2)
    empty = dict(evidence, gradient_schema=[], gradient_leaf_count=0)
    with pytest.raises(p.PilotError, match="nonempty"):
        p.validate_step_evidence("phase-a", [{"local_step": 1, **empty}, {"local_step": 2, **empty}], 2)


def test_output_and_phase_b_dependency_gates(tmp_path):
    p = load_pilot()
    assert p.ensure_absent(tmp_path / "new") is None
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(p.PilotError, match="already exists"):
        p.ensure_absent(existing)
    with pytest.raises(p.PilotError, match="Phase A"):
        p.validate_phase_b_dependency({}, {}, tmp_path / "missing")


def test_marker_order_and_report_aggregation():
    p = load_pilot()
    a = {"status": "ok", "steps": [{"global_step": 1}, {"global_step": 2}], "provider_calls": 2, "optimizer_updates": 2, "wall_seconds": 1.0}
    b = {"status": "ok", "steps": [{"global_step": 3}], "provider_calls": 1, "optimizer_updates": 1, "wall_seconds": 2.0}
    final = p.aggregate_reports(a, b)
    assert final["provider_calls"] == 3
    assert final["optimizer_updates"] == 3
    assert final["global_progression"] == [0, 1, 2, 3]
    assert final["total_active_wall_seconds"] == 3.0
    assert final["contract_digest"]
    assert "optimizer_state_continuity" in final["non_claims"]


def test_catalog_commands_retire_attempt1_and_keep_attempt3_non_default():
    p = load_pilot()
    import scripts.finetune_ds4 as finetune
    old = {"ds4-segmented-pilot-phase-a", "ds4-segmented-pilot-phase-b"}
    assert old.isdisjoint(finetune.MLX_STEPS)
    assert old.isdisjoint(finetune.BACKEND_STEPS["local-mlx"])
    assert old.isdisjoint(finetune.COMMAND_STEPS)
    assert {"ds4-segmented-pilot-attempt-3-phase-a", "ds4-segmented-pilot-attempt-3-phase-b"} <= set(finetune.MLX_STEPS)
    assert not {"ds4-segmented-pilot-attempt-2-phase-a", "ds4-segmented-pilot-attempt-2-phase-b"}.intersection(finetune.MLX_STEPS)
    assert old.isdisjoint(finetune.DEFAULT_BACKEND_STEPS["local-mlx"])
    assert "ds4-segmented-pilot-attempt-3-phase-a" not in finetune.DEFAULT_BACKEND_STEPS["local-mlx"]
    assert "ds4-segmented-pilot-attempt-3-phase-b" not in finetune.DEFAULT_BACKEND_STEPS["local-mlx"]


def test_no_real_asset_access_during_import(tmp_path, monkeypatch):
    touched = []
    original = pathlib.Path.open
    def trap(self, *args, **kwargs):
        if str(self).startswith("/Volumes/Data NVME"):
            touched.append(str(self))
            raise AssertionError("real asset access")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(pathlib.Path, "open", trap)
    load_pilot()
    assert touched == []


def test_watchdog_install_and_cancel_are_explicitly_bounded(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    alarms = []
    monkeypatch.setattr(p.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(p.signal, "alarm", lambda seconds: alarms.append(seconds))
    watchdog = p._install_timeout_watchdog(7)
    watchdog.cancel()
    assert alarms == [7, 0]
    assert watchdog.event.is_set()


def test_timeout_and_retry_contracts_are_explicit():
    p = load_pilot()
    assert p.PHASES["phase-a"]["timeout"] == 2700
    assert p.PHASES["phase-b"]["timeout"] == 1500
    assert p.PILOT_TOTAL_BUDGET == 4200
    assert p.RETRY_POLICY == "none"
    assert p.FALLBACK_POLICY == "none"


def test_report_write_is_atomic(tmp_path):
    p = load_pilot()
    report = tmp_path / "report.json"
    p.atomic_write_json(report, {"status": "ok"})
    assert json.loads(report.read_text()) == {"status": "ok"}
    assert not list(tmp_path.glob("*.tmp"))


def test_smoke_source_hash_guard():
    smoke = ROOT / "scripts" / "ds4_segmented_smoke.py"
    digest = hashlib.sha256(smoke.read_bytes()).hexdigest()
    assert digest == "ec17950d985578f3cd97b51734527bfc47e6c399ebe84fad8ffcb12beae18ad8"


def test_canonical_digest_rejects_dtype_size_gaps_and_trailing_bytes(tmp_path):
    p = load_pilot()
    wrong_dtype = tmp_path / "wrong-dtype.safetensors"
    write_safetensors(wrong_dtype, {"x": ("F32", [2], b"x")})
    with pytest.raises(p.PilotError, match="byte size"):
        p.canonical_tensor_digest(wrong_dtype)

    gap = tmp_path / "gap.safetensors"
    header = {"x": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]}}
    encoded = json.dumps(header).encode()
    gap.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"ab")
    with pytest.raises(p.PilotError, match="contiguous"):
        p.canonical_tensor_digest(gap)

    trailing = tmp_path / "trailing.safetensors"
    header = {"x": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}
    encoded = json.dumps(header).encode()
    trailing.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"ab")
    with pytest.raises(p.PilotError, match="trailing"):
        p.canonical_tensor_digest(trailing)


def test_validate_pins_requires_every_effective_contract_value():
    p = load_pilot()
    parser = p.build_parser()
    args = parser.parse_args(["--phase", "phase-a"])
    with pytest.raises(p.PilotError, match="required"):
        p.validate_pins(args, p.PHASES["phase-a"], {})


def test_phase_b_dependency_requires_complete_binding_and_exact_report_identity(tmp_path, monkeypatch):
    p = load_pilot()
    source = tmp_path / "0000002_adapters.safetensors"
    write_safetensors(source, {"x": ("U8", [1], b"x")})
    report_path = tmp_path / "phase-a-report.json"
    monkeypatch.setitem(p.PHASES["phase-a"], "report", str(report_path))
    report = {
        "status": "ok",
        "phase": "phase-a",
        "provider_calls": 2,
        "optimizer_updates": 2,
        "contract_digest": "contract",
        "resume_source": {"path": str(source), "file_sha256": p.file_sha256(source), "canonical_tensor_digest_v1": p.canonical_tensor_digest(source)["canonical_tensor_digest_v1"]},
    }
    report_path.write_text(json.dumps(report))
    marker = {
        "status": "ok",
        "phase": "phase-a",
        "report_path": str(report_path),
        "report_sha256": p.file_sha256(report_path),
        "contract_digest": "contract",
        "resume_source": str(source),
        "resume_source_file_sha256": report["resume_source"]["file_sha256"],
        "resume_source_canonical_tensor_digest_v1": report["resume_source"]["canonical_tensor_digest_v1"],
    }
    p.validate_phase_b_dependency(report, marker, source)

    unrelated = tmp_path / "unrelated-report.json"
    unrelated.write_bytes(report_path.read_bytes())
    marker["report_path"] = str(unrelated)
    marker["report_sha256"] = p.file_sha256(unrelated)
    with pytest.raises(p.PilotError, match="canonical"):
        p.validate_phase_b_dependency(report, marker, source)

    marker["report_path"] = str(report_path)
    substituted = dict(report, provider_calls=99)
    report_path.write_text(json.dumps(substituted))
    marker["report_sha256"] = p.file_sha256(report_path)
    with pytest.raises(p.PilotError, match="payload"):
        p.validate_phase_b_dependency(report, marker, source)

    old_bytes = json.dumps(report).encode()
    report_path.write_bytes(old_bytes)
    marker["report_sha256"] = hashlib.sha256(old_bytes).hexdigest()
    substituted = dict(report, mutation="hash-read-boundary")
    original_hash = p.file_sha256
    def replace_after_hash(path, *args, **kwargs):
        result = original_hash(path, *args, **kwargs)
        if pathlib.Path(path).resolve() == report_path.resolve():
            report_path.write_text(json.dumps(substituted))
        return result
    monkeypatch.setattr(p, "file_sha256", replace_after_hash)
    with pytest.raises(p.PilotError, match="payload"):
        p.validate_phase_b_dependency(substituted, marker, source)
    monkeypatch.setattr(p, "file_sha256", original_hash)

    duplicate = json.dumps(report).replace('"status": "ok"', '"status": "ok", "status": "ok"', 1)
    report_path.write_text(duplicate)
    marker["report_sha256"] = p.file_sha256(report_path)
    with pytest.raises(p.PilotError, match="JSON"):
        p.validate_phase_b_dependency(report, marker, source)

    report_path.write_text(json.dumps(report))
    marker["report_sha256"] = p.file_sha256(report_path)
    marker.pop("contract_digest")
    with pytest.raises(p.PilotError, match="binding"):
        p.validate_phase_b_dependency(report, marker, source)


def test_callback_uses_vendor_one_dict_contract_and_validation():
    p = load_pilot()
    callback = p._PilotTrainingCallback("phase-a")
    callback.on_val_loss_report({"iteration": 0, "val_loss": 1.0, "val_time": 0.2})
    callback.on_train_loss_report({
        "iteration": 1,
        "train_loss": 1.0,
        "learning_rate": 1e-5,
        "tokens_per_second": 2.0,
        "iterations_per_second": 4.0,
    })
    assert callback.validation_records[0]["iteration"] == 0
    assert callback.records[0]["global_step"] == 1
    assert callback.records[0]["train_step_wall_seconds"] == 0.25
    with pytest.raises(p.PilotError):
        callback.on_train_loss_report({"iteration": 2, "train_loss": float("nan"), "learning_rate": 1e-5, "tokens_per_second": 1.0, "iterations_per_second": 1.0})


def test_training_contract_uses_dataset_tokenizer_optimizer_and_supported_args():
    p = load_pilot()
    calls = {}

    class FakeModel:
        def freeze(self):
            calls["freeze"] = True
        def load_weights(self, path, strict=False):
            calls["load_weights"] = (path, strict)
        def trainable_parameters(self):
            return {"lora": FakeArray([1.0])}

    class FakeArray:
        def __init__(self, values=(), dtype="float32", value=1.0):
            self.shape = tuple(values) if isinstance(values, tuple) else (len(values),)
            self.dtype = dtype
            self.value = value
        def item(self):
            return self.value

    class LocalMX:
        float32 = "float32"
        int32 = "int32"
        class Random:
            def seed(self, value):
                calls.setdefault("seed", []).append(("mx.random.seed", value))
        random = Random()
        def eval(self, *values):
            calls["eval"] = True
        def save_safetensors(self, path, values):
            calls.setdefault("saves", []).append(path)

    class FakeTrainingArgs:
        def __init__(self, **kwargs):
            calls["training_args"] = kwargs

    class FakeOptimizer:
        pass

    class FakeAPI:
        mx = LocalMX()
        tree_flatten = staticmethod(fake_tree_flatten)
        TrainingArgs = FakeTrainingArgs
        Optimizer = FakeOptimizer
        def load(self, model, **kwargs):
            calls["load"] = (model, kwargs)
            return FakeModel(), "tokenizer"
        def load_dataset(self, args, tokenizer):
            calls["dataset"] = (args, tokenizer)
            return "train", "valid", "test"
        def linear_to_lora_layers(self, model, num_layers, config):
            calls["lora"] = (model, num_layers, config)
        def make_optimizer(self, learning_rate):
            calls["optimizer"] = learning_rate
            return FakeOptimizer()
        def train(self, model, optimizer, train_dataset, val_dataset, **kwargs):
            calls["train"] = {"model": model, "optimizer": optimizer, "train_dataset": train_dataset, "val_dataset": val_dataset, **kwargs}

    args = p.build_parser().parse_args([
        "--phase", "phase-a", "--model", p.PILOT_MODEL, "--data", p.PILOT_DATA,
        "--adapter-path", p.PHASES["phase-a"]["adapter_path"], "--config", p.PILOT_CONFIG,
        "--train", "--fine-tune-type", "lora", "--num-layers", "16", "--iters", "2",
        "--batch-size", "1", "--learning-rate", "1e-5", "--max-seq-length", "4096",
        "--mask-prompt", "--grad-checkpoint", "--grad-accumulation-steps", "1",
        "--seed", "0", "--optimizer", "adam", "--val-batches", "25",
        "--steps-per-report", "1", "--steps-per-eval", "2", "--save-every", "1", "--segment-size", "1",
    ])
    provider = p._StepObservingProvider(lambda *_: ((FakeArray((), "float32", 1.0), FakeArray((), "int32", 1)), {"lora": FakeArray([1.0])}), "phase-a", FakeAPI.mx, fake_tree_flatten)
    callback = p._PilotTrainingCallback("phase-a")
    output = pathlib.Path("/tmp/pilot-contract-test")
    p._execute_training(args, p.PHASES["phase-a"], FakeAPI(), output, provider, callback, config={"rank": 8, "scale": 20.0, "dropout": 0.0, "keys": []})
    assert calls["dataset"] == (args, "tokenizer")
    assert calls["freeze"] is True
    assert calls["optimizer"] == 1e-5
    assert calls["training_args"]["adapter_file"] == str(output / "adapters.safetensors")
    assert calls["train"]["optimizer"].__class__ is FakeOptimizer
    assert calls["train"]["train_dataset"] == "train"
    assert calls["train"]["val_dataset"] == "valid"
    assert calls["train"]["args"].__class__ is FakeTrainingArgs


def test_provider_records_finite_schema_mask_and_rejects_mutation():
    p = load_pilot()
    mx = FakeMX()
    tokens = FakeArray(shape=(1, 3), dtype="int32")
    lengths = FakeArray(shape=(1, 2), dtype="int32")
    lengths.tolist = lambda: [[0, 2]]

    class Array:
        shape = (2,)
        dtype = "float32"
    scalar_loss = FakeArray(shape=(), dtype="float32", value=1.0)
    scalar_tokens = FakeArray(shape=(), dtype="int32", value=2)
    provider = p._StepObservingProvider(lambda *_: ((scalar_loss, scalar_tokens), {"grad": Array()}), "phase-a", mx, fake_tree_flatten,
                                         expected_schema=[{"path": "grad", "shape": [2], "dtype": "float32"}])
    result = provider(object(), tokens, lengths)
    assert result[0][1].item() == 2
    assert provider.records[0]["gradient_schema"] == [{"path": "grad", "shape": [2], "dtype": "float32"}]
    assert provider.records[0]["mask_tokens_match"] is True

    class ChangedArray(Array):
        shape = (3,)
    changed = p._StepObservingProvider(lambda *_: ((scalar_loss, scalar_tokens), {"grad": Array()}), "phase-a", mx, fake_tree_flatten,
                                       expected_schema=[{"path": "grad", "shape": [2], "dtype": "float32"}])
    changed(object(), tokens, lengths)
    changed.delegate = lambda *_: ((scalar_loss, scalar_tokens), {"grad": ChangedArray()})
    with pytest.raises(p.PilotError, match="schema"):
        changed(object(), tokens, lengths)
    flat = p._StepObservingProvider(lambda *_: (scalar_loss, scalar_tokens, {"grad": Array()}), "phase-a", mx, fake_tree_flatten)
    with pytest.raises(p.PilotError, match="provider must return"):
        flat(object(), tokens, lengths)


def test_provider_requires_scalar_mlx_dtypes_and_exact_mask_length():
    p = load_pilot()
    mx = FakeMX()
    tokens = FakeArray(shape=(1, 4), dtype="int32")
    lengths = FakeArray(shape=(1, 2), dtype="int32")
    lengths.tolist = lambda: [[0, 99]]
    scalar_loss = FakeArray(shape=(), dtype="float32", value=1.0)
    scalar_tokens = FakeArray(shape=(), dtype="int32", value=2)
    provider = p._StepObservingProvider(lambda *_: ((scalar_loss, scalar_tokens), {"grad": FakeArray()}),
                                         "phase-a", mx, fake_tree_flatten)
    with pytest.raises(p.PilotError, match="token count"):
        provider(object(), tokens, lengths)
    bad_loss = FakeArray(shape=(1,), dtype="float32", value=1.0)
    provider.delegate = lambda *_: ((bad_loss, scalar_tokens), {"grad": FakeArray()})
    lengths.tolist = lambda: [[0, 3]]
    with pytest.raises(p.PilotError, match="scalar"):
        provider(object(), tokens, lengths)


def test_seed_application_order_is_explicit():
    p = load_pilot()
    calls = []
    mx = FakeMX(calls)
    import numpy as np
    original = np.random.seed
    np.random.seed = lambda value: calls.append(("numpy.random.seed", value))
    try:
        assert p._apply_phase_seeds(mx, 0) == ["numpy.random.seed", "mx.random.seed"]
    finally:
        np.random.seed = original
    assert calls[:2] == [("numpy.random.seed", 0), ("mx.random.seed", 0)]


def test_real_mlx_nested_trainable_save_is_flattened(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from mlx.utils import tree_flatten
    p = load_pilot()
    class Model:
        def trainable_parameters(self):
            return {"outer": {"weight": mx.ones((1,), dtype=mx.float32)}}
    path = tmp_path / "nested.safetensors"
    schema = p._save_trainable({"mx": mx, "tree_flatten": tree_flatten}, Model(), path)
    assert schema == [{"path": "outer.weight", "shape": [1], "dtype": "float32"}]
    assert "outer.weight" in mx.load(str(path))


def test_identity_comparison_excludes_dynamic_resources_and_binds_digest():
    p = load_pilot()
    immutable = {"assets": {"model": "a"}, "runtime": {"mlx": "0.31.2"}}
    p.compare_immutable_identity({"immutable": immutable, "dynamic_resources": {"available": 1}},
                                {"immutable": immutable, "dynamic_resources": {"available": 2}})
    with pytest.raises(p.PilotError, match="immutable identity"):
        p.compare_immutable_identity({"immutable": immutable}, {"immutable": {"assets": {"model": "b"}}})
    digest_a = p.contract_digest("phase-a", {"command": ["a"]}, {"immutable": immutable})
    digest_b = p.contract_digest("phase-a", {"command": ["b"]}, {"immutable": immutable})
    assert digest_a != digest_b


def test_validation_callback_cardinality_and_order_are_pinned():
    p = load_pilot()
    assert p._expected_validation_steps(p.PHASES["phase-a"]) == [0, 1]
    assert p._expected_validation_steps(p.PHASES["phase-b"]) == [0]
    callback = p._PilotTrainingCallback("phase-a")
    callback.on_val_loss_report({"iteration": 0, "val_loss": 1.0, "val_time": 0.1})
    assert [item["iteration"] for item in callback.validation_records] != [0, 1]


def test_aggregate_reports_rejects_progression_and_budget_mutations():
    p = load_pilot()
    a = {"status": "ok", "steps": [{"global_step": 1}, {"global_step": 2}], "provider_calls": 2, "optimizer_updates": 2, "wall_seconds": 1.0}
    b = {"status": "ok", "steps": [{"global_step": 4}], "provider_calls": 1, "optimizer_updates": 1, "wall_seconds": 2.0}
    with pytest.raises(p.PilotError, match="progression"):
        p.aggregate_reports(a, b)
    b["steps"][0]["global_step"] = 3
    b["wall_seconds"] = p.PILOT_TOTAL_BUDGET
    with pytest.raises(p.PilotError, match="budget"):
        p.aggregate_reports(a, b)


def test_phase_b_identity_mismatch_stops_before_training(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    model = tmp_path / "model"; model.mkdir()
    data = tmp_path / "data"; data.mkdir()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0, "keys": ["k"]}}))
    phase_a_dir = work / "phase-a"; phase_a_dir.mkdir()
    phase_b_dir = work / "phase-b"
    source = phase_a_dir / "0000002_adapters.safetensors"
    write_safetensors(source, {"x": ("U8", [1], b"x")})
    phase_a_report_path = tmp_path / "phase-a-report.json"
    source_info = p.canonical_tensor_digest(source)
    phase_a_report = {"status": "ok", "phase": "phase-a", "provider_calls": 2, "optimizer_updates": 2,
                      "contract_digest": "phase-a-contract", "identity_manifest": {"immutable": {"asset": "old"}},
                      "resume_source": source_info}
    phase_a_report_path.write_text(json.dumps(phase_a_report))
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(work))
    monkeypatch.setattr(p, "PILOT_MODEL", str(model))
    monkeypatch.setattr(p, "PILOT_DATA", str(data))
    monkeypatch.setattr(p, "PILOT_CONFIG", str(config))
    monkeypatch.setattr(p, "PILOT_FINAL_REPORT", tmp_path / "pilot-report.json")
    monkeypatch.setitem(p.PHASES["phase-a"], "report", str(phase_a_report_path))
    monkeypatch.setitem(p.PHASES["phase-a"], "adapter_path", str(phase_a_dir))
    monkeypatch.setitem(p.PHASES["phase-b"], "report", str(tmp_path / "phase-b-report.json"))
    monkeypatch.setitem(p.PHASES["phase-b"], "adapter_path", str(phase_b_dir))
    monkeypatch.setitem(p.PHASES["phase-b"], "resume_adapter_file", str(source))
    marker = {"status": "ok", "phase": "phase-a", "report_path": str(phase_a_report_path),
              "report_sha256": p.file_sha256(phase_a_report_path), "contract_digest": "phase-a-contract",
              "resume_source": str(source), "resume_source_file_sha256": source_info["file_sha256"],
              "resume_source_canonical_tensor_digest_v1": source_info["canonical_tensor_digest_v1"]}
    p._marker_path("phase-a", "ok").write_text(json.dumps(marker))
    args = p.build_parser().parse_args([
        "--phase", "phase-b", "--model", str(model), "--data", str(data), "--adapter-path", str(phase_b_dir),
        "--config", str(config), "--resume-adapter-file", str(source), "--train", "--fine-tune-type", "lora",
        "--num-layers", "16", "--iters", "1", "--batch-size", "1", "--learning-rate", "1e-5",
        "--max-seq-length", "4096", "--mask-prompt", "--grad-checkpoint", "--grad-accumulation-steps", "1",
        "--seed", "0", "--optimizer", "adam", "--val-batches", "25", "--steps-per-report", "1",
        "--steps-per-eval", "1", "--save-every", "1", "--segment-size", "1"])
    called = []
    def forbidden(*_args, **_kwargs):
        called.append(True)
        raise AssertionError("training must not start after identity drift")
    monkeypatch.setattr(p, "_execute_training", forbidden)
    report = p.run_phase(args, preflight=lambda *_: {"immutable": {"asset": "new"}, "dynamic_resources": {"free": 1}})
    assert report["status"] == "fail"
    assert "immutable identity" in report["error"]
    assert called == []


def test_lock_default_resolves_mutated_workspace_without_real_path(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(tmp_path))
    lock = p._acquire_ft_lock()
    try:
        assert lock == tmp_path / ".ds4-ft.lock"
        assert lock.is_file()
        assert str(lock).startswith(str(tmp_path))
    finally:
        p._release_ft_lock()
    assert not (tmp_path / ".ds4-ft.lock").exists()


def test_success_evidence_rolls_back_all_ok_markers_on_final_marker_failure(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(work))
    monkeypatch.setattr(p, "PILOT_FINAL_REPORT", tmp_path / "pilot-report.json")
    monkeypatch.setitem(p.PHASES["phase-a"], "report", str(tmp_path / "phase-a-report.json"))
    monkeypatch.setitem(p.PHASES["phase-b"], "report", str(tmp_path / "phase-b-report.json"))
    report = {"status": "ok", "phase": "phase-b", "output_path": str(work / "phase-b"),
              "steps": [{"global_step": 3}], "provider_calls": 1, "optimizer_updates": 1,
              "wall_seconds": 1.0, "contract_digest": "contract"}
    phase_a = {"status": "ok", "steps": [{"global_step": 1}, {"global_step": 2}],
               "provider_calls": 2, "optimizer_updates": 2, "wall_seconds": 1.0,
               "contract_digest": "a"}
    pathlib.Path(p.PHASES["phase-a"]["report"]).write_text(json.dumps(phase_a))
    original = p.atomic_write_json
    calls = []
    def fail_final_marker(path, value):
        calls.append(pathlib.Path(path))
        if pathlib.Path(path).name == ".ds4-segmented-pilot-ok":
            raise OSError("injected final marker failure")
        return original(path, value)
    monkeypatch.setattr(p, "atomic_write_json", fail_final_marker)
    with pytest.raises(OSError, match="final marker"):
        p._write_success_evidence("phase-b", report)
    assert not list(work.glob("*.ds4-segmented-pilot-*-ok"))
    assert not (work / ".ds4-segmented-pilot-ok").exists()
    assert calls


def test_terminal_mutation_matrix_covers_every_required_boundary():
    p = load_pilot()
    assert p.TERMINAL_MUTATION_POINTS == (
        "parser/config", "preflight", "start-save", "provider", "callback",
        "checkpoint-validation", "phase-report", "final-report", "phase-marker",
        "final-marker", "timeout", "watchdog-cancellation", "lock-release",
        "phase-b-success", "phase-b-failure", "final-aggregation",
    )
    source = pathlib.Path(__file__).read_text()
    tree = ast.parse(source)
    terminal_names = {"run_phase", "_write_success_evidence", "_write_failure_evidence",
                      "_acquire_ft_lock", "_release_ft_lock", "_install_timeout_watchdog"}
    uncovered = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
            continue
        names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
        attrs = {item.attr for item in ast.walk(node) if isinstance(item, ast.Attribute)}
        if terminal_names.intersection(names | attrs):
            guarded = any(isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
                          and item.func.id == "install_tmp_fs_guard" for item in ast.walk(node))
            if not guarded:
                uncovered.append(node.name)
    assert uncovered == []


def test_catalog_rendering_retains_defaults_and_binds_attempt3_wrapper():
    import scripts.finetune_ds4 as finetune
    import scripts.ds4_segmented_pilot as pilot
    class Args:
        hf_model = "/tmp/hf"
        dataset_root = "/tmp/data"
        mlx_work = "/tmp/mlx"
        ds4_root = "/tmp/ds4"
        ds4_gguf = None
        split_dir = "mlx-4096"
        fused_hf_model = None
        ds4_imatrix = None
        adapter_ds4 = None
        attempt3_authorization_file = None
        attempt3_authorization_json = pilot.canonical_attempt3_authorization(
            "phase-a", pilot.attempt3_phase_specs()["phase-a"],
            catalog_source=ROOT / "scripts" / "finetune_ds4.py")
    catalog = finetune.command_catalog(Args())
    command = catalog["ds4-segmented-pilot-attempt-3-phase-a"][0]
    assert command.startswith("bash -lc 'set -euo pipefail")
    assert "--attempt 3 --phase phase-a --launch-check-only --log-path \"$LOG\"" in command
    assert command.index("--launch-check-only") < command.index('mkdir -p "$(dirname "$LOG")"')
    assert command.index('set -o noclobber; exec 3>"$LOG"') < command.index("--attempt 3 --phase phase-a --log-path \"$LOG\" --log-fd 3")
    assert "tee /dev/fd/3" in command
    assert 'tee "$LOG"' not in command and "tee -a" not in command
    assert catalog["smoke-train"] == ["cd /tmp/mlx && unset SSLKEYLOGFILE && . /tmp/mlx/.venv/bin/activate && mlx_lm.lora --config /tmp/mlx/lora-config.json --model /tmp/mlx/model-4bit --train --data /tmp/data/mlx-4096 --adapter-path /tmp/mlx/adapters-smoke --fine-tune-type lora --iters 20 --batch-size 1 --learning-rate 1e-5 --max-seq-length 4096 --mask-prompt --grad-checkpoint"]
    assert catalog["continue-train"] == ["cd /tmp/mlx && unset SSLKEYLOGFILE && . /tmp/mlx/.venv/bin/activate && mlx_lm.lora --config /tmp/mlx/lora-config.json --model /tmp/mlx/model-4bit --train --data /tmp/data/mlx-4096 --adapter-path /tmp/mlx/adapters --resume-adapter-file /tmp/mlx/adapters/adapters.safetensors --fine-tune-type lora --iters 15000 --batch-size 1 --learning-rate 5e-6 --max-seq-length 4096 --mask-prompt --grad-checkpoint --steps-per-report 10 --steps-per-eval 200"]


@pytest.mark.parametrize("phase,failure_number", [("phase-a", 1), ("phase-a", 2),
                                                     ("phase-b", 1), ("phase-b", 2),
                                                     ("phase-b", 3), ("phase-b", 4)])
def test_every_success_evidence_write_boundary_rolls_back_ok_markers(tmp_path, monkeypatch, phase, failure_number):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(work))
    monkeypatch.setattr(p, "PILOT_FINAL_REPORT", tmp_path / "pilot-report.json")
    monkeypatch.setitem(p.PHASES["phase-a"], "report", str(tmp_path / "phase-a-report.json"))
    monkeypatch.setitem(p.PHASES["phase-b"], "report", str(tmp_path / "phase-b-report.json"))
    pathlib.Path(p.PHASES["phase-a"]["report"]).write_text(json.dumps({
        "status": "ok", "phase": "phase-a", "steps": [{"global_step": 1}, {"global_step": 2}],
        "provider_calls": 2, "optimizer_updates": 2, "wall_seconds": 1.0,
    }))
    report = ({"status": "ok", "phase": "phase-a", "output_path": str(work / "phase-a"),
               "steps": [{"global_step": 1}, {"global_step": 2}], "provider_calls": 2,
               "optimizer_updates": 2, "wall_seconds": 1.0, "contract_digest": "contract"}
              if phase == "phase-a" else
              {"status": "ok", "phase": "phase-b", "output_path": str(work / "phase-b"),
               "steps": [{"global_step": 3}], "provider_calls": 1, "optimizer_updates": 1,
               "wall_seconds": 1.0, "contract_digest": "contract"})
    original = p.atomic_write_json
    calls = 0
    def fail_at_boundary(path, value):
        nonlocal calls
        calls += 1
        if calls == failure_number:
            raise OSError(f"injected boundary {failure_number}")
        return original(path, value)
    monkeypatch.setattr(p, "atomic_write_json", fail_at_boundary)
    with pytest.raises(OSError, match=f"boundary {failure_number}"):
        p._write_success_evidence(phase, report)
    assert all(not marker.exists() for marker in p._ok_marker_paths())


@pytest.mark.parametrize("failure_number", [1, 2, 3, 4])
def test_each_failure_evidence_boundary_preserves_only_bound_fail_markers(tmp_path, monkeypatch, failure_number):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(work))
    monkeypatch.setattr(p, "PILOT_FINAL_REPORT", tmp_path / "pilot-report.json")
    monkeypatch.setitem(p.PHASES["phase-b"], "report", str(tmp_path / "phase-b-report.json"))
    report = {"status": "fail", "phase": "phase-b", "error": "injected"}
    original = p.atomic_write_json
    calls = 0
    def fail_at_boundary(path, value):
        nonlocal calls
        calls += 1
        if calls == failure_number:
            raise OSError(f"failure boundary {failure_number}")
        return original(path, value)
    monkeypatch.setattr(p, "atomic_write_json", fail_at_boundary)
    with pytest.raises(Exception):
        p._write_failure_evidence("phase-b", report)
    assert all(not marker.exists() for marker in p._ok_marker_paths())
    for marker in (p._marker_path("phase-b", "fail"), work / ".ds4-segmented-pilot-fail"):
        if marker.exists():
            payload = json.loads(marker.read_text())
            assert payload["report_sha256"] == p.file_sha256(pathlib.Path(payload["report_path"]))


def test_attempt_gate_precedes_config_failure_and_preserves_prior_evidence(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    (work / "phase-a").mkdir()
    report_path = tmp_path / "phase-a-report.json"
    report_bytes = b'{"status":"ok","phase":"phase-a"}\n'
    report_path.write_bytes(report_bytes)
    marker_path = work / ".ds4-segmented-pilot-phase-a-ok"
    marker_bytes = b'{"status":"ok"}\n'
    marker_path.write_bytes(marker_bytes)
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(work))
    monkeypatch.setattr(p, "PILOT_FINAL_REPORT", tmp_path / "pilot-report.json")
    monkeypatch.setitem(p.PHASES["phase-a"], "adapter_path", str(work / "phase-a"))
    monkeypatch.setitem(p.PHASES["phase-a"], "report", str(report_path))
    args = type("Args", (), {"phase": "phase-a", "config": str(tmp_path / "missing.json")})()
    result = p.run_phase(args)
    assert result["status"] == "fail"
    assert report_path.read_bytes() == report_bytes
    assert marker_path.read_bytes() == marker_bytes
    assert not (work / ".ds4-segmented-pilot-phase-a-fail").exists()


def test_attempt_gate_failure_preserves_prior_evidence_byte_for_byte(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    output = work / "phase-a"
    output.mkdir()
    report_path = tmp_path / "phase-a-report.json"
    report_bytes = b'{"status":"ok","phase":"phase-a"}\n'
    report_path.write_bytes(report_bytes)
    marker_path = work / ".ds4-segmented-pilot-phase-a-ok"
    marker_bytes = b'{"status":"ok","report_path":"phase-a-report.json"}\n'
    marker_path.write_bytes(marker_bytes)
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(work))
    monkeypatch.setattr(p, "PILOT_FINAL_REPORT", tmp_path / "pilot-report.json")
    monkeypatch.setitem(p.PHASES["phase-a"], "adapter_path", str(output))
    monkeypatch.setitem(p.PHASES["phase-a"], "report", str(report_path))
    monkeypatch.setattr(p, "_read_config", lambda *_args: {})
    monkeypatch.setattr(p, "validate_pins", lambda *_args: {})
    args = type("Args", (), {"phase": "phase-a", "config": None})()
    result = p.run_phase(args, preflight=lambda *_args: {})
    assert result["status"] == "fail"
    assert report_path.read_bytes() == report_bytes
    assert marker_path.read_bytes() == marker_bytes
    assert not (work / ".ds4-segmented-pilot-phase-a-fail").exists()


def test_partial_lock_acquisition_removes_unowned_lock(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    original_fsync = p.os.fsync
    def fail_fsync(_fd):
        raise OSError("injected lock fsync failure")
    monkeypatch.setattr(p.os, "fsync", fail_fsync)
    with pytest.raises(p.PilotError, match="lock acquisition failed"):
        p._acquire_ft_lock(tmp_path)
    assert not (tmp_path / ".ds4-ft.lock").exists()
    assert p._LOCK_OWNED_PATH is None and p._LOCK_OWNED_TOKEN is None
    monkeypatch.setattr(p.os, "fsync", original_fsync)


def test_watchdog_install_failure_disarms_signal_alarm(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    alarms = []
    monkeypatch.setattr(p.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(p.signal, "alarm", lambda seconds: alarms.append(seconds))
    class BadThread:
        def __init__(self, **_kwargs):
            pass
        def start(self):
            raise OSError("injected thread start failure")
    monkeypatch.setattr(p.threading, "Thread", BadThread)
    with pytest.raises(OSError, match="thread start"):
        p._install_timeout_watchdog(10)
    assert alarms == [10, 0]


def test_watchdog_event_construction_failure_disarms_signal_alarm(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    alarms = []
    monkeypatch.setattr(p.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(p.signal, "alarm", lambda seconds: alarms.append(seconds))
    def bad_event():
        raise OSError("injected Event construction")
    monkeypatch.setattr(p.threading, "Event", bad_event)
    with pytest.raises(OSError, match="Event construction"):
        p._install_timeout_watchdog(10)
    assert alarms == [10, 0]


def test_partial_lock_acquisition_preserves_recoverable_owner_when_cleanup_unlink_fails(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    original_unlink = pathlib.Path.unlink
    calls = []
    def fail_lock_unlink(path, *args, **kwargs):
        if path == tmp_path / ".ds4-ft.lock":
            calls.append(path)
            raise OSError("injected partial cleanup unlink")
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(p.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("injected lock fsync failure")))
    monkeypatch.setattr(pathlib.Path, "unlink", fail_lock_unlink)
    with pytest.raises(p.PilotError, match="lock acquisition failed"):
        p._acquire_ft_lock(tmp_path)
    assert calls == [tmp_path / ".ds4-ft.lock"]
    assert (tmp_path / ".ds4-ft.lock").is_file()
    assert p._LOCK_OWNED_PATH == tmp_path / ".ds4-ft.lock"
    assert p._LOCK_OWNED_TOKEN == str(os.getpid())
    monkeypatch.undo()
    p._release_ft_lock()
    assert not (tmp_path / ".ds4-ft.lock").exists()
    assert p._LOCK_OWNED_PATH is None and p._LOCK_OWNED_TOKEN is None


def test_release_unlink_failure_preserves_owner_for_exactly_once_retry(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    lock = p._acquire_ft_lock(tmp_path)
    original_unlink = pathlib.Path.unlink
    calls = []
    def fail_once(path, *args, **kwargs):
        if path == lock and not calls:
            calls.append(path)
            raise OSError("injected release unlink")
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(pathlib.Path, "unlink", fail_once)
    with pytest.raises(OSError, match="release unlink"):
        p._release_ft_lock()
    assert calls == [lock]
    assert lock.is_file()
    assert p._LOCK_OWNED_PATH == lock
    assert p._LOCK_OWNED_TOKEN == str(os.getpid())
    p._release_ft_lock()
    assert calls == [lock]
    assert not lock.exists()
    assert p._LOCK_OWNED_PATH is None and p._LOCK_OWNED_TOKEN is None


def test_run_phase_quarantines_retained_partial_lock_after_acquire_failure(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    lock = work / ".ds4-ft.lock"
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(work))
    monkeypatch.setattr(p, "PILOT_FINAL_REPORT", tmp_path / "pilot-report.json")
    monkeypatch.setitem(p.PHASES["phase-a"], "adapter_path", str(work / "phase-a"))
    monkeypatch.setitem(p.PHASES["phase-a"], "report", str(tmp_path / "phase-a-report.json"))
    monkeypatch.setattr(p, "_read_config", lambda *_args: {})
    monkeypatch.setattr(p, "validate_pins", lambda *_args: {})
    monkeypatch.setattr(p, "_install_timeout_watchdog", lambda _seconds: type("Watchdog", (), {"cancel": lambda self: None})())
    original_fsync = p.os.fsync
    fsync_calls = 0
    def fail_owner_fsync(fd):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 1:
            raise OSError("injected owner fsync failure")
        return original_fsync(fd)
    original_unlink = pathlib.Path.unlink
    def fail_lock_unlink(path, *args, **kwargs):
        if path == lock:
            raise OSError("injected cleanup unlink failure")
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(p.os, "fsync", fail_owner_fsync)
    monkeypatch.setattr(pathlib.Path, "unlink", fail_lock_unlink)

    result = p.run_phase(type("Args", (), {"phase": "phase-a", "config": None})())

    assert result["status"] == "fail"
    assert result["lock_lifecycle"]["acquired"] is False
    assert result["lock_lifecycle"]["release_attempts"] == 1
    assert result["lock_lifecycle"]["quarantined"] is True
    assert not lock.exists()
    assert list(work.glob(".ds4-ft.lock.quarantine.*"))
    assert p._LOCK_OWNED_PATH is None and p._LOCK_OWNED_TOKEN is None


def test_run_phase_partial_progress_orders_failure_cleanup_and_marker(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    events = []
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(work))
    monkeypatch.setattr(p, "PILOT_FINAL_REPORT", tmp_path / "pilot-report.json")
    monkeypatch.setitem(p.PHASES["phase-a"], "adapter_path", str(work / "phase-a"))
    monkeypatch.setitem(p.PHASES["phase-a"], "report", str(tmp_path / "phase-a-report.json"))
    monkeypatch.setattr(p, "_read_config", lambda *_args: {})
    monkeypatch.setattr(p, "validate_pins", lambda *_args: {})
    monkeypatch.setattr(p, "_runtime_preflight", lambda *_args: {})
    class TraceWatchdog:
        def cancel(self):
            events.append("watchdog cancel")
    monkeypatch.setattr(p, "_install_timeout_watchdog", lambda _seconds: TraceWatchdog())
    def fail_after_one_update(_args, _phase, _api, _output, provider, callback, **_kwargs):
        provider.records.append({"local_step": 1, "provider_call": 1})
        callback.records.append({"local_step": 1, "optimizer_update_ordinal": 1})
        events.append("training failure")
        raise p.PilotError("injected failure after one completed update")
    monkeypatch.setattr(p, "_execute_training", fail_after_one_update)
    original_release = p._release_ft_lock
    def release_with_trace():
        events.append("release")
        return original_release()
    monkeypatch.setattr(p, "_release_ft_lock", release_with_trace)
    original_failure = p._write_failure_evidence
    def failure_with_trace(phase, report):
        events.append("failure report")
        return original_failure(phase, report)
    monkeypatch.setattr(p, "_write_failure_evidence", failure_with_trace)
    original_atomic = p.atomic_write_json
    def atomic_with_trace(path, value):
        result = original_atomic(path, value)
        if pathlib.Path(path).name == ".ds4-segmented-pilot-phase-a-fail":
            events.append("fail marker")
        return result
    monkeypatch.setattr(p, "atomic_write_json", atomic_with_trace)

    result = p.run_phase(type("Args", (), {"phase": "phase-a", "config": None})(),
                         api={"tree_flatten": fake_tree_flatten, "mx": FakeMX()})

    assert result["status"] == "fail"
    assert result["lock_lifecycle"] == {
        "path": str(work / ".ds4-ft.lock"), "acquired": True,
        "released": True, "release_attempts": 1,
    }
    assert events == ["training failure", "watchdog cancel", "release", "failure report", "fail marker"]


def test_watchdog_cancel_failure_still_releases_lock_and_writes_failure(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(work))
    monkeypatch.setattr(p, "PILOT_FINAL_REPORT", tmp_path / "pilot-report.json")
    monkeypatch.setitem(p.PHASES["phase-a"], "adapter_path", str(work / "phase-a"))
    monkeypatch.setitem(p.PHASES["phase-a"], "report", str(tmp_path / "phase-a-report.json"))
    monkeypatch.setattr(p, "validate_pins", lambda *_args: {})
    class BadWatchdog:
        def cancel(self):
            raise OSError("injected watchdog cancellation failure")
    monkeypatch.setattr(p, "_install_timeout_watchdog", lambda _seconds: BadWatchdog())
    monkeypatch.setattr(p, "_runtime_preflight", lambda *_args: (_ for _ in ()).throw(p.PilotError("preflight stop")))
    args = type("Args", (), {"phase": "phase-a", "config": None})()
    result = p.run_phase(args)
    assert result["status"] == "fail"
    assert "watchdog cancellation" in result["error"]
    assert not (work / ".ds4-ft.lock").exists()
    assert (tmp_path / "phase-a-report.json").is_file()
    assert (work / ".ds4-segmented-pilot-phase-a-fail").is_file()


def test_rollback_unlink_failure_moves_ok_marker_out_of_evidence_namespace(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(work))
    monkeypatch.setattr(p, "PILOT_FINAL_REPORT", tmp_path / "pilot-report.json")
    monkeypatch.setitem(p.PHASES["phase-a"], "report", str(tmp_path / "phase-a-report.json"))
    monkeypatch.setitem(p.PHASES["phase-b"], "report", str(tmp_path / "phase-b-report.json"))
    pathlib.Path(p.PHASES["phase-a"]["report"]).write_text(json.dumps({
        "status": "ok", "phase": "phase-a", "steps": [{"global_step": 1}, {"global_step": 2}],
        "provider_calls": 2, "optimizer_updates": 2, "wall_seconds": 1.0,
    }))
    report = {"status": "ok", "phase": "phase-b", "output_path": str(work / "phase-b"),
              "steps": [{"global_step": 3}], "provider_calls": 1, "optimizer_updates": 1,
              "wall_seconds": 1.0, "contract_digest": "contract"}
    original_write = p.atomic_write_json
    original_unlink = pathlib.Path.unlink
    def fail_final(path, value):
        if pathlib.Path(path).name == ".ds4-segmented-pilot-ok":
            raise OSError("injected final marker failure")
        return original_write(path, value)
    def fail_once(path, *args, **kwargs):
        if path.name == ".ds4-segmented-pilot-phase-b-ok":
            pathlib.Path.unlink = original_unlink
            raise OSError("injected rollback unlink failure")
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(p, "atomic_write_json", fail_final)
    monkeypatch.setattr(pathlib.Path, "unlink", fail_once)
    with pytest.raises(OSError, match="final marker"):
        p._write_success_evidence("phase-b", report)
    assert not (work / ".ds4-segmented-pilot-phase-b-ok").exists()
    assert not (work / ".ds4-segmented-pilot-ok").exists()


@pytest.mark.parametrize("stage", ["parser/config", "preflight", "start-save", "provider",
                                    "callback", "checkpoint-validation", "timeout", "lock-release"])
def test_terminal_failure_matrix_executes_each_real_stage_and_cleans_evidence(tmp_path, monkeypatch, stage):
    p = load_pilot()
    work = tmp_path / "work"
    model = tmp_path / "model"
    data = tmp_path / "data"
    config = tmp_path / "config.json"
    work.mkdir(); model.mkdir(); data.mkdir(); config.write_text("{}")
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(work))
    monkeypatch.setattr(p, "PILOT_MODEL", str(model))
    monkeypatch.setattr(p, "PILOT_DATA", str(data))
    monkeypatch.setattr(p, "PILOT_CONFIG", str(config))
    monkeypatch.setattr(p, "PILOT_FINAL_REPORT", tmp_path / "pilot-report.json")
    monkeypatch.setitem(p.PHASES["phase-a"], "adapter_path", str(work / "phase-a"))
    monkeypatch.setitem(p.PHASES["phase-a"], "report", str(tmp_path / "phase-a-report.json"))
    monkeypatch.setitem(p.PHASES["phase-b"], "report", str(tmp_path / "phase-b-report.json"))
    monkeypatch.setattr(p, "_read_config", lambda *_args: {})
    monkeypatch.setattr(p, "validate_pins", lambda *_args: {"command": []})
    args = p.build_parser().parse_args([
        "--phase", "phase-a", "--model", str(model), "--data", str(data),
        "--adapter-path", str(work / "phase-a"), "--config", str(config), "--train",
        "--fine-tune-type", "lora", "--num-layers", "16", "--iters", "2",
        "--batch-size", "1", "--learning-rate", "1e-5", "--max-seq-length", "4096",
        "--mask-prompt", "--grad-checkpoint", "--grad-accumulation-steps", "1",
        "--seed", "0", "--optimizer", "adam", "--val-batches", "25",
        "--steps-per-report", "1", "--steps-per-eval", "2", "--save-every", "1",
        "--segment-size", "1",
    ])
    error = p.PilotError(f"injected {stage}")
    install_tmp_fs_guard(monkeypatch, tmp_path)

    class Model:
        def freeze(self):
            pass
        def trainable_parameters(self):
            return {"x": FakeArray()}
        def load_weights(self, *_args, **_kwargs):
            pass

    class TrainingArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    training_args_type = TrainingArgs
    class API:
        mx = FakeMX()
        tree_flatten = staticmethod(fake_tree_flatten)
        TrainingArgs = training_args_type
        def load(self, *_args, **_kwargs):
            return Model(), object()
        def load_dataset(self, *_args):
            return [], [], []
        def linear_to_lora_layers(self, *_args, **_kwargs):
            pass
        def make_optimizer(self, _lr):
            return object()
        def make_provider(self, **_kwargs):
            def provider(*_args):
                raise error
            return provider
        def train(self, _model, _optimizer, _train, _valid, **kwargs):
            if stage == "provider":
                kwargs["loss_and_grad"](_model)
            elif stage == "callback":
                kwargs["training_callback"].on_train_loss_report({"train_loss": float("nan")})

    preflight = lambda *_: {}
    release_calls = []
    if stage == "parser/config":
        monkeypatch.setattr(p, "_read_config", lambda *_: (_ for _ in ()).throw(error))
    elif stage == "preflight":
        preflight = lambda *_: (_ for _ in ()).throw(error)
    elif stage == "start-save":
        monkeypatch.setattr(p, "_save_trainable", lambda *_: (_ for _ in ()).throw(error))
        preflight = lambda *_: {}
    elif stage == "timeout":
        previous_handler = p.signal.getsignal(p.signal.SIGALRM)
        def deliver_timeout(*_args):
            p.signal.raise_signal(p.signal.SIGALRM)
        preflight = deliver_timeout
    elif stage == "lock-release":
        original_unlink = pathlib.Path.unlink
        def fail_release_once(path, *args, **kwargs):
            if path == work / ".ds4-ft.lock" and not release_calls:
                release_calls.append(path)
                raise OSError("injected actual release unlink")
            return original_unlink(path, *args, **kwargs)
        monkeypatch.setattr(pathlib.Path, "unlink", fail_release_once)

    try:
        result = p.run_phase(args, api=API(), preflight=preflight)
    finally:
        if stage == "timeout":
            p.signal.signal(p.signal.SIGALRM, previous_handler)

    assert result["status"] == "fail"
    assert not any(marker.exists() for marker in p._ok_marker_paths())
    assert (tmp_path / "phase-a-report.json").is_file()
    assert (work / ".ds4-segmented-pilot-phase-a-fail").is_file()
    if stage == "lock-release":
        assert release_calls == [work / ".ds4-ft.lock"]
        assert (work / ".ds4-ft.lock").is_file()
        p._release_ft_lock()
        assert release_calls == [work / ".ds4-ft.lock"]
    assert not (work / ".ds4-ft.lock").exists()


@pytest.mark.parametrize("mode", ["success", "failure", "final-aggregation", "phase-report", "final-report", "phase-marker", "final-marker"])
def test_phase_b_terminal_paths_run_end_to_end(tmp_path, monkeypatch, mode):
    p = load_pilot()
    work = tmp_path / "work"
    model = tmp_path / "model"
    data = tmp_path / "data"
    config = tmp_path / "config.json"
    resume = tmp_path / "phase-a-adapters.safetensors"
    work.mkdir(); model.mkdir(); data.mkdir(); config.write_text("{}")
    resume.write_bytes(b"resume")
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(work))
    monkeypatch.setattr(p, "PILOT_MODEL", str(model))
    monkeypatch.setattr(p, "PILOT_DATA", str(data))
    monkeypatch.setattr(p, "PILOT_CONFIG", str(config))
    monkeypatch.setattr(p, "PILOT_FINAL_REPORT", tmp_path / "pilot-report.json")
    monkeypatch.setitem(p.PHASES["phase-a"], "report", str(tmp_path / "phase-a-report.json"))
    monkeypatch.setitem(p.PHASES["phase-a"], "adapter_path", str(work / "phase-a"))
    monkeypatch.setitem(p.PHASES["phase-b"], "report", str(tmp_path / "phase-b-report.json"))
    monkeypatch.setitem(p.PHASES["phase-b"], "adapter_path", str(work / "phase-b"))
    monkeypatch.setitem(p.PHASES["phase-b"], "resume_adapter_file", str(resume))
    phase_a_report = {"status": "ok", "phase": "phase-a", "provider_calls": 2,
                      "optimizer_updates": 2, "contract_digest": "phase-a-contract",
                      "identity_manifest": {"immutable": {"identity": "same"}},
                      "steps": [{"global_step": 1}, {"global_step": 2}], "wall_seconds": 1.0}
    pathlib.Path(p.PHASES["phase-a"]["report"]).write_text(json.dumps(phase_a_report))
    p._marker_path("phase-a", "ok").write_text(json.dumps({"status": "ok", "phase": "phase-a"}))
    monkeypatch.setattr(p, "_read_config", lambda *_: {})
    monkeypatch.setattr(p, "validate_pins", lambda *_: {"command": []})
    monkeypatch.setattr(p, "validate_phase_b_dependency", lambda *_: None)
    source_info = {"canonical_tensor_digest_v1": "source-digest", "file_sha256": p.file_sha256(resume)}
    monkeypatch.setattr(p, "canonical_tensor_digest", lambda *_: dict(source_info))
    monkeypatch.setattr(p, "_validate_artifacts", lambda *_: {
        "start": {"canonical_tensor_digest_v1": "source-digest", "tensors": []},
        "checkpoints": [{"canonical_tensor_digest_v1": "final-digest", "tensors": []}],
        "final": {"canonical_tensor_digest_v1": "final-digest", "tensors": []},
    })
    install_tmp_fs_guard(monkeypatch, tmp_path)
    args = p.build_parser().parse_args([
        "--phase", "phase-b", "--model", str(model), "--data", str(data),
        "--adapter-path", str(work / "phase-b"), "--config", str(config),
        "--resume-adapter-file", str(resume), "--train", "--fine-tune-type", "lora",
        "--num-layers", "16", "--iters", "1", "--batch-size", "1",
        "--learning-rate", "1e-5", "--max-seq-length", "4096", "--mask-prompt",
        "--grad-checkpoint", "--grad-accumulation-steps", "1", "--seed", "0",
        "--optimizer", "adam", "--val-batches", "25", "--steps-per-report", "1",
        "--steps-per-eval", "1", "--save-every", "1", "--segment-size", "1",
    ])
    if mode == "failure":
        preflight = lambda *_: (_ for _ in ()).throw(p.PilotError("injected Phase B failure"))
    else:
        preflight = lambda *_: {"immutable": {"identity": "same"}}
    def fake_execute(_args, _phase, _api, _output, provider, callback, config=None):
        provider.records.append({"local_step": 1, "provider_call": 1, "loss": 1.0, "n_tokens": 1,
                                 "gradients_finite": True, "gradient_schema": [{"path": "x", "shape": [1], "dtype": "float32"}],
                                 "gradient_leaf_count": 1, "expected_mask_tokens": 1, "mask_tokens_match": True})
        callback.records.append({"local_step": 1, "global_step": 3})
        callback.validation_records.append({"iteration": 0, "val_loss": 1.0, "val_time": 0.1})
    monkeypatch.setattr(p, "_execute_training", fake_execute)
    if mode == "final-aggregation":
        monkeypatch.setattr(p, "aggregate_reports", lambda *_: (_ for _ in ()).throw(p.PilotError("injected final aggregation")))
    elif mode in {"phase-report", "final-report", "phase-marker", "final-marker"}:
        target = {
            "phase-report": "phase-b-report.json", "final-report": "pilot-report.json",
            "phase-marker": ".ds4-segmented-pilot-phase-b-ok", "final-marker": ".ds4-segmented-pilot-ok",
        }[mode]
        original_atomic_write = p.atomic_write_json
        injected = []
        def fail_evidence_boundary(path, value):
            if pathlib.Path(path).name == target and not injected:
                injected.append(path)
                raise OSError(f"injected {mode}")
            return original_atomic_write(path, value)
        monkeypatch.setattr(p, "atomic_write_json", fail_evidence_boundary)
    result = p.run_phase(args, api={"tree_flatten": fake_tree_flatten, "mx": FakeMX()}, preflight=preflight)
    assert result["status"] == ("ok" if mode == "success" else "fail"), result
    assert (tmp_path / "phase-b-report.json").is_file()
    assert (work / ".ds4-ft.lock").exists() is False
    if mode == "success":
        assert (work / ".ds4-segmented-pilot-phase-b-ok").is_file()
        assert (work / ".ds4-segmented-pilot-ok").is_file()
        assert json.loads((p.PILOT_FINAL_REPORT).read_text())["global_progression"] == [0, 1, 2, 3]
    else:
        assert not any(marker.exists() for marker in p._ok_marker_paths())
        assert (work / ".ds4-segmented-pilot-phase-b-fail").is_file()


def test_synthetic_terminal_filesystem_operations_stay_under_resolved_tmp_root(tmp_path, monkeypatch):
    p = load_pilot()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(work))
    monkeypatch.setattr(p, "PILOT_FINAL_REPORT", tmp_path / "pilot-report.json")
    monkeypatch.setitem(p.PHASES["phase-a"], "report", str(tmp_path / "phase-a-report.json"))
    root = install_tmp_fs_guard(monkeypatch, tmp_path)
    p._acquire_ft_lock()
    p._release_ft_lock()
    p._write_failure_evidence("phase-a", {"status": "fail", "phase": "phase-a", "error": "synthetic"})
    assert root == tmp_path.resolve()


def test_failure_markers_bind_surviving_reports_after_success_rollback(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(work))
    monkeypatch.setattr(p, "PILOT_FINAL_REPORT", tmp_path / "pilot-report.json")
    monkeypatch.setitem(p.PHASES["phase-b"], "report", str(tmp_path / "phase-b-report.json"))
    p._marker_path("phase-b", "ok").write_text("stale")
    p._write_failure_evidence("phase-b", {"status": "fail", "phase": "phase-b", "error": "x"})
    assert all(not marker.exists() for marker in p._ok_marker_paths())
    for marker in (p._marker_path("phase-b", "fail"), work / ".ds4-segmented-pilot-fail"):
        payload = json.loads(marker.read_text())
        report_path = pathlib.Path(payload["report_path"])
        assert payload["status"] == "fail"
        assert payload["report_sha256"] == p.file_sha256(report_path)


def test_run_phase_fake_terminal_path_writes_report_then_marker(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    model = tmp_path / "model"
    data = tmp_path / "data"
    model.mkdir(); data.mkdir()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0, "keys": ["self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_proj"]}}))
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(work))
    monkeypatch.setattr(p, "PILOT_MODEL", str(model))
    monkeypatch.setattr(p, "PILOT_DATA", str(data))
    monkeypatch.setattr(p, "PILOT_CONFIG", str(config))
    monkeypatch.setattr(p, "PILOT_FINAL_REPORT", tmp_path / "pilot-report.json")
    monkeypatch.setitem(p.PHASES["phase-a"], "adapter_path", str(work / "phase-a"))
    monkeypatch.setitem(p.PHASES["phase-a"], "report", str(tmp_path / "phase-a-report.json"))
    args = p.build_parser().parse_args([
        "--phase", "phase-a", "--model", str(model), "--data", str(data), "--adapter-path", str(work / "phase-a"),
        "--config", str(config), "--train", "--fine-tune-type", "lora", "--num-layers", "16", "--iters", "2",
        "--batch-size", "1", "--learning-rate", "1e-5", "--max-seq-length", "4096", "--mask-prompt",
        "--grad-checkpoint", "--grad-accumulation-steps", "1", "--seed", "0", "--optimizer", "adam",
        "--val-batches", "25", "--steps-per-report", "1", "--steps-per-eval", "2", "--save-every", "1", "--segment-size", "1",
    ])

    class Array:
        shape = (1,)
        dtype = "float32"

    class Scalar:
        def __init__(self, dtype, value):
            self.shape = ()
            self.dtype = dtype
            self.value = value
        def item(self):
            return self.value

    class Tokens:
        shape = (1, 2)
        dtype = "int32"

    class Lengths:
        shape = (1, 2)
        dtype = "int32"
        def tolist(self):
            return [[0, 1]]

    def save(path, _values):
        name = pathlib.Path(path).name
        raw = b"a" if "phase-a-start" in name else b"b" if "0000001" in name else b"c"
        write_safetensors(pathlib.Path(path), {"x": ("U8", [1], raw)})

    class MX:
        float32 = "float32"
        int32 = "int32"
        class Random:
            def seed(self, _value):
                pass
        random = Random()
        eval = staticmethod(lambda *_: None)
        save_safetensors = staticmethod(save)

    class Model:
        def freeze(self): pass
        def trainable_parameters(self): return {"x": Array()}
        def load_weights(self, *_args, **_kwargs): pass

    class TrainingArgsType:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)

    class API:
        mx = MX()
        tree_flatten = staticmethod(fake_tree_flatten)
        TrainingArgs = TrainingArgsType
        def load(self, *_args, **_kwargs): return Model(), object()
        def load_dataset(self, *_args): return [], [], []
        def linear_to_lora_layers(self, *_args, **_kwargs): pass
        def make_provider(self, **_kwargs): return lambda *_args: ((Scalar("float32", 1.0), Scalar("int32", 1)), {"x": Array()})
        def make_optimizer(self, _lr): return object()
        def train(self, model, optimizer, train_dataset, val_dataset, **kwargs):
            provider, callback, training_args = kwargs["loss_and_grad"], kwargs["training_callback"], kwargs["args"]
            for step, byte in ((1, b"b"), (2, b"c")):
                callback.on_val_loss_report({"iteration": step - 1, "val_loss": 1.0, "val_time": 0.1})
                provider(model, Tokens(), Lengths())
                callback.on_train_loss_report({"iteration": step, "train_loss": 1.0, "learning_rate": 1e-5, "tokens_per_second": 1.0, "iterations_per_second": 1.0})
                save(str(pathlib.Path(training_args.adapter_file).parent / f"{step:07d}_adapters.safetensors"), {})
            save(training_args.adapter_file, {})

    report = p.run_phase(args, api=API(), preflight=lambda *_: {"identity": "fake"})
    assert report["status"] == "ok"
    saved_report = json.loads((tmp_path / "phase-a-report.json").read_text())
    assert saved_report["provider_calls"] == 2
    assert saved_report["commands"]
    assert saved_report["lock_lifecycle"] == {"path": str(work / ".ds4-ft.lock"), "acquired": True, "released": True, "release_attempts": 1}
    marker = json.loads(p._marker_path("phase-a", "ok").read_text())
    assert marker["output_path"] == str(work / "phase-a")
    assert marker["timestamp"] > 0
    assert marker["exit_code"] == 0
    assert marker["contract_digest"] == saved_report["contract_digest"]


def _fake_psutil(monkeypatch, pilot, processes, available=64 * 1024**3, disk=2 * 1024**3):
    class Memory:
        def __init__(self, rss):
            self.rss = rss

    class VM:
        def __init__(self):
            self.available = available

    class FakePsutil:
        AccessDenied = type("AccessDenied", (Exception,), {})
        NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        ZombieProcess = type("ZombieProcess", (FakePsutil if False else Exception,), {})
        def virtual_memory(self):
            return VM()
        def process_iter(self, _attrs):
            return iter(processes)

    class Disk:
        free = disk

    fake = FakePsutil()
    for proc in processes:
        proc._memory = getattr(proc, "_memory", None)
    monkeypatch.setitem(sys.modules, "psutil", fake)
    monkeypatch.setattr(pilot, "shutil_disk_free", lambda _path: disk)
    return fake, Memory


def test_resource_gate_skips_only_allowed_process_races_and_preserves_sorted_evidence(monkeypatch):
    p = load_pilot()
    fake, Memory = _fake_psutil(monkeypatch, p, [])
    class Process:
        def __init__(self, pid, outcome):
            self.pid = pid
            self.outcome = outcome
        def memory_info(self):
            if isinstance(self.outcome, BaseException):
                raise self.outcome
            return Memory(self.outcome)
    denied = fake.AccessDenied(); denied.pid = 0
    gone = fake.NoSuchProcess(); gone.pid = 101
    zombie = fake.ZombieProcess(); zombie.pid = 202
    processes = [Process(9, 4), Process(101, gone), Process(0, denied), Process(202, zombie), Process(3, 8)]
    fake.process_iter = lambda _attrs: iter(processes)
    monkeypatch.setattr(p.os, "getpid", lambda: 999)
    evidence = p._resource_gate()
    assert evidence["competing_processes"] == [[3, 8], [9, 4]]
    assert evidence["allowed_process_skips"] == {
        "total": 3,
        "counts_by_type": {"AccessDenied": 1, "NoSuchProcess": 1, "ZombieProcess": 1},
        "pids_by_type": {"AccessDenied": [0], "NoSuchProcess": [101], "ZombieProcess": [202]},
        "unknown_pid_counts_by_type": {"AccessDenied": 0, "NoSuchProcess": 0, "ZombieProcess": 0},
    }


def test_resource_gate_keeps_strict_resource_boundaries_and_current_pid_exclusion(monkeypatch):
    p = load_pilot()
    fake, Memory = _fake_psutil(monkeypatch, p, [])
    class Process:
        def __init__(self, pid, rss):
            self.pid, self.rss = pid, rss
        def memory_info(self):
            assert self.pid != 77
            return Memory(self.rss)
    fake.process_iter = lambda _attrs: iter([Process(77, 100 * 1024**3), Process(8, 50 * 1024**3)])
    monkeypatch.setattr(p.os, "getpid", lambda: 77)
    assert p._resource_gate()["competing_processes"] == [[8, 50 * 1024**3]]
    fake.virtual_memory = lambda: type("VM", (), {"available": 32 * 1024**3 - 1})()
    with pytest.raises(p.PilotError, match="32 GiB"):
        p._resource_gate()


def test_resource_gate_unknown_process_error_fails_closed_with_structured_context(monkeypatch):
    p = load_pilot()
    fake, Memory = _fake_psutil(monkeypatch, p, [])
    class Process:
        pid = 123
        def memory_info(self):
            raise RuntimeError("observer drift")
    fake.process_iter = lambda _attrs: iter([Process()])
    with pytest.raises(p.ResourceObserverError) as caught:
        p._resource_gate()
    assert caught.value.stage == "process-rss"
    assert caught.value.exception_type == "RuntimeError"
    assert caught.value.pid == 123


def test_resource_gate_competitor_and_disk_boundaries_remain_strict(monkeypatch):
    p = load_pilot()
    fake, Memory = _fake_psutil(monkeypatch, p, [], disk=1 * 1024**3)
    class Process:
        pid = 321
        def memory_info(self):
            return Memory(50 * 1024**3 + 1)
    fake.process_iter = lambda _attrs: iter([Process()])
    with pytest.raises(p.PilotError, match="competing process"):
        p._resource_gate()
    class AtLimit:
        pid = 321
        def memory_info(self):
            return Memory(0)
    fake.process_iter = lambda _attrs: iter([AtLimit()])
    assert p._resource_gate()["disk_free"] == 1 * 1024**3
    monkeypatch.setattr(p, "shutil_disk_free", lambda _path: 1 * 1024**3 - 1)
    with pytest.raises(p.PilotError, match="disk free"):
        p._resource_gate()


def test_resource_gate_enumeration_and_oserror_fail_closed(monkeypatch):
    p = load_pilot()
    fake, Memory = _fake_psutil(monkeypatch, p, [])
    fake.process_iter = lambda _attrs: (_ for _ in ()).throw(OSError("enumeration drift"))
    with pytest.raises(p.ResourceObserverError) as caught:
        p._resource_gate()
    assert caught.value.stage == "process-enumeration"
    class Process:
        pid = 44
        def memory_info(self):
            raise OSError("rss drift")
    fake.process_iter = lambda _attrs: iter([Process()])
    with pytest.raises(p.ResourceObserverError) as caught:
        p._resource_gate()
    assert caught.value.stage == "process-rss"
    assert caught.value.exception_type == "OSError"


@pytest.mark.parametrize("stage", ["iterator", "pid", "virtual-memory", "disk-free", "resource-evidence"])
def test_resource_gate_observer_failure_matrix_is_structured_and_fail_closed(monkeypatch, stage):
    p = load_pilot()
    fake, Memory = _fake_psutil(monkeypatch, p, [])
    if stage == "iterator":
        class BrokenIterator:
            def __iter__(self): return self
            def __next__(self): raise RuntimeError("iterator drift")
        fake.process_iter = lambda _attrs: BrokenIterator()
    elif stage == "pid":
        class Process:
            @property
            def pid(self): raise RuntimeError("pid drift")
        fake.process_iter = lambda _attrs: iter([Process()])
    elif stage == "virtual-memory":
        class BrokenInt:
            def __int__(self): raise RuntimeError("memory conversion drift")
        fake.virtual_memory = lambda: type("VM", (), {"available": BrokenInt()})()
    elif stage == "disk-free":
        monkeypatch.setattr(p, "shutil_disk_free", lambda _path: (_ for _ in ()).throw(RuntimeError("disk drift")))
    else:
        monkeypatch.setattr(p, "_resource_evidence", lambda *_args: (_ for _ in ()).throw(ValueError("evidence drift")))
    with pytest.raises(p.ResourceObserverError) as caught:
        p._resource_gate()
    expected = {"iterator": "process-enumeration", "pid": "process-pid", "virtual-memory": "virtual-memory",
                "disk-free": "disk-free", "resource-evidence": "resource-evidence"}[stage]
    assert caught.value.stage == expected
    assert caught.value.exception_type


def test_resource_gate_unknown_skip_accounting_is_repeated_and_deterministic(monkeypatch):
    p = load_pilot()
    fake, Memory = _fake_psutil(monkeypatch, p, [])
    class Process:
        @property
        def pid(self):
            error = fake.NoSuchProcess()
            raise error
    fake.process_iter = lambda _attrs: iter([Process(), Process()])
    evidence = p._resource_gate()
    assert evidence["allowed_process_skips"]["counts_by_type"]["NoSuchProcess"] == 2
    assert evidence["allowed_process_skips"]["pids_by_type"]["NoSuchProcess"] == []
    assert evidence["allowed_process_skips"]["unknown_pid_counts_by_type"]["NoSuchProcess"] == 2


@pytest.mark.parametrize("error_name", ["NoSuchProcess", "ZombieProcess"])
def test_resource_gate_later_competitor_survives_allowed_process_race(monkeypatch, error_name):
    p = load_pilot()
    fake, Memory = _fake_psutil(monkeypatch, p, [])
    class Process:
        def __init__(self, pid, outcome):
            self.pid, self.outcome = pid, outcome
        def memory_info(self):
            if isinstance(self.outcome, BaseException):
                raise self.outcome
            return Memory(self.outcome)
    vanished = getattr(fake, error_name)()
    vanished.pid = 41
    fake.process_iter = lambda _attrs: iter([
        Process(41, vanished), Process(99, 50 * 1024**3 + 1),
    ])
    with pytest.raises(p.PilotError, match="competing process"):
        p._resource_gate()


@pytest.mark.parametrize("stage", ["process-pid", "process-rss"])
def test_resource_gate_integer_normalization_errors_fail_closed(monkeypatch, stage):
    p = load_pilot()
    fake, Memory = _fake_psutil(monkeypatch, p, [])
    if stage == "process-pid":
        class Process:
            @property
            def pid(self):
                return "not-an-int"
        fake.process_iter = lambda _attrs: iter([Process()])
    else:
        class Process:
            pid = 55
            def memory_info(self):
                return type("RSS", (), {"rss": "not-an-int"})()
        fake.process_iter = lambda _attrs: iter([Process()])
    with pytest.raises(p.ResourceObserverError) as caught:
        p._resource_gate()
    assert caught.value.stage == stage
    assert caught.value.pid == (None if stage == "process-pid" else 55)


def test_attempt3_namespace_is_exact_and_attempt1_manifest_is_separate():
    p = load_pilot()
    paths = p.attempt3_namespace()
    assert paths["phase-a-output"] == pathlib.Path("/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-a")
    assert paths["phase-b-resume"] == paths["phase-a-output"] / "0000002_adapters.safetensors"
    assert paths["phase-a-log"] == p.REPO_ROOT / "agent-output/cmux-14-5-attempt-3/phase-a-log.txt"
    assert paths["final-report"] == p.REPO_ROOT / "agent-output/cmux-14-5-attempt-3/pilot-report.json"
    assert all("attempt-3" in str(path) for key, path in paths.items() if key not in {"phase-a-output", "phase-b-output", "phase-b-resume"})
    assert all("attempt-3" not in item["path"] for item in p.ATTEMPT1_EVIDENCE_SHA256)


def test_attempt3_namespace_binds_every_exact_destination_and_has_no_future_suffix(tmp_path):
    p = load_pilot()
    paths = p.attempt3_namespace(repo_root=tmp_path, workspace=tmp_path / "workspace")
    assert set(paths) == {
        "phase-a-output", "phase-a-start-checkpoint", "phase-a-step1-checkpoint",
        "phase-a-step2-checkpoint", "phase-a-final-checkpoint", "phase-a-config",
        "phase-a-log", "phase-a-report", "phase-a-ok", "phase-a-fail",
        "phase-b-output", "phase-b-start-checkpoint", "phase-b-step1-checkpoint",
        "phase-b-final-checkpoint", "phase-b-config", "phase-b-resume",
        "phase-b-log", "phase-b-report", "phase-b-ok", "phase-b-fail",
        "final-report", "final-ok", "final-fail",
    }
    assert all("attempt-3" in str(path) for path in paths.values())
    assert not any("attempt-3-2" in str(path) or "attempt-4" in str(path) for path in paths.values())


@pytest.mark.parametrize("key", [
    "phase-a-output", "phase-a-start-checkpoint", "phase-a-step1-checkpoint",
    "phase-a-step2-checkpoint", "phase-a-final-checkpoint", "phase-a-config",
    "phase-a-log", "phase-a-report", "phase-a-ok", "phase-a-fail",
    "phase-b-output", "phase-b-start-checkpoint", "phase-b-step1-checkpoint",
    "phase-b-final-checkpoint", "phase-b-config", "phase-b-log", "phase-b-report",
    "phase-b-ok", "phase-b-fail", "final-report", "final-ok", "final-fail",
])
def test_attempt3_phase_a_collision_matrix_is_fail_closed(tmp_path, monkeypatch, key):
    p = load_pilot()
    paths = p.attempt3_namespace(repo_root=tmp_path, workspace=tmp_path / "workspace")
    collision = paths[key]
    collision.parent.mkdir(parents=True, exist_ok=True)
    if key.endswith("output"):
        collision.mkdir()
    else:
        collision.write_bytes(b"historical")
    monkeypatch.setattr(p, "verify_attempt1_historical_evidence", lambda **_: [])
    with pytest.raises(p.PilotError, match="collision"):
        p.check_attempt3_launch("phase-a", paths["phase-a-log"], repo_root=tmp_path, workspace=tmp_path / "workspace")
    assert collision.exists()
    assert not paths["phase-a-report"].exists() if key != "phase-a-report" else True


@pytest.mark.parametrize("key", [
    "phase-b-output", "phase-b-start-checkpoint", "phase-b-step1-checkpoint",
    "phase-b-final-checkpoint", "phase-b-config", "phase-b-log", "phase-b-report",
    "phase-b-ok", "phase-b-fail", "final-report", "final-ok", "final-fail",
])
def test_attempt3_phase_b_collision_matrix_is_fail_closed(tmp_path, monkeypatch, key):
    p = load_pilot()
    paths, report, _ = _write_valid_attempt3_phase_a(tmp_path, p)
    collision = paths[key]
    collision.parent.mkdir(parents=True, exist_ok=True)
    if key.endswith("output"):
        collision.mkdir()
    else:
        collision.write_bytes(b"historical")
    monkeypatch.setattr(p, "verify_attempt1_historical_evidence", lambda **_: report["attempt_1_historical_evidence"])
    with pytest.raises(p.PilotError, match=re.escape(str(collision))):
        p.check_attempt3_launch("phase-b", paths["phase-b-log"], repo_root=tmp_path, workspace=tmp_path / "workspace")
    assert collision.exists()


def test_attempt3_phase_b_admits_valid_a2_evidence_before_b2_destinations(tmp_path, monkeypatch):
    p = load_pilot()
    paths, report, _ = _write_valid_attempt3_phase_a(tmp_path, p)
    monkeypatch.setattr(p, "verify_attempt1_historical_evidence", lambda **_: report["attempt_1_historical_evidence"])
    result = p.check_attempt3_launch("phase-b", paths["phase-b-log"], repo_root=tmp_path, workspace=tmp_path / "workspace")
    assert result["phase"] == "phase-b"
    assert result["log_path"] == str(paths["phase-b-log"])
    assert not paths["phase-b-output"].exists()
    assert not paths["final-report"].exists()


def test_attempt3_dependency_rejects_report_marker_checkpoint_and_log_mutations(tmp_path):
    p = load_pilot()
    paths, report, marker = _write_valid_attempt3_phase_a(tmp_path, p)
    cases = ("report_sha256", "contract_digest", "resume_source_file_sha256", "resume_source_canonical_tensor_digest_v1")
    for field in cases:
        mutated_marker = dict(marker)
        mutated_marker[field] = "mutated"
        with pytest.raises(p.PilotError):
            p.validate_phase_b_dependency(report, mutated_marker, paths["phase-b-resume"],
                                          phase_spec=p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"])
    mutated_report = dict(report)
    mutated_report["effective"] = {"command": ["synthetic"], "log_path": str(tmp_path / "wrong.log")}
    with pytest.raises(p.PilotError):
        p.validate_phase_b_dependency(mutated_report, marker, paths["phase-b-resume"],
                                      phase_spec=p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"])


def test_attempt3_historical_manifest_path_size_and_digest_are_verified(tmp_path, monkeypatch):
    p = load_pilot()
    historical = tmp_path / "historical.bin"
    historical.write_bytes(b"abc")
    entry = {"path": historical.name, "size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    monkeypatch.setattr(p, "ATTEMPT1_EVIDENCE_SHA256", (entry,))
    assert p.verify_attempt1_historical_evidence(repo_root=tmp_path)[0]["sha256"] == entry["sha256"]
    for field, value in (("path", "missing.bin"), ("size", 4), ("sha256", "0" * 64)):
        mutated = dict(entry, **{field: value})
        monkeypatch.setattr(p, "ATTEMPT1_EVIDENCE_SHA256", (mutated,))
        with pytest.raises(p.PilotError):
            p.verify_attempt1_historical_evidence(repo_root=tmp_path)


def test_attempt3_log_fd_attestation_rejects_wrong_descriptor(tmp_path):
    p = load_pilot()
    log = tmp_path / "phase-a-log.txt"
    log.write_bytes(b"log")
    with log.open("ab") as handle:
        p._attest_log_fd(log, handle.fileno())
    other = tmp_path / "other.log"
    other.write_bytes(b"other")
    with other.open("ab") as handle:
        with pytest.raises(p.PilotError, match="FD collision"):
            p._attest_log_fd(log, handle.fileno())


def test_attempt3_collision_rejects_before_any_mutation(tmp_path, monkeypatch):
    p = load_pilot()
    namespace = p.attempt3_namespace(repo_root=tmp_path, workspace=tmp_path / "workspace")
    collision = namespace["phase-a-log"]
    collision.parent.mkdir(parents=True)
    collision.write_bytes(b"historical")
    monkeypatch.setattr(p, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(tmp_path / "workspace"))
    with pytest.raises(p.PilotError, match="collision"):
        p.check_attempt3_launch("phase-a", collision)
    assert collision.read_bytes() == b"historical"
    assert not (tmp_path / "workspace").exists()


def test_attempt3_namespace_contains_all_exact_checkpoint_and_config_paths():
    p = load_pilot()
    paths = p.attempt3_namespace(repo_root=pathlib.Path("/tmp/repo"), workspace=pathlib.Path("/tmp/work"))
    assert set(paths) == {
        "phase-a-output", "phase-a-start-checkpoint", "phase-a-step1-checkpoint",
        "phase-a-step2-checkpoint", "phase-a-final-checkpoint", "phase-a-config",
        "phase-a-log", "phase-a-report", "phase-a-ok", "phase-a-fail",
        "phase-b-output", "phase-b-resume", "phase-b-start-checkpoint",
        "phase-b-step1-checkpoint", "phase-b-final-checkpoint", "phase-b-config",
        "phase-b-log", "phase-b-report", "phase-b-ok", "phase-b-fail",
        "final-report", "final-ok", "final-fail",
    }


def test_canonical_attempt3_validator_rejects_incomplete_report_before_dependency(tmp_path):
    p = load_pilot()
    paths, report, _ = _write_valid_attempt3_phase_a(tmp_path, p)
    report.pop("provider_evidence")
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
    with pytest.raises(p.PilotError, match="canonical A3 report"):
        p.validate_canonical_attempt3_report(report, phase_spec=spec,
                                             report_path=paths["phase-a-report"])


@pytest.mark.parametrize("mutation", [
    "effective", "identity", "provider", "validation", "artifact", "commands", "historical",
    "namespace", "output", "report", "resume", "contract", "coordinated",
])
def test_canonical_attempt3_report_mutation_matrix_is_fail_closed(tmp_path, mutation):
    p = load_pilot()
    paths, original, _ = _write_valid_attempt3_phase_a(tmp_path, p)
    report = json.loads(json.dumps(original))
    if mutation == "effective":
        report["effective"]["learning_rate"] = 2e-5
    elif mutation == "identity":
        report["identity_manifest"]["immutable"]["provider_sha256"] = "0" * 64
    elif mutation == "provider":
        report["provider_evidence"][0]["n_tokens"] = 2
    elif mutation == "validation":
        report["validation_evidence"][0]["iteration"] = 99
    elif mutation == "artifact":
        report["artifacts"]["checkpoints"][1]["path"] = str(tmp_path / "substituted.safetensors")
    elif mutation == "commands":
        report["commands"] = ["substituted"]
    elif mutation == "historical":
        report["attempt_1_historical_evidence"].append({"path": "extra", "size": 0, "sha256": "0" * 64})
    elif mutation == "namespace":
        report["namespace_paths"]["phase-a-step2-checkpoint"] = str(tmp_path / "substituted.safetensors")
    elif mutation == "output":
        report["output_path"] = str(tmp_path / "substituted-output")
    elif mutation == "report":
        report["report_path"] = str(tmp_path / "substituted-report.json")
    elif mutation == "resume":
        report["resume_source"]["canonical_tensor_digest_v1"] = "0" * 64
    elif mutation == "contract":
        report["contract_digest"] = "0" * 64
    elif mutation == "coordinated":
        report["effective"]["model"] = "/substituted-model"
        report["identity_manifest"]["immutable"]["git_head"] = "2" * 40
        report["contract_digest"] = p.contract_digest("phase-a", report["effective"], report["identity_manifest"],
                                                       phase_spec=p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"])
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
    with pytest.raises(p.PilotError):
        p.validate_canonical_attempt3_report(report, phase_spec=spec, report_path=paths["phase-a-report"])


def test_canonical_attempt3_rejects_coordinated_command_and_identity_substitutions(tmp_path):
    p = load_pilot()
    paths, original, _ = _write_valid_attempt3_phase_a(tmp_path, p)
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
    spec["trusted_identity"] = original["identity_manifest"]
    report = json.loads(json.dumps(original))
    report["effective"]["command"] = ["substituted-command"]
    report["commands"] = ["substituted-command"]
    report["identity_manifest"]["immutable"]["git_head"] = "2" * 40
    report["contract_digest"] = p.contract_digest("phase-a", report["effective"], report["identity_manifest"], phase_spec=spec)
    with pytest.raises(p.PilotError):
        p.validate_canonical_attempt3_report(report, phase_spec=spec, report_path=paths["phase-a-report"])


def test_canonical_attempt3_rejects_provider_schema_substitution_after_digest_recompute(tmp_path):
    p = load_pilot()
    paths, original, _ = _write_valid_attempt3_phase_a(tmp_path, p)
    report = json.loads(json.dumps(original))
    report["provider_evidence"][0].pop("gradient_paths")
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
    report["contract_digest"] = p.contract_digest("phase-a", report["effective"], report["identity_manifest"], phase_spec=spec)
    with pytest.raises(p.PilotError):
        p.validate_canonical_attempt3_report(report, phase_spec=spec, report_path=paths["phase-a-report"])


@pytest.mark.parametrize("mutation", ["provider_swap", "update_swap", "elapsed", "learning_rate_bool",
                                       "loss_bool", "gradient_path", "gradient_duplicate_path", "gradient_shape", "gradient_dtype"])
def test_canonical_attempt3_rejects_coordinated_provider_update_value_mutations(tmp_path, mutation):
    p = load_pilot()
    paths, original, _ = _write_valid_attempt3_phase_a(tmp_path, p)
    report = json.loads(json.dumps(original))
    if mutation == "provider_swap":
        report["provider_evidence"][0]["provider_call"], report["provider_evidence"][1]["provider_call"] = 2, 1
    elif mutation == "update_swap":
        report["steps"][0]["local_step"], report["steps"][1]["local_step"] = 2, 1
        report["steps"][0]["optimizer_update_ordinal"], report["steps"][1]["optimizer_update_ordinal"] = 2, 1
    elif mutation == "elapsed":
        report["provider_evidence"][0]["provider_elapsed_seconds"] = "0.1"
    elif mutation == "learning_rate_bool":
        report["steps"][0]["learning_rate"] = True
    elif mutation == "loss_bool":
        report["provider_evidence"][0]["loss"] = True
    elif mutation == "gradient_path":
        report["provider_evidence"][0]["gradient_schema"][0]["path"] = "bad path"
        report["provider_evidence"][0]["gradient_paths"] = ["bad path"]
        report["provider_evidence"][0]["gradient_shapes"] = {"bad path": [1]}
        report["provider_evidence"][0]["gradient_dtypes"] = {"bad path": "float32"}
    elif mutation == "gradient_duplicate_path":
        report["provider_evidence"][0]["gradient_schema"] = [
            {"path": "x", "shape": [1], "dtype": "float32"},
            {"path": "x", "shape": [1], "dtype": "float32"},
        ]
        report["provider_evidence"][0]["gradient_leaf_count"] = 2
        report["provider_evidence"][0]["gradient_paths"] = ["x", "x"]
        report["provider_evidence"][0]["gradient_shapes"] = {"x": [1]}
        report["provider_evidence"][0]["gradient_dtypes"] = {"x": "float32"}
    elif mutation == "gradient_shape":
        report["provider_evidence"][0]["gradient_schema"][0]["shape"] = ["1"]
        report["provider_evidence"][0]["gradient_shapes"]["x"] = ["1"]
    elif mutation == "gradient_dtype":
        report["provider_evidence"][0]["gradient_schema"][0]["dtype"] = "not-a-dtype"
        report["provider_evidence"][0]["gradient_dtypes"]["x"] = "not-a-dtype"
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
    with pytest.raises(p.PilotError):
        p.validate_canonical_attempt3_report(report, phase_spec=spec, report_path=paths["phase-a-report"])


def test_validate_artifacts_consumes_canonical_namespace_entries(tmp_path):
    p = load_pilot()
    paths, _, _ = _write_valid_attempt3_phase_a(tmp_path, p)
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
    mutated = dict(spec)
    mutated["namespace_paths"] = dict(spec["namespace_paths"])
    mutated["namespace_paths"]["phase-a-config"] = str(tmp_path / "wrong-config.json")
    with pytest.raises(p.PilotError, match="missing required pilot artifact"):
        p._validate_artifacts("phase-a", paths["phase-a-output"], phase_spec=mutated)


@pytest.mark.parametrize("key", [
    "phase-a-config", "phase-a-start-checkpoint", "phase-a-step1-checkpoint",
    "phase-a-step2-checkpoint", "phase-a-final-checkpoint",
])
def test_validate_artifacts_binds_each_phase_a_checkpoint_and_config_path(tmp_path, key):
    p = load_pilot()
    paths, _, _ = _write_valid_attempt3_phase_a(tmp_path, p)
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
    mutated = dict(spec)
    mutated["namespace_paths"] = dict(spec["namespace_paths"])
    mutated["namespace_paths"][key] = str(tmp_path / "mutated" / key)
    with pytest.raises(p.PilotError):
        p._validate_artifacts("phase-a", paths["phase-a-output"], phase_spec=mutated)


@pytest.mark.parametrize("key", ["phase-b-config", "phase-b-start-checkpoint", "phase-b-step1-checkpoint", "phase-b-final-checkpoint"])
def test_validate_artifacts_binds_each_phase_b_checkpoint_and_config_path(tmp_path, key):
    p = load_pilot()
    workspace = tmp_path / "workspace"
    paths = p.attempt3_namespace(repo_root=tmp_path, workspace=workspace)
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=workspace)["phase-b"]
    paths["phase-b-output"].mkdir(parents=True)
    for name, payload in (("resume-start.safetensors", b"a"), ("0000001_adapters.safetensors", b"b"), ("adapters.safetensors", b"b")):
        write_safetensors(paths["phase-b-output"] / name, {"x": ("U8", [1], payload)})
    paths["phase-b-config"].write_text("{}", encoding="utf-8")
    mutated = dict(spec)
    mutated["namespace_paths"] = dict(spec["namespace_paths"])
    mutated["namespace_paths"][key] = str(tmp_path / "mutated" / key)
    with pytest.raises(p.PilotError):
        p._validate_artifacts("phase-b", paths["phase-b-output"], phase_spec=mutated)


def _r8_rebind_consumer(report, phase_spec, p, tmp_path, key):
    rebound_report = json.loads(json.dumps(report))
    rebound_spec = json.loads(json.dumps(phase_spec))
    mutated_path = tmp_path / "mutated consumer paths" / key
    rebound_spec["namespace_paths"] = dict(phase_spec["namespace_paths"])
    rebound_report["namespace_paths"] = dict(report["namespace_paths"])
    rebound_spec["namespace_paths"][key] = str(mutated_path)
    rebound_report["namespace_paths"][key] = str(mutated_path)
    if key == "phase-b-resume":
        mutated_path.parent.mkdir(parents=True, exist_ok=True)
        write_safetensors(mutated_path, {"x": ("U8", [1], b"x")})
        rebound_spec["resume_adapter_file"] = str(mutated_path)
        rebound_report["effective"]["resume_adapter_file"] = str(mutated_path)
        command = p.canonical_attempt3_command(rebound_spec["phase"], rebound_spec)
        rebound_report["effective"]["command"] = command
        rebound_report["commands"] = command
        rebound_report["authorization"]["canonical_command_sha256"] = hashlib.sha256(
            json.dumps(command, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    rebound_report["contract_digest"] = p.contract_digest(
        rebound_report["phase"], rebound_report["effective"],
        rebound_report["identity_manifest"], phase_spec=rebound_spec,
        authorization=rebound_report.get("authorization"),
        phase_a_admission_lineage=rebound_report.get("phase_a_admission_lineage"))
    return rebound_report, rebound_spec, mutated_path


def _r8_write_dependency_snapshot(paths, report, marker, p):
    paths["phase-a-report"].write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    marker = dict(marker)
    marker["report_sha256"] = p.file_sha256(paths["phase-a-report"])
    marker["contract_digest"] = report["contract_digest"]
    paths["phase-a-ok"].write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")
    return marker


def _r8_remove_publications(paths, phase):
    keys = (f"{phase}-report", f"{phase}-ok", f"{phase}-fail", "final-report", "final-ok", "final-fail")
    for key in keys:
        paths[key].unlink(missing_ok=True)


@pytest.mark.parametrize("key", [
    "phase-a-start-checkpoint", "phase-a-step1-checkpoint", "phase-a-step2-checkpoint",
    "phase-a-final-checkpoint", "phase-a-config",
])
def test_r8_canonical_a2_consumer_matrix_reaches_named_artifact_boundary(tmp_path, key):
    p = load_pilot()
    paths, report, _ = _write_valid_attempt3_phase_a(tmp_path, p)
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
    rebound_report, rebound_spec, mutated_path = _r8_rebind_consumer(report, spec, p, tmp_path, key)
    with pytest.raises(p.PilotError, match=re.escape(f"missing required pilot artifact: {mutated_path}")) as caught:
        p.validate_canonical_attempt3_report(rebound_report, phase_spec=rebound_spec,
                                             report_path=paths["phase-a-report"])
    assert "canonical A3 namespace path substitution" not in str(caught.value)
    assert "canonical A3 contract digest mismatch" not in str(caught.value)


@pytest.mark.parametrize("key", [
    "phase-b-start-checkpoint", "phase-b-step1-checkpoint", "phase-b-final-checkpoint",
    "phase-b-config", "phase-b-resume",
])
def test_r8_canonical_b2_consumer_matrix_reaches_named_artifact_boundary(tmp_path, key):
    p = load_pilot()
    paths, report, spec, _, _ = _write_valid_attempt3_phase_b(tmp_path, p)
    rebound_report, rebound_spec, mutated_path = _r8_rebind_consumer(report, spec, p, tmp_path, key)
    expected = ("canonical A3 resume checkpoint binding mismatch" if key == "phase-b-resume"
                else f"missing required pilot artifact: {mutated_path}")
    with pytest.raises(p.PilotError, match=re.escape(expected)) as caught:
        p.validate_canonical_attempt3_report(rebound_report, phase_spec=rebound_spec,
                                             report_path=paths["phase-b-report"])
    assert "canonical A3 namespace path substitution" not in str(caught.value)
    assert "canonical A3 contract digest mismatch" not in str(caught.value)


@pytest.mark.parametrize("key", [
    "phase-a-start-checkpoint", "phase-a-step1-checkpoint", "phase-a-step2-checkpoint",
    "phase-a-final-checkpoint", "phase-a-config",
])
def test_r8_a2_success_admission_rejects_each_consumer_without_publication(tmp_path, monkeypatch, key):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    paths, report, marker = _write_valid_attempt3_phase_a(tmp_path, p)
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
    rebound_report, rebound_spec, mutated_path = _r8_rebind_consumer(report, spec, p, tmp_path, key)
    _r8_remove_publications(paths, "phase-a")
    calls = []
    real_writer = p._write_success_evidence
    monkeypatch.setattr(p, "_write_success_evidence", lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(p.PilotError, match=re.escape(f"missing required pilot artifact: {mutated_path}")):
        p.validate_canonical_attempt3_report(rebound_report, phase_spec=rebound_spec,
                                             report_path=paths["phase-a-report"])
        p._write_success_evidence("phase-a", rebound_report, phase_spec=rebound_spec)
    assert calls == []
    assert all(not paths[name].exists() for name in ("phase-a-report", "phase-a-ok", "phase-a-fail",
                                                       "final-report", "final-ok", "final-fail"))
    assert real_writer is not None


@pytest.mark.parametrize("key", [
    "phase-b-start-checkpoint", "phase-b-step1-checkpoint", "phase-b-final-checkpoint",
    "phase-b-config", "phase-b-resume",
])
def test_r8_b2_success_admission_rejects_each_consumer_without_publication(tmp_path, monkeypatch, key):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    paths, report, spec, phase_a_report, phase_a_marker = _write_valid_attempt3_phase_b(tmp_path, p)
    a2_report_bytes = paths["phase-a-report"].read_bytes()
    a2_marker_bytes = paths["phase-a-ok"].read_bytes()
    rebound_report, rebound_spec, mutated_path = _r8_rebind_consumer(report, spec, p, tmp_path, key)
    _r8_remove_publications(paths, "phase-b")
    calls = []
    monkeypatch.setattr(p, "_write_success_evidence", lambda *args, **kwargs: calls.append((args, kwargs)))
    expected = ("canonical A3 resume checkpoint binding mismatch" if key == "phase-b-resume"
                else f"missing required pilot artifact: {mutated_path}")
    with pytest.raises(p.PilotError, match=re.escape(expected)):
        p.validate_canonical_attempt3_report(rebound_report, phase_spec=rebound_spec,
                                             report_path=paths["phase-b-report"])
        p._write_success_evidence("phase-b", rebound_report, phase_spec=rebound_spec)
    assert calls == []
    assert all(not paths[name].exists() for name in ("phase-b-report", "phase-b-ok", "phase-b-fail",
                                                       "final-report", "final-ok", "final-fail"))
    assert paths["phase-a-report"].read_bytes() == a2_report_bytes
    assert paths["phase-a-ok"].read_bytes() == a2_marker_bytes
    assert phase_a_report and phase_a_marker


@pytest.mark.parametrize("key", [
    "phase-a-start-checkpoint", "phase-a-step1-checkpoint", "phase-a-step2-checkpoint",
    "phase-a-final-checkpoint", "phase-a-config",
])
def test_r8_phase_b_dependency_a2_matrix_reaches_named_consumer(tmp_path, key):
    p = load_pilot()
    paths, report, marker = _write_valid_attempt3_phase_a(tmp_path, p)
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
    rebound_report, rebound_spec, mutated_path = _r8_rebind_consumer(report, spec, p, tmp_path, key)
    rebound_marker = _r8_write_dependency_snapshot(paths, rebound_report, marker, p)
    with pytest.raises(p.PilotError, match=re.escape(f"missing required pilot artifact: {mutated_path}")) as caught:
        p.validate_phase_b_dependency(rebound_report, rebound_marker, paths["phase-b-resume"],
                                      phase_spec=rebound_spec)
    assert "canonical A3 namespace path substitution" not in str(caught.value)
    assert "Phase A report hash binding mismatch" not in str(caught.value)


def test_r8_phase_b_dependency_resume_matrix_reaches_resume_binding(tmp_path):
    p = load_pilot()
    paths, report, marker = _write_valid_attempt3_phase_a(tmp_path, p)
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
    rebound_report, rebound_spec, mutated_path = _r8_rebind_consumer(
        report, spec, p, tmp_path, "phase-b-resume")
    rebound_marker = _r8_write_dependency_snapshot(paths, rebound_report, marker, p)
    with pytest.raises(p.PilotError, match="Phase A resume source path binding missing|canonical A3 contract digest mismatch") as caught:
        p.validate_phase_b_dependency(rebound_report, rebound_marker, mutated_path,
                                      phase_spec=rebound_spec)
    assert "canonical A3 namespace path substitution" not in str(caught.value)
    assert "Phase A report hash binding mismatch" not in str(caught.value)


R8_CATALOG_APPLICABILITY = {
    "phase-a-start-checkpoint": "executable",
    "phase-a-step1-checkpoint": "N/A — derived training output",
    "phase-a-step2-checkpoint": "N/A — derived training output",
    "phase-a-final-checkpoint": "executable",
    "phase-a-config": "executable",
    "phase-b-resume": "executable",
    "phase-b-start-checkpoint": "executable",
    "phase-b-step1-checkpoint": "N/A — derived training output",
    "phase-b-final-checkpoint": "executable",
    "phase-b-config": "executable",
}
R8_EXECUTABLE_CATALOG_KEYS = (
    "phase-a-start-checkpoint", "phase-a-final-checkpoint", "phase-a-config",
    "phase-b-resume", "phase-b-start-checkpoint", "phase-b-final-checkpoint", "phase-b-config",
)
R8_DERIVED_CATALOG_NA_KEYS = (
    "phase-a-step1-checkpoint", "phase-a-step2-checkpoint", "phase-b-step1-checkpoint",
)


@pytest.mark.parametrize("key", R8_EXECUTABLE_CATALOG_KEYS)
def test_r8_catalog_each_key_changes_observable_consumer(tmp_path, monkeypatch, key):
    import scripts.finetune_ds4 as finetune

    phase = "phase-b" if key.startswith("phase-b-") else "phase-a"
    root = tmp_path / "catalog repo with spaces"
    root.mkdir()
    pilot_script = root / "pilot stub with spaces.py"
    pilot_script.write_text(
        "import json, os, pathlib, sys\n"
        "obs = pathlib.Path(os.environ['R8_OBS'])\n"
        "kind = 'launch' if '--launch-check-only' in sys.argv else 'train'\n"
        "with obs.open('a', encoding='utf-8') as stream:\n"
        "    json.dump({'kind': kind, 'argv': sys.argv[1:]}, stream); stream.write('\\n')\n"
        "if kind == 'train': print('training-output')\n",
        encoding="utf-8",
    )
    namespace_root = root / "namespace with spaces"
    paths = {name: namespace_root / name for name in (
        "phase-a-output", "phase-a-start-checkpoint", "phase-a-step1-checkpoint", "phase-a-step2-checkpoint",
        "phase-a-final-checkpoint", "phase-a-config", "phase-a-log", "phase-a-report", "phase-a-ok", "phase-a-fail",
        "phase-b-output", "phase-b-start-checkpoint", "phase-b-step1-checkpoint", "phase-b-final-checkpoint",
        "phase-b-config", "phase-b-resume", "phase-b-log", "phase-b-report", "phase-b-ok", "phase-b-fail",
        "final-report", "final-ok", "final-fail")}
    identity = {name: root / name for name in ("interpreter", "model", "data", "config")}
    identity["interpreter"] = pathlib.Path(sys.executable)
    identity["script"] = pilot_script
    specs = {
        "phase-a": {"phase": "phase-a", "attempt": 3, "iters": 2, "steps_per_eval": 2,
                    "adapter_path": str(paths["phase-a-output"]), "resume_adapter_file": None,
                    "namespace_paths": {name: str(value) for name, value in paths.items()}},
        "phase-b": {"phase": "phase-b", "attempt": 3, "iters": 1, "steps_per_eval": 1,
                    "adapter_path": str(paths["phase-b-output"]), "resume_adapter_file": str(paths["phase-b-resume"]),
                    "namespace_paths": {name: str(value) for name, value in paths.items()}},
    }
    monkeypatch.setattr(finetune, "_central_attempt3_launch_identity", lambda **_: identity)
    monkeypatch.setattr(finetune, "_central_attempt3_namespace", lambda **_: paths)
    monkeypatch.setattr(finetune, "_central_attempt3_phase_specs", lambda **_: specs)

    def run_case(mutated=False):
        if mutated:
            paths[key] = namespace_root / "mutated" / key
            specs[phase]["namespace_paths"] = {name: str(value) for name, value in paths.items()}
            if key == "phase-b-resume":
                specs[phase]["resume_adapter_file"] = str(paths[key])
        else:
            specs[phase]["namespace_paths"] = {name: str(value) for name, value in paths.items()}
        paths[f"{phase}-log"].unlink(missing_ok=True)
        command = finetune._pilot_attempt3_command(
            root, pilot_script, namespace_root, phase,
            **_r10_wrapper_kwargs(finetune, root, namespace_root, phase))
        argv = shlex.split(command)
        assert argv[:2] == ["bash", "-lc"] and len(argv) == 3
        obs = root / ("mutated-observations.jsonl" if mutated else "baseline-observations.jsonl")
        env = os.environ.copy()
        env["R8_OBS"] = str(obs)
        result = subprocess.run(argv, cwd=root, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        log = pathlib.Path(paths[f"{phase}-log"])
        assert log.is_file() and "training-output" in log.read_text(encoding="utf-8")
        return [json.loads(line) for line in obs.read_text(encoding="utf-8").splitlines()], log.read_text(encoding="utf-8")

    baseline, baseline_log = run_case()
    mutated, mutated_log = run_case(mutated=True)
    assert [item["kind"] for item in baseline] == ["launch", "train"]
    assert [item["kind"] for item in mutated] == ["launch", "train"]
    if key == "phase-b-resume":
        value = str(paths[key])
        assert all(value in item["argv"] for item in (mutated[0], mutated[1]))
        assert value in mutated_log
    else:
        assert str(paths[key]) in mutated_log, key


def test_r8_catalog_applicability_matrix_is_exact():
    assert set(R8_CATALOG_APPLICABILITY) == set(R8_EXECUTABLE_CATALOG_KEYS) | set(R8_DERIVED_CATALOG_NA_KEYS)
    assert all(R8_CATALOG_APPLICABILITY[key] == "executable" for key in R8_EXECUTABLE_CATALOG_KEYS)
    assert all(R8_CATALOG_APPLICABILITY[key] == "N/A — derived training output"
               for key in R8_DERIVED_CATALOG_NA_KEYS)


@pytest.mark.parametrize("key", R8_DERIVED_CATALOG_NA_KEYS)
def test_r8_catalog_marks_derived_checkpoint_outputs_na(key):
    assert R8_CATALOG_APPLICABILITY[key] == "N/A — derived training output"
    assert key not in R8_EXECUTABLE_CATALOG_KEYS


def test_r8_valid_b2_publication_binds_phase_and_final_ok_evidence(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    paths, report, spec, phase_a_report, phase_a_marker = _write_valid_attempt3_phase_b(tmp_path, p)
    monkeypatch.setattr(p, "verify_attempt1_historical_evidence", lambda **_: report["attempt_1_historical_evidence"])
    _r8_remove_publications(paths, "phase-b")
    p.validate_canonical_attempt3_report(report, phase_spec=spec, report_path=paths["phase-b-report"])
    p._write_success_evidence("phase-b", report, phase_spec=spec,
                               phase_a_authorization=phase_a_report["authorization"],
                               phase_a_admission_lineage=report["phase_a_admission_lineage"],
                               phase_b_authorization=report["authorization"],
                               trusted_identity=phase_a_report["identity_manifest"],
                               trusted_attempt_1_historical_evidence=phase_a_report["attempt_1_historical_evidence"],
                               trusted_attempt_2_historical_evidence=phase_a_report["attempt_2_historical_evidence"])
    phase_report = json.loads(paths["phase-b-report"].read_text(encoding="utf-8"))
    final_report = json.loads(paths["final-report"].read_text(encoding="utf-8"))
    phase_marker = json.loads(paths["phase-b-ok"].read_text(encoding="utf-8"))
    final_marker = json.loads(paths["final-ok"].read_text(encoding="utf-8"))
    required_marker_keys = {"phase", "status", "attempt", "namespace", "report_path", "report_sha256",
                            "output_path", "timestamp", "exit_code", "contract_digest"}
    assert phase_report["status"] == "ok"
    assert phase_marker.keys() == required_marker_keys
    assert phase_marker["phase"] == "phase-b" and phase_marker["attempt"] == 3
    assert phase_marker["report_path"] == str(paths["phase-b-report"])
    assert phase_marker["report_sha256"] == p.file_sha256(paths["phase-b-report"])
    assert phase_marker["contract_digest"] == report["contract_digest"]
    assert phase_marker["output_path"] == report["output_path"]
    assert phase_marker["timestamp"] > 0 and phase_marker["exit_code"] == 0
    assert final_report["status"] == "ok" and final_report["global_progression"] == [0, 1, 2, 3]
    assert final_marker.keys() == required_marker_keys
    assert final_marker["phase"] == "phase-b" and final_marker["attempt"] == 3
    assert final_marker["report_path"] == str(paths["final-report"])
    assert final_marker["report_sha256"] == p.file_sha256(paths["final-report"])
    assert final_marker["contract_digest"] == final_report["contract_digest"]
    assert final_marker["output_path"] == report["output_path"]
    assert final_marker["timestamp"] > 0 and final_marker["exit_code"] == 0
    assert paths["phase-b-fail"].exists() is False and paths["final-fail"].exists() is False
    assert phase_a_report and phase_a_marker


def _write_valid_attempt3_phase_a(tmp_path, p):
    workspace = tmp_path / "workspace"
    paths = p.attempt3_namespace(repo_root=tmp_path, workspace=workspace)
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=workspace)["phase-a"]
    paths["phase-a-output"].mkdir(parents=True)
    for key, raw in (("phase-a-start-checkpoint", b"a"), ("phase-a-step1-checkpoint", b"b"),
                     ("phase-a-step2-checkpoint", b"c"), ("phase-a-final-checkpoint", b"c")):
        write_safetensors(paths[key], {"x": ("U8", [1], raw)})
    paths["phase-a-config"].write_text("{}", encoding="utf-8")
    paths["phase-a-log"].parent.mkdir(parents=True, exist_ok=True)
    paths["phase-a-log"].write_bytes(b"phase-a")
    identity = {
        "immutable": {
            "interpreter": p.PILOT_INTERPRETER, "python_version": [3, 13, 0], "mlx_version": "0.31.2",
            "mlx_lm_resolved_module": "/tmp/vendor/mlx-lm/mlx_lm/__init__.py", "vendor_head": p.PILOT_VENDOR_SHA,
            "vendor_gitlink": f"160000 {p.PILOT_VENDOR_SHA} 0\tvendor/mlx-lm", "vendor_inner_clean": True,
            "provider_sha256": p.PILOT_PROVIDER_SHA256, "smoke_sha256": p.PILOT_SMOKE_SHA256,
            "pilot_source_sha256": "a" * 64, "config_sha256": "b" * 64, "provenance_sha256": "c" * 64,
            "provenance_split_hashes": {name: "d" * 64 for name in ("train.jsonl", "valid.jsonl", "test.jsonl")},
            "model_manifest": [{"path": "model", "size": 1, "sha256": "e" * 64}],
            "dataset_manifest": [{"path": "train.jsonl", "size": 1, "sha256": "f" * 64}],
            "lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0, "keys": ["self_attn.q_a_proj"]},
            "git_head": "1" * 40,
        },
        "dynamic_resources": {
            "available_memory": 32 * 1024**3, "disk_free": 1 * 1024**3,
            "competing_processes": [],
            "allowed_process_skips": p._resource_evidence(
                32 * 1024**3, 1 * 1024**3, [],
                {name: {"counts_by_type": 0, "pids_by_type": [], "unknown_pid_counts_by_type": 0}
                 for name in ("AccessDenied", "NoSuchProcess", "ZombieProcess")},
            )["allowed_process_skips"],
        },
        "repository_status": [],
    }
    effective = {key: value for key, value in p.COMMON_VALUES.items()}
    effective.update({"phase": "phase-a", "model": p.PILOT_MODEL, "data": p.PILOT_DATA, "config": p.PILOT_CONFIG,
                      "adapter_path": str(paths["phase-a-output"]), "resume_adapter_file": None,
                      "train": True, "test": False, "hf_dataset": False, "attempt": 3,
                      "log_path": str(paths["phase-a-log"]), "iters": 2, "steps_per_eval": 2,
                      "command": p.canonical_attempt3_command("phase-a", spec)})
    artifacts = p._validate_artifacts("phase-a", paths["phase-a-output"], phase_spec=spec)
    provider = []
    steps = []
    for local in (1, 2):
        provider.append({"phase": "phase-a", "provider_call": local, "local_step": local, "global_step": local,
                         "loss": 1.0, "loss_dtype": "float32", "token_dtype": "int32", "n_tokens": 1,
                         "expected_mask_tokens": 1, "mask_tokens_match": True,
                         "gradient_schema": [{"path": "x", "shape": [1], "dtype": "float32"}],
                         "gradient_leaf_count": 1, "gradient_paths": ["x"], "gradient_shapes": {"x": [1]},
                         "gradient_dtypes": {"x": "float32"}, "gradients_finite": True, "provider_elapsed_seconds": 0.1})
        steps.append({"phase": "phase-a", "local_step": local, "global_step": local, "loss": 1.0,
                      "learning_rate": 1e-5, "tokens_per_second": 1.0, "iterations_per_second": 1.0,
                      "train_step_wall_seconds": 1.0, "optimizer_update_ordinal": local,
                      "provider_call": local, "checkpoint": artifacts["checkpoints"][local - 1]})
    historical_attempt2 = {"synthetic": "attempt-2-history"}
    authorization = {"revision": "0" * 40, "canonical_command_sha256": hashlib.sha256(
        json.dumps(p.canonical_attempt3_command("phase-a", spec), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "pilot_source_sha256": "1" * 64, "catalog_source_sha256": "2" * 64,
        "protected_files_manifest_sha256": "3" * 64,
        "attempt2_runtime_manifest_sha256": p._ATTEMPT2_HISTORICAL_MANIFEST["sha256"]}
    report = {"status": "ok", "phase": "phase-a", "attempt": 3, "namespace": p.ATTEMPT3_NAMESPACE,
              "effective": effective, "identity_manifest": identity,
              "contract_digest": p.contract_digest("phase-a", effective, identity, phase_spec=spec,
                                                   historical_evidence=list(p.ATTEMPT1_EVIDENCE_SHA256),
                                                   historical_attempt2_evidence=historical_attempt2,
                                                   authorization=authorization),
              "provider_calls": 2, "optimizer_updates": 2, "steps": steps, "provider_evidence": provider,
              "validation_evidence": [{"iteration": 0, "val_loss": 1.0, "val_time": 0.1},
                                      {"iteration": 1, "val_loss": 1.0, "val_time": 0.1}],
              "artifacts": artifacts, "output_path": str(paths["phase-a-output"]),
              "commands": p.canonical_attempt3_command("phase-a", spec),
              "wall_seconds": 1.0, "retry": p.RETRY_POLICY, "fallback": p.FALLBACK_POLICY,
              "non_claims": list(p.NON_CLAIMS), "global_mapping": [0, 1, 2],
              "attempt_1_historical_evidence": list(p.ATTEMPT1_EVIDENCE_SHA256),
              "attempt_2_historical_evidence": historical_attempt2, "authorization": authorization,
              "resume_source": artifacts["checkpoints"][-1], "report_path": str(paths["phase-a-report"]),
              "log_path": str(paths["phase-a-log"]), "ok_marker_path": str(paths["phase-a-ok"]),
              "fail_marker_path": str(paths["phase-a-fail"]), "final_report_path": str(paths["final-report"]),
              "namespace_paths": {key: str(value) for key, value in paths.items()},
              "lock_lifecycle": {}, "watchdog_cancelled": True, "exit_code": 0}
    spec["trusted_identity"] = identity
    paths["phase-a-report"].parent.mkdir(parents=True, exist_ok=True)
    paths["phase-a-report"].write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    marker = {"status": "ok", "phase": "phase-a", "attempt": 3, "namespace": p.ATTEMPT3_NAMESPACE,
              "report_path": str(paths["phase-a-report"]), "report_sha256": p.file_sha256(paths["phase-a-report"]),
              "output_path": str(paths["phase-a-output"]), "timestamp": 1.0, "exit_code": 0,
              "contract_digest": report["contract_digest"],
              "resume_source": str(paths["phase-b-resume"]),
              "resume_source_file_sha256": report["resume_source"]["file_sha256"],
              "resume_source_canonical_tensor_digest_v1": report["resume_source"]["canonical_tensor_digest_v1"]}
    paths["phase-a-ok"].write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")
    return paths, report, marker


def _write_valid_attempt3_phase_b(tmp_path, p):
    paths, phase_a_report, phase_a_marker = _write_valid_attempt3_phase_a(tmp_path, p)
    workspace = tmp_path / "workspace"
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=workspace)["phase-b"]
    paths["phase-b-output"].mkdir(parents=True)
    for key, raw in (("phase-b-start-checkpoint", b"d"), ("phase-b-step1-checkpoint", b"e"),
                     ("phase-b-final-checkpoint", b"e")):
        write_safetensors(paths[key], {"x": ("U8", [1], raw)})
    paths["phase-b-config"].write_text("{}", encoding="utf-8")
    paths["phase-b-log"].parent.mkdir(parents=True, exist_ok=True)
    paths["phase-b-log"].write_bytes(b"phase-b")
    artifacts = p._validate_artifacts("phase-b", paths["phase-b-output"], phase_spec=spec)
    report = json.loads(json.dumps(phase_a_report))
    effective = dict(report["effective"])
    effective.update({"phase": "phase-b", "adapter_path": str(paths["phase-b-output"]),
                      "resume_adapter_file": str(paths["phase-b-resume"]), "log_path": str(paths["phase-b-log"]),
                      "iters": 1, "steps_per_eval": 1,
                      "command": p.canonical_attempt3_command("phase-b", spec)})
    phase_b_authorization = dict(report["authorization"])
    phase_b_authorization["canonical_command_sha256"] = hashlib.sha256(json.dumps(
        p.canonical_attempt3_command("phase-b", spec), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    provider = [{"phase": "phase-b", "provider_call": 1, "local_step": 1, "global_step": 3,
                 "loss": 1.0, "loss_dtype": "float32", "token_dtype": "int32", "n_tokens": 1,
                 "expected_mask_tokens": 1, "mask_tokens_match": True,
                 "gradient_schema": [{"path": "x", "shape": [1], "dtype": "float32"}],
                 "gradient_leaf_count": 1, "gradient_paths": ["x"], "gradient_shapes": {"x": [1]},
                 "gradient_dtypes": {"x": "float32"}, "gradients_finite": True,
                 "provider_elapsed_seconds": 0.1}]
    lineage = p.capture_phase_a_admission_lineage(
        paths["phase-a-report"], paths["phase-a-ok"], phase_a_report, phase_a_marker)
    report.update({"phase": "phase-b", "effective": effective,
                   "phase_a_admission_lineage": lineage,
                   "contract_digest": p.contract_digest("phase-b", effective, report["identity_manifest"], phase_spec=spec,
                                                        historical_evidence=report["attempt_1_historical_evidence"],
                                                        historical_attempt2_evidence=report["attempt_2_historical_evidence"],
                                                        authorization=phase_b_authorization,
                                                        phase_a_admission_lineage=lineage),
                   "provider_calls": 1, "optimizer_updates": 1, "steps": [{
                       "phase": "phase-b", "local_step": 1, "global_step": 3, "loss": 1.0,
                       "learning_rate": 1e-5, "tokens_per_second": 1.0, "iterations_per_second": 1.0,
                       "train_step_wall_seconds": 1.0, "optimizer_update_ordinal": 1,
                       "provider_call": 1, "checkpoint": artifacts["checkpoints"][0]}],
                   "provider_evidence": provider, "validation_evidence": [{"iteration": 0, "val_loss": 1.0, "val_time": 0.1}],
                   "artifacts": artifacts, "output_path": str(paths["phase-b-output"]),
                   "commands": p.canonical_attempt3_command("phase-b", spec), "global_mapping": [0, 3],
                   "authorization": phase_b_authorization,
                   "resume_source": p.canonical_tensor_digest(paths["phase-b-resume"]),
                   "report_path": str(paths["phase-b-report"]), "log_path": str(paths["phase-b-log"]),
                   "ok_marker_path": str(paths["phase-b-ok"]), "fail_marker_path": str(paths["phase-b-fail"]),
                   "final_report_path": str(paths["final-report"]),
                   "namespace_paths": {key: str(value) for key, value in paths.items()}})
    paths["phase-b-report"].parent.mkdir(parents=True, exist_ok=True)
    paths["phase-b-report"].write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    return paths, report, spec, phase_a_report, phase_a_marker


def _trusted_final_kwargs(report, phase_a_report):
    return {
        "trusted_phase_a_admission_lineage": report["phase_a_admission_lineage"],
        "trusted_phase_a_authorization": phase_a_report["authorization"],
        "trusted_phase_b_authorization": report["authorization"],
        "trusted_identity": phase_a_report["identity_manifest"],
        "trusted_attempt_1_historical_evidence": phase_a_report["attempt_1_historical_evidence"],
        "trusted_attempt_2_historical_evidence": phase_a_report["attempt_2_historical_evidence"],
    }


def test_valid_b2_canonical_report_publishes_phase_and_final_evidence(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    paths, report, spec, phase_a_report, phase_a_marker = _write_valid_attempt3_phase_b(tmp_path, p)
    monkeypatch.setattr(p, "verify_attempt1_historical_evidence", lambda **_: report["attempt_1_historical_evidence"])
    p.validate_canonical_attempt3_report(report, phase_spec=spec, report_path=paths["phase-b-report"])
    p._write_success_evidence("phase-b", report, phase_spec=spec,
                               phase_a_authorization=phase_a_report["authorization"],
                               phase_a_admission_lineage=report["phase_a_admission_lineage"],
                               phase_b_authorization=report["authorization"],
                               trusted_identity=phase_a_report["identity_manifest"],
                               trusted_attempt_1_historical_evidence=phase_a_report["attempt_1_historical_evidence"],
                               trusted_attempt_2_historical_evidence=phase_a_report["attempt_2_historical_evidence"])
    assert paths["phase-b-report"].is_file()
    assert paths["final-report"].is_file()
    assert paths["phase-b-ok"].is_file()
    assert paths["final-ok"].is_file()
    assert json.loads(paths["final-report"].read_text(encoding="utf-8"))["status"] == "ok"


@pytest.mark.parametrize("key", [
    "phase-b-output", "phase-b-start-checkpoint", "phase-b-step1-checkpoint", "phase-b-final-checkpoint",
    "phase-b-config", "phase-b-resume", "phase-b-log", "phase-b-report", "phase-b-ok", "phase-b-fail",
    "final-report", "final-ok", "final-fail",
])
def test_valid_b2_canonical_report_rejects_each_namespace_binding_mutation(tmp_path, key):
    p = load_pilot()
    paths, report, spec, _, _ = _write_valid_attempt3_phase_b(tmp_path, p)
    mutated = json.loads(json.dumps(report))
    mutated["namespace_paths"][key] = str(tmp_path / "mutated" / key)
    with pytest.raises(p.PilotError):
        p.validate_canonical_attempt3_report(mutated, phase_spec=spec, report_path=paths["phase-b-report"])


def test_attempt3_launch_check_rejects_invalid_a2_before_b2_log_creation(tmp_path, monkeypatch):
    p = load_pilot()
    paths, report, _ = _write_valid_attempt3_phase_a(tmp_path, p)
    report["status"] = "fail"
    paths["phase-a-report"].write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(p, "verify_attempt1_historical_evidence", lambda **_: [])
    with pytest.raises(p.PilotError, match="Phase A report and OK marker"):
        p.check_attempt3_launch("phase-b", paths["phase-b-log"], repo_root=tmp_path, workspace=tmp_path / "workspace")
    assert not paths["phase-b-log"].exists()
    assert not paths["phase-b-output"].exists()
    assert not paths["final-report"].exists()


def test_launch_check_cli_rejects_current_identity_substitution_before_b2_log(tmp_path, monkeypatch, capsys):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    monkeypatch.setattr(p, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setattr(p, "PILOT_MODEL", str(tmp_path / "model"))
    monkeypatch.setattr(p, "PILOT_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(p, "PILOT_CONFIG", str(tmp_path / "lora-config.json"))
    monkeypatch.setattr(p, "PILOT_INTERPRETER", sys.executable)
    paths, report, _ = _write_valid_attempt3_phase_a(tmp_path, p)
    current = json.loads(json.dumps(report["identity_manifest"]))
    current["immutable"]["git_head"] = "2" * 40
    monkeypatch.setattr(p, "_read_config", lambda _path: {})
    monkeypatch.setattr(p, "_runtime_preflight", lambda *_args: current)
    monkeypatch.setattr(p, "verify_attempt1_historical_evidence", lambda **_kwargs: list(p.ATTEMPT1_EVIDENCE_SHA256))
    authorization_json = json.dumps(report["authorization"], sort_keys=True, separators=(",", ":"))
    result = p.main(["--attempt", "3", "--phase", "phase-b", "--launch-check-only",
                     "--authorization-json", authorization_json, "--log-path", str(paths["phase-b-log"])])
    assert result == 1
    error = capsys.readouterr().err
    assert "immutable identity substituted" in error or "filesystem escape" in error
    assert not paths["phase-b-log"].exists()
    assert not paths["phase-b-output"].exists()
    assert not paths["final-report"].exists()


def test_attempt3_contract_digest_binds_manifest_and_namespace_paths(tmp_path):
    p = load_pilot()
    paths = p.attempt3_namespace(repo_root=tmp_path, workspace=tmp_path / "workspace")
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
    effective = {"log_path": str(paths["phase-a-log"]), "pin": "fixed"}
    identity = {"immutable": {"identity": "same"}}
    first = p.contract_digest("phase-a", effective, identity, phase_spec=spec)
    changed_manifest = list(p.ATTEMPT1_EVIDENCE_SHA256)
    changed_manifest[0] = {**changed_manifest[0], "size": changed_manifest[0]["size"] + 1}
    p.ATTEMPT1_EVIDENCE_SHA256 = tuple(changed_manifest)
    assert p.contract_digest("phase-a", effective, identity, phase_spec=spec) != first
    changed_spec = dict(spec)
    changed_spec["namespace_paths"] = dict(spec["namespace_paths"])
    changed_spec["namespace_paths"]["phase-b-log"] = str(tmp_path / "mutated.log")
    assert p.contract_digest("phase-a", effective, identity, phase_spec=changed_spec) != p.contract_digest("phase-a", effective, identity, phase_spec=spec)


def test_attempt3_contract_digest_binds_every_namespace_path_mutation(tmp_path):
    p = load_pilot()
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
    effective = {"command": ["synthetic"], "log_path": spec["namespace_paths"]["phase-a-log"]}
    identity = {"immutable": {"identity": "same"}}
    baseline = p.contract_digest("phase-a", effective, identity, phase_spec=spec)
    for key in spec["namespace_paths"]:
        mutated = dict(spec)
        mutated["namespace_paths"] = dict(spec["namespace_paths"])
        mutated["namespace_paths"][key] = f"{mutated['namespace_paths'][key]}.mutated"
        assert p.contract_digest("phase-a", effective, identity, phase_spec=mutated) != baseline, key


def test_attempt3_catalog_is_non_default_and_uses_fd_attested_noclobber_wrapper():
    import scripts.finetune_ds4 as finetune
    import scripts.ds4_segmented_pilot as pilot
    class Args:
        hf_model = "/tmp/hf"; dataset_root = "/tmp/data"; mlx_work = "/tmp/mlx"
        ds4_root = "/tmp/ds4"; ds4_gguf = None; split_dir = "mlx-4096"
        fused_hf_model = None; ds4_imatrix = None; adapter_ds4 = None
        attempt3_authorization_file = None
        attempt3_authorization_json = pilot.canonical_attempt3_authorization(
            "phase-a", pilot.attempt3_phase_specs()["phase-a"],
            catalog_source=pathlib.Path(finetune.__file__).resolve())
    catalog = finetune.command_catalog(Args())
    assert "ds4-segmented-pilot-attempt-3-phase-a" in finetune.MLX_STEPS
    assert "ds4-segmented-pilot-attempt-3-phase-b" in finetune.MLX_STEPS
    assert "ds4-segmented-pilot-attempt-3-phase-a" not in finetune.DEFAULT_BACKEND_STEPS["local-mlx"]
    command = catalog["ds4-segmented-pilot-attempt-3-phase-a"][0]
    assert "--launch-check-only" in command
    assert "set -o noclobber; exec 3>" in command
    assert "tee /dev/fd/3" in command
    assert "tee \"" not in command
    assert "--attempt 3" in command


@pytest.mark.parametrize("phase", ["phase-a", "phase-b"])
def test_attempt3_catalog_uses_every_central_namespace_binding(monkeypatch, phase):
    import scripts.finetune_ds4 as finetune
    keys = ("phase-a-output", "phase-a-start-checkpoint", "phase-a-step1-checkpoint",
            "phase-a-step2-checkpoint", "phase-a-final-checkpoint", "phase-a-config",
            "phase-a-log", "phase-a-report", "phase-a-ok", "phase-a-fail",
            "phase-b-output", "phase-b-start-checkpoint", "phase-b-step1-checkpoint",
            "phase-b-final-checkpoint", "phase-b-config", "phase-b-resume", "phase-b-log",
            "phase-b-report", "phase-b-ok", "phase-b-fail", "final-report", "final-ok", "final-fail")
    sentinel = {key: pathlib.Path(f"/sentinel/{key}") for key in keys}
    phase_specs = {
        "phase-a": {"iters": 2, "steps_per_eval": 2, "adapter_path": str(sentinel["phase-a-output"]),
                     "resume_adapter_file": None},
        "phase-b": {"iters": 1, "steps_per_eval": 1, "adapter_path": str(sentinel["phase-b-output"]),
                     "resume_adapter_file": str(sentinel["phase-b-resume"])},
    }
    monkeypatch.setattr(finetune, "_central_attempt3_namespace", lambda **_kwargs: sentinel)
    monkeypatch.setattr(finetune, "_central_attempt3_phase_specs", lambda **_kwargs: phase_specs)
    command = finetune._pilot_attempt3_command(
        pathlib.Path("/repo"), pathlib.Path("/repo/pilot.py"), pathlib.Path("/work"), phase,
        **_r10_wrapper_kwargs(finetune, pathlib.Path("/repo"), pathlib.Path("/work"), phase))
    for path in sentinel.values():
        assert str(path) in command


@pytest.mark.parametrize("phase", ["phase-a", "phase-b"])
@pytest.mark.parametrize("field", ["interpreter", "model", "data", "config"])
def test_attempt3_catalog_central_launch_mutation_changes_both_command_positions(monkeypatch, field, phase):
    import scripts.finetune_ds4 as finetune
    base = {name: pathlib.Path(f"/base/{name}") for name in ("interpreter", "model", "data", "config", "script")}
    mutated = dict(base)
    mutated[field] = pathlib.Path(f"/mutated/{field}")
    paths = {key: pathlib.Path(f"/sentinel/{key}") for key in (
        "phase-a-output", "phase-a-start-checkpoint", "phase-a-step1-checkpoint", "phase-a-step2-checkpoint",
        "phase-a-final-checkpoint", "phase-a-config", "phase-a-log", "phase-a-report", "phase-a-ok", "phase-a-fail",
        "phase-b-output", "phase-b-start-checkpoint", "phase-b-step1-checkpoint", "phase-b-final-checkpoint", "phase-b-config",
        "phase-b-resume", "phase-b-log", "phase-b-report", "phase-b-ok", "phase-b-fail", "final-report", "final-ok", "final-fail")}
    phase_specs = {
        "phase-a": {"iters": 2, "steps_per_eval": 2, "adapter_path": str(paths["phase-a-output"]), "resume_adapter_file": None},
        "phase-b": {"iters": 1, "steps_per_eval": 1, "adapter_path": str(paths["phase-b-output"]), "resume_adapter_file": str(paths["phase-b-resume"])},
    }
    monkeypatch.setattr(finetune, "_central_attempt3_namespace", lambda **_kwargs: paths)
    monkeypatch.setattr(finetune, "_central_attempt3_phase_specs", lambda **_kwargs: phase_specs)
    monkeypatch.setattr(finetune, "_central_attempt3_launch_identity", lambda **_kwargs: base)
    baseline = finetune._pilot_attempt3_command(
        pathlib.Path("/repo"), pathlib.Path("/repo/pilot.py"), pathlib.Path("/work"), phase,
        **_r10_wrapper_kwargs(finetune, pathlib.Path("/repo"), pathlib.Path("/work"), phase))
    monkeypatch.setattr(finetune, "_central_attempt3_launch_identity", lambda **_kwargs: mutated)
    changed = finetune._pilot_attempt3_command(
        pathlib.Path("/repo"), pathlib.Path("/repo/pilot.py"), pathlib.Path("/work"), phase,
        **_r10_wrapper_kwargs(finetune, pathlib.Path("/repo"), pathlib.Path("/work"), phase))
    assert changed != baseline
    assert str(mutated[field]) in changed
    if field != "interpreter":
        assert changed.count(str(mutated[field])) == 1



@pytest.mark.parametrize("phase", ["phase-a", "phase-b"])
def test_attempt3_catalog_binds_central_launch_identity_in_launch_check_and_training(monkeypatch, phase):
    import scripts.finetune_ds4 as finetune
    identity = {
        "interpreter": pathlib.Path("/sentinel/python"),
        "model": pathlib.Path("/sentinel/model"),
        "data": pathlib.Path("/sentinel/data"),
        "config": pathlib.Path("/sentinel/config.json"),
        "script": pathlib.Path("/sentinel/pilot.py"),
    }
    paths = {key: pathlib.Path(f"/sentinel/{key}") for key in (
        "phase-a-output", "phase-a-start-checkpoint", "phase-a-step1-checkpoint", "phase-a-step2-checkpoint",
        "phase-a-final-checkpoint", "phase-a-config", "phase-a-log", "phase-a-report", "phase-a-ok", "phase-a-fail",
        "phase-b-output", "phase-b-start-checkpoint", "phase-b-step1-checkpoint", "phase-b-final-checkpoint", "phase-b-config",
        "phase-b-resume", "phase-b-log", "phase-b-report", "phase-b-ok", "phase-b-fail", "final-report", "final-ok", "final-fail")}
    phase_specs = {
        "phase-a": {"iters": 2, "steps_per_eval": 2, "adapter_path": str(paths["phase-a-output"]), "resume_adapter_file": None},
        "phase-b": {"iters": 1, "steps_per_eval": 1, "adapter_path": str(paths["phase-b-output"]), "resume_adapter_file": str(paths["phase-b-resume"])},
    }
    monkeypatch.setattr(finetune, "_central_attempt3_launch_identity", lambda **_kwargs: identity)
    monkeypatch.setattr(finetune, "_central_attempt3_namespace", lambda **_kwargs: paths)
    monkeypatch.setattr(finetune, "_central_attempt3_phase_specs", lambda **_kwargs: phase_specs)
    command = finetune._pilot_attempt3_command(
        pathlib.Path("/repo"), pathlib.Path("/repo/pilot.py"), pathlib.Path("/work"), phase,
        **_r10_wrapper_kwargs(finetune, pathlib.Path("/repo"), pathlib.Path("/work"), phase))
    assert command.count("/sentinel/python") == 1
    assert command.count("/sentinel/pilot.py") == 1
    for flag, variable in (("--model", '"$MODEL"'), ("--data", '"$DATA"'), ("--config", '"$CONFIG"')):
        assert command.count(f"{flag} {variable}") == 2
    for value in identity.values():
        assert str(value) in command


@pytest.mark.parametrize("phase", ["phase-a", "phase-b"])
@pytest.mark.parametrize("train_exit", [0, 17])
def test_attempt3_catalog_wrapper_executes_temp_phase_with_spaces_and_propagates_status(
        tmp_path, monkeypatch, phase, train_exit):
    import scripts.finetune_ds4 as finetune

    root = tmp_path / "repo with spaces"
    root.mkdir()
    pilot_script = root / "pilot stub with spaces.py"
    pilot_script.write_text(
        "import os, pathlib, sys\n"
        "trace = pathlib.Path(os.environ['STUB_TRACE'])\n"
        "kind = 'launch-check' if '--launch-check-only' in sys.argv else 'train'\n"
        "with trace.open('a', encoding='utf-8') as stream:\n"
        "    stream.write(kind + '\\n')\n"
        "if kind == 'train':\n"
        "    print('training-output')\n"
        "    raise SystemExit(int(os.environ['STUB_TRAIN_EXIT']))\n",
        encoding="utf-8",
    )
    sentinel_root = tmp_path / "namespace with spaces"
    keys = (
        "phase-a-output", "phase-a-start-checkpoint", "phase-a-step1-checkpoint",
        "phase-a-step2-checkpoint", "phase-a-final-checkpoint", "phase-a-config", "phase-a-log",
        "phase-a-report", "phase-a-ok", "phase-a-fail", "phase-b-output", "phase-b-start-checkpoint",
        "phase-b-step1-checkpoint", "phase-b-final-checkpoint", "phase-b-config", "phase-b-resume",
        "phase-b-log", "phase-b-report", "phase-b-ok", "phase-b-fail", "final-report", "final-ok",
        "final-fail",
    )
    paths = {key: sentinel_root / key for key in keys}
    identity = {
        "interpreter": pathlib.Path(sys.executable), "model": sentinel_root / "model with spaces",
        "data": sentinel_root / "data with spaces", "config": sentinel_root / "config with spaces.json",
        "script": pilot_script,
    }
    phase_specs = {
        "phase-a": {"iters": 2, "steps_per_eval": 2, "adapter_path": str(paths["phase-a-output"]),
                    "resume_adapter_file": None},
        "phase-b": {"iters": 1, "steps_per_eval": 1, "adapter_path": str(paths["phase-b-output"]),
                    "resume_adapter_file": str(paths["phase-b-resume"])},
    }
    monkeypatch.setattr(finetune, "_central_attempt3_launch_identity", lambda **_kwargs: identity)
    monkeypatch.setattr(finetune, "_central_attempt3_namespace", lambda **_kwargs: paths)
    monkeypatch.setattr(finetune, "_central_attempt3_phase_specs", lambda **_kwargs: phase_specs)
    command = finetune._pilot_attempt3_command(
        root, pilot_script, sentinel_root, phase,
        **_r10_wrapper_kwargs(finetune, root, sentinel_root, phase))
    argv = shlex.split(command)
    assert len(argv) == 3
    assert argv[:2] == ["bash", "-lc"]

    env = os.environ.copy()
    env.update({"STUB_TRACE": str(tmp_path / "trace with spaces.txt"), "STUB_TRAIN_EXIT": str(train_exit)})
    result = subprocess.run(argv, cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == train_exit
    assert pathlib.Path(env["STUB_TRACE"]).read_text(encoding="utf-8").splitlines() == ["launch-check", "train"]
    assert paths[f"{phase}-log"].is_file()
    log = paths[f"{phase}-log"].read_text(encoding="utf-8")
    assert "attempt-3 namespace" in log
    assert "training-output" in log


@pytest.mark.parametrize("key", [
    "phase-a-output", "phase-a-start-checkpoint", "phase-a-step1-checkpoint", "phase-a-step2-checkpoint",
    "phase-a-final-checkpoint", "phase-a-config", "phase-a-log", "phase-a-report", "phase-a-ok", "phase-a-fail",
    "phase-b-output", "phase-b-start-checkpoint", "phase-b-step1-checkpoint", "phase-b-final-checkpoint",
    "phase-b-config", "phase-b-resume", "phase-b-log", "phase-b-report", "phase-b-ok", "phase-b-fail",
    "final-report", "final-ok", "final-fail",
])
def test_attempt3_namespace_key_mutation_rejected_by_dependency_admission(tmp_path, monkeypatch, key):
    p = load_pilot()
    paths, report, marker = _write_valid_attempt3_phase_a(tmp_path, p)
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
    spec["namespace_paths"] = dict(spec["namespace_paths"])
    spec["namespace_paths"][key] = str(tmp_path / "mutated" / key)
    monkeypatch.setattr(p, "verify_attempt1_historical_evidence", lambda **_: report["attempt_1_historical_evidence"])
    with pytest.raises(p.PilotError):
        p.validate_phase_b_dependency(report, marker, paths["phase-b-resume"], phase_spec=spec)


def test_attempt3_phase_specs_bind_checkpoint_and_config_consumers_to_namespace(tmp_path):
    p = load_pilot()
    paths = p.attempt3_namespace(repo_root=tmp_path, workspace=tmp_path / "workspace")
    specs = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")
    assert specs["phase-a"]["adapter_path"] == str(paths["phase-a-output"])
    assert specs["phase-a"]["namespace_paths"]["phase-a-config"] == str(paths["phase-a-config"])
    assert specs["phase-b"]["adapter_path"] == str(paths["phase-b-output"])
    assert specs["phase-b"]["resume_adapter_file"] == str(paths["phase-b-resume"])
    assert specs["phase-b"]["namespace_paths"]["phase-b-start-checkpoint"] == str(paths["phase-b-start-checkpoint"])
    assert specs["phase-b"]["namespace_paths"]["phase-b-step1-checkpoint"] == str(paths["phase-b-step1-checkpoint"])
    assert specs["phase-b"]["namespace_paths"]["phase-b-final-checkpoint"] == str(paths["phase-b-final-checkpoint"])
    assert specs["phase-b"]["namespace_paths"]["phase-b-config"] == str(paths["phase-b-config"])


def test_attempt3_phase_specs_bind_every_checkpoint_and_config_consumer_to_namespace(tmp_path):
    p = load_pilot()
    paths = p.attempt3_namespace(repo_root=tmp_path, workspace=tmp_path / "workspace")
    specs = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")
    for phase, keys in {
        "phase-a": ("phase-a-output", "phase-a-start-checkpoint", "phase-a-step1-checkpoint",
                    "phase-a-step2-checkpoint", "phase-a-final-checkpoint", "phase-a-config"),
        "phase-b": ("phase-b-output", "phase-b-start-checkpoint", "phase-b-step1-checkpoint",
                    "phase-b-final-checkpoint", "phase-b-config", "phase-b-resume"),
    }.items():
        assert specs[phase]["adapter_path"] == str(paths[f"{phase}-output"])
        for key in keys:
            assert specs[phase]["namespace_paths"][key] == str(paths[key])


@pytest.mark.parametrize("module_version", [_MISSING_MODULE_VERSION, None, "999.0-spoof"])
def test_mlx_distribution_metadata_is_authoritative_for_runtime_identity(tmp_path, monkeypatch, module_version):
    p = configure_runtime_preflight_fixture(monkeypatch, tmp_path, "0.31.2", module_version)
    identity = p._runtime_preflight(type("Args", (), {})(), {}, {})
    assert identity["immutable"]["mlx_version"] == "0.31.2"


@pytest.mark.parametrize(
    "metadata_value,module_version,diagnostic",
    [
        ("0.31.1", "0.31.2", "MLX distribution metadata version mismatch"),
        ("0.31.2+local", "0.31.2", "MLX distribution metadata version mismatch"),
        ("0.31.3", _MISSING_MODULE_VERSION, "MLX distribution metadata version mismatch"),
        (importlib.metadata.PackageNotFoundError("mlx"), "0.31.2", "MLX distribution metadata unavailable"),
        (RuntimeError("metadata broken"), "0.31.2", "MLX distribution metadata error: RuntimeError"),
        ("", "0.31.2", "MLX distribution metadata malformed"),
        (None, "0.31.2", "MLX distribution metadata malformed"),
        (312, "0.31.2", "MLX distribution metadata malformed"),
    ],
)
def test_mlx_distribution_metadata_fail_closed_at_production_preflight(
        tmp_path, monkeypatch, metadata_value, module_version, diagnostic):
    p = configure_runtime_preflight_fixture(monkeypatch, tmp_path, metadata_value, module_version)
    with pytest.raises(p.PilotError, match=re.escape(diagnostic)):
        p._runtime_preflight(type("Args", (), {})(), {}, {})


def test_mlx_module_import_remains_mandatory_when_metadata_is_correct(tmp_path, monkeypatch):
    p = configure_runtime_preflight_fixture(monkeypatch, tmp_path, "0.31.2")
    monkeypatch.setitem(sys.modules, "mlx", None)
    with pytest.raises(p.PilotError, match="runtime source identity unavailable"):
        p._runtime_preflight(type("Args", (), {})(), {}, {})


@pytest.mark.parametrize(
    "metadata_value,diagnostic",
    [
        ("0.31.1", "MLX distribution metadata version mismatch"),
        ("0.31.2+local", "MLX distribution metadata version mismatch"),
        (importlib.metadata.PackageNotFoundError("mlx"), "MLX distribution metadata unavailable"),
        (RuntimeError("metadata broken"), "MLX distribution metadata error: RuntimeError"),
        ("", "MLX distribution metadata malformed"),
        (None, "MLX distribution metadata malformed"),
        (312, "MLX distribution metadata malformed"),
    ],
)
def test_mlx_metadata_direct_launch_check_is_no_write_and_no_call(
        tmp_path, monkeypatch, capsys, metadata_value, diagnostic):
    p = configure_runtime_preflight_fixture(monkeypatch, tmp_path, metadata_value, "999.0-spoof")
    install_tmp_fs_guard(monkeypatch, tmp_path)
    monkeypatch.setattr(p, "_read_config", lambda _path: {})
    monkeypatch.setattr(p, "verify_attempt1_historical_evidence", lambda **_kwargs: [])
    training_api_calls = []
    monkeypatch.setattr(p, "_load_training_api", lambda: training_api_calls.append(True))
    paths = p.attempt3_namespace()
    assert all(not path.exists() for path in p._attempt3_absent_paths(paths))

    result = p.main(["--attempt", "3", "--phase", "phase-a", "--launch-check-only",
                     "--log-path", str(paths["phase-a-log"])])

    assert result == 1
    assert diagnostic in capsys.readouterr().err
    assert training_api_calls == []
    assert all(not path.exists() for path in p._attempt3_absent_paths(paths))


@pytest.mark.parametrize(
    "metadata_code,diagnostic",
    [
        ("mismatch", "MLX distribution metadata version mismatch"),
        ("suffix", "MLX distribution metadata version mismatch"),
        ("missing", "MLX distribution metadata unavailable"),
        ("error", "MLX distribution metadata error: RuntimeError"),
        ("empty", "MLX distribution metadata malformed"),
        ("none", "MLX distribution metadata malformed"),
        ("integer", "MLX distribution metadata malformed"),
    ],
)
def test_mlx_metadata_generated_attempt3_wrapper_rejects_before_log_or_training(
        tmp_path, monkeypatch, metadata_code, diagnostic):
    import scripts.finetune_ds4 as finetune

    root = tmp_path / "wrapper repo with spaces"
    root.mkdir()
    namespace_root = root / "namespace with spaces"
    model = root / "model with spaces"
    data = root / "data with spaces"
    config = root / "config with spaces.json"
    model.mkdir()
    data.mkdir()
    config.write_text("{}", encoding="utf-8")
    runner = root / "pilot runner with spaces.py"
    runner.write_text(textwrap.dedent(f"""
        import importlib
        import importlib.metadata
        import importlib.util
        import json
        import os
        import pathlib
        import sys
        import types

        root = pathlib.Path(os.environ["CASE_ROOT"])
        production = {str(SCRIPT)!r}
        spec = importlib.util.spec_from_file_location("production_pilot", production)
        p = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = p
        spec.loader.exec_module(p)
        if not hasattr(p, "importlib"):
            p.importlib = importlib
        p.REPO_ROOT = root
        p.PILOT_WORKSPACE = os.environ["WORKSPACE"]
        p.PILOT_INTERPRETER = sys.executable
        p.PILOT_MODEL = os.environ["MODEL"]
        p.PILOT_DATA = os.environ["DATA"]
        p.PILOT_CONFIG = os.environ["CONFIG"]
        p.PILOT_PROVENANCE = "provenance.json"
        p.PILOT_VENDOR_SHA = "vendor-sha"
        pathlib.Path(p.PILOT_PROVENANCE).write_text("{{}}", encoding="utf-8")
        vendor_module = root / "vendor" / "mlx-lm" / "mlx_lm" / "__init__.py"
        vendor_module.parent.mkdir(parents=True, exist_ok=True)
        vendor_module.write_text("", encoding="utf-8")
        fake_mlx = types.ModuleType("mlx")
        fake_mlx.__version__ = "999.0-spoof"
        fake_mlx_lm = types.ModuleType("mlx_lm")
        fake_mlx_lm.__file__ = str(vendor_module)
        sys.modules["mlx"] = fake_mlx
        sys.modules["mlx_lm"] = fake_mlx_lm

        def metadata_version(_name):
            code = os.environ["METADATA_CODE"]
            if code == "missing":
                raise p.importlib.metadata.PackageNotFoundError("mlx")
            if code == "error":
                raise RuntimeError("metadata broken")
            return {{"mismatch": "0.31.1", "suffix": "0.31.2+local",
                     "empty": "", "none": None, "integer": 312}}[code]

        p.importlib.metadata.version = metadata_version
        p._validate_lora_config = lambda _config: {{"rank": 8, "scale": 20.0,
            "dropout": 0.0, "keys": ["synthetic"]}}
        p._resource_gate = lambda: {{"available_memory": 1, "disk_free": 1,
            "competing_processes": []}}
        p._validate_provenance_splits = lambda _path, _manifest: {{
            name: "hash" for name in ("train.jsonl", "valid.jsonl", "test.jsonl")}}
        p._file_manifest = lambda path: [{{"path": pathlib.Path(path).name,
            "size": 0, "sha256": "0" * 64}}]
        def fake_file_sha256(path):
            path = pathlib.Path(path)
            if path.name == "segmented_loss_and_grad.py":
                return p.PILOT_PROVIDER_SHA256
            if path.name == "ds4_segmented_smoke.py":
                return p.PILOT_SMOKE_SHA256
            return "0" * 64
        p.file_sha256 = fake_file_sha256
        def fake_git(*args):
            if "rev-parse" in args and "vendor/mlx-lm" in args:
                return p.PILOT_VENDOR_SHA
            if "--stage" in args:
                return f"160000 {{p.PILOT_VENDOR_SHA}} 0\\tvendor/mlx-lm"
            return ""
        p._git_output = fake_git
        p.verify_attempt1_historical_evidence = lambda **_kwargs: []
        trace = pathlib.Path(os.environ["TRACE"])
        kind = "launch-check" if "--launch-check-only" in sys.argv else "train"
        with trace.open("a", encoding="utf-8") as stream:
            stream.write(kind + "\\n")
        p._load_training_api = lambda: trace.open("a", encoding="utf-8").write("training-api\\n")
        raise SystemExit(p.main(sys.argv[1:]))
    """), encoding="utf-8")

    pilot = load_pilot()
    paths = pilot.attempt3_namespace(repo_root=root, workspace=namespace_root)
    phase_specs = pilot.attempt3_phase_specs(repo_root=root, workspace=namespace_root)
    identity = {"interpreter": pathlib.Path(sys.executable), "model": model, "data": data,
                "config": config, "script": runner}
    monkeypatch.setattr(finetune, "_central_attempt3_launch_identity", lambda **_kwargs: identity)
    monkeypatch.setattr(finetune, "_central_attempt3_namespace", lambda **_kwargs: paths)
    monkeypatch.setattr(finetune, "_central_attempt3_phase_specs", lambda **_kwargs: phase_specs)
    command = finetune._pilot_attempt3_command(
        root, runner, namespace_root, "phase-a",
        **_r10_wrapper_kwargs(finetune, root, namespace_root, "phase-a"))
    trace = root / "trace.txt"
    env = os.environ.copy()
    env.update({"CASE_ROOT": str(root), "WORKSPACE": str(namespace_root), "MODEL": str(model),
                "DATA": str(data), "CONFIG": str(config), "METADATA_CODE": metadata_code,
                "TRACE": str(trace)})
    result = subprocess.run(shlex.split(command), cwd=root, env=env, capture_output=True, text=True)

    assert result.returncode == 1
    assert diagnostic in result.stderr
    assert trace.read_text(encoding="utf-8").splitlines() == ["launch-check"]
    assert all(not path.exists() for path in paths.values())


def test_attempt2_historical_verifier_uses_immutable_roots(tmp_path, monkeypatch):
    p = load_pilot()
    canonical = p._ATTEMPT2_CANONICAL_EXPECTED
    monkeypatch.setattr(p, "_ATTEMPT2_HISTORICAL_EXPECTED", {"files": []})
    monkeypatch.setattr(p, "_ATTEMPT2_HISTORICAL_MANIFEST", {"path": "substituted", "size": 0, "sha256": "0" * 64})
    assert p._ATTEMPT2_CANONICAL_EXPECTED is canonical
    with pytest.raises(p.PilotError):
        p.verify_attempt2_historical_evidence(repo_root=tmp_path)


def test_attempt3_authorization_schema_and_parser_are_explicit(tmp_path):
    p = load_pilot()
    parser = p.build_parser()
    assert "--authorization-json" in parser.format_help()
    with pytest.raises(p.PilotError):
        p.validate_attempt3_authorization({})
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
    command_hash = hashlib.sha256(json.dumps(
        p.canonical_attempt3_command("phase-a", spec), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    authorization = {"revision": "0" * 40, "canonical_command_sha256": command_hash,
                     "pilot_source_sha256": "1" * 64, "catalog_source_sha256": "2" * 64,
                     "protected_files_manifest_sha256": "3" * 64,
                     "attempt2_runtime_manifest_sha256": p._ATTEMPT2_HISTORICAL_MANIFEST["sha256"]}
    assert p.validate_attempt3_authorization(authorization, phase="phase-a", phase_spec=spec,
                                             repo_root=tmp_path, verify_sources=False) == authorization
    with pytest.raises(p.PilotError, match="canonical command"):
        p.validate_attempt3_authorization({**authorization, "canonical_command_sha256": "f" * 64},
                                          phase="phase-a", phase_spec=spec,
                                          repo_root=tmp_path, verify_sources=False)
    with pytest.raises(p.PilotError, match="canonical command"):
        p.validate_attempt3_authorization(authorization, phase="phase-b", phase_spec=spec,
                                          repo_root=tmp_path, verify_sources=False)
    for key in authorization:
        mutated = dict(authorization)
        mutated[key] = "bad"
        with pytest.raises(p.PilotError):
            p.validate_attempt3_authorization(mutated, phase="phase-a", phase_spec=spec,
                                              repo_root=tmp_path, verify_sources=False)


def test_attempt3_contract_and_report_lineage_require_attempt2_history(tmp_path):
    p = load_pilot()
    _, report, _ = _write_valid_attempt3_phase_a(tmp_path, p)
    assert "attempt_2_historical_evidence" in report
    assert "attempt_2_historical_evidence" in p._A2_REPORT_KEYS


def test_validate_artifacts_accepts_null_metadata_at_every_phase_a_checkpoint(tmp_path):
    p = load_pilot()
    workspace = tmp_path / "workspace"
    paths = p.attempt3_namespace(repo_root=tmp_path, workspace=workspace)
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=workspace)["phase-a"]
    paths["phase-a-output"].mkdir(parents=True)
    for key, raw in (("phase-a-start-checkpoint", b"a"), ("phase-a-step1-checkpoint", b"b"),
                     ("phase-a-step2-checkpoint", b"c"), ("phase-a-final-checkpoint", b"c")):
        write_safetensors(paths[key], {"x": ("U8", [1], raw)}, metadata=None)
    paths["phase-a-config"].write_text("{}", encoding="utf-8")
    result = p._validate_artifacts("phase-a", paths["phase-a-output"], phase_spec=spec)
    assert len(result["checkpoints"]) == 2
    assert all(item["tensors"] == [{"name": "x", "dtype": "U8", "shape": [1], "nbytes": 1}]
               for item in (result["start"], *result["checkpoints"], result["final"]))


def test_invalid_metadata_cannot_publish_success_evidence(tmp_path):
    p = load_pilot()
    workspace = tmp_path / "workspace"
    paths = p.attempt3_namespace(repo_root=tmp_path, workspace=workspace)
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=workspace)["phase-a"]
    paths["phase-a-output"].mkdir(parents=True)
    for key, raw in (("phase-a-start-checkpoint", b"a"), ("phase-a-step1-checkpoint", b"b"),
                     ("phase-a-step2-checkpoint", b"c"), ("phase-a-final-checkpoint", b"c")):
        write_safetensors(paths[key], {"x": ("U8", [1], raw)}, metadata="invalid")
    paths["phase-a-config"].write_text("{}", encoding="utf-8")
    before = paths["phase-a-step1-checkpoint"].read_bytes()
    with pytest.raises(p.PilotError):
        p._validate_artifacts("phase-a", paths["phase-a-output"], phase_spec=spec)
    assert paths["phase-a-step1-checkpoint"].read_bytes() == before
    assert not paths["phase-a-ok"].exists() and not paths["final-ok"].exists()


def test_mlx_written_null_metadata_is_digestible_and_loadable(tmp_path):
    mx = pytest.importorskip("mlx.core")
    p = load_pilot()
    path = tmp_path / "mlx-null.safetensors"
    values = {"x": mx.array([[1.0, 2.0]], dtype=mx.float32)}
    mx.save_safetensors(str(path), values)
    header_length = int.from_bytes(path.read_bytes()[:8], "little")
    header = json.loads(path.read_bytes()[8:8 + header_length])
    assert "__metadata__" in header and header["__metadata__"] is None
    digest = p.canonical_tensor_digest(path)
    loaded = mx.load(str(path))
    mx.eval(loaded["x"])
    assert digest["tensors"] == [{"name": "x", "dtype": "F32", "shape": [1, 2], "nbytes": 8}]
    assert loaded["x"].tolist() == values["x"].tolist()


def test_attempt3_catalog_wrapper_binds_authorization_json(tmp_path):
    import scripts.finetune_ds4 as finetune
    import scripts.ds4_segmented_pilot as pilot
    phase_spec = pilot.attempt3_phase_specs()["phase-a"]
    authorization = pilot.canonical_attempt3_authorization("phase-a", phase_spec,
                                                           catalog_source=ROOT / "scripts" / "finetune_ds4.py")
    authorization_file = tmp_path / "reviewed-attempt3-authorization.json"
    authorization_file.write_text(authorization, encoding="utf-8")
    class Args:
        hf_model = "/tmp/hf"; dataset_root = "/tmp/data"; mlx_work = "/tmp/mlx"
        ds4_root = "/tmp/ds4"; ds4_gguf = None; split_dir = "mlx-4096"
        fused_hf_model = None; ds4_imatrix = None; adapter_ds4 = None; mlx_lm_source = "fork"
        attempt3_authorization_file = str(authorization_file); attempt3_authorization_json = None
    command = finetune.command_catalog(Args())["ds4-segmented-pilot-attempt-3-phase-a"][0]
    assert "--authorization-json" in command
    assert "canonical_command_sha256" in command


def test_attempt3_catalog_requires_external_authorization_and_does_not_mint_one(tmp_path):
    import scripts.finetune_ds4 as finetune
    class Args:
        hf_model = "/tmp/hf"; dataset_root = "/tmp/data"; mlx_work = "/tmp/mlx"
        ds4_root = "/tmp/ds4"; ds4_gguf = None; split_dir = "mlx-4096"
        fused_hf_model = None; mlx_lm_source = "fork"
        fused_hf_model = None; ds4_imatrix = None; adapter_ds4 = None
        attempt3_authorization_file = None; attempt3_authorization_json = None
    catalog = finetune.command_catalog(Args())
    assert "ds4-segmented-pilot-attempt-3-phase-a" not in catalog
    assert "ds4-segmented-pilot-attempt-3-phase-b" not in catalog


def test_attempt2_historical_verifier_reserves_real_phase_b_output_path():
    p = load_pilot()
    assert "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-b" in p._ATTEMPT2_HISTORICAL_ABSENT_PATHS


def test_attempt2_verifier_ignores_all_replaceable_compatibility_roots(tmp_path, monkeypatch):
    p = load_pilot()
    replacements = {
        "_ATTEMPT2_CANONICAL_PRELOG": ("substituted", 0, "0" * 64),
        "_ATTEMPT2_CANONICAL_MANIFEST": ("substituted", 0, "0" * 64),
        "_ATTEMPT2_CANONICAL_FILES": (("substituted", 0, "0" * 64),),
        "_ATTEMPT2_CANONICAL_EXPECTED": {"files": []},
        "_ATTEMPT2_CANONICAL_FACTS": {},
        "_ATTEMPT2_CANONICAL_ABSENT_PATHS": (),
    }
    for name, replacement in replacements.items():
        monkeypatch.setattr(p, name, replacement)
    with pytest.raises(p.PilotError):
        p.verify_attempt2_historical_evidence(repo_root=tmp_path)


def test_phase_b_launch_uses_separate_authorization_from_phase_a(tmp_path):
    p = load_pilot()
    paths, phase_a_report, _ = _write_valid_attempt3_phase_a(tmp_path, p)
    phase_b_spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-b"]
    phase_b_authorization = dict(phase_a_report["authorization"])
    phase_b_authorization["canonical_command_sha256"] = hashlib.sha256(json.dumps(
        p.canonical_attempt3_command("phase-b", phase_b_spec), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    p.verify_attempt1_historical_evidence = lambda **_: phase_a_report["attempt_1_historical_evidence"]
    result = p.check_attempt3_launch("phase-b", paths["phase-b-log"], repo_root=tmp_path,
                                     workspace=tmp_path / "workspace",
                                     authorization_json=phase_b_authorization)
    assert result["authorization"] == phase_b_authorization


def test_final_publication_revalidates_phase_a_lineage_before_success(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    paths, report, spec, _, _ = _write_valid_attempt3_phase_b(tmp_path, p)
    phase_a_report = json.loads(paths["phase-a-report"].read_text(encoding="utf-8"))
    phase_a_report["attempt"] = 2
    paths["phase-a-report"].write_text(json.dumps(phase_a_report, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(p, "_remove_ok_markers", lambda *_: None)
    with pytest.raises(p.PilotError, match="canonical A3|Phase A report|attempt|Phase B requires"):
        p._write_success_evidence(
            "phase-b", report, phase_spec=spec,
            phase_a_authorization=report["phase_a_admission_lineage"]["authorization"],
            phase_a_admission_lineage=report["phase_a_admission_lineage"])
    assert not paths["final-ok"].exists()


def test_r4_attempt3_marker_schema_is_exact_and_fail_closed(tmp_path):
    p = load_pilot()
    paths, report, _ = _write_valid_attempt3_phase_a(tmp_path, p)
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
    marker = p._marker_fields("phase-a", "ok", paths["phase-a-report"], report, spec)
    source = report["resume_source"]
    marker.update({
        "resume_source": source["path"],
        "resume_source_file_sha256": source["file_sha256"],
        "resume_source_canonical_tensor_digest_v1": source["canonical_tensor_digest_v1"],
    })
    p._validate_attempt3_marker(marker, phase="phase-a", phase_spec=spec,
                                report=report, marker_path=paths["phase-a-ok"])
    for mutation in ("missing", "extra", "type", "value"):
        mutated = json.loads(json.dumps(marker))
        if mutation == "missing":
            del mutated["resume_source_file_sha256"]
        elif mutation == "extra":
            mutated["unexpected"] = True
        elif mutation == "type":
            mutated["exit_code"] = "0"
        else:
            mutated["namespace"] = "wrong"
        with pytest.raises(p.PilotError):
            p._validate_attempt3_marker(mutated, phase="phase-a", phase_spec=spec,
                                        report=report, marker_path=paths["phase-a-ok"])


def test_r4_phase_b_report_requires_immutable_phase_a_admission_lineage(tmp_path):
    p = load_pilot()
    paths, report, spec, _, _ = _write_valid_attempt3_phase_b(tmp_path, p)
    report.pop("phase_a_admission_lineage")
    with pytest.raises(p.PilotError, match="admission lineage"):
        p.validate_canonical_attempt3_report(report, phase_spec=spec,
                                             report_path=paths["phase-b-report"])


def test_r4_phase_a_admission_lineage_rejects_post_admission_report_or_marker_mutation(tmp_path):
    p = load_pilot()
    paths, report, spec, phase_a_report, phase_a_marker = _write_valid_attempt3_phase_b(tmp_path, p)
    lineage = p.capture_phase_a_admission_lineage(
        paths["phase-a-report"], paths["phase-a-ok"], phase_a_report, phase_a_marker)
    report["phase_a_admission_lineage"] = lineage
    report["contract_digest"] = p.contract_digest(
        "phase-b", report["effective"], report["identity_manifest"], phase_spec=spec,
        historical_evidence=report["attempt_1_historical_evidence"],
        historical_attempt2_evidence=report["attempt_2_historical_evidence"],
        authorization=report["authorization"], phase_a_admission_lineage=lineage)
    paths["phase-a-report"].write_bytes(paths["phase-a-report"].read_bytes() + b"\n")
    with pytest.raises(p.PilotError, match="admission lineage|Phase A report hash"):
        p.validate_canonical_attempt3_report(report, phase_spec=spec,
                                             report_path=paths["phase-b-report"])


def test_r5_final_report_requires_both_phase_report_hashes_and_exact_schema(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    paths, report, spec, phase_a_report, _ = _write_valid_attempt3_phase_b(tmp_path, p)
    p._write_success_evidence("phase-b", report, phase_spec=spec,
                              phase_a_authorization=phase_a_report["authorization"],
                              phase_a_admission_lineage=report["phase_a_admission_lineage"],
                              phase_b_authorization=report["authorization"],
                              trusted_identity=phase_a_report["identity_manifest"],
                              trusted_attempt_1_historical_evidence=phase_a_report["attempt_1_historical_evidence"],
                              trusted_attempt_2_historical_evidence=phase_a_report["attempt_2_historical_evidence"])
    final = json.loads(paths["final-report"].read_text(encoding="utf-8"))
    phase_a = json.loads(paths["phase-a-report"].read_text(encoding="utf-8"))
    phase_b = json.loads(paths["phase-b-report"].read_text(encoding="utf-8"))
    for mutation in ("missing", "extra", "type", "value"):
        mutated = json.loads(json.dumps(final))
        if mutation == "missing":
            del mutated["phase_a_report_sha256"]
        elif mutation == "extra":
            mutated["unexpected"] = True
        elif mutation == "type":
            mutated["phase_b_report_sha256"] = 1
        else:
            mutated["phase_b_report_sha256"] = "0" * 64
        with pytest.raises(p.PilotError):
            p.validate_final_attempt3_report(mutated, phase_a, phase_b,
                                             **_trusted_final_kwargs(report, phase_a_report))


def test_r5_final_writer_rejects_coordinated_b3_lineage_substitution(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    paths, report, spec, phase_a_report, _ = _write_valid_attempt3_phase_b(tmp_path, p)
    trusted = json.loads(json.dumps(report["phase_a_admission_lineage"]))
    substituted = json.loads(json.dumps(trusted))
    substituted["authorization"]["canonical_command_sha256"] = "f" * 64
    report["phase_a_admission_lineage"] = substituted
    report["contract_digest"] = p.contract_digest(
        "phase-b", report["effective"], report["identity_manifest"], phase_spec=spec,
        historical_evidence=report["attempt_1_historical_evidence"],
        historical_attempt2_evidence=report["attempt_2_historical_evidence"],
        authorization=report["authorization"], phase_a_admission_lineage=substituted)
    monkeypatch.setattr(p, "_remove_ok_markers", lambda *_: None)
    with pytest.raises(p.PilotError, match="retained pre-training lineage|admission lineage"):
        p._write_success_evidence("phase-b", report, phase_spec=spec,
                                  phase_a_authorization=phase_a_report["authorization"],
                                  phase_a_admission_lineage=trusted)
    assert not paths["final-ok"].exists()


@pytest.mark.parametrize("metadata", ["", "metadata", [], [1], 0, 1.5, False, True])
@pytest.mark.parametrize("checkpoint_key", ["phase-a-start-checkpoint", "phase-a-step1-checkpoint",
                                             "phase-a-step2-checkpoint", "phase-a-final-checkpoint"])
def test_every_invalid_metadata_type_fails_at_each_phase_a_checkpoint(tmp_path, metadata, checkpoint_key):
    p = load_pilot()
    workspace = tmp_path / "workspace"
    paths = p.attempt3_namespace(repo_root=tmp_path, workspace=workspace)
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=workspace)["phase-a"]
    paths["phase-a-output"].mkdir(parents=True)
    for key, raw in (("phase-a-start-checkpoint", b"a"), ("phase-a-step1-checkpoint", b"b"),
                     ("phase-a-step2-checkpoint", b"c"), ("phase-a-final-checkpoint", b"c")):
        write_safetensors(paths[key], {"x": ("U8", [1], raw)}, metadata=metadata if key == checkpoint_key else None)
    paths["phase-a-config"].write_text("{}", encoding="utf-8")
    before = paths[checkpoint_key].read_bytes()
    with pytest.raises(p.PilotError):
        p._validate_artifacts("phase-a", paths["phase-a-output"], phase_spec=spec)
    assert paths[checkpoint_key].read_bytes() == before


def test_catalog_binds_distinct_external_phase_authorizations(tmp_path):
    import scripts.finetune_ds4 as finetune
    import scripts.ds4_segmented_pilot as pilot
    specs = pilot.attempt3_phase_specs()
    auth_a = pilot.canonical_attempt3_authorization("phase-a", specs["phase-a"],
                                                     catalog_source=ROOT / "scripts" / "finetune_ds4.py")
    auth_b = pilot.canonical_attempt3_authorization("phase-b", specs["phase-b"],
                                                     catalog_source=ROOT / "scripts" / "finetune_ds4.py")
    class Args:
        hf_model = "/tmp/hf"; dataset_root = "/tmp/data"; mlx_work = "/tmp/mlx"
        ds4_root = "/tmp/ds4"; ds4_gguf = None; split_dir = "mlx-4096"
        fused_hf_model = None; ds4_imatrix = None; adapter_ds4 = None; mlx_lm_source = "fork"
        attempt3_authorization_file = None; attempt3_authorization_json = None
        attempt3_phase_a_authorization_file = None; attempt3_phase_a_authorization_json = auth_a
        attempt3_phase_b_authorization_file = None; attempt3_phase_b_authorization_json = auth_b
    catalog = finetune.command_catalog(Args())
    command_a = catalog["ds4-segmented-pilot-attempt-3-phase-a"][0]
    command_b = catalog["ds4-segmented-pilot-attempt-3-phase-b"][0]
    assert auth_a in command_a and auth_b in command_b and auth_a in command_b


def _install_r6_attempt2_fixture(tmp_path, monkeypatch, p):
    def mapped(raw):
        return tmp_path / "attempt2" / raw.lstrip("/").replace("/", "__")

    monkeypatch.setattr(p, "_historical_path", lambda raw, _root: mapped(raw))
    monkeypatch.setattr(p, "_attempt2_expected_path", lambda raw, _root: mapped(raw))
    prelog_raw = "agent-output/cmux-14-5-attempt-2/phase-a2-prelog-failure.md"
    manifest_raw = "agent-output/cmux-14-5-attempt-2/phase-a2-runtime-failure-manifest.json"
    file_paths = [
        "agent-output/cmux-14-5-attempt-2/phase-a-log.txt",
        "agent-output/cmux-14-5-attempt-2/phase-a-report.json",
        "agent-output/cmux-14-5-attempt-2/pilot-report.json",
        "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-a-fail",
        "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-fail",
        "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/0000001_adapters.safetensors",
        "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/0000002_adapters.safetensors",
        "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/adapter_config.json",
        "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/adapters.safetensors",
        "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/phase-a-start.safetensors",
    ]
    report = {"attempt": 2, "namespace": "ds4-segmented-pilot-attempt-2", "effective": {}, "identity_manifest": {}}
    payloads = {
        raw: (json.dumps(report, sort_keys=True).encode() if raw.endswith("pilot-report.json")
              else json.dumps(report, sort_keys=True).encode() if raw.endswith("phase-a-report.json")
              else f"synthetic:{index}".encode())
        for index, raw in enumerate(file_paths)
    }
    prelog_path = mapped(prelog_raw)
    prelog_path.parent.mkdir(parents=True, exist_ok=True)
    prelog_path.write_bytes(b"")
    files = []
    for raw, payload in payloads.items():
        path = mapped(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files.append((raw, len(payload), hashlib.sha256(payload).hexdigest()))
    expected_manifest = {
        "attempt": 2, "command_sha256": "a" * 64, "error": "synthetic failure", "exit_code": 1,
        "files": [{"path": raw, "size": size, "sha256": sha} for raw, size, sha in files],
        "lock_absent_after_exit": True,
        "lock_lifecycle": {"acquired": True, "path": "/synthetic/.ds4-ft.lock", "release_attempts": 1, "released": True},
        "namespace": "ds4-segmented-pilot-attempt-2", "ok_markers_absent": True, "retry_performed": False,
        "revision": "synthetic", "status": "fail", "story": "14.5b",
        "training_log_observations": {"checkpoint1_saved": True}, "wall_seconds": 1.0, "watchdog_cancelled": True,
    }
    manifest_payload = json.dumps(expected_manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_path = mapped(manifest_raw)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_payload)
    fixture = {
        "prelog_expected": (prelog_raw, 0, hashlib.sha256(b"").hexdigest()),
        "manifest_expected": (manifest_raw, len(manifest_payload), hashlib.sha256(manifest_payload).hexdigest()),
        "files_expected": tuple(files),
        "absent_expected": (
            "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-a-ok",
            "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-b-ok",
            "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-b-fail",
            "/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-ok",
            "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-b",
            "agent-output/cmux-14-5-attempt-2/phase-b-log.txt",
            "agent-output/cmux-14-5-attempt-2/phase-b-report.json",
            "/Volumes/Data NVME/mlx-ft/ds4/.ds4-ft.lock",
        ),
        "expected_manifest": expected_manifest,
    }
    return fixture, expected_manifest, report


def test_r6_attempt2_verifier_accepts_authentic_ten_file_eight_absence_fixture(tmp_path, monkeypatch):
    p = load_pilot()
    fixture, expected_manifest, report = _install_r6_attempt2_fixture(tmp_path, monkeypatch, p)
    result = p._verify_historical_evidence_snapshot(repo_root=tmp_path, **fixture)
    assert len(result["files"]) == 10
    assert len(p._ATTEMPT2_CANONICAL_ABSENT_PATHS) == 8
    assert result["manifest_snapshot"] == expected_manifest
    assert result["report_identity"]["attempt"] == report["attempt"]


@pytest.mark.parametrize("mutation", ["manifest", "target", "absence"])
def test_r6_attempt2_verifier_rejects_outer_semantic_target_and_absence_mutations(tmp_path, monkeypatch, mutation):
    p = load_pilot()
    fixture, expected_manifest, _ = _install_r6_attempt2_fixture(tmp_path, monkeypatch, p)
    if mutation == "manifest":
        broken = dict(expected_manifest)
        broken["exit_code"] = 0
        fixture["expected_manifest"] = broken
    elif mutation == "target":
        fixture["files_expected"] = (fixture["files_expected"][0][0], fixture["files_expected"][0][1], "0" * 64), *fixture["files_expected"][1:]
    else:
        absent = tmp_path / "attempt2" / "synthetic-absence"
        absent.touch()
        monkeypatch.setattr(p, "_historical_path", lambda raw, _root: absent if raw == p._ATTEMPT2_CANONICAL_ABSENT_PATHS[0] else tmp_path / "attempt2" / raw.lstrip("/").replace("/", "__"))
    with pytest.raises(p.PilotError):
        p._verify_historical_evidence_snapshot(repo_root=tmp_path, **fixture)


def test_r6_final_validator_requires_trusted_pretraining_lineage_and_authorization(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    paths, report, spec, phase_a_report, _ = _write_valid_attempt3_phase_b(tmp_path, p)
    p._write_success_evidence("phase-b", report, phase_spec=spec,
                              phase_a_authorization=phase_a_report["authorization"],
                              phase_a_admission_lineage=report["phase_a_admission_lineage"],
                              phase_b_authorization=report["authorization"],
                              trusted_identity=phase_a_report["identity_manifest"],
                              trusted_attempt_1_historical_evidence=phase_a_report["attempt_1_historical_evidence"],
                              trusted_attempt_2_historical_evidence=phase_a_report["attempt_2_historical_evidence"])
    final = json.loads(paths["final-report"].read_text(encoding="utf-8"))
    phase_a = json.loads(paths["phase-a-report"].read_text(encoding="utf-8"))
    phase_b = json.loads(paths["phase-b-report"].read_text(encoding="utf-8"))
    with pytest.raises(p.PilotError, match="trusted.*(lineage|authorization)|pre-training"):
        p.validate_final_attempt3_report(final, phase_a, phase_b)


@pytest.mark.parametrize("field", [
    "status", "provider_calls", "optimizer_updates", "steps", "global_progression",
    "total_active_wall_seconds", "phase_a", "phase_b", "output_path", "contract_digest",
    "non_claims", "attempt", "namespace", "phase_a_report_path", "phase_b_report_path",
    "phase_a_contract_digest", "phase_b_contract_digest", "phase_authorizations",
    "attempt_1_historical_evidence", "attempt_2_historical_evidence",
    "phase_a_report_sha256", "phase_b_report_sha256",
])
def test_r6_final_validator_rejects_every_final_field_type_or_value_mutation(tmp_path, monkeypatch, field):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    paths, report, spec, phase_a_report, _ = _write_valid_attempt3_phase_b(tmp_path, p)
    p._write_success_evidence("phase-b", report, phase_spec=spec,
                              phase_a_authorization=phase_a_report["authorization"],
                              phase_a_admission_lineage=report["phase_a_admission_lineage"],
                              phase_b_authorization=report["authorization"],
                              trusted_identity=phase_a_report["identity_manifest"],
                              trusted_attempt_1_historical_evidence=phase_a_report["attempt_1_historical_evidence"],
                              trusted_attempt_2_historical_evidence=phase_a_report["attempt_2_historical_evidence"])
    final = json.loads(paths["final-report"].read_text(encoding="utf-8"))
    phase_a = json.loads(paths["phase-a-report"].read_text(encoding="utf-8"))
    phase_b = json.loads(paths["phase-b-report"].read_text(encoding="utf-8"))
    mutated = json.loads(json.dumps(final))
    if field in {"provider_calls", "optimizer_updates", "attempt"}:
        mutated[field] = True
    elif field == "status":
        mutated[field] = 1
    elif field == "steps":
        mutated[field] = []
    elif field == "global_progression":
        mutated[field] = [0, 1, 2, 4]
    elif field == "total_active_wall_seconds":
        mutated[field] = "not-a-number"
    elif field in {"phase_a", "phase_b"}:
        mutated[field] = {}
    elif field == "output_path":
        mutated[field] = str(tmp_path / "substituted-output")
    elif field in {"contract_digest", "phase_a_contract_digest", "phase_b_contract_digest"}:
        mutated[field] = "0" * 64
    elif field == "non_claims":
        mutated[field] = []
    elif field == "namespace":
        mutated[field] = "substituted"
    elif field in {"phase_a_report_path", "phase_b_report_path"}:
        mutated[field] = str(tmp_path / "substituted-report.json")
    elif field == "phase_authorizations":
        mutated[field] = {}
    elif field == "attempt_1_historical_evidence":
        mutated[field] = []
    elif field == "attempt_2_historical_evidence":
        mutated[field] = {}
    else:
        mutated[field] = 1
    with pytest.raises(p.PilotError):
        p.validate_final_attempt3_report(
            mutated, phase_a, phase_b,
            **_trusted_final_kwargs(report, phase_a_report),
        )


@pytest.mark.parametrize("key", sorted({
    "phase", "status", "attempt", "namespace", "report_path", "report_sha256",
    "output_path", "timestamp", "exit_code", "contract_digest",
}))
def test_r6_marker_validator_rejects_exact_json_type_and_value_mutations(tmp_path, key):
    p = load_pilot()
    paths, report, _ = _write_valid_attempt3_phase_a(tmp_path, p)
    spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
    marker = p._marker_fields("phase-a", "ok", paths["phase-a-report"], report, spec)
    source = report["resume_source"]
    marker.update({
        "resume_source": source["path"],
        "resume_source_file_sha256": source["file_sha256"],
        "resume_source_canonical_tensor_digest_v1": source["canonical_tensor_digest_v1"],
    })
    mutated = json.loads(json.dumps(marker))
    mutated[key] = True if key in {"attempt", "timestamp", "exit_code"} else 1
    with pytest.raises(p.PilotError):
        p._validate_attempt3_marker(mutated, phase="phase-a", phase_spec=spec,
                                    report=report, marker_path=paths["phase-a-ok"])


def test_r6_tensor_digest_wraps_unhashable_dtype_as_pilot_error(tmp_path):
    p = load_pilot()
    path = tmp_path / "bad-dtype.safetensors"
    header = b'{"x":{"dtype":[],"shape":[1],"data_offsets":[0,1]}}'
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"x")
    with pytest.raises(p.PilotError, match="tensor schema|dtype"):
        p.canonical_tensor_digest(path)


def test_r7_tensor_descriptor_rejects_extra_keys(tmp_path):
    p = load_pilot()
    path = tmp_path / "extra-key.safetensors"
    header = b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1],"unexpected":true}}'
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"x")
    with pytest.raises(p.PilotError, match="tensor schema|keys"):
        p.canonical_tensor_digest(path)


@pytest.mark.parametrize("mutation", ["missing", "type", "value", "dtype-value", "shape-value", "layout"])
def test_r7_tensor_descriptor_schema_matrix_rejects_independent_mutations(tmp_path, mutation):
    p = load_pilot()
    descriptor = {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}
    if mutation == "missing":
        descriptor.pop("dtype")
    elif mutation == "type":
        descriptor["shape"] = {"value": 1}
    elif mutation == "value":
        descriptor["data_offsets"] = [1, 1]
    elif mutation == "dtype-value":
        descriptor["dtype"] = "F32"
    elif mutation == "shape-value":
        descriptor["shape"] = [2]
    else:
        descriptor["layout"] = "row-major"
    header = json.dumps({"x": descriptor}, separators=(",", ":")).encode()
    path = tmp_path / f"bad-{mutation}.safetensors"
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"x")
    with pytest.raises(p.PilotError, match="tensor schema|shape|offsets|contiguous"):
        p.canonical_tensor_digest(path)


@pytest.mark.parametrize("field", ["name", "dtype", "shape"])
def test_r7_tensor_descriptor_value_mutations_change_canonical_digest(tmp_path, field):
    p = load_pilot()
    first_header = {"x": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}
    second_header = json.loads(json.dumps(first_header))
    if field == "name":
        second_header = {"y": second_header.pop("x")}
    elif field == "dtype":
        second_header["x"]["dtype"] = "I8"
    elif field == "shape":
        second_header["x"]["shape"] = [1, 1]
    else:
        raise AssertionError(field)
    first = tmp_path / f"{field}-first.safetensors"
    second = tmp_path / f"{field}-second.safetensors"
    first_header_bytes = json.dumps(first_header, separators=(",", ":")).encode()
    second_header_bytes = json.dumps(second_header, separators=(",", ":")).encode()
    first.write_bytes(len(first_header_bytes).to_bytes(8, "little") + first_header_bytes + b"x")
    second.write_bytes(len(second_header_bytes).to_bytes(8, "little") + second_header_bytes + b"x")
    assert p.canonical_tensor_digest(first)["canonical_tensor_digest_v1"] != p.canonical_tensor_digest(second)["canonical_tensor_digest_v1"]


def test_r7_tensor_payload_mutation_changes_canonical_digest(tmp_path):
    p = load_pilot()
    header = b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    first.write_bytes(len(header).to_bytes(8, "little") + header + b"x")
    second.write_bytes(len(header).to_bytes(8, "little") + header + b"y")
    assert p.canonical_tensor_digest(first)["canonical_tensor_digest_v1"] != p.canonical_tensor_digest(second)["canonical_tensor_digest_v1"]


def test_r7_final_validator_rejects_coordinated_phase_snapshot_mutation(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    paths, report, spec, phase_a_report, _ = _write_valid_attempt3_phase_b(tmp_path, p)
    p._write_success_evidence("phase-b", report, phase_spec=spec,
                              phase_a_authorization=phase_a_report["authorization"],
                              phase_a_admission_lineage=report["phase_a_admission_lineage"],
                              phase_b_authorization=report["authorization"],
                              trusted_identity=phase_a_report["identity_manifest"],
                              trusted_attempt_1_historical_evidence=phase_a_report["attempt_1_historical_evidence"],
                              trusted_attempt_2_historical_evidence=phase_a_report["attempt_2_historical_evidence"])
    final = json.loads(paths["final-report"].read_text(encoding="utf-8"))
    phase_a = json.loads(paths["phase-a-report"].read_text(encoding="utf-8"))
    phase_b = json.loads(paths["phase-b-report"].read_text(encoding="utf-8"))
    mutated_a = json.loads(json.dumps(phase_a))
    mutated_a["provider_calls"] = 99
    mutated_a["optimizer_updates"] = 99
    mutated_final = json.loads(json.dumps(final))
    mutated_final["phase_a"] = mutated_a
    mutated_final["provider_calls"] = mutated_a["provider_calls"] + phase_b["provider_calls"]
    mutated_final["optimizer_updates"] = mutated_a["optimizer_updates"] + phase_b["optimizer_updates"]
    mutated_final["steps"] = mutated_a["steps"] + phase_b["steps"]
    with pytest.raises(p.PilotError, match="snapshot|report hash|canonical A3"):
        p.validate_final_attempt3_report(
            mutated_final, mutated_a, phase_b,
            **_trusted_final_kwargs(report, phase_a_report),
        )


def test_r8_final_validator_rejects_trusted_root_substitution(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    paths, report, spec, phase_a_report, _ = _write_valid_attempt3_phase_b(tmp_path, p)
    p._write_success_evidence(
        "phase-b", report, phase_spec=spec,
        phase_a_authorization=phase_a_report["authorization"],
        phase_a_admission_lineage=report["phase_a_admission_lineage"],
        phase_b_authorization=report["authorization"],
        trusted_identity=phase_a_report["identity_manifest"],
        trusted_attempt_1_historical_evidence=phase_a_report["attempt_1_historical_evidence"],
        trusted_attempt_2_historical_evidence=phase_a_report["attempt_2_historical_evidence"],
    )
    final = json.loads(paths["final-report"].read_text(encoding="utf-8"))
    phase_a = json.loads(paths["phase-a-report"].read_text(encoding="utf-8"))
    phase_b = json.loads(paths["phase-b-report"].read_text(encoding="utf-8"))
    phase_b["identity_manifest"]["immutable"]["git_head"] = "2" * 40
    paths["phase-b-report"].write_text(json.dumps(phase_b, sort_keys=True), encoding="utf-8")
    with pytest.raises(p.PilotError, match="identity"):
        p.validate_final_attempt3_report(
            final, phase_a, phase_b,
            **_trusted_final_kwargs(report, phase_a_report),
        )


def test_r8_final_validator_uses_snapshot_hashes_without_report_reopen(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    paths, report, spec, phase_a_report, _ = _write_valid_attempt3_phase_b(tmp_path, p)
    p._write_success_evidence(
        "phase-b", report, phase_spec=spec,
        phase_a_authorization=phase_a_report["authorization"],
        phase_a_admission_lineage=report["phase_a_admission_lineage"],
        phase_b_authorization=report["authorization"],
        trusted_identity=phase_a_report["identity_manifest"],
        trusted_attempt_1_historical_evidence=phase_a_report["attempt_1_historical_evidence"],
        trusted_attempt_2_historical_evidence=phase_a_report["attempt_2_historical_evidence"],
    )
    final = json.loads(paths["final-report"].read_text(encoding="utf-8"))
    phase_a = json.loads(paths["phase-a-report"].read_text(encoding="utf-8"))
    phase_b = json.loads(paths["phase-b-report"].read_text(encoding="utf-8"))
    original_file_sha256 = p.file_sha256

    def guarded_file_sha256(path, *args, **kwargs):
        if pathlib.Path(path) in {paths["phase-a-report"], paths["phase-b-report"]}:
            raise AssertionError(f"phase report hash reopened after snapshot capture: {path}")
        return original_file_sha256(path, *args, **kwargs)

    monkeypatch.setattr(p, "file_sha256", guarded_file_sha256)
    p.validate_final_attempt3_report(
        final, phase_a, phase_b,
        **_trusted_final_kwargs(report, phase_a_report),
    )


def _r9_set_nested(value, path, replacement):
    cursor = value
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement


@pytest.mark.parametrize("path,replacement", [
    (("attempt",), 3),
    (("command_sha256",), "0" * 64),
    (("error",), "substituted"),
    (("exit_code",), 0),
    (("lock_absent_after_exit",), False),
    (("namespace",), "substituted"),
    (("ok_markers_absent",), False),
    (("retry_performed",), True),
    (("revision",), "substituted"),
    (("status",), "ok"),
    (("story",), "substituted"),
    (("wall_seconds",), 0.0),
    (("watchdog_cancelled",), False),
    (("lock_lifecycle", "acquired"), False),
    (("lock_lifecycle", "path"), "/substituted-lock"),
    (("lock_lifecycle", "release_attempts"), 2),
    (("lock_lifecycle", "released"), False),
    (("training_log_observations", "checkpoint1_saved"), False),
    (("training_log_observations", "checkpoint2_saved"), False),
    (("training_log_observations", "iter1_loss"), "substituted"),
    (("training_log_observations", "iter1_val_loss"), "substituted"),
    (("training_log_observations", "iter2_loss"), "substituted"),
    (("training_log_observations", "iter2_val_loss"), "substituted"),
])
def test_r9_attempt2_rejects_every_semantic_and_nested_manifest_field(tmp_path, monkeypatch, path, replacement):
    p = load_pilot()
    fixture, expected_manifest, _ = _install_r6_attempt2_fixture(tmp_path, monkeypatch, p)
    mutated = json.loads(json.dumps(expected_manifest))
    _r9_set_nested(mutated, path, replacement)
    fixture["expected_manifest"] = mutated
    with pytest.raises(p.PilotError):
        p._verify_historical_evidence_snapshot(repo_root=tmp_path, **fixture)


@pytest.mark.parametrize("index", range(10))
@pytest.mark.parametrize("field", ["path", "size", "sha256"])
def test_r9_attempt2_rejects_each_runtime_descriptor_path_size_and_hash(tmp_path, monkeypatch, index, field):
    p = load_pilot()
    fixture, _, _ = _install_r6_attempt2_fixture(tmp_path, monkeypatch, p)
    descriptors = list(fixture["files_expected"])
    raw, size, sha256 = descriptors[index]
    if field == "path":
        raw = raw + ".substituted"
    elif field == "size":
        size += 1
    else:
        sha256 = "0" * 64
    descriptors[index] = (raw, size, sha256)
    fixture["files_expected"] = tuple(descriptors)
    with pytest.raises(p.PilotError):
        p._verify_historical_evidence_snapshot(repo_root=tmp_path, **fixture)


@pytest.mark.parametrize("index", range(10))
def test_r9_attempt2_rejects_each_runtime_target_byte_mutation(tmp_path, monkeypatch, index):
    p = load_pilot()
    fixture, _, _ = _install_r6_attempt2_fixture(tmp_path, monkeypatch, p)
    raw = fixture["files_expected"][index][0]
    target = p._attempt2_expected_path(raw, tmp_path)
    payload = target.read_bytes()
    target.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    with pytest.raises(p.PilotError):
        p._verify_historical_evidence_snapshot(repo_root=tmp_path, **fixture)


@pytest.mark.parametrize("index", range(8))
def test_r9_attempt2_rejects_each_absence_fact_violation(tmp_path, monkeypatch, index):
    p = load_pilot()
    fixture, _, _ = _install_r6_attempt2_fixture(tmp_path, monkeypatch, p)
    raw = fixture["absent_expected"][index]
    p._historical_path(raw, tmp_path).touch()
    with pytest.raises(p.PilotError, match="absence"):
        p._verify_historical_evidence_snapshot(repo_root=tmp_path, **fixture)


@pytest.mark.parametrize("field", ["path", "size", "sha256"])
def test_r9_attempt2_rejects_each_prelog_outer_descriptor_mutation(tmp_path, monkeypatch, field):
    p = load_pilot()
    fixture, _, _ = _install_r6_attempt2_fixture(tmp_path, monkeypatch, p)
    raw, size, sha256 = fixture["prelog_expected"]
    if field == "path":
        raw += ".substituted"
    elif field == "size":
        size += 1
    else:
        sha256 = "0" * 64
    fixture["prelog_expected"] = (raw, size, sha256)
    with pytest.raises(p.PilotError):
        p._verify_historical_evidence_snapshot(repo_root=tmp_path, **fixture)


@pytest.mark.parametrize("field", ["path", "size", "sha256"])
def test_r9_attempt2_rejects_each_manifest_outer_descriptor_mutation(tmp_path, monkeypatch, field):
    p = load_pilot()
    fixture, _, _ = _install_r6_attempt2_fixture(tmp_path, monkeypatch, p)
    raw, size, sha256 = fixture["manifest_expected"]
    if field == "path":
        raw += ".substituted"
    elif field == "size":
        size += 1
    else:
        sha256 = "0" * 64
    fixture["manifest_expected"] = (raw, size, sha256)
    with pytest.raises(p.PilotError):
        p._verify_historical_evidence_snapshot(repo_root=tmp_path, **fixture)


def test_r9_attempt2_rejects_duplicate_key_manifest_json(tmp_path, monkeypatch):
    p = load_pilot()
    fixture, _, _ = _install_r6_attempt2_fixture(tmp_path, monkeypatch, p)
    raw = b'{"attempt":2,"attempt":2}'
    path = p._historical_path(fixture["manifest_expected"][0], tmp_path)
    path.write_bytes(raw)
    fixture["manifest_expected"] = (fixture["manifest_expected"][0], len(raw), hashlib.sha256(raw).hexdigest())
    with pytest.raises(p.PilotError, match="JSON"):
        p._verify_historical_evidence_snapshot(repo_root=tmp_path, **fixture)


def test_r9_attempt2_rejects_coordinated_manifest_path_hash_target_substitution(tmp_path, monkeypatch):
    p = load_pilot()
    fixture, expected_manifest, _ = _install_r6_attempt2_fixture(tmp_path, monkeypatch, p)
    index = 6
    raw, _, _ = fixture["files_expected"][index]
    target = p._attempt2_expected_path(raw, tmp_path)
    payload = bytes([target.read_bytes()[0] ^ 1]) + target.read_bytes()[1:]
    target.write_bytes(payload)
    replacement_hash = hashlib.sha256(payload).hexdigest()
    descriptors = list(fixture["files_expected"])
    descriptors[index] = (raw + ".substituted", len(payload), replacement_hash)
    fixture["files_expected"] = tuple(descriptors)
    manifest = json.loads(json.dumps(expected_manifest))
    manifest["files"][index] = {"path": raw + ".substituted", "size": len(payload), "sha256": replacement_hash}
    manifest_path = p._historical_path(fixture["manifest_expected"][0], tmp_path)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_path.write_bytes(encoded)
    fixture["manifest_expected"] = (fixture["manifest_expected"][0], len(encoded), hashlib.sha256(encoded).hexdigest())
    with pytest.raises(p.PilotError, match="manifest"):
        p._verify_historical_evidence_snapshot(repo_root=tmp_path, **fixture)


@pytest.mark.parametrize("roots", [
    ("trusted_phase_a_authorization",),
    ("trusted_phase_b_authorization",),
    ("trusted_identity",),
    ("trusted_attempt_1_historical_evidence",),
    ("trusted_attempt_2_historical_evidence",),
    ("trusted_phase_a_admission_lineage",),
    ("trusted_phase_a_authorization", "trusted_phase_b_authorization"),
    ("trusted_identity", "trusted_attempt_1_historical_evidence"),
    ("trusted_attempt_1_historical_evidence", "trusted_attempt_2_historical_evidence"),
])
def test_r9_final_validator_rejects_every_trusted_root_substitution_and_combination(tmp_path, monkeypatch, roots):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    paths, report, spec, phase_a_report, _ = _write_valid_attempt3_phase_b(tmp_path, p)
    trusted = _trusted_final_kwargs(report, phase_a_report)
    p._write_success_evidence("phase-b", report, phase_spec=spec,
                              phase_a_authorization=phase_a_report["authorization"],
                              phase_a_admission_lineage=report["phase_a_admission_lineage"],
                              phase_b_authorization=report["authorization"],
                              trusted_identity=phase_a_report["identity_manifest"],
                              trusted_attempt_1_historical_evidence=phase_a_report["attempt_1_historical_evidence"],
                              trusted_attempt_2_historical_evidence=phase_a_report["attempt_2_historical_evidence"])
    final = json.loads(paths["final-report"].read_text(encoding="utf-8"))
    phase_a = json.loads(paths["phase-a-report"].read_text(encoding="utf-8"))
    phase_b = json.loads(paths["phase-b-report"].read_text(encoding="utf-8"))
    mutated = json.loads(json.dumps(trusted))
    for root in roots:
        value = mutated[root]
        if isinstance(value, dict):
            value = json.loads(json.dumps(value))
            if "revision" in value:
                value["revision"] = "9" * 40
            elif "immutable" in value:
                value["immutable"]["git_head"] = "9" * 40
            else:
                value["substituted"] = True
        elif isinstance(value, list):
            value = json.loads(json.dumps(value))
            value[0] = "9" * 64
        mutated[root] = value
    with pytest.raises(p.PilotError):
        p.validate_final_attempt3_report(final, phase_a, phase_b, **mutated)


@pytest.mark.parametrize("phase,key", [
    ("phase-a", "phase-a-start-checkpoint"), ("phase-a", "phase-a-step1-checkpoint"),
    ("phase-a", "phase-a-step2-checkpoint"), ("phase-a", "phase-a-final-checkpoint"),
    ("phase-b", "phase-b-start-checkpoint"), ("phase-b", "phase-b-step1-checkpoint"),
    ("phase-b", "phase-b-final-checkpoint"),
])
@pytest.mark.parametrize("metadata", [_METADATA_OMITTED, None, {"source": "r9"}])
def test_r9_metadata_absent_null_object_matrix_covers_every_a3_b3_artifact(tmp_path, phase, key, metadata):
    p = load_pilot()
    if phase == "phase-a":
        paths, _, marker = _write_valid_attempt3_phase_a(tmp_path, p)
        spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
    else:
        paths, _, spec, _, _ = _write_valid_attempt3_phase_b(tmp_path, p)
    raw_payloads = {
        "phase-a-start-checkpoint": b"a", "phase-a-step1-checkpoint": b"b",
        "phase-a-step2-checkpoint": b"c", "phase-a-final-checkpoint": b"c",
        "phase-b-start-checkpoint": b"d", "phase-b-step1-checkpoint": b"e",
        "phase-b-final-checkpoint": b"e",
    }
    write_safetensors(paths[key], {"x": ("U8", [1], raw_payloads[key])}, metadata=metadata)
    result = p._validate_artifacts(phase, paths[f"{phase}-output"], phase_spec=spec)
    assert result["start"]["tensors"] == [{"name": "x", "dtype": "U8", "shape": [1], "nbytes": 1}]


@pytest.mark.parametrize("phase,key", [
    ("phase-a", "phase-a-start-checkpoint"), ("phase-a", "phase-a-step1-checkpoint"),
    ("phase-a", "phase-a-step2-checkpoint"), ("phase-a", "phase-a-final-checkpoint"),
    ("phase-b", "phase-b-start-checkpoint"), ("phase-b", "phase-b-step1-checkpoint"),
    ("phase-b", "phase-b-final-checkpoint"),
])
def test_r9_invalid_metadata_position_rejects_canonical_publication_before_success_markers(tmp_path, phase, key):
    p = load_pilot()
    if phase == "phase-a":
        paths, report, _ = _write_valid_attempt3_phase_a(tmp_path, p)
        spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
        phase_a_report = None
    else:
        paths, report, spec, phase_a_report, _ = _write_valid_attempt3_phase_b(tmp_path, p)
    raw_payloads = {
        "phase-a-start-checkpoint": b"a", "phase-a-step1-checkpoint": b"b",
        "phase-a-step2-checkpoint": b"c", "phase-a-final-checkpoint": b"c",
        "phase-b-start-checkpoint": b"d", "phase-b-step1-checkpoint": b"e",
        "phase-b-final-checkpoint": b"e",
    }
    write_safetensors(paths[key], {"x": ("U8", [1], raw_payloads[key])}, metadata="invalid")
    for marker_key in ("phase-a-ok", "phase-b-ok", "final-ok"):
        paths[marker_key].unlink(missing_ok=True)
    kwargs = {"historical_evidence": report["attempt_1_historical_evidence"],
              "historical_attempt2_evidence": report["attempt_2_historical_evidence"],
              "trusted_identity": report["identity_manifest"]}
    if phase == "phase-b":
        kwargs.update({"trusted_phase_a_authorization": phase_a_report["authorization"],
                       "trusted_phase_a_admission_lineage": report["phase_a_admission_lineage"]})
    with pytest.raises(p.PilotError, match="metadata|artifact"):
        p.validate_canonical_attempt3_report(report, phase_spec=spec,
                                             report_path=paths[f"{phase}-report"], **kwargs)
    assert not paths["phase-a-ok"].exists()
    assert not paths["phase-b-ok"].exists()
    assert not paths["final-ok"].exists()


@pytest.mark.parametrize("kind", ["phase-a", "phase-b", "final"])
def test_r9_marker_missing_extra_type_value_and_resume_matrix(tmp_path, monkeypatch, kind):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    if kind == "phase-a":
        paths, report, marker = _write_valid_attempt3_phase_a(tmp_path, p)
        spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
        marker_path = paths["phase-a-ok"]
        final = False
    else:
        paths, report, spec, phase_a_report, _ = _write_valid_attempt3_phase_b(tmp_path, p)
        if kind == "phase-b":
            marker = p._marker_fields("phase-b", "ok", paths["phase-b-report"], report, spec)
            marker_path = paths["phase-b-ok"]
            final = False
        else:
            p._write_success_evidence("phase-b", report, phase_spec=spec,
                                      phase_a_authorization=phase_a_report["authorization"],
                                      phase_a_admission_lineage=report["phase_a_admission_lineage"],
                                      phase_b_authorization=report["authorization"],
                                      trusted_identity=phase_a_report["identity_manifest"],
                                      trusted_attempt_1_historical_evidence=phase_a_report["attempt_1_historical_evidence"],
                                      trusted_attempt_2_historical_evidence=phase_a_report["attempt_2_historical_evidence"])
            marker_path = paths["final-ok"]
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            report = json.loads(paths["final-report"].read_text(encoding="utf-8"))
            final = True
    expected = set(p._ATTEMPT3_MARKER_BASE_KEYS)
    if kind == "phase-a":
        expected = set(p._ATTEMPT3_MARKER_PHASE_A_KEYS)
    for key in sorted(expected):
        mutated = json.loads(json.dumps(marker))
        del mutated[key]
        with pytest.raises(p.PilotError):
            p._validate_attempt3_marker(mutated, phase="phase-b" if kind != "phase-a" else "phase-a",
                                        phase_spec=spec, report=report, marker_path=marker_path, final=final)
    mutated = json.loads(json.dumps(marker))
    mutated["unexpected"] = True
    with pytest.raises(p.PilotError):
        p._validate_attempt3_marker(mutated, phase="phase-b" if kind != "phase-a" else "phase-a",
                                    phase_spec=spec, report=report, marker_path=marker_path, final=final)
    for key in sorted(expected):
        mutated = json.loads(json.dumps(marker))
        if key in {"status", "phase", "namespace", "report_path", "output_path", "resume_source"}:
            mutated[key] = 1
        elif key in {"attempt", "exit_code"}:
            mutated[key] = 4
        elif key == "timestamp":
            mutated[key] = 0
        else:
            mutated[key] = "0" * 64
        with pytest.raises(p.PilotError):
            p._validate_attempt3_marker(mutated, phase="phase-b" if kind != "phase-a" else "phase-a",
                                        phase_spec=spec, report=report, marker_path=marker_path, final=final)


@pytest.mark.parametrize("phase,failures", [("phase-a", 2), ("phase-b", 4)])
def test_r9_attempt3_success_publication_rolls_back_every_write_seam(tmp_path, monkeypatch, phase, failures):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    if phase == "phase-a":
        paths, report, _ = _write_valid_attempt3_phase_a(tmp_path, p)
        spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")["phase-a"]
        kwargs = {}
    else:
        paths, report, spec, phase_a_report, _ = _write_valid_attempt3_phase_b(tmp_path, p)
        kwargs = {"phase_a_authorization": phase_a_report["authorization"],
                  "phase_a_admission_lineage": report["phase_a_admission_lineage"],
                  "phase_b_authorization": report["authorization"],
                  "trusted_identity": phase_a_report["identity_manifest"],
                  "trusted_attempt_1_historical_evidence": phase_a_report["attempt_1_historical_evidence"],
                  "trusted_attempt_2_historical_evidence": phase_a_report["attempt_2_historical_evidence"]}
    original = p.atomic_write_json
    phase_a_marker_bytes = paths["phase-a-ok"].read_bytes() if phase == "phase-b" else None
    for failure_number in range(1, failures + 1):
        calls = 0
        def fail_at_seam(path, value):
            nonlocal calls
            calls += 1
            if calls == failure_number:
                raise OSError(f"r9 seam {failure_number}")
            return original(path, value)
        monkeypatch.setattr(p, "atomic_write_json", fail_at_seam)
        def remove_tmp_markers(_attempt):
            for marker_key in ("phase-a-ok", "phase-b-ok", "final-ok"):
                paths[marker_key].unlink(missing_ok=True)
        monkeypatch.setattr(p, "_remove_ok_markers", remove_tmp_markers)
        remove_tmp_markers(3)
        if phase_a_marker_bytes is not None:
            paths["phase-a-ok"].write_bytes(phase_a_marker_bytes)
        with pytest.raises(OSError, match=f"r9 seam {failure_number}"):
            p._write_success_evidence(phase, report, phase_spec=spec, **kwargs)
        assert not paths["phase-a-ok"].exists()
        assert not paths["phase-b-ok"].exists()
        assert not paths["final-ok"].exists()
        monkeypatch.setattr(p, "atomic_write_json", original)


def test_r10_attempt3_wrapper_requires_explicit_authorization(tmp_path):
    import scripts.finetune_ds4 as finetune

    with pytest.raises(Exception, match="authorization"):
        finetune._pilot_attempt3_command(
            ROOT, ROOT / "scripts" / "ds4_segmented_pilot.py", tmp_path, "phase-a")


def test_r10_generic_authorization_never_mints_or_exposes_phase_b3():
    import scripts.finetune_ds4 as finetune
    import scripts.ds4_segmented_pilot as pilot

    class Args:
        hf_model = "/tmp/hf"
        dataset_root = "/tmp/data"
        mlx_work = "/tmp/mlx"
        ds4_root = "/tmp/ds4"
        ds4_gguf = None
        split_dir = "mlx-4096"
        fused_hf_model = None
        ds4_imatrix = None
        adapter_ds4 = None
        mlx_lm_source = "release"
        attempt3_authorization_file = None
        attempt3_authorization_json = pilot.canonical_attempt3_authorization(
            "phase-a", pilot.attempt3_phase_specs()["phase-a"],
            catalog_source=ROOT / "scripts" / "finetune_ds4.py")
        attempt3_phase_a_authorization_file = None
        attempt3_phase_a_authorization_json = None
        attempt3_phase_b_authorization_file = None
        attempt3_phase_b_authorization_json = None

    catalog = finetune.command_catalog(Args())
    assert "ds4-segmented-pilot-attempt-3-phase-a" in catalog
    assert "ds4-segmented-pilot-attempt-3-phase-b" not in catalog


def test_r10_phase_b3_requires_distinct_phase_a_and_phase_b_authorizations():
    import scripts.finetune_ds4 as finetune
    import scripts.ds4_segmented_pilot as pilot

    class Args:
        hf_model = "/tmp/hf"
        dataset_root = "/tmp/data"
        mlx_work = "/tmp/mlx"
        ds4_root = "/tmp/ds4"
        ds4_gguf = None
        split_dir = "mlx-4096"
        fused_hf_model = None
        ds4_imatrix = None
        adapter_ds4 = None
        mlx_lm_source = "release"
        attempt3_authorization_file = None
        attempt3_authorization_json = None
        attempt3_phase_a_authorization_file = None
        attempt3_phase_a_authorization_json = pilot.canonical_attempt3_authorization(
            "phase-a", pilot.attempt3_phase_specs()["phase-a"],
            catalog_source=ROOT / "scripts" / "finetune_ds4.py")
        attempt3_phase_b_authorization_file = None
        attempt3_phase_b_authorization_json = pilot.canonical_attempt3_authorization(
            "phase-b", pilot.attempt3_phase_specs()["phase-b"],
            catalog_source=ROOT / "scripts" / "finetune_ds4.py")

    catalog = finetune.command_catalog(Args())
    command_a = catalog["ds4-segmented-pilot-attempt-3-phase-a"][0]
    command_b = catalog["ds4-segmented-pilot-attempt-3-phase-b"][0]
    assert Args.attempt3_phase_a_authorization_json in command_a
    assert Args.attempt3_phase_b_authorization_json in command_b
    assert Args.attempt3_phase_a_authorization_json in command_b
    assert Args.attempt3_phase_b_authorization_json != Args.attempt3_phase_a_authorization_json


@pytest.mark.parametrize("kind", ["prelog", "manifest"])
def test_r10_attempt2_outer_bytes_and_manifest_semantics_are_immutable(tmp_path, monkeypatch, kind):
    p = load_pilot()
    fixture, expected_manifest, _ = _install_r6_attempt2_fixture(tmp_path, monkeypatch, p)
    if kind == "prelog":
        path = p._historical_path(fixture["prelog_expected"][0], tmp_path)
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        manifest_path = p._historical_path(fixture["manifest_expected"][0], tmp_path)
        mutated = json.loads(json.dumps(expected_manifest))
        mutated["error"] = "substituted"
        payload = json.dumps(mutated, sort_keys=True, separators=(",", ":")).encode()
        manifest_path.write_bytes(payload)
        fixture["manifest_expected"] = (
            fixture["manifest_expected"][0], len(payload), hashlib.sha256(payload).hexdigest())
    with pytest.raises(p.PilotError):
        p._verify_historical_evidence_snapshot(repo_root=tmp_path, **fixture)


@pytest.mark.parametrize("header_kind", [
    "missing-key", "extra-key", "duplicate-key", "bad-dtype", "bad-shape",
    "bad-layout", "payload-out-of-bounds", "two-tensor-overlap",
])
def test_r10_safetensors_negative_matrix_is_fail_closed(tmp_path, header_kind):
    p = load_pilot()
    path = tmp_path / f"{header_kind}.safetensors"
    import struct
    valid = {"x": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}
    if header_kind == "missing-key":
        header = {"x": {"dtype": "U8", "shape": [1]}}
        raw_header = json.dumps(header).encode()
        payload = b"x"
    elif header_kind == "extra-key":
        header = {"x": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1], "extra": 1}}
        raw_header = json.dumps(header).encode()
        payload = b"x"
    elif header_kind == "duplicate-key":
        item = json.dumps(valid["x"], separators=(",", ":"))
        raw_header = f'{{"x":{item},"x":{item}}}'.encode()
        payload = b"x"
    elif header_kind == "bad-dtype":
        header = {"x": {"dtype": "NOPE", "shape": [1], "data_offsets": [0, 1]}}
        raw_header = json.dumps(header).encode()
        payload = b"x"
    elif header_kind == "bad-shape":
        header = {"x": {"dtype": "U8", "shape": ["1"], "data_offsets": [0, 1]}}
        raw_header = json.dumps(header).encode()
        payload = b"x"
    elif header_kind == "bad-layout":
        raw_header = b"[]"
        payload = b""
    elif header_kind == "payload-out-of-bounds":
        header = {"x": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}}
        raw_header = json.dumps(header).encode()
        payload = b"x"
    else:
        header = {
            "x": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
            "y": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]},
        }
        raw_header = json.dumps(header).encode()
        payload = b"xyz"
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)
    with pytest.raises(p.PilotError):
        p.canonical_tensor_digest(path)


@pytest.mark.parametrize("phase,key", [
    ("phase-a", "phase-a-start-checkpoint"), ("phase-a", "phase-a-step1-checkpoint"),
    ("phase-a", "phase-a-step2-checkpoint"), ("phase-a", "phase-a-final-checkpoint"),
    ("phase-b", "phase-b-start-checkpoint"), ("phase-b", "phase-b-step1-checkpoint"),
    ("phase-b", "phase-b-final-checkpoint"),
])
@pytest.mark.parametrize("metadata", ["invalid", [], 1, 1.5, True])
def test_r10_every_invalid_metadata_type_blocks_artifact_publication(tmp_path, phase, key, metadata):
    import struct
    p = load_pilot()
    if phase == "phase-a":
        paths, _, _ = _write_valid_attempt3_phase_a(tmp_path, p)
        spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")[phase]
    else:
        paths, _, spec, _, _ = _write_valid_attempt3_phase_b(tmp_path, p)
    for marker_key in ("phase-a-ok", "phase-b-ok", "final-ok"):
        paths[marker_key].unlink(missing_ok=True)
    raw = paths[key].read_bytes()
    payload = raw[8 + struct.unpack("<Q", raw[:8])[0]:]
    write_safetensors(paths[key], {"x": ("U8", [len(payload)], payload)}, metadata=metadata)
    with pytest.raises(p.PilotError, match="metadata|artifact"):
        p._validate_artifacts(phase, paths[f"{phase}-output"], phase_spec=spec)
    assert not paths["phase-a-ok"].exists()
    assert not paths["phase-b-ok"].exists()
    assert not paths["final-ok"].exists()


@pytest.mark.parametrize("phase,field", [
    ("phase-a", "revision"), ("phase-a", "canonical_command_sha256"),
    ("phase-a", "pilot_source_sha256"), ("phase-a", "catalog_source_sha256"),
    ("phase-a", "protected_files_manifest_sha256"),
    ("phase-a", "attempt2_runtime_manifest_sha256"),
    ("phase-b", "revision"), ("phase-b", "canonical_command_sha256"),
    ("phase-b", "pilot_source_sha256"), ("phase-b", "catalog_source_sha256"),
    ("phase-b", "protected_files_manifest_sha256"),
    ("phase-b", "attempt2_runtime_manifest_sha256"),
])
def test_r10_final_validator_rejects_each_phase_authorization_snapshot_field(tmp_path, monkeypatch, phase, field):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    paths, report_b, spec_b, report_a, _ = _write_valid_attempt3_phase_b(tmp_path, p)
    p._write_success_evidence(
        "phase-b", report_b, phase_spec=spec_b,
        phase_a_authorization=report_a["authorization"],
        phase_a_admission_lineage=report_b["phase_a_admission_lineage"],
        phase_b_authorization=report_b["authorization"],
        trusted_identity=report_a["identity_manifest"],
        trusted_attempt_1_historical_evidence=report_a["attempt_1_historical_evidence"],
        trusted_attempt_2_historical_evidence=report_a["attempt_2_historical_evidence"])
    final = json.loads(paths["final-report"].read_text(encoding="utf-8"))
    phase_a = json.loads(paths["phase-a-report"].read_text(encoding="utf-8"))
    phase_b = json.loads(paths["phase-b-report"].read_text(encoding="utf-8"))
    target = phase_a if phase == "phase-a" else phase_b
    target["authorization"][field] = "f" * (40 if field == "revision" else 64)
    target["contract_digest"] = p.contract_digest(
        phase, target["effective"], target["identity_manifest"],
        phase_spec=p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")[phase],
        historical_evidence=target["attempt_1_historical_evidence"],
        historical_attempt2_evidence=target["attempt_2_historical_evidence"],
        authorization=target["authorization"],
        phase_a_admission_lineage=target.get("phase_a_admission_lineage"))
    target_path = paths[f"{phase}-report"]
    target_path.write_text(json.dumps(target, sort_keys=True), encoding="utf-8")
    if phase == "phase-a":
        phase_a = target
    else:
        phase_b = target
    final["phase_a"] = phase_a
    final["phase_b"] = phase_b
    final["phase_a_report_sha256"] = p.file_sha256(paths["phase-a-report"])
    final["phase_b_report_sha256"] = p.file_sha256(paths["phase-b-report"])
    final["phase_a_contract_digest"] = phase_a["contract_digest"]
    final["phase_b_contract_digest"] = phase_b["contract_digest"]
    final["contract_digest"] = hashlib.sha256(json.dumps({
        "phase_a": phase_a["contract_digest"], "phase_b": phase_b["contract_digest"],
        "phase_authorizations": {"phase-a": phase_a["authorization"], "phase-b": phase_b["authorization"]}},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(p.PilotError):
        p.validate_final_attempt3_report(final, phase_a, phase_b,
                                         **_trusted_final_kwargs(report_b, report_a))


@pytest.mark.parametrize("phase,root", [
    ("phase-a", "identity"), ("phase-b", "identity"),
    ("phase-a", "attempt-1"), ("phase-b", "attempt-1"),
    ("phase-a", "attempt-2"), ("phase-b", "attempt-2"),
    ("phase-b", "lineage"),
    ("phase-b", "identity+attempt-1"), ("phase-b", "attempt-1+attempt-2"),
])
def test_r10_final_validator_rejects_rebound_snapshot_roots(tmp_path, monkeypatch, phase, root):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    paths, report_b, spec_b, report_a, _ = _write_valid_attempt3_phase_b(tmp_path, p)
    p._write_success_evidence(
        "phase-b", report_b, phase_spec=spec_b,
        phase_a_authorization=report_a["authorization"],
        phase_a_admission_lineage=report_b["phase_a_admission_lineage"],
        phase_b_authorization=report_b["authorization"],
        trusted_identity=report_a["identity_manifest"],
        trusted_attempt_1_historical_evidence=report_a["attempt_1_historical_evidence"],
        trusted_attempt_2_historical_evidence=report_a["attempt_2_historical_evidence"])
    final = json.loads(paths["final-report"].read_text(encoding="utf-8"))
    phase_a = json.loads(paths["phase-a-report"].read_text(encoding="utf-8"))
    phase_b = json.loads(paths["phase-b-report"].read_text(encoding="utf-8"))
    target = phase_a if phase == "phase-a" else phase_b
    if "identity" in root:
        target["identity_manifest"]["immutable"]["git_head"] = "f" * 40
    if "attempt-1" in root:
        target["attempt_1_historical_evidence"][0]["sha256"] = "0" * 64
    if "attempt-2" in root:
        target["attempt_2_historical_evidence"]["synthetic"] = "substituted"
    if root == "lineage":
        target["phase_a_admission_lineage"]["report"]["sha256"] = "0" * 64
    phase_spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")[phase]
    target["contract_digest"] = p.contract_digest(
        phase, target["effective"], target["identity_manifest"], phase_spec=phase_spec,
        historical_evidence=target["attempt_1_historical_evidence"],
        historical_attempt2_evidence=target["attempt_2_historical_evidence"],
        authorization=target["authorization"],
        phase_a_admission_lineage=target.get("phase_a_admission_lineage"))
    paths[f"{phase}-report"].write_text(json.dumps(target, sort_keys=True), encoding="utf-8")
    if phase == "phase-a":
        phase_a = target
    else:
        phase_b = target
    final.update({
        "phase_a": phase_a, "phase_b": phase_b,
        "phase_a_report_sha256": p.file_sha256(paths["phase-a-report"]),
        "phase_b_report_sha256": p.file_sha256(paths["phase-b-report"]),
        "phase_a_contract_digest": phase_a["contract_digest"],
        "phase_b_contract_digest": phase_b["contract_digest"],
    })
    final["contract_digest"] = hashlib.sha256(json.dumps({
        "phase_a": phase_a["contract_digest"], "phase_b": phase_b["contract_digest"],
        "phase_authorizations": {"phase-a": phase_a["authorization"], "phase-b": phase_b["authorization"]}},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(p.PilotError):
        p.validate_final_attempt3_report(final, phase_a, phase_b,
                                         **_trusted_final_kwargs(report_b, report_a))


@pytest.mark.parametrize("kind,field", [
    ("phase-a", "phase"), ("phase-a", "status"), ("phase-a", "attempt"),
    ("phase-a", "namespace"), ("phase-a", "report_path"),
    ("phase-a", "report_sha256"), ("phase-a", "output_path"),
    ("phase-a", "timestamp"), ("phase-a", "exit_code"),
    ("phase-a", "contract_digest"), ("phase-a", "resume_source"),
    ("phase-a", "resume_source_file_sha256"),
    ("phase-a", "resume_source_canonical_tensor_digest_v1"),
    ("phase-b", "phase"), ("phase-b", "status"), ("phase-b", "attempt"),
    ("phase-b", "namespace"), ("phase-b", "report_path"),
    ("phase-b", "report_sha256"), ("phase-b", "output_path"),
    ("phase-b", "timestamp"), ("phase-b", "exit_code"), ("phase-b", "contract_digest"),
    ("final", "phase"), ("final", "status"), ("final", "attempt"),
    ("final", "namespace"), ("final", "report_path"),
    ("final", "report_sha256"), ("final", "output_path"),
    ("final", "timestamp"), ("final", "exit_code"), ("final", "contract_digest"),
])
def test_r10_marker_valid_type_wrong_value_matrix(tmp_path, monkeypatch, kind, field):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    if kind == "phase-a":
        paths, report, _ = _write_valid_attempt3_phase_a(tmp_path, p)
        spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")[kind]
        marker_path = paths["phase-a-ok"]
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        final = False
    elif kind == "phase-b":
        paths, report, spec, _, _ = _write_valid_attempt3_phase_b(tmp_path, p)
        marker = p._marker_fields("phase-b", "ok", paths["phase-b-report"], report, spec)
        marker_path = paths["phase-b-ok"]
        final = False
    else:
        paths, report, spec, phase_a_report, _ = _write_valid_attempt3_phase_b(tmp_path, p)
        p._write_success_evidence(
            "phase-b", report, phase_spec=spec,
            phase_a_authorization=phase_a_report["authorization"],
            phase_a_admission_lineage=report["phase_a_admission_lineage"],
            phase_b_authorization=report["authorization"],
            trusted_identity=phase_a_report["identity_manifest"],
            trusted_attempt_1_historical_evidence=phase_a_report["attempt_1_historical_evidence"],
            trusted_attempt_2_historical_evidence=phase_a_report["attempt_2_historical_evidence"])
        marker_path = paths["final-ok"]
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        report = json.loads(paths["final-report"].read_text(encoding="utf-8"))
        final = True
    mutated = json.loads(json.dumps(marker))
    if field == "phase":
        mutated[field] = "phase-b" if kind == "phase-a" else "phase-a"
    elif field == "status":
        mutated[field] = "fail"
    elif field == "attempt":
        mutated[field] = 4
    elif field == "namespace":
        mutated[field] = "substituted"
    elif field in {"report_path", "output_path", "resume_source"}:
        mutated[field] = str(tmp_path / "substituted")
    elif field in {"report_sha256", "contract_digest", "resume_source_file_sha256",
                   "resume_source_canonical_tensor_digest_v1"}:
        mutated[field] = "0" * 64
    elif field == "timestamp":
        mutated[field] = 0.0
    else:
        mutated[field] = 1
    with pytest.raises(p.PilotError):
        p._validate_attempt3_marker(
            mutated, phase="phase-b" if kind in {"phase-b", "final"} else "phase-a",
            phase_spec=spec, report=report, marker_path=marker_path, final=final)


@pytest.mark.parametrize("kind", ["phase-a", "phase-b", "final"])
def test_r11_marker_every_field_rejects_independent_wrong_json_type(tmp_path, monkeypatch, kind):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    if kind == "phase-a":
        paths, report, _ = _write_valid_attempt3_phase_a(tmp_path, p)
        spec = p.attempt3_phase_specs(repo_root=tmp_path, workspace=tmp_path / "workspace")[kind]
        marker_path = paths["phase-a-ok"]
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        final = False
        phase = "phase-a"
    elif kind == "phase-b":
        paths, report, spec, _, _ = _write_valid_attempt3_phase_b(tmp_path, p)
        marker = p._marker_fields("phase-b", "ok", paths["phase-b-report"], report, spec)
        marker_path = paths["phase-b-ok"]
        final = False
        phase = "phase-b"
    else:
        paths, report, spec, phase_a_report, _ = _write_valid_attempt3_phase_b(tmp_path, p)
        p._write_success_evidence(
            "phase-b", report, phase_spec=spec,
            phase_a_authorization=phase_a_report["authorization"],
            phase_a_admission_lineage=report["phase_a_admission_lineage"],
            phase_b_authorization=report["authorization"],
            trusted_identity=phase_a_report["identity_manifest"],
            trusted_attempt_1_historical_evidence=phase_a_report["attempt_1_historical_evidence"],
            trusted_attempt_2_historical_evidence=phase_a_report["attempt_2_historical_evidence"])
        marker_path = paths["final-ok"]
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        report = json.loads(paths["final-report"].read_text(encoding="utf-8"))
        final = True
        phase = "phase-b"
    expected = set(p._ATTEMPT3_MARKER_PHASE_A_KEYS if kind == "phase-a" else p._ATTEMPT3_MARKER_BASE_KEYS)
    for field in sorted(expected):
        mutated = json.loads(json.dumps(marker))
        if field in {"attempt", "exit_code"}:
            mutated[field] = "3"
        elif field == "timestamp":
            mutated[field] = "now"
        elif field in {"report_sha256", "contract_digest", "resume_source_file_sha256",
                       "resume_source_canonical_tensor_digest_v1"}:
            mutated[field] = 0
        else:
            mutated[field] = None
        with pytest.raises(p.PilotError):
            p._validate_attempt3_marker(mutated, phase=phase, phase_spec=spec,
                                        report=report, marker_path=marker_path, final=final)


def test_r11_loaded_mlx_negative_schema_and_payload_matrix_is_fail_closed(tmp_path):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    p = load_pilot()
    model = nn.Linear(2, 2)
    expected = model.parameters()
    valid = tmp_path / "valid.safetensors"
    mx.save_safetensors(str(valid), expected)
    original = json.loads(valid.read_bytes()[8:8 + int.from_bytes(valid.read_bytes()[:8], "little")])
    payload_start = 8 + int.from_bytes(valid.read_bytes()[:8], "little")
    payload = valid.read_bytes()[payload_start:]

    cases = ["missing-name", "extra-name", "duplicate-name", "bad-dtype", "bad-shape",
             "bad-layout", "bad-payload"]
    for case in cases:
        path = tmp_path / f"{case}.safetensors"
        header = json.loads(json.dumps(original))
        raw_header = None
        payload_case = payload
        if case == "missing-name":
            del header["bias"]
        elif case == "extra-name":
            header["extra"] = {"dtype": "F32", "shape": [1],
                                "data_offsets": [len(payload), len(payload) + 4]}
            payload_case = payload + b"\\x00" * 4
        elif case == "duplicate-name":
            item = json.dumps(original["weight"], separators=(",", ":"))
            raw_header = ("{" + json.dumps("__metadata__") + ":null," +
                          json.dumps("weight") + ":" + item + "," +
                          json.dumps("weight") + ":" + item + "," +
                          json.dumps("bias") + ":" + json.dumps(original["bias"], separators=(",", ":")) + "}").encode()
        elif case == "bad-dtype":
            header["weight"]["dtype"] = "F16"
        elif case == "bad-shape":
            header["weight"]["shape"] = [1, 4]
        elif case == "bad-layout":
            header["weight"]["data_offsets"] = [1, 17]
        else:
            payload_case = bytes([payload[0] ^ 1]) + payload[1:]
        if raw_header is None:
            raw_header = json.dumps(header, separators=(",", ":")).encode()
        path.write_bytes(len(raw_header).to_bytes(8, "little") + raw_header + payload_case)
        with pytest.raises(Exception):
            loaded = mx.load(str(path))
            mx.eval(*loaded.values())
            target = nn.Linear(2, 2)
            target.load_weights(str(path), strict=True)
            assert set(loaded) == set(expected)
            for name in expected:
                assert loaded[name].tolist() == expected[name].tolist()
            p.canonical_tensor_digest(path)


def test_r11_attempt2_coordinated_mutation_reaches_path_hash_and_target_guards(tmp_path, monkeypatch):
    p = load_pilot()
    fixture, expected_manifest, _ = _install_r6_attempt2_fixture(tmp_path, monkeypatch, p)
    immutable_files = fixture["files_expected"]
    target_index = 6
    original_path, original_size, original_sha = fixture["files_expected"][target_index]
    substituted_path = original_path + ".substituted"
    target = p._attempt2_expected_path(original_path, tmp_path)
    substituted = p._attempt2_expected_path(substituted_path, tmp_path)
    substituted.parent.mkdir(parents=True, exist_ok=True)
    mutated_payload = bytes([target.read_bytes()[0] ^ 1]) + target.read_bytes()[1:]
    substituted.write_bytes(mutated_payload)
    mutated_sha = hashlib.sha256(mutated_payload).hexdigest()
    files = list(fixture["files_expected"])
    files[target_index] = (substituted_path, len(mutated_payload), mutated_sha)
    expected = json.loads(json.dumps(expected_manifest))
    expected["files"][target_index] = {"path": substituted_path, "size": len(mutated_payload), "sha256": mutated_sha}
    manifest_path = p._historical_path(fixture["manifest_expected"][0], tmp_path)
    manifest_payload = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    manifest_path.write_bytes(manifest_payload)
    fixture["manifest_expected"] = (fixture["manifest_expected"][0], len(manifest_payload), hashlib.sha256(manifest_payload).hexdigest())
    fixture["expected_manifest"] = expected
    fixture["files_expected"] = tuple(files)
    calls = []
    original_snapshot = p._verify_historical_snapshot
    def traced_snapshot(path, size, sha, label, **kwargs):
        calls.append(("snapshot", str(path), size, sha, label))
        return original_snapshot(path, size, sha, label, **kwargs)
    monkeypatch.setattr(p, "_verify_historical_snapshot", traced_snapshot)
    p._verify_historical_evidence_snapshot(repo_root=tmp_path, **fixture)
    assert any(call[1] == str(substituted) and call[2:] == (len(mutated_payload), mutated_sha, "historical runtime evidence")
               for call in calls)
    with pytest.raises(p.PilotError, match="target bindings"):
        p._verify_historical_evidence_snapshot(repo_root=tmp_path, immutable_files_expected=immutable_files, **fixture)
    assert target.read_bytes() != mutated_payload


def test_r11_public_attempt2_guard_rejects_rebound_target_root(tmp_path, monkeypatch):
    p = load_pilot()
    rebound = list(p._ATTEMPT2_VERIFIER_FILES)
    rebound[6] = (rebound[6][0] + ".substituted", rebound[6][1], rebound[6][2])
    monkeypatch.setattr(p, "_ATTEMPT2_VERIFIER_FILES", tuple(rebound))
    with pytest.raises(p.PilotError, match="target bindings"):
        p.verify_attempt2_historical_evidence(repo_root=tmp_path)


def test_r12_weakened_attempt2_verifier_accepts_coordinated_substitution_but_public_guard_rejects(
        tmp_path, monkeypatch):
    p = load_pilot()
    fixture, expected_manifest, _ = _install_r6_attempt2_fixture(tmp_path, monkeypatch, p)
    immutable_files = fixture["files_expected"]
    target_index = 6
    original_path, _, _ = immutable_files[target_index]
    substituted_path = original_path + ".substituted"
    target = p._attempt2_expected_path(original_path, tmp_path)
    substituted = p._attempt2_expected_path(substituted_path, tmp_path)
    substituted.parent.mkdir(parents=True, exist_ok=True)
    mutated_payload = bytes([target.read_bytes()[0] ^ 1]) + target.read_bytes()[1:]
    substituted.write_bytes(mutated_payload)
    mutated_sha = hashlib.sha256(mutated_payload).hexdigest()
    files = list(immutable_files)
    files[target_index] = (substituted_path, len(mutated_payload), mutated_sha)
    expected = json.loads(json.dumps(expected_manifest))
    expected["files"][target_index] = {
        "path": substituted_path, "size": len(mutated_payload), "sha256": mutated_sha}
    manifest_path = p._historical_path(fixture["manifest_expected"][0], tmp_path)
    manifest_payload = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    manifest_path.write_bytes(manifest_payload)
    fixture["manifest_expected"] = (
        fixture["manifest_expected"][0], len(manifest_payload), hashlib.sha256(manifest_payload).hexdigest())
    fixture["expected_manifest"] = expected
    fixture["files_expected"] = tuple(files)

    original_snapshot = p._verify_historical_snapshot
    def weakened(path, size, sha, label, **kwargs):
        if label == "historical runtime evidence":
            if kwargs.get("parse_json"):
                return original_snapshot(path, size, sha, label, **kwargs)
            return {"path": str(path), "size": size, "sha256": sha}, None
        return original_snapshot(path, size, sha, label, **kwargs)

    monkeypatch.setattr(p, "_verify_historical_snapshot", weakened)
    accepted = p._verify_historical_evidence_snapshot(repo_root=tmp_path, **fixture)
    assert accepted["files"][target_index] == {
        "path": str(substituted), "size": len(mutated_payload), "sha256": mutated_sha}
    assert accepted["report_identity"]["attempt"] == 2
    assert accepted["report_identity"]["namespace"] == "ds4-segmented-pilot-attempt-2"

    rebound = list(p._ATTEMPT2_VERIFIER_FILES)
    rebound[target_index] = files[target_index]
    monkeypatch.setattr(p, "_ATTEMPT2_VERIFIER_FILES", tuple(rebound))
    with pytest.raises(p.PilotError, match=r"^attempt-2 runtime target bindings are not immutable$") as exc_info:
        p.verify_attempt2_historical_evidence(repo_root=tmp_path)
    assert "missing report" not in str(exc_info.value)


def _run_r12_phase_publication_seam(tmp_path, monkeypatch, phase_name, failure_number):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    workspace = tmp_path / "workspace"
    phase_specs = p.attempt3_phase_specs(repo_root=tmp_path, workspace=workspace)
    if phase_name == "phase-b":
        _write_valid_attempt3_phase_a(tmp_path, p)
    phase = dict(phase_specs[phase_name])
    paths = p.attempt3_namespace(repo_root=tmp_path, workspace=workspace)
    phase.update({"iters": 0, "timeout": 60, "adapter_path": str(paths[f"{phase_name}-output"]),
                  "namespace_paths": {key: str(value) for key, value in paths.items()},
                  "report": str(paths[f"{phase_name}-report"]), "log_path": str(paths[f"{phase_name}-log"])})
    authorization = {"revision": "0" * 40, "canonical_command_sha256": "0" * 64,
                     "pilot_source_sha256": "0" * 64, "catalog_source_sha256": "0" * 64,
                     "protected_files_manifest_sha256": "0" * 64,
                     "attempt2_runtime_manifest_sha256": "0" * 64}
    authorization_json = json.dumps(authorization, sort_keys=True, separators=(",", ":"))
    args = types.SimpleNamespace(
        phase=phase_name, attempt=3, config=None, authorization_json=authorization_json,
        phase_a_authorization_json=(authorization_json if phase_name == "phase-b" else None),
        log_fd=None, log_path=paths[f"{phase_name}-log"])
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(workspace))
    monkeypatch.setattr(p, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(p, "_phase_spec", lambda *_args, **_kwargs: phase)
    monkeypatch.setattr(p, "attempt3_phase_specs", lambda **_kwargs: phase_specs)
    monkeypatch.setattr(p, "_check_attempt_gates", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(p, "_install_timeout_watchdog", lambda _seconds: types.SimpleNamespace(cancel=lambda: None))
    monkeypatch.setattr(p, "_read_config", lambda _path: {})
    monkeypatch.setattr(p, "validate_pins", lambda *_args: {"command": []})
    monkeypatch.setattr(p, "verify_attempt1_historical_evidence", lambda **_kwargs: [])
    monkeypatch.setattr(p, "verify_attempt2_historical_evidence", lambda **_kwargs: {})
    monkeypatch.setattr(p, "_acquire_ft_lock", lambda _workspace: None)
    monkeypatch.setattr(p, "_release_ft_lock", lambda: None)
    monkeypatch.setattr(p, "_runtime_preflight", lambda *_args: {"immutable": {"identity": "synthetic"}})
    monkeypatch.setattr(p, "compare_immutable_identity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(p, "contract_digest", lambda *_args, **_kwargs: "c" * 64)
    monkeypatch.setattr(p, "_execute_training", lambda *_args, **_kwargs: None)
    info = {"path": "synthetic", "file_sha256": "a" * 64, "canonical_tensor_digest_v1": "b" * 64,
            "tensors": [{"name": "x", "dtype": "U8", "shape": [1], "nbytes": 1}]}
    monkeypatch.setattr(p, "_validate_artifacts", lambda *_args, **_kwargs: {
        "start": info, "checkpoints": [info], "final": info})
    monkeypatch.setattr(p, "canonical_tensor_digest", lambda *_args, **_kwargs: {
        "path": "synthetic", "file_sha256": "a" * 64,
        "canonical_tensor_digest_v1": "b" * 64,
        "tensors": info["tensors"]})
    monkeypatch.setattr(p, "validate_step_evidence", lambda *_args: None)
    monkeypatch.setattr(p, "_expected_validation_steps", lambda _phase: [])
    monkeypatch.setattr(p, "_StepObservingProvider", lambda *_args, **_kwargs: types.SimpleNamespace(records=[]))
    monkeypatch.setattr(p, "_PilotTrainingCallback", lambda _phase: types.SimpleNamespace(records=[], validation_records=[]))
    monkeypatch.setattr(p, "validate_canonical_attempt3_report", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(p, "validate_phase_b_dependency", lambda *_args, **_kwargs: None)
    def aggregate_for_seam(phase_a, phase_b, strict_lineage=False):
        final = dict(phase_b)
        final.update({"phase": "phase-b", "status": "ok", "attempt": 3,
                      "contract_digest": "c" * 64, "global_progression": [0, 1, 2, 3]})
        return final
    monkeypatch.setattr(p, "aggregate_reports", aggregate_for_seam)
    monkeypatch.setattr(p, "validate_final_attempt3_report", lambda *_args, **_kwargs: None)
    original = p.atomic_write_json
    calls = 0
    def fail_at_seam(path, value):
        nonlocal calls
        calls += 1
        if calls == failure_number:
            raise OSError(f"r12 seam {failure_number}")
        return original(path, value)
    monkeypatch.setattr(p, "atomic_write_json", fail_at_seam)
    monkeypatch.setattr(p, "_LOCK_OWNED_PATH", None)
    monkeypatch.setattr(p, "_LOCK_OWNED_PARTIAL", False)
    result = p.run_phase(args, api={"tree_flatten": lambda value: value, "mx": object()})
    return p, paths, result


@pytest.mark.parametrize("phase_name,seams", [("phase-a", 2), ("phase-b", 4)])
def test_r12_run_phase_publishes_exact_failure_reports_and_markers_at_every_seam(
        tmp_path, monkeypatch, phase_name, seams):
    for failure_number in range(1, seams + 1):
        with monkeypatch.context() as scoped:
            p, paths, result = _run_r12_phase_publication_seam(
                tmp_path / f"case-{failure_number}", scoped, phase_name, failure_number)
        assert result["status"] == "fail"
        assert result["phase"] == phase_name
        assert result["attempt"] == 3
        assert result["namespace"] == p.ATTEMPT3_NAMESPACE
        assert result["exit_code"] == 1
        assert result["error"] == f"report/marker write failed: r12 seam {failure_number}"
        phase_report_path = paths[f"{phase_name}-report"]
        final_report_path = paths["final-report"]
        phase_report = json.loads(phase_report_path.read_text(encoding="utf-8"))
        final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
        assert phase_report == result
        assert final_report == dict(phase_report, phase_report=str(phase_report_path))
        assert final_report["phase_report"] == str(phase_report_path)
        assert final_report["status"] == "fail"
        assert final_report["phase"] == phase_name
        assert final_report["attempt"] == 3
        assert final_report["namespace"] == p.ATTEMPT3_NAMESPACE
        assert final_report["exit_code"] == 1
        assert final_report["contract_digest"] == phase_report["contract_digest"]
        for marker_key, report_path in ((f"{phase_name}-fail", phase_report_path),
                                        ("final-fail", final_report_path)):
            marker_path = paths[marker_key]
            assert marker_path.exists()
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            assert marker["status"] == "fail"
            assert marker["phase"] == phase_name
            assert marker["attempt"] == 3
            assert marker["namespace"] == p.ATTEMPT3_NAMESPACE
            assert marker["exit_code"] == 1
            assert marker["report_path"] == str(report_path)
            assert marker["report_sha256"] == p.file_sha256(report_path)
            assert marker["contract_digest"] == final_report["contract_digest"]
        assert not paths["phase-a-ok"].exists()
        assert not paths["phase-b-ok"].exists()
        assert not paths["final-ok"].exists()


def test_r10_run_phase_write_seam_returns_bound_attempt3_failure_evidence(tmp_path, monkeypatch):
    p = load_pilot()
    install_tmp_fs_guard(monkeypatch, tmp_path)
    workspace = tmp_path / "workspace"
    paths = p.attempt3_namespace(repo_root=tmp_path, workspace=workspace)
    phase = p.attempt3_phase_specs(repo_root=tmp_path, workspace=workspace)["phase-a"]
    phase = dict(phase)
    phase["iters"] = 0
    phase["timeout"] = 60
    phase["adapter_path"] = str(paths["phase-a-output"])
    phase["namespace_paths"] = {key: str(value) for key, value in paths.items()}
    phase["report"] = str(paths["phase-a-report"])
    phase["log_path"] = str(paths["phase-a-log"])
    authorization = {
        "revision": "0" * 40,
        "canonical_command_sha256": "0" * 64,
        "pilot_source_sha256": "0" * 64,
        "catalog_source_sha256": "0" * 64,
        "protected_files_manifest_sha256": "0" * 64,
        "attempt2_runtime_manifest_sha256": "0" * 64,
    }
    args = types.SimpleNamespace(
        phase="phase-a", attempt=3, config=None,
        authorization_json=json.dumps(authorization, sort_keys=True, separators=(",", ":")),
        phase_a_authorization_json=None, log_fd=None)
    monkeypatch.setattr(p, "PILOT_WORKSPACE", str(workspace))
    monkeypatch.setattr(p, "_phase_spec", lambda *_args: phase)
    monkeypatch.setattr(p, "_check_attempt_gates", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(p, "_install_timeout_watchdog", lambda _seconds: types.SimpleNamespace(cancel=lambda: None))
    monkeypatch.setattr(p, "_read_config", lambda _path: {})
    monkeypatch.setattr(p, "validate_pins", lambda *_args: {"command": []})
    monkeypatch.setattr(p, "verify_attempt1_historical_evidence", lambda: [])
    monkeypatch.setattr(p, "verify_attempt2_historical_evidence", lambda: {})
    monkeypatch.setattr(p, "_acquire_ft_lock", lambda _workspace: None)
    monkeypatch.setattr(p, "_release_ft_lock", lambda: None)
    monkeypatch.setattr(p, "_runtime_preflight", lambda *_args: {"immutable": {"identity": "synthetic"}})
    monkeypatch.setattr(p, "contract_digest", lambda *_args, **_kwargs: "c" * 64)
    monkeypatch.setattr(p, "_execute_training", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(p, "_validate_artifacts", lambda *_args, **_kwargs: {
        "start": {"path": "start", "file_sha256": "a" * 64,
                   "canonical_tensor_digest_v1": "b" * 64},
        "checkpoints": [{"path": "checkpoint", "file_sha256": "a" * 64,
                          "canonical_tensor_digest_v1": "b" * 64}],
        "final": {"path": "final", "file_sha256": "a" * 64,
                   "canonical_tensor_digest_v1": "b" * 64},
    })
    monkeypatch.setattr(p, "validate_step_evidence", lambda *_args: None)
    monkeypatch.setattr(p, "_expected_validation_steps", lambda _phase: [])
    monkeypatch.setattr(p, "_StepObservingProvider", lambda *_args, **_kwargs: types.SimpleNamespace(records=[]))
    monkeypatch.setattr(p, "_PilotTrainingCallback", lambda _phase: types.SimpleNamespace(records=[], validation_records=[]))
    monkeypatch.setattr(p, "_write_success_evidence", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("r10 seam")))
    monkeypatch.setattr(p, "_LOCK_OWNED_PATH", None)
    monkeypatch.setattr(p, "_LOCK_OWNED_PARTIAL", False)
    result = p.run_phase(args, api={"tree_flatten": lambda value: value, "mx": object()})
    assert result["status"] == "fail"
    assert result["attempt"] == 3
    assert result["phase"] == "phase-a"
    assert "report/marker write failed" in result["error"]
    assert result["contract_digest"] == "c" * 64
    fail_marker = paths["phase-a-fail"]
    assert fail_marker.is_file()
    marker = json.loads(fail_marker.read_text(encoding="utf-8"))
    assert marker["phase"] == "phase-a"
    assert marker["attempt"] == 3
    assert marker["status"] == "fail"
    assert marker["report_path"] == str(paths["phase-a-report"])
    assert marker["report_sha256"] == p.file_sha256(paths["phase-a-report"])
    assert not paths["phase-a-ok"].exists()
