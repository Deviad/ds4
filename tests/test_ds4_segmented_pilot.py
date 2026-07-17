from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import pathlib
import shlex
import sys
import textwrap

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ds4_segmented_pilot.py"


def load_pilot():
    spec = importlib.util.spec_from_file_location("ds4_segmented_pilot_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def write_safetensors(path: pathlib.Path, tensors: dict[str, tuple[str, list[int], bytes]], metadata=None):
    offset = 0
    header = {}
    payload = bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        payload.extend(raw)
        offset += len(raw)
    if metadata is not None:
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


def test_catalog_commands_are_explicit_and_non_default():
    source = (ROOT / "scripts" / "finetune_ds4.py").read_text(encoding="utf-8")
    assert "ds4-segmented-pilot-phase-a" in source
    assert "ds4-segmented-pilot-phase-b" in source
    p = load_pilot()
    import scripts.finetune_ds4 as finetune
    assert "ds4-segmented-pilot-phase-a" in finetune.MLX_STEPS
    assert "ds4-segmented-pilot-phase-b" in finetune.MLX_STEPS
    assert "ds4-segmented-pilot-phase-a" not in finetune.DEFAULT_BACKEND_STEPS["local-mlx"]
    assert "ds4-segmented-pilot-phase-b" not in finetune.DEFAULT_BACKEND_STEPS["local-mlx"]


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


def test_catalog_rendering_is_byte_exact_and_defaults_are_guarded():
    import scripts.finetune_ds4 as finetune
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
    catalog = finetune.command_catalog(Args())
    expected_a = textwrap.dedent("""\
        bash -lc 'set -o pipefail
        cd "/Users/spotted/projects/ds4-finetuning"
        unset SSLKEYLOGFILE
        PYTHONUNBUFFERED=1 PYTHONPATH="/Users/spotted/projects/ds4-finetuning/python-envs/mlx/src:/Users/spotted/projects/ds4-finetuning/vendor/mlx-lm" \\
        "/Volumes/Data NVME/mlx-ft/ds4/.venv/bin/python" \\
        "/Users/spotted/projects/ds4-finetuning/scripts/ds4_segmented_pilot.py" \\
          --phase phase-a \\
          --model "/Volumes/Data NVME/mlx-ft/ds4/model-4bit" \\
          --data "/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke" \\
          --adapter-path "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-a" \\
          --config "/Volumes/Data NVME/mlx-ft/ds4/lora-config.json" \\
          --train --fine-tune-type lora --num-layers 16 \\
          --iters 2 --batch-size 1 --learning-rate 1e-5 \\
          --max-seq-length 4096 --mask-prompt --grad-checkpoint \\
          --grad-accumulation-steps 1 --seed 0 --optimizer adam \\
          --val-batches 25 --steps-per-report 1 --steps-per-eval 2 \\
          --save-every 1 --segment-size 1 \\
          2>&1 | tee "/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5/phase-a-log.txt"'
        """).rstrip("\n")
    expected_b = expected_a.replace("phase-a", "phase-b").replace("phase-a-log", "phase-b-log").replace("iters 2", "iters 1").replace("steps-per-eval 2", "steps-per-eval 1")
    resume = ' \\\n  --resume-adapter-file "/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-a/0000002_adapters.safetensors"'
    expected_b = expected_b.replace('  --config "/Volumes/Data NVME/mlx-ft/ds4/lora-config.json" \\\n', '  --config "/Volumes/Data NVME/mlx-ft/ds4/lora-config.json"' + resume + ' \\\n')
    assert catalog["ds4-segmented-pilot-phase-a"] == [expected_a]
    assert catalog["ds4-segmented-pilot-phase-b"] == [expected_b]
    assert catalog["smoke-train"] == ["cd /tmp/mlx && unset SSLKEYLOGFILE && . /tmp/mlx/.venv/bin/activate && mlx_lm.lora --config /tmp/mlx/lora-config.json --model /tmp/mlx/model-4bit --train --data /tmp/data/mlx-4096 --adapter-path /tmp/mlx/adapters-smoke --fine-tune-type lora --iters 20 --batch-size 1 --learning-rate 1e-5 --max-seq-length 4096 --mask-prompt --grad-checkpoint"]
    assert catalog["continue-train"] == ["cd /tmp/mlx && unset SSLKEYLOGFILE && . /tmp/mlx/.venv/bin/activate && mlx_lm.lora --config /tmp/mlx/lora-config.json --model /tmp/mlx/model-4bit --train --data /tmp/data/mlx-4096 --adapter-path /tmp/mlx/adapters --resume-adapter-file /tmp/mlx/adapters/adapters.safetensors --fine-tune-type lora --iters 15000 --batch-size 1 --learning-rate 5e-6 --max-seq-length 4096 --mask-prompt --grad-checkpoint --steps-per-report 10 --steps-per-eval 200"]
    assert "ds4-segmented-pilot-phase-a" not in finetune.DEFAULT_BACKEND_STEPS["local-mlx"]
    assert "ds4-segmented-pilot-phase-b" not in finetune.DEFAULT_BACKEND_STEPS["local-mlx"]


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
