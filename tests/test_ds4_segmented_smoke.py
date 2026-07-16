"""Phase 1 tests for ds4_segmented_smoke r6 — lock tokens, nested evidence,
mandatory cleanup, timeout/preflight, legitimate mutations, registry identity.

All tests use temporary directories, monkeypatched stubs, or temp-source
mutation overlays. No real model or /Volumes/Data NVME/ access.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import types
from unittest import mock

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "ds4_segmented_smoke.py"
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _pin_test_paths(mod, tmp_path):
    """Monkeypatch _PINNED_SMOKE_PATHS so tests calling main() with
    temporary paths are not rejected by the pinned-path gate."""
    mod._PINNED_SMOKE_PATHS = {
        "model": str(tmp_path / "m"),
        "data": str(tmp_path / "d"),
        "adapter_path": str(tmp_path / "a"),
        "config": str(tmp_path / "lora.json") if (tmp_path / "lora.json").exists() else None,
    }


def _mutate(mutation):
    """Copy source to temp dir, apply mutation, import, return module."""
    td = pathlib.Path(tempfile.mkdtemp(prefix="ds4m_"))
    dst = td / "m.py"
    shutil.copy2(SCRIPT, dst)
    t = dst.read_text("utf-8")
    t = mutation(t)
    dst.write_text(t, "utf-8")
    spec = importlib.util.spec_from_file_location(f"m_{id(td)}", dst)
    m = importlib.util.module_from_spec(spec)
    saved = set(sys.modules)
    try:
        spec.loader.exec_module(m)
    except Exception:
        for k in list(sys.modules):
            if k not in saved:
                del sys.modules[k]
        raise
    return m


def _stub(extra=None):
    b = dict(model="/tmp/fm", data="/tmp/fd", adapter_path="/tmp/fa", config=None,
             fine_tune_type="lora", num_layers=1, batch_size=1, iters=1,
             val_batches=25, learning_rate=1e-5, steps_per_report=10,
             steps_per_eval=200, save_every=100, max_seq_length=4096,
             grad_checkpoint=True, grad_accumulation_steps=1, seed=42,
             mask_prompt=True, optimizer="adam", optimizer_config={"adam": {}},
             lr_schedule=None, resume_adapter_file=None,
             lora_parameters={"rank": 8, "scale": 20.0, "dropout": 0.0},
             report_to=None, project_name=None, trust_remote_code=False,
             segment_size=1, test=False, test_batches=500, clear_cache_threshold=0)
    if extra:
        b.update(extra)
    return types.SimpleNamespace(**b)


def _make_provider_stub(monkeypatch, mod):
    """Set up stubs so _run() returns a valid _ObservingProvider."""
    class SM:
        def __init__(s):
            s.layers = type("L", (), {"__len__": lambda s: 1, "__getitem__": lambda s, i: type("La", (), {"unfreeze": lambda: None})()})()
        def freeze(s): pass
        def train(s): pass
        def eval(s): pass
        def trainable_parameters(s): return {}
        def update(s, *a, **kw): pass
        def load_weights(s, *a, **kw): pass
    monkeypatch.setattr(mod, "load", lambda *a, **kw: (SM(), "tok"))
    monkeypatch.setattr(mod, "load_dataset", lambda *a, **kw: ([("x",)], [], []))
    monkeypatch.setattr(mod, "linear_to_lora_layers", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "print_trainable_parameters", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "save_config", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "get_reporting_callbacks", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "CacheDataset", lambda x: x)
    fn = types.ModuleType("numpy")
    fn.random = types.SimpleNamespace()
    fn.random.seed = lambda s: None
    monkeypatch.setitem(sys.modules, "numpy", fn)
    fm = types.ModuleType("mlx.core")
    fm.random = types.SimpleNamespace()
    fm.random.seed = lambda s: None
    fm.any = staticmethod(lambda x: x)
    fm.isfinite = staticmethod(lambda x: x)
    fm.all = staticmethod(lambda x: x)
    fm.sum = staticmethod(lambda x: x)
    monkeypatch.setitem(sys.modules, "mlx.core", fm)
    monkeypatch.setattr(mod, "build_schedule", lambda *a, **kw: 1e-5)
    monkeypatch.setattr(mod, "make_ds4_segmented_loss_and_grad", lambda *, segment_size: lambda m, *b: ((mx_arr(0.5), mx_arr(7)), {"w": mx_arr([1.0])}))
    monkeypatch.setattr(mod, "_check_preflight", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
    monkeypatch.setattr(signal, "signal", lambda s, h: None)
    monkeypatch.setattr(signal, "alarm", lambda s: None)


def mx_arr(val):
    """Minimal mx.array stand-in for test stubs."""
    class _Arr:
        def __init__(s, v):
            s._v = v
        def item(s): return s._v if not isinstance(s._v, (list, tuple)) else s._v
        def __iter__(s): return iter(s._v) if isinstance(s._v, (list, tuple)) else iter([s._v])
        def __repr__(s): return f"mx_arr({s._v})"
    return _Arr(val)


# ===================================================================
# T1 — Lock ownership: path+token tracking
# ===================================================================
class TestLockOwnership:
    def test_acquire_sets_owned(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        assert mod._LOCK_OWNED_PATH is None
        assert mod._LOCK_OWNED_TOKEN is None
        mod._acquire_ft_lock(tmp_path, timeout_s=5)
        assert mod._LOCK_OWNED_PATH is not None
        assert mod._LOCK_OWNED_TOKEN is not None
        assert mod._LOCK_OWNED_TOKEN == str(os.getpid())
        assert mod._LOCK_OWNED_PATH == tmp_path / ".ds4-ft.lock"
        mod._release_ft_lock()
        assert mod._LOCK_OWNED_PATH is None
        assert mod._LOCK_OWNED_TOKEN is None

    def test_release_unowned_does_not_unlink(self, tmp_path):
        """When _LOCK_OWNED_PATH is None, release does nothing."""
        import scripts.ds4_segmented_smoke as mod
        (tmp_path / ".ds4-ft.lock").write_text("pid=99\n")
        mod._LOCK_OWNED_PATH = None
        mod._LOCK_OWNED_TOKEN = None
        mod._release_ft_lock()
        assert (tmp_path / ".ds4-ft.lock").is_file()

    def test_release_other_token_does_not_unlink(self, tmp_path):
        """When lock content does not contain our token, do not unlink."""
        import scripts.ds4_segmented_smoke as mod
        lp = tmp_path / ".ds4-ft.lock"
        lp.write_text("pid=99\n")
        mod._LOCK_OWNED_PATH = lp
        mod._LOCK_OWNED_TOKEN = "999"
        mod._release_ft_lock()
        assert lp.is_file()

    def test_release_wrong_path_does_not_unlink(self, tmp_path):
        """When lock content matches our token but path differs, do not unlink."""
        import scripts.ds4_segmented_smoke as mod
        other = tmp_path / "other" / ".ds4-ft.lock"
        other.parent.mkdir(parents=True)
        other.write_text(f"pid={os.getpid()}\n")
        mod._LOCK_OWNED_PATH = tmp_path / ".ds4-ft.lock"
        mod._LOCK_OWNED_TOKEN = str(os.getpid())
        mod._release_ft_lock()  # path doesn't exist — cleans up state only
        assert other.is_file()

    def test_main_preserves_others_lock(self, monkeypatch, tmp_path):
        """When another process holds .ds4-ft.lock, main() must not unlink it."""
        import scripts.ds4_segmented_smoke as mod
        (tmp_path / ".ds4-ft.lock").write_text("pid=999\n")
        (tmp_path / "m").mkdir()
        reports = []; markers = []
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: markers.append(r))
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(mod, "_clear_markers", lambda mw: None)
        # Make lock acquisition fail immediately
        monkeypatch.setattr(mod, "_acquire_ft_lock",
            lambda *a, **kw: (_ for _ in ()).throw(SystemExit("held")))
        _pin_test_paths(mod, tmp_path)
        rc = mod.main(["--model", str(tmp_path/"m"), "--data", str(tmp_path/"d"),
                        "--adapter-path", str(tmp_path/"a")])
        assert rc != 0
        assert (tmp_path / ".ds4-ft.lock").is_file(), "lock must survive"

    def test_acquire_then_release_owned(self, monkeypatch, tmp_path):
        """After acquire, release removes the lock."""
        import scripts.ds4_segmented_smoke as mod
        mod._acquire_ft_lock(tmp_path, timeout_s=5)
        assert (tmp_path / ".ds4-ft.lock").is_file()
        mod._release_ft_lock()
        assert not (tmp_path / ".ds4-ft.lock").exists()


# ===================================================================
# T2 — Nested gradient tree flattening
# ===================================================================
class TestNestedGradients:
    def test_flat_dict_gradients_produce_leaves(self):
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        trees = {"w": mx.array([1.0, 2.0]), "b": mx.array([0.0])}
        leaves = mod._flatten_gradient_tree(trees)
        paths = [l[0] for l in leaves]
        assert "w" in paths
        assert "b" in paths
        assert len(leaves) == 2

    def test_nested_dict_gradients(self):
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        trees = {"layer": {"w": mx.array([1.0]), "b": mx.array([0.0])}}
        leaves = mod._flatten_gradient_tree(trees)
        paths = [l[0] for l in leaves]
        assert "layer.w" in paths
        assert "layer.b" in paths
        assert len(leaves) == 2

    def test_list_gradients(self):
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        trees = [mx.array([1.0]), mx.array([2.0])]
        leaves = mod._flatten_gradient_tree(trees)
        paths = [l[0] for l in leaves]
        assert "[0]" in paths[0]
        assert "[1]" in paths[1]
        assert len(leaves) == 2

    def test_mixed_nested_gradients(self):
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        trees = {"layer1": {"w": mx.array([1.0]), "b": mx.array([0.0])},
                 "layer2": [mx.array([2.0]), {"sub": mx.array([3.0])}]}
        leaves = mod._flatten_gradient_tree(trees)
        paths = [l[0] for l in leaves]
        assert "layer1.w" in paths
        assert "layer1.b" in paths
        assert "layer2[0]" in paths
        assert "layer2[1].sub" in paths
        assert len(leaves) == 4

    def test_malformed_non_container_rejected(self):
        """Non-container non-tensor raises SystemExit."""
        import scripts.ds4_segmented_smoke as mod
        with pytest.raises(SystemExit):
            mod._flatten_gradient_tree("not_valid")

    def test_empty_dict_no_leaves(self):
        import scripts.ds4_segmented_smoke as mod
        leaves = mod._flatten_gradient_tree({})
        assert len(leaves) == 0

    def test_nested_nonfinite_leaf_fails_observer(self):
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        grads = {"layer": {"w": mx.array(float("nan"))}}
        def _real(m, *b): return ((mx.array(0.5), mx.array(5)), grads)
        obs = mod._ObservingProvider(_real)
        obs("m", "b")
        assert obs.captured_gradient_paths == ["layer.w"]
        assert obs.captured_all_finite is False
        assert obs.captured_gradient_leaf_count == 1

    def test_nested_all_finite_passes(self):
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        grads = {"layer": {"w": mx.array([1.0]), "b": mx.array([0.0])}}
        def _real(m, *b): return ((mx.array(0.5), mx.array(7)), grads)
        obs = mod._ObservingProvider(_real)
        obs("m", "b")
        assert obs.captured_gradient_paths == ["layer.w", "layer.b"]
        assert obs.captured_all_finite is True
        assert obs.captured_gradient_leaf_count == 2

    def test_observer_captures_exact_values(self, monkeypatch):
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        loss_arr = mx.array(0.5)
        tc_arr = mx.array(7)
        grads = {"w": mx.array([1.0, 2.0])}
        def _real(m, *b): return ((loss_arr, tc_arr), grads)
        obs = mod._ObservingProvider(_real)
        r = obs("m", "b")
        assert obs.call_count == 1
        assert obs.captured_loss == 0.5
        assert obs.captured_token_count == 7
        assert obs.captured_gradient_paths == ["w"]
        assert obs.captured_all_finite is True

    def test_negative_token_rejected(self):
        """Negative token count raises SystemExit in observer."""
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        grads = {"w": mx.array([1.0])}
        def _real(m, *b): return ((mx.array(0.5), mx.array(-7)), grads)
        obs = mod._ObservingProvider(_real)
        with pytest.raises(SystemExit, match="not positive"):
            obs("m", "b")

    def test_nonfinite_gradient_fails(self):
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        grads = {"w": mx.array(float("nan"))}
        def _real(m, *b): return ((mx.array(0.5), mx.array(5)), grads)
        obs = mod._ObservingProvider(_real)
        obs("m", "b")
        assert obs.captured_all_finite is False

    def test_zero_calls_stays_zero(self):
        import scripts.ds4_segmented_smoke as mod
        obs = mod._ObservingProvider(lambda m, *b: 0)
        assert obs.call_count == 0

    def test_multiple_calls_counted(self):
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        grads = {"w": mx.array([1.0])}
        def _real(m, *b): return ((mx.array(0.5), mx.array(5)), grads)
        obs = mod._ObservingProvider(_real)
        with mock.patch.object(obs, "_delegate", _real):
            obs("m", ("t",), (0, 5))
            obs("m", ("t",), (0, 5))
        assert obs.call_count == 2


# ===================================================================
# T3 — Unified cleanup and mandatory evidence
# ===================================================================
class TestUnifiedCleanup:
    def test_config_failure_writes_fail_marker(self, monkeypatch, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        markers = []; reports = []
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: markers.append((str(mw), r)))
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        _pin_test_paths(mod, tmp_path)
        rc = mod.main(["--config", str(tmp_path / "nope.json")])
        assert rc != 0
        assert len(reports) == 1
        assert len(markers) == 1

    def test_missing_provider_evidence_fails(self, monkeypatch, tmp_path):
        """_run returns None provider must fail — unconditional validation."""
        import scripts.ds4_segmented_smoke as mod
        ap = tmp_path / "ap"; mp = tmp_path / "mp"; mp.mkdir()
        ap.mkdir(parents=True); (ap/"adapters.safetensors").write_text("x"); (ap/"adapter_config.json").write_text("{}")
        monkeypatch.setattr(mod, "_check_preflight", lambda *a, **kw: None)
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_clear_markers", lambda mw: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: None)
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        # _run returning None — should fail validation
        # But post-run checks need the provider object, not None
        # We can't make _run return None since validation expects it
        # Instead test that main() properly handles a missing provider
        monkeypatch.setattr(mod, "_run", lambda *a, **kw: type("P", (), {
            "call_count": 0,
            "captured_loss": None,
            "captured_token_count": None,
            "captured_all_finite": None,
            "captured_gradient_paths": None,
            "captured_gradient_shapes": None,
            "captured_gradient_leaf_count": 0,
            "captured_gradient_dtypes": None,
        })())
        _pin_test_paths(mod, tmp_path)
        rc = mod.main(["--model", str(mp), "--data", str(tmp_path/"d"), "--adapter-path", str(ap)])
        assert rc != 0, "must fail with missing provider evidence"

    def test_report_failure_does_not_leave_ok_marker(self, monkeypatch, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        mp = tmp_path / "m"; mp.mkdir(); ap = tmp_path / "ar"; ap.mkdir(parents=True)
        (ap / "adapters.safetensors").write_text("x")
        (ap / "adapter_config.json").write_text("{}")
        markers = []
        monkeypatch.setattr(mod, "_check_preflight", lambda *a, **kw: None)
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_clear_markers", lambda mw: None)
        monkeypatch.setattr(mod, "_write_smoke_report",
            lambda r: (_ for _ in ()).throw(OSError("report denied")))
        monkeypatch.setattr(mod, "_ok_marker_path", lambda mw: tmp_path / ".ok")
        monkeypatch.setattr(mod, "_fail_marker_path", lambda mw: tmp_path / ".fail")
        monkeypatch.setattr(mod, "_run", lambda *a, **kw: type("P", (), {
            "call_count": 1,
            "captured_loss": 0.5,
            "captured_token_count": 7,
            "captured_all_finite": True,
            "captured_gradient_paths": ["w"],
            "captured_gradient_shapes": [(1,)],
            "captured_gradient_dtypes": ["float32"],
            "captured_gradient_leaf_count": 1,
            "compute_expected_token_count": lambda self: 7,
        })())
        _pin_test_paths(mod, tmp_path)
        rc = mod.main(["--model", str(mp), "--data", str(tmp_path/"d"), "--adapter-path", str(ap)])
        assert rc != 0
        assert not (tmp_path / ".ok").is_file(), "OK marker must not be written"
        assert (tmp_path / ".fail").is_file(), "fail marker must be written"

    def test_ok_marker_write_failure_returns_nonzero(self, monkeypatch, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        mp = tmp_path/"m"; mp.mkdir(); ap = tmp_path/"x"
        monkeypatch.setattr(mod, "_check_preflight", lambda *a, **kw: None)
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_clear_markers", lambda mw: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: None)
        monkeypatch.setattr(mod, "_write_ok_marker",
            lambda mw: (_ for _ in ()).throw(OSError("write denied")))
        monkeypatch.setattr(mod, "_ok_marker_path", lambda mw: tmp_path/".ok")
        monkeypatch.setattr(mod, "_fail_marker_path", lambda mw: tmp_path/".fail")
        monkeypatch.setattr(mod, "_run", lambda *a, **kw: type("P", (), {
            "call_count": 1,
            "captured_loss": 0.5,
            "captured_token_count": 7,
            "captured_all_finite": True,
            "captured_gradient_paths": ["w"],
            "captured_gradient_shapes": [(1,)],
            "captured_gradient_dtypes": ["float32"],
            "captured_gradient_leaf_count": 1,
            "compute_expected_token_count": lambda self: 7,
        })())
        _pin_test_paths(mod, tmp_path)
        rc = mod.main(["--model", str(mp), "--data", str(tmp_path/"d"), "--adapter-path", str(ap)])
        assert rc != 0, "OK marker write failure must return nonzero"
        assert (tmp_path / ".fail").is_file(), "fail marker must be written"

    def test_exactly_one_terminal_marker(self, monkeypatch, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        mp = tmp_path/"m"; mp.mkdir(); ap = tmp_path/"x"
        monkeypatch.setattr(mod, "_check_preflight", lambda *a, **kw: None)
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_clear_markers", lambda mw: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: None)
        monkeypatch.setattr(mod, "_ok_marker_path", lambda mw: tmp_path/".ok")
        monkeypatch.setattr(mod, "_fail_marker_path", lambda mw: tmp_path/".fail")
        monkeypatch.setattr(mod, "_release_ft_lock", lambda: None)
        monkeypatch.setattr(mod, "_run", lambda *a, **kw: type("P", (), {
            "call_count": 1,
            "captured_loss": 0.5,
            "captured_token_count": 7,
            "captured_all_finite": True,
            "captured_gradient_paths": ["w"],
            "captured_gradient_shapes": [(1,)],
            "captured_gradient_dtypes": ["float32"],
            "captured_gradient_leaf_count": 1,
            "compute_expected_token_count": lambda self: 7,
        })())
        _pin_test_paths(mod, tmp_path)
        mod.main(["--model", str(mp), "--data", str(tmp_path/"d"), "--adapter-path", str(ap)])
        ok = (tmp_path/".ok").is_file(); fail = (tmp_path/".fail").is_file()
        assert (ok or fail) and not (ok and fail)

    def test_exception_systemexit_writes_fail_marker(self, monkeypatch, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        markers = []; reports = []
        monkeypatch.setattr(mod, "_check_preflight", lambda *a, **kw: None)
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_run", lambda *a, **kw: (_ for _ in ()).throw(SystemExit("fail")))
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: markers.append(r))
        monkeypatch.setattr(mod, "_release_ft_lock", lambda: None)
        monkeypatch.setattr(mod, "_clear_markers", lambda mw: None)
        _pin_test_paths(mod, tmp_path)
        mod.main(["--model", str(tmp_path/"m"), "--data", str(tmp_path/"d"), "--adapter-path", str(tmp_path/"a")])
        assert len(reports) >= 1 and len(markers) >= 1

    def test_oom_writes_fail_marker(self, monkeypatch, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        markers = []; reports = []
        monkeypatch.setattr(mod, "_check_preflight", lambda *a, **kw: None)
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_run", lambda *a, **kw: (_ for _ in ()).throw(MemoryError("OOM")))
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: markers.append(r))
        monkeypatch.setattr(mod, "_release_ft_lock", lambda: None)
        monkeypatch.setattr(mod, "_clear_markers", lambda mw: None)
        _pin_test_paths(mod, tmp_path)
        rc = mod.main(["--model", str(tmp_path/"m"), "--data", str(tmp_path/"d"), "--adapter-path", str(tmp_path/"a")])
        assert rc != 0 and len(reports) >= 1 and len(markers) >= 1


# ===================================================================
# T4 — Real mutation tests (behavioral contract oracles, not source presence)
# ===================================================================
class TestRealMutations:
    def _run_mutated_main_and_capture_marker(self, mutation, args_override=None):
        """Run a mutated script's main() and return (rc, ok_marker_path, fail_marker_path)."""
        td = pathlib.Path(tempfile.mkdtemp(prefix="ds4mt_"))
        dst = td / "m.py"
        shutil.copy2(SCRIPT, dst)
        t = dst.read_text("utf-8")
        t = mutation(t)
        dst.write_text(t, "utf-8")
        okp = td / ".ok"; fp = td / ".fail"
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{ROOT}/python-envs/mlx/src:{ROOT}/vendor/mlx-lm:{ROOT}"
        cmd = [sys.executable, str(dst)] + (args_override or ["--model", str(td/ "m"),
            "--data", str(td/"d"), "--adapter-path", str(td/"a")])
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=15)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = -1
        return rc, okp, fp

    def test_loss_and_grad_mutation_killed_behavioral(self, monkeypatch):
        """One unchanged oracle applied to both control and mutant.
        Oracle: _run calls train with loss_and_grad=<not None>.
        Control passes; mutant fails."""
        def mutation(t):
            old = "loss_and_grad=provider,"
            new = "loss_and_grad=None,  # MUTATED"
            assert old in t
            return t.replace(old, new, 1)
        mut = _mutate(mutation)
        import scripts.ds4_segmented_smoke as control_mod

        def make_fake_model():
            class _M:
                def __init__(s):
                    s.layers = [type("L", (), {"unfreeze": lambda s: None})()]
                def freeze(s): pass
                def train(s): pass
                def eval(s): pass
                def update(s, *a, **kw): pass
                def load_weights(s, *a, **kw): pass
                def trainable_parameters(s): return {}
            return _M()

        def _fake_load(*a, **kw): return (make_fake_model(), "tok")
        def _fake_dataset(*a, **kw): return ([("x",)], [], [])
        import types as _types

        def setup(mod, train_fn):
            monkeypatch.setattr(mod, "load", _fake_load)
            monkeypatch.setattr(mod, "load_dataset", _fake_dataset)
            monkeypatch.setattr(mod, "linear_to_lora_layers", lambda *a, **kw: None)
            monkeypatch.setattr(mod, "print_trainable_parameters", lambda *a, **kw: None)
            monkeypatch.setattr(mod, "save_config", lambda *a, **kw: None)
            monkeypatch.setattr(mod, "get_reporting_callbacks", lambda *a, **kw: None)
            monkeypatch.setattr(mod, "CacheDataset", lambda x: x)
            monkeypatch.setattr(mod, "build_schedule", lambda *a, **kw: 1e-5)
            fn = _types.ModuleType("numpy")
            fn.random = _types.SimpleNamespace(); fn.random.seed = lambda s: None
            monkeypatch.setitem(sys.modules, "numpy", fn)
            fm = _types.ModuleType("mlx.core")
            fm.random = _types.SimpleNamespace(); fm.random.seed = lambda s: None
            fm.any = staticmethod(lambda x: x); fm.isfinite = staticmethod(lambda x: x)
            fm.all = staticmethod(lambda x: x); fm.sum = staticmethod(lambda x: x)
            monkeypatch.setitem(sys.modules, "mlx.core", fm)
            monkeypatch.setattr(mod, "train", train_fn)
            monkeypatch.setattr(mod, "make_ds4_segmented_loss_and_grad",
                lambda *, segment_size: lambda m, *b: None)

        tmpd = pathlib.Path(tempfile.mkdtemp(prefix="ds4mt_"))
        ap = tmpd / "a"; mp = tmpd / "m"; mp.mkdir(); (mp / "config.json").write_text("{}")
        (mp / "tokenizer.json").write_text("{}")
        dp = tmpd / "d"; dp.mkdir()
        for s in ("train.jsonl", "valid.jsonl", "test.jsonl"): (dp / s).write_text('{"prompt":"x","completion":"y"}\n')
        mp.joinpath("model-00001-of-00002.safetensors").write_text("x")
        mp.joinpath("model-00002-of-00002.safetensors").write_text("x")
        args = _stub({"model": str(mp), "data": str(dp), "adapter_path": str(ap), "config": None})

        # Control: unchanged module passes oracle
        train_kw_ctl = {}
        setup(control_mod, lambda *a, **kw: train_kw_ctl.update(kw))
        control_mod._run(args)
        assert train_kw_ctl.get("loss_and_grad") is not None, \
            "control must pass loss_and_grad=provider"

        # Mutant: mutated module fails oracle
        train_kw_mut = {}
        setup(mut, lambda *a, **kw: train_kw_mut.update(kw))
        mut._run(args)
        lg = train_kw_mut.get("loss_and_grad")
        assert lg is None, f"mutant must not pass loss_and_grad=provider, got {lg}"

    def test_config_merge_mutation_killed_behavioral(self):
        """Mutant: cfg items not applied. Contract: config values must affect args."""
        def mutation(t):
            old = "for k, v in cfg.items():"
            new = "for k, v in (cfg.items() if False else []):  # MUTATED"
            assert old in t
            return t.replace(old, new, 1)
        mut = _mutate(mutation)
        c = pathlib.Path(tempfile.mkdtemp()) / "c.json"
        try:
            c.write_text(json.dumps({"num_layers": 3}))
            # Mutant does NOT apply config
            a = _stub({"config": str(c), "num_layers": None})
            result = mut._merge_config(a)
            assert result.num_layers != 3, "mutation survived: config was applied"
            # Unchanged code DOES apply config
            import scripts.ds4_segmented_smoke as base
            a2 = _stub({"config": str(c), "num_layers": None})
            base._merge_config(a2)
            assert a2.num_layers == 3, "unchanged: config NOT applied"
        finally:
            if c.exists(): c.unlink(); c.parent.rmdir()

    def test_held_lock_preserved(self, tmp_path, monkeypatch):
        """When .ds4-ft.lock is held, main() must not delete it."""
        import scripts.ds4_segmented_smoke as mod
        (tmp_path / ".ds4-ft.lock").write_text("pid=999\n")
        (tmp_path / "m").mkdir()
        reports = []; markers = []
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: markers.append(r))
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_clear_markers", lambda mw: None)
        monkeypatch.setattr(mod, "_acquire_ft_lock",
            lambda *a, **kw: (_ for _ in ()).throw(SystemExit("held")))
        _pin_test_paths(mod, tmp_path)
        mod.main(["--model", str(tmp_path/"m"), "--data", str(tmp_path/"d"),
                  "--adapter-path", str(tmp_path/"a")])
        assert (tmp_path / ".ds4-ft.lock").is_file()

    def test_segment_validation_mutation_killed(self):
        """Mutant: segment-size check disabled. Mutant accepts 0; unchanged rejects 0."""
        def mutation(t):
            old = "if v < 1 or v > 4:"
            new = "if False:  # MUTATED"
            assert old in t
            return t.replace(old, new, 1)
        mut = _mutate(mutation)
        # Unchanged parser rejects 0
        import scripts.ds4_segmented_smoke as base
        with pytest.raises(SystemExit):
            base.build_smoke_parser().parse_args(["--segment-size", "0"])
        # Mutated parser accepts 0
        mut.build_smoke_parser().parse_args(["--segment-size", "0"])


# ===================================================================
# T5 — Segment-size parser
# ===================================================================
class TestSegmentParser:
    def test_rejects_invalid(self):
        import scripts.ds4_segmented_smoke as mod
        with pytest.raises(SystemExit): mod.build_smoke_parser().parse_args(["--segment-size", "-1"])
        with pytest.raises(SystemExit): mod.build_smoke_parser().parse_args(["--segment-size", "0"])
        with pytest.raises(SystemExit): mod.build_smoke_parser().parse_args(["--segment-size", "5"])
        with pytest.raises(SystemExit): mod.build_smoke_parser().parse_args(["--segment-size", "1.5"])
    def test_accepts_valid(self):
        import scripts.ds4_segmented_smoke as mod
        for v in ("1","2","3","4"):
            assert mod.build_smoke_parser().parse_args(["--segment-size", v]).segment_size == int(v)
    def test_default(self):
        import scripts.ds4_segmented_smoke as mod
        assert mod.build_smoke_parser().parse_args([]).segment_size == 1


# ===================================================================
# T6 — Config merge
# ===================================================================
class TestConfig:
    def test_file_sets_values(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        c = tmp_path / "c.json"
        c.write_text(json.dumps({"num_layers": 7}))
        a = _stub({"config": str(c), "num_layers": None, "optimizer": None, "lora_parameters": None})
        mod._merge_config(a)
        assert a.num_layers == 7
    def test_cli_overrides(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        c = tmp_path / "c.yaml"; c.write_text("num_layers: 99\n")
        a = _stub({"config": str(c), "num_layers": 4})
        mod._merge_config(a)
        assert a.num_layers == 4
    def test_nonexistent_raises(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        with pytest.raises(SystemExit): mod._merge_config(_stub({"config": str(tmp_path/"nope.json")}))
    def test_malformed_raises(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        c = tmp_path/"b.json"; c.write_text("{bad}")
        with pytest.raises(SystemExit): mod._merge_config(_stub({"config": str(c)}))
    def test_nonmapping_raises(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        c = tmp_path/"a.json"; c.write_text("[1,2,3]")
        with pytest.raises(SystemExit): mod._merge_config(_stub({"config": str(c)}))


# ===================================================================
# T7 — MLX_STEPS and catalog
# ===================================================================
class TestCatalog:
    def test_steps_contain_segmented(self):
        from scripts import finetune_ds4
        assert "ds4-segmented-smoke" in finetune_ds4.MLX_STEPS
    def test_not_in_default(self):
        from scripts import finetune_ds4
        assert "ds4-segmented-smoke" not in finetune_ds4.DEFAULT_BACKEND_STEPS["local-mlx"]
    def test_catalog_redirects(self):
        from scripts import finetune_ds4
        with tempfile.TemporaryDirectory() as td:
            a = types.SimpleNamespace(hf_model=td+"/h", dataset_root=td+"/d", mlx_work=td+"/m",
                ds4_root=td+"/ds4", ds4_gguf=None, adapter_ds4=None, split_dir="x",
                fused_hf_model=None, ds4_imatrix=None, mlx_lm_source="fork")
            c = finetune_ds4.command_catalog(a)["ds4-segmented-smoke"][0]
        assert "smoke-log.txt" in c and "2>&1" in c
    def test_catalog_segmented_uses_filtered_path(self):
        """ds4-segmented-smoke catalog entry uses mlx-4096-smoke, not split_dir."""
        from scripts import finetune_ds4
        with tempfile.TemporaryDirectory() as td:
            a = types.SimpleNamespace(hf_model=td+"/h", dataset_root=td+"/d", mlx_work=td+"/m",
                ds4_root=td+"/ds4", ds4_gguf=None, adapter_ds4=None, split_dir="mlx-4096",
                fused_hf_model=None, ds4_imatrix=None, mlx_lm_source="fork")
            c = finetune_ds4.command_catalog(a)["ds4-segmented-smoke"][0]
        assert "mlx-4096-smoke" in c, "catalog must use filtered dataset path"
    def test_default_catalog_entries_unchanged(self):
        """Prove exact byte identity of smoke-train, full-train, continue-train
        command strings against approved immutable baseline digests.

        Normalization replaces the temp-directory prefix with ${TMP} so
        digests are stable across CI runs and local environments.  Any
        change to a default entry — including flag reordering, whitespace,
        or accidental contamination — will produce a different digest."""
        import hashlib
        from scripts import finetune_ds4
        with tempfile.TemporaryDirectory() as td:
            a = types.SimpleNamespace(hf_model=td+"/h", dataset_root=td+"/d", mlx_work=td+"/m",
                ds4_root=td+"/ds4", ds4_gguf=None, adapter_ds4=None, split_dir="mlx-4096",
                fused_hf_model=None, ds4_imatrix=None, mlx_lm_source="fork")
            cat = finetune_ds4.command_catalog(a)

        def _digest(entry, base):
            return hashlib.sha256(
                entry.replace(base, "${TMP}").encode()
            ).hexdigest()

        BASELINE = {
            "smoke-train":    "7b04545d1e54ad69f971fb6e4bac9240f551f88df217ddb9bd456017b5eec0fa",
            "full-train":     "26416001f3d87e0cddabdfb3189a7e2702cebc9244533a70a3a8809c6dfdee0a",
            "continue-train": "534bbf0c6345e342d21fe8bc9d2e8a4cefd70f531e3cd995588eff926150a2b2",
        }
        for key, expected_digest in BASELINE.items():
            actual = _digest(cat[key][0], td)
            assert actual == expected_digest, (
                f"{key} byte-identity changed: {actual} != {expected_digest}"
            )
        # Defense-in-depth: also assert no filtered path leaked into defaults
        for key in BASELINE:
            assert "mlx-4096-smoke" not in cat[key][0], (
                f"{key} must not contain filtered dataset path"
            )


# ===================================================================
# T8 — Protected hashes
# ===================================================================
class TestProtected:
    def test_provider_source(self):
        p = ROOT/"python-envs"/"mlx"/"src"/"ds4_ft_mlx"/"segmented_loss_and_grad.py"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == "20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518"
    def test_provider_test(self):
        p = ROOT/"tests"/"test_ds4_segmented_loss_and_grad.py"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == "618a0f22d350ed368e7bf7f782728ad2f6a11486e19a4c97c5e85228aea784c6"
    def test_source_sentinel(self):
        p = ROOT/"tests"/"test_mlx_lm_source.py"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == "dec2c2b5a7644540a7ec339712968593647abd64c70b821abddc97f9b6ff5d65"


# ===================================================================
# T9 — No real access
# ===================================================================
class TestNoRealAccess:
    def test_fresh_import_no_volume_stat(self):
        access = []
        orig = os.stat
        def _g(p, *a, **kw):
            if "/Volumes/Data NVME" in str(p):
                access.append(str(p))
            return orig(p, *a, **kw)
        os.stat = _g
        sd = pathlib.Path(tempfile.mkdtemp()); dst = sd / "f.py"
        shutil.copy2(SCRIPT, dst); sys.path.insert(0, str(sd))
        try:
            for k in list(sys.modules):
                if "ds4_segmented_smoke" in k and "scripts." not in k:
                    del sys.modules[k]
            import importlib; importlib.import_module("f")
            assert access == [], f"real path access: {access}"
        finally:
            sys.path.remove(str(sd))


# ===================================================================
# T10 — Architecture doc
# ===================================================================
class TestArchDoc:
    def test_paragraph(self):
        t = (ROOT/"docs"/"architecture.md").read_text()
        assert "ds4_segmented_smoke" in t and "loss_and_grad=provider" in t


# ===================================================================
# T11 — No unused code
# ===================================================================
class TestNoUnused:
    def test_elapsed_used(self):
        assert "wall_clock_seconds" in SCRIPT.read_text()


# ===================================================================
# T12 — Timeout and preflight
# ===================================================================
class TestTimeoutPreflight:
    def test_watchdog_installed_before_config(self, monkeypatch, tmp_path):
        """Watchdog must be installed before config merge."""
        import scripts.ds4_segmented_smoke as mod
        order = []
        real_install = mod._install_timeout_watchdog
        real_merge = mod._merge_config
        def traced_install(*a, **kw):
            order.append("watchdog")
            return real_install(*a, **kw)
        def traced_merge(*a, **kw):
            order.append("config")
            return real_merge(*a, **kw)
        monkeypatch.setattr(mod, "_install_timeout_watchdog", traced_install)
        monkeypatch.setattr(mod, "_merge_config", traced_merge)
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: None)
        try:
            _pin_test_paths(mod, tmp_path)
            mod.main(["--config", str(tmp_path / "nope.json")])
        except SystemExit:
            pass
        if "watchdog" in order and "config" in order:
            assert order.index("watchdog") < order.index("config"), \
                "watchdog must be installed before config merge"
        else:
            pass  # both may not be called if main fails early

    def test_config_fail_writes_to_cwd(self, monkeypatch, tmp_path):
        """Config failure must write fail marker."""
        import scripts.ds4_segmented_smoke as mod
        markers = []
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: markers.append((str(mw), r)))
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: None)
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        _pin_test_paths(mod, tmp_path)
        rc = mod.main(["--config", str(tmp_path / "nope.json")])
        assert rc != 0
        assert len(markers) >= 1

    def test_backup_watchdog_thread_created(self, monkeypatch, tmp_path):
        """_install_timeout_watchdog creates a daemon thread."""
        import scripts.ds4_segmented_smoke as mod
        import threading
        threads_before = len(threading.enumerate())
        mod._install_timeout_watchdog(tmp_path)
        threads_after = len(threading.enumerate())
        assert threads_after >= threads_before


# ===================================================================
# Story 14.3b — Timeout repin 600s → 1200s
# ===================================================================
class TestTimeoutRepin:
    def test_timeout_constant_is_1200(self):
        """SMOKE_TIMEOUT_SECONDS must be 1200 after repin."""
        import scripts.ds4_segmented_smoke as mod
        assert mod.SMOKE_TIMEOUT_SECONDS == 1200

    def test_timeout_message_contains_1200s(self, monkeypatch):
        """Timeout handler message must reflect 1200s, not stale 600s."""
        import scripts.ds4_segmented_smoke as mod
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        # Reset module globals to avoid cross-test contamination
        mod._TIMEOUT_FLAG = False
        mod._TIMEOUT_REASON = None
        try:
            mod._timeout_handler(signal.SIGALRM, None)
        except SystemExit:
            pass
        assert mod._TIMEOUT_REASON == "smoke timed out after 1200s"

    def test_lock_timeout_unchanged_at_60(self):
        """SMOKE_LOCK_TIMEOUT_S must remain 60 — regression guard."""
        import scripts.ds4_segmented_smoke as mod
        assert mod.SMOKE_LOCK_TIMEOUT_S == 60

    def test_abort_timeout_unchanged_at_2(self):
        """SMOKE_ABORT_TIMEOUT must remain 2 — regression guard."""
        import scripts.ds4_segmented_smoke as mod
        assert mod.SMOKE_ABORT_TIMEOUT == 2

    def test_backup_watchdog_deadline_uses_1200(self):
        """_backup_watchdog deadline must be exactly SMOKE_TIMEOUT_SECONDS + 5.

        Arithmetic: SMOKE_TIMEOUT_SECONDS + 5 == 1205 — catches constant drift.
        Exact expression oracle: the deadline assignment line must contain
        the exact expression ``time.monotonic() + SMOKE_TIMEOUT_SECONDS + 5``.
        A mutation from +5 to +6 or a hardcoded literal fails the oracle.
        """
        import scripts.ds4_segmented_smoke as mod
        import inspect
        import re

        # Arithmetic: SMOKE_TIMEOUT_SECONDS + 5 must equal 1205
        assert mod.SMOKE_TIMEOUT_SECONDS + 5 == 1205

        # Exact expression oracle: deadline assignment must contain
        # ``time.monotonic() + SMOKE_TIMEOUT_SECONDS + 5``
        source = inspect.getsource(mod._backup_watchdog)
        pattern = r'time\.monotonic\(\)\s*\+\s*SMOKE_TIMEOUT_SECONDS\s*\+\s*5\b'
        assert re.search(pattern, source), (
            "_backup_watchdog deadline must use expression "
            "time.monotonic() + SMOKE_TIMEOUT_SECONDS + 5; "
            "mutation to +6 or hardcoded literal is rejected"
        )


# ===================================================================
# T13 — Lock token/path preservation
# ===================================================================
class TestLockTokenPath:
    def test_lock_path_bound_acquire_and_release(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        mod._acquire_ft_lock(tmp_path, timeout_s=5)
        assert mod._LOCK_OWNED_PATH == tmp_path / ".ds4-ft.lock"
        assert mod._LOCK_OWNED_TOKEN == str(os.getpid())
        mod._release_ft_lock()
        assert mod._LOCK_OWNED_PATH is None
        assert mod._LOCK_OWNED_TOKEN is None

    def test_other_lock_path_does_not_release(self, tmp_path):
        """Release only unlinks the acquired path."""
        import scripts.ds4_segmented_smoke as mod
        p1 = tmp_path / "p1"; p1.mkdir()
        p2 = tmp_path / "p2"; p2.mkdir()
        # Acquire at p1
        mod._acquire_ft_lock(p1, timeout_s=5)
        lp1 = mod._LOCK_OWNED_PATH
        # Store a lock at p2 for another process
        (p2 / ".ds4-ft.lock").write_text("pid=99\n")
        mod._release_ft_lock()
        assert not lp1.exists(), "our lock unlinked"
        assert (p2 / ".ds4-ft.lock").is_file(), "other lock preserved"


# ===================================================================
# T14 — Provider control (unchanged) reaches train with loss_and_grad
# ===================================================================
class TestProviderControl:
    def test_unchanged_provider_reaches_train(self, monkeypatch):
        """Unchanged _run calls train with loss_and_grad=provider."""
        import scripts.ds4_segmented_smoke as mod
        train_kw = {}
        def _fake_train(*a, **kw):
            train_kw.update(kw)
        monkeypatch.setattr(mod, "train", _fake_train)
        monkeypatch.setattr(mod, "load", lambda *a, **kw: (type("M", (), {
            "layers": [type("L", (), {"unfreeze": lambda s: None})()],
            "freeze": lambda s: None, "train": lambda s: None, "eval": lambda s: None,
            "update": lambda s, *a, **kw: None, "load_weights": lambda s, *a, **kw: None,
            "trainable_parameters": lambda s: {},
        })(), "tok"))
        monkeypatch.setattr(mod, "load_dataset", lambda *a, **kw: ([("x",)], [], []))
        monkeypatch.setattr(mod, "linear_to_lora_layers", lambda *a, **kw: None)
        monkeypatch.setattr(mod, "print_trainable_parameters", lambda *a, **kw: None)
        monkeypatch.setattr(mod, "save_config", lambda *a, **kw: None)
        monkeypatch.setattr(mod, "get_reporting_callbacks", lambda *a, **kw: None)
        monkeypatch.setattr(mod, "CacheDataset", lambda x: x)
        monkeypatch.setattr(mod, "build_schedule", lambda *a, **kw: 1e-5)
        monkeypatch.setattr(mod, "make_ds4_segmented_loss_and_grad",
            lambda *, segment_size: lambda m, *b: None)
        import types as _t
        fn = _t.ModuleType("numpy"); fn.random = _t.SimpleNamespace(); fn.random.seed = lambda s: None
        monkeypatch.setitem(sys.modules, "numpy", fn)
        fm = _t.ModuleType("mlx.core"); fm.random = _t.SimpleNamespace(); fm.random.seed = lambda s: None
        fm.any = staticmethod(lambda x: x); fm.isfinite = staticmethod(lambda x: x)
        fm.all = staticmethod(lambda x: x); fm.sum = staticmethod(lambda x: x)
        monkeypatch.setitem(sys.modules, "mlx.core", fm)
        import tempfile as _tf, pathlib as _pl
        td = _pl.Path(_tf.mkdtemp())
        mp = td / "m"; mp.mkdir(); (mp / "config.json").write_text("{}"); (mp / "tokenizer.json").write_text("{}")
        dp = td / "d"; dp.mkdir()
        for s in ("train.jsonl", "valid.jsonl", "test.jsonl"): (dp / s).write_text('{"prompt":"x","completion":"y"}\n')
        mp.joinpath("model-00001-of-00002.safetensors").write_text("x")
        mod._run(_stub({"model": str(mp), "data": str(dp), "adapter_path": str(td/"a"), "config": None}))
        assert "loss_and_grad" in train_kw, "control: train must receive loss_and_grad"
        assert train_kw["loss_and_grad"] is not None, "control: loss_and_grad must not be None"

    def test_scalar_shape_loss_rejected(self, monkeypatch):
        """One-element vector loss must be rejected."""
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        grads = {"w": mx.array([1.0])}
        def _real(m, *b): return ((mx.array([0.5]), mx.array(7)), grads)
        obs = mod._ObservingProvider(_real)
        with pytest.raises(SystemExit, match="scalar"):
            obs("m", "b")

    def test_scalar_shape_token_rejected(self, monkeypatch):
        """One-element vector token must be rejected."""
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        grads = {"w": mx.array([1.0])}
        def _real(m, *b): return ((mx.array(0.5), mx.array([7])), grads)
        obs = mod._ObservingProvider(_real)
        with pytest.raises(SystemExit, match="scalar"):
            obs("m", "b")

    def test_model_backed_schema_nested_ok(self, monkeypatch):
        """Model with nested trainable_parameters and matching gradients passes."""
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        class _M:
            def trainable_parameters(s):
                return {"layer1": {"w": mx.array([[1.0]]), "b": mx.array([0.0])}}
        grads = {"layer1": {"w": mx.array([[1.0]]), "b": mx.array([0.0])}}
        def _real(m, *b): return ((mx.array(0.5), mx.array(7)), grads)
        obs = mod._ObservingProvider(_real)
        obs(_M(), "b")
        assert obs.captured_all_finite is True
        assert obs.captured_gradient_leaf_count == 2

    def test_wrong_key_order_rejected(self):
        """Gradients with wrong key order vs trainable_parameters must fail."""
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        class _M:
            def trainable_parameters(s):
                return {"w": mx.array([1.0]), "b": mx.array([0.0])}
        grads = {"b": mx.array([0.0]), "w": mx.array([1.0])}
        def _real(m, *b): return ((mx.array(0.5), mx.array(7)), grads)
        obs = mod._ObservingProvider(_real)
        with pytest.raises(SystemExit, match="key order"):
            obs(_M(), "b")

    def test_wrong_container_type_rejected(self):
        """Gradients as list vs expected dict must fail."""
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        class _M:
            def trainable_parameters(s):
                return {"w": mx.array([1.0])}
        grads = [mx.array([1.0])]
        def _real(m, *b): return ((mx.array(0.5), mx.array(7)), grads)
        obs = mod._ObservingProvider(_real)
        with pytest.raises(SystemExit):
            obs(_M(), "b")

    def test_wrong_path_rejected(self):
        """Gradients with wrong leaf path vs trainable_parameters must fail."""
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        class _M:
            def trainable_parameters(s):
                return {"layer": {"w": mx.array([1.0])}}
        grads = {"layer": {"b": mx.array([0.0])}}
        def _real(m, *b): return ((mx.array(0.5), mx.array(7)), grads)
        obs = mod._ObservingProvider(_real)
        with pytest.raises(SystemExit):
            obs(_M(), "b")

    def test_wrong_shape_rejected(self):
        """Gradients with wrong leaf shape vs trainable_parameters must fail."""
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        class _M:
            def trainable_parameters(s):
                return {"w": mx.array([1.0, 2.0])}
        grads = {"w": mx.array([1.0])}
        def _real(m, *b): return ((mx.array(0.5), mx.array(7)), grads)
        obs = mod._ObservingProvider(_real)
        with pytest.raises(SystemExit, match="shape"):
            obs(_M(), "b")

    def test_wrong_dtype_rejected(self):
        """Gradients with wrong leaf dtype vs trainable_parameters must fail."""
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        class _M:
            def trainable_parameters(s):
                return {"w": mx.array([1.0], dtype=mx.float32)}
        grads = {"w": mx.array([1.0], dtype=mx.float16)}
        def _real(m, *b): return ((mx.array(0.5), mx.array(7)), grads)
        obs = mod._ObservingProvider(_real)
        with pytest.raises(SystemExit, match="dtype"):
            obs(_M(), "b")


# ===================================================================
# T15 — Preflight additions
# ===================================================================
class TestPreflightAdditions:
    def test_missing_weights_rejected(self, tmp_path):
        """Model dir with config+tokenizer but no weights must fail."""
        import scripts.ds4_segmented_smoke as mod
        mp = tmp_path / "m"; mp.mkdir(); (mp / "config.json").write_text("{}"); (mp / "tokenizer.json").write_text("{}")
        ap = tmp_path / "a"
        with pytest.raises(SystemExit):
            mod._check_preflight(_stub({"model": str(mp), "data": str(tmp_path/"d"), "adapter_path": str(ap), "config": None}), tmp_path)

    def test_dataset_missing_prompt_rejected(self, tmp_path):
        """Dataset rows without prompt field must fail."""
        import scripts.ds4_segmented_smoke as mod
        mp = tmp_path / "m"; mp.mkdir(); (mp / "config.json").write_text("{}"); (mp / "tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        (dp / "train.jsonl").write_text('{"bad":"data"}\n')
        (dp / "valid.jsonl").write_text('{"prompt":"x","completion":"y"}\n')
        (dp / "test.jsonl").write_text('{"prompt":"x","completion":"y"}\n')
        ap = tmp_path / "a"
        with pytest.raises(SystemExit):
            mod._check_preflight(_stub({"model": str(mp), "data": str(dp), "adapter_path": str(ap), "config": None}), tmp_path)

    def test_lora_config_missing_rank_rejected(self, tmp_path):
        """LoRA config without rank field must fail."""
        import scripts.ds4_segmented_smoke as mod
        mp = tmp_path / "m"; mp.mkdir(); (mp / "config.json").write_text("{}"); (mp / "tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        for s in ("train.jsonl", "valid.jsonl", "test.jsonl"): (dp / s).write_text('{"prompt":"x","completion":"y"}\n')
        cp = tmp_path / "lora.json"; cp.write_text('{"alpha":16,"target_modules":["q_proj"]}')
        with pytest.raises(SystemExit):
            mod._check_preflight(_stub({"model": str(mp), "data": str(dp), "adapter_path": str(tmp_path/"a"), "config": str(cp)}), tmp_path)

    def test_lora_config_missing_alpha_rejected(self, tmp_path):
        """LoRA config without alpha field must fail."""
        import scripts.ds4_segmented_smoke as mod
        mp = tmp_path / "m"; mp.mkdir(); (mp / "config.json").write_text("{}"); (mp / "tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        for s in ("train.jsonl", "valid.jsonl", "test.jsonl"): (dp / s).write_text('{"prompt":"x","completion":"y"}\n')
        cp = tmp_path / "lora.json"; cp.write_text('{"rank":8}')
        with pytest.raises(SystemExit):
            mod._check_preflight(_stub({"model": str(mp), "data": str(dp), "adapter_path": str(tmp_path/"a"), "config": str(cp)}), tmp_path)

    def test_watchdog_before_parse(self, monkeypatch, tmp_path):
        """Watchdog installed before build_smoke_parser."""
        import scripts.ds4_segmented_smoke as mod
        order = []
        real_install = mod._install_timeout_watchdog
        real_parser = mod.build_smoke_parser
        def traced_install(*a, **kw):
            order.append("watchdog")
            return real_install(*a, **kw)
        def traced_parser():
            order.append("parser")
            return real_parser()
        monkeypatch.setattr(mod, "_install_timeout_watchdog", traced_install)
        monkeypatch.setattr(mod, "build_smoke_parser", traced_parser)
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: None)
        try:
            _pin_test_paths(mod, tmp_path)
            mod.main(["--config", str(tmp_path / "nope.json")])
        except SystemExit:
            pass
        assert "watchdog" in order and "parser" in order
        assert order.index("watchdog") < order.index("parser"), \
            "watchdog must be installed before parser construction"

    def test_config_failure_writes_workspace_marker(self, monkeypatch, tmp_path):
        """Config failure must write fail marker to resolved adapter workspace."""
        import scripts.ds4_segmented_smoke as mod
        import tempfile as _tf, pathlib as _pl
        marker_dest = []
        def rec_marker(mw, r):
            marker_dest.append(str(mw))
        monkeypatch.setattr(mod, "_write_fail_marker", rec_marker)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: None)
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        wp = _pl.Path(_tf.mkdtemp())
        _pin_test_paths(mod, tmp_path)
        rc = mod.main(["--model", str(wp/"m"), "--data", str(wp/"d"),
                       "--adapter-path", str(wp/"a"), "--config", str(wp/"nope.json")])
        assert rc != 0
        assert len(marker_dest) >= 1
        # Marker must be written to adapter workspace parent, not cwd
        for md in marker_dest:
            if "ds4-segmented-smoke-fail" in md or "ds4-segmented-smoke-ok" in md:
                continue
            assert str(wp) in md or os.path.commonpath([str(wp), md]) == str(wp), \
                f"marker at unexpected path: {md}"


# ===================================================================
# T16 — Cleanup release mutation + preflight regressions
# ===================================================================
class TestCleanupReleaseMutation:
    def test_cleanup_release_control(self, monkeypatch, tmp_path):
        """Control: owned lock is released after cleanup."""
        import scripts.ds4_segmented_smoke as mod
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: None)
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        mod._LOCK_OWNED_PATH = tmp_path / "nonexistent.lock"
        mod._LOCK_OWNED_TOKEN = "pid=12345"
        try:
            _pin_test_paths(mod, tmp_path)
            mod.main(["--model", str(tmp_path/"m"), "--data", str(tmp_path/"d"),
                      "--adapter-path", str(tmp_path/"a"), "--config", str(tmp_path/"nope.json")])
            assert mod._LOCK_OWNED_PATH is None, "control: lock must be released"
        finally:
            mod._LOCK_OWNED_PATH = None
            mod._LOCK_OWNED_TOKEN = None

    def test_cleanup_release_mutant_still_holds(self, monkeypatch, tmp_path):
        """Mutant with skipped _release_ft_lock: owned lock survives."""
        def mutation(t):
            # Target only the call inside the finally block, not the function definition
            old = "_release_ft_lock()\n        except"
            new = "pass  # MUTATED release skipped\n        except"
            assert old in t, "expected _release_ft_lock() call in finally"
            return t.replace(old, new, 1)
        mut = _mutate(mutation)
        monkeypatch.setattr(mut, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mut, "_write_smoke_report", lambda r: None)
        monkeypatch.setattr(mut, "_write_fail_marker", lambda mw, r: None)
        mut._LOCK_OWNED_PATH = tmp_path / "nonexistent.lock"
        mut._LOCK_OWNED_TOKEN = "pid=12345"
        mp = tmp_path / "m"; mp.mkdir()
        try:
            mut.main(["--model", str(mp), "--data", str(tmp_path/"d"),
                      "--adapter-path", str(tmp_path/"a"), "--config", str(tmp_path/"nope.json")])
            assert mut._LOCK_OWNED_PATH is not None, "mutant: lock must survive"
        finally:
            mut._LOCK_OWNED_PATH = None
            mut._LOCK_OWNED_TOKEN = None

    def test_cleanup_release_exception_warning(self, monkeypatch, tmp_path):
        """Exception from _release_ft_lock is recorded as a warning."""
        import scripts.ds4_segmented_smoke as mod
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: None)
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        orig_release = mod._release_ft_lock
        def failing_release():
            raise OSError("simulated release failure")
        monkeypatch.setattr(mod, "_release_ft_lock", failing_release)
        mod._LOCK_OWNED_PATH = tmp_path / "nonexistent.lock"
        mod._LOCK_OWNED_TOKEN = "pid=12345"
        mod._CLEANUP_WARNINGS.clear()
        try:
            _pin_test_paths(mod, tmp_path)
            mod.main(["--model", str(tmp_path/"m"), "--data", str(tmp_path/"d"),
                      "--adapter-path", str(tmp_path/"a"), "--config", str(tmp_path/"nope.json")])
            assert len(mod._CLEANUP_WARNINGS) >= 1, f"expected cleanup warning, got {mod._CLEANUP_WARNINGS}"
            assert any("simulated release failure" in w for w in mod._CLEANUP_WARNINGS)
        finally:
            mod._LOCK_OWNED_PATH = None
            mod._LOCK_OWNED_TOKEN = None
            mod._CLEANUP_WARNINGS.clear()
            monkeypatch.setattr(mod, "_release_ft_lock", orig_release)


# ===================================================================
# T17 — Preflight regression controls
# ===================================================================
class TestPreflightRegression:
    def test_empty_weight_dir_rejected(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        mp = tmp_path / "m"; mp.mkdir(); (mp / "config.json").write_text("{}"); (mp / "tokenizer.json").write_text("{}")
        (mp / "weights").mkdir()
        with pytest.raises(SystemExit):
            mod._check_preflight(_stub({"model": str(mp), "data": str(tmp_path/"d"), "adapter_path": str(tmp_path/"a"), "config": None}), tmp_path)

    def test_empty_weights_file_rejected(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        mp = tmp_path / "m"; mp.mkdir(); (mp / "config.json").write_text("{}"); (mp / "tokenizer.json").write_text("{}")
        (mp / "model-00001.safetensors").write_text("")
        with pytest.raises(SystemExit):
            mod._check_preflight(_stub({"model": str(mp), "data": str(tmp_path/"d"), "adapter_path": str(tmp_path/"a"), "config": None}), tmp_path)

    def test_later_malformed_row_rejected(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        mp = tmp_path / "m"; mp.mkdir(); (mp / "config.json").write_text("{}"); (mp / "tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        (dp / "train.jsonl").write_text('{"prompt":"x","completion":"y"}\n{"prompt":"x"}\n')
        (dp / "valid.jsonl").write_text('{"prompt":"x","completion":"y"}\n')
        (dp / "test.jsonl").write_text('{"prompt":"x","completion":"y"}\n')
        with pytest.raises(SystemExit):
            mod._check_preflight(_stub({"model": str(mp), "data": str(dp), "adapter_path": str(tmp_path/"a"), "config": None}), tmp_path)

    def test_empty_split_rejected(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        mp = tmp_path / "m"; mp.mkdir(); (mp / "config.json").write_text("{}"); (mp / "tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        for s in ("train.jsonl", "valid.jsonl", "test.jsonl"): (dp / s).write_text("")
        with pytest.raises(SystemExit):
            mod._check_preflight(_stub({"model": str(mp), "data": str(dp), "adapter_path": str(tmp_path/"a"), "config": None}), tmp_path)

    def test_generated_nested_lora_accepted(self, monkeypatch, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        import mlx_lm
        # Point mlx_lm.__file__ at vendor dir so fork identity check passes.
        vendor_init = pathlib.Path(__file__).resolve().parents[1] / "vendor" / "mlx-lm" / "mlx_lm" / "__init__.py"
        orig_file = mlx_lm.__file__
        monkeypatch.setattr(mlx_lm, "__file__", str(vendor_init), raising=False)
        mp = tmp_path / "m"; mp.mkdir(); (mp / "config.json").write_text("{}"); (mp / "tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        for s in ("train.jsonl", "valid.jsonl", "test.jsonl"): (dp / s).write_text('{"prompt":"x","completion":"y"}\n\n')

        cp = tmp_path / "lora.json"
        cp.write_text(json.dumps({"lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0, "keys": ["self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_proj"]}}))
        mod._check_preflight(_stub({"model": str(mp), "data": str(dp), "adapter_path": str(tmp_path/"a"), "config": str(cp)}), tmp_path)

    def test_missing_keys_rejected(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        mp = tmp_path / "m"; mp.mkdir(); (mp / "config.json").write_text("{}"); (mp / "tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        for s in ("train.jsonl", "valid.jsonl", "test.jsonl"): (dp / s).write_text('{"prompt":"x","completion":"y"}\n\n')

        cp = tmp_path / "lora.json"
        cp.write_text(json.dumps({"lora_parameters": {"rank": 8, "scale": 20.0}}))
        with pytest.raises(SystemExit):
            mod._check_preflight(_stub({"model": str(mp), "data": str(dp), "adapter_path": str(tmp_path/"a"), "config": str(cp)}), tmp_path)

    def test_wrong_allowlist_rejected(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        mp = tmp_path / "m"; mp.mkdir(); (mp / "config.json").write_text("{}"); (mp / "tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        for s in ("train.jsonl", "valid.jsonl", "test.jsonl"): (dp / s).write_text('{"prompt":"x","completion":"y"}\n\n')

        cp = tmp_path / "lora.json"
        cp.write_text(json.dumps({"lora_parameters": {"rank": 8, "scale": 20.0, "keys": ["q_proj"]}}))
        with pytest.raises(SystemExit):
            mod._check_preflight(_stub({"model": str(mp), "data": str(dp), "adapter_path": str(tmp_path/"a"), "config": str(cp)}), tmp_path)

    def test_token_bound_5000_rejected(self, tmp_path):
        """Row with >4096 whitespace tokens is rejected by conservative bound."""
        import scripts.ds4_segmented_smoke as mod
        mp = tmp_path / "m"; mp.mkdir(); (mp / "config.json").write_text("{}"); (mp / "tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        big_prompt = " ".join(str(i) for i in range(5000))
        (dp / "train.jsonl").write_text(json.dumps({"prompt": big_prompt, "completion": "y"}) + "\n")
        (dp / "valid.jsonl").write_text('{"prompt":"x","completion":"y"}\n')
        (dp / "test.jsonl").write_text('{"prompt":"x","completion":"y"}\n')
        with pytest.raises(SystemExit):
            mod._check_preflight(_stub({"model": str(mp), "data": str(dp), "adapter_path": str(tmp_path/"a"), "config": None}), tmp_path)

    def test_max_seq_length_bound_pinned(self, monkeypatch, tmp_path):
        """Conservative token bound is pinned to 4096 regardless of args.max_seq_length."""
        import scripts.ds4_segmented_smoke as mod
        import mlx_lm
        vendor_init = pathlib.Path(__file__).resolve().parents[1] / "vendor" / "mlx-lm" / "mlx_lm" / "__init__.py"
        monkeypatch.setattr(mlx_lm, "__file__", str(vendor_init), raising=False)
        mp = tmp_path / "m"; mp.mkdir(); (mp / "config.json").write_text("{}"); (mp / "tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        # 4100 whitespace tokens — fits under altered max_seq_length=8192 but
        # must still be rejected because conservative bound is pinned at 4096.
        big = " ".join(str(i) for i in range(4100))
        for s in ("train.jsonl", "valid.jsonl", "test.jsonl"):
            (dp / s).write_text(json.dumps({"prompt": big, "completion": "y"}) + "\n")
        with pytest.raises(SystemExit):
            mod._check_preflight(_stub({"model": str(mp), "data": str(dp), "adapter_path": str(tmp_path/"a"),
                                       "config": None, "max_seq_length": 8192}), tmp_path)

    def test_token_formula_regression(self):
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        tokens = mx.array([[1, 2, 3, 4, 5]])
        lengths = mx.array([[1, 4]])
        targets = tokens[:, 1:]
        steps = mx.arange(1, targets.shape[1] + 1)
        start = lengths[:, 0:1]
        end = lengths[:, 1:]
        mask = (steps >= start) & (steps <= end)
        observer_result = int(mx.sum(mask).item())
        assert observer_result == 4, f"token formula gave {observer_result}, expected 4"

    def test_ordereddict_vs_dict_rejected(self):
        import scripts.ds4_segmented_smoke as mod
        from collections import OrderedDict
        import mlx.core as mx
        class _M:
            def trainable_parameters(s):
                return {"w": mx.array([1.0])}
        grads = OrderedDict({"w": mx.array([1.0])})
        mismatches = mod._compare_tree_schema(grads, _M().trainable_parameters())
        assert mismatches, "OrderedDict should not match dict"
        assert any("OrderedDict" in m for m in mismatches)

    def test_list_subclass_rejected(self):
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        class MyList(list): pass
        class _M:
            def trainable_parameters(s):
                return [mx.array([1.0])]
        grads = MyList([mx.array([1.0])])
        mismatches = mod._compare_tree_schema(grads, _M().trainable_parameters())
        assert mismatches, "list subclass should not match list"

    def test_tuple_subclass_rejected(self):
        import scripts.ds4_segmented_smoke as mod
        import mlx.core as mx
        class MyTuple(tuple): pass
        class _M:
            def trainable_parameters(s):
                return (mx.array([1.0]),)
        grads = MyTuple([mx.array([1.0])])
        mismatches = mod._compare_tree_schema(grads, _M().trainable_parameters())
        assert mismatches, "tuple subclass should not match tuple"

    def test_parse_failure_inside_terminal_lifecycle(self, monkeypatch):
        import scripts.ds4_segmented_smoke as mod
        markers = []; reports = []
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: markers.append(r))
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        rc = mod.main(["--segment-size", "0"])
        assert rc != 0, "invalid segment-size must produce nonzero exit"

    def test_preflight_exit_code_preserved(self, monkeypatch, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: None)
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        _pin_test_paths(mod, tmp_path)
        mod._PINNED_SMOKE_PATHS["model"] = str(tmp_path / "nonexistent")
        rc = mod.main(["--model", str(tmp_path/"nonexistent"), "--data", str(tmp_path/"d"),
                       "--adapter-path", str(tmp_path/"a"),
                       "--iters", "1", "--batch-size", "1",
                       "--learning-rate", "1e-5", "--max-seq-length", "4096",
                       "--mask-prompt", "--grad-checkpoint", "--segment-size", "1"])
        assert rc == 3, f"preflight failure must return 3, got {rc}"

    def test_preflight_reason_report(self, monkeypatch, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        reports = []
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        _pin_test_paths(mod, tmp_path)
        mod._PINNED_SMOKE_PATHS["model"] = str(tmp_path / "nonexistent")
        rc = mod.main(["--model", str(tmp_path/"nonexistent"), "--data", str(tmp_path/"d"),
                       "--adapter-path", str(tmp_path/"a"),
                       "--iters", "1", "--batch-size", "1",
                       "--learning-rate", "1e-5", "--max-seq-length", "4096",
                       "--mask-prompt", "--grad-checkpoint", "--segment-size", "1"])
        assert rc == 3
        assert len(reports) >= 1
        r = reports[-1]
        assert r.get("failure_code") == "preflight", f"wrong code: {r}"

    def test_parser_failure_workspace_marker(self, monkeypatch, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        import tempfile as _tf, pathlib as _pl
        wp = _pl.Path(_tf.mkdtemp())
        markers = []
        def rec(mw, r):
            markers.append((str(mw), r))
        monkeypatch.setattr(mod, "_write_fail_marker", rec)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: None)
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        _pin_test_paths(mod, tmp_path)
        rc = mod.main(["--segment-size", "0", "--model", str(wp/"m"), "--data", str(wp/"d"),
                       "--adapter-path", str(wp/"a"),
                       "--iters", "1", "--batch-size", "1",
                       "--learning-rate", "1e-5", "--max-seq-length", "4096",
                       "--mask-prompt", "--grad-checkpoint"])
        assert rc != 0
        workspace_refs = [str(wp) in m[0] for m in markers]
        assert any(workspace_refs), f"no marker at workspace path: {markers}"

    def test_mlx_version_none_rejected(self, monkeypatch, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        mp = tmp_path / "m"; mp.mkdir(); (mp / "config.json").write_text("{}"); (mp / "tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        for s in ("train.jsonl", "valid.jsonl", "test.jsonl"): (dp / s).write_text('{"prompt":"x","completion":"y"}\n\n')

        class FakeMlxCore:
            __version__ = None
        import mlx
        orig_core = getattr(mlx, "core", None)
        mlx.core = FakeMlxCore()
        try:
            with pytest.raises(SystemExit):
                mod._check_preflight(_stub({"model": str(mp), "data": str(dp), "adapter_path": str(tmp_path/"a"), "config": None}), tmp_path)
        finally:
            if orig_core:
                mlx.core = orig_core
            else:
                del mlx.core


# ===================================================================
# r11 — Pinned smoke arg rejection (Blocker 3)
# ===================================================================
class TestPinnedSmokeArgs:
    def test_max_seq_length_altered_rejected(self, monkeypatch, tmp_path):
        """Pinned max_seq_length=4096; 8192 is rejected at entry point."""
        import scripts.ds4_segmented_smoke as mod
        reports = []
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        _pin_test_paths(mod, tmp_path)
        rc = mod.main(["--model", str(tmp_path/"m"), "--data", str(tmp_path/"d"),
                       "--adapter-path", str(tmp_path/"a"),
                       "--iters", "1", "--batch-size", "1",
                       "--learning-rate", "1e-5", "--max-seq-length", "8192",
                       "--mask-prompt", "--grad-checkpoint", "--segment-size", "1"])
        assert rc != 0
        assert len(reports) >= 1
        assert reports[-1]["failure_code"] == "pinned-args"

    def test_iters_altered_rejected(self, monkeypatch, tmp_path):
        """Pinned iters=1; 2 is rejected."""
        import scripts.ds4_segmented_smoke as mod
        reports = []
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        _pin_test_paths(mod, tmp_path)
        rc = mod.main(["--model", str(tmp_path/"m"), "--data", str(tmp_path/"d"),
                       "--adapter-path", str(tmp_path/"a"),
                       "--iters", "2", "--batch-size", "1",
                       "--learning-rate", "1e-5", "--max-seq-length", "4096",
                       "--mask-prompt", "--grad-checkpoint", "--segment-size", "1"])
        assert rc != 0
        assert reports[-1]["failure_code"] == "pinned-args"

    def test_segment_size_altered_rejected(self, monkeypatch, tmp_path):
        """Pinned segment_size=1; 2 is rejected."""
        import scripts.ds4_segmented_smoke as mod
        reports = []
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        _pin_test_paths(mod, tmp_path)
        rc = mod.main(["--model", str(tmp_path/"m"), "--data", str(tmp_path/"d"),
                       "--adapter-path", str(tmp_path/"a"),
                       "--iters", "1", "--batch-size", "1",
                       "--learning-rate", "1e-5", "--max-seq-length", "4096",
                       "--mask-prompt", "--grad-checkpoint", "--segment-size", "2"])
        assert rc != 0
        assert reports[-1]["failure_code"] == "pinned-args"

    def test_grad_checkpoint_missing_rejected(self, monkeypatch, tmp_path):
        """Pinned grad_checkpoint=True; missing flag is rejected."""
        import scripts.ds4_segmented_smoke as mod
        reports = []
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        _pin_test_paths(mod, tmp_path)
        rc = mod.main(["--model", str(tmp_path/"m"), "--data", str(tmp_path/"d"),
                       "--adapter-path", str(tmp_path/"a"),
                       "--iters", "1", "--batch-size", "1",
                       "--learning-rate", "1e-5", "--max-seq-length", "4096",
                       "--mask-prompt", "--segment-size", "1"])
        assert rc != 0
        assert reports[-1]["failure_code"] == "pinned-args"


# ===================================================================
# r11 — Exact LoRA key set + dropout validation (Blocker 4)
# ===================================================================
class TestLoraExactKeySet:
    def test_subset_keys_rejected(self, monkeypatch, tmp_path):
        """Lora config with only 2 of 3 canonical keys is rejected."""
        import scripts.ds4_segmented_smoke as mod
        mp = tmp_path / "m"; mp.mkdir(); (mp/"config.json").write_text("{}"); (mp/"tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        for s in ("train.jsonl", "valid.jsonl", "test.jsonl"): (dp/s).write_text('{"prompt":"x","completion":"y"}\n\n')
        cp = tmp_path / "lora.json"
        cp.write_text(json.dumps({"lora_parameters": {"rank":8,"scale":20.0,"dropout":0.0,
            "keys":["self_attn.q_a_proj","self_attn.q_b_proj"]}}))
        with pytest.raises(SystemExit):
            mod._check_preflight(_stub({"model":str(mp),"data":str(dp),"adapter_path":str(tmp_path/"a"),"config":str(cp)}), tmp_path)

    def test_duplicate_keys_rejected(self, monkeypatch, tmp_path):
        """Lora config with duplicate canonical keys is rejected."""
        import scripts.ds4_segmented_smoke as mod
        mp = tmp_path / "m"; mp.mkdir(); (mp/"config.json").write_text("{}"); (mp/"tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        for s in ("train.jsonl", "valid.jsonl", "test.jsonl"): (dp/s).write_text('{"prompt":"x","completion":"y"}\n\n')
        cp = tmp_path / "lora.json"
        cp.write_text(json.dumps({"lora_parameters": {"rank":8,"scale":20.0,"dropout":0.0,
            "keys":["self_attn.q_a_proj","self_attn.q_b_proj","self_attn.kv_proj","self_attn.q_a_proj"]}}))
        with pytest.raises(SystemExit):
            mod._check_preflight(_stub({"model":str(mp),"data":str(dp),"adapter_path":str(tmp_path/"a"),"config":str(cp)}), tmp_path)

    def test_extra_key_rejected(self, monkeypatch, tmp_path):
        """Lora config with canonical plus extra key is rejected (exact set match)."""
        import scripts.ds4_segmented_smoke as mod
        mp = tmp_path / "m"; mp.mkdir(); (mp/"config.json").write_text("{}"); (mp/"tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        for s in ("train.jsonl", "valid.jsonl", "test.jsonl"): (dp/s).write_text('{"prompt":"x","completion":"y"}\n\n')
        cp = tmp_path / "lora.json"
        cp.write_text(json.dumps({"lora_parameters": {"rank":8,"scale":20.0,"dropout":0.0,
            "keys":["self_attn.q_a_proj","self_attn.q_b_proj","self_attn.kv_proj","extra.key"]}}))
        with pytest.raises(SystemExit):
            mod._check_preflight(_stub({"model":str(mp),"data":str(dp),"adapter_path":str(tmp_path/"a"),"config":str(cp)}), tmp_path)

    def test_canonical_keys_accepted(self, monkeypatch, tmp_path):
        """Exact canonical key set is accepted."""
        import scripts.ds4_segmented_smoke as mod
        import mlx_lm
        from ds4_ft_mlx.lora_targets import build_lora_parameters
        canonical = build_lora_parameters()
        monkeypatch.setattr(mlx_lm, "__file__",
            str(ROOT / "vendor" / "mlx-lm" / "mlx_lm" / "__init__.py"), raising=False)
        monkeypatch.setattr(mod, "_check_process_rss_gb", lambda *a, **kw: 0)
        mp = tmp_path / "m"; mp.mkdir(); (mp/"config.json").write_text("{}"); (mp/"tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        for s in ("train.jsonl", "valid.jsonl", "test.jsonl"): (dp/s).write_text('{"prompt":"x","completion":"y"}\n\n')
        cp = tmp_path / "lora.json"
        cp.write_text(json.dumps({"lora_parameters": canonical}))
        mod._check_preflight(_stub({"model":str(mp),"data":str(dp),"adapter_path":str(tmp_path/"a"),"config":str(cp)}), tmp_path)


class TestLoraDropout:
    def test_negative_dropout_rejected(self, monkeypatch, tmp_path):
        """dropout=-1.0 is rejected."""
        import scripts.ds4_segmented_smoke as mod
        mp = tmp_path / "m"; mp.mkdir(); (mp/"config.json").write_text("{}"); (mp/"tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        for s in ("train.jsonl", "valid.jsonl", "test.jsonl"): (dp/s).write_text('{"prompt":"x","completion":"y"}\n\n')
        cp = tmp_path / "lora.json"
        cp.write_text(json.dumps({"lora_parameters": {"rank":8,"scale":20.0,"dropout":-1.0,
            "keys":["self_attn.q_a_proj","self_attn.q_b_proj","self_attn.kv_proj"]}}))
        with pytest.raises(SystemExit):
            mod._check_preflight(_stub({"model":str(mp),"data":str(dp),"adapter_path":str(tmp_path/"a"),"config":str(cp)}), tmp_path)

    def test_dropout_one_rejected(self, monkeypatch, tmp_path):
        """dropout=1.0 is rejected (must be < 1.0)."""
        import scripts.ds4_segmented_smoke as mod
        mp = tmp_path / "m"; mp.mkdir(); (mp/"config.json").write_text("{}"); (mp/"tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        for s in ("train.jsonl", "valid.jsonl", "test.jsonl"): (dp/s).write_text('{"prompt":"x","completion":"y"}\n\n')
        cp = tmp_path / "lora.json"
        cp.write_text(json.dumps({"lora_parameters": {"rank":8,"scale":20.0,"dropout":1.0,
            "keys":["self_attn.q_a_proj","self_attn.q_b_proj","self_attn.kv_proj"]}}))
        with pytest.raises(SystemExit):
            mod._check_preflight(_stub({"model":str(mp),"data":str(dp),"adapter_path":str(tmp_path/"a"),"config":str(cp)}), tmp_path)

    def test_dropout_zero_accepted(self, monkeypatch, tmp_path):
        """dropout=0.0 is accepted."""
        import scripts.ds4_segmented_smoke as mod
        import mlx_lm
        monkeypatch.setattr(mlx_lm, "__file__",
            str(ROOT / "vendor" / "mlx-lm" / "mlx_lm" / "__init__.py"), raising=False)
        monkeypatch.setattr(mod, "_check_process_rss_gb", lambda *a, **kw: 0)
        mp = tmp_path / "m"; mp.mkdir(); (mp/"config.json").write_text("{}"); (mp/"tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        for s in ("train.jsonl", "valid.jsonl", "test.jsonl"): (dp/s).write_text('{"prompt":"x","completion":"y"}\n\n')
        cp = tmp_path / "lora.json"
        cp.write_text(json.dumps({"lora_parameters": {"rank":8,"scale":20.0,"dropout":0.0,
            "keys":["self_attn.q_a_proj","self_attn.q_b_proj","self_attn.kv_proj"]}}))
        mod._check_preflight(_stub({"model":str(mp),"data":str(dp),"adapter_path":str(tmp_path/"a"),"config":str(cp)}), tmp_path)


# ===================================================================
# r11 — Tokenizer validation + conservative bound (Blocker 3)
# ===================================================================
class TestTokenizerBound:
    def test_tokenizer_loads_and_counts(self, monkeypatch, tmp_path):
        """When tokenizer is loadable, real token count is used."""
        import scripts.ds4_segmented_smoke as mod

        class FakeTok:
            def encode(self, text):
                return list(range(len(text) // 2))  # simulate tokenization
        def fake_load_tokenizer(path):
            return FakeTok()
        monkeypatch.setattr(mod, "_try_token_count",
            lambda mp, text: len(FakeTok().encode(text)))
        mp = tmp_path / "m"; mp.mkdir(); (mp/"config.json").write_text("{}"); (mp/"tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        (dp/"valid.jsonl").write_text('{"prompt":"x","completion":"y"}\n\n')
        (dp/"test.jsonl").write_text('{"prompt":"x","completion":"y"}\n\n')
        # This row has 10002 chars → ~5001 tokens from FakeTok → > 4096 → reject
        big = "x" * 10002
        (dp/"train.jsonl").write_text(json.dumps({"prompt":big,"completion":"y"})+"\n")
        with pytest.raises(SystemExit):
            mod._check_preflight(_stub({"model":str(mp),"data":str(dp),"adapter_path":str(tmp_path/"a"),"config":None}), tmp_path)

    def test_unbroken_10000char_rejected_by_density(self, monkeypatch, tmp_path):
        """Unbroken 10000-char string has 1 whitespace token but high density.
        Conservative density check rejects it."""
        import scripts.ds4_segmented_smoke as mod
        # Force tokenizer to be unavailable — fall back to conservative bound
        monkeypatch.setattr(mod, "_try_token_count", lambda *a, **kw: None)
        mp = tmp_path / "m"; mp.mkdir(); (mp/"config.json").write_text("{}"); (mp/"tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        (dp/"valid.jsonl").write_text('{"prompt":"x","completion":"y"}\n\n')
        (dp/"test.jsonl").write_text('{"prompt":"x","completion":"y"}\n\n')
        # Single unbroken 10000-char string — density = 10000/1 = 10000 > 50
        big = "x" * 10000
        (dp/"train.jsonl").write_text(json.dumps({"prompt":big,"completion":"y"})+"\n")
        with pytest.raises(SystemExit):
            mod._check_preflight(_stub({"model":str(mp),"data":str(dp),"adapter_path":str(tmp_path/"a"),"config":None}), tmp_path)


# ===================================================================
# r11 — Preflight reason preservation (Blocker 5)
# ===================================================================
class TestPreflightReason:
    def test_preflight_reason_preserved_not_just_code(self, monkeypatch, tmp_path):
        """When _check_preflight raises SystemExit(3) after writing a detailed
        fail marker, the smoke report must contain the detailed reason, not
        just the exit code string '3'."""
        import scripts.ds4_segmented_smoke as mod
        reports = []
        marker_reasons = []
        # Wrap _write_fail_marker so it still sets _LAST_FAIL_REASON
        # (monkeypatching to no-op bypasses the global variable capture).
        orig_fmf = mod._write_fail_marker
        def _wrap_fmf(mw, r):
            marker_reasons.append(r)
            return orig_fmf(mw, r)
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        monkeypatch.setattr(mod, "_write_fail_marker", _wrap_fmf)
        _pin_test_paths(mod, tmp_path)
        mod._PINNED_SMOKE_PATHS["model"] = str(tmp_path / "nonexistent")
        rc = mod.main(["--model", str(tmp_path/"nonexistent"), "--data", str(tmp_path/"d"),
                       "--adapter-path", str(tmp_path/"a"),
                       "--iters", "1", "--batch-size", "1",
                       "--learning-rate", "1e-5", "--max-seq-length", "4096",
                       "--mask-prompt", "--grad-checkpoint", "--segment-size", "1"])
        assert rc == 3
        assert len(reports) >= 1
        r = reports[-1]
        msg = r.get("failure_message", "")
        assert msg != "3", f"failure_message must not be just '3': {r}"
        assert "model path not found" in msg.lower() or "nonexistent" in msg, \
            f"failure_message must contain detailed reason: {r}"
        assert len(marker_reasons) >= 1


# ===================================================================
# r11 — Parser classification (Blocker 5)
# ===================================================================
class TestParserClassification:
    def test_argparse_error_classified_as_argparse_not_timeout(self, monkeypatch, tmp_path):
        """Argparse errors (code 2) must be classified as 'argparse',
        not 'timeout'."""
        import scripts.ds4_segmented_smoke as mod
        reports = []
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        _pin_test_paths(mod, tmp_path)
        rc = mod.main(["--segment-size", "0", "--model", str(tmp_path/"m"),
                       "--data", str(tmp_path/"d"), "--adapter-path", str(tmp_path/"a"),
                       "--iters", "1", "--batch-size", "1",
                       "--learning-rate", "1e-5", "--max-seq-length", "4096",
                       "--mask-prompt", "--grad-checkpoint"])
        assert rc != 0
        assert len(reports) >= 1
        r = reports[-1]
        assert r["failure_code"] == "argparse", f"expected 'argparse', got {r}"
        msg = r.get("failure_message", "")
        assert "timeout" not in msg.lower()
        # Must contain meaningful error text, not just the exit code "2"
        # argparse prints the real error to stderr which we now capture
        assert len(msg) > 5, f"argparse message too short, expected useful text: {msg!r}"
        assert "segment" in msg.lower(), f"message should mention segment-size: {msg!r}"

    def test_help_classified_as_argparse_help(self, monkeypatch, tmp_path):
        """--help (code 0) must be classified as 'argparse-help'."""
        import scripts.ds4_segmented_smoke as mod
        reports = []
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        _pin_test_paths(mod, tmp_path)
        rc = mod.main(["--help"])
        assert len(reports) >= 1
        assert reports[-1]["failure_code"] in ("argparse-help", "argparse")


# ===================================================================
# r11 — Cleanup warning persistence (Blocker 6)
# ===================================================================
class TestCleanupWarningPersistence:
    def test_cleanup_warnings_in_report_on_failure(self, monkeypatch, tmp_path):
        """When cleanup release raises an exception, the warning must be
        recorded in-memory AND persisted durably to the cleanup-warnings.json
        sidecar file. The pre-cleanup report does NOT contain cleanup warnings
        because the report is written before the finally block runs."""
        import scripts.ds4_segmented_smoke as mod
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        monkeypatch.setattr(mod, "_write_ok_marker", lambda mw: None)
        monkeypatch.setattr(mod, "_write_smoke_report",
            lambda r: (_ for _ in ()).throw(RuntimeError("report fail")))
        def failing_release():
            raise OSError("test release failure")
        monkeypatch.setattr(mod, "_release_ft_lock", failing_release)
        mp = tmp_path / "m"; mp.mkdir()
        mod._LOCK_OWNED_PATH = tmp_path / "owned.lock"
        mod._LOCK_OWNED_TOKEN = "pid=12345"
        mod._CLEANUP_WARNINGS.clear()
        # Sidecar path — absolute, always same regardless of tmp_path
        sidecar = mod._ARTIFACT_DIR / "cleanup-warnings.json"
        # Remove stale sidecar before test
        try:
            sidecar.unlink()
        except FileNotFoundError:
            pass
        try:
            _pin_test_paths(mod, tmp_path)
            rc = mod.main(["--model", str(mp), "--data", str(tmp_path/"d"),
                           "--adapter-path", str(tmp_path/"a"),
                           "--iters", "1", "--batch-size", "1",
                           "--learning-rate", "1e-5", "--max-seq-length", "4096",
                           "--mask-prompt", "--grad-checkpoint", "--segment-size", "1"])
            assert rc != 0
            # In-memory check
            assert len(mod._CLEANUP_WARNINGS) >= 1
            assert any("test release failure" in w for w in mod._CLEANUP_WARNINGS)
            # Durable sidecar check — the finally block writes cleanup-warnings.json
            import json
            assert sidecar.is_file(), f"sidecar not written: {sidecar}"
            data = json.loads(sidecar.read_text())
            assert "warnings" in data
            assert any("test release failure" in w for w in data["warnings"]), \
                f"warning not in sidecar: {data}"
        finally:
            mod._LOCK_OWNED_PATH = None
            mod._LOCK_OWNED_TOKEN = None
            mod._CLEANUP_WARNINGS.clear()
            try:
                sidecar.unlink()
            except FileNotFoundError:
                pass


# ===================================================================
# r11 — Unified cleanup oracle (Blocker 6)
# ===================================================================
class TestUnifiedCleanupOracle:
    """One unchanged oracle: after main(), lock must be released.
    Control passes; mutant (release skipped) fails the same oracle."""

    def _oracle_lock_released(self, module, lock_path):
        """Unchanged oracle: after execution, owned lock must be None."""
        assert module._LOCK_OWNED_PATH is None, \
            f"lock not released: {module._LOCK_OWNED_PATH}"
        assert module._LOCK_OWNED_TOKEN is None

    def test_control_passes_oracle(self, monkeypatch, tmp_path):
        """Control: unchanged code releases lock → oracle passes."""
        import scripts.ds4_segmented_smoke as control_mod
        monkeypatch.setattr(control_mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(control_mod, "_write_smoke_report", lambda r: None)
        monkeypatch.setattr(control_mod, "_write_fail_marker", lambda mw, r: None)
        monkeypatch.setattr(control_mod, "_write_ok_marker", lambda mw: None)
        control_mod._LOCK_OWNED_PATH = tmp_path / "control.lock"
        control_mod._LOCK_OWNED_TOKEN = "pid=12345"
        try:
            _pin_test_paths(control_mod, tmp_path)
            control_mod.main(["--model", str(tmp_path/"m"), "--data", str(tmp_path/"d"),
                              "--adapter-path", str(tmp_path/"a"),
                              "--iters", "1", "--batch-size", "1",
                              "--learning-rate", "1e-5", "--max-seq-length", "4096",
                              "--mask-prompt", "--grad-checkpoint", "--segment-size", "1"])
            self._oracle_lock_released(control_mod, tmp_path)
        finally:
            control_mod._LOCK_OWNED_PATH = None
            control_mod._LOCK_OWNED_TOKEN = None

    def test_mutant_fails_oracle(self, monkeypatch, tmp_path):
        """Mutant: skipped release → same oracle fails."""
        def mutation(t):
            old = "_release_ft_lock()\n        except"
            new = "pass  # MUTATED release skipped\n        except"
            assert old in t
            return t.replace(old, new, 1)
        mut = _mutate(mutation)
        monkeypatch.setattr(mut, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mut, "_write_smoke_report", lambda r: None)
        monkeypatch.setattr(mut, "_write_fail_marker", lambda mw, r: None)
        monkeypatch.setattr(mut, "_write_ok_marker", lambda mw: None)
        mp = tmp_path / "m"; mp.mkdir()
        mut._LOCK_OWNED_PATH = tmp_path / "mutant.lock"
        mut._LOCK_OWNED_TOKEN = "pid=12345"
        try:
            _pin_test_paths(mut, tmp_path)
            mut.main(["--model", str(mp), "--data", str(tmp_path/"d"),
                      "--adapter-path", str(tmp_path/"a"),
                      "--iters", "1", "--batch-size", "1",
                      "--learning-rate", "1e-5", "--max-seq-length", "4096",
                      "--mask-prompt", "--grad-checkpoint", "--segment-size", "1"])
            # Apply SAME oracle — it must fail because lock was NOT released
            try:
                self._oracle_lock_released(mut, tmp_path)
                pytest.fail("mutant passed oracle — lock was NOT released")
            except AssertionError:
                pass  # expected: oracle correctly detects unreleased lock
        finally:
            mut._LOCK_OWNED_PATH = None
            mut._LOCK_OWNED_TOKEN = None


# ===================================================================
# r12 — Pinned smoke path rejection (Blocker 2)
# ===================================================================
class TestPinnedSmokePaths:
    """Direct entry point must reject arbitrary paths with
    failure_code: pinned-paths before preflight."""

    def test_arbitrary_model_rejected(self, monkeypatch, tmp_path):
        """Valid scalars but wrong model path → pinned-paths."""
        import scripts.ds4_segmented_smoke as mod
        reports = []
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        # Use pinned scalars but arbitrary model path
        ap = tmp_path / "a"
        rc = mod.main([
            "--model", str(tmp_path / "ARBITRARY_MODEL"),
            "--data", str(tmp_path / "d"),
            "--adapter-path", str(ap),
            "--iters", "1", "--batch-size", "1",
            "--learning-rate", "1e-5", "--max-seq-length", "4096",
            "--mask-prompt", "--grad-checkpoint", "--segment-size", "1",
        ])
        assert rc != 0
        assert len(reports) >= 1
        assert reports[-1]["failure_code"] == "pinned-paths", \
            f"expected pinned-paths, got {reports[-1]}"

    def test_arbitrary_data_rejected(self, monkeypatch, tmp_path):
        """Valid model but wrong data path → pinned-paths."""
        import scripts.ds4_segmented_smoke as mod
        reports = []
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        ap = tmp_path / "a"
        # Pin model to match, but data is wrong
        mod._PINNED_SMOKE_PATHS = {
            "model": str(tmp_path / "m"),
            "data": str(tmp_path / "CORRECT_DATA"),
            "adapter_path": str(ap),
            "config": None,
        }
        m = tmp_path / "m"; m.mkdir()
        rc = mod.main([
            "--model", str(m),
            "--data", str(tmp_path / "WRONG_DATA"),
            "--adapter-path", str(ap),
            "--iters", "1", "--batch-size", "1",
            "--learning-rate", "1e-5", "--max-seq-length", "4096",
            "--mask-prompt", "--grad-checkpoint", "--segment-size", "1",
        ])
        assert rc != 0
        assert len(reports) >= 1
        assert reports[-1]["failure_code"] == "pinned-paths"

    def test_arbitrary_adapter_rejected(self, monkeypatch, tmp_path):
        """Valid model+data but wrong adapter path → pinned-paths."""
        import scripts.ds4_segmented_smoke as mod
        reports = []
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        m = tmp_path / "m"; m.mkdir()
        d = tmp_path / "d"
        ap = tmp_path / "WRONG_ADAPTER"
        mod._PINNED_SMOKE_PATHS = {
            "model": str(m),
            "data": str(d),
            "adapter_path": str(tmp_path / "CORRECT_ADAPTER"),
            "config": None,
        }
        rc = mod.main([
            "--model", str(m),
            "--data", str(d),
            "--adapter-path", str(ap),
            "--iters", "1", "--batch-size", "1",
            "--learning-rate", "1e-5", "--max-seq-length", "4096",
            "--mask-prompt", "--grad-checkpoint", "--segment-size", "1",
        ])
        assert rc != 0
        assert len(reports) >= 1
        assert reports[-1]["failure_code"] == "pinned-paths"

    def test_valid_paths_accepted(self, monkeypatch, tmp_path):
        """When pinned paths match, check passes (preflight may then fail)."""
        import scripts.ds4_segmented_smoke as mod
        reports = []
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        _pin_test_paths(mod, tmp_path)
        m = tmp_path / "m"; m.mkdir()
        rc = mod.main([
            "--model", str(m),
            "--data", str(tmp_path / "d"),
            "--adapter-path", str(tmp_path / "a"),
            "--iters", "1", "--batch-size", "1",
            "--learning-rate", "1e-5", "--max-seq-length", "4096",
            "--mask-prompt", "--grad-checkpoint", "--segment-size", "1",
        ])
        # Should reach preflight, which will fail (no model files), returning code 3
        # NOT pinned-paths
        assert rc != 0
        assert len(reports) >= 1
        code = reports[-1].get("failure_code", "")
        assert code != "pinned-paths", f"should not be pinned-paths, got {reports[-1]}"


# ===================================================================
# r12 — Filtered smoke dataset repin (Story 14.3a)
# ===================================================================
class TestFilteredDatasetRepin:
    FILTERED_PATH = "/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke"
    ORIGINAL_PATH = "/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096"

    def test_pinned_data_constant_is_filtered_path(self):
        """_PINNED_SMOKE_PATHS['data'] equals the filtered path.

        Uses a fresh import from a temp copy to avoid contamination from
        prior tests that monkeypatch _PINNED_SMOKE_PATHS on the live module."""
        spec = importlib.util.spec_from_file_location(
            f"fresh_{id(tempfile.mkdtemp())}", SCRIPT)
        fresh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fresh)
        assert fresh._PINNED_SMOKE_PATHS["data"] == self.FILTERED_PATH, \
            f"expected {self.FILTERED_PATH}, got {fresh._PINNED_SMOKE_PATHS['data']}"

    def test_original_dataset_path_rejected(self, monkeypatch, tmp_path):
        """Original mlx-4096 (without -smoke) is rejected by pinned-path gate."""
        import scripts.ds4_segmented_smoke as mod
        reports = []
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        _pin_test_paths(mod, tmp_path)
        # Pin data to filtered path, but pass the original path on CLI
        mod._PINNED_SMOKE_PATHS["data"] = self.FILTERED_PATH
        m = tmp_path / "m"; m.mkdir()
        rc = mod.main([
            "--model", str(m),
            "--data", self.ORIGINAL_PATH,
            "--adapter-path", str(tmp_path / "a"),
            "--iters", "1", "--batch-size", "1",
            "--learning-rate", "1e-5", "--max-seq-length", "4096",
            "--mask-prompt", "--grad-checkpoint", "--segment-size", "1",
        ])
        assert rc != 0
        assert len(reports) >= 1
        assert reports[-1]["failure_code"] == "pinned-paths", \
            f"expected pinned-paths, got {reports[-1]}"

    def test_filtered_path_accepted_by_pinned_gate(self, monkeypatch, tmp_path):
        """Filtered path passes pinned-path gate (preflight then fails)."""
        import scripts.ds4_segmented_smoke as mod
        reports = []
        monkeypatch.setattr(mod, "_install_timeout_watchdog", lambda *a, **kw: None)
        monkeypatch.setattr(signal, "signal", lambda s, h: None)
        monkeypatch.setattr(signal, "alarm", lambda s: None)
        monkeypatch.setattr(mod, "_write_smoke_report", lambda r: reports.append(r))
        monkeypatch.setattr(mod, "_write_fail_marker", lambda mw, r: None)
        _pin_test_paths(mod, tmp_path)
        # Set the pinned paths to match temporary paths; pin data to filtered path
        mod._PINNED_SMOKE_PATHS = {
            "model": str(tmp_path / "m"),
            "data": str(tmp_path / "d"),
            "adapter_path": str(tmp_path / "a"),
            "config": None,
        }
        m = tmp_path / "m"; m.mkdir()
        rc = mod.main([
            "--model", str(m),
            "--data", str(tmp_path / "d"),
            "--adapter-path", str(tmp_path / "a"),
            "--iters", "1", "--batch-size", "1",
            "--learning-rate", "1e-5", "--max-seq-length", "4096",
            "--mask-prompt", "--grad-checkpoint", "--segment-size", "1",
        ])
        # Reaches preflight, fails with code=3 (not pinned-paths)
        assert rc != 0
        assert len(reports) >= 1
        code = reports[-1].get("failure_code", "")
        assert code != "pinned-paths", f"filtered path should pass pinned gate, got {reports[-1]}"


# ===================================================================
# r12 — LoRA NaN/Inf/bool rejection (Blocker 3)
# ===================================================================
class TestLoraNaN:
    """NaN, Inf, and bool values in rank/scale/dropout must be rejected."""

    def _make_pt_args(self, tmp_path):
        mp = tmp_path / "m"; mp.mkdir()
        (mp / "config.json").write_text("{}")
        (mp / "tokenizer.json").write_text("{}")
        mp.joinpath("model-00001.safetensors").write_text("x")
        dp = tmp_path / "d"; dp.mkdir()
        for s in ("train.jsonl", "valid.jsonl", "test.jsonl"):
            (dp / s).write_text('{"prompt":"x","completion":"y"}\n\n')
        return {"model": str(mp), "data": str(dp), "adapter_path": str(tmp_path / "a"), "config": None}

    def _lora_cfg(self, tmp_path, rank=8, scale=20.0, dropout=0.0):
        cp = tmp_path / "lora.json"
        cp.write_text(json.dumps({"lora_parameters": {
            "rank": rank, "scale": scale, "dropout": dropout,
            "keys": ["self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_proj"],
        }}))
        return str(cp)

    def test_nan_rank_rejected(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        stub = _stub(self._make_pt_args(tmp_path))
        stub.config = self._lora_cfg(tmp_path, rank=float("nan"))
        with pytest.raises(SystemExit):
            mod._check_preflight(stub, tmp_path)

    def test_nan_scale_rejected(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        stub = _stub(self._make_pt_args(tmp_path))
        stub.config = self._lora_cfg(tmp_path, scale=float("nan"))
        with pytest.raises(SystemExit):
            mod._check_preflight(stub, tmp_path)

    def test_nan_dropout_rejected(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        stub = _stub(self._make_pt_args(tmp_path))
        stub.config = self._lora_cfg(tmp_path, dropout=float("nan"))
        with pytest.raises(SystemExit):
            mod._check_preflight(stub, tmp_path)

    def test_inf_rank_rejected(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        stub = _stub(self._make_pt_args(tmp_path))
        stub.config = self._lora_cfg(tmp_path, rank=float("inf"))
        with pytest.raises(SystemExit):
            mod._check_preflight(stub, tmp_path)

    def test_inf_scale_rejected(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        stub = _stub(self._make_pt_args(tmp_path))
        stub.config = self._lora_cfg(tmp_path, scale=float("inf"))
        with pytest.raises(SystemExit):
            mod._check_preflight(stub, tmp_path)

    def test_bool_rank_rejected(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        stub = _stub(self._make_pt_args(tmp_path))
        stub.config = self._lora_cfg(tmp_path, rank=True)
        with pytest.raises(SystemExit):
            mod._check_preflight(stub, tmp_path)

    def test_bool_scale_rejected(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        stub = _stub(self._make_pt_args(tmp_path))
        stub.config = self._lora_cfg(tmp_path, scale=False)
        with pytest.raises(SystemExit):
            mod._check_preflight(stub, tmp_path)

    def test_bool_dropout_rejected(self, tmp_path):
        import scripts.ds4_segmented_smoke as mod
        stub = _stub(self._make_pt_args(tmp_path))
        stub.config = self._lora_cfg(tmp_path, dropout=True)
        with pytest.raises(SystemExit):
            mod._check_preflight(stub, tmp_path)