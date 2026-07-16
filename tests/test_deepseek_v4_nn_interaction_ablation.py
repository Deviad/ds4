"""Story 13.3b-5h diagnostic-only interaction-ablation tests."""

from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_HELPER = _ROOT / "tests" / "helpers" / "routed_fp4_multilayer_peak_probe.py"
_FIXTURE = _ROOT / "tests" / "fixtures" / "deepseek_v4_nn_multilayer_peak_topology.json"
_REPORT = _ROOT / "agent-output" / "cmux-13-3b" / "interaction-ablation-report.json"

_REQUIRED_ROW_FIELDS = {
    "case", "depth", "experts", "nonempty_experts", "tokens", "hidden", "intermediate", "top_k",
    "hc_mult", "compression_ratio", "checkpoint", "component_mask", "active_baseline",
    "active_after_graph", "pre_boundary_active", "post_boundary_active", "peak_forward", "peak_backward",
    "active_final", "cache_final", "custom_kernel_nodes", "duration_s", "finite_loss", "finite_gradients",
    "exit_code", "signal", "error", "loss", "forward_exact", "forward_max_abs_diff",
    "materialized_loss_abs_diff", "materialized_lora_grad_max_abs_diff", "materialized_lora_grad_max_rel_diff",
    "materialized_input_grad_max_abs_diff", "materialized_input_grad_max_rel_diff", "materialized_allclose_atol_rtol_1e_5",
}
_EVIDENCE_FIELDS = {
    "ordinal", "cell_identity_sha256", "child_pid", "child_nonce", "started_monotonic_ns",
    "finished_monotonic_ns", "trainable_key_count", "trainable_keys_sha256", "input_gradient_shape",
    "input_gradient_nonzero", "measured_custom_kernel_nodes",
}


def _helper_module():
    spec = importlib.util.spec_from_file_location("routed_fp4_multilayer_peak_probe", _HELPER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_helper_snippet(body: str, timeout: int = 220) -> dict:
    code = f"""
import importlib.util
import json
from pathlib import Path
HELPER = Path({str(_HELPER)!r})
spec = importlib.util.spec_from_file_location("routed_fp4_multilayer_peak_probe", HELPER)
assert spec and spec.loader
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
{textwrap.dedent(body)}
"""
    cp = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, text=True, capture_output=True, timeout=timeout)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    return json.loads(cp.stdout.splitlines()[-1])


def _load_report() -> dict:
    assert _REPORT.is_file(), "run routed_fp4_multilayer_peak_probe.py interaction-run to create the 5h report"
    return json.loads(_REPORT.read_text())


def test_interaction_ablation_helper_test_fixture_and_report_are_tracked():
    paths = [_HELPER, _FIXTURE, Path(__file__), _REPORT]
    for path in paths:
        assert path.is_file(), path
    tracked = subprocess.check_output(
        ["git", "ls-files", *[str(p.relative_to(_ROOT)) for p in paths]],
        text=True,
        cwd=_ROOT,
    ).splitlines()
    assert set(tracked) == {str(p.relative_to(_ROOT)) for p in paths}


def test_interaction_ablation_plan_exact_25_rows_order_thresholds_and_identity():
    h = _helper_module()
    plan = json.loads(subprocess.check_output([sys.executable, str(_HELPER), "interaction-plan"], text=True, cwd=_ROOT))
    cells = plan["cells"]
    assert plan["story"] == "13.3b-5h"
    assert plan["memory_limit"] == 8_000_000_000
    assert plan["timeout_s"] <= 180
    assert plan["real_assets_accessed"] is False
    assert plan["classification_thresholds"] == {"collapsed_max_delta": 227_479_736, "persistent_min_delta": 941_632_822}
    assert len(cells) == 25
    assert plan["planned_cell_list_sha256"] == h.interaction_plan_identity_hash(cells)
    assert plan["cell_identity_sha256"] == [h.interaction_cell_identity(c) for c in cells]
    assert len(set(plan["cell_identity_sha256"])) == 25
    expected = []
    expected += [("composed_checkpoint_on", d, True) for d in (1, 2, 4, 8, 16, 43)]
    expected += [("composed_checkpoint_off", d, False) for d in (1, 4, 8)]
    expected += [("attention_forward_routed_vjp", d, True) for d in (1, 8, 16, 43)]
    expected += [("routed_forward_attention_vjp", d, True) for d in (1, 8, 16, 43)]
    expected += [("materialized_attention_routed_boundary", d, True) for d in (1, 8, 16, 43)]
    expected += [("composed_layer_sequential", d, True) for d in (1, 8, 16, 43)]
    observed = [(c["case"], c["depth"], c["checkpoint"]) for c in cells]
    assert observed == expected
    for cell in cells:
        assert cell | {
            "experts": 8, "nonempty_experts": 8, "tokens": 64, "hidden": 128, "intermediate": 64,
            "top_k": 2, "hc_mult": 4, "compression_ratio": 4, "component_mask": "no_shared",
        } == cell
        assert "/Volumes/" not in json.dumps(cell)


def test_interaction_ablation_active_baseline_is_pinned_immediately_before_vjp_graph():
    h = _helper_module()
    for fn_name in ("_run_interaction_one_graph", "_run_interaction_materialized"):
        source = inspect.getsource(getattr(h, fn_name))
        vjp_pos = source.index("value, grads = mx.value_and_grad")
        baseline_pos = source.rfind('row["active_baseline"] = mx.get_active_memory()', 0, vjp_pos)
        assert baseline_pos > source.index('row["peak_forward"] = mx.get_peak_memory()')
        assert "interaction_forward_once" not in source[baseline_pos:vjp_pos]
    sequential = inspect.getsource(h._run_interaction_sequential)
    loop_pos = sequential.index("for layer in model.model.pipeline_layers:")
    assert 'row["active_baseline"] = mx.get_active_memory()' not in sequential[:loop_pos]
    assert "stage_baselines" in sequential
    assert "baseline_for_peak_b" in sequential


def test_interaction_ablation_materialized_parity_uses_elementwise_allclose_counterexample():
    result = _run_helper_snippet(
        """
mx = h._setup_mlx()
a = mx.array([1.5e-5, 1.0], dtype=mx.float32)
b = mx.array([0.0, 1.0], dtype=mx.float32)
array_abs, array_rel, array_allclose = h._max_array_diff(a, b, mx)
grad_abs, grad_rel, keys_equal, grad_allclose, keys, key_hash, mismatches = h._max_grad_diff({"leaf": a}, {"leaf": b}, mx)
print(json.dumps({
    "array_abs": array_abs,
    "array_rel": array_rel,
    "array_allclose": array_allclose,
    "grad_abs": grad_abs,
    "grad_rel": grad_rel,
    "keys_equal": keys_equal,
    "grad_allclose": grad_allclose,
    "mismatches": mismatches,
}))
"""
    )
    assert result["array_abs"] == pytest.approx(1.5e-5)
    assert result["array_allclose"] is False
    assert result["keys_equal"] is True
    assert result["mismatches"] == []
    assert result["grad_allclose"] is False


def test_interaction_ablation_vjp_legs_are_gradient_partition_sensitive():
    result = _run_helper_snippet(
        """
import copy
mx = h._setup_mlx()
from mlx_lm.tuner.trainer import grad_checkpoint
cells = {c["case"]: c for c in h.interaction_plan_cells() if c["depth"] == 1}
out = {}
for mode in ("attention_forward_routed_vjp", "routed_forward_attention_vjp"):
    model, keys, key_hash, shapes, ids, input_probe = h._make_interaction_model(copy.deepcopy(cells[mode]), mx)
    grad_checkpoint(model.layers[0])
    h._materialize_resident(mx, h._tree_values(model.parameters()), ids, input_probe)
    def loss_for(params, probe):
        model.update(params)
        y = h.interaction_forward_once(model, ids, probe, mode, {})
        return mx.mean(y * y) + h._zero_param_anchor(params, mx)
    value, grads = mx.value_and_grad(loss_for, argnums=(0, 1))(model.trainable_parameters(), input_probe)
    lora_grads, input_grad = grads
    grad_vals = h._tree_values(lora_grads)
    nodes = h._count_custom_kernel_nodes(mx, [value, *grad_vals, input_grad])
    mx.eval(value, *grad_vals, input_grad)
    per_key = {
        k: float(mx.max(mx.abs(v.astype(mx.float32))).item())
        for k, v in dict(h._tree_flat(lora_grads)).items()
    }
    out[mode] = {
        "nodes": nodes,
        "loss": float(value.item()),
        "input_grad_max_abs": float(mx.max(mx.abs(input_grad.astype(mx.float32))).item()),
        "per_key_lora_grad_max_abs": per_key,
    }
print(json.dumps(out))
"""
    )
    routed_vjp = result["attention_forward_routed_vjp"]
    attention_vjp = result["routed_forward_attention_vjp"]
    assert routed_vjp["nodes"] == 12
    assert routed_vjp["input_grad_max_abs"] > 0.0
    assert all(v == 0.0 for v in routed_vjp["per_key_lora_grad_max_abs"].values())
    assert attention_vjp["nodes"] == 4
    assert attention_vjp["input_grad_max_abs"] > 0.0
    assert any(v > 0.0 for v in attention_vjp["per_key_lora_grad_max_abs"].values())


def test_interaction_ablation_runtime_control_semantics_independent_oracles():
    result = _run_helper_snippet(
        """
import copy
cells = {c["case"]: c for c in h.interaction_plan_cells() if c["depth"] == 1}
out = {}
for name in ("composed_checkpoint_on", "attention_forward_routed_vjp", "routed_forward_attention_vjp", "materialized_attention_routed_boundary", "composed_layer_sequential"):
    row = h.run_interaction_cell(copy.deepcopy(cells[name]))
    out[name] = {
        "error": row.get("error"),
        "finite_loss": row.get("finite_loss"),
        "finite_gradients": row.get("finite_gradients"),
        "forward_exact": row.get("forward_exact"),
        "forward_max_abs_diff": row.get("forward_max_abs_diff"),
        "input_gradient_nonzero": row.get("_cell_evidence", {}).get("input_gradient_nonzero"),
        "trainable_key_count": row.get("_cell_evidence", {}).get("trainable_key_count"),
        "nodes": row.get("custom_kernel_nodes"),
        "materialized_allclose": row.get("materialized_allclose_atol_rtol_1e_5"),
        "materialized_lora_grad_max_abs_diff": row.get("materialized_lora_grad_max_abs_diff"),
        "materialized_input_grad_max_abs_diff": row.get("materialized_input_grad_max_abs_diff"),
    }
print(json.dumps(out))
"""
    )
    for metrics in result.values():
        assert metrics["error"] is None
        assert metrics["finite_loss"] is True
        assert metrics["finite_gradients"] is True
        assert metrics["input_gradient_nonzero"] is True
        assert metrics["trainable_key_count"] == 6
        assert metrics["nodes"] is not None and metrics["nodes"] > 0
    assert result["attention_forward_routed_vjp"]["forward_exact"] is True
    assert result["attention_forward_routed_vjp"]["forward_max_abs_diff"] == 0.0
    assert result["routed_forward_attention_vjp"]["forward_exact"] is True
    assert result["routed_forward_attention_vjp"]["forward_max_abs_diff"] == 0.0
    assert result["materialized_attention_routed_boundary"]["materialized_allclose"] is True
    assert result["materialized_attention_routed_boundary"]["materialized_lora_grad_max_abs_diff"] <= 1e-5
    assert result["materialized_attention_routed_boundary"]["materialized_input_grad_max_abs_diff"] <= 1e-5


def test_interaction_ablation_report_schema_identity_seriality_and_complete_outcomes():
    h = _helper_module()
    report = _load_report()
    metadata = report["metadata"]
    rows = report["rows"]
    evidence = report["cell_evidence"]
    assert metadata["story"] == "13.3b-5h"
    assert metadata["memory_limit"] == 8_000_000_000
    assert metadata["timeout_s"] <= 180
    assert metadata["execution_mode"] == "serial_fresh_subprocess_per_cell"
    assert metadata["max_concurrency"] == 1
    assert metadata["real_assets_accessed"] is False
    assert metadata["classification"] == "not-classified-by-coder"
    assert metadata["planned_cell_count"] == metadata["executed_cell_count"] == 25
    assert metadata["classification_thresholds"] == {"collapsed_max_delta": 227_479_736, "persistent_min_delta": 941_632_822}
    assert metadata["planned_cell_list_sha256"] == metadata["executed_cell_list_sha256"]
    assert metadata["fixture"]["fixture_sha256"] == h.sha256_path(_FIXTURE)
    assert metadata["fixture"]["source_hashes"] == h.validate_topology_fixture(_FIXTURE)["source_hashes"]
    assert len(rows) == len(evidence) == 25
    assert [(r["case"], r["depth"], r["checkpoint"]) for r in rows] == [(c["case"], c["depth"], c["checkpoint"]) for c in h.interaction_plan_cells()]

    nonces = set()
    previous_finish = None
    for ordinal, (row, ev) in enumerate(zip(rows, evidence)):
        assert set(row) == _REQUIRED_ROW_FIELDS
        assert set(ev) == _EVIDENCE_FIELDS
        assert ev["ordinal"] == ordinal
        assert ev["cell_identity_sha256"] == h.interaction_cell_identity(row)
        assert ev["cell_identity_sha256"] == metadata["planned_cell_identity_sha256"][ordinal]
        assert ev["child_pid"] != metadata["parent_pid"]
        assert ev["child_nonce"] not in nonces
        nonces.add(ev["child_nonce"])
        assert ev["started_monotonic_ns"] < ev["finished_monotonic_ns"]
        if previous_finish is not None:
            assert ev["started_monotonic_ns"] >= previous_finish
        previous_finish = ev["finished_monotonic_ns"]
        assert row["exit_code"] == 0
        assert row["signal"] is None
        assert row["error"] is None
        assert row["finite_loss"] is True
        assert row["finite_gradients"] is True
        assert row["active_baseline"] is not None
        assert row["active_after_graph"] is not None
        assert row["peak_forward"] is not None
        assert row["peak_backward"] is not None
        assert row["custom_kernel_nodes"] is not None
        assert row["loss"] is not None
        assert ev["trainable_key_count"] == 6 * min(row["depth"], 16)
        assert ev["input_gradient_shape"] == [1, 64, 4, 128]
        assert ev["input_gradient_nonzero"] is True
        assert ev["measured_custom_kernel_nodes"] == row["custom_kernel_nodes"]
        if row["case"] in {"attention_forward_routed_vjp", "routed_forward_attention_vjp"}:
            assert row["forward_exact"] is True
            assert row["forward_max_abs_diff"] == 0.0
        if row["case"] == "materialized_attention_routed_boundary":
            assert row["materialized_allclose_atol_rtol_1e_5"] is True
            assert row["materialized_lora_grad_max_abs_diff"] <= 1e-5
            assert row["materialized_input_grad_max_abs_diff"] <= 1e-5


def test_interaction_ablation_source_guards_and_no_classification_oracles():
    source = _HELPER.read_text()
    assert "INTERACTION_REPORT" in source
    assert "interaction_plan_cells" in source
    assert "run_interaction_cell" in source
    assert "nan_to_num" not in source
    assert "collapsed" not in source.partition("def interaction_plan")[2].partition("def")[0].lower() or "classification_thresholds" in source
    assert "delta <=" not in source
    assert "persistent" not in source.partition("def run_interaction_report")[2].lower()
