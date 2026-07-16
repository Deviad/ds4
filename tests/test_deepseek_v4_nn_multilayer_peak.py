"""Story 13.3b-5g diagnostic-only multi-layer peak report tests."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_HELPER = _ROOT / "tests" / "helpers" / "routed_fp4_multilayer_peak_probe.py"
_FIXTURE = _ROOT / "tests" / "fixtures" / "deepseek_v4_nn_multilayer_peak_topology.json"
_REPORT = _ROOT / "agent-output" / "cmux-13-3b" / "multilayer-peak-report.json"

_REQUIRED_FIELDS = {
    "case", "depth", "experts", "nonempty_experts", "tokens", "hidden", "intermediate", "top_k",
    "hc_mult", "compression_ratio", "checkpoint", "component_mask", "active_baseline",
    "active_after_graph", "peak_forward", "peak_backward", "active_final", "cache_final",
    "custom_kernel_nodes", "duration_s", "finite_loss", "finite_gradients", "exit_code", "signal", "error",
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


def _load_report() -> dict:
    assert _REPORT.is_file(), "run routed_fp4_multilayer_peak_probe.py to create the diagnostic report"
    return json.loads(_REPORT.read_text())


def _run_helper_snippet(body: str, timeout: int = 180) -> dict:
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


def _compile_eval_inside_compile_expectation(timeout: int = 30) -> dict:
    code = """
import json
import mlx.core as mx
x = mx.ones((2, 3, 16), dtype=mx.float32)
@mx.compile
def bad(x):
    y = mx.stop_gradient(x)
    mx.eval(y)
    return y
try:
    bad(x)
except ValueError as exc:
    print(json.dumps({"raised": True, "error": str(exc)}))
else:
    raise SystemExit("expected ValueError for eval inside compile")
"""
    cp = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, text=True, capture_output=True, timeout=timeout)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    return json.loads(cp.stdout.splitlines()[-1])


def test_multilayer_peak_helper_fixture_and_report_are_tracked():
    paths = [_HELPER, _FIXTURE, Path(__file__), _REPORT]
    for path in paths:
        assert path.is_file(), path
    tracked = subprocess.check_output(
        ["git", "ls-files", *[str(p.relative_to(_ROOT)) for p in paths]],
        text=True,
        cwd=_ROOT,
    ).splitlines()
    assert set(tracked) == {str(p.relative_to(_ROOT)) for p in paths}


def test_multilayer_peak_plan_exact_order_identity_and_no_real_assets():
    h = _helper_module()
    plan = json.loads(subprocess.check_output([sys.executable, str(_HELPER), "plan"], text=True, cwd=_ROOT))
    cells = plan["cells"]
    assert len(cells) == 64
    assert plan["memory_limit"] == 8_000_000_000
    assert plan["timeout_s"] <= 180
    assert plan["real_assets_accessed"] is False
    assert plan["planned_cell_list_sha256"] == h.plan_identity_hash(cells)
    assert plan["cell_identity_sha256"] == [h.cell_identity(c) for c in cells]
    assert all("/Volumes/" not in json.dumps(cell) for cell in cells)

    expected = []
    expected += [("probe_b_depth", d, 8, True, "full", 0) for d in (1, 2, 4, 8, 16, 43)]
    expected += [("probe_b_depth", d, 8, False, "full", 0) for d in (1, 4, 8)]
    for depth in (1, 8, 16, 43):
        for experts in (2, 8, 32, 128, 256):
            expected.append(("probe_c_routed_one_graph", depth, experts, None, "routed_only", 0))
            expected.append(("probe_c_routed_sequential", depth, experts, None, "routed_only", 0))
    for component in ("full", "no_routed", "no_shared", "no_attention", "routed_only"):
        for cr in (0, 4, 128):
            expected.append(("probe_d_ablation", 43, 8, True, component, cr))
    observed = [(c["case"], c["depth"], c["experts"], c["checkpoint"], c["component_mask"], c["compression_ratio"]) for c in cells]
    assert observed == expected
    assert len(set(plan["cell_identity_sha256"])) == 64


def test_multilayer_peak_fixture_is_load_bearing_and_provenanced(tmp_path: Path):
    h = _helper_module()
    derived = h.validate_topology_fixture(_FIXTURE)
    assert derived["fixture_path"] == "tests/fixtures/deepseek_v4_nn_multilayer_peak_topology.json"
    assert derived["schema_version"] == 1
    assert derived["layer_count"] == 43
    assert derived["num_hidden_layers"] == 43
    assert derived["hidden_size"] == 128
    assert derived["moe_intermediate_size"] == 64
    assert derived["n_routed_experts"] == 8
    assert derived["num_experts_per_tok"] == 2
    assert derived["hc_mult"] == 4
    assert set(derived["source_hashes"]) == set(h.PROVENANCE_SOURCES)

    fixture = json.loads(_FIXTURE.read_text())
    mutations = []
    bad = copy.deepcopy(fixture); del bad["purpose"]; mutations.append(bad)
    bad = copy.deepcopy(fixture); bad["layers"][7]["index"] = 8; mutations.append(bad)
    bad = copy.deepcopy(fixture); bad["layers"][0]["hidden_size"] = True; mutations.append(bad)
    bad = copy.deepcopy(fixture); bad["layers"][0]["hidden_size"] = 130; mutations.append(bad)
    bad = copy.deepcopy(fixture); bad["layers"][1]["experts"] = 9; mutations.append(bad)
    bad = copy.deepcopy(fixture); bad["provenance"]["durable_authority"][0]["sha256"] = "0" * 64; mutations.append(bad)
    for idx, payload in enumerate(mutations):
        path = tmp_path / f"bad-{idx}.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError):
            h.validate_topology_fixture(path)


def test_multilayer_peak_report_schema_matrix_evidence_and_metadata_complete():
    h = _helper_module()
    report = _load_report()
    rows = report["rows"]
    metadata = report["metadata"]
    evidence = report["cell_evidence"]
    assert metadata["memory_limit"] == 8_000_000_000
    assert metadata["timeout_s"] <= 180
    assert metadata["real_assets_accessed"] is False
    assert metadata["classification"] == "not-authorized-by-diagnostic"
    assert metadata["execution_mode"] == "serial_fresh_subprocess_per_cell"
    assert metadata["max_concurrency"] == 1
    assert metadata["planned_cell_count"] == metadata["executed_cell_count"] == 64
    assert len(rows) == len(evidence) == 64
    assert metadata["planned_cell_list_sha256"] == metadata["executed_cell_list_sha256"]
    assert metadata["probe_c_comparison_contract"] == h._probe_c_comparison_contract()
    assert metadata["fixture"]["fixture_sha256"] == h.sha256_path(_FIXTURE)
    assert metadata["fixture"]["source_hashes"] == h.validate_topology_fixture(_FIXTURE)["source_hashes"]

    nonces = set()
    previous_finish = None
    for ordinal, (row, ev) in enumerate(zip(rows, evidence)):
        assert set(row) == _REQUIRED_FIELDS
        assert set(ev) == _EVIDENCE_FIELDS
        assert ev["ordinal"] == ordinal
        assert ev["cell_identity_sha256"] == h.cell_identity(row)
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
        assert ev["measured_custom_kernel_nodes"] == row["custom_kernel_nodes"]
        if row["case"].startswith("probe_c"):
            assert ev["trainable_key_count"] is None
            assert ev["input_gradient_shape"] is None
            assert row["nonempty_experts"] == min(row["experts"], row["tokens"] * row["top_k"])
            assert row["custom_kernel_nodes"] == 6 * row["depth"] * row["nonempty_experts"]
        else:
            assert ev["trainable_key_count"] == 6 * min(row["depth"], 16)
            assert ev["input_gradient_shape"] == [1, row["tokens"], row["hc_mult"], row["hidden"]]
            assert ev["input_gradient_nonzero"] is True


def test_multilayer_peak_probe_c_comparison_contract_and_assignments():
    report = _load_report()
    contract = report["metadata"]["probe_c_comparison_contract"]
    assert contract["low_e_control"] == {"experts": 2, "top_k": 2, "assignments": 128, "comparable_to_expert_count_axis": False}
    assert contract["normalized_expert_count_axis"] == {"experts": [8, 32, 128, 256], "top_k": 6, "assignments": 384}
    for row in report["rows"]:
        if not row["case"].startswith("probe_c"):
            continue
        assignments = row["tokens"] * row["top_k"]
        if row["experts"] == 2:
            assert row["top_k"] == 2
            assert assignments == 128
        else:
            assert row["experts"] in (8, 32, 128, 256)
            assert row["top_k"] == 6
            assert assignments == 384


def test_multilayer_peak_checkpoint_coverage_and_parity():
    report = _load_report()
    coverage = report["probe_a"]
    assert coverage["layer_count"] == 43
    assert coverage["pipeline_layer_count"] == 43
    assert coverage["decoder_layer_count"] == 43
    assert coverage["checkpointed_forward_calls"] == 43
    assert coverage["checkpoint_wrapper_name"] == "checkpointed_fn"
    assert coverage["finite_loss_off"] is True
    assert coverage["finite_loss_on"] is True
    assert coverage["finite_gradients_off"] is True
    assert coverage["finite_gradients_on"] is True
    assert coverage["gradient_key_count_off"] > 0
    assert coverage["gradient_key_count_on"] == coverage["gradient_key_count_off"]
    assert coverage["gradient_keys_sha256_on"] == coverage["gradient_keys_sha256_off"]
    assert coverage["gradient_key_sets_equal"] is True
    assert coverage["gradient_shapes_equal"] is True
    assert coverage["gradient_allclose_atol_rtol_1e_5"] is True
    assert coverage["max_gradient_abs_diff"] <= 1e-5
    assert coverage["max_gradient_rel_diff"] <= 1e-5
    exec_ev = coverage["fresh_child_execution"]
    assert exec_ev["child_pid"] != report["metadata"]["parent_pid"]
    assert exec_ev["child_nonce"]
    assert exec_ev["started_monotonic_ns"] < exec_ev["finished_monotonic_ns"]
    assert exec_ev["parent_observed_exit_code"] == 0
    assert exec_ev["parent_observed_signal"] is None
    assert exec_ev["parent_observed_error"] is None


def test_multilayer_peak_runtime_lora_conversion_keys_and_shapes():
    result = _run_helper_snippet(
        """
mx = h._setup_mlx()
from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import Model, ModelArgs
depth = 2
model = Model(ModelArgs.from_dict(h._config(depth, T=4, H=128, I=64, E=8, K=2, hc=4, cr=0)))
h._fill_model_fp4(model, mx)
keys, key_hash, shapes = h._prepare_lora_model(model, depth)
print(json.dumps({"keys": keys, "key_hash": key_hash, "shapes": shapes, "expected": h._expected_lora_keys(depth)}))
"""
    )
    keys = result["keys"]
    assert keys == result["expected"]
    assert result["key_hash"] == _helper_module()._hash_keys(keys)
    assert len(keys) == 12
    assert all("self_attn" in key and any(proj in key for proj in _helper_module().LORA_KEYS) for key in keys)
    assert not any("mlp" in key or "embed" in key or "lm_head" in key for key in keys)
    for key, shape in result["shapes"].items():
        if key.endswith("lora_a"):
            assert shape[-1] == 8
        elif key.endswith("lora_b"):
            assert shape[0] == 8
        else:
            raise AssertionError(key)


def test_multilayer_peak_runtime_raw_input_gradient_is_not_sanitized():
    result = _run_helper_snippet(
        """
mx = h._setup_mlx()
def fail_nan_to_num(*_args, **_kwargs):
    raise AssertionError("input gradients must remain raw; nan_to_num is forbidden")
mx.nan_to_num = fail_nan_to_num
row = h.run_model_cell(h.plan_cells()[0])
print(json.dumps({
    "error": row.get("error"),
    "finite_gradients": row.get("finite_gradients"),
    "input_gradient_nonzero": row.get("_cell_evidence", {}).get("input_gradient_nonzero"),
}))
""",
        timeout=220,
    )
    assert result == {"error": None, "finite_gradients": True, "input_gradient_nonzero": True}


def test_multilayer_peak_runtime_masked_component_independent_oracles_and_mutations():
    result = _run_helper_snippet(
        """
mx = h._setup_mlx()
from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4_nn import Model, ModelArgs
MASKS = ("full", "no_routed", "no_shared", "no_attention", "routed_only")

def make_model():
    model = Model(ModelArgs.from_dict(h._config(1, T=4, H=128, I=64, E=8, K=2, hc=4, cr=0)))
    h._fill_model_fp4(model, mx)
    return model

def token_ids():
    return mx.arange(4, dtype=mx.int32).reshape(1, 4)

def initial_hidden(model, ids):
    embedded = model.model.embed_tokens(ids)
    return mx.broadcast_to(mx.expand_dims(embedded, -2), (*embedded.shape[:-1], model.args.hc_mult, model.args.hidden_size))

def finite(value):
    return bool(mx.all(mx.isfinite(value)).item())

def nonzero(value):
    return bool(mx.any(mx.abs(value.astype(mx.float32)) > 0).item())

def zero_expression_metrics():
    x = (mx.arange(1 * 4 * 4 * 8, dtype=mx.float32).reshape(1, 4, 4, 8) + 1.0) / 17.0
    y = x - mx.stop_gradient(x)
    def loss(z):
        return mx.sum(z - mx.stop_gradient(z))
    _, grad = mx.value_and_grad(loss)(x)
    mx.eval(y, grad)
    return {
        "shape": list(y.shape),
        "dtype": str(y.dtype),
        "forward_max_abs": float(mx.max(mx.abs(y)).item()),
        "grad_min": float(mx.min(grad).item()),
        "grad_max": float(mx.max(grad).item()),
        "grad_nonzero": nonzero(grad),
    }

def helper_identity_metrics():
    residual = (mx.arange(1 * 4 * 4 * 128, dtype=mx.float32).reshape(1, 4, 4, 128) + 1.0) / 251.0
    like = mx.zeros((1, 4, 128), dtype=mx.float32)
    actual = h._zero_residual_identity_like(residual, like)
    expected = mx.mean(residual - mx.stop_gradient(residual), axis=-2).astype(like.dtype)
    def actual_loss(z):
        return mx.sum(h._zero_residual_identity_like(z, like))
    def expected_loss(z):
        return mx.sum(mx.mean(z - mx.stop_gradient(z), axis=-2).astype(like.dtype))
    _, actual_grad = mx.value_and_grad(actual_loss)(residual)
    _, expected_grad = mx.value_and_grad(expected_loss)(residual)
    mx.eval(actual, expected, actual_grad, expected_grad)
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "forward_max_abs": float(mx.max(mx.abs(actual)).item()),
        "expected_diff": float(mx.max(mx.abs(actual - expected)).item()),
        "grad_diff": float(mx.max(mx.abs(actual_grad - expected_grad)).item()),
        "grad_nonzero": nonzero(actual_grad),
    }

def routed_metrics():
    model = make_model()
    ids = token_ids()
    layer = model.model.pipeline_layers[0]
    hidden = initial_hidden(model, ids)
    post, comb, collapsed = layer.attn_hc(hidden)
    attn_input = layer.input_layernorm(collapsed)
    attn_output = layer.self_attn(attn_input, cache=None)
    after_attention = mx.expand_dims(post.astype(hidden.dtype), -1) * mx.expand_dims(attn_output, -2) + (
        comb.astype(hidden.dtype).transpose(0, 1, 3, 2) @ hidden
    )
    mlp_input = layer.post_attention_layernorm(layer.ffn_hc(after_attention)[2])
    actual = h._routed_only(layer.mlp, mlp_input, ids)
    expected = layer.mlp(mlp_input, input_ids=ids) - layer.mlp.shared_experts(mlp_input)
    def actual_loss(z):
        y = h._routed_only(layer.mlp, z, ids)
        return mx.sum(y * y)
    def expected_loss(z):
        y = layer.mlp(z, input_ids=ids) - layer.mlp.shared_experts(z)
        return mx.sum(y * y)
    _, actual_grad = mx.value_and_grad(actual_loss)(mlp_input)
    _, expected_grad = mx.value_and_grad(expected_loss)(mlp_input)
    mx.eval(actual, expected, actual_grad, expected_grad)
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "forward_max_abs": float(mx.max(mx.abs(actual - expected)).item()),
        "gradient_max_abs": float(mx.max(mx.abs(actual_grad - expected_grad)).item()),
        "actual_grad_nonzero": nonzero(actual_grad),
        "expected_grad_nonzero": nonzero(expected_grad),
    }

def model_mask_metrics():
    model = make_model()
    ids = token_ids()
    keys, key_hash, _shapes = h._prepare_lora_model(model, 1)
    hidden = initial_hidden(model, ids)
    ordinary = model(ids)
    masked_full = h.model_forward_once(model, ids, None, "full")
    mx.eval(ordinary, masked_full)
    masks = {}
    for mask in MASKS:
        probe = mx.zeros_like(hidden)
        forward = h.model_forward_once(model, ids, None, mask)
        def loss_for_probe(input_probe):
            y = h.model_forward_once(model, ids, input_probe, mask)
            return mx.mean(y * y)
        _, input_grad = mx.value_and_grad(loss_for_probe)(probe)
        mx.eval(forward, input_grad)
        masks[mask] = {
            "shape": list(forward.shape),
            "dtype": str(forward.dtype),
            "finite_forward": finite(forward),
            "finite_input_grad": finite(input_grad),
            "input_grad_shape": list(input_grad.shape),
            "input_grad_nonzero": nonzero(input_grad),
            "keys_unchanged": h._tree_keys(model.trainable_parameters()) == keys,
        }
    return {
        "full_forward_max_abs": float(mx.max(mx.abs(ordinary - masked_full)).item()),
        "ordinary_shape": list(ordinary.shape),
        "ordinary_dtype": str(ordinary.dtype),
        "keys": keys,
        "key_hash": key_hash,
        "expected_keys": h._expected_lora_keys(1),
        "masks": masks,
    }

def collect():
    return {
        "zero_expression": zero_expression_metrics(),
        "identity": helper_identity_metrics(),
        "routed": routed_metrics(),
        "model_masks": model_mask_metrics(),
    }

normal = collect()
orig_routed = h._routed_only
orig_identity = h._zero_residual_identity_like
h._routed_only = lambda _block, x, _input_ids: mx.zeros_like(x)
h._zero_residual_identity_like = lambda _residual, like: mx.zeros_like(like)
mutated = collect()
h._routed_only = orig_routed
h._zero_residual_identity_like = orig_identity
print(json.dumps({"normal": normal, "mutated": mutated}))
""",
        timeout=220,
    )
    normal = result["normal"]
    assert normal["zero_expression"] == {
        "shape": [1, 4, 4, 8],
        "dtype": "mlx.core.float32",
        "forward_max_abs": 0.0,
        "grad_min": 1.0,
        "grad_max": 1.0,
        "grad_nonzero": True,
    }
    assert normal["identity"] == {
        "shape": [1, 4, 128],
        "dtype": "mlx.core.float32",
        "forward_max_abs": 0.0,
        "expected_diff": 0.0,
        "grad_diff": 0.0,
        "grad_nonzero": True,
    }
    assert normal["routed"]["shape"] == [1, 4, 128]
    assert normal["routed"]["dtype"] == "mlx.core.float32"
    assert normal["routed"]["forward_max_abs"] <= 1e-6
    assert normal["routed"]["gradient_max_abs"] <= 1e-6
    assert normal["routed"]["actual_grad_nonzero"] is True
    assert normal["routed"]["expected_grad_nonzero"] is True

    model_masks = normal["model_masks"]
    assert model_masks["full_forward_max_abs"] == 0.0
    assert model_masks["ordinary_shape"] == [1, 4, 128]
    assert model_masks["ordinary_dtype"] == "mlx.core.float32"
    assert model_masks["keys"] == model_masks["expected_keys"] == _helper_module()._expected_lora_keys(1)
    assert model_masks["key_hash"] == _helper_module()._hash_keys(model_masks["keys"])
    for mask, metrics in model_masks["masks"].items():
        assert mask in {"full", "no_routed", "no_shared", "no_attention", "routed_only"}
        assert metrics == {
            "shape": [1, 4, 128],
            "dtype": "mlx.core.float32",
            "finite_forward": True,
            "finite_input_grad": True,
            "input_grad_shape": [1, 4, 4, 128],
            "input_grad_nonzero": True,
            "keys_unchanged": True,
        }

    mutated = result["mutated"]
    assert mutated["identity"]["grad_nonzero"] is False
    assert mutated["identity"]["grad_diff"] > 0.0
    assert mutated["routed"]["forward_max_abs"] > 1e-4
    assert mutated["routed"]["gradient_max_abs"] > 1e-7
    assert mutated["routed"]["actual_grad_nonzero"] is False
    assert mutated["routed"]["expected_grad_nonzero"] is True


def test_multilayer_peak_runtime_dot_boundary_and_probe_c_measurement():
    result = _run_helper_snippet(
        """
row = h.run_routed_cell({
    "case": "probe_c_routed_one_graph", "depth": 1, "experts": 2, "nonempty_experts": 2,
    "tokens": 64, "hidden": 64, "intermediate": 32, "top_k": 2, "hc_mult": None,
    "compression_ratio": 0, "checkpoint": None, "component_mask": "routed_only",
})
print(json.dumps({
    "error": row.get("error"),
    "active_after_graph_present": row.get("active_after_graph") is not None,
    "custom_kernel_nodes": row.get("custom_kernel_nodes"),
    "measured_custom_kernel_nodes": row.get("_cell_evidence", {}).get("measured_custom_kernel_nodes"),
}))
"""
    )
    assert result["error"] is None
    assert result["active_after_graph_present"] is True
    assert result["custom_kernel_nodes"] == 12
    assert result["measured_custom_kernel_nodes"] == result["custom_kernel_nodes"]


def test_multilayer_peak_compile_state_isolated_from_diagnostic_subprocesses():
    before = _compile_eval_inside_compile_expectation()
    diagnostic = _run_helper_snippet(
        """
row, evidence = h.run_cell_subprocess(h.plan_cells()[0], 0)
print(json.dumps({
    "error": row.get("error"),
    "exit_code": row.get("exit_code"),
    "child_pid": evidence.get("child_pid"),
    "child_nonce": evidence.get("child_nonce"),
}))
""",
        timeout=220,
    )
    after = _compile_eval_inside_compile_expectation()
    assert before["raised"] is True
    assert after["raised"] is True
    assert diagnostic["error"] is None
    assert diagnostic["exit_code"] == 0
    assert diagnostic["child_pid"]
    assert diagnostic["child_nonce"]


def test_multilayer_peak_malformed_child_output_and_outcome_override_handling():
    h = _helper_module()
    nonce = "nonce-123"
    assert h._parse_child_stdout("not json\n{}\n", nonce) == (None, None)
    pid, row = h._parse_child_stdout(
        json.dumps({"child_nonce": nonce, "pid": 123}) + "\n" + json.dumps({"case": "probe_b_depth", "exit_code": 99, "signal": "TIMEOUT"}),
        nonce,
    )
    assert pid == 123
    assert row["exit_code"] == 99 and row["signal"] == "TIMEOUT"
    normalized = h.normalize_process_row(row, h.plan_cells()[0], -9, "killed by parent")
    assert normalized["exit_code"] is None
    assert normalized["signal"] in {"SIGKILL", "KILL"}
    assert normalized["error"]

def test_multilayer_peak_helper_source_guards_against_reviewer_red_patterns():
    source = _HELPER.read_text()
    assert "model(ids) * model(ids)" not in source
    assert "chain(z) * chain(z)" not in source
    assert "layer(z) * layer(z)" not in source
    assert "active_after_graph\"] = row[\"active_baseline\"]" not in source
    assert "linear_to_lora_layers(model, min(depth, 16)" in source
    assert "model.freeze()" in source
    assert "mx.export_to_dot" in source


def test_multilayer_peak_exit_signal_timeout_normalization_unit_paths():
    h = _helper_module()
    cell = h.plan_cells()[0]
    timeout = h.normalize_process_row(h.base_row(cell), cell, None, "late")
    assert timeout["exit_code"] is None and timeout["signal"] == "TIMEOUT" and timeout["error"]
    signaled = h.normalize_process_row(h.base_row(cell), cell, -9, "killed")
    assert signaled["exit_code"] is None and signaled["signal"] in {"SIGKILL", "KILL"} and signaled["error"]
    failed = h.normalize_process_row(h.base_row(cell), cell, 2, "bad")
    assert failed["exit_code"] == 2 and failed["signal"] is None and failed["error"]
    ok = h.normalize_process_row(h.base_row(cell), cell, 0, "")
    assert ok["exit_code"] == 0 and ok["signal"] is None and ok["error"] is None
