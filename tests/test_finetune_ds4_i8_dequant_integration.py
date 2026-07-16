#!/usr/bin/env python3
import inspect
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import finetune_ds4


EXPECTED_REAL_MODE_PROOF_IDS = ("B0a-1", "B0a-2", "B0a-3", "B0b-a-1", "B0b-a-2", "B2-a-1", "B2-a-2", "B2-a-3", "B2-a-4")


def _b2a1_spec():
    return next(spec for spec in finetune_ds4._real_mode_forward_proof_specs() if spec["id"] == "B2-a-1")


def _b2a2_spec():
    return next(spec for spec in finetune_ds4._real_mode_forward_proof_specs() if spec["id"] == "B2-a-2")


def _b2a3_spec():
    return next(spec for spec in finetune_ds4._real_mode_forward_proof_specs() if spec["id"] == "B2-a-3")


def _stub_proofs(status="ok"):
    proofs = []
    for spec in finetune_ds4._real_mode_forward_proof_specs():
        proof = {
            "id": spec["id"],
            "name": spec["name"],
            "status": status,
            "max_abs_error": 0.0 if status == "ok" else None,
            "config": dict(spec["config"]),
            "reference": spec["reference"],
            "covered": list(spec.get("covered", [])),
            "not_covered": list(spec.get("not_covered", [])),
        }
        if spec["id"] in {"B2-a-1", "B2-a-2", "B2-a-3"}:
            proof.update({
                "evidence_class": "B2-partial",
                "reference_max_abs_error": 0.0,
                "reference_tolerance": 1e-3,
                "quantization_gap": 0.01,
                "non_degenerate": True,
                "block_scale_nonunit": True,
                "i8_branch_reached": True,
            })
        if spec["id"] == "B2-a-2":
            proof["secondary_multilayer"] = {
                "num_hidden_layers": 3,
                "max_abs_error": 0.0,
                "reference_max_abs_error": 0.0,
                "ok": True,
            }
        if spec["id"] == "B2-a-3":
            proof.update({
                "multi_expert_combine_exercised": True,
                "num_experts_per_tok": 2,
                "n_routed_experts": 4,
                "tie_free_margin": 0.156,
                "routing_subset_stable": True,
                "secondary_topk": {
                    "num_experts_per_tok": 3,
                    "max_abs_error": 0.0,
                    "reference_max_abs_error": 0.0,
                    "ok": True,
                },
            })
        proofs.append(proof)
    return proofs


class I8DequantIntegrationProofTests(unittest.TestCase):
    def test_b2_a1_spec_present_and_shaped(self):
        specs = finetune_ds4._real_mode_forward_proof_specs()
        self.assertEqual([spec["id"] for spec in specs], list(EXPECTED_REAL_MODE_PROOF_IDS))
        spec = _b2a1_spec()
        self.assertEqual(spec["name"], "real-mode-single-layer-i8-block-scale-dequant-moe")
        self.assertEqual(spec["evidence_class"], "B2-partial")
        self.assertEqual(spec["reference"], "_integrated_layer_forward")
        self.assertIn("raw branch", spec["isolation_reference"])
        self.assertEqual(spec["config"]["expert_dtype"], "i8")
        self.assertEqual(spec["config"]["hidden_size"], 16)
        self.assertEqual(spec["config"]["moe_intermediate_size"], 16)
        self.assertEqual(spec["config"]["block_size"], 16)
        self.assertEqual(spec["config"]["scale_axis"], 1)
        self.assertEqual(spec["config"]["n_routed_experts"], 2)
        self.assertEqual(spec["config"]["num_experts_per_tok"], 1)
        self.assertEqual(spec["config"]["n_shared_experts"], 1)
        self.assertEqual(spec["config"]["seq_len"], 2)

    def test_b2_a1_report_counts_eight_with_stub(self):
        report = finetune_ds4._real_mode_proofs_report(_stub_proofs())
        self.assertEqual(report["category"], "B0-partial")
        self.assertEqual(report["proofs_total"], 9)
        self.assertEqual(report["proofs_ok"], 9)
        self.assertEqual(report["proofs_skipped"], 0)
        self.assertEqual(report["proofs_failed"], 0)
        self.assertEqual([proof["id"] for proof in report["proofs"]], list(EXPECTED_REAL_MODE_PROOF_IDS))
        self.assertIn("B0 NOT", report["b0_partial_progress"])
        self.assertIn("B2", report["b0_partial_progress"])
        self.assertFalse(any("full_forward_parity" in proof for proof in report["proofs"]))

    def test_b2_a2_spec_present_and_shaped(self):
        spec = _b2a2_spec()
        self.assertEqual(spec["name"], "real-mode-multilayer-i8-block-scale-dequant-moe")
        self.assertEqual(spec["evidence_class"], "B2-partial")
        self.assertEqual(spec["reference"], "_integrated_multilayer_forward")
        self.assertIn("raw branch", spec["isolation_reference"])
        self.assertEqual(spec["config"]["expert_dtype"], "i8")
        self.assertEqual(spec["config"]["num_hidden_layers"], 2)
        self.assertEqual(spec["config"]["hidden_size"], 16)
        self.assertEqual(spec["config"]["moe_intermediate_size"], 16)
        self.assertEqual(spec["config"]["block_size"], 16)
        self.assertEqual(spec["config"]["scale_axis"], 1)
        self.assertEqual(spec["config"]["secondary_num_hidden_layers"], 3)

    def test_b2_a2_not_covered_closes_multilayer(self):
        spec = _b2a2_spec()
        not_covered = "\n".join(spec["not_covered"])
        self.assertIn("FP4", not_covered)
        self.assertIn("real checkpoint payload", not_covered)
        self.assertIn("expert parallel kernels", not_covered)
        self.assertIn("B1", not_covered)
        self.assertIn("B3", not_covered)
        self.assertNotIn("multi-layer I8 stacking", not_covered)

    def test_b2_a3_spec_present_and_shaped(self):
        spec = _b2a3_spec()
        self.assertEqual(spec["name"], "real-mode-topk-multi-expert-i8-block-scale-dequant-moe")
        self.assertEqual(spec["evidence_class"], "B2-partial")
        self.assertEqual(spec["reference"], "_integrated_layer_forward")
        self.assertIn("raw branch", spec["isolation_reference"])
        self.assertEqual(spec["config"]["expert_dtype"], "i8")
        self.assertEqual(spec["config"]["num_hidden_layers"], 1)
        self.assertEqual(spec["config"]["hidden_size"], 16)
        self.assertEqual(spec["config"]["moe_intermediate_size"], 16)
        self.assertEqual(spec["config"]["n_routed_experts"], 4)
        self.assertEqual(spec["config"]["num_experts_per_tok"], 2)
        self.assertEqual(spec["config"]["secondary_num_experts_per_tok"], 3)
        self.assertEqual(spec["config"]["block_size"], 16)
        self.assertEqual(spec["config"]["scale_axis"], 1)
        self.assertEqual(spec["config"]["seq_len"], 2)
        covered = "\n".join(spec["covered"])
        self.assertIn("argsort", covered)
        self.assertIn("multi-term denom", covered)
        self.assertIn("factor=0", covered)

    def test_b2_a3_not_covered_defers_fp4_payload_kernels_b1_b3(self):
        spec = _b2a3_spec()
        not_covered = "\n".join(spec["not_covered"])
        self.assertIn("packed FP4", not_covered)
        self.assertIn("real checkpoint payload", not_covered)
        self.assertIn("expert parallel kernels", not_covered)
        self.assertIn("B1", not_covered)
        self.assertIn("B3", not_covered)
        self.assertNotIn("top-k>1", not_covered)
        self.assertNotIn("multi-expert I8", not_covered)

    def test_i8_dequant_proof_status_truth_table(self):
        ok, reason = finetune_ds4._i8_dequant_proof_status(0.0, 1e-4, 1e-3, True, True, 1e-5)
        self.assertEqual(ok, "ok")
        self.assertIsNone(reason)
        status, reason = finetune_ds4._i8_dequant_proof_status(2e-5, 1e-4, 1e-3, True, True, 1e-5)
        self.assertEqual(status, "failed")
        self.assertIn("isolation", reason)
        status, reason = finetune_ds4._i8_dequant_proof_status(0.0, 2e-3, 1e-3, True, True, 1e-5)
        self.assertEqual(status, "failed")
        self.assertIn("reference_max_abs_error", reason)
        status, reason = finetune_ds4._i8_dequant_proof_status(0.0, 1e-4, 1e-3, False, True, 1e-5)
        self.assertEqual(status, "failed")
        self.assertIn("degenerate", reason)
        status, reason = finetune_ds4._i8_dequant_proof_status(0.0, 1e-4, 1e-3, True, False, 1e-5)
        self.assertEqual(status, "failed")
        self.assertIn("branch", reason)

    def test_b2_a1_not_covered_lists_fp4_payload_kernels(self):
        spec = _b2a1_spec()
        not_covered = "\n".join(spec["not_covered"])
        self.assertIn("FP4", not_covered)
        self.assertIn("real checkpoint payload", not_covered)
        self.assertIn("expert parallel kernels", not_covered)
        self.assertIn("multi-layer I8", not_covered)
        self.assertIn("B1", not_covered)
        self.assertIn("B3", not_covered)

    def test_b2_a1_runner_writes_no_marker(self):
        source = "\n".join([
            inspect.getsource(finetune_ds4._run_i8_dequant_integration_proof),
            inspect.getsource(finetune_ds4._make_real_mode_i8_dequant_weights),
            inspect.getsource(finetune_ds4._make_real_mode_multilayer_i8_dequant_weights),
            inspect.getsource(finetune_ds4._real_mode_proofs_report),
        ])
        self.assertNotIn("_write_gate_marker", source)
        self.assertNotIn("model-4bit", source)
        self.assertNotIn(".deepseek-", source)

    def test_no_mlx_or_torch_imports_required(self):
        # The mlx venv's editable ds4_ft_mlx install preloads torch/mlx at
        # interpreter site-init (verified: -S -> torch*/mlx* = 0/0, default ->
        # 1050/32), so an absolute assertNotIn(..., sys.modules) is env-broken.
        # Snapshot-diff: the readiness path itself must add no new torch/mlx
        # (top-level or submodule, incl mlx.core via startswith("mlx.")) keys to
        # sys.modules. (ADR 0018.)
        def _is_torch_or_mlx(name):
            return (name == "torch" or name.startswith("torch.")
                    or name == "mlx" or name.startswith("mlx."))
        before = {k for k in sys.modules if _is_torch_or_mlx(k)}
        _b2a1_spec()
        finetune_ds4._real_mode_proofs_report(_stub_proofs())
        added = {k for k in sys.modules if _is_torch_or_mlx(k)} - before
        self.assertFalse(
            added,
            f"I8 dequant integration readiness path imported torch/mlx (new sys.modules keys): {sorted(added)}",
        )


if __name__ == "__main__":
    unittest.main()
