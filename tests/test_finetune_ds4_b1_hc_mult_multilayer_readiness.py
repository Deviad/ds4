#!/usr/bin/env python3
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import finetune_ds4


BLOCKERS = (
    "full attention parity with RoPE/cache/sinks/compressor/indexer",
    "full decoder-layer hyperconnection residual mixing and final hyperhead parity",
    "full MoE parity with packed FP4/I8 expert dequant and expert kernels",
    "full MLX shimmed-checkpoint load/forward and MLX generation smoke (Track B only; DS4-GGUF base generation is its own Track-A gate, .ds4-gguf-generate-ok, not a forward-parity blocker)",
)
EXPECTED_REAL_MODE_PROOF_IDS = ("B0a-1", "B0a-2", "B0a-3", "B0b-a-1", "B0b-a-2", "B2-a-1", "B2-a-2", "B2-a-3")
VALIDATE_REJECT_MESSAGE = "real DeepSeek V4 model stacked multi-layer forward currently supports only hc_mult=1; hc_mult>1 multi-layer parity is not proven"
REFERENCE_REJECT_MESSAGE = "integrated multi-layer reference currently supports only hc_mult=1"


def stub_partials(status="ok"):
    return [
        {
            "fixture": name,
            "status": status,
            "max_abs_error": 0.0,
            "covered": [f"covered:{name}"],
            "not_covered": [],
        }
        for name in finetune_ds4.FORWARD_PARITY_FIXTURE_NAMES
    ]


def stub_real_mode_proofs(status="ok"):
    proofs = [
        {
            "id": proof_id,
            "name": proof_id,
            "status": status,
            "max_abs_error": 0.0 if status == "ok" else None,
            "config": {},
            "reference": "stub",
            "covered": [f"covered:{proof_id}"],
            "not_covered": ["full B0/B1/B2/B3 parity"],
        }
        for proof_id in EXPECTED_REAL_MODE_PROOF_IDS
    ]
    return {
        "category": "B0-partial",
        "description": "stubbed real-mode proofs",
        "tolerance": 1e-5,
        "proofs_total": len(proofs),
        "proofs_ok": sum(1 for item in proofs if item["status"] == "ok"),
        "proofs_skipped": sum(1 for item in proofs if item["status"] == "skipped"),
        "proofs_failed": sum(1 for item in proofs if item["status"] not in {"ok", "skipped"}),
        "proofs": proofs,
        "b0_partial_progress": "DATA ONLY — B0 NOT marked satisfied.",
        "not_covered": ["stateful cache real-mode forward", "real checkpoint load"],
    }


def double_blocked_probe():
    return {
        "probe_status": "double-blocked",
        "import_ok": True,
        "import_error": None,
        "validate_real_mode_matrix": [
            {"num_hidden_layers": 1, "hc_mult": 1, "constructs": True, "gate_rejects": False, "reject_message": None},
            {"num_hidden_layers": 1, "hc_mult": 2, "constructs": True, "gate_rejects": False, "reject_message": None},
            {"num_hidden_layers": 2, "hc_mult": 1, "constructs": True, "gate_rejects": False, "reject_message": None},
            {"num_hidden_layers": 2, "hc_mult": 2, "constructs": False, "gate_rejects": True, "reject_message": VALIDATE_REJECT_MESSAGE},
            {"num_hidden_layers": 3, "hc_mult": 2, "constructs": False, "gate_rejects": True, "reject_message": VALIDATE_REJECT_MESSAGE},
        ],
        "production_gate": {
            "function": "Model._validate_real_mode",
            "rejects_multilayer_hc_gt1": True,
            "reject_message_substring": "hc_mult>1 multi-layer parity is not proven",
            "observed_message": VALIDATE_REJECT_MESSAGE,
            "construction_calls_validate": True,
            "vendor_reference": "vendor/mlx_lm_models/deepseek_v4.py",
            "approx_line": 1694,
            "line_is_evidence_only": True,
        },
        "reference_gate": {
            "function": "_integrated_multilayer_forward",
            "rejects_hc_gt1": True,
            "reject_message_substring": REFERENCE_REJECT_MESSAGE,
            "observed_message": REFERENCE_REJECT_MESSAGE,
            "vendor_reference": "vendor/mlx_lm_models/deepseek_v4.py",
            "approx_line": 1102,
            "line_is_evidence_only": True,
        },
        "transformers_setter": {
            "function": "set_transformers_integrated_weights",
            "single_layer_only": True,
            "touches_layers_0": True,
            "has_layer_loop": False,
        },
    }


def skipped_probe():
    return {
        "probe_status": "skipped",
        "import_ok": False,
        "import_error": "MLX real-mode dependencies unavailable: No module named 'mlx'",
        "validate_real_mode_matrix": [],
        "production_gate": {
            "function": "Model._validate_real_mode",
            "rejects_multilayer_hc_gt1": False,
            "reject_message_substring": "hc_mult>1 multi-layer parity is not proven",
        },
        "reference_gate": {
            "function": "_integrated_multilayer_forward",
            "rejects_hc_gt1": False,
            "reject_message_substring": REFERENCE_REJECT_MESSAGE,
        },
        "transformers_setter": {"function": "set_transformers_integrated_weights", "single_layer_only": False},
    }


def gate_lifted_probe():
    probe = double_blocked_probe()
    probe["probe_status"] = "open"
    for row in probe["validate_real_mode_matrix"]:
        if row["num_hidden_layers"] > 1 and row["hc_mult"] > 1:
            row["constructs"] = True
            row["gate_rejects"] = False
            row["reject_message"] = None
    probe["production_gate"] = {
        **probe["production_gate"],
        "rejects_multilayer_hc_gt1": False,
        "observed_message": None,
    }
    probe["reference_gate"] = {
        **probe["reference_gate"],
        "rejects_hc_gt1": False,
        "observed_message": None,
    }
    probe["transformers_setter"] = {
        **probe["transformers_setter"],
        "single_layer_only": False,
        "has_layer_loop": True,
    }
    return probe


class B1HcMultMultilayerReadinessTests(unittest.TestCase):
    def test_builder_double_blocked_is_not_ready(self):
        block = finetune_ds4._b1_hc_mult_multilayer_readiness(double_blocked_probe())
        self.assertEqual(block["block"], "b1-hc-mult-multilayer-readiness")
        self.assertIs(block["proof_available"], False)
        self.assertIs(block["hc_mult_multi_layer_allowed"], False)
        self.assertIs(block["double_blocked"], True)
        self.assertEqual(block["probe_status"], "double-blocked")
        self.assertEqual(block["status"], "fail-closed")
        self.assertEqual(block["decision"], "not-ready")
        self.assertIs(block["fail_closed"], True)
        self.assertIs(block["ready"], False)
        self.assertIn("UNPROVEN", block["verdict"])
        self.assertIn("double-blocked", block["verdict"])

    def test_builder_skipped_probe_is_not_ready(self):
        block = finetune_ds4._b1_hc_mult_multilayer_readiness(skipped_probe())
        self.assertIs(block["proof_available"], False)
        self.assertIs(block["hc_mult_multi_layer_allowed"], False)
        self.assertIs(block["double_blocked"], False)
        self.assertEqual(block["probe_status"], "skipped")
        self.assertEqual(block["decision"], "not-ready")
        self.assertIs(block["fail_closed"], True)
        self.assertIs(block["ready"], False)
        self.assertIs(block["evidence"]["import_ok"], False)
        self.assertIn("No module named", block["evidence"]["import_error"])

    def test_builder_gate_lifted_probe_still_not_ready(self):
        block = finetune_ds4._b1_hc_mult_multilayer_readiness(gate_lifted_probe())
        self.assertIs(block["hc_mult_multi_layer_allowed"], True)
        self.assertIs(block["double_blocked"], False)
        self.assertIs(block["proof_available"], False)
        self.assertEqual(block["decision"], "not-ready")
        self.assertEqual(block["status"], "fail-closed")
        self.assertIs(block["fail_closed"], True)
        self.assertIs(block["ready"], False)
        self.assertIn("necessary-but-insufficient", block["no_false_positive_policy"])

    def test_evidence_pins_both_gate_messages(self):
        block = finetune_ds4._b1_hc_mult_multilayer_readiness(double_blocked_probe())
        evidence = block["evidence"]
        self.assertEqual(evidence["production_gate"]["function"], "Model._validate_real_mode")
        self.assertEqual(evidence["production_gate"]["reject_message_substring"], "hc_mult>1 multi-layer parity is not proven")
        self.assertEqual(evidence["reference_gate"]["function"], "_integrated_multilayer_forward")
        self.assertEqual(evidence["reference_gate"]["reject_message_substring"], REFERENCE_REJECT_MESSAGE)
        self.assertIs(evidence["transformers_setter"]["single_layer_only"], True)
        rejected_cells = {
            (row["num_hidden_layers"], row["hc_mult"]): row
            for row in evidence["validate_real_mode_matrix"]
        }
        self.assertIs(rejected_cells[(2, 2)]["gate_rejects"], True)
        self.assertIs(rejected_cells[(3, 2)]["gate_rejects"], True)
        self.assertEqual(rejected_cells[(2, 2)]["reject_message"], VALIDATE_REJECT_MESSAGE)

    def test_b0a_evidence_labeled_insufficient(self):
        block = finetune_ds4._b1_hc_mult_multilayer_readiness(double_blocked_probe())
        self.assertEqual([item["id"] for item in block["b0a_evidence"]], ["B0a-1", "B0a-2", "B0a-3"])
        for item in block["b0a_evidence"]:
            self.assertIs(item["proves_full_b1"], False)
            self.assertEqual(item["proving_story"], "11.32")
        self.assertIn("do NOT compose", block["b0a_non_claim"])
        self.assertGreaterEqual(len(block["future_proof_criteria"]), 5)
        self.assertIn("_validate_real_mode", block["future_proof_criteria"][0])

    def test_default_seam_probe_is_skipped_fail_closed(self):
        probe = finetune_ds4._default_b1_hc_mult_multilayer_seam_probe()
        self.assertEqual(probe["probe_status"], "skipped")
        block = finetune_ds4._b1_hc_mult_multilayer_readiness(probe)
        self.assertIs(block["proof_available"], False)
        self.assertIs(block["hc_mult_multi_layer_allowed"], False)
        self.assertEqual(block["decision"], "not-ready")
        self.assertIs(block["fail_closed"], True)
        self.assertIs(block["ready"], False)

    def test_additive_block_does_not_change_honesty_counters(self):
        with tempfile.TemporaryDirectory() as td:
            mlx_work = pathlib.Path(td) / "mlx"
            report = finetune_ds4.build_forward_parity_readiness_report(
                mlx_work=mlx_work,
                partials=stub_partials(),
                blockers=BLOCKERS,
                marker_present=False,
                model_constructs=True,
                non_forward_gate_status={".deepseek-v4-import-ok": "present"},
                generated_at="2026-06-20T00:00:00+00:00",
                real_mode_proofs=stub_real_mode_proofs(),
                b1_hc_mult_multilayer_seam_probe=gate_lifted_probe(),
            )
            self.assertFalse((mlx_work / ".deepseek-v4-forward-parity-ok").exists())
            self.assertFalse((mlx_work / "model-4bit").exists())
        self.assertEqual(report["schema"], 1)
        self.assertIs(report["full_forward_parity"], False)
        self.assertIs(report["marker_earned"], False)
        self.assertEqual(report["status"], "not-ready")
        self.assertEqual(report["blockers_count"], 4)
        self.assertEqual(report["real_mode_proofs"]["proofs_total"], 8)
        self.assertEqual(report["coverage"]["fixtures_total"], 19)
        self.assertIn("b1_hc_mult_multilayer_readiness", report)
        self.assertGreater(list(report).index("b1_hc_mult_multilayer_readiness"), list(report).index("stateful_decode_readiness"))
        self.assertLess(list(report).index("b1_hc_mult_multilayer_readiness"), list(report).index("marker_write_criteria"))
        self.assertIs(report["b1_hc_mult_multilayer_readiness"]["hc_mult_multi_layer_allowed"], True)
        self.assertIs(report["b1_hc_mult_multilayer_readiness"]["ready"], False)
        self.assertIs(report["b1_hc_mult_multilayer_readiness"]["proof_available"], False)

    def test_no_mlx_or_torch_imports_required(self):
        # The mlx venv's editable ds4_ft_mlx install preloads torch/mlx at
        # interpreter site-init (verified: -S -> torch*/mlx* = 0/0, default ->
        # 1050/32), so an absolute assertNotIn(..., sys.modules) is env-broken.
        # Snapshot-diff: the readiness path itself must add no new torch/mlx
        # (top-level or submodule) keys to sys.modules. (ADR 0018.)
        def _is_torch_or_mlx(name):
            return (name == "torch" or name.startswith("torch.")
                    or name == "mlx" or name.startswith("mlx."))
        before = {k for k in sys.modules if _is_torch_or_mlx(k)}
        finetune_ds4._b1_hc_mult_multilayer_readiness(double_blocked_probe())
        with tempfile.TemporaryDirectory() as td:
            finetune_ds4.build_forward_parity_readiness_report(
                mlx_work=pathlib.Path(td) / "mlx",
                partials=stub_partials(),
                blockers=BLOCKERS,
                marker_present=False,
                model_constructs=True,
                non_forward_gate_status={},
                generated_at="2026-06-20T00:00:00+00:00",
                real_mode_proofs=stub_real_mode_proofs(),
                b1_hc_mult_multilayer_seam_probe=double_blocked_probe(),
            )
        added = {k for k in sys.modules if _is_torch_or_mlx(k)} - before
        self.assertFalse(
            added,
            f"B1 readiness path imported torch/mlx (new sys.modules keys): {sorted(added)}",
        )


if __name__ == "__main__":
    unittest.main()
