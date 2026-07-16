#!/usr/bin/env python3
import pathlib
import sys
import tempfile
import unittest

from scripts import finetune_ds4
from tests.test_finetune_ds4_forward_parity_readiness import (
    BLOCKERS,
    EXPECTED_FIXTURE_NAMES,
    EXPECTED_REAL_MODE_PROOF_IDS,
    stub_partials,
    stub_real_mode_proofs,
)


EXPECTED_CANDIDATES = [
    "official_hf_inference_convert_py",
    "official_hf_inference_kernel_py",
    "transformers_deepseek_v4_modeling",
    "our_torch_numpy_int8_decode_e8m0",
    "fp8_shim_f8_e8m0_to_bf16",
    "ds4_cpu_harness",
]
EXPECTED_VERDICTS = [
    "assumes_other_packing",
    "no_i8_e8m0_block16_op",
    "delegates_no_i8_backend",
    "circular",
    "circular",
    "built_adjudicated_independent",
]


def present_probe():
    return {
        "probe_status": "present-probed",
        "probe_ran": True,
        "probe_kind": "header-only/introspection",
        "import_ok": True,
        "import_error": None,
        "official_inference_convert_py_present": True,
        "official_inference_kernel_py_present": True,
        "convert_py_assertion_string_confirmed": True,
        "transformers_deepseek_v4_experts_present": True,
        "ds4_c_checked": True,
        "ds4_c_routed_i8_e8m0_block16_op_present": False,
        "read_payload_bytes": False,
        "payload_bytes_decoded": False,
    }


class B2RoutedDequantTrustedReferenceReadinessTests(unittest.TestCase):
    def block(self, probe=None):
        probe = finetune_ds4._default_b2_routed_dequant_trusted_reference_probe() if probe is None else probe
        return finetune_ds4._b2_routed_dequant_trusted_reference_readiness(probe)

    def test_default_skip_is_fail_closed_and_payload_safe(self):
        probe = finetune_ds4._default_b2_routed_dequant_trusted_reference_probe()
        block = finetune_ds4._b2_routed_dequant_trusted_reference_readiness(probe)

        self.assertEqual(block["block"], "b2-routed-dequant-trusted-reference-readiness")
        self.assertEqual(block["blocker_id"], "B2")
        self.assertEqual(block["blocker"], finetune_ds4.FORWARD_PARITY_BLOCKER_DESCRIPTIONS[2])
        self.assertEqual(block["probe_status"], "skipped")
        self.assertIs(block["probe_ran"], False)
        self.assertEqual(block["status"], "fail-closed")
        self.assertEqual(block["decision"], "not-ready")
        self.assertIs(block["fail_closed"], True)
        self.assertIs(block["ready"], False)
        self.assertIs(block["proof_available"], True)
        self.assertIs(block["aggregate_decode_reference_available"], False)
        self.assertIs(block["decode_trusted_reference_available"], True)
        self.assertIs(block["real_payload_decoded"], True)
        self.assertIs(block["can_decode_payload"], False)
        self.assertEqual(block["routed_layout"], {"axis": 1, "block_size": 16, "weight_dtype": "I8", "scale_dtype": "F8_E8M0"})
        self.assertEqual(len(block["candidate_reference_landscape"]), 6)
        self.assertIs(block["evidence"]["read_payload_bytes"], False)
        self.assertIs(block["evidence"]["payload_bytes_decoded"], False)
        self.assertIs(block.get("read_payload_bytes", False), False)
        self.assertIs(block.get("payload_bytes_decoded", False), False)

    def test_landscape_has_all_six_candidates_with_correct_verdicts(self):
        block = self.block(present_probe())
        landscape = block["candidate_reference_landscape"]

        self.assertEqual([item["candidate"] for item in landscape], EXPECTED_CANDIDATES)
        self.assertEqual([item["verdict"] for item in landscape], EXPECTED_VERDICTS)
        self.assertTrue(all(item["binding_adr"] == "ADR 0007 §4" for item in landscape))
        self.assertEqual(landscape[0]["assumes_packing"], "e2m1fn-fp4-per-32")
        self.assertIs(landscape[1]["i8_e8m0_block16_op_present"], False)
        self.assertEqual(landscape[1]["ops_present"], ["fp8_gemm(128-block)", "fp4_gemm(32-block)", "act_quant"])
        self.assertEqual(landscape[2]["decoration"], "@use_experts_implementation backends")
        self.assertIs(landscape[2]["backend_defines_i8_e8m0_block16"], False)
        self.assertEqual(landscape[3]["formula"], "int8 * decode_e8m0(scale)")
        self.assertIn("decode_e8m0", landscape[4]["formula"])
        self.assertIs(landscape[5]["independence_undetermined"], False)

    def test_convert_py_mismatch_is_pinned_machine_readable(self):
        mismatch = self.block(present_probe())["convert_py_mismatch"]

        self.assertEqual(mismatch["assumes_packing"], "e2m1fn-fp4-per-32")
        self.assertEqual(mismatch["assertion"], "scale.size(1) == in_dim // fp4_block_size")
        self.assertEqual(mismatch["fp4_block_size"], 32)
        self.assertEqual(mismatch["expected_scale_dim1_for_real_w1"], 64)
        self.assertEqual(mismatch["observed_scale_dim1_for_real_w1"], 128)
        self.assertIs(mismatch["assertion_fails_on_real_checkpoint"], True)
        self.assertEqual(mismatch["binding_adr"], "ADR 0007 §4")

    def test_no_false_positive_policy_and_future_proof_criteria_are_pinned(self):
        block = self.block(present_probe())

        self.assertIn("NEVER flips ready", block["no_false_positive_policy"])
        self.assertIn("independence_test", block["no_false_positive_policy"])
        self.assertIn("<=1e-5", block["no_false_positive_policy"])
        criteria = "\n".join(block["future_proof_criteria"])
        self.assertIn("independent trusted routed I8+F8_E8M0", criteria)
        self.assertIn("real_mode_proofs entry", criteria)
        self.assertIn(".deepseek-v4-forward-parity-ok", criteria)
        self.assertIn("landscape block alone is never a marker", criteria)
        self.assertIn("circular", block["independence_test"]["policy"])
        self.assertIn("forbidden slop", block["independence_test"]["policy"])
        self.assertEqual(block["independence_test"]["determination_owner"], "Architect (future criterion), recorded here, NOT a BA/Coder in-slice build.")
        self.assertIn("no new DS4-CPU harness is built in 11.41", block["gap_registry"][0])

    def test_present_probe_still_fail_closed(self):
        block = self.block(present_probe())

        self.assertEqual(block["probe_status"], "present-probed")
        self.assertIs(block["probe_ran"], True)
        self.assertEqual(block["status"], "fail-closed")
        self.assertEqual(block["decision"], "not-ready")
        self.assertIs(block["fail_closed"], True)
        self.assertIs(block["ready"], False)
        self.assertIs(block["aggregate_decode_reference_available"], False)
        self.assertIs(block["decode_trusted_reference_available"], True)
        self.assertIs(block["real_payload_decoded"], True)
        self.assertIs(block["proof_available"], True)

    def test_readiness_report_counters_unchanged_and_markers_absent(self):
        with tempfile.TemporaryDirectory() as td:
            mlx = pathlib.Path(td) / "mlx"
            report = finetune_ds4.build_forward_parity_readiness_report(
                mlx_work=mlx,
                partials=stub_partials(),
                blockers=BLOCKERS,
                marker_present=False,
                model_constructs=True,
                non_forward_gate_status={".deepseek-v4-import-ok": "present"},
                generated_at="2026-06-20T00:00:00+00:00",
                real_mode_proofs=stub_real_mode_proofs(),
                b2_routed_dequant_trusted_reference_seam_probe=present_probe(),
            )

            self.assertEqual(report["schema"], 1)
            self.assertEqual(report["real_mode_proofs"]["proofs_total"], len(EXPECTED_REAL_MODE_PROOF_IDS))
            self.assertEqual(report["real_mode_proofs"]["proofs_ok"], len(EXPECTED_REAL_MODE_PROOF_IDS))
            self.assertEqual(report["blockers_count"], 4)
            self.assertEqual(report["coverage"]["fixtures_total"], len(EXPECTED_FIXTURE_NAMES))
            self.assertIs(report["full_forward_parity"], False)
            self.assertIs(report["marker_earned"], False)
            self.assertFalse(any("b2-routed-dequant-trusted-reference" in item.get("id", "") for item in report["real_mode_proofs"]["proofs"]))
            self.assertIn("b2_routed_dequant_trusted_reference_readiness", report)
            self.assertIs(report["b2_routed_dequant_trusted_reference_readiness"]["fail_closed"], True)
            self.assertFalse((mlx / ".deepseek-v4-forward-parity-ok").exists())
            self.assertFalse((mlx / "model-4bit").exists())

    def test_builder_does_not_import_mlx_or_torch(self):
        before = set(sys.modules)
        finetune_ds4._b2_routed_dequant_trusted_reference_readiness(present_probe())
        added = set(sys.modules) - before
        self.assertNotIn("mlx", added)
        self.assertNotIn("torch", added)


if __name__ == "__main__":
    unittest.main()
