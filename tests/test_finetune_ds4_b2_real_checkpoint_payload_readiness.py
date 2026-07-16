#!/usr/bin/env python3
import sys
import unittest

from scripts import finetune_ds4
from tests.test_deepseek_v4_dequant_parity import DECLARED_FP8_QUANT, real_packing_header_fixture


REPO_ROOT = finetune_ds4.pathlib.Path(__file__).resolve().parents[1]
MLX_SRC = REPO_ROOT / "python-envs" / "mlx" / "src"
if str(MLX_SRC) not in sys.path:
    sys.path.insert(0, str(MLX_SRC))

from ds4_ft_mlx import deepseek_v4_dequant as dq  # noqa: E402


class B2RealCheckpointPayloadReadinessTests(unittest.TestCase):
    def classified_probe(self):
        classification = dq.classify_checkpoint_expert_packing(
            real_packing_header_fixture(),
            declared_quant=DECLARED_FP8_QUANT,
        )
        return {
            "probe_status": "classified",
            "import_ok": True,
            "import_error": None,
            "probe_target": "original HF F8 checkpoint (HF_MODEL/DS4_HF_MODEL)",
            "checkpoint_dir": "/tmp/original-hf-f8",
            "index_sha256": "f" * 64,
            "declared_quant": dict(DECLARED_FP8_QUANT),
            "classification": classification,
            "routed_layout": dq.resolve_routed_block_layout(classification),
            "routed_layout_status": "shape_authoritative",
            "reconciled": dq.reconcile_routed_block_layout(classification),
            "read_payload_bytes": False,
            "payload_bytes_decoded": False,
        }

    def test_default_skip_is_fail_closed_and_payload_safe(self):
        probe = finetune_ds4._default_b2_real_checkpoint_payload_probe()
        block = finetune_ds4._b2_real_checkpoint_payload_readiness(probe)

        self.assertEqual(block["block"], "b2-real-checkpoint-payload-readiness")
        self.assertEqual(block["blocker_id"], "B2")
        self.assertEqual(block["blocker"], finetune_ds4.FORWARD_PARITY_BLOCKER_DESCRIPTIONS[2])
        self.assertEqual(block["status"], "fail-closed")
        self.assertEqual(block["decision"], "not-ready")
        self.assertIs(block["fail_closed"], True)
        self.assertIs(block["ready"], False)
        self.assertIs(block["proof_available"], False)
        self.assertIs(block["real_payload_classified"], False)
        self.assertIs(block["can_decode_payload"], False)
        self.assertIs(block["real_payload_decoded"], True)
        self.assertIs(block["read_payload_bytes"], False)
        self.assertIs(block["payload_bytes_decoded"], False)
        self.assertIsNone(block["routed"])
        self.assertIsNone(block["shared"])
        self.assertEqual(block["evidence"]["probe_status"], "skipped")

    def test_classified_probe_records_real_payload_geometry_but_stays_fail_closed(self):
        block = finetune_ds4._b2_real_checkpoint_payload_readiness(self.classified_probe())

        self.assertIs(block["real_payload_classified"], True)
        self.assertIs(block["proof_available"], False)
        self.assertEqual(block["status"], "fail-closed")
        self.assertEqual(block["decision"], "not-ready")
        self.assertIs(block["fail_closed"], True)
        self.assertIs(block["ready"], False)
        self.assertIs(block["fp4_absent"], True)
        self.assertIs(block["adr_0007_binding"], True)
        self.assertEqual(block["index_sha256"], "f" * 64)
        self.assertIs(block["read_payload_bytes"], False)
        self.assertIs(block["payload_bytes_decoded"], False)

        self.assertEqual(block["routed"]["weight_dtype"], "I8")
        self.assertEqual(block["routed"]["scale_dtype"], "F8_E8M0")
        self.assertEqual(block["routed"]["geometry"], {"axis": 1, "block_size": 16})
        self.assertEqual(block["routed"]["weight_shape_example"], [2048, 2048])
        self.assertEqual(block["routed"]["scale_shape_example"], [2048, 128])
        self.assertEqual(block["routed"]["layout_status"], "shape_authoritative")

        self.assertEqual(block["shared"]["weight_dtype"], "F8_E4M3")
        self.assertEqual(block["shared"]["scale_dtype"], "F8_E8M0")
        self.assertEqual(block["shared"]["geometry"], {"block_size": [128, 128]})
        self.assertEqual(block["shared"]["weight_shape_example"], [2048, 4096])
        self.assertEqual(block["shared"]["scale_shape_example"], [16, 32])

        declared = block["declared_quantization_config"]
        self.assertIs(declared["advisory"], True)
        self.assertEqual(declared["weight_block_size"], [128, 128])
        self.assertEqual(declared["fmt"], "e4m3")
        self.assertEqual(declared["scale_fmt"], "ue8m0")
        self.assertEqual(declared["quant_method"], "fp8")
        self.assertIs(block["routed_block_size_discrepancy"], True)

    def test_missing_trusted_reference_strings_are_exactly_fail_closed(self):
        block = finetune_ds4._b2_real_checkpoint_payload_readiness(self.classified_probe())

        self.assertEqual(block["dequantize_expert_packed_i8_status"], {
            "full_metadata_path": "non-raising (returns decoded floats, Story 11.22, circular per ADR 0007 §4)",
            "no_metadata_guard": "raises NotImplementedError (block_size_missing/scale_axis_missing, guard intact)",
            "fp4_path": "raises (fail-closed; packed FP4 expert dequant remains unproven)",
            "nuance": "Story 11.22 opened the full-metadata i8 dispatch path; 11.43 adjudicates independence via the Metal MSL surface without touching dequantize_expert_packed gate. No gate lift.",
        })
        self.assertEqual(block["dequantize_expert_packed_fp4_status"], "NotImplementedError (raises)")
        self.assertIs(block["decode_trusted_reference_available"], True)
        self.assertIs(block["can_decode_payload"], False)
        self.assertIs(block["real_payload_decoded"], True)
        self.assertIs(block["proof_available"], False)
        required = block["trusted_reference_required"]
        self.assertIn("independent routed I8+F8_E8M0 1-D block_size=16 axis=1", required)
        self.assertIn("official DeepseekV4 routed-expert dequant", required)
        self.assertIn("DS4-CPU harness", required)
        self.assertIn("circular", required)
        self.assertIn("Fp8Dequantize", required)
        self.assertIn("Mxfp4Dequantize", required)

    def test_i8_dispatch_evidence_prevents_synthetic_overclaim(self):
        block = finetune_ds4._b2_real_checkpoint_payload_readiness(self.classified_probe())
        evidence = block["i8_dispatch_evidence"]

        self.assertEqual(evidence["synthetic_proof_ids"], ["B2-a-1", "B2-a-2", "B2-a-3"])
        self.assertIs(evidence["synthetic_i8_dispatch_proven"], True)
        self.assertIs(evidence["real_checkpoint_payload_decode_proven"], False)
        self.assertIs(evidence["fp8_shim_is_independent_reference"], False)
        self.assertIn("synthetic", evidence["note"])
        self.assertIn("real-checkpoint", evidence["note"])
        self.assertEqual(
            [item["id"] for item in block["b2_partial_evidence"]],
            ["B2-a-1", "B2-a-2", "B2-a-3"],
        )
        self.assertTrue(all(item["proves_real_payload_decode"] is False for item in block["b2_partial_evidence"]))
        self.assertIn("ONLY synthetic I8", block["b2_partial_non_claim"])
        self.assertIn("production `_moe_mlx` is the expert kernel", "\n".join(block["gap_registry"]))

    def test_no_false_positive_policy_and_future_proof_criteria_are_pinned(self):
        block = finetune_ds4._b2_real_checkpoint_payload_readiness(self.classified_probe())

        self.assertIs(block["real_payload_classified"], True)
        self.assertIs(block["ready"], False)
        self.assertEqual(block["decision"], "not-ready")
        self.assertIn("BOTH", block["no_false_positive_policy"])
        self.assertIn("independent trusted routed I8+F8_E8M0", block["no_false_positive_policy"])
        self.assertIn("reviewed real-mode real-payload decode proof", block["no_false_positive_policy"])
        criteria = "\n".join(block["future_proof_criteria"])
        self.assertIn("independent trusted routed I8+F8_E8M0", criteria)
        self.assertIn("public load_weights()", criteria)
        self.assertIn("<=1e-5", criteria)
        self.assertIn("shared F8_E4M3+F8_E8M0", criteria)
        self.assertIn(".deepseek-v4-forward-parity-ok", criteria)

    def test_builder_does_not_import_mlx_or_torch(self):
        # The mlx venv's editable ds4_ft_mlx install preloads torch/mlx at
        # interpreter site-init (verified: -S -> torch*/mlx* = 0/0, default ->
        # 1050/32), so an absolute assertNotIn(..., sys.modules) is env-broken.
        # Snapshot-diff: the readiness path itself must add no new torch/mlx
        # (top-level or submodule) keys to sys.modules. (ADR 0018.)
        def _is_torch_or_mlx(name):
            return (name == "torch" or name.startswith("torch.")
                    or name == "mlx" or name.startswith("mlx."))
        before = {k for k in sys.modules if _is_torch_or_mlx(k)}
        finetune_ds4._b2_real_checkpoint_payload_readiness(self.classified_probe())
        added = {k for k in sys.modules if _is_torch_or_mlx(k)} - before
        self.assertFalse(
            added,
            f"B2 real-checkpoint readiness path imported torch/mlx (new sys.modules keys): {sorted(added)}",
        )


if __name__ == "__main__":
    unittest.main()
