#!/usr/bin/env python3
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import finetune_ds4
from tests.test_finetune_ds4_forward_parity_readiness import BLOCKERS, stub_partials


EXPECTED_REAL_MODE_PROOF_IDS_1134 = ("B0a-1", "B0a-2", "B0a-3", "B0b-a-1", "B0b-a-2", "B2-a-1", "B2-a-2", "B2-a-3", "B2-a-4")


def _b0ba2_spec():
    return next(spec for spec in finetune_ds4._real_mode_forward_proof_specs() if spec["id"] == "B0b-a-2")


class FakeArray:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class CsaTopkPrimitiveProofTests(unittest.TestCase):
    def test_spec_schema_granularity_reference_and_honesty(self):
        specs = finetune_ds4._real_mode_forward_proof_specs()
        self.assertEqual([spec["id"] for spec in specs], list(EXPECTED_REAL_MODE_PROOF_IDS_1134))
        spec = next(spec for spec in specs if spec["id"] == "B0b-a-2")
        self.assertEqual(spec["name"], "csa-topk-sparse-selection-primitive")
        self.assertEqual(spec["granularity"], "csa-attention-sublayer-primitive")
        self.assertEqual(spec["reference"], "tiny_compressor_indexer_attention_reference(index_topk=1)")
        self.assertEqual(spec["config"], {
            "num_hidden_layers": 1,
            "hc_mult": 1,
            "compression_ratio": 4,
            "final_projection": False,
            "seq_len": 12,
            "index_topk": 1,
            "compressed_len": 3,
        })
        covered = "\n".join(spec["covered"])
        not_covered = "\n".join(spec["not_covered"])
        self.assertIn("_csa_attention_mlx(index_topk<compressed_len)", covered)
        self.assertIn("tiny_compressor_indexer_attention_reference", covered)
        self.assertIn("production real-mode forward strict index_topk", not_covered)
        self.assertIn("MLA latent", not_covered)
        self.assertIn("FlashMLA FP8", not_covered)
        self.assertFalse(any("granularity" in spec for spec in specs if spec["id"] != "B0b-a-2"))

    def test_report_counts_and_fail_closed_honesty_fields_stay_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            report = finetune_ds4.build_forward_parity_readiness_report(
                mlx_work=pathlib.Path(td) / "mlx",
                partials=stub_partials(),
                blockers=BLOCKERS,
                marker_present=False,
                model_constructs=True,
                non_forward_gate_status={},
                generated_at="2026-06-19T00:00:00+00:00",
            )
        self.assertIs(report["full_forward_parity"], False)
        self.assertIs(report["marker_earned"], False)
        self.assertEqual(report["blockers_count"], 4)
        self.assertEqual(report["coverage"]["fixtures_total"], 19)
        self.assertEqual(report["coverage"]["fixtures_failed"], 0)
        self.assertEqual(report["real_mode_proofs"]["proofs_total"], 9)
        self.assertEqual(report["real_mode_proofs"]["proofs_skipped"], 9)
        self.assertIn("csa-attention-sublayer-primitive", report["real_mode_proofs"]["description"])
        self.assertIn("production real-mode forward does NOT expose index_topk", report["real_mode_proofs"]["description"])
        self.assertIn("B0 NOT marked satisfied", report["real_mode_proofs"]["b0_partial_progress"])
        self.assertIn("strict index_topk pruning inside production real-mode forward", "\n".join(report["real_mode_proofs"]["not_covered"]))

    def test_csa_topk_status_requires_accuracy_and_non_degeneracy(self):
        self.assertEqual(
            finetune_ds4._csa_topk_proof_status(0.0, True, 1e-5)[0],
            "ok",
        )
        status, reason = finetune_ds4._csa_topk_proof_status(0.0, False, 1e-5)
        self.assertEqual(status, "failed")
        self.assertIn("pruning_non_degenerate=false", reason)
        status, reason = finetune_ds4._csa_topk_proof_status(2e-5, True, 1e-5)
        self.assertEqual(status, "failed")
        self.assertIn("max_abs_error", reason)

    def test_run_csa_topk_primitive_proof_calls_direct_primitive_and_reference(self):
        calls = {"arrays": []}
        fake_args = types.SimpleNamespace(
            hidden_size=4,
            num_attention_heads=1,
            head_dim=4,
            q_lora_rank=4,
            o_lora_rank=4,
            qk_rope_head_dim=4,
            o_groups=1,
            compression_ratio=4,
            index_n_heads=2,
            index_head_dim=2,
            rms_norm_eps=1e-6,
            compress_rope_theta=10000.0,
        )
        fake_weights = {"w": [[1.0]]}

        def fake_array(value):
            calls["arrays"].append(value)
            return value

        def fake_csa_attention_mlx(args, x, weights, *, index_topk=None):
            calls.setdefault("primitive", []).append((args, x, weights, index_topk))
            if index_topk == 1:
                return FakeArray([[[0.25, 0.5, 0.75, 1.0]]])
            return FakeArray([[[1.25, 0.5, 0.75, 1.0]]])

        class FakeSpec:
            def __init__(self, *args):
                calls["spec_args"] = args

        def fake_reference(spec, q, kv, weights, *, rms_norm_eps, rope_theta, index_topk=None):
            calls["reference"] = (spec, q, kv, weights, rms_norm_eps, rope_theta, index_topk)
            return {"attended": [[0.25, 0.5, 0.75, 1.0]]}

        mlx_module = types.ModuleType("mlx")
        mlx_core = types.ModuleType("mlx.core")
        mlx_core.array = fake_array
        mlx_module.core = mlx_core
        vendor_module = types.ModuleType("deepseek_v4")
        vendor_module._csa_attention_mlx = fake_csa_attention_mlx
        attention_module = types.ModuleType("deepseek_v4_attention_spec")
        attention_module.DeepSeekV4AttentionSpec = FakeSpec
        attention_module.tiny_compressor_indexer_attention_reference = fake_reference
        module_patches = {
            "mlx": mlx_module,
            "mlx.core": mlx_core,
            "ds4_ft_mlx": types.ModuleType("ds4_ft_mlx"),
            "ds4_ft_mlx.deepseek_v4_attention_spec": attention_module,
            "ds4_ft_mlx.vendor": types.ModuleType("vendor"),
            "ds4_ft_mlx.vendor.mlx_lm_models": types.ModuleType("mlx_lm_models"),
            "ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4": vendor_module,
        }
        with mock.patch.dict(sys.modules, module_patches), \
             mock.patch.object(finetune_ds4, "_ensure_mlx_project_src_on_path"), \
             mock.patch.object(finetune_ds4, "_build_real_mode_tiny_args", return_value=fake_args) as build_args, \
             mock.patch.object(finetune_ds4, "_make_real_mode_csa_weights", return_value=fake_weights) as make_weights:
            result = finetune_ds4._run_one_real_mode_forward_proof(_b0ba2_spec())

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["granularity"], "csa-attention-sublayer-primitive")
        self.assertEqual(result["max_abs_error"], 0.0)
        self.assertIs(result["pruning_non_degenerate"], True)
        self.assertEqual(result["pruning_diff_vs_full_topk"], 1.0)
        self.assertEqual(result["config"]["index_topk"], 1)
        self.assertEqual(result["config"]["compressed_len"], 3)
        build_args.assert_called_once_with(1, 1, compression_ratio=4, index_n_heads=2, index_head_dim=2)
        make_weights.assert_called_once_with()
        self.assertEqual([call[3] for call in calls["primitive"]], [1, 3])
        self.assertEqual(calls["reference"][-1], 1)
        q = calls["reference"][1]
        self.assertEqual(len(q), 12)
        self.assertEqual(q, calls["reference"][2])
        self.assertEqual(q[0], [0.1, 0.2552, 0.3873, 0.4942])
        self.assertEqual(calls["spec_args"], (4, 1, 4, 4, 4, 4, 1, 4, 2, 2))

    def test_degenerate_primitive_output_fails_even_when_reference_matches(self):
        calls = {}
        fake_args = types.SimpleNamespace(
            hidden_size=4,
            num_attention_heads=1,
            head_dim=4,
            q_lora_rank=4,
            o_lora_rank=4,
            qk_rope_head_dim=4,
            o_groups=1,
            compression_ratio=4,
            index_n_heads=2,
            index_head_dim=2,
            rms_norm_eps=1e-6,
            compress_rope_theta=10000.0,
        )

        def fake_csa_attention_mlx(args, x, weights, *, index_topk=None):
            calls.setdefault("topks", []).append(index_topk)
            return FakeArray([[[0.25, 0.5, 0.75, 1.0]]])

        mlx_module = types.ModuleType("mlx")
        mlx_core = types.ModuleType("mlx.core")
        mlx_core.array = lambda value: value
        mlx_module.core = mlx_core
        vendor_module = types.ModuleType("deepseek_v4")
        vendor_module._csa_attention_mlx = fake_csa_attention_mlx
        attention_module = types.ModuleType("deepseek_v4_attention_spec")
        attention_module.DeepSeekV4AttentionSpec = lambda *args: object()
        attention_module.tiny_compressor_indexer_attention_reference = lambda *args, **kwargs: {"attended": [[0.25, 0.5, 0.75, 1.0]]}
        module_patches = {
            "mlx": mlx_module,
            "mlx.core": mlx_core,
            "ds4_ft_mlx": types.ModuleType("ds4_ft_mlx"),
            "ds4_ft_mlx.deepseek_v4_attention_spec": attention_module,
            "ds4_ft_mlx.vendor": types.ModuleType("vendor"),
            "ds4_ft_mlx.vendor.mlx_lm_models": types.ModuleType("mlx_lm_models"),
            "ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4": vendor_module,
        }
        with mock.patch.dict(sys.modules, module_patches), \
             mock.patch.object(finetune_ds4, "_ensure_mlx_project_src_on_path"), \
             mock.patch.object(finetune_ds4, "_build_real_mode_tiny_args", return_value=fake_args), \
             mock.patch.object(finetune_ds4, "_make_real_mode_csa_weights", return_value={"w": [[1.0]]}):
            result = finetune_ds4._run_one_real_mode_forward_proof(_b0ba2_spec())

        self.assertEqual(calls["topks"], [1, 3])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["max_abs_error"], 0.0)
        self.assertIs(result["pruning_non_degenerate"], False)
        self.assertIn("pruning_non_degenerate=false", result["reason"])

    def test_no_mlx_or_torch_imports_required_for_unit_tests(self):
        # The mlx venv preloads torch/mlx (incl mlx.core) at site-init (ADR 0018),
        # so assertNotIn(..., sys.modules) is env-broken. This test exercises
        # NO path-under-test (the primitive unit tests inject a FAKE mlx module
        # and restore sys.modules), so a snapshot-diff would be a tautology
        # (added always empty -> assertFalse always passes = dead code = slop,
        # forbidden AGENTS.md). Assert the import-purity invariant STATICALLY:
        # the test module's own source pulls no torch/mlx/mlx.core at module
        # level (AST + source-substring verified). (ADR 0018 Amendment, Story 11.45.)
        import inspect
        source = inspect.getsource(sys.modules[__name__])
        self.assertNotIn("import " + "torch", source)
        self.assertNotIn("import " + "mlx" + ".core", source)
        self.assertNotIn("import " + "mlx" + "\n", source)  # bare `import mlx`, not mlx.core / mlx_lm
        self.assertNotIn("from " + "torch", source)
        self.assertNotIn("from " + "mlx", source)


if __name__ == "__main__":
    unittest.main()
