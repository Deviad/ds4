#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_SRC = REPO_ROOT / "python-envs" / "mlx" / "src"


class DeepSeekV4ForwardParityTests(unittest.TestCase):
    def setUp(self):
        self._old_path = list(sys.path)
        sys.path.insert(0, str(MLX_SRC))

    def tearDown(self):
        sys.path[:] = self._old_path

    def test_forward_parity_blockers_are_explicit(self):
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import forward_parity_blockers

        blockers = forward_parity_blockers()
        self.assertEqual(len(blockers), 4)
        self.assertEqual(blockers[0], "full attention parity with RoPE/cache/sinks/compressor/indexer")
        self.assertEqual(blockers[1], "full decoder-layer hyperconnection residual mixing and final hyperhead parity")
        self.assertEqual(blockers[2], "full MoE parity with packed FP4/I8 expert dequant and expert kernels")
        self.assertIn("full MLX shimmed-checkpoint load/forward and MLX generation smoke (Track B only; DS4-GGUF base generation is its own Track-A gate, .ds4-gguf-generate-ok, not a forward-parity blocker)", blockers)
        self.assertNotIn("over the shimmed checkpoint and generation smoke", "\n".join(blockers))

    def test_tiny_embedding_norm_head_fixture_matches_manual_reference(self):
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, ModelArgs

        args = ModelArgs.from_dict({
            "model_type": "deepseek_v4",
            "vocab_size": 3,
            "hidden_size": 2,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "q_lora_rank": 2,
            "qk_rope_head_dim": 1,
            "n_routed_experts": 1,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 2,
            "expert_dtype": "fp4",
            "forward_parity_fixture": "embedding-rmsnorm-head",
            "rms_norm_eps": 0.0,
        })
        model = Model(args)
        model.load_tiny_weights({
            "embed.weight": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "norm.weight": [1.0, 0.5],
            "lm_head.weight": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        })
        logits = model([[0, 1]])
        # token 0: rms=sqrt((1^2+2^2)/2)=sqrt(2.5), norm=[1/sqrt(2.5), 2/sqrt(2.5)*0.5]
        # token 1: rms=sqrt((3^2+4^2)/2)=sqrt(12.5), norm=[3/sqrt(12.5), 4/sqrt(12.5)*0.5]
        import math
        expected = [[
            [1.0 / math.sqrt(2.5), 1.0 / math.sqrt(2.5), 2.0 / math.sqrt(2.5)],
            [3.0 / math.sqrt(12.5), 2.0 / math.sqrt(12.5), 5.0 / math.sqrt(12.5)],
        ]]
        for got_row, expected_row in zip(logits[0], expected[0], strict=True):
            for got, exp in zip(got_row, expected_row, strict=True):
                self.assertAlmostEqual(got, exp, places=12)

    def test_forward_parity_check_fails_closed_and_removes_stale_marker(self):
        from scripts import finetune_ds4

        with tempfile.TemporaryDirectory() as td:
            mlx = Path(td) / "mlx"
            mlx.mkdir()
            marker = mlx / ".deepseek-v4-forward-parity-ok"
            marker.write_text("stale\n", encoding="utf-8")
            args = type("Args", (), {"mlx_work": str(mlx)})()
            with self.assertRaises(finetune_ds4.PlanError):
                finetune_ds4.deepseek_v4_forward_parity_check(args)
            self.assertFalse(marker.exists())
            report = (mlx / "deepseek-v4-forward-parity-partial.json")
            self.assertTrue(report.exists())
            report_data = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(report_data["partial"]["fixture"], "embedding-rmsnorm-head")
            self.assertEqual(report_data["additional_partials"][0]["fixture"], "sliding-attention-no-compressor-no-rope")
            self.assertEqual(report_data["additional_partials"][1]["fixture"], "rope-tail-pairwise")
            self.assertEqual(report_data["additional_partials"][2]["fixture"], "sink-cache-inverse-rope-hca-bias")
            self.assertEqual(report_data["additional_partials"][3]["fixture"], "hyperconnection-hc1-collapse")
            self.assertEqual(report_data["additional_partials"][4]["fixture"], "hyperconnection-hc2-pre-post-comb")
            self.assertEqual(report_data["additional_partials"][5]["fixture"], "hyperconnection-hc2-transformers")
            self.assertEqual(report_data["additional_partials"][6]["fixture"], "csa-compressor-forward")
            self.assertEqual(report_data["additional_partials"][7]["fixture"], "csa-topk-indexer-gather-mask")
            self.assertEqual(report_data["additional_partials"][8]["fixture"], "hca-compressor-forward")
            self.assertEqual(report_data["additional_partials"][9]["fixture"], "final-hyperhead-hc2-collapse")
            self.assertEqual(report_data["additional_partials"][10]["fixture"], "topk-moe-unquantized")
            self.assertEqual(report_data["additional_partials"][11]["fixture"], "hash-moe-tid2eid-unquantized")
            integrated_partials = [p for p in report_data["additional_partials"] if p.get("fixture") == "integrated-layer-attention-moe"]
            try:
                import torch  # noqa: F401
                import transformers  # noqa: F401
                has_torch = True
            except ImportError:
                has_torch = False
            if has_torch:
                self.assertEqual(integrated_partials[0]["status"], "ok")
            elif integrated_partials:
                self.assertEqual(integrated_partials[0]["status"], "skipped")

    def test_integrated_layer_forward_matches_transformers_reference(self):
        try:
            import torch
            from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
            from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Model
        except ImportError:
            self.skipTest("torch/transformers not available in this environment")

        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, ModelArgs, _make_integrated_tiny_weights

        args = ModelArgs.from_dict({
            "model_type": "deepseek_v4",
            "vocab_size": 4,
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "q_lora_rank": 4,
            "o_lora_rank": 4,
            "qk_rope_head_dim": 4,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 2,
            "n_shared_experts": 1,
            "expert_dtype": "fp4",
            "forward_parity_fixture": "integrated-layer",
            "rms_norm_eps": 1e-6,
            "hc_mult": 1,
            "hc_eps": 1e-6,
            "hc_sinkhorn_iters": 1,
            "layer_types": ["sliding_attention"],
            "mlp_layer_types": ["moe"],
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.0,
            "swiglu_limit": 10.0,
            "rope_theta": 10000.0,
            "sliding_window": 128,
            "o_groups": 1,
        })
        pure = Model(args)
        weights = _make_integrated_tiny_weights(nonzero_hc=True)
        pure.load_integrated_weights(weights)

        config = DeepseekV4Config(
            vocab_size=4,
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            q_lora_rank=4,
            o_lora_rank=4,
            qk_rope_head_dim=4,
            num_experts_per_tok=1,
            n_routed_experts=2,
            moe_intermediate_size=2,
            n_shared_experts=1,
            expert_dtype="fp4",
            rms_norm_eps=1e-6,
            hc_mult=1,
            hc_eps=1e-6,
            hc_sinkhorn_iters=1,
            layer_types=["sliding_attention"],
            mlp_layer_types=["moe"],
            scoring_func="sqrtsoftplus",
            routed_scaling_factor=1.0,
            swiglu_limit=10.0,
            rope_theta=10000.0,
            sliding_window=128,
            o_groups=1,
            rope_parameters={
                "main": {"rope_type": "default", "rope_theta": 10000.0},
                "compress": {"rope_type": "default", "rope_theta": 160000.0},
            },
        )
        ref_model = DeepseekV4Model(config)
        ref_model.eval()
        ref_model.hc_head = torch.nn.Identity()
        ref_model.norm = torch.nn.Identity()
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import set_transformers_integrated_weights
        set_transformers_integrated_weights(ref_model, weights)

        input_ids = [[0, 1]]
        pure_out = pure(input_ids)
        with torch.no_grad():
            ref_out = ref_model(torch.tensor(input_ids, dtype=torch.long)).last_hidden_state
        ref = ref_out.squeeze(0).squeeze(1).tolist()

        max_abs_error = 0.0
        for got_row, ref_row in zip(pure_out[0], ref, strict=True):
            for got, want in zip(got_row, ref_row, strict=True):
                max_abs_error = max(max_abs_error, abs(float(got) - float(want)))
        self.assertLessEqual(max_abs_error, 1e-5, f"integrated layer max_abs_error {max_abs_error}")

    def test_integrated_layer_fixture_accepts_finite_sink_and_rejects_missing_reference(self):
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, ModelArgs, _make_integrated_tiny_weights, run_integrated_layer_fixture

        args = ModelArgs.from_dict({
            "model_type": "deepseek_v4",
            "vocab_size": 4,
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "q_lora_rank": 4,
            "o_lora_rank": 4,
            "qk_rope_head_dim": 4,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 2,
            "n_shared_experts": 1,
            "expert_dtype": "fp4",
            "forward_parity_fixture": "integrated-layer",
            "rms_norm_eps": 1e-6,
            "hc_mult": 1,
            "hc_eps": 1e-6,
            "hc_sinkhorn_iters": 1,
            "layer_types": ["sliding_attention"],
            "mlp_layer_types": ["moe"],
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.0,
            "swiglu_limit": 10.0,
            "rope_theta": 10000.0,
            "sliding_window": 128,
            "o_groups": 1,
        })
        model = Model(args)
        weights = _make_integrated_tiny_weights()
        weights["sinks"] = [0.0]
        model.load_integrated_weights(weights)
        got = model([[0, 1]])
        self.assertEqual(len(got), 1)
        self.assertEqual(len(got[0]), 2)
        self.assertEqual(len(got[0][0]), args.hidden_size)

        weights["sinks"] = [-1e9]
        model.load_integrated_weights(weights)
        with self.assertRaisesRegex(ValueError, "requires a trusted reference_output"):
            run_integrated_layer_fixture(reference_output=None)


if __name__ == "__main__":
    unittest.main()
