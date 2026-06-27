#!/usr/bin/env python3
import json
import math
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_SRC = REPO_ROOT / "python-envs" / "mlx" / "src"


def _softplus_ref(x: float) -> float:
    if x > 20:
        return x
    if x < -20:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _score_ref(logit: float) -> float:
    return math.sqrt(_softplus_ref(logit))


def _routing_logits(router: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(float(x) * float(w) for x, w in zip(vector, row)) for row in router]


def _routing_fixture_256():
    from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig

    cfg = MoEConfig(hidden_size=4, moe_intermediate_size=1, n_routed_experts=256, num_experts_per_tok=6, n_shared_experts=1)
    hidden = [[0.7, -1.1, 2.3, 0.5]]
    router = []
    for i in range(cfg.n_routed_experts):
        router.append([
            ((i % 17) - 8) * 0.07,
            ((i % 13) - 6) * -0.03,
            ((i % 5) - 2) * 0.11,
            (((i // 7) % 9) - 4) * 0.02,
        ])
    bias = [(((i * 37) % 23) - 11) * 0.01 for i in range(cfg.n_routed_experts)]
    return cfg, hidden, {"router.weight": router, "router.e_score_correction_bias": bias}


class DeepSeekV4MoESpecTests(unittest.TestCase):
    def _assert_model_4bit_absent_or_valid_current_artifact(self):
        model_4bit = Path("/Volumes/Data NVME/mlx-ft/ds4/model-4bit")
        if not model_4bit.exists():
            return
        cfg = json.loads((model_4bit / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg.get("model_type"), "deepseek_v4_nn")
        self.assertEqual(cfg.get("quantization", {}).get("bits"), 4)

    def setUp(self):
        self._old_path = list(sys.path)
        sys.path.insert(0, str(MLX_SRC))

    def tearDown(self):
        sys.path[:] = self._old_path

    def test_classifies_routed_shared_and_router_tensor_names(self):
        from ds4_ft_mlx.deepseek_v4_moe_spec import classify_moe_tensor_name

        routed = classify_moe_tensor_name("layers.7.ffn.experts.12.w1.weight")
        self.assertEqual(routed.family, "routed_expert")
        self.assertEqual(routed.layer, 7)
        self.assertEqual(routed.expert_id, 12)
        self.assertEqual(routed.projection, "w1")
        self.assertEqual(routed.kind, "weight")

        shared = classify_moe_tensor_name("layers.7.ffn.shared_experts.w3.scale")
        self.assertEqual(shared.family, "shared_expert")
        self.assertEqual(shared.layer, 7)
        self.assertIsNone(shared.expert_id)
        self.assertEqual(shared.projection, "w3")
        self.assertEqual(shared.kind, "scale")

        gate = classify_moe_tensor_name("layers.7.ffn.gate.tid2eid")
        self.assertEqual(gate.family, "router")
        self.assertEqual(gate.kind, "tid2eid")
        self.assertIsNone(classify_moe_tensor_name("layers.7.attn.wq_a.weight"))

    def test_expert_w1_w2_w3_shape_semantics_are_explicit(self):
        from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig, expected_expert_shape

        cfg = MoEConfig(hidden_size=16, moe_intermediate_size=6, n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=3)
        self.assertEqual(expected_expert_shape(cfg, "routed_expert", "w1"), (6, 16))
        self.assertEqual(expected_expert_shape(cfg, "routed_expert", "w3"), (6, 16))
        self.assertEqual(expected_expert_shape(cfg, "routed_expert", "w2"), (16, 6))
        self.assertEqual(expected_expert_shape(cfg, "shared_expert", "w1"), (18, 16))
        self.assertEqual(expected_expert_shape(cfg, "shared_expert", "w3"), (18, 16))
        self.assertEqual(expected_expert_shape(cfg, "shared_expert", "w2"), (16, 18))

    def test_validates_complete_tiny_moe_manifest_and_router_metadata(self):
        from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig, TensorMeta, validate_moe_manifest

        cfg = MoEConfig(hidden_size=16, moe_intermediate_size=6, n_routed_experts=2, num_experts_per_tok=1, n_shared_experts=1)
        manifest = {
            "layers.0.ffn.gate.weight": TensorMeta(shape=(2, 16), dtype="BF16"),
            "layers.0.ffn.gate.bias": TensorMeta(shape=(2,), dtype="BF16"),
            "layers.0.ffn.gate.tid2eid": TensorMeta(shape=(2,), dtype="I32"),
            "layers.0.ffn.experts.0.w1.weight": TensorMeta(shape=(6, 16), dtype="BF16"),
            "layers.0.ffn.experts.0.w2.weight": TensorMeta(shape=(16, 6), dtype="BF16"),
            "layers.0.ffn.experts.0.w3.weight": TensorMeta(shape=(6, 16), dtype="BF16"),
            "layers.0.ffn.experts.1.w1.weight": TensorMeta(shape=(6, 16), dtype="BF16"),
            "layers.0.ffn.experts.1.w2.weight": TensorMeta(shape=(16, 6), dtype="BF16"),
            "layers.0.ffn.experts.1.w3.weight": TensorMeta(shape=(6, 16), dtype="BF16"),
            "layers.0.ffn.shared_experts.w1.weight": TensorMeta(shape=(6, 16), dtype="BF16"),
            "layers.0.ffn.shared_experts.w2.weight": TensorMeta(shape=(16, 6), dtype="BF16"),
            "layers.0.ffn.shared_experts.w3.weight": TensorMeta(shape=(6, 16), dtype="BF16"),
        }
        report = validate_moe_manifest(manifest, cfg, trusted_dequant=False)
        self.assertTrue(report.ok)
        self.assertEqual(report.router_layers[0].weight_shape, (2, 16))
        self.assertTrue(report.router_layers[0].has_bias)
        self.assertTrue(report.router_layers[0].has_tid2eid)
        self.assertEqual(report.routed_experts_by_layer, {0: [0, 1]})
        self.assertEqual(report.shared_layers, [0])

    def test_partial_moe_manifest_is_not_ok_and_lists_missing_tensors(self):
        from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig, TensorMeta, validate_moe_manifest

        cfg = MoEConfig(hidden_size=16, moe_intermediate_size=6, n_routed_experts=2, num_experts_per_tok=1, n_shared_experts=1)
        manifest = {
            "layers.0.ffn.gate.weight": TensorMeta(shape=(2, 16), dtype="BF16"),
            "layers.0.ffn.experts.0.w1.weight": TensorMeta(shape=(6, 16), dtype="BF16"),
        }
        report = validate_moe_manifest(manifest, cfg, trusted_dequant=False)
        self.assertFalse(report.ok)
        self.assertIn("layers.0.ffn.experts.0.w2.weight", report.missing_required_tensors)
        self.assertIn("layers.0.ffn.experts.1.w1.weight", report.missing_required_tensors)
        self.assertIn("layers.0.ffn.shared_experts.w1.weight", report.missing_required_tensors)

    def test_tiny_topk_moe_fixture_matches_manual_reference(self):
        from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig, tiny_topk_moe_forward

        cfg = MoEConfig(hidden_size=1, moe_intermediate_size=1, n_routed_experts=2, num_experts_per_tok=1, n_shared_experts=1)
        weights = {
            "router.weight": [[1.0], [-1.0]],
            "router.e_score_correction_bias": [0.0, 3.0],
            "experts.0.w1": [[12.0]],
            "experts.0.w2": [[1.0]],
            "experts.0.w3": [[20.0]],
            "experts.1.w1": [[1.0]],
            "experts.1.w2": [[2.0]],
            "experts.1.w3": [[3.0]],
            "shared.w1": [[0.0]],
            "shared.w2": [[0.0]],
            "shared.w3": [[0.0]],
        }
        got = tiny_topk_moe_forward(
            cfg,
            [[1.0], [-1.0]],
            weights,
            scoring_func="sqrtsoftplus",
            routed_scaling_factor=1.5,
            swiglu_limit=10.0,
        )
        # token +1 chooses expert 1 because correction bias is added before top-k;
        # output = routed_scaling_factor * w2(silu(w1*x) * w3*x)
        expected_pos = 1.5 * 2.0 * (1.0 / (1.0 + math.exp(-1.0))) * 3.0
        # token -1 also chooses expert 1; up projection is clamped to [-10,10]
        expected_neg = 1.5 * 2.0 * (-1.0 / (1.0 + math.exp(1.0))) * -3.0
        self.assertAlmostEqual(got[0][0], expected_pos, places=12)
        self.assertAlmostEqual(got[1][0], expected_neg, places=12)

    def test_topk_routing_256_top6_selection_matches_independent_reference(self):
        from ds4_ft_mlx.deepseek_v4_moe_spec import tiny_topk_moe_routing

        cfg, hidden, weights = _routing_fixture_256()
        router = weights["router.weight"]
        bias = weights["router.e_score_correction_bias"]
        logits = _routing_logits(router, hidden[0])
        scores = [_score_ref(logit) for logit in logits]
        expected = sorted(range(cfg.n_routed_experts), key=lambda i: (-(scores[i] + bias[i]), i))[: cfg.num_experts_per_tok]

        routing = tiny_topk_moe_routing(cfg, hidden, weights)

        self.assertEqual(routing[0]["selected"], expected)
        self.assertEqual(len(routing[0]["selected"]), 6)

    def test_topk_routing_tie_break_prefers_lower_index(self):
        """Equal score+bias ties are deterministic: lower expert index wins."""
        from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig, tiny_topk_moe_routing

        cfg = MoEConfig(hidden_size=1, moe_intermediate_size=1, n_routed_experts=256, num_experts_per_tok=6, n_shared_experts=1)
        weights = {
            "router.weight": [[0.0] for _ in range(cfg.n_routed_experts)],
            "router.e_score_correction_bias": [1.0 for _ in range(cfg.n_routed_experts)],
        }

        routing = tiny_topk_moe_routing(cfg, [[0.0]], weights)

        self.assertEqual(routing[0]["selected"], [0, 1, 2, 3, 4, 5])
        self.assertNotIn(6, routing[0]["selected"])

    def test_router_scores_sqrtsoftplus_match_reference_at_scale(self):
        from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig, tiny_topk_moe_routing

        cfg = MoEConfig(hidden_size=1, moe_intermediate_size=1, n_routed_experts=256, num_experts_per_tok=6, n_shared_experts=1)
        router = [[((i % 31) - 15) * 0.25] for i in range(cfg.n_routed_experts)]
        router[0] = [40.0]
        router[1] = [-40.0]
        weights = {"router.weight": router, "router.e_score_correction_bias": [0.0] * cfg.n_routed_experts}

        routing = tiny_topk_moe_routing(cfg, [[1.0]], weights)

        logits = _routing_logits(router, [1.0])
        expected_scores = [_score_ref(logit) for logit in logits]
        self.assertEqual(len(routing[0]["scores"]), 256)
        for got, want in zip(routing[0]["scores"], expected_scores, strict=True):
            self.assertAlmostEqual(got, want, places=12)

    def test_contribution_weights_use_unbiased_scores_only(self):
        from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig, tiny_topk_moe_routing

        cfg = MoEConfig(hidden_size=1, moe_intermediate_size=1, n_routed_experts=256, num_experts_per_tok=6, n_shared_experts=1)
        router = [[-5.0] for _ in range(cfg.n_routed_experts)]
        for idx in range(5):
            router[idx] = [4.0]
        router[200] = [-4.0]
        bias = [0.0 for _ in range(cfg.n_routed_experts)]
        bias[200] = 20.0
        weights = {"router.weight": router, "router.e_score_correction_bias": bias}
        scaling = 1.75

        routing = tiny_topk_moe_routing(cfg, [[1.0]], weights, routed_scaling_factor=scaling)[0]

        self.assertEqual(routing["selected"][0], 200)
        selected_scores = [_score_ref(router[idx][0]) for idx in routing["selected"]]
        denom = sum(selected_scores) + 1e-20
        expected_weights = [(score / denom) * scaling for score in selected_scores]
        for got, want in zip(routing["contribution_weights"], expected_weights, strict=True):
            self.assertAlmostEqual(got, want, places=12)
        biased_wrong = ((_score_ref(router[200][0]) + bias[200]) / sum(_score_ref(router[idx][0]) + bias[idx] for idx in routing["selected"])) * scaling
        self.assertNotAlmostEqual(routing["contribution_weights"][0], biased_wrong, places=6)

    def test_routing_combine_uses_synthetic_experts_no_decode(self):
        from ds4_ft_mlx.deepseek_v4_dequant import dequantize_expert_packed
        from ds4_ft_mlx.deepseek_v4_moe_spec import tiny_topk_moe_routing

        cfg, hidden, weights = _routing_fixture_256()
        routing = tiny_topk_moe_routing(cfg, hidden, weights)[0]
        expert_outputs = {idx: [idx / 100.0, -idx / 200.0] for idx in routing["selected"]}
        combined = [0.0, 0.0]
        for factor, idx in zip(routing["contribution_weights"], routing["selected"], strict=True):
            combined = [value + factor * expert_value for value, expert_value in zip(combined, expert_outputs[idx], strict=True)]
        expected = [
            sum(factor * expert_outputs[idx][0] for factor, idx in zip(routing["contribution_weights"], routing["selected"], strict=True)),
            sum(factor * expert_outputs[idx][1] for factor, idx in zip(routing["contribution_weights"], routing["selected"], strict=True)),
        ]
        self.assertEqual(combined, expected)
        with self.assertRaises(NotImplementedError):
            dequantize_expert_packed("i8", b"\x00", scales=b"\x7f", shape=(1,))
        with self.assertRaisesRegex(ValueError, "requires non-None scales"):
            dequantize_expert_packed("fp4", b"\x00" * 16, scales=None, shape=(1, 32))

    def test_tiny_topk_moe_forward_byte_identical_after_routing_extract(self):
        from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig, tiny_topk_moe_forward

        cfg = MoEConfig(hidden_size=1, moe_intermediate_size=1, n_routed_experts=2, num_experts_per_tok=1, n_shared_experts=1)
        weights = {
            "router.weight": [[1.0], [-1.0]],
            "router.e_score_correction_bias": [0.0, 3.0],
            "experts.0.w1": [[12.0]],
            "experts.0.w2": [[1.0]],
            "experts.0.w3": [[20.0]],
            "experts.1.w1": [[1.0]],
            "experts.1.w2": [[2.0]],
            "experts.1.w3": [[3.0]],
            "shared.w1": [[0.0]],
            "shared.w2": [[0.0]],
            "shared.w3": [[0.0]],
        }
        got = tiny_topk_moe_forward(cfg, [[1.0], [-1.0]], weights, scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, swiglu_limit=10.0)
        expected = [
            [1.5 * 2.0 * (1.0 / (1.0 + math.exp(-1.0))) * 3.0],
            [1.5 * 2.0 * (-1.0 / (1.0 + math.exp(1.0))) * -3.0],
        ]
        self.assertEqual(got, expected)

    def test_routing_proof_does_not_relax_vendor_caps(self):
        from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, ModelArgs

        cfg = MoEConfig(hidden_size=4, moe_intermediate_size=1, n_routed_experts=256, num_experts_per_tok=6, n_shared_experts=1)
        self.assertEqual(cfg.n_routed_experts, 256)
        with self.assertRaisesRegex(NotImplementedError, "num_hidden_layers"):
            Model(ModelArgs())

    def test_routing_is_deterministic(self):
        from ds4_ft_mlx.deepseek_v4_moe_spec import tiny_topk_moe_routing

        cfg, hidden, weights = _routing_fixture_256()
        first = tiny_topk_moe_routing(cfg, hidden, weights)
        second = tiny_topk_moe_routing(cfg, hidden, weights)

        self.assertEqual(first[0]["selected"], second[0]["selected"])
        self.assertEqual(first[0]["contribution_weights"], second[0]["contribution_weights"])
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

    def test_forward_marker_and_model_4bit_stay_absent_for_routing_slice(self):
        marker = Path("/Volumes/Data NVME/mlx-ft/ds4/.deepseek-v4-forward-parity-ok")
        self.assertFalse(marker.exists(), f"{marker} must not be written by Story 11.15d")
        self._assert_model_4bit_absent_or_valid_current_artifact()

    def test_topk_routing_matches_torch_selection_or_skip(self):
        from ds4_ft_mlx.deepseek_v4_moe_spec import tiny_topk_moe_routing

        try:
            import torch
        except ImportError:
            self.skipTest("torch not available in this environment")

        cfg, hidden, weights = _routing_fixture_256()
        # Add a tiny monotonic bias to avoid relying on torch.topk tie behavior.
        weights = dict(weights)
        weights["router.e_score_correction_bias"] = [i * 1e-6 for i in range(cfg.n_routed_experts)]
        routing = tiny_topk_moe_routing(cfg, hidden, weights)[0]
        router = torch.tensor(weights["router.weight"], dtype=torch.float64)
        vector = torch.tensor(hidden[0], dtype=torch.float64)
        logits = router @ vector
        scores = torch.sqrt(torch.nn.functional.softplus(logits))
        biased = scores + torch.tensor(weights["router.e_score_correction_bias"], dtype=torch.float64)
        expected = torch.topk(biased, k=cfg.num_experts_per_tok).indices.tolist()

        self.assertEqual(routing["selected"], expected)
        for got, want in zip(routing["scores"], scores.tolist(), strict=True):
            self.assertAlmostEqual(got, want, places=12)

    def test_tiny_hash_moe_fixture_uses_tid2eid_selection(self):
        from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig, tiny_hash_moe_forward

        cfg = MoEConfig(hidden_size=1, moe_intermediate_size=1, n_routed_experts=2, num_experts_per_tok=1, n_shared_experts=1)
        weights = {
            "router.weight": [[1.0], [1.0]],
            "router.tid2eid": [[1], [0]],
            "experts.0.w1": [[1.0]],
            "experts.0.w2": [[1.0]],
            "experts.0.w3": [[1.0]],
            "experts.1.w1": [[2.0]],
            "experts.1.w2": [[3.0]],
            "experts.1.w3": [[4.0]],
            "shared.w1": [[0.0]],
            "shared.w2": [[0.0]],
            "shared.w3": [[0.0]],
        }
        got = tiny_hash_moe_forward(
            cfg,
            [[1.0], [1.0]],
            input_ids=[0, 1],
            weights=weights,
            scoring_func="sqrtsoftplus",
            routed_scaling_factor=1.5,
            swiglu_limit=10.0,
        )
        expected_token0 = 1.5 * 3.0 * (2.0 / (1.0 + math.exp(-2.0))) * 4.0
        expected_token1 = 1.5 * (1.0 / (1.0 + math.exp(-1.0)))
        self.assertAlmostEqual(got[0][0], expected_token0, places=12)
        self.assertAlmostEqual(got[1][0], expected_token1, places=12)

    def test_shape_mismatch_mentions_projection_semantics(self):
        from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig, MoESpecError, TensorMeta, validate_moe_manifest

        cfg = MoEConfig(hidden_size=16, moe_intermediate_size=6, n_routed_experts=1, num_experts_per_tok=1, n_shared_experts=1)
        manifest = {
            "layers.0.ffn.gate.weight": TensorMeta(shape=(1, 16), dtype="BF16"),
            "layers.0.ffn.experts.0.w2.weight": TensorMeta(shape=(6, 16), dtype="BF16"),
        }
        with self.assertRaisesRegex(MoESpecError, r"w2.*expected \(16, 6\).*hidden, intermediate"):
            validate_moe_manifest(manifest, cfg, trusted_dequant=False)

    def test_fp4_and_i8_experts_fail_closed_until_trusted_dequant_parity_exists(self):
        from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig, MoEQuantizationBlocked, TensorMeta, validate_moe_manifest

        cfg = MoEConfig(hidden_size=16, moe_intermediate_size=6, n_routed_experts=1, num_experts_per_tok=1, n_shared_experts=1)
        manifest = {
            "layers.0.ffn.gate.weight": TensorMeta(shape=(1, 16), dtype="BF16"),
            "layers.0.ffn.experts.0.w1.weight": TensorMeta(shape=(6, 16), dtype="FP4"),
        }
        with self.assertRaisesRegex(MoEQuantizationBlocked, "FP4.*trusted dequant parity.*layers.0.ffn.experts.0.w1.weight"):
            validate_moe_manifest(manifest, cfg, trusted_dequant=False)

        manifest["layers.0.ffn.experts.0.w1.weight"] = TensorMeta(shape=(6, 16), dtype="I8")
        with self.assertRaisesRegex(MoEQuantizationBlocked, "I8.*trusted dequant parity.*layers.0.ffn.experts.0.w1.weight"):
            validate_moe_manifest(manifest, cfg, trusted_dequant=False)

    def test_trusted_dequant_flag_allows_quantized_shape_checks_without_claiming_parity(self):
        from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig, TensorMeta, validate_moe_manifest

        cfg = MoEConfig(hidden_size=16, moe_intermediate_size=6, n_routed_experts=1, num_experts_per_tok=1, n_shared_experts=1)
        manifest = {
            "layers.0.ffn.gate.weight": TensorMeta(shape=(1, 16), dtype="BF16"),
            "layers.0.ffn.experts.0.w1.weight": TensorMeta(shape=(6, 16), dtype="FP4"),
            "layers.0.ffn.experts.0.w2.weight": TensorMeta(shape=(16, 6), dtype="FP4"),
            "layers.0.ffn.experts.0.w3.weight": TensorMeta(shape=(6, 16), dtype="FP4"),
            "layers.0.ffn.shared_experts.w1.weight": TensorMeta(shape=(6, 16), dtype="BF16"),
            "layers.0.ffn.shared_experts.w2.weight": TensorMeta(shape=(16, 6), dtype="BF16"),
            "layers.0.ffn.shared_experts.w3.weight": TensorMeta(shape=(6, 16), dtype="BF16"),
        }
        report = validate_moe_manifest(manifest, cfg, trusted_dequant=True)
        self.assertTrue(report.ok)
        self.assertIn("layers.0.ffn.experts.0.w1.weight", report.quantized_expert_tensors)

    def test_tiny_topk_moe_i8_matches_pytorch_reference(self):
        from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig, tiny_topk_moe_i8_forward

        try:
            import torch
        except ImportError:
            self.skipTest("torch not available in this environment")

        cfg = MoEConfig(hidden_size=16, moe_intermediate_size=16, n_routed_experts=2, num_experts_per_tok=1, n_shared_experts=1)
        block_size = 16
        scale_axis = 1
        hidden_size = cfg.hidden_size
        intermediate = cfg.moe_intermediate_size

        def _make_i8_weight(shape):
            weight_f32 = torch.randn(shape, dtype=torch.float32) * 0.1
            scale = weight_f32.abs().amax(dim=scale_axis, keepdim=True) / 127.0
            scale = torch.where(scale == 0, torch.ones_like(scale) * 1e-6, scale)
            i8 = (weight_f32 / scale).round().clamp(-128, 127).to(torch.int8)
            scale_bf16 = scale.to(torch.bfloat16)
            return i8, scale_bf16

        shapes = {
            "w1": (intermediate, hidden_size),
            "w2": (hidden_size, intermediate),
            "w3": (intermediate, hidden_size),
        }
        weights: dict[str, object] = {
            "router.weight": [[0.1] * hidden_size, [-0.1] * hidden_size],
            "router.e_score_correction_bias": [0.0, 0.0],
            "shared.w1": [[0.01] * hidden_size for _ in range(intermediate)],
            "shared.w2": [[0.01] * intermediate for _ in range(hidden_size)],
            "shared.w3": [[0.01] * hidden_size for _ in range(intermediate)],
        }
        reference_float: dict[str, dict[str, torch.Tensor]] = {}
        for eid in range(cfg.n_routed_experts):
            for proj, shape in shapes.items():
                i8, scale_bf16 = _make_i8_weight(shape)
                weights[f"experts.{eid}.{proj}.weight"] = bytes(i8.numpy().tobytes())
                weights[f"experts.{eid}.{proj}.scale"] = bytes(scale_bf16.contiguous().view(torch.uint8).numpy().tobytes())
                scale_f32 = scale_bf16.float()
                scale_repeated = scale_f32.repeat_interleave(block_size, dim=scale_axis)
                reference_float[f"{eid}.{proj}"] = (i8.float() * scale_repeated).numpy()

        hidden = torch.randn(2, hidden_size, dtype=torch.float32) * 0.1
        hidden_list = hidden.tolist()

        got = tiny_topk_moe_i8_forward(
            cfg,
            hidden_list,
            weights,
            scoring_func="sqrtsoftplus",
            routed_scaling_factor=1.0,
            swiglu_limit=10.0,
            block_size=block_size,
            scale_axis=scale_axis,
        )

        # Build a PyTorch reference using the same dequantized floats.
        def _silu_ref(x):
            return x / (1.0 + torch.exp(-x))

        expected: list[list[float]] = []
        for token_idx, vector in enumerate(hidden):
            logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
            for eid in range(cfg.n_routed_experts):
                logits[eid] = sum(vector[d].item() * weights["router.weight"][eid][d] for d in range(hidden_size))
            scores = torch.sqrt(torch.nn.functional.softplus(logits))
            top_idx = int(torch.argmax(scores).item())
            w1 = torch.from_numpy(reference_float[f"{top_idx}.w1"])
            w2 = torch.from_numpy(reference_float[f"{top_idx}.w2"])
            w3 = torch.from_numpy(reference_float[f"{top_idx}.w3"])
            gate = (vector @ w1.T).clamp(max=10.0)
            up = (vector @ w3.T).clamp(min=-10.0, max=10.0)
            expert_out = (_silu_ref(gate) * up) @ w2.T
            shared_w1 = torch.tensor(weights["shared.w1"], dtype=torch.float32)
            shared_w2 = torch.tensor(weights["shared.w2"], dtype=torch.float32)
            shared_w3 = torch.tensor(weights["shared.w3"], dtype=torch.float32)
            shared_gate = (vector @ shared_w1.T).clamp(max=10.0)
            shared_up = (vector @ shared_w3.T).clamp(min=-10.0, max=10.0)
            shared_out = (_silu_ref(shared_gate) * shared_up) @ shared_w2.T
            expected.append((expert_out + shared_out).tolist())

        max_abs_error = 0.0
        for got_row, ref_row in zip(got, expected):
            for gv, rv in zip(got_row, ref_row):
                max_abs_error = max(max_abs_error, abs(gv - rv))
        self.assertLessEqual(max_abs_error, 1e-3)

    def test_run_tiny_topk_moe_i8_fixture_ok_or_skipped(self):
        from ds4_ft_mlx.deepseek_v4_moe_spec import run_tiny_topk_moe_i8_fixture

        report = run_tiny_topk_moe_i8_fixture()
        self.assertEqual(report["fixture"], "topk-moe-i8-block-scale")
        if report["status"] == "skipped":
            self.assertIn("torch", report.get("reason", "").lower())
        else:
            self.assertEqual(report["status"], "ok")
            self.assertLessEqual(report["max_abs_error"], 1e-3)


if __name__ == "__main__":
    unittest.main()
