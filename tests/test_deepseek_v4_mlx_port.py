#!/usr/bin/env python3
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_SRC = REPO_ROOT / "python-envs" / "mlx" / "src"


_TRANSFORMERS_WEIGHT_NAMES = {
    "embed.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
    "input_layernorm.weight": "model.layers.0.input_layernorm.weight",
    "post_attention_layernorm.weight": "model.layers.0.post_attention_layernorm.weight",
    "attn_hc.fn": "model.layers.0.attn_hc.fn",
    "attn_hc.base": "model.layers.0.attn_hc.base",
    "attn_hc.scale": "model.layers.0.attn_hc.scale",
    "ffn_hc.fn": "model.layers.0.ffn_hc.fn",
    "ffn_hc.base": "model.layers.0.ffn_hc.base",
    "ffn_hc.scale": "model.layers.0.ffn_hc.scale",
    "q_a_proj.weight": "model.layers.0.self_attn.q_a_proj.weight",
    "q_norm.weight": "model.layers.0.self_attn.q_a_norm.weight",
    "q_b_proj.weight": "model.layers.0.self_attn.q_b_proj.weight",
    "kv_proj.weight": "model.layers.0.self_attn.kv_proj.weight",
    "kv_norm.weight": "model.layers.0.self_attn.kv_norm.weight",
    "o_a_proj.weight": "model.layers.0.self_attn.o_a_proj.weight",
    "o_b_proj.weight": "model.layers.0.self_attn.o_b_proj.weight",
    "sinks": "model.layers.0.self_attn.sinks",
    "mlp.gate.weight": "model.layers.0.mlp.gate.weight",
    "mlp.gate.e_score_correction_bias": "model.layers.0.mlp.gate.e_score_correction_bias",
    "mlp.experts.0.w1.weight": "model.layers.0.mlp.experts.0.w1.weight",
    "mlp.experts.0.w2.weight": "model.layers.0.mlp.experts.0.w2.weight",
    "mlp.experts.0.w3.weight": "model.layers.0.mlp.experts.0.w3.weight",
    "mlp.experts.1.w1.weight": "model.layers.0.mlp.experts.1.w1.weight",
    "mlp.experts.1.w2.weight": "model.layers.0.mlp.experts.1.w2.weight",
    "mlp.experts.1.w3.weight": "model.layers.0.mlp.experts.1.w3.weight",
    "mlp.shared_experts.w1.weight": "model.layers.0.mlp.shared_experts.w1.weight",
    "mlp.shared_experts.w2.weight": "model.layers.0.mlp.shared_experts.w2.weight",
    "mlp.shared_experts.w3.weight": "model.layers.0.mlp.shared_experts.w3.weight",
    "hc_head.fn": "model.hc_head.fn",
    "hc_head.base": "model.hc_head.base",
    "hc_head.scale": "model.hc_head.scale",
}


def _to_transformers_weight_names(weights):
    mapped = {}
    for key, value in weights.items():
        if key in _TRANSFORMERS_WEIGHT_NAMES:
            mapped[_TRANSFORMERS_WEIGHT_NAMES[key]] = value
        elif key.startswith("mlp.experts.") and key.endswith(".scale"):
            mapped[f"model.layers.0.{key}"] = value
        else:
            raise KeyError(key)
    return mapped


class DeepSeekV4MlxPortTests(unittest.TestCase):
    def setUp(self):
        self._old_path = list(sys.path)
        self._old_modules = dict(sys.modules)
        sys.path.insert(0, str(MLX_SRC))

    def tearDown(self):
        sys.path[:] = self._old_path
        for name in list(sys.modules):
            if name not in self._old_modules:
                # mlx uses nanobind bindings that cannot be safely reloaded.
                if name == "mlx" or name.startswith("mlx."):
                    continue
                del sys.modules[name]
        for name, module in self._old_modules.items():
            sys.modules[name] = module

    def install_fake_mlx_lm_models(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        models_dir = Path(tmp.name) / "mlx_lm" / "models"
        models_dir.mkdir(parents=True)
        mlx_lm = types.ModuleType("mlx_lm")
        mlx_lm.__path__ = [str(models_dir.parent)]
        models = types.ModuleType("mlx_lm.models")
        models.__path__ = [str(models_dir)]
        sys.modules["mlx_lm"] = mlx_lm
        sys.modules["mlx_lm.models"] = models
        return models

    def test_registers_project_controlled_deepseek_v4_module_path(self):
        models = self.install_fake_mlx_lm_models()
        from ds4_ft_mlx.mlx_lm_plugin import install_deepseek_v4_plugin, plugin_models_path

        self.assertNotIn(str(plugin_models_path()), list(models.__path__))
        install_deepseek_v4_plugin()
        self.assertIn(str(plugin_models_path()), list(models.__path__))
        mod = importlib.import_module("mlx_lm.models.deepseek_v4")
        self.assertTrue(mod.__file__.startswith(str(plugin_models_path())))
        self.assertEqual(mod.ModelArgs.model_type, "deepseek_v4")

    def test_startup_pth_hook_is_reproducible(self):
        from ds4_ft_mlx.mlx_lm_plugin import install_startup_pth_hook

        with tempfile.TemporaryDirectory() as td:
            hook = install_startup_pth_hook(td)
            self.assertEqual(hook.name, "ds4_ft_mlx_mlx_lm_plugin.pth")
            self.assertIn("install_deepseek_v4_plugin", hook.read_text(encoding="utf-8"))

    def test_mlx_tail_rope_helper_matches_python_reference_and_preserves_prefix(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.deepseek_v4_attention_spec import apply_rope_tail
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _apply_rope_tail_mlx, _rope_tail_tables_mlx

        x = mx.array([[[10.0, 20.0, 1.0, 2.0, 3.0, 4.0], [30.0, 40.0, 5.0, 6.0, 7.0, 8.0]]])
        cos, sin = _rope_tail_tables_mlx(head_dim=6, qk_rope_head_dim=4, rope_theta=10000.0, seq_len=2)
        got = _apply_rope_tail_mlx(x, cos, sin, qk_rope_head_dim=4).tolist()

        self.assertEqual(got[0][0][:2], [10.0, 20.0])
        self.assertEqual(got[0][1][:2], [30.0, 40.0])
        expected = []
        for pos, row in enumerate(x.tolist()[0]):
            cos_full = [value for value in cos[pos].tolist() for _ in (0, 1)]
            sin_full = [value for value in sin[pos].tolist() for _ in (0, 1)]
            expected.append(apply_rope_tail(row, nope_dim=2, cos=cos_full, sin=sin_full))
        for got_row, exp_row in zip(got[0], expected):
            for gv, ev in zip(got_row, exp_row):
                self.assertAlmostEqual(gv, ev, places=6)

    def test_mlx_tail_rope_tables_use_qk_dim_not_full_head_dim(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.deepseek_v4_attention_spec import _rope_cos_sin_for_position
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _rope_tail_tables_mlx

        cos, sin = _rope_tail_tables_mlx(head_dim=6, qk_rope_head_dim=4, rope_theta=10000.0, seq_len=2)
        expected_cos, expected_sin = _rope_cos_sin_for_position(position=1, rope_dim=4, theta=10000.0)
        full_head_cos, full_head_sin = _rope_cos_sin_for_position(position=1, rope_dim=6, theta=10000.0)
        self.assertEqual(len(cos[1].tolist()), 2)
        for got, expected in zip(cos[1].tolist(), expected_cos[::2]):
            self.assertAlmostEqual(got, expected, places=6)
        for got, expected in zip(sin[1].tolist(), expected_sin[::2]):
            self.assertAlmostEqual(got, expected, places=6)
        self.assertGreater(abs(cos[1].tolist()[1] - full_head_cos[2]), 1e-4)
        self.assertGreater(abs(sin[1].tolist()[1] - full_head_sin[2]), 1e-4)

    def test_mlx_tail_rope_helper_broadcasts_over_4d_multihead_shape(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.deepseek_v4_attention_spec import apply_rope_tail
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _apply_rope_tail_mlx, _rope_tail_tables_mlx

        x = mx.array([[
            [[100.0, 200.0, 1.0, 2.0, 3.0, 4.0], [300.0, 400.0, 5.0, 6.0, 7.0, 8.0]],
            [[500.0, 600.0, 9.0, 10.0, 11.0, 12.0], [700.0, 800.0, 13.0, 14.0, 15.0, 16.0]],
        ]])
        cos, sin = _rope_tail_tables_mlx(head_dim=6, qk_rope_head_dim=4, rope_theta=10000.0, seq_len=2)
        got = _apply_rope_tail_mlx(x, cos, sin, qk_rope_head_dim=4).tolist()
        for pos in range(2):
            cos_full = [value for value in cos[pos].tolist() for _ in (0, 1)]
            sin_full = [value for value in sin[pos].tolist() for _ in (0, 1)]
            for head in range(2):
                original = x.tolist()[0][pos][head]
                expected = apply_rope_tail(original, nope_dim=2, cos=cos_full, sin=sin_full)
                self.assertEqual(got[0][pos][head][:2], original[:2])
                for gv, ev in zip(got[0][pos][head], expected):
                    self.assertAlmostEqual(gv, ev, places=5)

    def test_mlx_tail_inverse_rope_restores_tail(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.deepseek_v4_attention_spec import apply_output_inverse_rope_tail, apply_rope_tail
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _apply_rope_tail_mlx, _rope_tail_tables_mlx

        x = mx.array([[7.0, 11.0, 1.0, 2.0, 3.0, 4.0], [13.0, 17.0, 5.0, 6.0, 7.0, 8.0]])
        cos, sin = _rope_tail_tables_mlx(head_dim=6, qk_rope_head_dim=4, rope_theta=10000.0, seq_len=2)
        rotated = _apply_rope_tail_mlx(x, cos, sin, qk_rope_head_dim=4)
        restored = _apply_rope_tail_mlx(rotated, cos, -sin, qk_rope_head_dim=4).tolist()

        for pos, row in enumerate(x.tolist()):
            cos_full = [value for value in cos[pos].tolist() for _ in (0, 1)]
            sin_full = [value for value in sin[pos].tolist() for _ in (0, 1)]
            py_rotated = apply_rope_tail(row, nope_dim=2, cos=cos_full, sin=sin_full)
            py_restored = apply_output_inverse_rope_tail(py_rotated, nope_dim=2, cos=cos_full, sin=sin_full)
            for gv, ev in zip(rotated.tolist()[pos], py_rotated):
                self.assertAlmostEqual(gv, ev, delta=1e-5)
            for gv, ev in zip(restored[pos], py_restored):
                self.assertAlmostEqual(gv, ev, delta=1e-5)
            for gv, ev in zip(restored[pos], row):
                self.assertAlmostEqual(gv, ev, delta=1e-5)

    def test_mlx_tail_rope_invalid_qk_rope_head_dim_boundaries_fail_closed(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _apply_rope_tail_mlx, _rope_tail_tables_mlx

        x = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        for rope_dim in (0, 3, 8):
            with self.subTest(rope_dim=rope_dim):
                with self.assertRaisesRegex(ValueError, "qk_rope_head_dim"):
                    _rope_tail_tables_mlx(head_dim=6, qk_rope_head_dim=rope_dim, rope_theta=10000.0, seq_len=1)
                with self.assertRaisesRegex(ValueError, "qk_rope_head_dim"):
                    _apply_rope_tail_mlx(x, mx.array([[1.0, 1.0]]), mx.array([[0.0, 0.0]]), qk_rope_head_dim=rope_dim)

    def _tiny_multihead_attention_args(
        self,
        *,
        o_groups=1,
        sliding_window=2,
        compression_ratio=0,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_size=4,
        head_dim=4,
        qk_rope_head_dim=2,
        o_lora_rank=8,
        index_head_dim=1,
        index_n_heads=1,
        n_routed_experts=2,
        num_experts_per_tok=1,
        n_shared_experts=1,
        moe_intermediate_size=2,
        scoring_func="sqrtsoftplus",
        hc_mult=1,
    ):
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import ModelArgs

        return ModelArgs.from_dict({
            "model_type": "deepseek_v4",
            "vocab_size": 8,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "q_lora_rank": 4,
            "o_lora_rank": o_lora_rank,
            "qk_rope_head_dim": qk_rope_head_dim,
            "index_head_dim": index_head_dim,
            "index_n_heads": index_n_heads,
            "n_routed_experts": n_routed_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "n_shared_experts": n_shared_experts,
            "moe_intermediate_size": moe_intermediate_size,
            "expert_dtype": "fp4",
            "rms_norm_eps": 1e-6,
            "hc_mult": hc_mult,
            "layer_types": ["sliding_attention"] * num_hidden_layers,
            "mlp_layer_types": ["moe"] * num_hidden_layers,
            "scoring_func": scoring_func,
            "routed_scaling_factor": 1.0,
            "swiglu_limit": 10.0,
            "rope_theta": 10000.0,
            "sliding_window": sliding_window,
            "o_groups": o_groups,
            "compression_ratio": compression_ratio,
        })

    def _tiny_multihead_attention_weights(self):
        return {
            "q_a_proj.weight": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "q_norm.weight": [1.0, 1.2, 0.8, 1.5],
            "q_b_proj.weight": [
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.25, 0.0],
                [0.0, 0.0, 0.0, 0.5],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.1],
            ],
            "kv_proj.weight": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "kv_norm.weight": [1.1, 0.9, 1.0, 1.3],
            "o_a_proj.weight": [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            "o_b_proj.weight": [
                [1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0],
                [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5],
            ],
            "sinks": [-1e9, -1e9],
        }

    def _tiny_multihead_hidden_states(self):
        return [
            [1.0, 0.5, -0.25, 0.75],
            [-0.3, 0.8, 0.2, -0.5],
            [0.6, -0.1, 0.4, 0.9],
        ]

    def _tiny_csa_args(
        self,
        *,
        compression_ratio=4,
        num_attention_heads=1,
        o_groups=1,
        num_key_value_heads=1,
        num_hidden_layers=1,
        n_routed_experts=2,
        num_experts_per_tok=1,
        hc_mult=1,
    ):
        return self._tiny_multihead_attention_args(
            hidden_size=4,
            head_dim=4,
            qk_rope_head_dim=4,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            o_groups=o_groups,
            o_lora_rank=4,
            index_head_dim=2,
            index_n_heads=2,
            compression_ratio=compression_ratio,
            num_hidden_layers=num_hidden_layers,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            hc_mult=hc_mult,
        )

    def _tiny_csa_hidden_states(self):
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5, 0.0],
            [0.25, 0.25, 0.25, 0.25],
        ]

    def _tiny_csa_weights(self):
        def dense(rows, cols, scale):
            return [
                [round(scale * (i + j + 1) * (1.0 if (i + j) % 2 == 0 else -1.0), 4) for j in range(cols)]
                for i in range(rows)
            ]

        return {
            "compressor_wkv": dense(4, 8, 0.5),
            "compressor_wgate": dense(4, 8, 0.25),
            "compressor_ape": dense(8, 4, 0.05),
            "compressor_norm": [1.0, 1.0, 1.0, 1.0],
            "indexer_wq_b": dense(4, 4, 0.5),
            "indexer_proj": dense(4, 2, 0.3),
            "indexer_compressor_wkv": dense(4, 4, 0.4),
            "indexer_compressor_wgate": dense(4, 4, 0.2),
            "indexer_compressor_ape": dense(4, 4, 0.05),
            "indexer_compressor_norm": [1.0, 1.0],
        }

    def test_csa_compressor_mlx_matches_python_reference(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_csa_compressor_forward, _rope_cos_sin
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _csa_compressor_mlx

        args = self._tiny_csa_args()
        hidden = self._tiny_csa_hidden_states()
        weights = self._tiny_csa_weights()
        expected = tiny_csa_compressor_forward(
            hidden,
            {
                "kv_proj": weights["compressor_wkv"],
                "gate_proj": weights["compressor_wgate"],
                "position_bias": [[weights["compressor_ape"][d][t] for d in range(2 * args.head_dim)] for t in range(args.compression_ratio)],
                "kv_norm": weights["compressor_norm"],
            },
            compress_rate=args.compression_ratio,
            rms_norm_eps=args.rms_norm_eps,
            rope_cos=_rope_cos_sin([0, 4], args.head_dim, args.compress_rope_theta)[0],
            rope_sin=_rope_cos_sin([0, 4], args.head_dim, args.compress_rope_theta)[1],
        )
        got = _csa_compressor_mlx(args, mx.array([hidden]), {key: mx.array(value) for key, value in weights.items()}).tolist()[0]
        max_abs_error = max(abs(float(gv) - float(ev)) for got_row, exp_row in zip(got, expected) for gv, ev in zip(got_row, exp_row))
        self.assertLessEqual(max_abs_error, 1e-5)

    def test_csa_indexer_mlx_matches_python_reference(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_csa_indexer_forward
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _csa_indexer_mlx

        args = self._tiny_csa_args()
        hidden = self._tiny_csa_hidden_states()
        weights = self._tiny_csa_weights()
        expected = tiny_csa_indexer_forward(
            hidden,
            hidden,
            {
                "kv_proj": weights["indexer_compressor_wkv"],
                "gate_proj": weights["indexer_compressor_wgate"],
                "position_bias": [[weights["indexer_compressor_ape"][d][t] for d in range(2 * args.index_head_dim)] for t in range(args.compression_ratio)],
                "kv_norm": weights["indexer_compressor_norm"],
                "q_b_proj": weights["indexer_wq_b"],
                "weights_proj": weights["indexer_proj"],
            },
            compress_rate=args.compression_ratio,
            index_n_heads=args.index_n_heads,
            index_head_dim=args.index_head_dim,
            index_topk=2,
            rms_norm_eps=args.rms_norm_eps,
            rope_theta=args.compress_rope_theta,
            position_ids=list(range(len(hidden))),
        )
        got = _csa_indexer_mlx(args, mx.array([hidden]), mx.array([hidden]), {key: mx.array(value) for key, value in weights.items()}, index_topk=2)
        got_scores = got["scores"].tolist()[0]
        max_abs_error = max(abs(float(gv) - float(ev)) for got_row, exp_row in zip(got_scores, expected["scores"]) for gv, ev in zip(got_row, exp_row))
        self.assertLessEqual(max_abs_error, 1e-5)
        self.assertEqual(got["topk_indices"], [expected["topk_indices"]])
        self.assertEqual(got["topk_mask"], [expected["topk_mask"]])
        expected_valid_mask = []
        for row in expected["topk_indices"]:
            selected = {entry for entry in row if entry >= 0}
            expected_valid_mask.append([entry in selected for entry in range(expected["compressed_len"])])
        self.assertEqual(got["valid_mask"].tolist(), [expected_valid_mask])

    def test_csa_attention_mlx_matches_python_reference_and_routes_attention(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec, tiny_compressor_indexer_attention_reference
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _attention_mlx, _csa_attention_mlx

        args = self._tiny_csa_args()
        hidden = self._tiny_csa_hidden_states()
        weights = self._tiny_csa_weights()
        expected = tiny_compressor_indexer_attention_reference(
            DeepSeekV4AttentionSpec(args.hidden_size, args.num_attention_heads, args.head_dim, args.q_lora_rank, args.o_lora_rank, args.qk_rope_head_dim, args.o_groups, args.compression_ratio, args.index_n_heads, args.index_head_dim),
            hidden,
            hidden,
            weights,
            rms_norm_eps=args.rms_norm_eps,
            rope_theta=args.compress_rope_theta,
            index_topk=2,
        )["attended"]
        mlx_weights = {key: mx.array(value) for key, value in weights.items()}
        got = _csa_attention_mlx(args, mx.array([hidden]), mlx_weights, index_topk=2).tolist()[0]
        routed = _attention_mlx(args, mx.array([hidden]), mlx_weights, index_topk=2).tolist()[0]
        max_abs_error = max(abs(float(gv) - float(ev)) for got_row, exp_row in zip(got, expected) for gv, ev in zip(got_row, exp_row))
        self.assertLessEqual(max_abs_error, 1e-5)
        self.assertEqual(routed, got)
        self.assertTrue(all(float(v) == 0.0 for v in got[0]))

    def test_csa_attention_mlx_short_sequence_matches_zero_reference(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec, tiny_compressor_indexer_attention_reference
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _csa_attention_mlx, _csa_indexer_mlx

        args = self._tiny_csa_args()
        hidden = self._tiny_csa_hidden_states()[: args.compression_ratio - 1]
        weights = self._tiny_csa_weights()
        expected = tiny_compressor_indexer_attention_reference(
            DeepSeekV4AttentionSpec(args.hidden_size, args.num_attention_heads, args.head_dim, args.q_lora_rank, args.o_lora_rank, args.qk_rope_head_dim, args.o_groups, args.compression_ratio, args.index_n_heads, args.index_head_dim),
            hidden,
            hidden,
            weights,
            rms_norm_eps=args.rms_norm_eps,
            rope_theta=args.compress_rope_theta,
            index_topk=2,
        )
        mlx_weights = {key: mx.array(value) for key, value in weights.items()}
        got = _csa_attention_mlx(args, mx.array([hidden]), mlx_weights, index_topk=2).tolist()[0]
        indexer = _csa_indexer_mlx(args, mx.array([hidden]), mx.array([hidden]), mlx_weights, index_topk=2)
        self.assertEqual(got, expected["attended"])
        self.assertEqual(indexer["topk_indices"], [expected["topk_indices"]])
        self.assertEqual(indexer["topk_mask"], [expected["topk_mask"]])
        self.assertEqual(indexer["scores"].tolist(), [[[], [], []]])

    def test_csa_attention_mlx_fails_closed_for_unproven_ratios_and_shapes(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, _attention_mlx

        weights = {key: mx.array(value) for key, value in self._tiny_csa_weights().items()}
        hidden = mx.array([self._tiny_csa_hidden_states()])
        with self.assertRaisesRegex(NotImplementedError, "compression_ratio=4"):
            _attention_mlx(self._tiny_csa_args(compression_ratio=2), hidden, weights, index_topk=1)
        with self.assertRaisesRegex(NotImplementedError, "requires explicit per-head sink logits"):
            _attention_mlx(self._tiny_csa_args(num_attention_heads=2), hidden, weights, index_topk=1)
        with self.assertRaisesRegex(NotImplementedError, "divisible by o_groups"):
            _attention_mlx(self._tiny_csa_args(o_groups=2), hidden, weights, index_topk=1)
        with self.assertRaisesRegex(NotImplementedError, "num_key_value_heads=1"):
            _attention_mlx(self._tiny_csa_args(num_key_value_heads=2), hidden, weights, index_topk=1)
        with self.assertRaisesRegex(NotImplementedError, "requires explicit per-head sink logits"):
            _attention_mlx(self._tiny_csa_args(hc_mult=4), hidden, weights, index_topk=1)

        Model(self._tiny_csa_args())
        with self.assertRaisesRegex(NotImplementedError, "compression_ratio=4"):
            Model(self._tiny_csa_args(compression_ratio=2))
        with self.assertRaisesRegex(NotImplementedError, "single-head"):
            Model(self._tiny_csa_args(num_attention_heads=2))
        with self.assertRaisesRegex(NotImplementedError, "single-head"):
            Model(self._tiny_csa_args(o_groups=2))
        with self.assertRaisesRegex(NotImplementedError, "hc_mult=1"):
            Model(self._tiny_csa_args(hc_mult=4))
        # Story 11.15g R2: gate #1 (num_hidden_layers<=3) and gate #11
        # (n_routed_experts<=4) relaxed; the CSA subset guards above
        # (compression_ratio=4 single-head, hc_mult=1) are NOT relaxed and
        # still fail closed. The two relaxed configs now construct; paired
        # parity = 11.15e structural + 11.14 stateless CSA composed per-layer
        # (gate #1/#5(b)) and ADR 0017 dequant + 11.15d routing + Q3 probe
        # (gate #11), witnessed in
        # tests/test_deepseek_v4_validate_real_mode_relaxation.py.
        csa_multilayer = Model(self._tiny_csa_args(num_hidden_layers=43))
        self.assertEqual(csa_multilayer.args.num_hidden_layers, 43)
        csa_many_experts = Model(self._tiny_csa_args(n_routed_experts=256, num_experts_per_tok=6))
        self.assertEqual(csa_many_experts.args.n_routed_experts, 256)

    def test_attention_mlx_multihead_disabled_sinks_o_groups1_matches_python_reference(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec, tiny_multihead_grouped_attention_reference
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _attention_mlx

        args = self._tiny_multihead_attention_args()
        weights = self._tiny_multihead_attention_weights()
        hidden_states = self._tiny_multihead_hidden_states()
        spec = DeepSeekV4AttentionSpec(
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            head_dim=args.head_dim,
            q_lora_rank=args.q_lora_rank,
            o_lora_rank=args.o_lora_rank,
            qk_rope_head_dim=args.qk_rope_head_dim,
            num_output_groups=args.o_groups,
            compression_ratio=args.compression_ratio,
        )
        expected = tiny_multihead_grouped_attention_reference(
            spec,
            hidden_states,
            weights,
            rms_norm_eps=args.rms_norm_eps,
            rope_theta=args.rope_theta,
            sliding_window=args.sliding_window,
        )
        got = _attention_mlx(args, mx.array([hidden_states]), {key: mx.array(value) for key, value in weights.items()}).tolist()[0]
        max_abs_error = 0.0
        for got_row, expected_row in zip(got, expected):
            for gv, ev in zip(got_row, expected_row):
                max_abs_error = max(max_abs_error, abs(float(gv) - float(ev)))
        self.assertLessEqual(max_abs_error, 1e-5)

    def test_attention_mlx_multihead_regresses_flattened_q_rmsnorm_mutant(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec, tiny_multihead_grouped_attention_reference
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _attention_mlx

        args = self._tiny_multihead_attention_args(sliding_window=0)
        weights = self._tiny_multihead_attention_weights()
        hidden_states = self._tiny_multihead_hidden_states()
        spec = DeepSeekV4AttentionSpec(args.hidden_size, args.num_attention_heads, args.head_dim, args.q_lora_rank, args.o_lora_rank, args.qk_rope_head_dim, args.o_groups)
        expected = tiny_multihead_grouped_attention_reference(spec, hidden_states, weights, rms_norm_eps=args.rms_norm_eps, rope_theta=args.rope_theta, sliding_window=args.sliding_window)
        got = _attention_mlx(args, mx.array([hidden_states]), {key: mx.array(value) for key, value in weights.items()}).tolist()[0]
        self.assertAlmostEqual(got[1][2], expected[1][2], delta=1e-5)
        # Regression guard: this fixture was chosen so flattening q RMSNorm
        # across all heads instead of normalizing each head changes this value.
        self.assertGreater(abs(expected[1][2] - 0.8513560561007422), 1e-4)

    def _tiny_grouped_multihead_attention_weights(self):
        weights = self._tiny_multihead_attention_weights()
        weights["o_a_proj.weight"] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.5],
            [0.25, 0.5, 0.0, 0.0],
            [0.0, -0.5, 1.0, 0.0],
            [0.0, 0.0, 0.75, -0.25],
            [1.5, 0.0, 0.0, 0.25],
            [0.0, 0.0, 1.25, 0.0],
            [0.0, 0.0, 0.0, -1.5],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.25, 0.0, 1.0],
            [-0.75, 0.0, 0.0, 0.5],
            [0.0, 1.25, 0.0, 0.0],
            [0.0, 0.0, -0.5, 0.5],
            [0.5, 0.5, 0.0, 0.0],
        ]
        weights["o_b_proj.weight"] = [
            [1.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, -0.2, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -0.7, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, -0.4, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.25, 0.0, 0.0, 0.0, -0.5, 0.0, 0.9, 0.0, 0.0, -0.1, 0.0, 0.3, 0.0],
        ]
        weights["sinks"] = [0.5, -0.75]
        return weights

    def test_attention_mlx_multihead_o_groups2_matches_python_reference(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec, tiny_multihead_grouped_attention_reference
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _attention_mlx

        args = self._tiny_multihead_attention_args(o_groups=2, sliding_window=2)
        weights = self._tiny_grouped_multihead_attention_weights()
        hidden_states = self._tiny_multihead_hidden_states()
        spec = DeepSeekV4AttentionSpec(args.hidden_size, args.num_attention_heads, args.head_dim, args.q_lora_rank, args.o_lora_rank, args.qk_rope_head_dim, args.o_groups)
        expected = tiny_multihead_grouped_attention_reference(spec, hidden_states, weights, rms_norm_eps=args.rms_norm_eps, rope_theta=args.rope_theta, sliding_window=args.sliding_window)
        got = _attention_mlx(args, mx.array([hidden_states]), {key: mx.array(value) for key, value in weights.items()}).tolist()[0]
        max_abs_error = 0.0
        for got_row, expected_row in zip(got, expected):
            for gv, ev in zip(got_row, expected_row):
                max_abs_error = max(max_abs_error, abs(float(gv) - float(ev)))
        self.assertLessEqual(max_abs_error, 1e-5)

    def test_attention_mlx_multihead_o_groups2_rejects_flat_o_a_shape(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _attention_mlx

        args = self._tiny_multihead_attention_args(o_groups=2)
        weights = self._tiny_grouped_multihead_attention_weights()
        # A flattened-output mutant would expect width num_heads*head_dim=8.
        # GroupedLinear requires per-group width heads_per_group*head_dim=4.
        weights["o_a_proj.weight"] = [[0.0] * (args.num_attention_heads * args.head_dim) for _ in range(args.o_groups * args.o_lora_rank)]
        with self.assertRaisesRegex(ValueError, "o_a_proj.weight shape mismatch"):
            _attention_mlx(args, mx.array([self._tiny_multihead_hidden_states()]), {key: mx.array(value) for key, value in weights.items()})

    def test_attention_mlx_multihead_o_groups_fail_closed_for_invalid_shapes_and_divisibility(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _attention_mlx

        hidden_states = mx.array([self._tiny_multihead_hidden_states()])
        weights = self._tiny_grouped_multihead_attention_weights()
        with self.assertRaisesRegex(NotImplementedError, "divisible by o_groups"):
            _attention_mlx(self._tiny_multihead_attention_args(o_groups=3), hidden_states, {key: mx.array(value) for key, value in weights.items()})
        bad_o_a = dict(weights)
        bad_o_a["o_a_proj.weight"] = bad_o_a["o_a_proj.weight"][:-1]
        with self.assertRaisesRegex(ValueError, "o_a_proj.weight shape mismatch"):
            _attention_mlx(self._tiny_multihead_attention_args(o_groups=2), hidden_states, {key: mx.array(value) for key, value in bad_o_a.items()})
        bad_o_b = dict(weights)
        bad_o_b["o_b_proj.weight"] = [row[:-1] for row in bad_o_b["o_b_proj.weight"]]
        with self.assertRaisesRegex(ValueError, "o_b_proj.weight shape mismatch"):
            _attention_mlx(self._tiny_multihead_attention_args(o_groups=2), hidden_states, {key: mx.array(value) for key, value in bad_o_b.items()})

    def test_attention_mlx_multihead_enabled_distinct_sinks_match_python_reference(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec, tiny_multihead_grouped_attention_reference
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _attention_mlx

        args = self._tiny_multihead_attention_args(sliding_window=2)
        weights = self._tiny_multihead_attention_weights()
        weights["sinks"] = [0.75, -0.25]
        hidden_states = self._tiny_multihead_hidden_states()
        spec = DeepSeekV4AttentionSpec(args.hidden_size, args.num_attention_heads, args.head_dim, args.q_lora_rank, args.o_lora_rank, args.qk_rope_head_dim, args.o_groups)
        expected = tiny_multihead_grouped_attention_reference(spec, hidden_states, weights, rms_norm_eps=args.rms_norm_eps, rope_theta=args.rope_theta, sliding_window=args.sliding_window)
        got = _attention_mlx(args, mx.array([hidden_states]), {key: mx.array(value) for key, value in weights.items()}).tolist()[0]
        max_abs_error = 0.0
        for got_row, expected_row in zip(got, expected):
            for gv, ev in zip(got_row, expected_row):
                max_abs_error = max(max_abs_error, abs(float(gv) - float(ev)))
        self.assertLessEqual(max_abs_error, 1e-5)

    def test_attention_mlx_sink_extra_bucket_regresses_additive_bias_or_no_sink(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec, tiny_multihead_grouped_attention_reference
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _attention_mlx

        args = self._tiny_multihead_attention_args(sliding_window=1)
        weights = self._tiny_multihead_attention_weights()
        hidden_states = [self._tiny_multihead_hidden_states()[0]]
        enabled = dict(weights)
        enabled["sinks"] = [0.0, 1.25]
        disabled = dict(weights)
        disabled["sinks"] = [-1e9, -1e9]
        spec = DeepSeekV4AttentionSpec(args.hidden_size, args.num_attention_heads, args.head_dim, args.q_lora_rank, args.o_lora_rank, args.qk_rope_head_dim, args.o_groups)
        expected = tiny_multihead_grouped_attention_reference(spec, hidden_states, enabled, rms_norm_eps=args.rms_norm_eps, rope_theta=args.rope_theta, sliding_window=args.sliding_window)
        no_sink = tiny_multihead_grouped_attention_reference(spec, hidden_states, disabled, rms_norm_eps=args.rms_norm_eps, rope_theta=args.rope_theta, sliding_window=args.sliding_window)
        got = _attention_mlx(args, mx.array([hidden_states]), {key: mx.array(value) for key, value in enabled.items()}).tolist()[0]
        self.assertAlmostEqual(got[0][0], expected[0][0], delta=1e-5)
        self.assertLess(abs(got[0][0]), abs(no_sink[0][0]))
        self.assertGreater(abs(no_sink[0][0] - expected[0][0]), 0.1)

    def test_attention_mlx_multihead_o_groups2_rejects_legacy_flat_weights(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _attention_mlx

        hidden_states = self._tiny_multihead_hidden_states()
        # These slice3 weights used a flat o_a shape [o_lora_rank, num_heads*head_dim].
        # With o_groups=2, GroupedLinear requires [o_groups*o_lora_rank, heads_per_group*head_dim].
        weights = self._tiny_multihead_attention_weights()
        mlx_weights = {key: mx.array(value) for key, value in weights.items()}
        with self.assertRaisesRegex(ValueError, "o_a_proj.weight shape mismatch"):
            _attention_mlx(self._tiny_multihead_attention_args(o_groups=2), mx.array([hidden_states]), mlx_weights)

    def test_attention_mlx_multihead_fail_closed_for_wrong_sink_shape(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _attention_mlx

        hidden_states = mx.array([self._tiny_multihead_hidden_states()])
        mlx_weights = {key: mx.array(value) for key, value in self._tiny_multihead_attention_weights().items()}
        for sinks in ([], [-1e9], [-1e9, -1e9, -1e9]):
            with self.subTest(sinks=sinks):
                wrong = dict(mlx_weights)
                wrong["sinks"] = mx.array(sinks)
                with self.assertRaisesRegex(NotImplementedError, "one sink logit per query head"):
                    _attention_mlx(self._tiny_multihead_attention_args(), hidden_states, wrong)

    def test_attention_mlx_multihead_fail_closed_for_compressors_and_missing_sinks(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _attention_mlx

        hidden_states = mx.array([self._tiny_multihead_hidden_states()])
        mlx_weights = {key: mx.array(value) for key, value in self._tiny_multihead_attention_weights().items()}
        with self.assertRaisesRegex(ValueError, "missing MLX CSA compressor weights"):
            _attention_mlx(self._tiny_multihead_attention_args(compression_ratio=4), hidden_states, mlx_weights)
        # ratio=2 remains unsupported: the divergent synthetic ratio=2
        # block-sparse path was removed; real compressed dispatch now supports
        # only CSA ratio=4 and HCA ratio=128.
        with self.assertRaisesRegex(NotImplementedError, "compressors/indexers"):
            _attention_mlx(self._tiny_multihead_attention_args(compression_ratio=2), hidden_states, mlx_weights)
        missing_sink_weights = dict(mlx_weights)
        del missing_sink_weights["sinks"]
        with self.assertRaisesRegex(NotImplementedError, "requires explicit per-head sink logits"):
            _attention_mlx(self._tiny_multihead_attention_args(), hidden_states, missing_sink_weights)

    def _tiny_topk_moe_weights(self, *, n_routed_experts=3):
        weights = {
            "mlp.gate.weight": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ][:n_routed_experts],
            "mlp.gate.e_score_correction_bias": [0.0, 0.2, -0.1, 0.05][:n_routed_experts],
            "mlp.shared_experts.w1.weight": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            "mlp.shared_experts.w2.weight": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            "mlp.shared_experts.w3.weight": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        }
        for eid in range(n_routed_experts):
            scale = float(eid + 1)
            weights[f"mlp.experts.{eid}.w1.weight"] = [[0.2 * scale, 0.0, 0.0, 0.0], [0.0, 0.1 * scale, 0.0, 0.0]]
            weights[f"mlp.experts.{eid}.w2.weight"] = [[scale, 0.0], [0.0, scale + 1.0], [0.5 * scale, 0.25], [0.1, -0.2 * scale]]
            weights[f"mlp.experts.{eid}.w3.weight"] = [[0.0, 0.0, 0.3 * scale, 0.0], [0.0, 0.0, 0.0, 0.4 * scale]]
        return weights

    def _moe_reference_weights(self, weights, *, n_routed_experts):
        ref = {
            "router.weight": weights["mlp.gate.weight"],
            "router.e_score_correction_bias": weights["mlp.gate.e_score_correction_bias"],
            "shared.w1": weights["mlp.shared_experts.w1.weight"],
            "shared.w2": weights["mlp.shared_experts.w2.weight"],
            "shared.w3": weights["mlp.shared_experts.w3.weight"],
        }
        for eid in range(n_routed_experts):
            for proj in ("w1", "w2", "w3"):
                ref[f"experts.{eid}.{proj}"] = weights[f"mlp.experts.{eid}.{proj}.weight"]
        return ref

    def test_moe_mlx_top2_unquantized_matches_python_reference_and_not_top1(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig, tiny_topk_moe_forward
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _moe_mlx

        args = self._tiny_multihead_attention_args(n_routed_experts=3, num_experts_per_tok=2)
        hidden_states = [[1.0, 0.5, 0.25, -0.75], [0.2, 1.1, 0.9, 0.4]]
        weights = self._tiny_topk_moe_weights(n_routed_experts=3)
        ref_weights = self._moe_reference_weights(weights, n_routed_experts=3)
        cfg = MoEConfig(hidden_size=4, moe_intermediate_size=2, n_routed_experts=3, num_experts_per_tok=2)
        expected = tiny_topk_moe_forward(cfg, hidden_states, ref_weights, scoring_func=args.scoring_func, routed_scaling_factor=args.routed_scaling_factor, swiglu_limit=args.swiglu_limit)
        top1 = tiny_topk_moe_forward(MoEConfig(hidden_size=4, moe_intermediate_size=2, n_routed_experts=3, num_experts_per_tok=1), hidden_states, ref_weights, scoring_func=args.scoring_func, routed_scaling_factor=args.routed_scaling_factor, swiglu_limit=args.swiglu_limit)
        got = _moe_mlx(args, mx.array([hidden_states]), {key: mx.array(value) for key, value in weights.items()}).tolist()[0]

        max_abs_error = 0.0
        max_top1_delta = 0.0
        for got_row, exp_row, top1_row in zip(got, expected, top1):
            for gv, ev, tv in zip(got_row, exp_row, top1_row):
                max_abs_error = max(max_abs_error, abs(float(gv) - float(ev)))
                max_top1_delta = max(max_top1_delta, abs(float(ev) - float(tv)))
        self.assertLessEqual(max_abs_error, 1e-5)
        self.assertGreater(max_top1_delta, 1e-4)

    def test_moe_mlx_topk_rejects_k_greater_than_experts_without_duplicate_routing(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _moe_mlx

        args = self._tiny_multihead_attention_args(n_routed_experts=3, num_experts_per_tok=2)
        args.num_experts_per_tok = 4
        with self.assertRaisesRegex(NotImplementedError, "cannot exceed n_routed_experts"):
            _moe_mlx(args, mx.array([[[1.0, 0.0, 0.0, 0.0]]]), {key: mx.array(value) for key, value in self._tiny_topk_moe_weights(n_routed_experts=3).items()})

    def test_moe_mlx_rejects_unsupported_scoring_func_direct_helper(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _moe_mlx

        args = self._tiny_multihead_attention_args(n_routed_experts=3, num_experts_per_tok=2, scoring_func="softmax")
        with self.assertRaisesRegex(NotImplementedError, "scoring_func='sqrtsoftplus'"):
            _moe_mlx(args, mx.array([[[1.0, 0.0, 0.0, 0.0]]]), {})

    def test_moe_mlx_top2_i8_preserves_block_scale_dequant_path(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.deepseek_v4_moe_spec import MoEConfig, tiny_topk_moe_forward
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import ModelArgs, _moe_mlx

        args = ModelArgs.from_dict({
            "model_type": "deepseek_v4",
            "vocab_size": 8,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "q_lora_rank": 4,
            "o_lora_rank": 4,
            "qk_rope_head_dim": 2,
            "n_routed_experts": 3,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "moe_intermediate_size": 16,
            "expert_dtype": "i8",
            "layer_types": ["sliding_attention"],
            "mlp_layer_types": ["moe"],
        })
        hidden_states = [[0.1 * (i + 1) for i in range(16)], [0.05 * (16 - i) for i in range(16)]]
        weights = {
            "mlp.gate.weight": [[0.01 * (eid + 1) for _ in range(16)] for eid in range(3)],
            "mlp.gate.e_score_correction_bias": [0.0, 0.1, -0.05],
            "mlp.shared_experts.w1.weight": [[0.0] * 16 for _ in range(16)],
            "mlp.shared_experts.w2.weight": [[0.0] * 16 for _ in range(16)],
            "mlp.shared_experts.w3.weight": [[0.0] * 16 for _ in range(16)],
        }
        ref = {
            "router.weight": weights["mlp.gate.weight"],
            "router.e_score_correction_bias": weights["mlp.gate.e_score_correction_bias"],
            "shared.w1": weights["mlp.shared_experts.w1.weight"],
            "shared.w2": weights["mlp.shared_experts.w2.weight"],
            "shared.w3": weights["mlp.shared_experts.w3.weight"],
        }
        for eid in range(3):
            for proj in ("w1", "w2", "w3"):
                i8 = [[(eid + 1) if row == col else 0 for col in range(16)] for row in range(16)]
                scale = [[0.01 * (1.0 + eid)]] * 16
                weights[f"mlp.experts.{eid}.{proj}.weight"] = i8
                weights[f"mlp.experts.{eid}.{proj}.scale"] = scale
                ref[f"experts.{eid}.{proj}"] = [[float(value) * scale[row][0] for value in row_values] for row, row_values in enumerate(i8)]
        expected = tiny_topk_moe_forward(MoEConfig(16, 16, 3, 2), hidden_states, ref, scoring_func=args.scoring_func, routed_scaling_factor=args.routed_scaling_factor, swiglu_limit=args.swiglu_limit)
        got = _moe_mlx(args, mx.array([hidden_states]), {key: mx.array(value) for key, value in weights.items()}).tolist()[0]
        max_abs_error = max(abs(float(gv) - float(ev)) for got_row, exp_row in zip(got, expected) for gv, ev in zip(got_row, exp_row))
        self.assertLessEqual(max_abs_error, 1e-5)

    def test_model_real_mode_top2_three_experts_matches_integrated_reference(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, _integrated_layer_forward, _make_integrated_tiny_weights

        args = self._tiny_multihead_attention_args(n_routed_experts=3, num_experts_per_tok=2)
        weights = _make_integrated_tiny_weights(nonzero_hc=True)
        hidden_states = self._tiny_multihead_hidden_states()
        weights["embed.weight"] = hidden_states + [[0.0, 0.0, 0.0, 0.0] for _ in range(args.vocab_size - len(hidden_states))]
        weights.update(self._tiny_multihead_attention_weights())
        weights.update(self._tiny_topk_moe_weights(n_routed_experts=3))
        expected = _integrated_layer_forward(args, [[0, 1, 2]], weights)

        model = Model(args)
        model.load_weights(weights)
        got = model([[0, 1, 2]])

        max_abs_error = max(abs(float(gv) - float(ev)) for got_batch, exp_batch in zip(got, expected) for got_token, exp_token in zip(got_batch, exp_batch) for gv, ev in zip(got_token, exp_token))
        self.assertLessEqual(max_abs_error, 1e-5)

    def test_real_model_csa_compressed_attention_matches_integrated_python_reference(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, _integrated_layer_forward, _make_integrated_tiny_weights

        args = self._tiny_csa_args()
        weights = _make_integrated_tiny_weights(nonzero_hc=True)
        hidden_states = self._tiny_csa_hidden_states()
        weights["embed.weight"] = hidden_states
        weights.update(self._tiny_csa_weights())
        expected = _integrated_layer_forward(args, [[0, 1, 2, 3, 4, 5, 6, 7]], weights)
        load_weights = dict(weights)
        for unused_key in ("q_a_proj.weight", "q_norm.weight", "q_b_proj.weight", "kv_proj.weight", "kv_norm.weight", "o_a_proj.weight", "o_b_proj.weight", "sinks"):
            del load_weights[unused_key]

        with self.assertRaisesRegex(ValueError, "unexpected real DeepSeek V4 weights: .*q_a_proj.weight"):
            Model(args).load_weights(weights)
        missing = dict(load_weights)
        del missing["compressor_wkv"]
        with self.assertRaisesRegex(ValueError, "missing real DeepSeek V4 weights: .*compressor_wkv"):
            Model(args).load_weights(missing)

        model = Model(args)
        model.load_weights(load_weights)
        got = model([[0, 1, 2, 3, 4, 5, 6, 7]])

        max_abs_error = max(abs(float(gv) - float(ev)) for got_batch, exp_batch in zip(got, expected) for got_token, exp_token in zip(got_batch, exp_batch) for gv, ev in zip(got_token, exp_token))
        self.assertLessEqual(max_abs_error, 1e-5)

    def test_finetune_b0b_a_real_mode_proof_matches_integrated_reference(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from scripts import finetune_ds4

        spec = next(spec for spec in finetune_ds4._real_mode_forward_proof_specs() if spec["id"] == "B0b-a-1")
        proof = finetune_ds4._run_one_real_mode_forward_proof(spec)

        self.assertEqual(proof["status"], "ok", proof.get("reason"))
        self.assertEqual(proof["config"]["compression_ratio"], 4)
        self.assertEqual(proof["config"]["seq_len"], 8)
        self.assertLessEqual(proof["max_abs_error"], finetune_ds4.REAL_MODE_FORWARD_TOLERANCE)
        self.assertIn("_csa_compressor_mlx", "\n".join(proof["covered"]))
        self.assertIn("_csa_indexer_mlx", "\n".join(proof["covered"]))

    def test_finetune_b2_a1_i8_dequant_integration_proof(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from scripts import finetune_ds4

        spec = next(spec for spec in finetune_ds4._real_mode_forward_proof_specs() if spec["id"] == "B2-a-1")
        proof = finetune_ds4._run_one_real_mode_forward_proof(spec)

        self.assertEqual(proof["status"], "ok", proof.get("reason"))
        self.assertEqual(proof["evidence_class"], "B2-partial")
        self.assertEqual(proof["config"]["expert_dtype"], "i8")
        self.assertEqual(proof["config"]["hidden_size"], 16)
        self.assertEqual(proof["config"]["moe_intermediate_size"], 16)
        self.assertLessEqual(proof["max_abs_error"], finetune_ds4.REAL_MODE_FORWARD_TOLERANCE)
        self.assertLessEqual(proof["reference_max_abs_error"], proof["reference_tolerance"])
        self.assertEqual(proof["reference_tolerance"], 1e-3)
        self.assertGreater(proof["quantization_gap"], finetune_ds4.REAL_MODE_FORWARD_TOLERANCE)
        self.assertIs(proof["non_degenerate"], True)
        self.assertIs(proof["block_scale_nonunit"], True)
        self.assertIs(proof["i8_branch_reached"], True)
        self.assertIn("_dequantize_i8_block_scale_mlx", "\n".join(proof["covered"]))
        self.assertIn("packed FP4", "\n".join(proof["not_covered"]))

    def test_finetune_b2_a2_multilayer_i8_dequant_integration_proof(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from scripts import finetune_ds4

        spec = next(spec for spec in finetune_ds4._real_mode_forward_proof_specs() if spec["id"] == "B2-a-2")
        proof = finetune_ds4._run_one_real_mode_forward_proof(spec)

        self.assertEqual(proof["status"], "ok", proof.get("reason"))
        self.assertEqual(proof["evidence_class"], "B2-partial")
        self.assertEqual(proof["config"]["num_hidden_layers"], 2)
        self.assertEqual(proof["config"]["hidden_size"], 16)
        self.assertEqual(proof["config"]["moe_intermediate_size"], 16)
        self.assertLessEqual(proof["max_abs_error"], finetune_ds4.REAL_MODE_FORWARD_TOLERANCE)
        self.assertLessEqual(proof["reference_max_abs_error"], proof["reference_tolerance"])
        self.assertEqual(proof["reference_tolerance"], 1e-3)
        self.assertGreater(proof["quantization_gap"], finetune_ds4.REAL_MODE_FORWARD_TOLERANCE)
        self.assertIs(proof["i8_branch_reached"], True)
        self.assertEqual(proof["secondary_multilayer"]["num_hidden_layers"], 3)
        self.assertIs(proof["secondary_multilayer"]["ok"], True)
        self.assertIn("_dequantize_i8_block_scale_mlx", "\n".join(proof["covered"]))
        self.assertIn("packed FP4", "\n".join(proof["not_covered"]))

    def test_finetune_b2_a3_topk_multi_expert_i8_dequant_integration_proof(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from scripts import finetune_ds4

        spec = next(spec for spec in finetune_ds4._real_mode_forward_proof_specs() if spec["id"] == "B2-a-3")
        proof = finetune_ds4._run_one_real_mode_forward_proof(spec)

        self.assertEqual(proof["status"], "ok", proof.get("reason"))
        self.assertEqual(proof["evidence_class"], "B2-partial")
        self.assertEqual(proof["config"]["num_hidden_layers"], 1)
        self.assertEqual(proof["config"]["n_routed_experts"], 4)
        self.assertEqual(proof["config"]["num_experts_per_tok"], 2)
        self.assertEqual(proof["config"]["hidden_size"], 16)
        self.assertEqual(proof["config"]["moe_intermediate_size"], 16)
        self.assertLessEqual(proof["max_abs_error"], finetune_ds4.REAL_MODE_FORWARD_TOLERANCE)
        self.assertLessEqual(proof["reference_max_abs_error"], proof["reference_tolerance"])
        self.assertEqual(proof["reference_tolerance"], 1e-3)
        self.assertGreater(proof["quantization_gap"], finetune_ds4.REAL_MODE_FORWARD_TOLERANCE)
        self.assertIs(proof["i8_branch_reached"], True)
        self.assertIs(proof["multi_expert_combine_exercised"], True)
        self.assertEqual(proof["num_experts_per_tok"], 2)
        self.assertEqual(proof["n_routed_experts"], 4)
        self.assertGreaterEqual(proof["tie_free_margin"], finetune_ds4.TIE_FREE_MARGIN_EPS)
        self.assertIs(proof["routing_subset_stable"], True)
        self.assertEqual(proof["secondary_topk"]["num_experts_per_tok"], 3)
        self.assertIs(proof["secondary_topk"]["ok"], True)
        self.assertIn("argsort", "\n".join(proof["covered"]))
        self.assertIn("packed FP4", "\n".join(proof["not_covered"]))

    def test_finetune_readiness_report_records_b0b_a_without_unblocking_b0(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from scripts import finetune_ds4

        with tempfile.TemporaryDirectory() as td:
            mlx_work = Path(td) / "mlx"
            mlx_work.mkdir()
            real_mode_proofs = finetune_ds4._run_real_mode_forward_proofs(mlx_work)
            partials = [
                {"fixture": name, "status": "ok", "max_abs_error": 0.0, "covered": [], "not_covered": []}
                for name in finetune_ds4.FORWARD_PARITY_FIXTURE_NAMES
            ]
            report = finetune_ds4.build_forward_parity_readiness_report(
                mlx_work=mlx_work,
                partials=partials,
                blockers=finetune_ds4.forward_parity_blockers(),
                marker_present=False,
                model_constructs=True,
                non_forward_gate_status={},
                generated_at="2026-06-19T00:00:00+00:00",
                real_mode_proofs=real_mode_proofs,
            )

            self.assertFalse((mlx_work / ".deepseek-v4-forward-parity-ok").exists())
        self.assertEqual(report["real_mode_proofs"]["proofs_total"], 9)
        self.assertEqual(report["real_mode_proofs"]["proofs_ok"], 8)
        b0b = next(proof for proof in report["real_mode_proofs"]["proofs"] if proof["id"] == "B0b-a-1")
        self.assertEqual(b0b["status"], "ok")
        self.assertLessEqual(b0b["max_abs_error"], finetune_ds4.REAL_MODE_FORWARD_TOLERANCE)
        self.assertEqual(report["coverage"]["fixtures_total"], 19)
        self.assertFalse(report["full_forward_parity"])
        self.assertFalse(report["marker_earned"])
        self.assertEqual(report["blockers_count"], 4)
        b2a = next(proof for proof in report["real_mode_proofs"]["proofs"] if proof["id"] == "B2-a-1")
        self.assertEqual(b2a["status"], "ok")
        self.assertLessEqual(b2a["max_abs_error"], finetune_ds4.REAL_MODE_FORWARD_TOLERANCE)
        self.assertLessEqual(b2a["reference_max_abs_error"], 1e-3)
        b2a2 = next(proof for proof in report["real_mode_proofs"]["proofs"] if proof["id"] == "B2-a-2")
        self.assertEqual(b2a2["status"], "ok")
        self.assertLessEqual(b2a2["max_abs_error"], finetune_ds4.REAL_MODE_FORWARD_TOLERANCE)
        self.assertLessEqual(b2a2["reference_max_abs_error"], 1e-3)
        self.assertIs(b2a2["secondary_multilayer"]["ok"], True)
        b2a3 = next(proof for proof in report["real_mode_proofs"]["proofs"] if proof["id"] == "B2-a-3")
        self.assertEqual(b2a3["status"], "ok")
        self.assertLessEqual(b2a3["max_abs_error"], finetune_ds4.REAL_MODE_FORWARD_TOLERANCE)
        self.assertLessEqual(b2a3["reference_max_abs_error"], 1e-3)
        self.assertIs(b2a3["multi_expert_combine_exercised"], True)
        self.assertGreaterEqual(b2a3["tie_free_margin"], finetune_ds4.TIE_FREE_MARGIN_EPS)
        self.assertIs(b2a3["routing_subset_stable"], True)
        self.assertIs(b2a3["secondary_topk"]["ok"], True)
        self.assertEqual(report["coverage"]["fixtures_total"], 19)
        self.assertFalse(report["full_forward_parity"])
        self.assertFalse(report["marker_earned"])
        self.assertEqual(report["blockers_count"], 4)
        self.assertIn("B0 NOT", report["real_mode_proofs"]["b0_partial_progress"])
        self.assertIn("B2 NOT", report["real_mode_proofs"]["b0_partial_progress"])

    def test_finetune_csa_real_mode_unsupported_configs_fail_closed(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from scripts import finetune_ds4
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model

        with self.assertRaisesRegex(NotImplementedError, "hc_mult=1"):
            Model(finetune_ds4._build_real_mode_tiny_args(1, 2, compression_ratio=4, index_n_heads=2, index_head_dim=2))
        with self.assertRaisesRegex(NotImplementedError, "compression_ratio=4"):
            Model(finetune_ds4._build_real_mode_tiny_args(1, 1, compression_ratio=2, index_n_heads=2, index_head_dim=2))
        multi_head_args = finetune_ds4._build_real_mode_tiny_args(1, 1, compression_ratio=4, index_n_heads=2, index_head_dim=2)
        multi_head_args.num_attention_heads = 2
        with self.assertRaisesRegex(NotImplementedError, "single-head"):
            Model(multi_head_args)

        model = Model(finetune_ds4._build_real_mode_tiny_args(1, 1, compression_ratio=4, index_n_heads=2, index_head_dim=2))
        missing = finetune_ds4._make_real_mode_csa_weights()
        del missing["compressor_wkv"]
        with self.assertRaisesRegex(ValueError, "missing real DeepSeek V4 weights: .*compressor_wkv"):
            model.load_weights(missing)

    def test_model_args_accept_real_flash_config_keys(self):
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import ModelArgs

        cfg = {
            "model_type": "deepseek_v4",
            "vocab_size": 129280,
            "hidden_size": 4096,
            "num_hidden_layers": 43,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "q_lora_rank": 1024,
            "o_lora_rank": 1024,
            "qk_rope_head_dim": 64,
            "index_head_dim": 128,
            "index_n_heads": 64,
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
            "n_shared_experts": 1,
            "moe_intermediate_size": 2048,
            "expert_dtype": "fp4",
            "rope_theta": 10000,
            "compress_rope_theta": 160000,
            "rope_scaling": {"type": "yarn", "factor": 16},
            "num_hash_layers": 3,
            "attention_bias": False,
        }
        args = ModelArgs.from_dict(cfg)
        self.assertEqual(args.model_type, "deepseek_v4")
        self.assertEqual(args.hidden_size, 4096)
        self.assertEqual(args.n_routed_experts, 256)
        self.assertEqual(args.expert_dtype, "fp4")
        self.assertEqual(args.rope_scaling["type"], "yarn")
        self.assertFalse(hasattr(args, "index_topk"))
        self.assertNotIn("index_topk", ModelArgs.__dataclass_fields__)

    def test_real_model_multihead_o_groups2_enabled_sinks_matches_integrated_python_reference(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, _integrated_layer_forward, _make_integrated_tiny_weights

        args = self._tiny_multihead_attention_args(o_groups=2, sliding_window=2)
        weights = _make_integrated_tiny_weights(nonzero_hc=True)
        hidden_states = self._tiny_multihead_hidden_states()
        weights["embed.weight"] = hidden_states + [[0.0, 0.0, 0.0, 0.0] for _ in range(args.vocab_size - len(hidden_states))]
        weights.update(self._tiny_grouped_multihead_attention_weights())
        expected = _integrated_layer_forward(args, [[0, 1, 2]], weights)

        model = Model(args)
        model.load_weights(_to_transformers_weight_names(weights))
        got = model([[0, 1, 2]])

        self.assertEqual(len(got), 1)
        self.assertEqual(len(got[0]), 3)
        self.assertEqual(len(got[0][0]), args.hidden_size)
        max_abs_error = 0.0
        for got_batch, exp_batch in zip(got, expected):
            for got_token, exp_token in zip(got_batch, exp_batch):
                for gv, ev in zip(got_token, exp_token):
                    max_abs_error = max(max_abs_error, abs(float(gv) - float(ev)))
        self.assertLessEqual(max_abs_error, 1e-5)

    def test_real_model_multihead_o_groups2_four_heads_matches_integrated_python_reference(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, _integrated_layer_forward, _make_integrated_tiny_weights

        args = self._tiny_multihead_attention_args(
            num_attention_heads=4,
            head_dim=2,
            qk_rope_head_dim=2,
            o_groups=2,
            o_lora_rank=4,
            sliding_window=2,
        )
        hidden_states = self._tiny_multihead_hidden_states()
        weights = _make_integrated_tiny_weights(nonzero_hc=True)
        weights["embed.weight"] = hidden_states + [[0.0, 0.0, 0.0, 0.0] for _ in range(args.vocab_size - len(hidden_states))]
        weights.update({
            "q_b_proj.weight": [
                [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 1.0],
                [0.25, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.25],
            ],
            "kv_proj.weight": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            "kv_norm.weight": [1.1, 0.9],
            "o_a_proj.weight": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.5, 0.25, 0.0, 0.0],
                [0.0, -0.5, 1.0, 0.0],
                [0.0, 0.0, 0.75, 0.5],
                [1.25, 0.0, 0.0, -0.25],
            ],
            "o_b_proj.weight": [
                [1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.4],
            ],
            "sinks": [0.5, -0.25, 0.75, -0.5],
        })
        expected = _integrated_layer_forward(args, [[0, 1, 2]], weights)

        model = Model(args)
        model.load_weights(_to_transformers_weight_names(weights))
        got = model([[0, 1, 2]])

        max_abs_error = 0.0
        for got_batch, exp_batch in zip(got, expected):
            for got_token, exp_token in zip(got_batch, exp_batch):
                for gv, ev in zip(got_token, exp_token):
                    max_abs_error = max(max_abs_error, abs(float(gv) - float(ev)))
        self.assertLessEqual(max_abs_error, 1e-5)

    def _build_multilayer_weights(self, num_layers, *, n_routed_experts=3, num_experts_per_tok=2, o_groups=2):
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import _make_integrated_tiny_weights

        hidden = self._tiny_multihead_hidden_states()
        pad = [[0.0, 0.0, 0.0, 0.0] for _ in range(8 - len(hidden))]
        weights = {"embed.weight": [list(r) for r in hidden] + pad}
        attention_weights = self._tiny_grouped_multihead_attention_weights() if o_groups > 1 else self._tiny_multihead_attention_weights()
        for i in range(num_layers):
            layer = _make_integrated_tiny_weights(nonzero_hc=True)
            layer.update(attention_weights)
            layer.update(self._tiny_topk_moe_weights(n_routed_experts=n_routed_experts))
            offset = 0.1 * (i + 1)
            scale = 1.0 + 0.15 * i
            layer["attn_hc.base"] = [v + offset for v in layer["attn_hc.base"]]
            layer["ffn_hc.base"] = [v - offset for v in layer["ffn_hc.base"]]
            for eid in range(n_routed_experts):
                for proj in ("w1", "w2", "w3"):
                    key = f"mlp.experts.{eid}.{proj}.weight"
                    layer[key] = [[v * scale for v in row] for row in layer[key]]
            prefix = f"layers.{i}."
            for key, value in layer.items():
                if key == "embed.weight":
                    continue
                weights[f"{prefix}{key}"] = value
        return weights

    def test_real_model_multilayer_2layer_matches_python_reference(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, _integrated_multilayer_forward

        args = self._tiny_multihead_attention_args(
            num_hidden_layers=2,
            n_routed_experts=3,
            num_experts_per_tok=2,
            o_groups=2,
            sliding_window=2,
        )
        weights = self._build_multilayer_weights(2)
        # Enable distinct sink logits per layer for a stronger test.
        weights["layers.0.sinks"] = [0.5, -0.25]
        weights["layers.1.sinks"] = [-0.1, 0.8]
        expected = _integrated_multilayer_forward(args, [[0, 1, 2]], weights)

        model = Model(args)
        model.load_weights(weights)
        got = model([[0, 1, 2]])

        self.assertEqual(len(got), 1)
        self.assertEqual(len(got[0]), 3)
        self.assertEqual(len(got[0][0]), args.hidden_size)
        max_abs_error = 0.0
        for got_batch, exp_batch in zip(got, expected):
            for got_token, exp_token in zip(got_batch, exp_batch):
                for gv, ev in zip(got_token, exp_token):
                    max_abs_error = max(max_abs_error, abs(float(gv) - float(ev)))
        self.assertLessEqual(max_abs_error, 1e-5)

    def test_real_model_multilayer_output_differs_from_single_layer(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model

        args2 = self._tiny_multihead_attention_args(
            num_hidden_layers=2,
            n_routed_experts=3,
            num_experts_per_tok=2,
            o_groups=2,
            sliding_window=2,
        )
        weights2 = self._build_multilayer_weights(2)
        model2 = Model(args2)
        model2.load_weights(weights2)
        got2 = model2([[0, 1, 2]])

        args1 = self._tiny_multihead_attention_args(
            num_hidden_layers=1,
            n_routed_experts=3,
            num_experts_per_tok=2,
            o_groups=2,
            sliding_window=2,
        )
        single_weights = {"embed.weight": weights2["embed.weight"]}
        for k, v in weights2.items():
            if k.startswith("layers.0."):
                single_weights[k[len("layers.0."):]] = v
        model1 = Model(args1)
        model1.load_weights(single_weights)
        got1 = model1([[0, 1, 2]])

        max_delta = max(
            abs(float(a) - float(b))
            for a_row, b_row in zip(got1, got2)
            for a_tok, b_tok in zip(a_row, b_row)
            for a, b in zip(a_tok, b_tok)
        )
        self.assertGreater(max_delta, 1e-4)

    def test_real_model_multilayer_fails_closed_for_missing_per_layer_weights(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model

        args = self._tiny_multihead_attention_args(
            num_hidden_layers=2,
            n_routed_experts=3,
            num_experts_per_tok=2,
            o_groups=2,
        )
        weights = self._build_multilayer_weights(2)
        # Remove all of layer 1's weights -> wrong per-layer weight count.
        incomplete = {k: v for k, v in weights.items() if not k.startswith("layers.1.")}
        model = Model(args)
        with self.assertRaisesRegex(ValueError, "missing real DeepSeek V4 weights"):
            model.load_weights(incomplete)

    def test_real_model_multilayer_accepts_three_layers(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, _integrated_multilayer_forward

        args = self._tiny_multihead_attention_args(
            num_hidden_layers=3,
            n_routed_experts=3,
            num_experts_per_tok=2,
            o_groups=2,
            sliding_window=2,
        )
        weights = self._build_multilayer_weights(3)
        expected = _integrated_multilayer_forward(args, [[0, 1, 2]], weights)
        model = Model(args)
        model.load_weights(weights)
        got = model([[0, 1, 2]])
        max_abs_error = 0.0
        for got_batch, exp_batch in zip(got, expected):
            for got_token, exp_token in zip(got_batch, exp_batch):
                for gv, ev in zip(got_token, exp_token):
                    max_abs_error = max(max_abs_error, abs(float(gv) - float(ev)))
        self.assertLessEqual(max_abs_error, 1e-5)

    def test_real_model_multilayer_applies_final_norm_and_lm_head(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, _integrated_multilayer_forward

        args = self._tiny_multihead_attention_args(
            num_hidden_layers=2,
            n_routed_experts=3,
            num_experts_per_tok=2,
            o_groups=2,
            sliding_window=2,
        )
        weights = self._build_multilayer_weights(2)
        weights["norm.weight"] = [1.1, 0.9, 1.25, 0.7]
        lm_head = [
            [1.0, -0.5, 0.25, 0.1],
            [0.0, 1.0, -0.3, 0.4],
            [-0.2, 0.6, 1.0, -0.1],
            [0.3, -0.1, 0.5, 1.0],
            [0.8, 0.2, -0.4, 0.6],
            [-0.6, 0.7, 0.15, -0.5],
            [0.45, -0.35, 0.9, 0.2],
            [-0.15, 0.55, -0.2, 0.85],
        ]
        weights["lm_head.weight"] = lm_head
        weights["layers.0.sinks"] = [0.5, -0.25]
        weights["layers.1.sinks"] = [-0.1, 0.8]

        expected = _integrated_multilayer_forward(args, [[0, 1, 2]], weights)
        model = Model(args)
        model.load_weights(weights)
        got = model([[0, 1, 2]])

        # Final lm_head projects hidden->vocab, so output width must be vocab_size.
        self.assertEqual(len(got), 1)
        self.assertEqual(len(got[0]), 3)
        self.assertEqual(len(got[0][0]), args.vocab_size)
        max_abs_error = 0.0
        for got_batch, exp_batch in zip(got, expected):
            for got_token, exp_token in zip(got_batch, exp_batch):
                for gv, ev in zip(got_token, exp_token):
                    max_abs_error = max(max_abs_error, abs(float(gv) - float(ev)))
        self.assertLessEqual(max_abs_error, 1e-5)

    def test_real_model_multilayer_allows_hc_mult_gt_one_hyperconnection(self):
        # Story 11.15g R2: gate #10 relaxed (multi-layer hc_mult>1).
        # Paired proven-component parity = 11.11 hyperconnection synthetic
        # (_hyperconnection_mlx handles hc_mult>1 + softmax/Sinkhorn); the
        # dedicated parity witness is
        # test_gate_10_multi_layer_hc_mult_2_hyperconnection_runs in
        # tests/test_deepseek_v4_validate_real_mode_relaxation.py.
        # Pre-relaxation this asserted NotImplementedError("hc_mult=1"); the
        # gate is now lifted, so the contract is construction-succeeds.
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model

        args = self._tiny_multihead_attention_args(
            num_hidden_layers=2,
            n_routed_experts=3,
            num_experts_per_tok=2,
            o_groups=2,
            sliding_window=2,
        )
        args.hc_mult = 2
        model = Model(args)
        self.assertEqual(model.args.num_hidden_layers, 2)
        self.assertEqual(model.args.hc_mult, 2)

    def test_model_args_fail_closed_for_unsupported_expert_dtype(self):
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import ModelArgs

        cfg = {
            "model_type": "deepseek_v4",
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "q_lora_rank": 8,
            "qk_rope_head_dim": 4,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 16,
            "expert_dtype": "mystery",
        }
        with self.assertRaisesRegex(ValueError, "unsupported DeepSeek V4 expert_dtype"):
            ModelArgs.from_dict(cfg)

    def test_real_model_closed_gates_still_fail_while_relaxed_gates_allow_construction(self):
        # Story 11.15g R2: gates #1 (num_hidden_layers<=3) and #11
        # (n_routed_experts<=4) were relaxed and moved to the
        # construction-succeeds witness below. The remaining closed guards
        # (CSA subset compression_ratio=4 single-head, MQA num_key_value_heads=1,
        # num_attention_heads divisible by o_groups, n_shared_experts=1,
        # scoring_func='sqrtsoftplus') are NOT relaxed and still fail closed.
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model

        closed_cases = [
            (self._tiny_multihead_attention_args(compression_ratio=4), "compressors"),
            (self._tiny_multihead_attention_args(num_key_value_heads=2), "num_key_value_heads=1"),
            (self._tiny_multihead_attention_args(num_attention_heads=3, o_groups=2), "divisible by o_groups"),
            (self._tiny_multihead_attention_args(n_shared_experts=2), "n_shared_experts=1"),
            (self._tiny_multihead_attention_args(scoring_func="softmax"), "scoring_func='sqrtsoftplus'"),
        ]
        for args, message in closed_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(NotImplementedError, message):
                    Model(args)

    def test_real_model_relaxed_gates_allow_construction_layers_and_experts(self):
        # Story 11.15g R2: gate #1 (num_hidden_layers in {1,2,3}) and gate #11
        # (n_routed_experts<=4) relaxed. Paired proven-component parity:
        #   gate #1  -> 11.15e structural shape-compat (header-only)
        #   gate #11 -> dequantize_i8_e8m0_block_scale (ADR 0017) + _moe_mlx
        #               top-k stub (11.15d) + Q3 probe L2_REL=0.000e+00 at 256
        # Dedicated parity witnesses live in
        # tests/test_deepseek_v4_validate_real_mode_relaxation.py
        # (test_gates_1_to_4_structural_43_layers_real_layer_vectors and
        # test_gate_11_moe_mlx_i8_composition_no_drift_at_scale).
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model

        four_layer = self._tiny_multihead_attention_args(num_hidden_layers=4)
        model = Model(four_layer)
        self.assertEqual(model.args.num_hidden_layers, 4)

        five_expert = self._tiny_multihead_attention_args(n_routed_experts=5)
        model = Model(five_expert)
        self.assertEqual(model.args.n_routed_experts, 5)

    def test_real_model_construction_allows_real_flash_layer_count_43_structural(self):
        # Story 11.15g R2: gate #1 relaxed (allow real num_hidden_layers=43).
        # Paired proven-component parity = 11.15e structural shape-compat
        # (header-only); forward parity is 11.15h scope. Pre-relaxation this
        # asserted NotImplementedError("num_hidden_layers in {1,2,3}"); the
        # gate is now lifted, so the contract is construction-succeeds. The
        # n_routed_experts=256 term also exercises relaxed gate #11.
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, ModelArgs

        flash_cfg = {
            "model_type": "deepseek_v4",
            "vocab_size": 129280,
            "hidden_size": 4096,
            "num_hidden_layers": 43,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "q_lora_rank": 1024,
            "o_lora_rank": 1024,
            "qk_rope_head_dim": 64,
            "index_head_dim": 128,
            "index_n_heads": 64,
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
            "moe_intermediate_size": 2048,
            "expert_dtype": "fp4",
            "layer_types": ["sliding_attention"] * 43,
            "mlp_layer_types": ["moe"] * 43,
        }
        args = ModelArgs.from_dict(flash_cfg)
        model = Model(args)
        self.assertEqual(model.args.num_hidden_layers, 43)
        self.assertEqual(len(model.args.layer_types), 43)
        self.assertEqual(len(model.args.mlp_layer_types), 43)
        self.assertEqual(model.args.n_routed_experts, 256)

    def test_deepseek_v4_sanitizer_strips_only_mtp_weights(self):
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import sanitize_weights

        weights = {
            "embed.weight": object(),
            "layers.0.attn.wq_a.weight": object(),
            "mtp.0.attn.wq_a.weight": object(),
            "mtp.0.ffn.experts.0.w1.weight": object(),
        }
        sanitized, removed = sanitize_weights(weights)
        self.assertEqual(sorted(sanitized), ["embed.weight", "layers.0.attn.wq_a.weight"])
        self.assertEqual(sorted(removed), ["mtp.0.attn.wq_a.weight", "mtp.0.ffn.experts.0.w1.weight"])

    def test_tensor_family_scanner_maps_expected_flash_names_and_fails_unknown(self):
        from ds4_ft_mlx.deepseek_v4_mapping import classify_tensor_name, scan_tensor_names

        self.assertEqual(classify_tensor_name("embed.weight"), "embedding")
        self.assertEqual(classify_tensor_name("layers.0.attn.wq_a.weight"), "attention.q_a")
        self.assertEqual(classify_tensor_name("layers.0.attn.wq_b.scale"), "attention.q_b_scale")
        self.assertEqual(classify_tensor_name("layers.0.attn.wkv.weight"), "attention.kv")
        self.assertEqual(classify_tensor_name("layers.0.attn.wo_a.weight"), "attention.output_a")
        self.assertEqual(classify_tensor_name("layers.0.ffn.experts.12.w1.weight"), "moe.expert.w1")
        self.assertEqual(classify_tensor_name("layers.0.ffn.experts.12.w2.scale"), "moe.expert.w2_scale")
        self.assertEqual(classify_tensor_name("layers.0.hc_linear.weight"), "hyperconnection")
        report = scan_tensor_names([
            "embed.weight",
            "layers.0.attn.wq_a.weight",
            "layers.0.ffn.experts.12.w1.weight",
            "totally.unknown.weight",
        ])
        self.assertFalse(report.ok)
        self.assertEqual(report.unmapped, ["totally.unknown.weight"])
        self.assertIn("attention.q_a", scan_tensor_names(["layers.0.attn.wq_a.weight"]).families)

    def test_tensor_family_scanner_blocks_review_required_mtp(self):
        from ds4_ft_mlx.deepseek_v4_mapping import MTP_POLICY_FAMILY, scan_tensor_names

        report = scan_tensor_names([
            "embed.weight",
            "mtp.0.hc_head_base",
            "mtp.0.layers.0.attn.wq_a.weight",
        ])
        self.assertFalse(report.ok)
        self.assertEqual(report.review_required, [
            "mtp.0.hc_head_base",
            "mtp.0.layers.0.attn.wq_a.weight",
        ])
        self.assertIn(MTP_POLICY_FAMILY, report.families)
        self.assertEqual(report.review_required_policy[MTP_POLICY_FAMILY]["action"], "fail-closed")
        self.assertFalse(report.review_required_policy[MTP_POLICY_FAMILY]["supported"])
        self.assertFalse(report.review_required_policy[MTP_POLICY_FAMILY]["ignored"])
        self.assertFalse(report.review_required_policy[MTP_POLICY_FAMILY]["stripped"])

    def test_mtp_policy_is_explicit_and_reportable(self):
        from ds4_ft_mlx.deepseek_v4_mapping import MTP_POLICY_FAMILY, mtp_policy, scan_tensor_names

        policy = mtp_policy()
        self.assertEqual(policy["family"], MTP_POLICY_FAMILY)
        self.assertEqual(policy["status"], "review-required")
        self.assertEqual(policy["action"], "fail-closed")
        self.assertFalse(policy["supported"])
        self.assertFalse(policy["ignored"])
        self.assertFalse(policy["stripped"])
        self.assertIn("Do not make mapping pass", policy["instruction"])

        report = scan_tensor_names(["mtp.0.e_proj.weight"])
        as_dict = report.to_dict()
        self.assertFalse(as_dict["ok"])
        self.assertEqual(as_dict["review_required"], ["mtp.0.e_proj.weight"])
        self.assertEqual(as_dict["review_required_policy"][MTP_POLICY_FAMILY], policy)

    def test_mtp_policy_doc_matches_fail_closed_mapping_policy(self):
        doc = REPO_ROOT / "docs" / "deepseek-v4-mtp-policy.md"
        text = doc.read_text(encoding="utf-8")
        self.assertIn("Current policy: fail-closed", text)
        self.assertIn("MTP is not currently supported", text)
        self.assertIn("MTP is not currently ignored", text)
        self.assertIn("MTP is not currently stripped", text)
        self.assertIn("Do not make mapping pass", text)

    def test_real_model_load_weights_and_forward_matches_transformers_reference(self):
        try:
            import torch
            from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
            from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Model
        except Exception as exc:
            self.skipTest(f"torch/transformers reference unavailable: {exc}")

        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, ModelArgs, _make_integrated_tiny_weights, set_transformers_integrated_weights

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
        weights = _make_integrated_tiny_weights(nonzero_hc=True)
        # Present weights with standard Transformers-compatible names as plain
        # Python tensors. load_weights also accepts mlx.core arrays via tolist().
        weights_std = _to_transformers_weight_names(weights)
        model.load_weights(weights_std)
        got = model([[0, 1]])

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
        set_transformers_integrated_weights(ref_model, weights)
        ref_model.hc_head = torch.nn.Identity()  # type: ignore[attr-defined]
        ref_model.norm = torch.nn.Identity()  # type: ignore[attr-defined]
        with torch.no_grad():
            ref_out = ref_model(torch.tensor([[0, 1]], dtype=torch.long)).last_hidden_state
        ref = ref_out.squeeze(0).reshape(-1, args.hidden_size).tolist()

        self.assertEqual(len(got), 1)
        self.assertEqual(len(got[0]), 2)
        max_abs_error = 0.0
        for got_row, ref_row in zip(got[0], ref):
            self.assertEqual(len(got_row), len(ref_row))
            for gv, rv in zip(got_row, ref_row):
                max_abs_error = max(max_abs_error, abs(float(gv) - float(rv)))
        self.assertLessEqual(max_abs_error, 1e-5)

    def test_real_model_load_weights_accepts_mlx_arrays(self):
        try:
            import mlx.core as mx
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
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
        weights = _make_integrated_tiny_weights(nonzero_hc=True)
        del weights["mlp.gate.e_score_correction_bias"]

        weights_mx = _to_transformers_weight_names({k: mx.array(v) for k, v in weights.items()})
        model.load_weights(weights_mx)
        params = model.parameters()
        self.assertTrue(all(type(v) is mx.array for v in params.values()))
        self.assertEqual(params["mlp.gate.e_score_correction_bias"].tolist(), [0.0, 0.0])
        got = model([[0, 1]])
        self.assertEqual(len(got), 1)
        self.assertEqual(len(got[0]), 2)
        self.assertEqual(len(got[0][0]), args.hidden_size)

    def test_real_model_hc_mult4_matches_integrated_python_reference(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import (
            Model,
            ModelArgs,
            _integrated_layer_forward,
            _make_integrated_tiny_weights,
        )

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
            "rms_norm_eps": 1e-6,
            "hc_mult": 4,
            "hc_eps": 1e-6,
            "hc_sinkhorn_iters": 2,
            "layer_types": ["sliding_attention"],
            "mlp_layer_types": ["moe"],
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.0,
            "swiglu_limit": 10.0,
            "rope_theta": 10000.0,
            "sliding_window": 128,
            "o_groups": 1,
        })
        weights = _make_integrated_tiny_weights(hc_mult=4, nonzero_hc=True)
        weights = {key: value for key, value in weights.items() if not key.startswith("hc_head.")}
        expected = _integrated_layer_forward(args, [[0, 1]], weights)

        model = Model(args)
        model.load_weights(_to_transformers_weight_names(weights))
        got = model([[0, 1]])

        self.assertEqual(len(got), 1)
        self.assertEqual(len(got[0]), 2)
        self.assertEqual(len(got[0][0]), 4)
        self.assertEqual(len(got[0][0][0]), args.hidden_size)
        max_abs_error = 0.0
        for got_batch, exp_batch in zip(got, expected):
            for got_token, exp_token in zip(got_batch, exp_batch):
                for got_stream, exp_stream in zip(got_token, exp_token):
                    for gv, ev in zip(got_stream, exp_stream):
                        max_abs_error = max(max_abs_error, abs(float(gv) - float(ev)))
        self.assertLessEqual(max_abs_error, 1e-5)

    def test_real_model_hc_mult4_final_hyperhead_norm_lm_head_matches_python_reference(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_hyperhead_collapse
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import (
            Model,
            ModelArgs,
            _integrated_layer_forward,
            _make_integrated_tiny_weights,
        )

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
            "rms_norm_eps": 1e-6,
            "hc_mult": 4,
            "hc_eps": 1e-6,
            "hc_sinkhorn_iters": 2,
            "layer_types": ["sliding_attention"],
            "mlp_layer_types": ["moe"],
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.0,
            "swiglu_limit": 10.0,
            "rope_theta": 10000.0,
            "sliding_window": 128,
            "o_groups": 1,
        })
        weights = _make_integrated_tiny_weights(hc_mult=4, nonzero_hc=True)
        fn_dim = args.hc_mult * args.hidden_size
        weights["hc_head.fn"] = [[1.0 if (j - i) % fn_dim == 1 else 0.0 for j in range(fn_dim)] for i in range(args.hc_mult)]
        weights["hc_head.base"] = [0.05, -0.05, 0.1, -0.1]
        weights["hc_head.scale"] = [0.3]
        norm_weight = [1.0, 0.5, 1.5, 2.0]
        lm_head = [
            [0.2, -0.1, 0.0, 0.3],
            [0.0, 0.25, -0.2, 0.1],
            [0.4, 0.0, 0.15, -0.05],
        ]
        stream = _integrated_layer_forward(args, [[0, 1]], weights)

        def _rms_norm(vector):
            mean_sq = sum(float(x) * float(x) for x in vector) / len(vector)
            scale = (mean_sq + args.rms_norm_eps) ** -0.5
            return [float(x) * scale * float(w) for x, w in zip(vector, norm_weight)]

        expected = []
        for batch in stream:
            out_batch = []
            for token_streams in batch:
                collapsed = tiny_hyperhead_collapse(
                    hidden_streams=[token_streams],
                    fn=weights["hc_head.fn"],
                    base=weights["hc_head.base"],
                    scale=weights["hc_head.scale"][0],
                    hc_mult=args.hc_mult,
                    eps=args.hc_eps,
                    rms_norm_eps=args.rms_norm_eps,
                )["collapsed"][0]
                normalized = _rms_norm(collapsed)
                out_batch.append([sum(float(x) * float(w) for x, w in zip(normalized, row)) for row in lm_head])
            expected.append(out_batch)

        model_weights = dict(weights)
        model_weights["norm.weight"] = norm_weight
        model_weights["lm_head.weight"] = lm_head
        model = Model(args)
        model.load_weights(_to_transformers_weight_names(model_weights))
        got = model([[0, 1]])

        self.assertEqual(len(got), 1)
        self.assertEqual(len(got[0]), 2)
        self.assertEqual(len(got[0][0]), len(lm_head))
        max_abs_error = 0.0
        for got_batch, exp_batch in zip(got, expected):
            for got_token, exp_token in zip(got_batch, exp_batch):
                for gv, ev in zip(got_token, exp_token):
                    max_abs_error = max(max_abs_error, abs(float(gv) - float(ev)))
        self.assertLessEqual(max_abs_error, 1e-5)

    def test_real_model_hc_mult4_final_norm_lm_head_fail_closed(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
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
            "rms_norm_eps": 1e-6,
            "hc_mult": 4,
            "hc_eps": 1e-6,
            "hc_sinkhorn_iters": 2,
            "layer_types": ["sliding_attention"],
            "mlp_layer_types": ["moe"],
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.0,
            "swiglu_limit": 10.0,
            "rope_theta": 10000.0,
            "sliding_window": 128,
            "o_groups": 1,
        })
        weights = _make_integrated_tiny_weights(hc_mult=4, nonzero_hc=True)
        weights = {key: value for key, value in weights.items() if not key.startswith("hc_head.")}
        mapped_weights = _to_transformers_weight_names(weights)
        mapped_weights["model.norm.weight"] = [1.0] * args.hidden_size
        mapped_weights["lm_head.weight"] = [[1.0, 0.0, 0.0, 0.0] for _ in range(args.vocab_size)]

        model = Model(args)
        model.load_weights(mapped_weights)
        with self.assertRaisesRegex(NotImplementedError, "hc_mult>1 forward does not support final norm/lm_head"):
            model([[0, 1]])

    def test_real_model_hc_mult4_incomplete_hc_head_fails_at_load(self):
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:
            self.skipTest(f"mlx not available: {exc}")
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
            "rms_norm_eps": 1e-6,
            "hc_mult": 4,
            "hc_eps": 1e-6,
            "hc_sinkhorn_iters": 2,
            "layer_types": ["sliding_attention"],
            "mlp_layer_types": ["moe"],
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.0,
            "swiglu_limit": 10.0,
            "rope_theta": 10000.0,
            "sliding_window": 128,
            "o_groups": 1,
        })
        weights = _make_integrated_tiny_weights(hc_mult=4, nonzero_hc=True)
        del weights["hc_head.scale"]
        with self.assertRaisesRegex(ValueError, "incomplete real DeepSeek V4 hc_head weights: hc_head.scale"):
            Model(args).load_weights(_to_transformers_weight_names(weights))

    def test_real_model_i8_block_scale_moe_matches_transformers_reference(self):
        try:
            import mlx.core as mx
            import torch
            from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
            from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Model
        except Exception as exc:
            self.skipTest(f"mlx/torch/transformers reference unavailable: {exc}")
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, ModelArgs, set_transformers_integrated_weights

        hidden = 16
        intermediate = 16
        identity = [[1.0 if i == j else 0.0 for j in range(hidden)] for i in range(hidden)]
        base_weights = {
            "embed.weight": [[(row * hidden + col + 1) * 0.01 for col in range(hidden)] for row in range(4)],
            "input_layernorm.weight": [1.0] * hidden,
            "post_attention_layernorm.weight": [1.0] * hidden,
            "q_a_proj.weight": identity,
            "q_norm.weight": [1.0] * hidden,
            "q_b_proj.weight": identity,
            "kv_proj.weight": identity,
            "kv_norm.weight": [1.0] * hidden,
            "o_a_proj.weight": identity,
            "o_b_proj.weight": identity,
            "sinks": [-1e9],
            "attn_hc.fn": [[0.0] * hidden for _ in range(3)],
            "attn_hc.base": [0.0, 0.0, 0.0],
            "attn_hc.scale": [0.0, 0.0, 0.0],
            "ffn_hc.fn": [[0.0] * hidden for _ in range(3)],
            "ffn_hc.base": [0.0, 0.0, 0.0],
            "ffn_hc.scale": [0.0, 0.0, 0.0],
            "mlp.gate.weight": [[0.05] * hidden, [-0.05] * hidden],
            "mlp.gate.e_score_correction_bias": [0.0, 0.0],
            "mlp.shared_experts.w1.weight": [[0.01] * hidden for _ in range(intermediate)],
            "mlp.shared_experts.w2.weight": [[0.01] * intermediate for _ in range(hidden)],
            "mlp.shared_experts.w3.weight": [[0.01] * hidden for _ in range(intermediate)],
        }

        float_weights = dict(base_weights)
        i8_weights = dict(base_weights)

        def _quantize_matrix(rows: int, cols: int, offset: int):
            dense = torch.tensor(
                [[((r * cols + c + offset) % 17 - 8) * 0.01 for c in range(cols)] for r in range(rows)],
                dtype=torch.float32,
            )
            scale = dense.abs().amax(dim=1, keepdim=True) / 127.0
            scale = torch.where(scale == 0, torch.ones_like(scale) * 1e-6, scale)
            q = (dense / scale).round().clamp(-128, 127).to(torch.int8)
            dequant = q.float() * scale.to(torch.bfloat16).float().repeat_interleave(16, dim=1)
            return q, scale.to(torch.bfloat16), dequant

        for eid in range(2):
            for proj, shape, offset in (("w1", (intermediate, hidden), 1), ("w2", (hidden, intermediate), 3), ("w3", (intermediate, hidden), 5)):
                q, scale, dequant = _quantize_matrix(shape[0], shape[1], offset + eid)
                key = f"mlp.experts.{eid}.{proj}"
                i8_weights[f"{key}.weight"] = mx.array(q.tolist(), dtype=mx.int8)
                i8_weights[f"{key}.scale"] = mx.array(scale.float().tolist(), dtype=mx.bfloat16)
                float_weights[f"{key}.weight"] = dequant.tolist()

        args = ModelArgs.from_dict({
            "model_type": "deepseek_v4",
            "vocab_size": 4,
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": hidden,
            "q_lora_rank": hidden,
            "o_lora_rank": hidden,
            "qk_rope_head_dim": hidden,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": intermediate,
            "n_shared_experts": 1,
            "expert_dtype": "i8",
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
        model.load_weights(_to_transformers_weight_names(i8_weights))
        got = model([[0, 1]])

        ref_config = DeepseekV4Config(
            vocab_size=4,
            hidden_size=hidden,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=hidden,
            q_lora_rank=hidden,
            o_lora_rank=hidden,
            qk_rope_head_dim=hidden,
            num_experts_per_tok=1,
            n_routed_experts=2,
            moe_intermediate_size=intermediate,
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
        ref_model = DeepseekV4Model(ref_config)
        ref_model.eval()
        set_transformers_integrated_weights(ref_model, float_weights)
        ref_model.hc_head = torch.nn.Identity()  # type: ignore[attr-defined]
        ref_model.norm = torch.nn.Identity()  # type: ignore[attr-defined]
        with torch.no_grad():
            ref = ref_model(torch.tensor([[0, 1]], dtype=torch.long)).last_hidden_state.squeeze(0).reshape(-1, hidden).tolist()

        max_abs_error = 0.0
        for got_row, ref_row in zip(got[0], ref):
            self.assertEqual(len(got_row), len(ref_row))
            for gv, rv in zip(got_row, ref_row):
                max_abs_error = max(max_abs_error, abs(float(gv) - float(rv)))
        self.assertLessEqual(max_abs_error, 1e-5)

    def test_real_model_mqa_gate_fails_closed_while_relaxed_n_routed_experts_allows_construction(self):
        # Story 11.15g R2: gate #6 (MQA num_key_value_heads=1) STAYS CLOSED
        # (ADR 0007 axis b retired) — block 1 still raises. Gate #11
        # (n_routed_experts<=4) relaxed — block 2 now constructs.
        # Paired parity for gate #11 = ADR 0017 dequant + _moe_mlx top-k
        # stub (11.15d) + Q3 probe L2_REL=0.000e+00 at 256, witnessed in
        # tests/test_deepseek_v4_validate_real_mode_relaxation.py.
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import Model, ModelArgs

        args = ModelArgs.from_dict({
            "model_type": "deepseek_v4",
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "q_lora_rank": 8,
            "qk_rope_head_dim": 4,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 16,
            "expert_dtype": "fp4",
            "layer_types": ["sliding_attention"],
            "mlp_layer_types": ["moe"],
        })
        with self.assertRaisesRegex(NotImplementedError, "num_key_value_heads=1"):
            Model(args)

        unproven_expert_count = ModelArgs.from_dict({
            "model_type": "deepseek_v4",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "q_lora_rank": 16,
            "o_lora_rank": 16,
            "qk_rope_head_dim": 16,
            "n_routed_experts": 5,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 16,
            "expert_dtype": "i8",
            "hc_mult": 1,
            "layer_types": ["sliding_attention"],
            "mlp_layer_types": ["moe"],
            "o_groups": 1,
        })
        model = Model(unproven_expert_count)
        self.assertEqual(model.args.n_routed_experts, 5)


if __name__ == "__main__":
    unittest.main()
