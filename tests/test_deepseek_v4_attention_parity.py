#!/usr/bin/env python3
import json
import math
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_SRC = REPO_ROOT / "python-envs" / "mlx" / "src"


class DeepSeekV4AttentionParityTests(unittest.TestCase):
    def setUp(self):
        self._old_path = list(sys.path)
        sys.path.insert(0, str(MLX_SRC))

    def tearDown(self):
        sys.path[:] = self._old_path

    def _assert_model_4bit_absent_or_valid_current_artifact(self):
        model_4bit = Path("/Volumes/Data NVME/mlx-ft/ds4/model-4bit")
        if not model_4bit.exists():
            return
        cfg = json.loads((model_4bit / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg.get("model_type"), "deepseek_v4_nn")
        self.assertEqual(cfg.get("quantization", {}).get("bits"), 4)

    def tiny_spec(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec

        return DeepSeekV4AttentionSpec(
            hidden_size=32,
            num_attention_heads=4,
            head_dim=10,
            q_lora_rank=6,
            o_lora_rank=3,
            qk_rope_head_dim=2,
            num_output_groups=2,
            compression_ratio=4,
            index_n_heads=5,
            index_head_dim=3,
        )

    def test_ds4_projection_tensor_shape_semantics(self):
        spec = self.tiny_spec()

        self.assertEqual(spec.q_dim, 40)
        self.assertEqual(spec.out_low_dim, 6)
        self.assertEqual(spec.ds4_tensor_shapes(), {
            "wq_a": (32, 6),
            "wq_b": (6, 40),
            "wkv": (32, 10),
            "wo_a": (20, 6),
            "wo_b": (6, 32),
        })

    def test_flash_mlx_attention_weight_shapes_match_checkpoint_formulas(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec

        spec = DeepSeekV4AttentionSpec(
            hidden_size=4096,
            num_attention_heads=64,
            head_dim=512,
            q_lora_rank=1024,
            o_lora_rank=1024,
            qk_rope_head_dim=64,
            num_output_groups=8,
        )
        self.assertEqual(spec.flash_mlx_safetensors_shapes(), {
            "q_a_proj.weight": (1024, 4096),
            "q_norm.weight": (1024,),
            "q_b_proj.weight": (32768, 1024),
            "kv_proj.weight": (512, 4096),
            "kv_norm.weight": (512,),
            "o_a_proj.weight": (8192, 4096),
            "o_b_proj.weight": (4096, 8192),
            "sinks": (64,),
        })
        self.assertEqual(spec.ds4_tensor_shapes(), {
            "wq_a": (4096, 1024),
            "wq_b": (1024, 32768),
            "wkv": (4096, 512),
            "wo_a": (4096, 8192),
            "wo_b": (8192, 4096),
        })

    def test_tiny_activation_shape_semantics_are_deterministic(self):
        spec = self.tiny_spec()

        self.assertEqual(spec.activation_shapes(batch=2, seq_len=3), {
            "q_a_out": (2, 3, 6),
            "q_b_out_flat": (2, 3, 40),
            "q_b_out_heads": (2, 3, 4, 10),
            "wkv_out": (2, 3, 10),
            "wo_a_grouped_in": (2, 3, 2, 20),
            "wo_a_out_flat": (2, 3, 6),
            "wo_b_out": (2, 3, 32),
        })

    def test_rope_nope_split_is_tail_rope(self):
        spec = self.tiny_spec()
        split = spec.rope_nope_split()

        self.assertEqual(split.nope_dim, 8)
        self.assertEqual(split.rope_dim, 2)
        self.assertEqual(split.nope_slice, (0, 8))
        self.assertEqual(split.rope_slice, (8, 10))
        self.assertEqual(split.describe(), "nope[0:8] rope[8:10]")

    def test_norm_sink_and_hyperconnection_shape_requirements_are_explicit(self):
        spec = self.tiny_spec()

        self.assertEqual(spec.norm_and_sink_shapes(), {
            "q_norm": (6,),
            "kv_norm": (10,),
            "attn_sink": (4,),
            "attn_norm": (32,),
        })
        self.assertEqual(spec.hyperconnection_shapes(hc_mult=4), {
            "hc_attn_base": (24,),
            "hc_attn_fn": (24, 128),
            "hc_attn_scale": (3,),
            "hc_ffn_base": (24,),
            "hc_ffn_fn": (24, 128),
            "hc_ffn_scale": (3,),
        })
        with self.assertRaisesRegex(ValueError, "hc_mult must be positive"):
            spec.hyperconnection_shapes(hc_mult=0)

    def test_compressor_and_indexer_shape_placeholders_are_explicit(self):
        spec = self.tiny_spec()

        self.assertEqual(spec.compressor_shapes(), {
            "compressor_ape": (20, 4),
            "compressor_wkv": (32, 20),
            "compressor_wgate": (32, 20),
            "compressor_norm": (10,),
        })
        self.assertEqual(spec.indexer_shapes(), {
            "indexer_wq_b": (6, 15),
            "indexer_proj": (32, 5),
            "indexer_compressor_ape": (6, 4),
            "indexer_compressor_wkv": (32, 6),
            "indexer_compressor_wgate": (32, 6),
            "indexer_compressor_norm": (3,),
        })

    def test_apply_rope_tail_rotates_only_rope_slice(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import apply_rope_tail

        got = apply_rope_tail([10.0, 20.0, 1.0, 2.0], nope_dim=2, cos=[0.0, 0.0], sin=[1.0, 1.0])
        self.assertEqual(got, [10.0, 20.0, -2.0, 1.0])

    def test_attention_sink_is_extra_logit_and_dropped_after_softmax(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import attention_scores_with_sink

        scores = attention_scores_with_sink([0.0, 0.0], sink=math.log(2.0))
        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(scores[0], 0.25, places=12)
        self.assertAlmostEqual(scores[1], 0.25, places=12)
        self.assertAlmostEqual(sum(scores), 0.5, places=12)
        stable_scores = attention_scores_with_sink([1000.0, 1000.0], sink=1000.0 + math.log(2.0))
        self.assertAlmostEqual(stable_scores[0], 0.25, places=12)
        self.assertAlmostEqual(stable_scores[1], 0.25, places=12)

    def _identity_out_in(self, size):
        return [[1.0 if row == col else 0.0 for col in range(size)] for row in range(size)]

    def _tiny_kv_spec(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec

        return DeepSeekV4AttentionSpec(
            hidden_size=4,
            num_attention_heads=2,
            head_dim=4,
            q_lora_rank=4,
            o_lora_rank=2,
            qk_rope_head_dim=2,
            num_output_groups=1,
        )

    def _tiny_kv_weights(self):
        return {
            "q_a_proj.weight": self._dense_weights(4, 4, 0.07),
            "q_norm.weight": [1.0, 1.1, 0.9, 1.2],
            "q_b_proj.weight": self._dense_weights(8, 4, 0.05),
            "kv_proj.weight": self._dense_weights(4, 4, 0.09),
            "kv_norm.weight": [1.0, 0.8, 1.15, 0.95],
            "o_a_proj.weight": self._dense_weights(2, 8, 0.04),
            "o_b_proj.weight": self._dense_weights(4, 2, 0.06),
            "sinks": [0.25, -0.4],
        }

    @staticmethod
    def _tiny_kv_hidden_states():
        return [
            [0.2, -0.1, 0.3, 0.7],
            [-0.4, 0.5, 0.1, -0.2],
            [0.6, 0.0, -0.3, 0.2],
            [0.1, 0.4, 0.8, -0.5],
            [-0.2, -0.6, 0.4, 0.3],
        ]

    @staticmethod
    def _max_abs_rows(a, b):
        return max(abs(float(x) - float(y)) for row_a, row_b in zip(a, b, strict=True) for x, y in zip(row_a, row_b, strict=True))

    def test_tiny_multihead_sink_is_extra_softmax_bucket_not_additive_bias(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec, tiny_multihead_grouped_attention_reference

        spec = DeepSeekV4AttentionSpec(
            hidden_size=2,
            num_attention_heads=1,
            head_dim=2,
            q_lora_rank=2,
            o_lora_rank=2,
            qk_rope_head_dim=2,
            num_output_groups=1,
        )
        weights = {
            "q_a_proj.weight": self._identity_out_in(2),
            "q_norm.weight": [1.0, 1.0],
            "q_b_proj.weight": self._identity_out_in(2),
            "kv_proj.weight": self._identity_out_in(2),
            "kv_norm.weight": [1.0, 1.0],
            "o_a_proj.weight": self._identity_out_in(2),
            "o_b_proj.weight": self._identity_out_in(2),
            "sinks": [math.sqrt(2.0)],
        }
        got = tiny_multihead_grouped_attention_reference(spec, [[1.0, 0.0]], weights, rms_norm_eps=0.0)
        # With one real key and an equal sink extra bucket, key probability is
        # 1/2. An additive-bias implementation would still assign probability 1
        # to the sole key and return sqrt(2) instead.
        self.assertAlmostEqual(got[0][0], math.sqrt(2.0) * 0.5, places=12)
        self.assertAlmostEqual(got[0][1], 0.0, places=12)
        weights["sinks"] = [-1e9]
        disabled = tiny_multihead_grouped_attention_reference(spec, [[1.0, 0.0]], weights, rms_norm_eps=0.0)
        self.assertAlmostEqual(disabled[0][0], math.sqrt(2.0), places=12)

    def test_tiny_multihead_q_norm_is_per_head_not_flattened(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec, tiny_multihead_grouped_attention_reference

        spec = DeepSeekV4AttentionSpec(
            hidden_size=4,
            num_attention_heads=2,
            head_dim=2,
            q_lora_rank=4,
            o_lora_rank=4,
            qk_rope_head_dim=2,
            num_output_groups=1,
        )
        weights = {
            "q_a_proj.weight": self._identity_out_in(4),
            "q_norm.weight": [1.0, 1.0, 1.0, 1.0],
            # Head 0 has much larger magnitude than head 1. A flattened q RMSNorm
            # would let head 0's scale suppress head 1 and changes token-1 output.
            "q_b_proj.weight": [[10.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.1, 0.0]],
            "kv_proj.weight": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            "kv_norm.weight": [1.0, 1.0],
            "o_a_proj.weight": self._identity_out_in(4),
            "o_b_proj.weight": self._identity_out_in(4),
            "sinks": [-1e9, -1e9],
        }
        got = tiny_multihead_grouped_attention_reference(spec, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]], weights)
        self.assertAlmostEqual(got[1][1], 1.2348673181418985, places=12)
        # Regression guard: a mutant that RMSNorms flattened q instead of each
        # head independently produces ~1.2337811194359771 here.
        self.assertGreater(abs(got[1][1] - 1.2337811194359771), 1e-4)

    def test_tiny_grouped_output_uses_independent_blocks(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec, tiny_multihead_grouped_attention_reference

        spec = DeepSeekV4AttentionSpec(
            hidden_size=2,
            num_attention_heads=2,
            head_dim=2,
            q_lora_rank=2,
            o_lora_rank=1,
            qk_rope_head_dim=2,
            num_output_groups=2,
        )
        weights = {
            "q_a_proj.weight": self._identity_out_in(2),
            "q_norm.weight": [1.0, 1.0],
            "q_b_proj.weight": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            "kv_proj.weight": self._identity_out_in(2),
            "kv_norm.weight": [1.0, 1.0],
            "o_a_proj.weight": [[1.0, 0.0], [3.0, 0.0]],
            "o_b_proj.weight": self._identity_out_in(2),
            "sinks": [-1e9, -1e9],
        }
        got = tiny_multihead_grouped_attention_reference(spec, [[1.0, 0.0]], weights, rms_norm_eps=0.0)
        self.assertAlmostEqual(got[0][0], math.sqrt(2.0), places=12)
        self.assertAlmostEqual(got[0][1], 3.0 * math.sqrt(2.0), places=12)

    def test_tiny_multihead_tail_rope_preserves_no_rope_prefix_and_rejects_invalid_rope_dim(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec, apply_rope_tail, tiny_multihead_grouped_attention_reference

        self.assertEqual(apply_rope_tail([10.0, 20.0, 1.0, 2.0], nope_dim=2, cos=[0.0, 0.0], sin=[1.0, 1.0])[:2], [10.0, 20.0])

        def _weights_for(spec):
            return {
                "q_a_proj.weight": self._identity_out_in(2),
                "q_norm.weight": [1.0, 1.0],
                "q_b_proj.weight": self._identity_out_in(2),
                "kv_proj.weight": self._identity_out_in(2),
                "kv_norm.weight": [1.0, 1.0],
                "o_a_proj.weight": self._identity_out_in(2),
                "o_b_proj.weight": self._identity_out_in(2),
                "sinks": [-1e9],
            }

        for rope_dim in (0, 1):
            spec = DeepSeekV4AttentionSpec(2, 1, 2, 2, 2, rope_dim)
            with self.subTest(rope_dim=rope_dim):
                with self.assertRaisesRegex(ValueError, "qk_rope_head_dim"):
                    tiny_multihead_grouped_attention_reference(spec, [[1.0, 0.0]], _weights_for(spec))
        with self.assertRaisesRegex(ValueError, "qk_rope_head_dim cannot exceed head_dim"):
            DeepSeekV4AttentionSpec(2, 1, 2, 2, 2, 4)

    def test_sliding_window_cache_update_returns_attention_full_and_persists_window_minus_one(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import sliding_window_cache_update

        update = sliding_window_cache_update([[1.0], [2.0]], [[3.0]], sliding_window=3)
        self.assertEqual(update["attention_kv"], [[1.0], [2.0], [3.0]])
        self.assertEqual(update["persisted_kv"], [[2.0], [3.0]])
        cold = sliding_window_cache_update([], [[1.0], [2.0]], sliding_window=4)
        self.assertEqual(cold["attention_kv"], [[1.0], [2.0]])
        self.assertEqual(cold["persisted_kv"], [[1.0], [2.0]])

    def test_incremental_attention_matches_full_reference_unbounded(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_incremental_attention, tiny_multihead_grouped_attention_reference

        spec = self._tiny_kv_spec()
        weights = self._tiny_kv_weights()
        hidden = self._tiny_kv_hidden_states()[:4]
        full = tiny_multihead_grouped_attention_reference(spec, hidden, weights, sliding_window=0)
        incremental = tiny_incremental_attention(spec, hidden, weights, sliding_window=0)

        self.assertLessEqual(self._max_abs_rows(incremental, full), 1e-5)
        self.assertEqual(incremental, tiny_incremental_attention(spec, hidden, weights, sliding_window=0))

    def test_incremental_attention_matches_reference_with_sliding_window(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_incremental_attention, tiny_multihead_grouped_attention_reference

        spec = self._tiny_kv_spec()
        weights = self._tiny_kv_weights()
        hidden = self._tiny_kv_hidden_states()
        full_w2 = tiny_multihead_grouped_attention_reference(spec, hidden, weights, sliding_window=2)
        incremental_w2 = tiny_incremental_attention(spec, hidden, weights, sliding_window=2)
        wrong_w3 = tiny_incremental_attention(spec, hidden, weights, sliding_window=3)

        self.assertLessEqual(self._max_abs_rows(incremental_w2, full_w2), 1e-5)
        self.assertGreater(self._max_abs_rows(wrong_w3, full_w2), 1e-5)

    def test_incremental_cache_persists_window_minus_one(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import IncrementalSlidingKVCache

        spec = self._tiny_kv_spec()
        weights = self._tiny_kv_weights()
        cache = IncrementalSlidingKVCache(spec, weights, sliding_window=3)
        for pos, hidden in enumerate(self._tiny_kv_hidden_states()):
            cache.step(hidden)
            self.assertEqual(len(cache.kv_rows), min(pos + 1, 2))
            if cache.kv_rows:
                self.assertEqual(len(cache.kv_rows[0]), spec.head_dim)
            self.assertEqual(cache.next_pos, pos + 1)

        unbounded = IncrementalSlidingKVCache(spec, weights, sliding_window=0)
        for pos, hidden in enumerate(self._tiny_kv_hidden_states()[:3]):
            unbounded.step(hidden)
            self.assertEqual(len(unbounded.kv_rows), pos + 1)

    def test_tiny_greedy_decode_is_deterministic_and_drift_free(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_greedy_decode, tiny_multihead_grouped_attention_reference

        spec = self._tiny_kv_spec()
        weights = self._tiny_kv_weights()
        prompt = self._tiny_kv_hidden_states()[:2]
        embed_table = self._tiny_kv_hidden_states()
        vocab_proj = self._dense_weights(len(embed_table), spec.hidden_size, 0.03)

        first = tiny_greedy_decode(spec, prompt, weights, embed_table=embed_table, vocab_proj=vocab_proj, steps=4, sliding_window=3)
        second = tiny_greedy_decode(spec, prompt, weights, embed_table=embed_table, vocab_proj=vocab_proj, steps=4, sliding_window=3)
        full = tiny_multihead_grouped_attention_reference(spec, first["hidden_states"], weights, sliding_window=3)

        self.assertEqual(first, second)
        self.assertEqual(len(first["token_ids"]), 4)
        self.assertLessEqual(self._max_abs_rows(first["step_outputs"], full[-4:]), 1e-5)

    def test_incremental_attention_fails_closed_for_csa(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec, IncrementalSlidingKVCache, tiny_greedy_decode, tiny_incremental_attention

        spec = DeepSeekV4AttentionSpec(
            hidden_size=4,
            num_attention_heads=1,
            head_dim=4,
            q_lora_rank=4,
            o_lora_rank=4,
            qk_rope_head_dim=4,
            num_output_groups=1,
            compression_ratio=4,
            index_n_heads=2,
            index_head_dim=2,
        )
        msg = "Ca-carry|deferred|compression_ratio=0"
        with self.assertRaisesRegex(NotImplementedError, msg):
            IncrementalSlidingKVCache(spec, {}, sliding_window=2)
        with self.assertRaisesRegex(NotImplementedError, msg):
            tiny_incremental_attention(spec, [[1.0, 0.0, 0.0, 0.0]], {}, sliding_window=2)
        with self.assertRaisesRegex(NotImplementedError, msg):
            tiny_greedy_decode(spec, [[1.0, 0.0, 0.0, 0.0]], {}, embed_table=[[1.0, 0.0, 0.0, 0.0]], vocab_proj=[[1.0, 0.0, 0.0, 0.0]], steps=1, sliding_window=2)

    def test_generation_markers_absent_and_real_blockers_unchanged(self):
        from ds4_ft_mlx.deepseek_v4_dequant import dequantize_expert_packed
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import forward_parity_blockers

        self.assertFalse(Path("/Volumes/Data NVME/mlx-ft/ds4/.deepseek-v4-forward-parity-ok").exists())
        self._assert_model_4bit_absent_or_valid_current_artifact()
        blockers = forward_parity_blockers()
        self.assertIn("full MLX shimmed-checkpoint load/forward and MLX generation smoke (Track B only; DS4-GGUF base generation is its own Track-A gate, .ds4-gguf-generate-ok, not a forward-parity blocker)", blockers)
        self.assertIn("full MoE parity with packed FP4/I8 expert dequant and expert kernels", blockers)
        with self.assertRaises(NotImplementedError):
            dequantize_expert_packed("i8", b"\x00", scales=b"\x7f", shape=(1,))
        with self.assertRaisesRegex(ValueError, "requires non-None scales"):
            dequantize_expert_packed("fp4", b"\x00" * 16, scales=None, shape=(1, 32))

    def test_output_inverse_rope_tail_undoes_value_rope_slice(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import apply_output_inverse_rope_tail, apply_rope_tail

        value_with_rope = apply_rope_tail([7.0, 1.0, 2.0], nope_dim=1, cos=[0.0, 0.0], sin=[1.0, 1.0])
        self.assertEqual(value_with_rope, [7.0, -2.0, 1.0])
        self.assertEqual(
            apply_output_inverse_rope_tail(value_with_rope, nope_dim=1, cos=[0.0, 0.0], sin=[1.0, 1.0]),
            [7.0, 1.0, 2.0],
        )

    def test_hca_block_bias_metadata_matches_causal_threshold_rule_and_no_bias_cases(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import hca_block_bias

        bias = hca_block_bias(position_ids=[0, 1, 2, 3], compressed_len=2, compression_ratio=2)
        neg_inf = float("-inf")
        self.assertEqual(bias, [
            [neg_inf, neg_inf],
            [0.0, neg_inf],
            [0.0, neg_inf],
            [0.0, 0.0],
        ])
        self.assertIsNone(hca_block_bias(position_ids=[7], compressed_len=2, compression_ratio=2))
        self.assertIsNone(hca_block_bias(position_ids=[1, 2], compressed_len=0, compression_ratio=2))

    def test_tiny_sink_cache_inverse_rope_fixture_reports_partial_coverage(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import run_tiny_sink_cache_inverse_rope_fixture

        report = run_tiny_sink_cache_inverse_rope_fixture()
        self.assertEqual(report["fixture"], "sink-cache-inverse-rope-hca-bias")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["max_abs_error"], 0.0)
        self.assertIn("attention sink extra logit", report["covered"])
        self.assertIn("compressor/indexer forward", report["not_covered"])

    def test_csa_topk_indexer_gather_respects_block_bias_and_pads(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import csa_topk_indexer_gather, hca_block_bias

        block_bias = hca_block_bias(position_ids=[1, 3, 5], compressed_len=3, compression_ratio=2)
        report = csa_topk_indexer_gather(
            [[0.1, 0.9, 0.8], [0.2, 0.7, 0.3], [0.2, 0.7, 0.9]],
            block_bias=block_bias,
            top_k=2,
        )
        self.assertEqual(report["gather_shape"], (3, 2))
        self.assertEqual(report["gather_indices"], [[0, -1], [1, 0], [2, 1]])
        self.assertEqual(report["gather_mask"], [[True, False], [True, True], [True, True]])
        self.assertEqual(report["eligible_counts"], [1, 2, 3])

    def test_tiny_csa_topk_indexer_fixture_reports_partial_coverage(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import run_tiny_csa_topk_indexer_fixture

        report = run_tiny_csa_topk_indexer_fixture()
        self.assertEqual(report["fixture"], "csa-topk-indexer-gather-mask")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["max_abs_error"], 0.0)
        self.assertIn("CSA top-k indexer gather shape", report["covered"])
        self.assertIn("compressor/indexer forward", report["not_covered"])

    def test_tiny_sliding_attention_no_compressor_matches_manual_reference(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec, tiny_sliding_attention_no_compressor

        spec = DeepSeekV4AttentionSpec(
            hidden_size=2,
            num_attention_heads=1,
            head_dim=2,
            q_lora_rank=2,
            o_lora_rank=2,
            qk_rope_head_dim=0,
            num_output_groups=1,
        )
        weights = {
            "wq_a": [[1.0, 0.0], [0.0, 1.0]],
            "q_norm": [1.0, 1.0],
            "wq_b": [[1.0, 0.0], [0.0, 1.0]],
            "wkv": [[1.0, 0.0], [0.0, 1.0]],
            "kv_norm": [1.0, 1.0],
            "wo_a": [[1.0, 0.0], [0.0, 1.0]],
            "wo_b": [[1.0, 0.0], [0.0, 1.0]],
        }
        got = tiny_sliding_attention_no_compressor(spec, [[1.0, 0.0], [0.0, 1.0]], weights, rms_norm_eps=0.0)
        exp_weight_1 = math.exp(math.sqrt(2.0)) / (1.0 + math.exp(math.sqrt(2.0)))
        exp_weight_0 = 1.0 - exp_weight_1
        expected = [
            [math.sqrt(2.0), 0.0],
            [exp_weight_0 * math.sqrt(2.0), exp_weight_1 * math.sqrt(2.0)],
        ]
        for got_row, expected_row in zip(got, expected):
            for got_value, expected_value in zip(got_row, expected_row):
                self.assertAlmostEqual(got_value, expected_value, places=12)

    def test_tiny_hyperconnection_fixture_matches_manual_reference(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_hyperconnection_forward

        result = tiny_hyperconnection_forward(
            hidden_streams=[[[2.0]]],
            fn=[[0.0], [0.0], [0.0]],
            base=[0.0, 0.0, 0.0],
            scale=[0.0, 0.0, 0.0],
            hc_mult=1,
            eps=0.0,
            sinkhorn_iters=1,
            rms_norm_eps=0.0,
        )
        self.assertEqual(result["pre"], [[0.5]])
        self.assertEqual(result["collapsed"], [[1.0]])
        self.assertEqual(result["post"], [[1.0]])
        self.assertEqual(result["comb"], [[[1.0]]])

    def test_tiny_hyperconnection_hc2_fixture_reports_multistream_partial_coverage(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import run_tiny_hyperconnection_hc2_fixture, tiny_hyperconnection_forward

        fn = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        result = tiny_hyperconnection_forward(
            hidden_streams=[[[3.0, 4.0], [0.0, 5.0]]],
            fn=fn,
            base=[0.1, -0.2, 0.3, -0.4, 0.2, -0.1, 0.05, -0.05],
            scale=[0.5, -0.25, 1.25],
            hc_mult=2,
            eps=0.0,
            sinkhorn_iters=1,
            rms_norm_eps=0.0,
        )
        expected = {
            "pre": [[0.628144306391142, 0.6241279919178186]],
            "collapsed": [[1.8844329191734261, 5.633217185153661]],
            "post": [[1.0085784333258068, 0.802624679775096]],
            "comb": [[[0.3597055166182544, 0.7932581563947655], [0.6402944833817457, 0.20674184360523457]]],
        }
        for key, expected_rows in expected.items():
            for got_row, expected_row in zip(result[key], expected_rows, strict=True):
                if key == "comb":
                    for got_inner, expected_inner in zip(got_row, expected_row, strict=True):
                        for got, want in zip(got_inner, expected_inner, strict=True):
                            self.assertAlmostEqual(got, want, places=12)
                else:
                    for got, want in zip(got_row, expected_row, strict=True):
                        self.assertAlmostEqual(got, want, places=12)

        report = run_tiny_hyperconnection_hc2_fixture()
        self.assertEqual(report["fixture"], "hyperconnection-hc2-pre-post-comb")
        self.assertEqual(report["status"], "ok")
        self.assertLessEqual(report["max_abs_error"], 1e-12)
        self.assertIn("nonzero hypernetwork projection", report["covered"])
        self.assertNotIn("nonzero hypernetwork projection parity against Transformers", report["not_covered"])
        self.assertIn("full decoder-layer residual placement", report["not_covered"])

    def test_tiny_final_hyperhead_collapse_matches_reference_formula(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import run_tiny_hyperhead_fixture, tiny_hyperhead_collapse

        got = tiny_hyperhead_collapse(
            hidden_streams=[[[3.0, 4.0], [0.0, 5.0]]],
            fn=[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            base=[0.1, -0.2],
            scale=0.5,
            hc_mult=2,
            eps=0.0,
            rms_norm_eps=0.0,
        )
        self.assertAlmostEqual(got["pre"][0][0], 0.628144306391142, places=12)
        self.assertAlmostEqual(got["pre"][0][1], 0.6241279919178186, places=12)
        self.assertAlmostEqual(got["collapsed"][0][0], 1.8844329191734261, places=12)
        self.assertAlmostEqual(got["collapsed"][0][1], 5.633217185153661, places=12)

        report = run_tiny_hyperhead_fixture()
        self.assertEqual(report["fixture"], "final-hyperhead-hc2-collapse")
        self.assertEqual(report["status"], "ok")
        self.assertLessEqual(report["max_abs_error"], 1e-12)
        self.assertIn("final hyperhead weighted stream collapse", report["covered"])
        self.assertIn("shared final RMSNorm after hyperhead", report["not_covered"])

    def test_tiny_hyperconnection_hc2_matches_transformers_reference(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import run_tiny_hyperconnection_hc2_transformers_fixture

        report = run_tiny_hyperconnection_hc2_transformers_fixture()
        if report["status"] == "skipped":
            self.skipTest(report.get("reason", "torch/transformers unavailable"))
        self.assertEqual(report["fixture"], "hyperconnection-hc2-transformers")
        self.assertEqual(report["status"], "ok")
        self.assertLessEqual(report["max_abs_error"], 1e-5)
        self.assertIn("Transformers DeepseekV4HyperConnection module reference", report["covered"])
        self.assertIn("full decoder-layer residual placement", report["not_covered"])

    def test_tiny_hca_compressor_forward_matches_transformers_reference(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import run_tiny_hca_compressor_fixture

        report = run_tiny_hca_compressor_fixture()
        if report["status"] == "skipped":
            self.skipTest(report.get("reason", "torch/transformers unavailable"))
        self.assertEqual(report["fixture"], "hca-compressor-forward")
        self.assertEqual(report["status"], "ok")
        self.assertLessEqual(report["max_abs_error"], 1e-5)
        self.assertIn("HCA compressor kv/gate projection", report["covered"])
        self.assertIn("CSA compressor Ca/Cb overlap", report["not_covered"])

    def test_tiny_hca_compressor_returns_empty_for_incomplete_window(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_hca_compressor_forward

        weights = {
            "kv_proj": [[1.0, 0.0], [0.0, 1.0]],
            "gate_proj": [[1.0, 0.0], [0.0, 1.0]],
            "position_bias": [[0.0, 0.0]],
            "kv_norm": [1.0, 1.0],
        }
        got = tiny_hca_compressor_forward(
            [[1.0, 0.0]],
            weights,
            compress_rate=2,
            rms_norm_eps=1e-6,
            rope_cos=[],
            rope_sin=[],
        )
        self.assertEqual(got, [])

    def test_tiny_csa_compressor_forward_matches_transformers_reference(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import run_tiny_csa_compressor_fixture

        report = run_tiny_csa_compressor_fixture()
        if report["status"] == "skipped":
            self.skipTest(report.get("reason", "torch/transformers unavailable"))
        self.assertEqual(report["fixture"], "csa-compressor-forward")
        self.assertEqual(report["status"], "ok")
        self.assertLessEqual(report["max_abs_error"], 1e-5)
        self.assertIn("CSA compressor Ca/Cb overlap", report["covered"])
        self.assertIn("CSA indexer scoring and top-k", report["not_covered"])

    def test_tiny_csa_compressor_returns_empty_for_incomplete_window(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_csa_compressor_forward

        weights = {
            "kv_proj": [[1.0, 0.0], [0.0, 1.0]],
            "gate_proj": [[1.0, 0.0], [0.0, 1.0]],
            "position_bias": [[0.0, 0.0]],
            "kv_norm": [1.0],
        }
        got = tiny_csa_compressor_forward(
            [[1.0, 0.0]],
            weights,
            compress_rate=2,
            rms_norm_eps=1e-6,
            rope_cos=[],
            rope_sin=[],
        )
        self.assertEqual(got, [])

    def test_tiny_csa_indexer_scorer_forward_matches_transformers_reference(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import run_tiny_csa_indexer_scorer_fixture

        report = run_tiny_csa_indexer_scorer_fixture()
        if report["status"] == "skipped":
            self.skipTest(report.get("reason", "torch/transformers unavailable"))
        self.assertEqual(report["fixture"], "csa-indexer-scorer-topk")
        self.assertEqual(report["status"], "ok")
        self.assertLessEqual(report["max_abs_error"], 1e-5)
        self.assertIn("Lightning indexer scoring", report["covered"])
        self.assertIn("stateful cache overlap", report["not_covered"])

    def test_unimplemented_forward_paths_fail_closed(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import (
            compressor_forward_placeholder,
            hyperconnection_forward_placeholder,
        )

        with self.assertRaisesRegex(NotImplementedError, "compressor forward parity"):
            compressor_forward_placeholder()
        with self.assertRaisesRegex(NotImplementedError, "hyperconnection forward parity"):
            hyperconnection_forward_placeholder()

    def test_invalid_specs_fail_closed(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec

        with self.assertRaisesRegex(ValueError, "qk_rope_head_dim cannot exceed head_dim"):
            DeepSeekV4AttentionSpec(
                hidden_size=32,
                num_attention_heads=4,
                head_dim=10,
                q_lora_rank=6,
                o_lora_rank=3,
                qk_rope_head_dim=11,
            )
        with self.assertRaisesRegex(ValueError, "num_attention_heads must be divisible"):
            DeepSeekV4AttentionSpec(
                hidden_size=32,
                num_attention_heads=5,
                head_dim=10,
                q_lora_rank=6,
                o_lora_rank=3,
                qk_rope_head_dim=2,
                num_output_groups=2,
            )
        with self.assertRaisesRegex(ValueError, "indexer dimensions require compression_ratio=4"):
            DeepSeekV4AttentionSpec(
                hidden_size=32,
                num_attention_heads=4,
                head_dim=10,
                q_lora_rank=6,
                o_lora_rank=3,
                qk_rope_head_dim=2,
                compression_ratio=2,
                index_n_heads=5,
                index_head_dim=3,
            )

    def test_tiny_hyperhead_collapse_matches_transformers_reference(self):
        try:
            import torch
            from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
            from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperHead
        except ImportError:
            self.skipTest("torch/transformers not available in this environment")

        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_hyperhead_collapse

        config = DeepseekV4Config(
            hidden_size=4,
            hc_mult=2,
            rms_norm_eps=1e-6,
            hc_eps=1e-6,
        )
        module = DeepseekV4HyperHead(config)
        module.eval()
        with torch.no_grad():
            module.hc_fn.copy_(torch.eye(8, dtype=torch.float32)[:2])
            module.hc_base.copy_(torch.tensor([0.1, -0.2], dtype=torch.float32))
            module.hc_scale.copy_(torch.tensor([0.5], dtype=torch.float32))

        hidden = torch.tensor([[[[3.0, 4.0, 0.0, 5.0], [1.0, 2.0, -1.0, 0.0]]]], dtype=torch.float32)
        with torch.no_grad():
            ref = module(hidden).squeeze(0).squeeze(0).tolist()

        got = tiny_hyperhead_collapse(
            hidden_streams=hidden.squeeze(0).tolist(),
            fn=module.hc_fn.tolist(),
            base=module.hc_base.tolist(),
            scale=float(module.hc_scale.item()),
            hc_mult=2,
            eps=1e-6,
            rms_norm_eps=1e-6,
        )
        collapsed = got["collapsed"][0]
        self.assertEqual(len(collapsed), len(ref))
        max_abs_error = max(abs(a - b) for a, b in zip(collapsed, ref))
        self.assertLessEqual(max_abs_error, 1e-5)

    def test_integrated_layer_hc_mult_matches_transformers_reference(self):
        try:
            import torch
            from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
            from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Model
        except ImportError:
            self.skipTest("torch/transformers not available in this environment")

        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import (
            Model,
            ModelArgs,
            _make_integrated_tiny_weights,
            run_integrated_layer_hc_mult_fixture,
            set_transformers_integrated_weights,
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
            "forward_parity_fixture": "integrated-layer",
            "rms_norm_eps": 1e-6,
            "hc_mult": 2,
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
        weights = _make_integrated_tiny_weights(hc_mult=2)
        pure_model = Model(args)
        pure_model.load_integrated_weights(weights)
        pure_out = pure_model([[0, 1]])

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
            hc_mult=2,
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
        ref = ref_out.tolist()  # [batch, seq, hc_mult, hidden]

        report = run_integrated_layer_hc_mult_fixture(reference_output=ref, weights=weights)
        if report["status"] == "skipped":
            self.skipTest(report.get("reason", "torch/transformers unavailable"))
        self.assertEqual(report["status"], "ok")
        self.assertLessEqual(report["max_abs_error"], 1e-5)


    # ------------------------------------------------------------------
    # Story 11 slice11: integrated compressor/indexer (CSA) attention ref
    # ------------------------------------------------------------------

    def _tiny_csa_attention_spec(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec

        return DeepSeekV4AttentionSpec(
            hidden_size=4,
            num_attention_heads=1,
            head_dim=4,
            q_lora_rank=4,
            o_lora_rank=4,
            qk_rope_head_dim=4,
            num_output_groups=1,
            compression_ratio=4,
            index_n_heads=2,
            index_head_dim=2,
        )

    @staticmethod
    def _dense_weights(rows, cols, scale=0.1):
        return [
            [round(scale * (i + j + 1) * (1.0 if (i + j) % 2 == 0 else -1.0), 4) for j in range(cols)]
            for i in range(rows)
        ]

    def _tiny_csa_attention_weights(self):
        # DS4 layout (input_dim, output_dim) consistent with compressor_shapes()
        # and indexer_shapes().  compressor_ape / indexer_compressor_ape are
        # (width, ratio) = (2*head_dim/index_head_dim, compression_ratio).
        return {
            "compressor_wkv": self._dense_weights(4, 8, 0.5),
            "compressor_wgate": self._dense_weights(4, 8, 0.25),
            "compressor_ape": self._dense_weights(8, 4, 0.05),
            "compressor_norm": [1.0, 1.0, 1.0, 1.0],
            "indexer_wq_b": self._dense_weights(4, 4, 0.5),
            "indexer_proj": self._dense_weights(4, 2, 0.3),
            "indexer_compressor_wkv": self._dense_weights(4, 4, 0.4),
            "indexer_compressor_wgate": self._dense_weights(4, 4, 0.2),
            "indexer_compressor_ape": self._dense_weights(4, 4, 0.05),
            "indexer_compressor_norm": [1.0, 1.0],
        }

    @staticmethod
    def _tiny_csa_hidden_states():
        # seq_len=8 -> 2 full windows of compression_ratio=4
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

    def _tiny_csa_hidden_states_12(self):
        return self._tiny_csa_hidden_states() + [
            [0.75, -0.25, 0.25, 0.0],
            [-0.5, 0.25, 0.75, 0.5],
            [0.1, -0.3, 0.6, 0.2],
            [0.4, 0.1, -0.2, 0.8],
        ]

    def _assert_close_matrix(self, got, want, tol=1e-5):
        self.assertEqual(len(got), len(want))
        for got_row, want_row in zip(got, want, strict=True):
            self.assertEqual(len(got_row), len(want_row))
            for got_value, want_value in zip(got_row, want_row, strict=True):
                if math.isinf(float(got_value)) or math.isinf(float(want_value)):
                    self.assertEqual(float(got_value), float(want_value))
                else:
                    self.assertLessEqual(abs(float(got_value) - float(want_value)), tol)

    def _assert_stateful_csa_matches_reference(self, got, ref, *, ratio=4, tol=1e-5):
        self.assertEqual(got["compressed_len"], ref["compressed_len"])
        self._assert_close_matrix(got["compressed_kv"], ref["compressed_kv"], tol=1e-12)
        self._assert_close_matrix(got["attended"], ref["attended"], tol=tol)
        self.assertEqual(got["topk_indices"], ref["topk_indices"])
        self.assertEqual(got["topk_mask"], ref["topk_mask"])
        for pos, (got_scores, ref_scores) in enumerate(zip(got["index_scores"], ref["index_scores"], strict=True)):
            valid = min((pos + 1) // ratio, got["compressed_len"])
            for entry in range(valid):
                self.assertLessEqual(abs(float(got_scores[entry]) - float(ref_scores[entry])), tol)
        for pos, (got_bias, ref_bias) in enumerate(zip(got["block_bias"], ref["block_bias"], strict=True)):
            valid = min((pos + 1) // ratio, got["compressed_len"])
            for entry in range(valid):
                self.assertEqual(float(got_bias[entry]), float(ref_bias[entry]))

    def test_stateful_csa_matches_full_reference_window_aligned(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_compressor_indexer_attention_reference, tiny_stateful_csa_attention

        spec = self._tiny_csa_attention_spec()
        weights = self._tiny_csa_attention_weights()
        hidden = self._tiny_csa_hidden_states_12()
        ref = tiny_compressor_indexer_attention_reference(spec, hidden, hidden, weights, index_topk=2)
        got = tiny_stateful_csa_attention(spec, hidden, hidden, weights, call_splits=[4, 4, 4], index_topk=2)

        self._assert_stateful_csa_matches_reference(got, ref)
        for pos in (0, 1, 2):
            self.assertTrue(all(float(v) == 0.0 for v in got["attended"][pos]))

    def test_stateful_csa_matches_full_reference_non_window_aligned(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import StatefulCSACache, tiny_compressor_indexer_attention_reference, tiny_stateful_csa_attention

        spec = self._tiny_csa_attention_spec()
        weights = self._tiny_csa_attention_weights()
        hidden = self._tiny_csa_hidden_states_12()
        ref = tiny_compressor_indexer_attention_reference(spec, hidden, hidden, weights, index_topk=2)
        got = tiny_stateful_csa_attention(spec, hidden, hidden, weights, call_splits=[3, 1, 5, 1, 2], index_topk=2)
        self._assert_stateful_csa_matches_reference(got, ref)

        corrupted = StatefulCSACache(spec, weights, index_topk=2)
        corrupted.step(hidden[:4], hidden[:4])
        corrupted._prev_window_hidden = []  # Teeth test: stale/missing Ca carry must be observable.
        tail = corrupted.step(hidden[4:], hidden[4:])
        self.assertGreater(self._max_abs_rows(tail["compressed_kv"], ref["compressed_kv"]), 1e-5)

    def test_stateful_csa_pending_and_prev_window_lifetime(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import StatefulCSACache

        spec = self._tiny_csa_attention_spec()
        weights = self._tiny_csa_attention_weights()
        cache = StatefulCSACache(spec, weights, index_topk=2)
        hidden = self._tiny_csa_hidden_states_12()
        for pos, row in enumerate(hidden):
            cache.step([row], [row])
            self.assertEqual(cache.abs_pos, pos + 1)
            self.assertLess(len(cache.pending_hidden), 4)
            self.assertEqual(len(cache.emitted_compressed_kv), (pos + 1) // 4)
            if pos < 3:
                self.assertEqual(cache.prev_window_hidden, [])
            elif pos < 7:
                self.assertEqual(cache.prev_window_hidden, hidden[:4])
            elif pos < 11:
                self.assertEqual(cache.prev_window_hidden, hidden[4:8])
            else:
                self.assertEqual(cache.prev_window_hidden, hidden[8:12])

    def test_stateful_csa_index_topk_exact_under_carry(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_compressor_indexer_attention_reference, tiny_stateful_csa_attention

        spec = self._tiny_csa_attention_spec()
        weights = self._tiny_csa_attention_weights()
        hidden = self._tiny_csa_hidden_states_12()
        ref = tiny_compressor_indexer_attention_reference(spec, hidden, hidden, weights, index_topk=1)
        got = tiny_stateful_csa_attention(spec, hidden, hidden, weights, call_splits=[5, 2, 1, 4], index_topk=1)

        self._assert_stateful_csa_matches_reference(got, ref)

    def test_stateful_csa_fails_closed_outside_subset(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import DeepSeekV4AttentionSpec, StatefulCSACache

        cases = [
            (DeepSeekV4AttentionSpec(4, 1, 4, 4, 4, 4, compression_ratio=2), {}, "compression_ratio=4"),
        ]
        for spec, weights, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(NotImplementedError, pattern):
                    StatefulCSACache(spec, weights, index_topk=1)
        with self.assertRaisesRegex(NotImplementedError, "num_key_value_heads=1"):
            StatefulCSACache(self._tiny_csa_attention_spec(), self._tiny_csa_attention_weights(), index_topk=1, num_key_value_heads=2)

    def test_stateful_csa_hc_mult_gt_1_single_chunk_matches_reference(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_compressor_indexer_attention_reference, tiny_stateful_csa_attention

        spec = self._tiny_csa_attention_spec()
        weights = self._tiny_csa_attention_weights()
        hidden = self._tiny_csa_hidden_states_12()
        ref = tiny_compressor_indexer_attention_reference(spec, hidden, hidden, weights, index_topk=1)
        got = tiny_stateful_csa_attention(spec, hidden, hidden, weights, call_splits=[len(hidden)], index_topk=1, hc_mult=2)

        self._assert_stateful_csa_matches_reference(got, ref)

    def test_stateful_csa_supports_hc_mult_gt_1(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import StatefulCSACache

        spec = self._tiny_csa_attention_spec()
        weights = self._tiny_csa_attention_weights()
        cache = StatefulCSACache(spec, weights, index_topk=1, hc_mult=2)
        hidden = self._tiny_csa_hidden_states_12()
        chunk = cache.step(hidden, hidden)

        self.assertEqual(len(chunk["attended"]), len(hidden))

    def test_stateful_csa_hc_mult_gt_1_is_deterministic_across_splits(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_stateful_csa_attention

        spec = self._tiny_csa_attention_spec()
        weights = self._tiny_csa_attention_weights()
        hidden = self._tiny_csa_hidden_states_12()
        one_chunk = tiny_stateful_csa_attention(spec, hidden, hidden, weights, call_splits=[12], index_topk=2, hc_mult=2)
        mixed = tiny_stateful_csa_attention(spec, hidden, hidden, weights, call_splits=[5, 2, 1, 4], index_topk=2, hc_mult=2)
        uniform = tiny_stateful_csa_attention(spec, hidden, hidden, weights, call_splits=[3, 3, 3, 3], index_topk=2, hc_mult=2)

        self.assertEqual(one_chunk, mixed)
        self.assertEqual(mixed, uniform)

    def test_stateful_csa_is_deterministic_across_splits(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_stateful_csa_attention

        spec = self._tiny_csa_attention_spec()
        weights = self._tiny_csa_attention_weights()
        hidden = self._tiny_csa_hidden_states_12()
        aligned = tiny_stateful_csa_attention(spec, hidden, hidden, weights, call_splits=[4, 4, 4], index_topk=2)
        non_aligned = tiny_stateful_csa_attention(spec, hidden, hidden, weights, call_splits=[3, 1, 5, 1, 2], index_topk=2)
        repeat = tiny_stateful_csa_attention(spec, hidden, hidden, weights, call_splits=[3, 1, 5, 1, 2], index_topk=2)

        self.assertEqual(non_aligned, repeat)
        self.assertEqual(aligned, non_aligned)

    def test_stateful_csa_markers_absent_and_11_16_cache_unchanged(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import IncrementalSlidingKVCache
        from ds4_ft_mlx.deepseek_v4_dequant import dequantize_expert_packed
        from ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4 import forward_parity_blockers

        self.assertFalse(Path("/Volumes/Data NVME/mlx-ft/ds4/.deepseek-v4-forward-parity-ok").exists())
        self._assert_model_4bit_absent_or_valid_current_artifact()
        blockers = forward_parity_blockers()
        self.assertIn("full MLX shimmed-checkpoint load/forward and MLX generation smoke (Track B only; DS4-GGUF base generation is its own Track-A gate, .ds4-gguf-generate-ok, not a forward-parity blocker)", blockers)
        self.assertIn("full MoE parity with packed FP4/I8 expert dequant and expert kernels", blockers)
        with self.assertRaises(NotImplementedError):
            dequantize_expert_packed("i8", b"\x00", scales=b"\x7f", shape=(1,))
        with self.assertRaisesRegex(ValueError, "requires non-None scales"):
            dequantize_expert_packed("fp4", b"\x00" * 16, scales=None, shape=(1, 32))

        cache = IncrementalSlidingKVCache(self._tiny_kv_spec(), self._tiny_kv_weights(), sliding_window=2)
        cache.step(self._tiny_kv_hidden_states()[0])
        self.assertEqual(cache.next_pos, 1)

    def test_tiny_multihead_csa_fusion_matches_transformers_reference(self):
        try:
            import torch
            from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
            from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
                DeepseekV4Attention,
                DeepseekV4RotaryEmbedding,
            )
        except Exception as exc:  # pragma: no cover - exercised only outside the MLX venv.
            self.skipTest(f"torch/transformers unavailable: {exc}")

        from ds4_ft_mlx.deepseek_v4_attention_spec import (
            DeepSeekV4AttentionSpec,
            tiny_multihead_csa_fusion_reference,
        )

        seq_len = 8
        hidden_size = 8
        head_dim = 4
        compress_rate = 4
        compress_theta = 160000.0
        rms_norm_eps = 1e-6

        def fill_matrix(param, *, scale):
            values = torch.arange(param.numel(), dtype=torch.float32).reshape_as(param)
            param.copy_(((values % 17.0) - 8.0) * scale)

        def fill_vector(param, *, start, step):
            values = start + step * torch.arange(param.numel(), dtype=torch.float32)
            param.copy_(values.reshape_as(param))

        for num_heads, o_groups in ((2, 1), (4, 1), (4, 2)):
            with self.subTest(num_heads=num_heads, o_groups=o_groups):
                config = DeepseekV4Config(
                    num_hidden_layers=1,
                    hidden_size=hidden_size,
                    num_attention_heads=num_heads,
                    head_dim=head_dim,
                    q_lora_rank=hidden_size,
                    o_lora_rank=4,
                    o_groups=o_groups,
                    layer_types=["compressed_sparse_attention"],
                    sliding_window=seq_len,
                    compress_rates={"compressed_sparse_attention": compress_rate, "hca": 128},
                    index_n_heads=2,
                    index_head_dim=head_dim,
                    index_topk=2,
                    partial_rotary_factor=1.0,
                    rope_theta=10000.0,
                    compress_rope_theta=compress_theta,
                    rms_norm_eps=rms_norm_eps,
                    attention_dropout=0.0,
                    rope_parameters={
                        "main": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
                        "compress": {"rope_type": "default", "rope_theta": compress_theta, "partial_rotary_factor": 1.0},
                    },
                )
                config._attn_implementation = "eager"
                module = DeepseekV4Attention(config, layer_idx=0).eval()

                with torch.no_grad():
                    fill_matrix(module.q_a_proj.weight, scale=0.017)
                    fill_matrix(module.q_b_proj.weight, scale=0.013)
                    fill_matrix(module.kv_proj.weight, scale=0.019)
                    fill_matrix(module.o_a_proj.weight, scale=0.011)
                    fill_matrix(module.o_b_proj.weight, scale=0.007)
                    fill_vector(module.q_a_norm.weight, start=0.85, step=0.03)
                    fill_vector(module.kv_norm.weight, start=0.9, step=0.025)
                    fill_vector(module.sinks, start=-0.2, step=0.35)
                    fill_matrix(module.compressor.kv_proj.weight, scale=0.021)
                    fill_matrix(module.compressor.gate_proj.weight, scale=0.009)
                    fill_matrix(module.compressor.position_bias, scale=0.015)
                    fill_vector(module.compressor.kv_norm.weight, start=0.82, step=0.04)
                    fill_matrix(module.compressor.indexer.kv_proj.weight, scale=0.018)
                    fill_matrix(module.compressor.indexer.gate_proj.weight, scale=0.008)
                    fill_matrix(module.compressor.indexer.position_bias, scale=0.012)
                    fill_vector(module.compressor.indexer.kv_norm.weight, start=0.88, step=0.035)
                    fill_matrix(module.compressor.indexer.q_b_proj.weight, scale=0.016)
                    fill_matrix(module.compressor.indexer.scorer.weights_proj.weight, scale=0.014)

                hidden = torch.tensor(
                    [[
                        [math.sin((pos + 1) * (dim + 1)) * 0.17 + (pos - dim) * 0.011 for dim in range(hidden_size)]
                        for pos in range(seq_len)
                    ]],
                    dtype=torch.float32,
                )
                position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
                attention_mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), dtype=torch.float32)
                for query_pos in range(seq_len):
                    for key_pos in range(seq_len):
                        if key_pos <= query_pos:
                            attention_mask[0, 0, query_pos, key_pos] = 0.0
                rotary = DeepseekV4RotaryEmbedding(config)
                with torch.no_grad():
                    cos, sin = rotary(hidden, position_ids=position_ids, layer_type="compress")
                    ref, _ = module(
                        hidden,
                        {"compress": (cos, sin)},
                        position_ids,
                        attention_mask,
                    )

                weights = {
                    "q_a_proj.weight": module.q_a_proj.weight.tolist(),
                    "q_norm.weight": module.q_a_norm.weight.tolist(),
                    "q_b_proj.weight": module.q_b_proj.weight.tolist(),
                    "kv_proj.weight": module.kv_proj.weight.tolist(),
                    "kv_norm.weight": module.kv_norm.weight.tolist(),
                    "o_a_proj.weight": module.o_a_proj.weight.tolist(),
                    "o_b_proj.weight": module.o_b_proj.weight.tolist(),
                    "sinks": module.sinks.tolist(),
                    "compressor_wkv": module.compressor.kv_proj.weight.T.tolist(),
                    "compressor_wgate": module.compressor.gate_proj.weight.T.tolist(),
                    "compressor_ape": module.compressor.position_bias.T.tolist(),
                    "compressor_norm": module.compressor.kv_norm.weight.tolist(),
                    "indexer_wq_b": module.compressor.indexer.q_b_proj.weight.T.tolist(),
                    "indexer_proj": module.compressor.indexer.scorer.weights_proj.weight.T.tolist(),
                    "indexer_compressor_wkv": module.compressor.indexer.kv_proj.weight.T.tolist(),
                    "indexer_compressor_wgate": module.compressor.indexer.gate_proj.weight.T.tolist(),
                    "indexer_compressor_ape": module.compressor.indexer.position_bias.T.tolist(),
                    "indexer_compressor_norm": module.compressor.indexer.kv_norm.weight.tolist(),
                }
                spec = DeepSeekV4AttentionSpec(
                    hidden_size=hidden_size,
                    num_attention_heads=num_heads,
                    head_dim=head_dim,
                    q_lora_rank=hidden_size,
                    o_lora_rank=4,
                    qk_rope_head_dim=head_dim,
                    num_output_groups=o_groups,
                    compression_ratio=compress_rate,
                    index_n_heads=2,
                    index_head_dim=head_dim,
                )
                got = tiny_multihead_csa_fusion_reference(
                    spec,
                    hidden.squeeze(0).tolist(),
                    weights,
                    sliding_window=seq_len,
                    rms_norm_eps=rms_norm_eps,
                    rope_theta=compress_theta,
                    position_ids=position_ids.squeeze(0).tolist(),
                    index_topk=config.index_topk,
                )

                max_abs_error = 0.0
                for got_row, ref_row in zip(got["output"], ref.squeeze(0).tolist(), strict=True):
                    for got_value, ref_value in zip(got_row, ref_row, strict=True):
                        max_abs_error = max(max_abs_error, abs(float(got_value) - float(ref_value)))
                self.assertLessEqual(max_abs_error, 1e-5)
                self.assertIn("stateless multi-head CSA fusion", got["covered"])
                self.assertIn("stateful multi-head CSA cache", got["not_covered"])
                self.assertTrue(
                    all(
                        value == 0.0 or (math.isinf(value) and value < 0.0)
                        for row in got["block_bias"]
                        for value in row
                    )
                )

    def test_tiny_stateful_csa_fusion_matches_transformers_reference(self):
        try:
            import torch
            from transformers.cache_utils import DynamicCache
            from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
            from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
                DeepseekV4Attention,
                DeepseekV4CSACache,
                DeepseekV4RotaryEmbedding,
            )
        except Exception as exc:  # pragma: no cover - exercised only outside the MLX venv.
            self.skipTest(f"torch/transformers unavailable: {exc}")

        from ds4_ft_mlx.deepseek_v4_attention_spec import (
            DeepSeekV4AttentionSpec,
            tiny_stateful_csa_fusion_reference,
        )

        seq_len = 8
        hidden_size = 8
        head_dim = 4
        compress_rate = 4
        sliding_window = 4
        compress_theta = 160000.0
        rms_norm_eps = 1e-6
        splits_cases = ([8], [3, 1, 4], [2, 2, 2, 2])

        def fill_matrix(param, *, scale):
            values = torch.arange(param.numel(), dtype=torch.float32).reshape_as(param)
            param.copy_(((values % 17.0) - 8.0) * scale)

        def fill_vector(param, *, start, step):
            values = start + step * torch.arange(param.numel(), dtype=torch.float32)
            param.copy_(values.reshape_as(param))

        def build_config(num_heads, o_groups):
            config = DeepseekV4Config(
                num_hidden_layers=1,
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                head_dim=head_dim,
                q_lora_rank=hidden_size,
                o_lora_rank=4,
                o_groups=o_groups,
                layer_types=["compressed_sparse_attention"],
                sliding_window=sliding_window,
                compress_rates={"compressed_sparse_attention": compress_rate, "heavily_compressed_attention": 128, "hca": 128},
                index_n_heads=2,
                index_head_dim=head_dim,
                index_topk=2,
                partial_rotary_factor=1.0,
                rope_theta=10000.0,
                compress_rope_theta=compress_theta,
                rms_norm_eps=rms_norm_eps,
                attention_dropout=0.0,
                rope_parameters={
                    "main": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
                    "compress": {"rope_type": "default", "rope_theta": compress_theta, "partial_rotary_factor": 1.0},
                },
            )
            config._attn_implementation = "eager"
            return config

        def build_module(config):
            module = DeepseekV4Attention(config, layer_idx=0).eval()
            with torch.no_grad():
                fill_matrix(module.q_a_proj.weight, scale=0.017)
                fill_matrix(module.q_b_proj.weight, scale=0.013)
                fill_matrix(module.kv_proj.weight, scale=0.019)
                fill_matrix(module.o_a_proj.weight, scale=0.011)
                fill_matrix(module.o_b_proj.weight, scale=0.007)
                fill_vector(module.q_a_norm.weight, start=0.85, step=0.03)
                fill_vector(module.kv_norm.weight, start=0.9, step=0.025)
                fill_vector(module.sinks, start=-0.2, step=0.35)
                fill_matrix(module.compressor.kv_proj.weight, scale=0.021)
                fill_matrix(module.compressor.gate_proj.weight, scale=0.015)
                fill_matrix(module.compressor.position_bias, scale=0.015)
                fill_vector(module.compressor.kv_norm.weight, start=0.82, step=0.04)
                fill_matrix(module.compressor.indexer.kv_proj.weight, scale=0.018)
                fill_matrix(module.compressor.indexer.gate_proj.weight, scale=0.008)
                fill_matrix(module.compressor.indexer.position_bias, scale=0.012)
                fill_vector(module.compressor.indexer.kv_norm.weight, start=0.88, step=0.035)
                fill_matrix(module.compressor.indexer.q_b_proj.weight, scale=0.016)
                fill_matrix(module.compressor.indexer.scorer.weights_proj.weight, scale=0.014)
            return module

        def make_csa_cache(config):
            cache = DynamicCache()
            cache.layers = [DeepseekV4CSACache(config)]
            return cache

        def window_causal_mask(num_q, past_len, start):
            kv_len = past_len + num_q
            mask = torch.full((1, 1, num_q, kv_len), float("-inf"), dtype=torch.float32)
            for q in range(num_q):
                abs_q = start + q
                for k in range(kv_len):
                    abs_k = start - past_len + k
                    if abs_k <= abs_q and abs_k >= abs_q - sliding_window + 1:
                        mask[0, 0, q, k] = 0.0
            return mask

        def run_real_forward(call_splits):
            module = build_module(config)
            rotary = DeepseekV4RotaryEmbedding(config)
            if call_splits == [seq_len]:
                with torch.no_grad():
                    cos, sin = rotary(hidden, position_ids=position_ids, layer_type="compress")
                    out, _ = module(
                        hidden,
                        {"compress": (cos, sin)},
                        position_ids,
                        window_causal_mask(seq_len, past_len=0, start=0),
                    )
                return out.squeeze(0).tolist()

            cache = make_csa_cache(config)
            outs = []
            offset = 0
            with torch.no_grad():
                for size in call_splits:
                    h = hidden[:, offset: offset + size, :]
                    pos = position_ids[:, offset: offset + size]
                    past_len = min(sliding_window - 1, offset)
                    cos, sin = rotary(h, position_ids=pos, layer_type="compress")
                    out, _ = module(
                        h,
                        {"compress": (cos, sin)},
                        pos,
                        window_causal_mask(size, past_len=past_len, start=offset),
                        past_key_values=cache,
                    )
                    outs.append(out)
                    offset += size
            return torch.cat(outs, dim=1).squeeze(0).tolist()

        def max_abs_rows(a, b):
            max_abs_error = 0.0
            for row_a, row_b in zip(a, b, strict=True):
                for value_a, value_b in zip(row_a, row_b, strict=True):
                    max_abs_error = max(max_abs_error, abs(float(value_a) - float(value_b)))
            return max_abs_error

        for num_heads, o_groups in ((1, 1), (4, 1), (4, 2), (2, 2)):
            with self.subTest(num_attention_heads=num_heads, o_groups=o_groups):
                config = build_config(num_heads, o_groups)
                hidden = torch.tensor(
                    [[
                        [math.sin((pos + 1) * (dim + 1)) * 0.17 + (pos - dim) * 0.011 for dim in range(hidden_size)]
                        for pos in range(seq_len)
                    ]],
                    dtype=torch.float32,
                )
                position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
                base_module = build_module(config)
                with torch.no_grad():
                    q_residual = base_module.q_a_norm(base_module.q_a_proj(hidden)).squeeze(0).tolist()
                weights = {
                    "q_a_proj.weight": base_module.q_a_proj.weight.tolist(),
                    "q_norm.weight": base_module.q_a_norm.weight.tolist(),
                    "q_b_proj.weight": base_module.q_b_proj.weight.tolist(),
                    "kv_proj.weight": base_module.kv_proj.weight.tolist(),
                    "kv_norm.weight": base_module.kv_norm.weight.tolist(),
                    "o_a_proj.weight": base_module.o_a_proj.weight.tolist(),
                    "o_b_proj.weight": base_module.o_b_proj.weight.tolist(),
                    "sinks": base_module.sinks.tolist(),
                    "compressor_wkv": base_module.compressor.kv_proj.weight.T.tolist(),
                    "compressor_wgate": base_module.compressor.gate_proj.weight.T.tolist(),
                    "compressor_ape": base_module.compressor.position_bias.T.tolist(),
                    "compressor_norm": base_module.compressor.kv_norm.weight.tolist(),
                    "indexer_wq_b": base_module.compressor.indexer.q_b_proj.weight.T.tolist(),
                    "indexer_proj": base_module.compressor.indexer.scorer.weights_proj.weight.T.tolist(),
                    "indexer_compressor_wkv": base_module.compressor.indexer.kv_proj.weight.T.tolist(),
                    "indexer_compressor_wgate": base_module.compressor.indexer.gate_proj.weight.T.tolist(),
                    "indexer_compressor_ape": base_module.compressor.indexer.position_bias.T.tolist(),
                    "indexer_compressor_norm": base_module.compressor.indexer.kv_norm.weight.tolist(),
                }
                spec = DeepSeekV4AttentionSpec(
                    hidden_size=hidden_size,
                    num_attention_heads=num_heads,
                    head_dim=head_dim,
                    q_lora_rank=hidden_size,
                    o_lora_rank=4,
                    qk_rope_head_dim=head_dim,
                    num_output_groups=o_groups,
                    compression_ratio=compress_rate,
                    index_n_heads=2,
                    index_head_dim=head_dim,
                )

                real_outputs = {tuple(splits): run_real_forward(list(splits)) for splits in splits_cases}
                spec_outputs = {}
                for splits in splits_cases:
                    got = tiny_stateful_csa_fusion_reference(
                        spec,
                        hidden.squeeze(0).tolist(),
                        q_residual,
                        weights,
                        call_splits=list(splits),
                        sliding_window=sliding_window,
                        rms_norm_eps=rms_norm_eps,
                        rope_theta=compress_theta,
                        position_ids=position_ids.squeeze(0).tolist(),
                        index_topk=config.index_topk,
                    )
                    spec_outputs[tuple(splits)] = got["output"]
                    self.assertLessEqual(max_abs_rows(got["output"], real_outputs[tuple(splits)]), 1e-5)
                    self.assertIn("stateful single-head CSA fusion", got["covered"])
                    self.assertIn("stateful multi-head CSA cache", got["not_covered"])
                    self.assertTrue(
                        all(
                            value == 0.0 or (math.isinf(value) and value < 0.0)
                            for row in got["block_bias"]
                            for value in row
                        )
                    )

                baseline = spec_outputs[(8,)]
                real_baseline = real_outputs[(8,)]
                for splits in splits_cases:
                    self.assertLessEqual(max_abs_rows(spec_outputs[tuple(splits)], baseline), 1e-5)
                    self.assertLessEqual(max_abs_rows(real_outputs[tuple(splits)], real_baseline), 1e-5)

    def test_tiny_csa_attention_reference_output_shapes(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_compressor_indexer_attention_reference

        spec = self._tiny_csa_attention_spec()
        weights = self._tiny_csa_attention_weights()
        hidden = self._tiny_csa_hidden_states()
        got = tiny_compressor_indexer_attention_reference(
            spec, hidden, hidden, weights, index_topk=2
        )
        self.assertEqual(len(got["compressed_kv"]), 2)
        self.assertEqual(len(got["compressed_kv"][0]), spec.head_dim)
        self.assertEqual(len(got["index_scores"]), 8)
        self.assertEqual(len(got["index_scores"][0]), 2)
        self.assertEqual(len(got["block_bias"]), 8)
        self.assertEqual(len(got["block_bias"][0]), 2)
        self.assertEqual(len(got["attended"]), 8)
        self.assertEqual(len(got["attended"][0]), spec.head_dim)
        self.assertEqual(got["compressed_len"], 2)

    def test_tiny_csa_attention_reference_causal_block_bias_threshold(self):
        # index_topk >= n_windows isolates the causal threshold component.
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_compressor_indexer_attention_reference

        spec = self._tiny_csa_attention_spec()
        weights = self._tiny_csa_attention_weights()
        hidden = self._tiny_csa_hidden_states()
        got = tiny_compressor_indexer_attention_reference(
            spec, hidden, hidden, weights, index_topk=2
        )
        bias = got["block_bias"]
        for pos in range(8):
            threshold = (pos + 1) // 4
            for entry in range(2):
                expected_masked = entry >= threshold
                got_masked = math.isinf(bias[pos][entry]) and bias[pos][entry] < 0
                self.assertEqual(
                    got_masked, expected_masked, f"pos={pos} entry={entry} bias={bias[pos][entry]}"
                )

    def test_tiny_csa_attention_reference_all_masked_row_is_zero(self):
        # Positions with causal threshold 0 (pos 0,1,2) have every entry
        # masked, so the attended output for those rows must be all zeros.
        # Position 3 has threshold (3+1)//4 == 1, so entry 0 is allowed and the
        # attended output should be nonzero.
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_compressor_indexer_attention_reference

        spec = self._tiny_csa_attention_spec()
        weights = self._tiny_csa_attention_weights()
        hidden = self._tiny_csa_hidden_states()
        got = tiny_compressor_indexer_attention_reference(
            spec, hidden, hidden, weights, index_topk=2
        )
        for pos in (0, 1, 2):
            for value in got["attended"][pos]:
                self.assertEqual(float(value), 0.0, f"pos={pos} expected zero attended output")
        self.assertTrue(any(abs(float(v)) > 0.0 for v in got["attended"][3]))

    def test_tiny_csa_attention_reference_compressor_matches_validated_helper(self):
        # The integrated reference must reuse the Transformers-validated CSA
        # compressor gating, so compressed_kv must match a direct call to
        # tiny_csa_compressor_forward with the same weights translated to the
        # internal (kv_proj/gate_proj/position_bias/kv_norm) names.
        from ds4_ft_mlx.deepseek_v4_attention_spec import (
            _rope_cos_sin,
            tiny_csa_compressor_forward,
            tiny_compressor_indexer_attention_reference,
        )

        spec = self._tiny_csa_attention_spec()
        weights = self._tiny_csa_attention_weights()
        hidden = self._tiny_csa_hidden_states()
        got = tiny_compressor_indexer_attention_reference(
            spec, hidden, hidden, weights, index_topk=2
        )
        ratio = 4
        n_windows = 2
        positions = [w * ratio for w in range(n_windows)]
        cos, sin = _rope_cos_sin(positions, spec.head_dim, 10000.0)
        internal = {
            "kv_proj": weights["compressor_wkv"],
            "gate_proj": weights["compressor_wgate"],
            "position_bias": [[weights["compressor_ape"][d][t] for d in range(2 * spec.head_dim)] for t in range(ratio)],
            "kv_norm": weights["compressor_norm"],
        }
        ref_compressed = tiny_csa_compressor_forward(
            hidden, internal, compress_rate=ratio, rms_norm_eps=1e-6, rope_cos=cos, rope_sin=sin
        )
        max_err = 0.0
        for got_row, ref_row in zip(got["compressed_kv"], ref_compressed):
            for gv, rv in zip(got_row, ref_row):
                max_err = max(max_err, abs(float(gv) - float(rv)))
        self.assertLessEqual(max_err, 1e-12)

    def test_tiny_csa_attention_reference_index_topk_restricts_block_bias(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_compressor_indexer_attention_reference

        spec = self._tiny_csa_attention_spec()
        weights = self._tiny_csa_attention_weights()
        hidden = self._tiny_csa_hidden_states()
        got = tiny_compressor_indexer_attention_reference(
            spec, hidden, hidden, weights, index_topk=1
        )
        # Each row may have at most one allowed (0.0) entry after top-k gather.
        for pos in range(8):
            allowed = sum(1 for v in got["block_bias"][pos] if not (math.isinf(v) and v < 0))
            self.assertLessEqual(allowed, 1, f"pos={pos} top-k=1 should allow at most one entry")

    def test_tiny_csa_attention_reference_rejects_non_csa_compression_ratio(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import (
            DeepSeekV4AttentionSpec,
            tiny_compressor_indexer_attention_reference,
        )

        spec = DeepSeekV4AttentionSpec(
            hidden_size=4,
            num_attention_heads=1,
            head_dim=4,
            q_lora_rank=4,
            o_lora_rank=4,
            qk_rope_head_dim=4,
            num_output_groups=1,
            compression_ratio=0,
        )
        with self.assertRaisesRegex(ValueError, "compression_ratio"):
            tiny_compressor_indexer_attention_reference(spec, [[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]], {})

    def test_tiny_csa_attention_reference_rejects_missing_weights(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import tiny_compressor_indexer_attention_reference

        spec = self._tiny_csa_attention_spec()
        hidden = self._tiny_csa_hidden_states()
        with self.assertRaisesRegex(ValueError, "missing"):
            tiny_compressor_indexer_attention_reference(spec, hidden, hidden, {"compressor_wkv": [[0.0] * 8] * 4})

    def test_run_tiny_compressor_indexer_attention_fixture_reports_partial_coverage(self):
        from ds4_ft_mlx.deepseek_v4_attention_spec import run_tiny_compressor_indexer_attention_fixture

        report = run_tiny_compressor_indexer_attention_fixture()
        self.assertEqual(report["fixture"], "compressor-indexer-attention")
        self.assertEqual(report["status"], "ok")
        self.assertLessEqual(report["max_abs_error"], 1e-12)
        self.assertIn("CSA compressor softmax-gated KV compression", report["covered"])
        self.assertIn("CSA Lightning indexer routing", report["covered"])
        self.assertIn("causal block-bias threshold", report["covered"])
        self.assertIn("MLX compressor/indexer port", report["not_covered"])
        self.assertIn("stateful cache overlap", report["not_covered"])
        self.assertIn("full DeepSeekV4Model layer integration", report["not_covered"])


if __name__ == "__main__":
    unittest.main()
