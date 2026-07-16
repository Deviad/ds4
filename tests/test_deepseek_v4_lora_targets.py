#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_SRC = REPO_ROOT / "python-envs" / "mlx" / "src"


class DeepSeekV4LoraTargetTests(unittest.TestCase):
    def setUp(self):
        self._old_path = list(sys.path)
        sys.path.insert(0, str(MLX_SRC))

    def tearDown(self):
        sys.path[:] = self._old_path

    def test_supported_aliases_are_ds4_runtime_internal_targets_only(self):
        from ds4_ft_mlx import lora_targets

        self.assertEqual(lora_targets.SUPPORTED_ALIASES, ("q_a", "q_b", "kv"))
        self.assertEqual(lora_targets.supported_ds4_targets(), ("attn_q_a", "attn_q_b", "attn_kv"))
        self.assertEqual(lora_targets.supported_mlx_keys(), ("self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_proj"))
        self.assertNotIn("output", lora_targets.supported_ds4_targets())
        self.assertNotIn("lm_head", lora_targets.supported_mlx_keys())

    def test_build_lora_parameters_uses_mlx_lm_keys_mechanism(self):
        from ds4_ft_mlx.lora_targets import build_lora_parameters

        params = build_lora_parameters(rank=8, scale=16.0, dropout=0.05)
        self.assertEqual(
            params,
            {
                "rank": 8,
                "scale": 16.0,
                "dropout": 0.05,
                "keys": ["self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_proj"],
            },
        )

    def test_resolve_aliases_accepts_only_explicit_supported_subset(self):
        from ds4_ft_mlx.lora_targets import resolve_target_aliases

        self.assertEqual(resolve_target_aliases(["kv", "q_a"]), ["self_attn.kv_proj", "self_attn.q_a_proj"])
        self.assertEqual(resolve_target_aliases(["q_a", "q_a", "q_b"]), ["self_attn.q_a_proj", "self_attn.q_b_proj"])
        with self.assertRaisesRegex(ValueError, "unsupported DS4 MLX LoRA target"):
            resolve_target_aliases(["all-linear"])
        with self.assertRaisesRegex(ValueError, "unsupported DS4 MLX LoRA target"):
            resolve_target_aliases(["default"])
        with self.assertRaisesRegex(ValueError, "unsupported DS4 MLX LoRA target"):
            resolve_target_aliases([])

    def test_validate_mlx_lora_parameters_rejects_default_all_linear_behavior(self):
        from ds4_ft_mlx.lora_targets import validate_lora_parameters

        with self.assertRaisesRegex(ValueError, "would default to all eligible linear"):
            validate_lora_parameters({"rank": 8, "scale": 20.0, "dropout": 0.0})
        with self.assertRaisesRegex(ValueError, "would default to all eligible linear"):
            validate_lora_parameters({"rank": 8, "scale": 20.0, "dropout": 0.0, "keys": []})

    def test_validate_mlx_lora_parameters_rejects_unsupported_module_families(self):
        from ds4_ft_mlx.lora_targets import validate_lora_parameters

        rejected = [
            "self_attn.o_proj",
            "self_attn.wo_a",
            "self_attn.wo_b",
            "lm_head",
            "output",
            "mlp.experts.0.w1",
            "mlp.switch_mlp.gate_proj",
            "self_attn.compressor.wkv",
            "self_attn.indexer.wq_b",
            "embed_tokens",
        ]
        for key in rejected:
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "unsupported DS4 MLX LoRA target"):
                    validate_lora_parameters({"rank": 8, "scale": 20.0, "dropout": 0.0, "keys": [key]})

    def test_validate_mlx_lora_parameters_accepts_only_allowed_keys(self):
        from ds4_ft_mlx.lora_targets import build_lora_parameters, validate_lora_parameters

        params = build_lora_parameters(targets=["q_b", "kv"], rank=4, scale=8.0, dropout=0.0)
        self.assertEqual(validate_lora_parameters(params), ("self_attn.q_b_proj", "self_attn.kv_proj"))

    def test_documented_mlx_lm_source_contract_is_inspection_only(self):
        from ds4_ft_mlx.lora_targets import MLX_LM_LORA_SOURCE_CONTRACT

        self.assertIn("linear_to_lora_layers", MLX_LM_LORA_SOURCE_CONTRACT)
        self.assertIn("config.get('keys'", MLX_LM_LORA_SOURCE_CONTRACT)
        self.assertIn("no CLI target-module argument", MLX_LM_LORA_SOURCE_CONTRACT)


if __name__ == "__main__":
    unittest.main()
