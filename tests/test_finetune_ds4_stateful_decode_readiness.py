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


def absent_probe():
    return {
        "probe_status": "absent",
        "import_ok": True,
        "import_error": None,
        "seam_param_tokens": list(finetune_ds4.STATEFUL_DECODE_SEAM_TOKEN_LIST),
        "surface": {
            "Model.__call__": {"params": ["self", "input_ids"], "has_seam_param": False},
            "Model._real_forward": {"params": ["self", "input_ids"], "has_seam_param": False},
            "Model._real_layer_forward": {"params": ["self", "args", "h", "layer_w"], "has_seam_param": False},
            "_attention_mlx": {"params": ["args", "x", "weights", "index_topk"], "has_seam_param": False},
            "_csa_attention_mlx": {"params": ["args", "x", "weights", "index_topk"], "has_seam_param": False},
        },
        "module_seam_symbols": [],
    }


def skipped_probe():
    return {
        "probe_status": "skipped",
        "import_ok": False,
        "import_error": "MLX real-mode dependencies unavailable: No module named 'mlx'",
        "seam_param_tokens": list(finetune_ds4.STATEFUL_DECODE_SEAM_TOKEN_LIST),
        "surface": {},
        "module_seam_symbols": [],
    }


def detected_probe():
    probe = absent_probe()
    probe["probe_status"] = "detected"
    probe["surface"] = {
        **probe["surface"],
        "Model._real_forward": {"params": ["self", "input_ids", "kv_cache"], "has_seam_param": True},
    }
    probe["module_seam_symbols"] = ["StatefulDecodeCache"]
    return probe


class StatefulDecodeReadinessTests(unittest.TestCase):
    def test_builder_absent_probe_is_not_ready(self):
        block = finetune_ds4._stateful_decode_readiness(absent_probe())
        self.assertEqual(block["block"], "stateful-decode-readiness")
        self.assertIs(block["seam_available"], False)
        self.assertEqual(block["probe_status"], "absent")
        self.assertEqual(block["status"], "fail-closed")
        self.assertEqual(block["decision"], "not-ready")
        self.assertIs(block["fail_closed"], True)
        self.assertIs(block["ready"], False)
        self.assertIn("stateless one-shot", block["verdict"])
        self.assertIn("future_proof_criteria", block)
        self.assertGreaterEqual(len(block["future_proof_criteria"]), 4)

    def test_builder_skipped_probe_is_not_ready(self):
        block = finetune_ds4._stateful_decode_readiness(skipped_probe())
        self.assertIs(block["seam_available"], False)
        self.assertEqual(block["probe_status"], "skipped")
        self.assertEqual(block["decision"], "not-ready")
        self.assertIs(block["fail_closed"], True)
        self.assertIs(block["ready"], False)
        self.assertIs(block["evidence"]["import_ok"], False)
        self.assertIn("No module named", block["evidence"]["import_error"])

    def test_builder_hypothetical_detected_still_not_ready(self):
        block = finetune_ds4._stateful_decode_readiness(detected_probe())
        self.assertIs(block["seam_available"], True)
        self.assertEqual(block["probe_status"], "detected")
        self.assertEqual(block["decision"], "not-ready")
        self.assertEqual(block["status"], "fail-closed")
        self.assertIs(block["fail_closed"], True)
        self.assertIs(block["ready"], False)
        self.assertIn("necessary-but-insufficient", block["no_false_positive_policy"])

    def test_seam_token_hit_excludes_false_positives(self):
        tokens = finetune_ds4.STATEFUL_DECODE_SEAM_TOKENS
        self.assertIs(finetune_ds4._seam_token_hit("DeepSeekV4AttentionSpec", tokens), False)
        self.assertIs(finetune_ds4._seam_token_hit("__cached__", tokens), False)
        self.assertIs(finetune_ds4._seam_token_hit("input_ids", tokens), False)
        self.assertIs(finetune_ds4._seam_token_hit("kv_cache", tokens), True)
        self.assertIs(finetune_ds4._seam_token_hit("cache_position", tokens), True)
        self.assertIs(finetune_ds4._seam_token_hit("offset", tokens), True)
        self.assertIs(finetune_ds4._seam_token_hit("KVCache", tokens), True)

    def test_spec_layer_seam_labeled_non_production(self):
        spec = finetune_ds4._stateful_decode_readiness(absent_probe())["spec_layer_seam"]
        self.assertIs(spec["proven_but_non_production"], True)
        self.assertIs(spec["is_production_runtime"], False)
        self.assertEqual(spec["module"], "ds4_ft_mlx.deepseek_v4_attention_spec")
        self.assertEqual(spec["proven_in_stories"], ["11.16", "11.27", "11.28", "11.29"])
        for symbol in (
            "StatefulCSACache.step",
            "StatefulCSACache.step_fusion",
            "IncrementalSlidingKVCache",
            "tiny_greedy_decode",
            "tiny_stateful_csa_fusion_reference",
        ):
            self.assertIn(symbol, spec["symbols"])

    def test_additive_block_does_not_change_honesty_counters(self):
        with tempfile.TemporaryDirectory() as td:
            report = finetune_ds4.build_forward_parity_readiness_report(
                mlx_work=pathlib.Path(td) / "mlx",
                partials=stub_partials(),
                blockers=BLOCKERS,
                marker_present=False,
                model_constructs=True,
                non_forward_gate_status={".deepseek-v4-import-ok": "present"},
                generated_at="2026-06-20T00:00:00+00:00",
                real_mode_proofs=stub_real_mode_proofs(),
                stateful_decode_seam_probe=detected_probe(),
            )
        self.assertEqual(report["schema"], 1)
        self.assertIs(report["full_forward_parity"], False)
        self.assertIs(report["marker_earned"], False)
        self.assertEqual(report["status"], "not-ready")
        self.assertEqual(report["blockers_count"], 4)
        self.assertEqual(report["coverage"]["fixtures_total"], 19)
        self.assertEqual(report["real_mode_proofs"]["proofs_total"], 8)
        self.assertIn("stateful_decode_readiness", report)
        self.assertGreater(list(report).index("stateful_decode_readiness"), list(report).index("real_mode_proofs"))
        self.assertLess(list(report).index("stateful_decode_readiness"), list(report).index("marker_write_criteria"))
        self.assertIs(report["stateful_decode_readiness"]["seam_available"], True)
        self.assertIs(report["stateful_decode_readiness"]["ready"], False)

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
        finetune_ds4._stateful_decode_readiness(absent_probe())
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
                stateful_decode_seam_probe=absent_probe(),
            )
        added = {k for k in sys.modules if _is_torch_or_mlx(k)} - before
        self.assertFalse(
            added,
            f"Stateful-decode readiness path imported torch/mlx (new sys.modules keys): {sorted(added)}",
        )

    def test_default_seam_probe_is_skipped_fail_closed(self):
        probe = finetune_ds4._default_stateful_decode_seam_probe()
        self.assertEqual(probe["probe_status"], "skipped")
        block = finetune_ds4._stateful_decode_readiness(probe)
        self.assertIs(block["seam_available"], False)
        self.assertEqual(block["decision"], "not-ready")
        self.assertIs(block["fail_closed"], True)
        self.assertIs(block["ready"], False)


if __name__ == "__main__":
    unittest.main()
