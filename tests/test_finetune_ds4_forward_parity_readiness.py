#!/usr/bin/env python3
import contextlib
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

from scripts import finetune_ds4


BLOCKERS = (
    "full attention parity with RoPE/cache/sinks/compressor/indexer",
    "full decoder-layer hyperconnection residual mixing and final hyperhead parity",
    "full MoE parity with packed FP4/I8 expert dequant and expert kernels",
    "full MLX shimmed-checkpoint load/forward and MLX generation smoke (Track B only; DS4-GGUF base generation is its own Track-A gate, .ds4-gguf-generate-ok, not a forward-parity blocker)",
)

EXPECTED_FIXTURE_NAMES = (
    "embedding-rmsnorm-head",
    "sliding-attention-no-compressor-no-rope",
    "rope-tail-pairwise",
    "sink-cache-inverse-rope-hca-bias",
    "hyperconnection-hc1-collapse",
    "hyperconnection-hc2-pre-post-comb",
    "hyperconnection-hc2-transformers",
    "csa-compressor-forward",
    "csa-topk-indexer-gather-mask",
    "hca-compressor-forward",
    "final-hyperhead-hc2-collapse",
    "topk-moe-unquantized",
    "hash-moe-tid2eid-unquantized",
    "topk-moe-i8-block-scale",
    "integrated-layer-attention-moe",
    "csa-indexer-scorer-topk",
    "hyperhead-transformers-reference",
    "integrated-layer-hc-mult",
    "csa-compressor-indexer-attention-mlx",
)
EXPECTED_REAL_MODE_PROOF_IDS = ("B0a-1", "B0a-2", "B0a-3", "B0b-a-1", "B0b-a-2", "B2-a-1", "B2-a-2", "B2-a-3", "B2-a-4")


def stub_partials(status="ok"):
    return [
        {
            "fixture": name,
            "status": status,
            "max_abs_error": 0.0,
            "covered": [f"covered:{name}"],
            "not_covered": [],
        }
        for name in EXPECTED_FIXTURE_NAMES
    ]


def stub_real_mode_proofs(status="ok"):
    proof_specs = (
        {
            "id": "B0a-1",
            "name": "real-mode-single-layer-hc1",
            "config": {"num_hidden_layers": 1, "hc_mult": 1, "compression_ratio": 0, "final_projection": False},
            "reference": "_integrated_layer_forward",
        },
        {
            "id": "B0a-2",
            "name": "real-mode-multilayer-hc1",
            "config": {"num_hidden_layers": 2, "hc_mult": 1, "compression_ratio": 0, "final_projection": True},
            "reference": "_integrated_multilayer_forward",
        },
        {
            "id": "B0a-3",
            "name": "real-mode-single-layer-hc2-hyperhead",
            "config": {"num_hidden_layers": 1, "hc_mult": 2, "compression_ratio": 0, "final_projection": False},
            "reference": "_integrated_layer_forward+tiny_hyperhead_collapse",
        },
        {
            "id": "B0b-a-1",
            "name": "real-mode-single-layer-csa-compressed",
            "config": {"num_hidden_layers": 1, "hc_mult": 1, "compression_ratio": 4, "final_projection": False, "seq_len": 8},
            "reference": "_integrated_layer_forward",
        },
        {
            "id": "B0b-a-2",
            "name": "csa-topk-sparse-selection-primitive",
            "granularity": "csa-attention-sublayer-primitive",
            "config": {"num_hidden_layers": 1, "hc_mult": 1, "compression_ratio": 4, "final_projection": False, "seq_len": 12, "index_topk": 1, "compressed_len": 3},
            "reference": "tiny_compressor_indexer_attention_reference(index_topk=1)",
        },
        {
            "id": "B2-a-1",
            "name": "real-mode-single-layer-i8-block-scale-dequant-moe",
            "evidence_class": "B2-partial",
            "config": {"num_hidden_layers": 1, "hc_mult": 1, "compression_ratio": 0, "hidden_size": 16, "moe_intermediate_size": 16, "n_routed_experts": 2, "num_experts_per_tok": 1, "n_shared_experts": 1, "expert_dtype": "i8", "block_size": 16, "scale_axis": 1, "seq_len": 2, "final_projection": False},
            "reference": "_integrated_layer_forward",
        },
        {
            "id": "B2-a-2",
            "name": "real-mode-multilayer-i8-block-scale-dequant-moe",
            "evidence_class": "B2-partial",
            "config": {"num_hidden_layers": 2, "hc_mult": 1, "compression_ratio": 0, "hidden_size": 16, "moe_intermediate_size": 16, "n_routed_experts": 2, "num_experts_per_tok": 1, "n_shared_experts": 1, "expert_dtype": "i8", "block_size": 16, "scale_axis": 1, "seq_len": 2, "final_projection": False, "secondary_num_hidden_layers": 3},
            "reference": "_integrated_multilayer_forward",
        },
        {
            "id": "B2-a-3",
            "name": "real-mode-topk-multi-expert-i8-block-scale-dequant-moe",
            "evidence_class": "B2-partial",
            "config": {"num_hidden_layers": 1, "hc_mult": 1, "compression_ratio": 0, "hidden_size": 16, "moe_intermediate_size": 16, "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1, "expert_dtype": "i8", "block_size": 16, "scale_axis": 1, "seq_len": 2, "final_projection": False, "secondary_num_experts_per_tok": 3},
            "reference": "_integrated_layer_forward",
        },
        {
            "id": "B2-a-4",
            "name": "metal-routed-i8-e8m0-dequant-isolation-proof",
            "evidence_class": "B2-partial-metal",
            "config": {},
            "reference": "OCP MX v1.0 spec (scale=2^(e-127), e=255->NaN block, e=0->2^(-127) subnormal, NO 2^(-6) factor)",
            "covered": [
                "compares Metal output against OCP-spec witness at max_abs <= 1e-5",
                "verifies NO forbidden OUR-Python symbols imported (grep audit)",
                "verified on real HF checkpoint bytes (snapshot 553034d), NOT synthetic",
                "production decode function ds4_e8m0_decode_i8 (inline device fn) shared between proof kernel and production fused kernels: no proof/production divergence",
            ],
            "not_covered": [
                "packed FP4 expert dequant (dequantize_expert_packed('fp4') still raises; B2 remainder)",
                "expert parallel kernels (production path _moe_mlx handles matmul but FP4 remains fail-closed)",
                "hc_mult>1 multi-layer stacking (B1)",
                "shimmed-checkpoint load/forward + MLX generation smoke (B3)",
                "production fused kernel matmul+SwiGLU integration (the standalone dump kernel is proof-only)",
                "real-payload dequant inside full _real_forward MoE branch (the isolation proof uses standalone dump kernel)",
            ],
        },
    )
    proofs = []
    for proof_spec in proof_specs:
        proof = {
            "id": proof_spec["id"],
            "name": proof_spec["name"],
            "status": status,
            "max_abs_error": 0.0 if status == "ok" else (None if status == "skipped" else 0.25),
            "config": dict(proof_spec["config"]),
            "reference": proof_spec["reference"],
            "covered": [f"covered:{proof_spec['id']}"],
            "not_covered": ["full B0/B1/B2/B3 parity"],
        }
        if "granularity" in proof_spec:
            proof["granularity"] = proof_spec["granularity"]
            proof["pruning_non_degenerate"] = status == "ok"
        if "evidence_class" in proof_spec:
            proof["evidence_class"] = proof_spec["evidence_class"]
            proof["reference_max_abs_error"] = 0.0 if status == "ok" else None
            proof["reference_tolerance"] = 1e-3
            proof["quantization_gap"] = 0.01 if status == "ok" else None
            proof["non_degenerate"] = status == "ok"
            proof["block_scale_nonunit"] = status == "ok"
            proof["i8_branch_reached"] = status == "ok"
            if proof_spec["id"] == "B2-a-2":
                proof["secondary_multilayer"] = {
                    "num_hidden_layers": 3,
                    "max_abs_error": 0.0 if status == "ok" else None,
                    "reference_max_abs_error": 0.0 if status == "ok" else None,
                    "ok": status == "ok",
                }
            if proof_spec["id"] == "B2-a-3":
                proof["multi_expert_combine_exercised"] = status == "ok"
                proof["num_experts_per_tok"] = 2
                proof["n_routed_experts"] = 4
                proof["tie_free_margin"] = 0.156 if status == "ok" else None
                proof["routing_subset_stable"] = status == "ok"
                proof["secondary_topk"] = {
                    "num_experts_per_tok": 3,
                    "max_abs_error": 0.0 if status == "ok" else None,
                    "reference_max_abs_error": 0.0 if status == "ok" else None,
                    "ok": status == "ok",
                }
        proofs.append(proof)
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


def _shape_of(value):
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        return (len(value),) + _shape_of(value[0])
    return ()


class ForwardParityReadinessTests(unittest.TestCase):
    def quiet(self, func, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return func(*args)

    def readiness_args(self, mlx, out=None):
        return type(
            "Args",
            (),
            {
                "mlx_work": str(mlx),
                "out": str(out) if out is not None else None,
            },
        )()

    def command_args(self, mlx):
        return type(
            "Args",
            (),
            {
                "hf_model": str(mlx / "hf"),
                "dataset_root": str(mlx / "dataset"),
                "mlx_work": str(mlx),
                "ds4_root": str(mlx / "ds4"),
                "ds4_gguf": None,
                "adapter_ds4": None,
                "split_dir": "mlx-4096",
                "fused_hf_model": None,
                "ds4_imatrix": None,
                "backend": "local-mlx",
                "step": "deepseek-v4-forward-parity-readiness",
                "execute": True,
                "yes": True,
            },
        )()

    def run_stubbed_readiness(self, mlx, out=None, partials=None, blockers=BLOCKERS, real_mode_proofs=None):
        partials = stub_partials() if partials is None else partials
        real_mode_proofs = stub_real_mode_proofs() if real_mode_proofs is None else real_mode_proofs
        args = self.readiness_args(mlx, out=out)
        with mock.patch.object(finetune_ds4, "_run_forward_parity_fixtures", return_value=partials), \
             mock.patch.object(finetune_ds4, "_run_real_mode_forward_proofs", return_value=real_mode_proofs), \
             mock.patch.object(finetune_ds4, "forward_parity_blockers", return_value=tuple(blockers)), \
             mock.patch.object(finetune_ds4, "_try_construct_tiny_forward_parity_model", return_value=True), \
             mock.patch.object(finetune_ds4, "_collect_non_forward_gate_status", return_value={
                 ".deepseek-v4-import-ok": "present",
                 ".deepseek-v4-tiny-config-ok": "present",
                 ".deepseek-v4-mapping-ok": "present",
                 ".deepseek-v4-mtp-exclusion-ok": "not-required",
                 ".deepseek-v4-dequant-parity-ok": "present",
             }), \
             mock.patch.object(finetune_ds4, "_probe_b2_real_checkpoint_payload", return_value=finetune_ds4._default_b2_real_checkpoint_payload_probe()), \
             mock.patch.object(finetune_ds4, "_probe_b2_routed_dequant_trusted_reference", return_value=finetune_ds4._default_b2_routed_dequant_trusted_reference_probe()), \
             mock.patch.object(finetune_ds4, "_write_gate_marker", side_effect=AssertionError("readiness must not write gate markers")):
            self.assertEqual(self.quiet(finetune_ds4.deepseek_v4_forward_parity_readiness, args), 0)
        out_path = pathlib.Path(out) if out is not None else mlx / "deepseek-v4-forward-parity-readiness.json"
        return json.loads(out_path.read_text(encoding="utf-8"))

    def test_schema_and_honesty_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            mlx = pathlib.Path(td) / "mlx"
            report = finetune_ds4.build_forward_parity_readiness_report(
                mlx_work=mlx,
                partials=stub_partials(),
                blockers=BLOCKERS,
                marker_present=False,
                model_constructs=True,
                non_forward_gate_status={".deepseek-v4-import-ok": "present"},
                generated_at="2026-06-19T00:00:00+00:00",
            )

        self.assertEqual(report["schema"], 1)
        self.assertEqual(report["report"], "deepseek-v4-forward-parity-readiness")
        self.assertEqual(report["status"], "not-ready")
        self.assertEqual(report["gate"], "deepseek-v4-forward-parity")
        self.assertIs(report["full_forward_parity"], False)
        self.assertIs(report["marker_earned"], False)
        self.assertIs(report["marker_present"], False)
        self.assertEqual(report["blockers"], list(BLOCKERS))
        self.assertEqual(report["blockers_count"], 4)
        self.assertIs(report["model_constructs_under_mlx_venv"], True)
        self.assertEqual(report["coverage"]["fixtures_total"], len(EXPECTED_FIXTURE_NAMES))
        self.assertEqual(report["coverage"]["fixtures_ok"], len(EXPECTED_FIXTURE_NAMES))
        self.assertEqual(report["coverage"]["fixtures_skipped"], 0)
        self.assertEqual(report["coverage"]["fixtures_failed"], 0)
        self.assertEqual([item["id"] for item in report["marker_write_criteria"]], ["B0", "B1", "B2", "B3", "GATE"])
        self.assertTrue(report["next_steps"][0].startswith("deepseek-v4-forward-parity-check"))
        self.assertEqual(report["next_steps"][-2:], ["convert-shimmed", "smoke-train"])
        self.assertNotIn("full-train", "\n".join(report["next_steps"]))
        self.assertIn("no marker written (.deepseek-v4-forward-parity-ok stays absent)", report["non_claims"])
        self.assertEqual(report["real_mode_proofs"]["category"], "B0-partial")
        self.assertEqual(report["real_mode_proofs"]["proofs_total"], len(EXPECTED_REAL_MODE_PROOF_IDS))
        self.assertEqual(report["real_mode_proofs"]["proofs_ok"], 0)
        self.assertEqual(report["real_mode_proofs"]["proofs_skipped"], len(EXPECTED_REAL_MODE_PROOF_IDS))
        self.assertEqual(report["real_mode_proofs"]["proofs_failed"], 0)
        self.assertEqual(report["real_mode_proofs"]["proofs"], [])
        b2_payload = report["b2_real_checkpoint_payload_readiness"]
        self.assertEqual(b2_payload["status"], "fail-closed")
        self.assertEqual(b2_payload["decision"], "not-ready")
        self.assertIs(b2_payload["fail_closed"], True)
        self.assertIs(b2_payload["ready"], False)
        self.assertIs(b2_payload["proof_available"], False)
        b2_trusted = report["b2_routed_dequant_trusted_reference_readiness"]
        self.assertEqual(b2_trusted["status"], "fail-closed")
        self.assertEqual(b2_trusted["decision"], "not-ready")
        self.assertIs(b2_trusted["fail_closed"], True)
        self.assertIs(b2_trusted["ready"], False)
        self.assertIs(b2_trusted["proof_available"], True)

    def test_real_mode_proofs_present_and_fail_closed(self):
        report = finetune_ds4.build_forward_parity_readiness_report(
            mlx_work=pathlib.Path("/tmp/mlx"),
            partials=stub_partials(),
            blockers=BLOCKERS,
            marker_present=False,
            model_constructs=True,
            non_forward_gate_status={},
            generated_at="2026-06-19T00:00:00+00:00",
            real_mode_proofs=stub_real_mode_proofs(),
        )

        self.assertIs(report["full_forward_parity"], False)
        self.assertIs(report["marker_earned"], False)
        self.assertEqual(report["status"], "not-ready")
        self.assertEqual(report["blockers_count"], 4)
        self.assertEqual(report["coverage"]["fixtures_total"], len(EXPECTED_FIXTURE_NAMES))
        self.assertEqual(report["real_mode_proofs"]["proofs_total"], len(EXPECTED_REAL_MODE_PROOF_IDS))
        self.assertEqual(report["real_mode_proofs"]["proofs_ok"], len(EXPECTED_REAL_MODE_PROOF_IDS))
        self.assertEqual([item["id"] for item in report["real_mode_proofs"]["proofs"]], list(EXPECTED_REAL_MODE_PROOF_IDS))
        self.assertFalse(any(item.get("fixture") in EXPECTED_REAL_MODE_PROOF_IDS for item in report["coverage"]["partials"]))
        self.assertIn("b2_real_checkpoint_payload_readiness", report)
        self.assertIs(report["b2_real_checkpoint_payload_readiness"]["proof_available"], False)
        self.assertIn("b2_routed_dequant_trusted_reference_readiness", report)
        self.assertIs(report["b2_routed_dequant_trusted_reference_readiness"]["proof_available"], True)
        self.assertFalse(any("real-checkpoint-payload" in item.get("id", "") for item in report["real_mode_proofs"]["proofs"]))
        self.assertFalse(any("routed-dequant-trusted-reference" in item.get("id", "") for item in report["real_mode_proofs"]["proofs"]))
        b0 = report["marker_write_criteria"][0]
        self.assertEqual(b0["id"], "B0")
        self.assertNotIn("satisfied", b0)
        self.assertIn("B0 NOT marked satisfied", report["real_mode_proofs"]["b0_partial_progress"])

    def test_real_mode_proofs_failed_or_skipped_still_not_ready(self):
        total = len(EXPECTED_REAL_MODE_PROOF_IDS)
        for status, expected_failed, expected_skipped in (("failed", total, 0), ("skipped", 0, total)):
            with self.subTest(status=status):
                report = finetune_ds4.build_forward_parity_readiness_report(
                    mlx_work=pathlib.Path("/tmp/mlx"),
                    partials=stub_partials(),
                    blockers=BLOCKERS,
                    marker_present=True,
                    model_constructs=True,
                    non_forward_gate_status={},
                    generated_at="2026-06-19T00:00:00+00:00",
                    real_mode_proofs=stub_real_mode_proofs(status=status),
                )
                self.assertIs(report["full_forward_parity"], False)
                self.assertIs(report["marker_earned"], False)
                self.assertEqual(report["blockers_count"], 4)
                self.assertEqual(report["real_mode_proofs"]["proofs_failed"], expected_failed)
                self.assertEqual(report["real_mode_proofs"]["proofs_skipped"], expected_skipped)
                self.assertEqual(report["coverage"]["fixtures_total"], len(EXPECTED_FIXTURE_NAMES))

    def test_command_surface_catalog_and_execute_guard(self):
        with tempfile.TemporaryDirectory() as td:
            mlx = pathlib.Path(td) / "mlx"
            self.assertIn("deepseek-v4-forward-parity-readiness", finetune_ds4.MLX_STEPS)
            self.assertLess(
                finetune_ds4.MLX_STEPS.index("model-4bit-conversion-plan"),
                finetune_ds4.MLX_STEPS.index("deepseek-v4-forward-parity-readiness"),
            )
            args = self.command_args(mlx)
            command = finetune_ds4.command_catalog(args)["deepseek-v4-forward-parity-readiness"][0]
            self.assertIn(".venv/bin/activate", command)
            self.assertIn("deepseek-v4-forward-parity-readiness", command)
            parsed = finetune_ds4.build_parser().parse_args(["deepseek-v4-forward-parity-readiness", "--mlx-work", str(mlx), "--out", str(mlx / "out.json")])
            self.assertIs(parsed.func, finetune_ds4.deepseek_v4_forward_parity_readiness)
            with self.assertRaisesRegex(finetune_ds4.PlanError, "top-level read-only Python subcommand"):
                finetune_ds4.run_command(args)

    def test_wrapper_writes_only_readiness_json_and_no_protected_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            mlx = pathlib.Path(td) / "mlx"
            mlx.mkdir()
            report = self.run_stubbed_readiness(mlx)
            report_path = mlx / "deepseek-v4-forward-parity-readiness.json"
            self.assertTrue(report_path.is_file())
            self.assertIs(report["full_forward_parity"], False)
            self.assertIs(report["marker_earned"], False)
            self.assertIn("b2_real_checkpoint_payload_readiness", report)
            self.assertIs(report["b2_real_checkpoint_payload_readiness"]["fail_closed"], True)
            self.assertIn("b2_routed_dequant_trusted_reference_readiness", report)
            self.assertIs(report["b2_routed_dequant_trusted_reference_readiness"]["fail_closed"], True)
            self.assertFalse((mlx / ".deepseek-v4-forward-parity-ok").exists())
            self.assertFalse((mlx / "model-4bit").exists())

    def test_protected_out_paths_are_rejected_before_fixture_run(self):
        with tempfile.TemporaryDirectory() as td:
            mlx = pathlib.Path(td) / "mlx"
            mlx.mkdir()
            protected_paths = [
                mlx / ".deepseek-v4-forward-parity-ok",
                mlx / ".deepseek-v4-anything-ok",
                mlx / ".anything-ok",
                mlx / "model-4bit" / "readiness.json",
                pathlib.Path(td) / ".deepseek-v4-external-ok",
            ]
            for out_path in protected_paths:
                with self.subTest(out_path=out_path):
                    args = self.readiness_args(mlx, out=out_path)
                    with mock.patch.object(finetune_ds4, "_run_forward_parity_fixtures", side_effect=AssertionError("fixtures must not run")):
                        with self.assertRaisesRegex(finetune_ds4.PlanError, "protected"):
                            finetune_ds4.deepseek_v4_forward_parity_readiness(args)
                    self.assertFalse(out_path.exists())
                    self.assertFalse((mlx / "deepseek-v4-forward-parity-readiness.json").exists())
                    self.assertFalse((mlx / ".deepseek-v4-forward-parity-ok").exists())
                    self.assertFalse((mlx / "model-4bit").exists())

    def test_readiness_json_is_not_a_gate_and_convert_shimmed_forward_parity_retired(self):
        with tempfile.TemporaryDirectory() as td:
            mlx = pathlib.Path(td) / "mlx"
            mlx.mkdir()
            report = self.run_stubbed_readiness(mlx)
            report_path = mlx / "deepseek-v4-forward-parity-readiness.json"
            with self.assertRaisesRegex(finetune_ds4.PlanError, "missing DeepSeek V4 architecture gate marker"):
                finetune_ds4._load_gate_marker(mlx, ".deepseek-v4-forward-parity-ok", "deepseek-v4-forward-parity")
            with self.assertRaisesRegex(finetune_ds4.PlanError, "invalid DeepSeek V4 gate marker"):
                finetune_ds4._load_gate_marker(mlx, "deepseek-v4-forward-parity-readiness.json", "deepseek-v4-forward-parity")

            (mlx / ".deepseek-v4-forward-parity-ok").write_text(
                json.dumps({
                    "schema": 1,
                    "gate": "deepseek-v4-forward-parity",
                    "status": "ok",
                    "report_path": str(report_path),
                    "report_sha256": finetune_ds4.sha256_file(report_path),
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(finetune_ds4.PlanError, "expected full_forward_parity=true"):
                finetune_ds4._validate_forward_parity_marker(mlx)
            (mlx / ".deepseek-v4-forward-parity-ok").unlink()

            shim = mlx / "hf-f8shim"
            shim.mkdir()
            index_path = shim / "model.safetensors.index.json"
            index_path.write_text(json.dumps({"weight_map": {"embed.weight": "s.safetensors"}}), encoding="utf-8")
            for marker, payload in {
                ".deepseek-v4-import-ok": {"schema": 1, "gate": "deepseek-v4-import", "status": "ok"},
                ".deepseek-v4-tiny-config-ok": {"schema": 1, "gate": "deepseek-v4-tiny-config", "status": "ok"},
                ".deepseek-v4-mapping-ok": {"schema": 1, "gate": "deepseek-v4-mapping", "status": "ok", "index_sha256": finetune_ds4.sha256_file(index_path)},
                ".deepseek-v4-dequant-parity-ok": {"schema": 1, "gate": "deepseek-v4-dequant-parity", "status": "ok"},
            }.items():
                (mlx / marker).write_text(json.dumps(payload), encoding="utf-8")
            # ADR 0022 §Decision 3: architecture-gate no longer raises on absent
            # forward-parity marker (retired); live convert-shimmed --execute
            # proceeds. Readiness JSON is still NOT a gate marker; marker
            # validator still rejects (asserted above). Honesty unchanged.
            finetune_ds4.validate_deepseek_v4_architecture_gates(mlx)
            self.assertIs(report["full_forward_parity"], False)

    def test_failed_fixture_and_empty_blockers_still_do_not_earn_marker(self):
        partials = stub_partials()
        partials[0] = {**partials[0], "status": "failed", "not_covered": ["synthetic failure"]}
        with tempfile.TemporaryDirectory() as td:
            report = finetune_ds4.build_forward_parity_readiness_report(
                mlx_work=pathlib.Path(td) / "mlx",
                partials=partials,
                blockers=(),
                marker_present=True,
                model_constructs=False,
                non_forward_gate_status={},
                generated_at="2026-06-19T00:00:00+00:00",
            )
        self.assertIs(report["full_forward_parity"], False)
        self.assertIs(report["marker_earned"], False)
        self.assertEqual(report["blockers"], list(BLOCKERS))
        self.assertEqual(report["blockers_count"], 4)
        self.assertGreaterEqual(report["coverage"]["fixtures_failed"], 1)
        self.assertIn("GATE", [item["id"] for item in report["marker_write_criteria"]])

    def test_fixture_name_drift_guard(self):
        self.assertEqual(finetune_ds4.FORWARD_PARITY_FIXTURE_NAMES, EXPECTED_FIXTURE_NAMES)
        self.assertEqual(len(finetune_ds4.FORWARD_PARITY_FIXTURE_NAMES), 19)

    def test_real_mode_proof_spec_drift_guard(self):
        specs = finetune_ds4._real_mode_forward_proof_specs()
        self.assertEqual([spec["id"] for spec in specs], list(EXPECTED_REAL_MODE_PROOF_IDS))
        self.assertEqual([spec["name"] for spec in specs], [
            "real-mode-single-layer-hc1",
            "real-mode-multilayer-hc1",
            "real-mode-single-layer-hc2-hyperhead",
            "real-mode-single-layer-csa-compressed",
            "csa-topk-sparse-selection-primitive",
            "real-mode-single-layer-i8-block-scale-dequant-moe",
            "real-mode-multilayer-i8-block-scale-dequant-moe",
            "real-mode-topk-multi-expert-i8-block-scale-dequant-moe",
            "metal-routed-i8-e8m0-dequant-isolation-proof",
        ])
        self.assertEqual([spec["config"] for spec in specs], [
            {"num_hidden_layers": 1, "hc_mult": 1, "compression_ratio": 0, "final_projection": False},
            {"num_hidden_layers": 2, "hc_mult": 1, "compression_ratio": 0, "final_projection": True},
            {"num_hidden_layers": 1, "hc_mult": 2, "compression_ratio": 0, "final_projection": False},
            {"num_hidden_layers": 1, "hc_mult": 1, "compression_ratio": 4, "final_projection": False, "seq_len": 8},
            {"num_hidden_layers": 1, "hc_mult": 1, "compression_ratio": 4, "final_projection": False, "seq_len": 12, "index_topk": 1, "compressed_len": 3},
            {"num_hidden_layers": 1, "hc_mult": 1, "compression_ratio": 0, "hidden_size": 16, "moe_intermediate_size": 16, "n_routed_experts": 2, "num_experts_per_tok": 1, "n_shared_experts": 1, "expert_dtype": "i8", "block_size": 16, "scale_axis": 1, "seq_len": 2, "final_projection": False},
            {"num_hidden_layers": 2, "hc_mult": 1, "compression_ratio": 0, "hidden_size": 16, "moe_intermediate_size": 16, "n_routed_experts": 2, "num_experts_per_tok": 1, "n_shared_experts": 1, "expert_dtype": "i8", "block_size": 16, "scale_axis": 1, "seq_len": 2, "final_projection": False, "secondary_num_hidden_layers": 3},
            {"num_hidden_layers": 1, "hc_mult": 1, "compression_ratio": 0, "hidden_size": 16, "moe_intermediate_size": 16, "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1, "expert_dtype": "i8", "block_size": 16, "scale_axis": 1, "seq_len": 2, "final_projection": False, "o_groups": 1, "secondary_num_experts_per_tok": 3},
            {"kernel": "metal/moe.metal", "kernel_function": "kernel_dsv4_routed_dequant_i8_e8m0_to_bf16", "weight_dtype": "I8", "scale_dtype": "F8_E8M0", "block_size": 16, "scale_axis": 1, "dedicated": True, "fusion": "production path fuses dequant+gemm in kernel Mul_mm_id_i8_e8m0_f32 and kernel_mul_mv_id_i8_e8m0_pair_swiglu_f32; this kernel is the standalone dequant dump for the isolation proof"},
        ])
        self.assertEqual([spec["reference"] for spec in specs], [
            "_integrated_layer_forward",
            "_integrated_multilayer_forward",
            "_integrated_layer_forward+tiny_hyperhead_collapse",
            "_integrated_layer_forward",
            "tiny_compressor_indexer_attention_reference(index_topk=1)",
            "_integrated_layer_forward",
            "_integrated_multilayer_forward",
            "_integrated_layer_forward",
            "OCP MX v1.0 spec (scale=2^(e-127), e=255->NaN block, e=0->2^(-127) subnormal, NO 2^(-6) factor)",
        ])
        self.assertFalse(any("seq_len" in spec["config"] for spec in specs[:3]))
        b0b = next(spec for spec in specs if spec["id"] == "B0b-a-1")
        self.assertIn("_csa_compressor_mlx", "\n".join(b0b["covered"]))
        self.assertIn("_csa_indexer_mlx", "\n".join(b0b["covered"]))
        self.assertIn("real cache / sliding-window stateful CSA", "\n".join(b0b["not_covered"]))
        topk = next(spec for spec in specs if spec["id"] == "B0b-a-2")
        self.assertEqual(topk["granularity"], "csa-attention-sublayer-primitive")
        self.assertIn("_csa_attention_mlx(index_topk<compressed_len)", "\n".join(topk["covered"]))
        for i8 in specs[-3:-1]:
            self.assertEqual(i8["evidence_class"], "B2-partial")
            self.assertEqual(i8["config"]["expert_dtype"], "i8")
            self.assertEqual(i8["config"]["hidden_size"], 16)

    def test_real_mode_csa_weight_shape_contract(self):
        weights = finetune_ds4._make_real_mode_csa_weights()
        csa_shapes = {
            "compressor_wkv": (4, 8),
            "compressor_wgate": (4, 8),
            "compressor_ape": (8, 4),
            "compressor_norm": (4,),
            "indexer_wq_b": (4, 4),
            "indexer_proj": (4, 2),
            "indexer_compressor_wkv": (4, 4),
            "indexer_compressor_wgate": (4, 4),
            "indexer_compressor_ape": (4, 4),
            "indexer_compressor_norm": (2,),
        }
        self.assertEqual(set(csa_shapes).issubset(weights), True)
        for key, shape in csa_shapes.items():
            self.assertEqual(_shape_of(weights[key]), shape, key)
        self.assertEqual(weights["compressor_wkv"][0][:4], [0.5, -1.0, 1.5, -2.0])
        self.assertEqual(_shape_of(weights["embed.weight"]), (8, 4))
        self.assertFalse({
            "q_a_proj.weight",
            "q_norm.weight",
            "q_b_proj.weight",
            "kv_proj.weight",
            "kv_norm.weight",
            "o_a_proj.weight",
            "o_b_proj.weight",
            "sinks",
        } & set(weights))

    def test_run_one_real_mode_forward_proof_uses_b0b_seq_len_and_csa_weights(self):
        calls = {}
        fake_args = types.SimpleNamespace(hc_mult=1, compression_ratio=4)
        fake_weights = {"sentinel": [[1.0]]}

        class FakeModel:
            def __init__(self, args):
                calls["model_args"] = args

            def load_weights(self, weights):
                calls["load_weights"] = weights

            def __call__(self, input_ids):
                calls["model_input_ids"] = input_ids
                return [[[float(token)] for token in input_ids[0]]]

        def fake_integrated_layer_forward(args, input_ids, weights):
            calls["reference"] = (args, input_ids, weights)
            return [[[float(token)] for token in input_ids[0]]]

        vendor_module = types.ModuleType("deepseek_v4")
        vendor_module.Model = FakeModel
        vendor_module._integrated_layer_forward = fake_integrated_layer_forward
        vendor_module._integrated_multilayer_forward = lambda *args, **kwargs: self.fail("unexpected multilayer reference")
        attention_module = types.ModuleType("deepseek_v4_attention_spec")
        attention_module.tiny_hyperhead_collapse = lambda *args, **kwargs: self.fail("unexpected hyperhead reference")
        module_patches = {
            "ds4_ft_mlx": types.ModuleType("ds4_ft_mlx"),
            "ds4_ft_mlx.deepseek_v4_attention_spec": attention_module,
            "ds4_ft_mlx.vendor": types.ModuleType("vendor"),
            "ds4_ft_mlx.vendor.mlx_lm_models": types.ModuleType("mlx_lm_models"),
            "ds4_ft_mlx.vendor.mlx_lm_models.deepseek_v4": vendor_module,
        }
        spec = next(spec for spec in finetune_ds4._real_mode_forward_proof_specs() if spec["id"] == "B0b-a-1")
        with mock.patch.dict(sys.modules, module_patches), \
             mock.patch.object(finetune_ds4, "_ensure_mlx_project_src_on_path"), \
             mock.patch.object(finetune_ds4, "_build_real_mode_tiny_args", return_value=fake_args) as build_args, \
             mock.patch.object(finetune_ds4, "_make_real_mode_csa_weights", return_value=fake_weights) as make_weights:
            result = finetune_ds4._run_one_real_mode_forward_proof(spec)
        self.assertEqual(result["status"], "ok")
        build_args.assert_called_once_with(1, 1, compression_ratio=4, index_n_heads=2, index_head_dim=2)
        make_weights.assert_called_once_with()
        self.assertIs(calls["model_args"], fake_args)
        self.assertIs(calls["load_weights"], fake_weights)
        self.assertEqual(calls["model_input_ids"], [[0, 1, 2, 3, 4, 5, 6, 7]])
        self.assertEqual(calls["reference"], (fake_args, [[0, 1, 2, 3, 4, 5, 6, 7]], fake_weights))

    def test_real_mode_not_covered_delta_removes_compressed_forward_but_keeps_b0_open(self):
        self.assertNotIn("real-mode compressed attention (compression_ratio != 0) forward", finetune_ds4.REAL_MODE_PROOFS_NOT_COVERED)
        self.assertIn("HCA compressor + non-tiny CSA configs wired into real-mode forward", finetune_ds4.REAL_MODE_PROOFS_NOT_COVERED)
        report = finetune_ds4._real_mode_proofs_report(stub_real_mode_proofs()["proofs"])
        self.assertEqual(report["proofs_total"], len(EXPECTED_REAL_MODE_PROOF_IDS))
        self.assertIn("B0 NOT", report["b0_partial_progress"])
        self.assertNotIn("compression_ratio != 0", "\n".join(report["not_covered"]))

    def test_no_mlx_or_torch_imports_required_for_unit_tests(self):
        before = set(sys.modules)
        finetune_ds4.build_forward_parity_readiness_report(
            mlx_work=pathlib.Path("/tmp/mlx"),
            partials=stub_partials(),
            blockers=BLOCKERS,
            marker_present=False,
            model_constructs=True,
            non_forward_gate_status={},
            generated_at="2026-06-19T00:00:00+00:00",
        )
        added = set(sys.modules) - before
        self.assertNotIn("mlx", added)
        self.assertNotIn("torch", added)


if __name__ == "__main__":
    unittest.main()
