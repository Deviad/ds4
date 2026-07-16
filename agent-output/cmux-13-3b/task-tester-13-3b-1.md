# Story 13.3b-1 — Test Manager Task Brief (independent validation)

## Role
Test Manager (neuralwatt/qwen3.6-35b · surface:85, fresh-context via /new). Independent validation of Coder 13.3b-1 (`1f5d8e4`).

## Context
- Coder 13.3b-1 complete: HCA real-dim MLX port + RoPE oracle. Commit `1f5d8e4` (parent `702199f`).
- Coder claims: AC0-AC3 GREEN, parity deltas tight (cos 7.2e-7, rotated 7.8e-3, compressed_kv 3.9e-6, output 6.3e-7), full suite 1 RED (pre-existing Test #8) / 561 PASS / 13 SKIP, FROZEN bodies byte-intact via AST, sha-pin cascade advanced for the `deepseek_v4.py` additive baseline change.
- Watch-item: Coder updated stale-hash sentinels as part of the sha-pin cascade SLICE-INVARIANT (per AGENTS.md, any edit to `deepseek_v4.py` MUST advance 3 sha16 pin sites). Test Manager MUST independently verify the cascade count + correctness.

## Your job — independent re-run + verification
1. `source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first).
2. `git log --oneline -2` — confirm `1f5d8e4` is HEAD.
3. Run full suite: `python3 -m pytest tests/ --no-header -q | tail -5`. Confirm exactly 1 RED (Test #8 `test_finetune_readiness_report_records_b0b_a_without_unblocking_b0`) / 561 PASS / 13 SKIP / 0 introduced RED.
4. Run the 4 NEW tests in isolation: `python3 -m pytest tests/test_compress_rope_oracle.py tests/test_hca_compressor_mlx_parity.py tests/test_attention_real_mlx_hca_parity.py tests/test_13_3b_1_dispatch_additive.py -v | tail -20`. Confirm all GREEN + note the parity assertion values.
5. Regression check: `python3 -m pytest tests/test_deepseek_v4_dequant_parity.py tests/test_deepseek_v4_mlx_port.py tests/test_deepseek_v4_nn_construct.py tests/test_deepseek_v4_nn_quantize.py tests/test_deepseek_v4_nn_lora.py tests/test_deepseek_v4_nn_forward.py tests/test_deepseek_v4_nn_moe_construct.py tests/test_deepseek_v4_nn_moe_quantize.py tests/test_deepseek_v4_nn_moe_forward.py tests/test_deepseek_v4_nn_hash_forward.py tests/test_deepseek_v4_nn_fp4_parity.py tests/test_deepseek_v4_nn_backward.py tests/test_deepseek_v4_nn_model_type_wiring.py tests/test_deepseek_v4_nn_sanitize.py -q | tail -5`. Confirm 13.2 FP4 (9) + 13.3a nn (12) all GREEN.
6. Tiny CSA fixtures (11.14/11.15) GREEN — confirm via grep for the test files then run.
7. FROZEN byte-intact verification — independent check:
   - `git show 1f5d8e4 -- python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py | head -200` — inspect the DIFF. Confirm ONLY additions (new module section + ONE dispatch branch) — NO body edits to `_csa_config_error`, `_require_csa_config`, `_csa_attention_mlx`, `_csa_compressor_mlx`, `_csa_indexer_mlx`, `_csa_windowed_compressor_mlx`, `_hyperconnection_mlx`, `_hyperhead_mlx`, `sanitize_weights`, `_apply_rope_full_mlx`, `_rope_tail_tables_mlx`, `_apply_rope_tail_mlx`, `_attention_mlx` cr=0 body, parity `Model`, 9 ADR-0017 forbidden symbols.
   - Sha-pin cascade verification: count how many sha16 pin sites reference `deepseek_v4.py` and confirm they ADVANCED to a new value (post-edit hash). The Coder's notes claim `sha16=7c9960107ef0919d` for `_attention_mlx` cr=0 body. Verify the pin sites in the test files (likely `test_finetune_ds4*.py` or `test_deepseek_v4_mlx_port.py`) reflect the NEW frozen baseline. Confirm exactly the right count of pins advanced (Coder claimed "3 sites" per SLICE-INVARIANT — verify count).
8. Anti-circularity spot-check (AC0 oracle): confirm the torch reference in `test_compress_rope_oracle.py` computes expected values from the SAME synthesized weights/config as the MLX path — NO second MLX primitive as oracle (ADR 0007 §4).
9. `git diff --check` clean (whitespace).

## PASS criteria
- 4 new AC tests GREEN with parity within claimed tolerances.
- Full suite: 1 RED (Test #8) / 561 PASS / 13 SKIP / 0 introduced RED.
- FROZEN bodies byte-intact (diff shows additions only).
- Sha-pin cascade advanced exactly the right number of sites to the correct new value.
- Anti-circularity honored (AC0 torch reference from same synthesized weights).
- Regression 13.2 + 13.3a + tiny CSA all GREEN.

## STOP-ESCALATE
If anything RED that shouldn't be (beyond Test #8), or FROZEN body edited (not additive), or sha-pin cascade broken — write `agent-output/cmux-13-3b/test-manager-13-3b-1-stop.md` + `{"status":"error",...}`.

## Deliverables
1. `agent-output/cmux-13-3b/test-report-13-3b-1.md` — run results, parity values, FROZEN byte-intact verdict, sha-pin cascade verdict, anti-circularity verdict, PASS/FAIL.
2. `.cmux-status/test-manager.done` (`{"status":"ok","role":"Test Manager"}`).
3. In-pane JSON: `{"status":"ok","role":"Test Manager"}` in surface:85 ONLY.

## Style
Caveman ultra default; byte-exact exempt (paths/SHAs/line numbers/test counts/parity deltas verbatim). Use `ctx_execute` for any large output (full suite tail, diffs). BEGIN NOW.
