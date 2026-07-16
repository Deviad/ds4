# Story 13.3b-2b — Test Manager Task Brief (independent validation)

## Role
Test Manager (neuralwatt/qwen3.6-35b · surface:85, fresh-context via /new). Independent validation of Coder 13.3b-2b (`aa93c5d`).

## Context
- Coder 13.3b-2b complete: CSA attention wiring. Commit `aa93c5d` (parent `a2c20b0`).
- Coder claims: AC1-AC2 GREEN; 3 new helpers `_csa_compressor_real_mlx:819` + `_csa_block_bias_mlx` + `_csa_attention_real_mlx:907`; chose NEW `_csa_attention_real_mlx` (NOT editing HCA `_attention_real_mlx` — keeps HCA byte-identical); rope option α (APE token-major no-transpose + compress-YaRN tail via reused 13.3b-1 helpers); parity: compressor 5.96e-6, attention 6.48e-7, block_bias -inf mask EXACT + finite 0.0; full suite 1 RED (Test #8) / 565 PASS / 13 SKIP; FROZEN bodies byte-intact via AST.
- Watch-items: (1) HCA `_attention_real_mlx` (now at `:1005`) byte-identical (Coder chose NEW helper, so HCA branch MUST be untouched); (2) rope option α + ape no-transpose (BA §0.2 — confirm `_csa_compressor_real_mlx` consumes ape token-major `(rate=4, 2*head_dim=1024)` NO transpose + applies compress-yarn-tail via REUSED 13.3b-1 helpers); (3) block_bias -inf mask EXACT (Coder claims finite 0.0 + -inf equal True); (4) sha-pin cascade (deepseek_v4.py edited additively → advance right count) — use rigorous whole-tree scan, NOT grep-subset (Test Manager 13.3b-1/2a under-counted).

## Your job — independent re-run + verification
1. `source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first).
2. `git log --oneline -2` — confirm `aa93c5d` HEAD.
3. Full suite: `python3 -m pytest tests/ --no-header -q | tail -5`. Confirm 1 RED (Test #8) / 565 PASS / 13 SKIP.
4. 2 NEW tests isolated: `python3 -m pytest tests/test_csa_compressor_real_mlx_parity.py tests/test_csa_attention_real_mlx_parity.py -v | tail -20`. Confirm GREEN + note parity deltas.
5. **HCA byte-intact** (watch-item 1): `git show aa93c5d -- python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py | grep -E 'def _attention_real_mlx|def _csa_compressor_real_mlx|def _csa_attention_real_mlx|def _csa_block_bias_mlx'`. Confirm: (a) `_attention_real_mlx` HCA branch body byte-identical (Coder chose NEW `_csa_attention_real_mlx`, so HCA `_attention_real_mlx` NOT edited); (b) 3 NEW helpers added additively.
6. **Rope option α + ape no-transpose** (watch-item 2): read `_csa_compressor_real_mlx:819` body. Confirm: (a) APE consumed token-major `(rate=4, 2*head_dim=1024)` NO transpose (BA §0.2 — the `ape[:out_dim,:].T` lives ONLY inside the bypassed FROZEN helper, not here); (b) compress-YaRN-tail rope via REUSED `_compress_rope_yarn_tail_tables_mlx:545` + `_apply_rope_tail_mlx:266`; (c) FROZEN `_csa_windowed_compressor_mlx:347` body byte-identical (bypassed, not edited).
7. **block_bias -inf mask EXACT** (watch-item 3): read `_csa_block_bias_mlx` body + the AC2 test. Confirm: (a) `-1` sentinel positions → `-inf` mask EXACT (boolean equality with torch); (b) valid pick positions → 0 (finite_max_abs == 0.0); (c) scatter correctness.
8. **FROZEN byte-intact** (independent): `git show aa93c5d -- deepseek_v4.py` — confirm ONLY additions (3 new helpers). NO body edits to: `_attention_real_mlx` HCA branch, `_csa_windowed_compressor_mlx:347`, `_hca_compressor_mlx`, `_indexer_mlx:620`, `_indexer_scorer_mlx:591`, `_compress_rope_yarn_tail_tables_mlx`, `_apply_rope_*`, `_rope_*_tables_mlx`, `_attention_mlx` cr=0 body, `_csa_attention_mlx:515`, `_csa_compressor_mlx:419`, `_csa_indexer_mlx:435`, `_csa_config_error:317`, `_causal_sliding_mask_mlx:914`, `_hyperconnection_mlx`, `_hyperhead_mlx`, `sanitize_weights`, parity `Model`, 9 ADR-0017 forbidden.
9. **Sha-pin cascade** (watch-item 4, rigorous): `grep -rn -E '0x[0-9a-f]{16}|sha16|deepseek_v4.*hash' tests/ scripts/finetune_ds4.py python-envs/mlx/src/ 2>/dev/null | grep -iv 'node_modules\|\.venv'`. Enumerate ALL pin sites referencing deepseek_v4.py content hash. Confirm: count advanced (each changed line references NEW post-edit hash), no phantom, no stale old hash in non-pycache. **Report precise count** (don't under-count like 13.3b-1/2a).
10. Regression: 13.3b-1 (AC0-AC3) + 13.3b-2a (AC1-AC2) + 13.3a nn (12 incl backward AC) + 13.2 FP4 (9) + tiny CSA (11.14/11.15) all GREEN.
11. Anti-circularity: `tests/test_csa_compressor_real_mlx_parity.py` + `tests/test_csa_attention_real_mlx_parity.py` — torch ref from SAME synthesized weights (ADR 0007 §4). NO second MLX primitive as oracle.
12. `git diff --check` clean.

## PASS criteria
- 2 new AC tests GREEN; block_bias -inf mask EXACT.
- Full suite: 1 RED (Test #8) / 565 PASS / 13 SKIP / 0 introduced RED.
- HCA `_attention_real_mlx` byte-identical (NEW helper chosen).
- Rope option α + ape no-transpose confirmed.
- FROZEN bodies byte-intact (additions only).
- Sha-pin cascade advanced exactly the right count (precisely enumerated, not under-counted).
- Anti-circularity honored.
- Regression all GREEN.

## STOP-ESCALATE
If anything unexpected RED, or FROZEN body edited, or HCA branch changed, or cascade broken — write `agent-output/cmux-13-3b/test-manager-13-3b-2b-stop.md` + `{"status":"error",...}`.

## Deliverables
1. `agent-output/cmux-13-3b/test-report-13-3b-2b.md` — run results, parity deltas, rope option α + ape no-transpose verdict, HCA byte-identical verdict, block_bias -inf mask EXACT verdict, FROZEN byte-intact verdict, sha-pin cascade precise count, anti-circularity verdict, PASS/FAIL.
2. `.cmux-status/test-manager.done` (`{"status":"ok","role":"Test Manager"}`).
3. In-pane JSON: `{"status":"ok","role":"Test Manager"}` in surface:85 ONLY.

## Style
Caveman ultra default; byte-exact exempt. Use `ctx_execute` for large output. BEGIN NOW.
