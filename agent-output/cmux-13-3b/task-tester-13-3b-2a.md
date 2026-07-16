# Story 13.3b-2a — Test Manager Task Brief (independent validation)

## Role
Test Manager (neuralwatt/qwen3.6-35b · surface:85, fresh-context via /new). Independent validation of Coder 13.3b-2a (`a2c20b0`).

## Context
- Coder 13.3b-2a complete: Indexer real-dim MLX port (`_indexer_scorer_mlx` + `_indexer_mlx`). Commit `a2c20b0` (parent `1f5d8e4`).
- Coder claims: AC1-AC2 GREEN, rope landmine resolved (reimplemented Ca/Cb overlap inline + reused 13.3b-1 rope helper; FROZEN `_csa_windowed_compressor_mlx:347` byte-identical), full suite 1 RED (Test #8) / 563 PASS / 13 SKIP, FROZEN bodies byte-intact via AST.
- Watch-items: (1) rope-landmine resolution — confirm `_indexer_mlx` reimplements Ca/Cb overlap inline (NOT a FROZEN body edit) + applies compress-yarn-tail via REUSED 13.3b-1 helper; (2) AC2 set-based assertion (NOT tie-order) — confirm the test asserts SET of non-sentinel picks + scores, not exact index order; (3) sha-pin cascade (deepseek_v4.py edited additively → advance right count of sha16 pin sites); (4) **Test Manager 13.3b-1 gap**: counted "2 sites" vs Reviewer's "3 sites". For 13.3b-2a, EXPLICITLY enumerate ALL sha-pin sites by scanning the whole tests/ + scripts/finetune_ds4.py tree for sha16 pins referencing deepseek_v4.py — don't infer from grep hits on a subset.

## Your job — independent re-run + verification
1. `source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first).
2. `git log --oneline -2` — confirm `a2c20b0` is HEAD.
3. Full suite: `python3 -m pytest tests/ --no-header -q | tail -5`. Confirm exactly 1 RED (Test #8 `test_finetune_readiness_report_records_b0b_a_without_unblocking_b0`) / 563 PASS / 13 SKIP.
4. Run 2 NEW tests in isolation: `python3 -m pytest tests/test_indexer_scorer_mlx_parity.py tests/test_indexer_mlx_parity.py -v | tail -20`. Confirm GREEN + note parity assertion strategy.
5. **AC2 set-based verification** (watch-item 2): read `tests/test_indexer_mlx_parity.py`. Confirm the test asserts: (a) `-1` sentinel mask exact (positions of sentinels match torch); (b) SET of non-sentinel valid picks == torch set per query; (c) scores at valid picks == torch within tol; (d) tie order NOT asserted. If the test asserts exact index order at ties → FLAG (would be a flaky/incorrect test).
6. **Rope-landmine verification** (watch-item 1): `git show a2c20b0 -- python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py | grep -E '_csa_windowed_compressor_mlx|_indexer_mlx|_compress_rope_yarn_tail_tables_mlx|_apply_rope_tail_mlx'`. Confirm: (a) `_indexer_mlx` reimplements Ca/Cb overlap pooling inline (additive new code); (b) applies compress-yarn-tail via REUSED 13.3b-1 helpers; (c) FROZEN `_csa_windowed_compressor_mlx:347` body NOT edited (byte-identical).
7. **FROZEN byte-intact** (independent): `git show a2c20b0 -- python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py` — confirm ONLY additions (2 new helpers). NO body edits to `_csa_windowed_compressor_mlx`, `_csa_compressor_mlx`, `_csa_indexer_mlx:435` (tiny), `_attention_mlx`, `_csa_*`, `_hyperconnection_mlx`, `_hyperhead_mlx`, `_hca_compressor_mlx`, `_attention_real_mlx`, `_compress_rope_yarn_tail_tables_mlx`, `_apply_rope_*`, `_rope_*_tables_mlx`, `sanitize_weights`, parity `Model`, 9 ADR-0017 forbidden.
8. **Sha-pin cascade** (watch-item 3, avoid 13.3b-1 gap): scan WHOLE tree for sha16 pins referencing deepseek_v4.py:
   - `grep -rn -E '0x[0-9a-f]{16}|sha16|deepseek_v4.*hash' tests/ scripts/finetune_ds4.py python-envs/mlx/src/ 2>/dev/null | grep -iv 'node_modules\|\.venv'`
   - Enumerate ALL pin sites. Confirm count advanced (each changed line references the NEW post-edit deepseek_v4.py hash). Confirm no phantom (no pin updated that didn't need to change).
   - Note: the exact count — report it precisely (don't under-count like 13.3b-1).
9. Regression: 13.3b-1 (AC0-AC3) + 13.3a nn (12 incl backward AC) + 13.2 FP4 (9) + tiny CSA (11.14/11.15) all GREEN.
10. Anti-circularity: `tests/test_indexer_scorer_mlx_parity.py` + `tests/test_indexer_mlx_parity.py` — confirm torch reference computes expected from SAME synthesized weights/config (ADR 0007 §4). NO second MLX primitive as oracle.
11. `git diff --check` clean.

## PASS criteria
- 2 new AC tests GREEN; AC2 set-based (not tie-order).
- Full suite: 1 RED (Test #8) / 563 PASS / 13 SKIP / 0 introduced RED.
- Rope landmine resolved correctly (inline reimpl + reused rope helper; FROZEN body byte-identical).
- FROZEN bodies byte-intact (additions only).
- Sha-pin cascade advanced exactly the right count (precisely enumerated, not under-counted).
- Anti-circularity honored.
- Regression all GREEN.

## STOP-ESCALATE
If anything unexpected RED, or FROZEN body edited, or AC2 asserts tie-order (flaky), or cascade broken — write `agent-output/cmux-13-3b/test-manager-13-3b-2a-stop.md` + `{"status":"error",...}`.

## Deliverables
1. `agent-output/cmux-13-3b/test-report-13-3b-2a.md` — run results, parity strategy, rope-landmine verdict, FROZEN byte-intact verdict, sha-pin cascade precise count, anti-circularity verdict, PASS/FAIL.
2. `.cmux-status/test-manager.done` (`{"status":"ok","role":"Test Manager"}`).
3. In-pane JSON: `{"status":"ok","role":"Test Manager"}` in surface:85 ONLY.

## Style
Caveman ultra default; byte-exact exempt. Use `ctx_execute` for large output. BEGIN NOW.
