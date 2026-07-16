# Story 13.3b-3 — Test Manager Task Brief (independent validation)

## Role
Test Manager (neuralwatt/qwen3.6-35b · surface:85, fresh-context via /new). Independent validation of Coder 13.3b-3 (`d1f1488`).

## Context
- Coder 13.3b-3 complete: INTEGRATION milestone (dispatch discriminator + AttentionNN guard lift + §4.1 nn wiring). Commit `d1f1488` (parent `aa93c5d`).
- Coder claims: AC1-AC4 GREEN; dispatch branch `if cr==4: return _csa_attention_real_mlx(...)` inserted in `_attention_mlx:1199-1204` (tiny + cr=128 byte-identical); AttentionNN guard `:153` lifted + new `AttentionCompressorNN` + `AttentionIndexerNN` submodules per §4.1 (CSA: compressor + indexer.compressor + indexer.weights_proj + indexer.wq_b; HCA: compressor only); `__call__` builds real weights dict + delegates to `_attention_mlx`; NO `_grouped_linear_mlx` (Q2 not needed); full suite **569 passed / 13 skipped / 0 RED** (Test #8 carve-out also now passes — Coder updated its stale proof-count expectation); FROZEN bodies byte-intact; sha-pin cascade advanced.
- Watch-items: (1) dispatch additive (tiny + cr=128 byte-identical, cr=0 body byte-identical); (2) GroupedLinear NOT introduced (Q2); (3) AttentionNN §4.1 wiring — compressor/indexer submodules + `__call__` delegation + cr=0 sliding path unchanged; (4) sha-pin cascade precise count (rigorous whole-tree scan).

## Your job — independent re-run + verification
1. `source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first).
2. `git log --oneline -2` — confirm `d1f1488` HEAD.
3. Full suite: `python3 -m pytest tests/ --no-header -q | tail -5`. Confirm 569 passed / 13 skipped / 0 RED (or 1 RED if Test #8 still flagged — Coder claims it now passes; verify).
4. NEW + extended tests isolated:
   - `python3 -m pytest tests/test_13_3b_1_dispatch_additive.py tests/test_13_3b_3_nn_mixed_layers_forward.py tests/test_csa_attention_real_mlx_parity.py tests/test_attention_real_mlx_hca_parity.py -v | tail -25`. Confirm GREEN.
5. **Dispatch additive** (watch-item 1): `git show d1f1488 -- python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py | head -60`. Confirm: ONLY the new `if cr==4: return _csa_attention_real_mlx(...)` branch inserted; tiny path (`_csa_config_error is None → _csa_attention_mlx`) byte-identical; cr=128 HCA fallback byte-identical; cr=0 body (`:1203+`) byte-identical.
6. **GroupedLinear NOT introduced** (watch-item 2): `grep -n '_grouped_linear_mlx' python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4.py`. Confirm ABSENT (Q2: not needed).
7. **AttentionNN §4.1 wiring** (watch-item 3): `git show d1f1488 -- python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/deepseek_v4_nn.py | head -100`. Confirm: guard `:153` lifted (no `if compression_ratio != 0: raise NotImplementedError`); new `AttentionCompressorNN` + `AttentionIndexerNN` submodules; CSA leaves (compressor.{wkv,wgate,ape,norm.weight} + indexer.compressor.{...} + indexer.weights_proj + indexer.wq_b); HCA leaves (compressor only); `__call__` builds real weights dict via `linear_weight` + delegates to `_attention_mlx`; cr=0 sliding path UNCHANGED.
8. **FROZEN byte-intact** (independent): `git show d1f1488 -- deepseek_v4.py` — confirm ONLY additive dispatch branch (no body edits to: cr=0 body, `_attention_real_mlx`, `_csa_attention_real_mlx`, `_csa_attention_mlx`, `_csa_config_error`, `_indexer_mlx`, `_indexer_scorer_mlx`, `_csa_compressor_real_mlx`, `_csa_block_bias_mlx`, `_hca_compressor_mlx`, `_compress_rope_yarn_tail_tables_mlx`, `_apply_rope_*`, `_rope_*_tables_mlx`, `_hyperconnection_mlx`, `_hyperhead_mlx`, `_causal_sliding_mask_mlx`, `sanitize_weights`, parity `Model`, FP4/i8 dequant, 9 ADR-0017 forbidden).
9. **sha-pin cascade** (watch-item 4, rigorous): `grep -rn -E '0x[0-9a-f]{16}|sha16|deepseek_v4.*hash' tests/ scripts/finetune_ds4.py python-envs/mlx/src/ 2>/dev/null | grep -iv 'node_modules\|\.venv'`. Enumerate ALL pin sites. Confirm count advanced (new post-`d1f1488` hashes), no phantom, no stale old hash. **Report precise count**.
10. Regression: 13.3b-1 (AC0-AC3) + 13.3b-2a (AC1-AC2) + 13.3b-2b (AC1-AC2) + 13.3a nn (12 incl backward AC) + 13.2 FP4 (9) + tiny CSA (11.14/11.15) all GREEN.
11. Anti-circularity: `test_13_3b_3_nn_mixed_layers_forward.py` — torch ref from SAME synthesized weights (ADR 0007 §4). NO second MLX primitive as oracle.
12. `git diff --check` clean.

## PASS criteria
- AC1-AC4 GREEN; dispatch additive (tiny + cr=128 + cr=0 byte-identical).
- GroupedLinear NOT introduced.
- AttentionNN §4.1 wiring correct (guard lifted, submodules, `__call__` delegation, cr=0 sliding unchanged).
- Full suite: 569 passed / 13 skipped / 0 introduced RED (verify Test #8 status).
- FROZEN bodies byte-intact (additions only).
- Sha-pin cascade advanced exactly the right count (precisely enumerated).
- Anti-circularity honored.
- Regression all GREEN.

## STOP-ESCALATE
If anything unexpected RED, or FROZEN body edited, or GroupedLinear introduced, or cr=0 body changed, or cascade broken — write `agent-output/cmux-13-3b/test-manager-13-3b-3-stop.md` + `{"status":"error",...}`.

## Deliverables
1. `agent-output/cmux-13-3b/test-report-13-3b-3.md` — run results, dispatch additive verdict, GroupedLinear-absent verdict, AttentionNN §4.1 wiring verdict, FROZEN byte-intact verdict, sha-pin cascade precise count, anti-circularity verdict, PASS/FAIL.
2. `.cmux-status/test-manager.done` (`{"status":"ok","role":"Test Manager"}`).
3. In-pane JSON: `{"status":"ok","role":"Test Manager"}` in surface:85 ONLY.

## Style
Caveman ultra default; byte-exact exempt. Use `ctx_execute` for large output. BEGIN NOW.
