# Story 13.3b-4 — Test Manager (validate Coder 13.3b-4 staged work)

## Slice context
- **Story**: 13.3b-4 — real ckpt convert / key remap + load. Convert side ONLY.
- **Coder done**: HEAD still `58194a9` (NO commit per commit-gating policy). All 6 deliverables `git add`-staged. You validate the STAGED work via `git diff --cached`.
- **Commit gating**: Coder MUST NOT commit. Supervisor commits only AFTER both you (Test Manager) AND Reviewer return GREEN. Your job = run independent validation, write `test-manager-13-3b-4-report.md`, emit verdict.

## What Coder staged (read via `git diff --cached`)
```
scripts/remap_ds4_nn_weights.py              (465 lines, NEW — the remap script)
tests/test_remap_ds4_nn_weights_keyset.py    (76L, NEW — AC1)
tests/test_remap_ds4_nn_weights_load.py     (198L, NEW — AC2)
tests/test_remap_ds4_nn_weights_regression.py (33L, NEW — AC3)
agent-output/cmux-13-3b/coder-13-3b-4-notes.md (69L)
.cmux-status/coder.done
```

## Your job — independent validation (NOT a rubber-stamp)

### 1. Run the slice's 3 NEW tests + full suite
```bash
unset SSLKEYLOGFILE
source python-envs/mlx/.venv/bin/activate
python3 -m pytest tests/test_remap_ds4_nn_weights_keyset.py tests/test_remap_ds4_nn_weights_load.py tests/test_remap_ds4_nn_weights_regression.py -v
python3 -m pytest tests/ --no-header -q  # full suite — expect 569+3 NEW / 13 / 0 RED
```

### 2. AC verification
- **AC1** (keyset match): `test_remap_ds4_nn_weights_keyset.py` — remapped key set == nn `model.parameters()` key set EXACTLY (no missing/extra). Confirm the expected key-set comes from `model.parameters()` actually probed (anti-circularity — NOT a hand-written list).
- **AC2** (sub-checkpoint load): `test_remap_ds4_nn_weights_load.py` — load succeeds strict on sub-checkpoint (1 CSA + 1 HCA + 1 sliding layer); forward finite. Full 162GB run NOT required (BA Q2 — shape-only + sub-checkpoint acceptable).
- **AC3** (tiny regression): `test_remap_ds4_nn_weights_regression.py` — tiny CSA 11.14/11.15 byte-identical pre/post remap (remap is real-ckpt-only, must not touch tiny path).

### 3. FROZEN / nn-port / shim byte-intactness (CRITICAL)
Run direct checks (NOT git diff for untracked-baseline files):
- `git diff --cached --stat` — confirm ONLY the 6 staged deliverables. NO edits to `deepseek_v4.py` (FROZEN bodies), `deepseek_v4_nn.py` (nn-port, ADR 0025), `shim_ds4_safetensors.py` (BA Q1 chose NEW script).
- Source-hash: hash the FROZEN bodies + nn-port + shim at HEAD `58194a9` vs staged. Must be byte-identical. (AST/source extraction not needed if no staged changes touch them — `git diff --cached --stat` confirms.)
- sha-pin cascade SLICE-INVARIANT: 13.3b-4 should NOT touch `deepseek_v4.py` or nn-port, so the sha-pin cascade should be a NO-OP. Verify no phantom/stale pin sites.

### 4. ape-verbatim watch-item (BA §0.4)
BA caught Architect §5.2 prose STALE. The wired real helpers consume ape token-major (NO transpose); ckpt headers IDENTICAL to nn leaves. Coder MUST copy ape verbatim (no `.T`). Verify in the staged `scripts/remap_ds4_nn_weights.py`: grep for `.T`, `transpose`, `swapaxes` on ape tensors — should be ABSENT (or only on non-ape tensors with justification).

### 5. Tracking-hygiene HARD RULE (AGENTS.md `58194a9`)
- Run `git ls-files` for every test file Coder created/modified: `tests/test_remap_ds4_nn_weights_keyset.py`, `_load.py`, `_regression.py`.
- All MUST be tracked (staged = will-be-tracked after commit). If any is untracked AND unstaged → block (FLAG for supervisor).
- Confirm `git status --short` shows them as `A` (added to index), NOT `??` (untracked).

### 6. Regression: 0 introduced RED
Full suite must be 569+3 NEW / 13 skip / 0 RED. If any pre-existing test flipped RED, flag it.

## Deliverables
1. `agent-output/cmux-13-3b/test-manager-13-3b-4-report.md` — full validation report (test runs + AC1-AC3 + byte-intactness + ape-verbatim + tracking + regression).
2. `.cmux-status/test-manager.done` marker.
3. In-pane JSON: `{"status":"ok","role":"Test Manager"}` in surface:85 ONLY.

## STOP-ESCALATE
Write `agent-output/cmux-13-3b/test-manager-13-3b-4-stop.md` if:
- AC1/AC2/AC3 fail (RED) and Coder's work is genuinely broken (not just a test bug).
- FROZEN/nn-port/shim byte-intactness violated (Coder edited a protected body).
- ape TRANSPOSED (BA §0.4 violated) → Coder r2.
- A test file Coder created is untracked/unstaged (HARD RULE violation).

## Pre-flight read
1. `agent-output/cmux-13-3b/coder-13-3b-4-notes.md` (Coder's own summary).
2. `agent-output/cmux-13-3b/requirements-13-3b-4.md` (BA SPEC + §0.4 ape correction + Q1-Q4).
3. `agent-output/cmux-13-3b/task-coder-13-3b-4.md` (the brief Coder worked from).
4. `AGENTS.md` "Tracking hygiene for test files participating in the baseline (HARD RULE)" section (committed `58194a9`).

## Build
`source python-envs/mlx/.venv/bin/activate` (unset SSLKEYLOGFILE first). Use `ctx_execute` for suite output / header scans. Caveman ultra default; byte-exact exempt. BEGIN NOW. Echo `{"status":"ok","role":"Test Manager"}` in this pane only.
