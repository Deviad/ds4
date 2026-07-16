# Story 14.4r2 — Architecture: Fresh-Clone Test Defect Micro-Revision

> Minimal fix for `test_active_memory_matrix` fresh-clone failure.
> No production edit, no commit, no push. Design only.

## Defect

`tests/test_ds4_segmented_loss_and_grad.py::test_active_memory_matrix`
at line 1021 writes to:

```python
(PROJECT_ROOT / "agent-output/cmux-14-2/memory-r4.log").write_text(...)
```

without ensuring the parent directory `agent-output/cmux-14-2/` exists.
The directory is NOT tracked in git (`git ls-files agent-output/cmux-14-2`
returns zero paths) and is now gitignored (`agent-output/` rule added in
the 14.4 cleanup). The local checkout had the directory from a prior run;
a fresh clone does not, causing:

```
FileNotFoundError: agent-output/cmux-14-2/memory-r4.log
```

This breaks the R14.4-4A fresh-clone 246-passed contract (1 failed /
245 passed in the Reviewer's synthetic staged clone).

## Fix (exact, minimal)

### C1. Create output parent before write

**File:** `tests/test_ds4_segmented_loss_and_grad.py`
**Line:** 1021-1023

**Before:**
```python
    (PROJECT_ROOT / "agent-output/cmux-14-2/memory-r4.log").write_text(
        "ACTIVE_MEMORY_MATRIX PASS 8/8\n" + "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )
```

**After:**
```python
    _log_path = PROJECT_ROOT / "agent-output/cmux-14-2/memory-r4.log"
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_path.write_text(
        "ACTIVE_MEMORY_MATRIX PASS 8/8\n" + "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )
```

This is the only edit to this file. `mkdir(parents=True, exist_ok=True)` is
idempotent: if the directory already exists (local checkout), it is a no-op.
In a fresh clone, it creates the full `agent-output/cmux-14-2/` path. The
output file itself remains untracked (gitignored) — it is runtime test
evidence, not a reproducibility input.

### Why not a temp directory or pytest tmp_path?

`test_active_memory_matrix` spawns a child subprocess that writes JSON
payloads to stdout; the parent test collects them and writes the summary
log. The log path `agent-output/cmux-14-2/memory-r4.log` is a project-
conventional evidence destination, not a test fixture. Using `tmp_path`
would change the evidence path contract and require updating downstream
references. The minimal fix preserves the existing path and just ensures
the parent exists.

## Protected-hash cascade

### Current state

| File | Current blob SHA-256 | Protected by |
|---|---|---|
| `tests/test_ds4_segmented_loss_and_grad.py` | `618a0f22d350ed368e7bf7f782728ad2f6a11486e19a4c97c5e85228aea784c6` | `TestProtected.test_provider_test` in `test_ds4_segmented_smoke.py` line 780 |
| `tests/test_ds4_segmented_smoke.py` | `7f980d06e1d173fb5cabd5ea092a00c535e3e66f8a79222edeecf9ddeec546e8` | Not protected by any hash assertion |
| `tests/test_mlx_lm_source.py` | `24325ef35e915b1cc3275d5fac31400c116e1adda6da0254b6158c7a6233dda9` | `TestProtected.test_source_sentinel` in `test_ds4_segmented_smoke.py` line 786 |
| `python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py` | `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518` | `TestProtected.test_provider_source` in `test_ds4_segmented_smoke.py` line 730 |

### Cascade chain

1. **C1 edits `test_ds4_segmented_loss_and_grad.py`** → blob hash changes
   from `618a0f22...` to `<NEW_HASH_A>`.

2. **C2 must update `test_ds4_segmented_smoke.py` line 780** → expected hash
   in `test_provider_test` changes from `618a0f22...` to `<NEW_HASH_A>`.

3. **C2 edit changes `test_ds4_segmented_smoke.py` blob hash** from
   `7f980d06...` to `<NEW_HASH_B>`. This file is NOT protected by any
   external hash assertion, so no further cascade.

### Files NOT affected

| File | Why unchanged |
|---|---|
| `segmented_loss_and_grad.py` | Production source — not edited. Blob `205721...` intact. |
| `test_mlx_lm_source.py` | Not edited. Blob `24325ef...` intact. |
| `test_ds4_segmented_loss_and_grad.py::test_path_a_manifest_reproducible_from_index` | Checks `git ls-files agent-output/cmux-13-3b` count/digest — Path A untouched. |
| Path A 365-file manifest | No Path A files added or removed. |
| ADR 0028 | Not edited. |

### Exact computation order

The Coder MUST compute `<NEW_HASH_A>` programmatically immediately after
editing `test_ds4_segmented_loss_and_grad.py`, BEFORE editing
`test_ds4_segmented_smoke.py`:

```bash
python3 -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('tests/test_ds4_segmented_loss_and_grad.py').read_bytes()).hexdigest())"
```

Then replace the expected hash at line 780 of `test_ds4_segmented_smoke.py`
with the computed value. Do NOT guess or hardcode the hash — compute it
from the actual file bytes after the edit.

## What does NOT change (binding)

| # | Invariant | Verification |
|---|-----------|-------------|
| N1 | `segmented_loss_and_grad.py` (provider source, `205721...`) unchanged | `TestProtected.test_provider_source` — no edits |
| N2 | `test_mlx_lm_source.py` (`24325ef...`) unchanged | `TestProtected.test_source_sentinel` — no edits |
| N3 | Path A 365-file count and digest `7241924d...` unchanged | `test_path_a_manifest_reproducible_from_index` — no Path A mutations |
| N4 | ADR 0028 (`0aa743...`) unchanged | Not edited |
| N5 | `test_path_a_manifest_reproducible_from_index` logic unchanged | Only `test_active_memory_matrix` is edited, not the Path A test |
| N6 | No production source edits | Only test file edited |
| N7 | No `git commit` or `git push` | Staging only |
| N8 | Vendor inner commit `80fab4e...` unchanged | No inner repo actions |
| N9 | Outer gitlink `80fab4e...` unchanged | No outer gitlink actions |

## TDD evidence

### Red stage (synthetic fresh-clone simulation)

1. Remove the stale directory: `rm -rf agent-output/cmux-14-2/`
2. Run the specific test:
   ```bash
   PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" \
     PYTHONDONTWRITEBYTECODE=1 \
     python-envs/mlx/.venv/bin/python -m pytest -q \
     tests/test_ds4_segmented_loss_and_grad.py::test_active_memory_matrix
   ```
3. **Expected (before fix):** `FileNotFoundError` — RED.
4. **Expected (after fix):** PASS — GREEN.

### Green stage

1. Apply C1 (add `mkdir` before `write_text`).
2. Compute new blob hash of `test_ds4_segmented_loss_and_grad.py`.
3. Apply C2 (update expected hash in `test_provider_test`).
4. Re-run from clean state:
   ```bash
   rm -rf agent-output/cmux-14-2/
   PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" \
     PYTHONDONTWRITEBYTECODE=1 \
     python-envs/mlx/.venv/bin/python -m pytest -q \
     tests/test_ds4_segmented_loss_and_grad.py::test_active_memory_matrix
   ```
5. **Expected:** PASS.

### Full suite verification

```bash
PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" \
  PYTHONDONTWRITEBYTECODE=1 \
  python-envs/mlx/.venv/bin/python -m pytest -q \
  tests/test_ds4_segmented_smoke.py tests/test_finetune_ds4.py \
  tests/test_mlx_lm_source.py tests/test_ds4_segmented_loss_and_grad.py \
  tests/test_ds4_gguf_base_smoke.py
```

**Expected:** 246 passed, 3 skipped, 1 warning, 2 subtests passed.

### Synthetic staged clone verification (Reviewer entry condition)

The Reviewer's procedure from R2-B1:

1. Export current index as a binary patch.
2. Clone outer HEAD to a temp directory (`--no-local --no-checkout`).
3. Apply the staged patch and create a synthetic outer commit.
4. `git submodule update --init --recursive` (fetches `80fab4e` from GitHub).
5. Ensure `agent-output/cmux-14-2/` does NOT pre-exist in the clone.
6. Run the exact five-file suite.
7. **Expected:** 246 passed, 0 failed.

This is the binding fresh-clone gate. The Test Manager must use this
procedure, not the local checkout, to produce a GREEN verdict.

## File scope

| Path | Action | Justification |
|------|--------|---------------|
| `tests/test_ds4_segmented_loss_and_grad.py` | EDIT (2 lines: add `_log_path` + `mkdir` before `write_text`) | Fix fresh-clone `FileNotFoundError` |
| `tests/test_ds4_segmented_smoke.py` | EDIT (1 line: update expected hash in `test_provider_test`) | Protected-hash cascade from C1 |

### Forbidden changes

- `python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py` (provider source `205721...`)
- `tests/test_mlx_lm_source.py` (sentinel `24325ef...`)
- Any production source file
- Any file under `agent-output/cmux-13-3b/`
- Any file under `vendor/mlx-lm/`
- `scripts/finetune_ds4.py`
- `docs/architecture.md`, `docs/backlog.md`, `docs/technical-spec.md`
- Any ADR

## Handoff force-add and manifest regeneration ordering

The Reviewer R2-B2 and R2-B3 findings require:
1. Force-add missing current-slice handoffs.
2. Rebuild manifests from the final index.

### Ordering

```
Phase 1: Fix test defect
  P1a. Edit test_ds4_segmented_loss_and_grad.py (C1)
  P1b. Compute new blob hash
  P1c. Edit test_ds4_segmented_smoke.py (C2 — cascade)
  P1d. git add tests/test_ds4_segmented_loss_and_grad.py tests/test_ds4_segmented_smoke.py

Phase 2: Force-add current-slice handoffs
  P2a. git add -f custom-handoffs/14-4-repo-hygiene/architecture-r2.md
  P2b. git add -f custom-handoffs/14-4-repo-hygiene/task-coder-r4.md  (if created)
  P2c. git add -f custom-handoffs/14-4-repo-hygiene/task-reviewer-r2.md
  P2d. git add -f custom-handoffs/14-4-repo-hygiene/task-tester-r2.md
  P2e. git add -f custom-handoffs/14-4-repo-hygiene/test-report-r2.md (or test-report-r3.md)
  P2f. git add -f custom-handoffs/14-4-repo-hygiene/review.md (updated with R2 findings closure)
  P2g. git add -f any other current-slice evidence files

Phase 3: Rebuild manifests from final index
  P3a. git diff --cached --name-status --diff-filter=A | wc -l  → count additions
  P3b. git diff --cached --name-status --diff-filter=M | wc -l  → count modifications
  P3c. git diff --cached --name-status --diff-filter=D | wc -l  → count deletions
  P3d. Rebuild agent-output/cmux-14-4/commit-manifest.txt with sorted, complete lists
  P3e. Verify deletion-manifest.txt still matches (should be unchanged — no new deletions)
  P3f. git add -f agent-output/cmux-14-4/commit-manifest.txt agent-output/cmux-14-4/deletion-manifest.txt
  P3g. Re-add manifests to the manifest (self-reference — the manifest files are staged additions themselves)

Phase 4: Final verification
  P4a. git diff --cached --check  → exit 0
  P4b. Run full five-file suite from local (should be 246 passed)
  P4c. Remove agent-output/cmux-14-2/ and rerun test_active_memory_matrix → PASS
  P4d. Reviewer runs synthetic staged clone gate
  P4e. Test Manager runs synthetic staged clone gate for GREEN verdict
```

### Critical ordering constraint

Manifests must be rebuilt as the LAST step (Phase 3), AFTER all handoffs
are force-added (Phase 2) and test files are staged (Phase 1). If
manifests are rebuilt before all files are staged, they will be stale.
If new handoff files are added after the manifest is rebuilt, the manifest
must be rebuilt again.

The manifest files themselves are staged additions in `agent-output/`
(which is gitignored), so they require `git add -f`. They are self-
referencing: the manifest lists itself as an addition.

## STOP-ESCALATE conditions

| # | Condition | Triggered? |
|---|---|---|
| 1 | The fix requires touching production source | No — only test file |
| 2 | The cascade requires touching a file beyond `test_ds4_segmented_smoke.py` | No — single cascade |
| 3 | Path A count or digest changes | No — no Path A files touched |
| 4 | `test_mlx_lm_source.py` blob changes | No — not edited |
| 5 | Provider source `segmented_loss_and_grad.py` blob changes | No — not edited |
| 6 | The `mkdir` call has side effects beyond test evidence | No — creates only a gitignored evidence directory |

No STOP conditions triggered. Design is unambiguous.

## Coder Phase 1 can proceed now

The design is complete. Coder should:

1. Apply C1 (2-line edit to `test_ds4_segmented_loss_and_grad.py`).
2. Compute new blob hash.
3. Apply C2 (1-line edit to `test_ds4_segmented_smoke.py`).
4. `git add` both test files.
5. Force-add all current-slice handoffs (Phase 2).
6. Rebuild manifests (Phase 3).
7. Run full suite + clean-directory test (Phase 4).
8. No `git commit`, no `git push`.