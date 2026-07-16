# Story 14.4 — Coder notes (repo reproducibility + stale cleanup)

## Summary

| Metric | Before | After |
|--------|--------|-------|
| Tracked files | 670 | 734 |
| Stale tracked removed | 0 | 21 (git rm --cached) |
| Untracked → tracked | 0 | 85 (git add) |
| Modified tracked | 7 | 11 staged |
| Inner vendor staged/uncommitted | 2 files | 0 (committed at 80fab4e) |
| Outer gitlink | 15b522f | 80fab4e |

## Phase execution (all serial)

### Phase A — Unstage BA marker
- `git reset HEAD .cmux-status/ba-14-4.done` — unstaged

### Phase B — Remove stale tracked files (21 files)
- 1 `.cmux-status/coder.done`
- 2 `.pipeline-slice.v1` (14-3-filtered, 14-3-timeout)
- 18 `.dispatch-epoch` (14-3, 14-3-filtered, 14-3-timeout)
- All `git rm --cached`'d; files preserved on disk

### Phase C — Track implementation files (85 files)
- 11 `python-envs/mlx/src/` — package init, specs, plugin, vendor
- 4 `scripts/` — convert_lora, fuse_lora_hf, make_synth_lora, smoke_fuse_serve
- 25 `tests/` — parity, integration, readiness, helpers, fixtures
- 25 `docs/adr/` — ADRs 0001-0023 + README
- 2 `docs/` — architecture dossier, MTP policy
- 18 `.pi/` — agent scaffolding, role-pipeline scripts, project overlay
- 2 `tests/ds4_lora_test`, `tests/test_q4k_dot` — QUARANTINED (Mach-O binaries → gitignored)

### Phase D — Inner vendor commit + gitlink update
- Inner commit: `80fab4e419a57f9465bb9e2f4e90010d645e124c`
- Message: "Story 14.1/14.2a: trainer loss_and_grad seam and test suite"
- Outer gitlink: `15b522f...` → `80fab4e...`

### Phase E — FORK_SHA cascade
- `scripts/finetune_ds4.py:119`: `MLX_LM_FORK_SHA` → 80fab4e
- `tests/test_mlx_lm_source.py:50`: `FORK_SHA` → 80fab4e
- `tests/test_ds4_segmented_smoke.py:783`: sentinel blob → `24325ef3...`
- New `test_mlx_lm_source.py` blob: `24325ef35e915b1cc3275d5fac31400c116e1adda6da0254b6158c7a6233dda9`

### Phase F — Doc pin references (9 updates)
- `docs/architecture.md`: pin + annotation
- `docs/backlog.md`: 7 occurrences annotated (historical evidence preserved, chain-of-custody notes added)
- `docs/technical-spec.md`: pin updated
- `docs/adr/0029`: pin updated + amendment section added

### Phase G — Gitignore
- Added: `*.dispatch-epoch`, `*.pid`, `*.rc`, `*.log`, `.pipeline-private/`, `.pipeline-slice.v1`, `agent-output/`, `custom-handoffs/`, `context.md`, `adapter-converter-*.md`, `tests/ds4_lora_test`, `tests/test_q4k_dot`

### Phase H — Legitimate modifications staged
- `.gitignore`, `AGENTS.md`, `agent-output/cmux-13-1/coder-notes.md` (with `-f`)

### Phase I — Manifests
- `agent-output/cmux-14-4/commit-manifest.txt` (force-added)
- `agent-output/cmux-14-4/deletion-manifest.txt` (force-added)

## Verification

| Check | Result |
|-------|--------|
| Full Epic 14 suite | **246 passed, 3 skipped, 2 subtests passed** |
| Path A count | **365** (unchanged) |
| Path A digest | `7241924d...` (verified) |
| Vendor gitlink == inner HEAD | `80fab4e` == `80fab4e` |
| Provider source SHA-256 | `20572191...` OK |
| Provider test SHA-256 | `618a0f22...` OK |
| Source sentinel (updated) | `24325ef3...` OK |
| `git diff --cached --check` | exit 0 |
| py_compile | 6/6 OK |
| All verdict test files tracked | 7/7 confirmed |
| Submodule status | `80fab4e` (v0.31.3-20-g80fab4e) |
| No commit, no push | ✅ |

## Preserved
- Path A 365-file evidence: intact
- `segmented_loss_and_grad.py`: unchanged
- `test_ds4_segmented_loss_and_grad.py`: unchanged (Path A stale cleanup STOPPED)
- Epic 14.3 smoke evidence: intact
- ADR 0028: unchanged
- All other protected hashes: intact (except source-sentinel which was explicitly authorized)
