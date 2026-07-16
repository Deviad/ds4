# Epic 14 / Story 14.4 — Repository reproducibility and stale-file cleanup

## Authorization

Operator authorized: inspect what belongs to current implementation, commit
legitimate slipped files, delete remaining stale/generated files. No push.

This BA slice defines classification and cleanup requirements only. No
deletions, no `git rm`, no `git add`, no `git commit`, no `git push` during BA.

## Trackable user story

```
As a DS4 fine-tuning repository maintainer (WHO), I want untracked
implementation files committed and stale generated/ephemeral files removed
from version control (WHAT), so that a fresh clone reproduces the full Epic 14
test baseline without missing imports or carrying transient junk (WHY).
```

## Evidence base

BA inspected (no mutations):

- `git status --porcelain`: 6 modified tracked files, 0 staged, ~1,608 untracked.
- `git ls-files`: 668 tracked files.
- `git ls-files --others --exclude-standard`: 1,608 untracked files.
- `git submodule status` / `git ls-tree HEAD vendor/mlx-lm`: outer gitlink
  `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`.
- Inner `vendor/mlx-lm` `git status --short`: 2 staged files
  (`mlx_lm/tuner/trainer.py`, `tests/test_tuner_trainer.py`), 2,627
  insertions / 14 deletions vs `HEAD`.
- `git show af08c25 --stat`: last commit content.
- Source imports in `scripts/ds4_segmented_smoke.py`, `tests/test_ds4_segmented_smoke.py`,
  `scripts/finetune_ds4.py`, `tests/test_finetune_ds4.py`.
- `docs/backlog.md` ADR reference count: 268.
- `.gitignore` current content.
- `.gitmodules`: single submodule `vendor/mlx-lm` →
  `git@github.com:Deviad/mlx-lm.git`.

## R14.4-1 — Classification: current implementation (must be committed)

### R14.4-1A — Untracked python-envs/mlx/src/ (15 files)

These are imported by committed Epic 14 source/tests. Without them, the
package `ds4_ft_mlx` is unimportable and the test baseline is
non-reproducible.

| File | Reason required |
|---|---|
| `python-envs/mlx/src/ds4_ft_mlx/__init__.py` | Package init; without it, `import ds4_ft_mlx` fails |
| `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_attention_spec.py` | DeepSeek V4 attention module spec; imported by nn modules |
| `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_checkpoint.py` | Checkpoint loader; referenced by tests and conversion |
| `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_mapping.py` | Tensor name mapping; required by remap scripts and tests |
| `python-envs/mlx/src/ds4_ft_mlx/deepseek_v4_moe_spec.py` | MoE module spec; imported by nn modules |
| `python-envs/mlx/src/ds4_ft_mlx/lora_targets.py` | LoRA target allowlist; referenced by `finetune_ds4.py` and `convert_lora_to_ds4.py` |
| `python-envs/mlx/src/ds4_ft_mlx/mlx_lm_plugin.py` | MLX-LM plugin integration; imported by the fork model registration |
| `python-envs/mlx/src/ds4_ft_mlx/numpy_real_forward_reference.py` | Numpy forward reference; imported by parity tests |
| `python-envs/mlx/src/ds4_ft_mlx/vendor/__init__.py` | Vendor sub-package init |
| `python-envs/mlx/src/ds4_ft_mlx/vendor/mlx_lm_models/__init__.py` | Vendor model sub-package init |
| `python-envs/mlx/src/sitecustomize.py` | PYTHONPATH sitecustomize; referenced by test environment setup |

Action: `git add` all 15. Coder must verify each import chain before staging.

### R14.4-1B — Untracked scripts/ (4 files)

| File | Reason required |
|---|---|
| `scripts/convert_lora_to_ds4.py` | Referenced by committed `tests/test_finetune_ds4.py` line 1077; production converter |
| `scripts/fuse_lora_hf.py` | Referenced in backlog Stories 5.3/5.4; production HF fuse path |
| `scripts/make_synth_lora.py` | Referenced in backlog; synthetic adapter generator for tests |
| `scripts/smoke_fuse_serve.sh` | Smoke script referenced by backlog smoke contracts |

Action: `git add` all 4.

### R14.4-1C — Untracked tests/ (27 files)

These participate in the test baseline. Per the AGENTS.md tracking hygiene hard
rule, any test file cited in verdicts MUST be tracked.

Key baseline participants:

| File | Role |
|---|---|
| `tests/test_convert_lora_to_ds4.py` | Unit tests for `convert_lora_to_ds4.py` |
| `tests/test_fuse_lora_hf.py` | Unit tests for `fuse_lora_hf.py` |
| `tests/test_make_synth_lora.py` | Unit tests for `make_synth_lora.py` |
| `tests/test_deepseek_v4_forward_parity.py` | Forward parity test |
| `tests/test_deepseek_v4_lora_targets.py` | LoRA target validation test |
| `tests/test_deepseek_v4_nn_interaction_ablation.py` | NN interaction ablation |
| `tests/test_deepseek_v4_nn_multilayer_peak.py` | Multilayer peak test |
| `tests/test_deepseek_v4_validate_real_mode_relaxation.py` | Real-mode relaxation validation |
| `tests/test_ds4_metal_routed_i8_e8m0_isolation.py` | Metal isolation test |
| `tests/test_finetune_ds4_b1_hc_mult_multilayer_readiness.py` | B1 readiness diagnostic |
| `tests/test_finetune_ds4_b2_real_checkpoint_payload_readiness.py` | B2 checkpoint payload readiness |
| `tests/test_finetune_ds4_b2_routed_dequant_trusted_reference_readiness.py` | B2 routed dequant readiness |
| `tests/test_finetune_ds4_csa_topk_primitive.py` | CSA topk primitive test |
| `tests/test_finetune_ds4_fuse_hf_gate.py` | Fuse HF gate test |
| `tests/test_finetune_ds4_fused_gate_perf.py` | Fused gate performance test |
| `tests/test_finetune_ds4_i8_dequant_integration.py` | I8 dequant integration test |
| `tests/test_finetune_ds4_stateful_decode_readiness.py` | Stateful decode readiness test |
| `tests/test_real_forward_intermediate_dump.py` | Real forward intermediate dump test |
| `tests/test_shim_ds4_safetensors.py` | Shim safetensors test |

Test helpers/fixtures:

| File | Role |
|---|---|
| `tests/_fuse_fixture.py` | Fuse test fixture |
| `tests/ds4_e8m0_ocp_witness.py` | E8M0 OCP witness module |
| `tests/ds4_f8_e4m3_e8m0_ocp_witness.py` | F8 E4M3 E8M0 OCP witness module |
| `tests/helpers/routed_fp4_multilayer_peak_probe.py` | Multilayer peak probe helper |
| `tests/fixtures/deepseek_v4_nn_multilayer_peak_topology.json` | Peak topology fixture |
| `tests/test_deepseek_v4_nn_backward.py` | NN backward test |
| `tests/test_deepseek_v4_nn_construct.py` | NN construction test |
| `tests/test_deepseek_v4_moe_parity.py` | MoE parity test |

Action: `git add` all 27. Reviewer must verify `git ls-files` for every test
file cited in the Epic 14.3b verdict, per the tracking hygiene hard rule.

### R14.4-1D — Untracked docs/adr/ (24 files)

ADRs 0001 through 0023 plus `docs/adr/README.md` are untracked. They are
referenced 268 times in `docs/backlog.md`, by `docs/architecture.md`, and by
`docs/technical-spec.md`. ADRs 0024-0029 are tracked.

Action: `git add` all 24. These are durable architecture decisions that are
already referenced as canonical by committed docs.

### R14.4-1E — Untracked docs/ (2 files)

| File | Reason |
|---|---|
| `docs/deepseek-v4-architecture-dossier.md` | Architecture dossier referenced by backlog Story 11.1 |
| `docs/deepseek-v4-mtp-policy.md` | MTP policy referenced by backlog |

Action: `git add` both.

### R14.4-1F — Untracked .pi/ (16 files)

Project-local Pi agent scaffolding used by the cmux role pipeline. These are
the reusable role-agent definitions and supervision scripts referenced by
`AGENTS.md` cmux orchestration section.

| Path | Role |
|---|---|
| `.pi/agents/README.md` | Agents directory documentation |
| `.pi/agents/project-overlay.md` | Project-specific role context |
| `.pi/agents/bin/check-role.sh` | Role completion check |
| `.pi/agents/bin/check-roster.sh` | Roster check |
| `.pi/agents/bin/dispatch-role.sh` | Role dispatch |
| `.pi/agents/bin/generate-agents-md.sh` | Agents MD generator |
| `.pi/agents/bin/launch-role.sh` | Role launcher |
| `.pi/agents/bin/mux-lib.sh` | Mux library |
| `.pi/agents/bin/notify-lib.sh` | Notification library |
| `.pi/agents/bin/notify-watch.sh` | Watch notification |
| `.pi/agents/bin/pipeline-lib.sh` | Pipeline library |
| `.pi/agents/bin/pipeline.sh` | Pipeline runner |
| `.pi/agents/bin/spawn-role-panes.sh` | Role pane spawner |
| `.pi/agents/bin/supervisor-poll.sh` | Supervisor polling |
| `.pi/agents/bin/wait-role-continuous.sh` | Continuous wait |
| `.pi/agents/bin/wait-role.sh` | Role wait |

Action: `git add` all 16. Note: `.pi/agents/pipeline.conf` is in `.gitignore`
(per-user personal settings) and should remain excluded.

### R14.4-1G — Vendor inner staged changes (CRITICAL)

The inner `vendor/mlx-lm` has two staged files that are NOT committed:

| File | Staged diff | Role |
|---|---|---|
| `vendor/mlx-lm/mlx_lm/tuner/trainer.py` | +240 / -14 lines | Story 14.1 trainer seam: custom `loss_and_grad` passthrough |
| `vendor/mlx-lm/tests/test_tuner_trainer.py` | +2401 lines | Story 14.1/14.2a trainer test suite |

These are REQUIRED by all Epic 14 tests. A fresh clone resolves the submodule
to `15b522f...` which does NOT include these changes → all Epic 14 tests fail.

**Resolution required:**

1. Commit the two staged files in the inner `vendor/mlx-lm` repo.
2. Update the outer gitlink to point to the new inner commit.
3. Record the new pin in `docs/architecture.md` (line 69 currently references
   `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`).
4. Update all protected-evidence references in `docs/backlog.md` that pin the
   inner `HEAD` / outer gitlink at `15b522f...` to the new commit hash.
5. The inner commit message must explain the changes are the Story 14.1/14.2a
   trainer seam, not a generic upstream patch.
6. ADR 0029 (`docs/adr/0029-pinned-mlx-lm-source-selection.md`, tracked) must
   be reviewed for consistency with the new pin.

**Architect must adjudicate:** whether the outer gitlink update requires an ADR
amendment or a new ADR, and whether the inner commit should be squashed or
preserved as a separate commit on the fork branch.

## R14.4-2 — Classification: stale/generated/ephemeral files

### R14.4-2A — Tracked stale files (must be `git rm`'d)

| Category | Count | Examples | Reason |
|---|---|---|---|
| `.dispatch-epoch` | 77 | `agent-output/cmux-13-3b/architect-13-3b-5c.dispatch-epoch` | Transient epoch timestamps |
| `.pid` | 5 | `agent-output/cmux-13-3b/coder-13-3b-5b-smoke-train.pid` | Process IDs |
| `.rc` | 4 | `agent-output/cmux-13-3b/coder-13-3b-5-smoke-train.rc` | Return codes |
| `.log` | 105 | `agent-output/cmux-13-3b/coder-13-3b-4b-convert.log` | Transient execution logs |
| `.pyc` / `__pycache__/` | 1 | `agent-output/cmux-13-3b/__pycache__/run_smoke_4096_routed_fp4.cpython-313.pyc` | Compiled bytecode (already gitignored pattern but force-added) |
| `.pipeline-private/` | 2 | `.pipeline-private/implementation-state.v1.tsv` files in `agent-output/cmux-13-3b/` | Pipeline runtime state |
| `.pipeline-slice.v1` | 2 | `.pipeline-slice.v1` files | Pipeline slice state |
| `.cmux-status/coder.done` | 1 | `.cmux-status/coder.done` (tracked despite `.gitignore` for `.cmux-status/`) | Runtime marker; gitignored but force-added |
| **Total** | **~197** | | |

Action: `git rm --cached` all ~197 files. Do NOT delete from working tree
(delution manifest only removes from version control; files remain locally).

**Deletion manifest:** Coder must produce `agent-output/cmux-14-4/deletion-manifest.txt`
listing every file path removed from version control, one per line, sorted
alphabetically.

### R14.4-2B — Untracked ephemeral files (must NOT be committed)

| Category | Count | Reason |
|---|---|---|
| `agent-output/` untracked | 1,260 | Handoff evidence only (AGENTS.md: "agent-output/ is handoff evidence only, not canonical docs") |
| `custom-handoffs/` untracked | 142 | Handoff artifact directories from 14-2, 14-3-family slices |
| `custom-handoffs/*/.dispatch-epoch` | within 142 | Transient dispatch epochs |
| `custom-handoffs/*/.pipeline-private/` | within 142 | Pipeline runtime state |
| `custom-handoffs/*/.pipeline-slice.v1` | within 142 | Pipeline slice state |

Action: these remain untracked. Gitignore patterns should be added to prevent
accidental future commits.

### R14.4-2C — Gitignore additions required

The following patterns must be added to `.gitignore` to prevent future tracking
of ephemeral files:

```gitignore
# dispatch / pipeline runtime state
*.dispatch-epoch
*.pid
*.rc
*.log
.pipeline-private/
.pipeline-slice.v1

# agent-output and custom-handoffs are handoff evidence only
agent-output/
custom-handoffs/
```

Note: `.cmux-status/` is already gitignored. `*.pyc` and `__pycache__/` are
already gitignored. `agent-output/` and `custom-handoffs/` are NOT currently
gitignored which is why so many files accumulate as untracked — adding them
keeps `git status` clean without committing.

**Architect must adjudicate:** whether `agent-output/` and `custom-handoffs/`
should be fully gitignored (making them invisible to `git status`) or only
pattern-gitignored within those directories. Evidence artifacts may need to
remain visible for supervisor inspection.

## R14.4-3 — Classification: ambiguous or user-owned files

### R14.4-3A — Root-level scratch files

| File | Status | Classification | Action |
|---|---|---|---|
| `context.md` | Untracked | Scratch context note; not imported by any committed source/test; referenced only by `adapter-converter-implementation-gpt55.md` | AMBIGUOUS — do not commit or delete without operator confirmation |
| `adapter-converter-implementation-gpt55.md` | Untracked | Scratch implementation note from a GPT-5.5 session; references `context.md` and `plan.md` (both absent) | AMBIGUOUS — do not commit or delete without operator confirmation |

BA recommendation: these are scratch notes from external LLM sessions. They
should not be committed (no canonical doc references them) but should also not
be deleted without operator confirmation since they may contain personal
context. Recommend adding `context.md` and `adapter-converter-*.md` to
`.gitignore` or deleting them with explicit operator approval.

## R14.4-4 — Reproducibility contract

### R14.4-4A — Fresh-clone test gate

After all R14.4-1 commits and R14.4-2 removals, a simulated fresh clone must
reproduce the Epic 14.3b test baseline:

1. All `python-envs/mlx/src/ds4_ft_mlx/` imports resolve (including `__init__.py`).
2. All `scripts/*.py` referenced by committed tests exist and import.
3. All `tests/*.py` cited in the Epic 14.3b Test Manager verdict are tracked.
4. The inner `vendor/mlx-lm` resolves to a commit that includes the trainer
   seam changes.
5. `git submodule update --init` produces a working tree where
   `mlx_lm/tuner/trainer.py` contains the 240-line addition.
6. No `.dispatch-epoch`, `.pid`, `.rc`, `.log`, `.pyc`, `.pipeline-private/`,
   or `.pipeline-slice.v1` files are tracked.
7. The exact-fork pytest suite runs and passes: 246 passed, 3 skipped, 1
   warning, 2 subtests passed (from Story 14.3b Test Manager r4).

### R14.4-4B — Tracking hygiene verification

Per the AGENTS.md hard rule:

- For every test file cited in the Story 14.3b Test Manager r4 verdict,
  `git ls-files -- <file>` must return the file.
- If any cited file is untracked after cleanup, the verdict is downgraded to
  "246 passed locally, reproducibility NOT verified" and the slice is returned.
- Reviewer must run `git ls-files` for every test file before adjudicating.

### R14.4-4C — Protected evidence preservation

The following must remain byte-identical and tracked:

| Protected artifact | Check |
|---|---|
| `segmented_loss_and_grad.py` | blob SHA-256 `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518` |
| `test_ds4_segmented_loss_and_grad.py` | blob SHA-256 `618a0f22d350ed368e7bf7f782728ad2f6a11486e19a4c97c5e85228aea784c6` |
| `test_mlx_lm_source.py` | blob SHA-256 `dec2c2b5a7644540a7ec339712968593647abd64c70b821abddc97f9b6ff5d65` |
| Path A 365-file evidence | digest `7241924d6f9be2eb0df974fdcb1717728c71b6ee77d41dea3fe8a8af55ebb2af` |
| Epic 14.3 smoke evidence | `agent-output/cmux-14-3/smoke-report.json` + `smoke-log.txt` |
| ADR 0028 | existing content intact |
| ADR 0029 | existing content intact (may need pin update for gitlink) |
| `docs/backlog.md` | Story 14.3/14.3a/14.3b closeout evidence preserved |

## R14.4-5 — Vendor pin resolution (CRITICAL)

### Current state

- `vendor/mlx-lm` outer gitlink: `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`
- Inner HEAD: `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe` (matches)
- Inner STAGED (uncommitted): 2 files, 2,627 insertions / 14 deletions

### Required resolution

1. Coder commits the two staged files in the inner `vendor/mlx-lm` repo.
2. Coder records the new inner HEAD hash.
3. Coder updates the outer gitlink to point to the new inner commit
   (`git add vendor/mlx-lm` after `cd vendor/mlx-lm && git commit`).
4. `docs/architecture.md` line 69 and all `docs/backlog.md` references to
   `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe` are updated to the new hash.
5. ADR 0029 is reviewed for consistency; if the pin changed, an ADR amendment
   or a new ADR records the new provenance.

### Pin tension risk

If the inner changes are committed but the outer gitlink is NOT updated, the
outer repo shows `vendor/mlx-lm` as dirty on every `git status`. If the outer
gitlink is updated but the inner changes are NOT committed, the submodule
cannot resolve and `git submodule update` fails. Both must happen in the same
commit slice.

**Architect must define:** the exact commit sequence (inner-first then
outer-gitlink, or a single outer commit that includes the gitlink bump), the
inner commit message, and whether a new ADR is required for the pin update.

## R14.4-6 — Commit manifest

Coder must produce `agent-output/cmux-14-4/commit-manifest.txt` listing:

1. All files `git add`'d (R14.4-1A through R14.4-1F), one per line, sorted.
2. The inner vendor commit hash and new outer gitlink hash (R14.4-5).
3. All files `git rm --cached`'d (R14.4-2A), one per line, sorted.
4. The `.gitignore` additions (R14.4-2C).
5. The deletion manifest `agent-output/cmux-14-4/deletion-manifest.txt`.

No `git commit` or `git push` during this slice — the manifest records what WILL
be committed when the operator authorizes it. The manifest itself is a tracked
and staged BA artifact.

## R14.4-7 — STOP/ESCALATE criteria

Emit error JSON and write `custom-handoffs/14-4-repo-hygiene/ba-stop.md` if:

- Any untracked implementation file cannot be linked to a committed import
  reference (orphan code with no consumer).
- Any tracked stale file is referenced by current source/tests (not actually
  stale).
- The vendor inner staged changes cannot be committed (e.g., inner repo is
  detached HEAD, upstream rejected, or inner changes conflict).
- Deleting any tracked file would remove evidence referenced by a current
  verdict or protected digest.
- An untracked ADR contradicts a tracked ADR (same number, different content).
- The scope of untracked files is so large that manual verification of each
  file's classification is infeasible in one slice.
- The operator has not confirmed whether `agent-output/` and `custom-handoffs/`
  should be fully gitignored (some evidence may need to remain visible for
  future slices).

## Acceptance criteria

1. `custom-handoffs/14-4-repo-hygiene/requirements.md` exists with testable
   classification (R14.4-1/2/3), reproducibility (R14.4-4), vendor pin
   (R14.4-5), commit manifest (R14.4-6), and STOP (R14.4-7) requirements with
   no unresolved placeholder. **Proof: file exists and every R14.4-* section
   has concrete criteria.**

2. `docs/backlog.md` contains a Story 14.4 user story in the exact form:
   `As a [type of user] (WHO), I want [some goal] (WHAT), so that [some reason]
   (WHY).` **Proof: grep for `14.4` and `As a.*WHO.*I want.*WHAT.*so
   that.*WHY` in `docs/backlog.md`.**

3. All three classification categories are exhaustive: current implementation
   (R14.4-1, ~68 files to commit), stale to remove (R14.4-2A, ~197 files),
   ambiguous (R14.4-3A, 2 files). **Proof: counts match `git ls-files` /
   `git ls-files --others` spot checks.**

4. Vendor pin tension is documented with exact hashes and resolution steps.
   **Proof: R14.4-5 references `15b522f...` and counts 2,627 / 14 lines.**

5. No deletions, `git rm`, `git add`, `git commit`, or `git push` were
   performed during BA. **Proof: `git diff --stat` shows only `docs/backlog.md`
   modified; `git diff --cached --stat` shows only new BA artifact files staged;
   `git log --oneline -1` shows no new commit.**

6. Path A permanent STOP, Epic 14.3b smoke evidence, and all protected
   hashes are preserved. **Proof: R14.4-4C protected evidence table.**

7. All BA verdict files and backlog edits are tracked/staged; no commit or
   push. **Proof: `git ls-files -- custom-handoffs/14-4-repo-hygiene/requirements.md
   docs/backlog.md` returns both (after staging); `git log --oneline -1` shows
   no new commit from BA.**