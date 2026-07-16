# Story 14.4 — Architecture: Repository Reproducibility and Stale-File Cleanup

> Design only. No file mutations except `architecture.md`, ADR docs, and
> `docs/architecture.md` if required. No deletion, commit, or push.
> The Coder executes the plan under Reviewer and Test Manager gates.

## Current state snapshot

- **Tracked files:** 670
- **Untracked files:** 1,610
- **Modified tracked files:** 7
- **Staged files (outer):** 3 (BA artifacts)
- **Last commit:** `af08c25` "Complete DS4 segmented provider smoke activation"

### Modified tracked files (working tree, not staged)

| File | Status | Adjudication |
|---|---|---|
| `.cmux-status/coder.done` | Modified | STALE — `git rm --cached` (runtime marker, gitignored) |
| `.gitignore` | Modified (adds `.cmux-status/` + `.pi/agents/pipeline.conf`) | KEEP — legitimate additions from prior slices; build new patterns on top |
| `AGENTS.md` | Modified (removes model routing section, clarifies Reviewer role) | KEEP — legitimate content update; stage as-is |
| `agent-output/cmux-13-1/coder-notes.md` | Modified | KEEP — legitimate history; stage as-is |
| `docs/technical-spec.md` | Modified (adds MLX-LM source selection docs, §6.4) | KEEP — legitimate; references `15b522f...` that must be updated to new pin |
| `vendor/mlx-lm` | Modified (inner staged 2 files) | VENDOR PIN RESOLUTION — see §6 |

### Staged in outer repo

| File | Adjudication |
|---|---|
| `.cmux-status/ba-14-4.done` | STALE — `git rm --cached` (runtime marker, gitignored) |
| `custom-handoffs/14-4-repo-hygiene/requirements.md` | KEEP — BA artifact |
| `docs/backlog.md` | KEEP — Story 14.4 added by BA |

---

## 1. Commit manifest — untracked implementation files (88 files)

All files below are untracked, not gitignored, and required for fresh-clone
reproducibility. Each is referenced by committed source/tests/docs.

### 1A. python-envs/mlx/src/ (11 files)

| File | Required by |
|---|---|
| `ds4_ft_mlx/__init__.py` | `import ds4_ft_mlx` in tests and scripts |
| `ds4_ft_mlx/deepseek_v4_attention_spec.py` | DeepSeek V4 attention module imports |
| `ds4_ft_mlx/deepseek_v4_checkpoint.py` | Checkpoint loader referenced by tests |
| `ds4_ft_mlx/deepseek_v4_mapping.py` | Tensor name mapping referenced by remap scripts |
| `ds4_ft_mlx/deepseek_v4_moe_spec.py` | MoE module imports |
| `ds4_ft_mlx/lora_targets.py` | `finetune_ds4.py` LoRA allowlist |
| `ds4_ft_mlx/mlx_lm_plugin.py` | MLX-LM fork model registration |
| `ds4_ft_mlx/numpy_real_forward_reference.py` | Parity tests |
| `ds4_ft_mlx/vendor/__init__.py` | Vendor sub-package init |
| `ds4_ft_mlx/vendor/mlx_lm_models/__init__.py` | Vendor model sub-package |
| `sitecustomize.py` | Test environment setup |

**Note:** BA listed 15 files; actual count is 11 (some sub-path discrepancy).
Coder must verify exact untracked count and add all that exist.

**Action:** `git add` all.

### 1B. scripts/ (4 files)

| File | Required by |
|---|---|
| `convert_lora_to_ds4.py` | `tests/test_finetune_ds4.py` line 1077 |
| `fuse_lora_hf.py` | Backlog Stories 5.3/5.4 |
| `make_synth_lora.py` | Backlog synthetic adapter generator |
| `smoke_fuse_serve.sh` | Backlog smoke contracts |

**Action:** `git add` all.

### 1C. tests/ (27 files)

Baseline participants and helpers:

`_fuse_fixture.py`, `ds4_e8m0_ocp_witness.py`, `ds4_f8_e4m3_e8m0_ocp_witness.py`,
`fixtures/deepseek_v4_nn_multilayer_peak_topology.json`,
`helpers/routed_fp4_multilayer_peak_probe.py`,
`test_convert_lora_to_ds4.py`, `test_deepseek_v4_forward_parity.py`,
`test_deepseek_v4_lora_targets.py`, `test_deepseek_v4_nn_interaction_ablation.py`,
`test_deepseek_v4_nn_multilayer_peak.py`,
`test_deepseek_v4_validate_real_mode_relaxation.py`,
`test_ds4_metal_routed_i8_e8m0_isolation.py`,
`test_finetune_ds4_b1_hc_mult_multilayer_readiness.py`,
`test_finetune_ds4_b2_real_checkpoint_payload_readiness.py`,
`test_finetune_ds4_b2_routed_dequant_trusted_reference_readiness.py`,
`test_finetune_ds4_csa_topk_primitive.py`, `test_finetune_ds4_fuse_hf_gate.py`,
`test_finetune_ds4_fused_gate_perf.py`, `test_finetune_ds4_i8_dequant_integration.py`,
`test_finetune_ds4_stateful_decode_readiness.py`, `test_fuse_lora_hf.py`,
`test_make_synth_lora.py`, `test_real_forward_intermediate_dump.py`,
`test_shim_ds4_safetensors.py`, `test_smoke_fuse_serve_ac7_template_relative.sh`

**Anomaly:** `ds4_lora_test` and `test_q4k_dot` are untracked files without
`.py` extension — they appear to be executable scripts or test binaries, not
Python test files. **Quarantine:** do NOT `git add` these without inspecting
content; they are likely scratch/test binaries. Add to ambiguous set (§4).

**Action:** `git add` the 27 files with `.py`/`.json`/`.sh` extensions.
**Quarantine:** `ds4_lora_test` and `test_q4k_dot` — inspect or add to gitignore.

### 1D. docs/adr/ (25 files, includes ADR 0022 numbering conflict)

ADRs 0001 through 0023 plus `README.md`. **BA listed 24 files; actual count
is 25** because two files share the `0022-` prefix:

| ADR 0022 File | Title |
|---|---|
| `0022-key-remap-adapter-fp4-expert-packing-contract.md` | Active key remap contract |
| `0022-strategic-pivot-local-mlx-qlora-parity-marker-retired.md` | Retirement record |

**Adjudication:** Both are legitimate. The filename suffix `-retired` on the
second file distinguishes it as a retirement/decommission record, not an
active decision. **No renumbering required.** Commit both as-is. The two-file
same-number pattern is analogous to "amendment" or "corrigendum" ADRs in
some systems — the name-level disambiguation is sufficient.

Full list of untracked ADRs:
```
docs/adr/0001-metal-graph-is-production-path.md
docs/adr/0002-parity-first-fail-closed-gates.md
docs/adr/0003-isolated-fine-tuning-environments.md
docs/adr/0004-project-local-pi-agent-scaffolding.md
docs/adr/0005-agent-output-is-not-canonical.md
docs/adr/0006-canonical-backlog.md
docs/adr/0007-expert-block-geometry-shape-authoritative.md
docs/adr/0008-two-track-parity-gguf-vs-mlx.md
docs/adr/0009-stateful-decode-readiness-diagnostic.md
docs/adr/0010-b1-hc-mult-multilayer-readiness-diagnostic.md
docs/adr/0011-b2-i8-dequant-realmode-moe-integration-proof.md
docs/adr/0012-b2-i8-dequant-realmode-moe-multilayer-integration-proof.md
docs/adr/0013-b2-i8-dequant-realmode-moe-topk-multi-expert-integration-proof.md
docs/adr/0014-b2-real-checkpoint-payload-readiness-diagnostic.md
docs/adr/0015-b2-routed-dequant-trusted-reference-readiness-diagnostic.md
docs/adr/0016-b2-ds4-cpu-harness-independence-adjudication.md
docs/adr/0017-b2-metal-carry-forward.md
docs/adr/0018-test-purity-snapshot-diff.md
docs/adr/0019-fusion-primary-adapter-serving.md
docs/adr/0020-story-12-3-ac4-hypothesis-retrospective.md
docs/adr/0021-loader-paired-tensor-synthesis-i8-e8m0.md
docs/adr/0022-key-remap-adapter-fp4-expert-packing-contract.md
docs/adr/0022-strategic-pivot-local-mlx-qlora-parity-marker-retired.md
docs/adr/0023-post-fuse-generation-coherence-cross-check.md
docs/adr/README.md
```

Already tracked: ADRs 0024 through 0029.

**Action:** `git add` all 25.

### 1E. docs/ (2 files)

| File | Required by |
|---|---|
| `docs/deepseek-v4-architecture-dossier.md` | Backlog Story 11.1 |
| `docs/deepseek-v4-mtp-policy.md` | Backlog MTP policy references |

**Action:** `git add` both.

### 1F. .pi/ (16 files)

Project-local Pi agent scaffolding referenced by `AGENTS.md` cmux orchestration
section and `ADR 0004-project-local-pi-agent-scaffolding.md` (being committed
in this same slice).

```
.pi/agents/README.md
.pi/agents/project-overlay.md
.pi/agents/bin/check-role.sh
.pi/agents/bin/check-roster.sh
.pi/agents/bin/dispatch-role.sh
.pi/agents/bin/generate-agents-md.sh
.pi/agents/bin/launch-role.sh
.pi/agents/bin/mux-lib.sh
.pi/agents/bin/notify-lib.sh
.pi/agents/bin/notify-watch.sh
.pi/agents/bin/pipeline-lib.sh
.pi/agents/bin/pipeline.sh
.pi/agents/bin/spawn-role-panes.sh
.pi/agents/bin/supervisor-poll.sh
.pi/agents/bin/wait-role-continuous.sh
.pi/agents/bin/wait-role.sh
```

Note: `.pi/agents/pipeline.conf` is already gitignored (per-user personal
settings) per the modified `.gitignore`. It remains excluded.

**Action:** `git add` all 16.

### Commit manifest total: 88 files (11 + 4 + 27 + 25 + 2 + 16 + 3 ambiguous quarantined)

---

## 2. Delete manifest — safe stale file removal (22 files)

These tracked stale files are OUTSIDE the Path A protected directory
(`agent-output/cmux-13-3b/`) and can be safely `git rm --cached` without
affecting any protected evidence.

### 2A. Stale files outside Path A (22 files)

| Category | Count | Examples |
|---|---|---|
| `.dispatch-epoch` in `custom-handoffs/` | 18 | `custom-handoffs/14-3/reviewer-r10.dispatch-epoch`, etc. |
| `.pipeline-slice.v1` in `custom-handoffs/` | 2 | `custom-handoffs/14-3-filtered/.pipeline-slice.v1`, `custom-handoffs/14-3-timeout/.pipeline-slice.v1` |
| `.cmux-status/` markers | 2 | `.cmux-status/ba-14-4.done`, `.cmux-status/coder.done` |

Full list of 22:
```
.cmux-status/ba-14-4.done
.cmux-status/coder.done
custom-handoffs/14-3-filtered/.pipeline-slice.v1
custom-handoffs/14-3-filtered/reviewer-r2.dispatch-epoch
custom-handoffs/14-3-filtered/reviewer.dispatch-epoch
custom-handoffs/14-3-filtered/tester-r2.dispatch-epoch
custom-handoffs/14-3-filtered/tester.dispatch-epoch
custom-handoffs/14-3-timeout/.pipeline-slice.v1
custom-handoffs/14-3-timeout/reviewer-r2.dispatch-epoch
custom-handoffs/14-3-timeout/reviewer-r3.dispatch-epoch
custom-handoffs/14-3-timeout/reviewer-r4.dispatch-epoch
custom-handoffs/14-3-timeout/reviewer.dispatch-epoch
custom-handoffs/14-3-timeout/tester-r2.dispatch-epoch
custom-handoffs/14-3-timeout/tester-r3.dispatch-epoch
custom-handoffs/14-3-timeout/tester-r4.dispatch-epoch
custom-handoffs/14-3-timeout/tester.dispatch-epoch
custom-handoffs/14-3/reviewer-r10.dispatch-epoch
custom-handoffs/14-3/reviewer-r11.dispatch-epoch
custom-handoffs/14-3/reviewer-r12.dispatch-epoch
custom-handoffs/14-3/tester-r10.dispatch-epoch
custom-handoffs/14-3/tester-r11.dispatch-epoch
custom-handoffs/14-3/tester-r12.dispatch-epoch
```

**Action:** `git rm --cached` all 22. Files remain on disk (working tree).
Coder must produce `agent-output/cmux-14-4/deletion-manifest.txt` listing
each path.

### First: unstage BA marker

Before `git rm --cached .cmux-status/ba-14-4.done`, first `git reset HEAD
.cmux-status/ba-14-4.done` to unstage it (it was `git add`'d by BA). Then
`git rm --cached` it.

---

## 3. STOP — Path A stale file conflict (174 files)

### The conflict

174 stale files are INSIDE `agent-output/cmux-13-3b/` (the Path A directory):

| Category | Count |
|---|---|
| `.dispatch-epoch` | 77 |
| `.pid` | 5 |
| `.rc` | 4 |
| `.log` | 87 |
| `.pyc` (force-added `.cpython-313.pyc`) | 1 |
| **Total** | **174** |

Path A is protected by:
- `tests/test_ds4_segmented_loss_and_grad.py::test_path_a_manifest_reproducible_from_index`
  — asserts `git ls-files agent-output/cmux-13-3b | wc -l == 365` and digest
  `7241924d6f9be2eb0df974fdcb1717728c71b6ee77d41dea3fe8a8af55ebb2af`.
- `tests/test_mlx_lm_source.py::test_protected_manifest_helpers_match_architect_baselines`
  — asserts the same count and digest via `compute_story_14_protected_manifest_report'.
- AGENTS.md: "Path A permanent STOP, the 365-file evidence manifest..."

Removing the 174 stale files drops the count from 365 to 191 and changes the
digest, breaking both tests.

The test file `tests/test_ds4_segmented_loss_and_grad.py` is protected at blob
SHA-256 `618a0f22d350ed368e7bf7f782728ad2f6a11486e19a4c97c5e85228aea784c6`.
Modifying it to change the count/digest expectation would change the blob hash,
breaking `TestProtected.test_provider_test` in `test_ds4_segmented_smoke.py`.

### Adjudication: PARTIAL STOP

The 174 Path A stale files CANNOT be `git rm --cached`'d in this slice without:

1. **Operator authorization for a Path A contract amendment** — Path A is
   "permanent STOP" per AGENTS.md; the operator must explicitly authorize the
   count change from 365 to 191.
2. **Amending the protected test** `test_ds4_segmented_loss_and_grad.py` —
   the count/digest expectations must be updated to the new values. This
   changes the blob hash from `618a0f22...` to a new hash.
3. **Cascading protected-hash update** in
   `tests/test_ds4_segmented_smoke.py::TestProtected::test_provider_test` —
   the expected blob hash must change from `618a0f22...` to the new hash.
4. **Amending `test_mlx_lm_source.py`** — the
   `test_protected_manifest_helpers_match_architect_baselines` test references
   the Path A count/digest; it must be updated. This changes the blob hash of
   `test_mlx_lm_source.py` from `dec2c2b5...` to a new hash.
5. **Cascading protected-hash update** in
   `tests/test_ds4_segmented_smoke.py::TestProtected::test_source_sentinel` —
   the expected blob hash must change.
6. **Updating `scripts/finetune_ds4.py::compute_story_14_protected_manifest_report`**
   — the hardcoded count/digest in the report helper must change.

This is 6 cascading changes across protected files. This exceeds the scope of
a single cleanup slice and requires operator authorization for a Path A
contract amendment.

### Recommendation

**Leave the 174 Path A stale files tracked.** Operator can authorize a
separate future slice for Path A cleanup if desired. The gitignore additions
(§5) will prevent NEW stale files from being tracked (the existing tracked
ones can be cleaned up later under separate authorization).

---

## 4. Quarantine — ambiguous or user-owned files

| File | Status | Classification | Action |
|---|---|---|---|
| `context.md` | Untracked | Scratch context note from external LLM session | DO NOT commit, DO NOT delete — add to gitignore (§5) |
| `adapter-converter-implementation-gpt55.md` | Untracked | Scratch from GPT-5.5 session | DO NOT commit, DO NOT delete — add to gitignore (§5) |
| `tests/ds4_lora_test` | Untracked, no `.py` ext | Likely executable binary or script | DO NOT commit — inspect or gitignore |
| `tests/test_q4k_dot` | Untracked, no `.py` ext | Likely executable binary or script | DO NOT commit — inspect or gitignore |

**For `context.md` and `adapter-converter-implementation-gpt55.md`:** Add to
gitignore as `context.md` and `adapter-converter-*.md`. Cannot delete without
operator confirmation.

**For `tests/ds4_lora_test` and `tests/test_q4k_dot`:** Coder must inspect
these files. If they are Python scripts with a shebang, rename to `.py` and
add to commit manifest. If they are compiled binaries, add to gitignore.
Leave in quarantine for now.

---

## 5. Gitignore additions

Build on the current modified `.gitignore` (which already adds
`.cmux-status/` and `.pi/agents/pipeline.conf`).

### New patterns to add

```gitignore
# dispatch / pipeline runtime state
*.dispatch-epoch
*.pid
*.rc
*.log
.pipeline-private/
.pipeline-slice.v1

# agent-output and custom-handoffs are handoff evidence only (AGENTS.md:ADR 0005)
agent-output/
custom-handoffs/

# ambiguous scratch files
context.md
adapter-converter-*.md
```

### Adjudication: full gitignore for `agent-output/` and `custom-handoffs/`

**Decision: YES — add `agent-output/` and `custom-handoffs/` to gitignore.**

Rationale:
1. AGENTS.md and ADR 0005 (being committed this slice) state
   "agent-output/ is handoff evidence only, not canonical docs."
2. `custom-handoffs/` is similarly transient handoff artifact storage.
3. Gitignoring these directories prevents accidental commits of 1,400+
   untracked files while keeping already-tracked files visible.
4. Already-tracked files in `agent-output/` (Path A 365 files, 14-3 smoke
   evidence, 13-1 coder notes) remain tracked — gitignore does not untrack
   existing tracked files.
5. New evidence files that must be tracked in future slices are added with
   `git add -f` (same as the existing pattern for the `.pyc` file and
   `.cmux-status/` markers).

The Coder may need to use `git add -f` for the deletion manifest and commit
manifest output files in `agent-output/cmux-14-4/`.

### `tests/ds4_lora_test` and `tests/test_q4k_dot`

If these are binaries, add:
```gitignore
tests/ds4_lora_test
tests/test_q4k_dot
```

If they are Python scripts, Codershould rename and track them instead.

---

## 6. Vendor pin resolution (CRITICAL)

### Current state

- `vendor/mlx-lm` outer gitlink: `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`
  (mode `160000`)
- Inner HEAD: `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`
- Inner branch: **detached HEAD** (no branch checked out)
- Inner staged: 2 files, 2,627 insertions / 14 deletions
  - `mlx_lm/tuner/trainer.py` (+240 / -14) — Story 14.1 trainer seam
  - `tests/test_tuner_trainer.py` (+2401) — Story 14.1/14.2a test suite

### Commit sequence (exact order)

**Step 1: Commit inner repo (detached HEAD is safe for submodules)**

```bash
cd vendor/mlx-lm
git commit -m "Story 14.1/14.2a: trainer loss_and_grad seam and test suite

Append optional loss_and_grad=None argument after training_callback
in train(). When provided, train() delegates the forward/backward
transform to the caller-supplied function instead of using the
default loss. Preserve all existing positional/keyword callers.

Add comprehensive test suite (2401 lines) covering:
- Default path (loss_and_grad=None) unchanged
- Custom provider path delegates correctly
- Host finite gate preserves nonfinite rejection
- Random rollback for compiled trainer phase 2
- Schema validation for provider output

This is the Story 14.1/14.2a trainer seam, not a generic upstream
patch. The outer repo gitlink will be updated to pin this commit."
NEW_INNER_HASH=$(git rev-parse HEAD)
cd ..
```

Detached HEAD is acceptable: the outer gitlink pins the exact commit hash,
not a branch. `git submodule update --init` will resolve to the gitlink
hash regardless of which branch it's on (or no branch at all).

**Step 2: Update outer gitlink**

```bash
git add vendor/mlx-lm
```

This stages the new gitlink hash in the outer repository index.

**Step 3: Update pin references in production source**

`scripts/finetune_ds4.py` line 119:
```python
MLX_LM_FORK_SHA = "15b522f593b7ca5fbc0cac6f7572d40859d2d8fe"
```
→ change to:
```python
MLX_LM_FORK_SHA = "<NEW_INNER_HASH>"
```

`tests/test_mlx_lm_source.py` line 50:
```python
FORK_SHA = "15b522f593b7ca5fbc0cac6f7572d40859d2d8fe"
```
→ change to:
```python
FORK_SHA = "<NEW_INNER_HASH>"
```

**Step 4: Update docs references**

`docs/architecture.md` line 69: change `15b522f...` to `<NEW_INNER_HASH>`.

`docs/backlog.md` — 7 occurrences of `15b522f...` at lines 4437, 4456, 4464,
4577, 4637, 4717, 4788. All must be updated to `<NEW_INNER_HASH>`.

`docs/technical-spec.md` — references `15b522f...` in the §6.4 MLX-LM source
selection section (added by the modified technical-spec). Must be updated.

### Cascading protected-hash amendments

Updating `FORK_SHA` in `tests/test_mlx_lm_source.py` changes its blob hash
from `dec2c2b5a7644540a7ec339712968593647abd64c70b821abddc97f9b6ff5d65`
to a new hash. This is a protected blob, but the update is explicitly
required by R14.4-5 (vendor pin resolution) and supersedes the R14.4-2
forbidden-list protection for this narrow constant change.

Cascading changes required:

| File | Change | Rationale |
|---|---|---|
| `tests/test_mlx_lm_source.py` | `FORK_SHA` constant → new hash | R14.4-5 authorizes |
| `scripts/finetune_ds4.py` | `MLX_LM_FORK_SHA` → new hash | R14.4-5 authorizes |
| `tests/test_ds4_segmented_smoke.py::TestProtected::test_source_sentinel` | Expected blob hash → new blob hash of `test_mlx_lm_source.py` | Protected-hash cascade — must update to match new file content |
| `docs/architecture.md` | `15b522f...` → new hash | Durable pin reference |
| `docs/backlog.md` | 7 `15b522f...` references → new hash | Historical pin references |
| `docs/technical-spec.md` | `15b522f...` reference → new hash | §6.4 pin reference |
| `ADR 0029` | Add amendment section recording the pin advance | Consistency |

**Note on `test_mlx_lm_source.py` blob hash:** The `TestProtected` test
class in `tests/test_ds4_segmented_smoke.py` checks:
```python
def test_source_sentinel(self):
    p = ROOT/"tests"/"test_mlx_lm_source.py"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == "dec2c2b5..."
```

After changing `FORK_SHA`, this assertion must be updated to the new blob
hash. Coder computes the new hash after editing: `python3 -c "import
hashlib; print(hashlib.sha256(open('tests/test_mlx_lm_source.py','rb').read()).hexdigest())"`

**Note on semantic source sentinel:** `SEMANTIC_MLX_SRC_COUNT = 20` and
`SEMANTIC_MLX_SRC_SHA256 = "58c588df..."` are computed from the FILESYSTEM
walk of `python-envs/mlx/src/`, not from git. Since committing the 11
untracked files does not change the filesystem, these values remain
unchanged and the tests pass. No cascade needed for semantic constants.

### ADR 0029 amendment

ADR 0029 currently records the pin at `15b522f...` as "Accepted for Story
14.0 bootstrap." Add a "## Amendment (Story 14.4)" section:

> The vendor inner pin advanced from `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`
> to `<NEW_HASH>` to include the Story 14.1/14.2a trainer seam
> (`loss_and_grad` passthrough) and test suite. The outer gitlink was updated
> to match. This is an authorized advancement of the submodule pin, not a
> source or identity change: the fork URL remains
> `git@github.com:Deviad/mlx-lm.git`, no branch-following is added, and the
> fork-identity verification (import path, PEP 610 metadata, module-level
> hash) remains the verification mechanism.

### No new ADR required

The pin advancement is an amendment to ADR 0029, not a new architectural
decision. ADR 0029's "Decision" section already says "pin the submodule"
without fixing the SHA for all time — the pin advancement is within scope.

---

## 7. Protected evidence preservation matrix

| Protected artifact | Check | This slice touches? |
|---|---|---|
| `segmented_loss_and_grad.py` | blob SHA-256 `205721...` | **No** — not edited |
| `test_ds4_segmented_loss_and_grad.py` | blob SHA-256 `618a0f...` | **No** — Path A stale cleanup is STOPPED (§3) |
| `test_mlx_lm_source.py` | blob SHA-256 `dec2c2b5...` | **YES** — `FORK_SHA` updated; new blob hash must be computed and recorded |
| `test_ds4_segmented_smoke.py::TestProtected` | Protects the sentinel hash | **YES** — `test_source_sentinel` expected value must be updated to the new blob hash of `test_mlx_lm_source.py` |
| Path A 365-file digest | `7241924d...` at count 365 | **No** — Path A stale cleanup STOPPED; count and digest preserved |
| ADR 0028 | blob SHA-256 | **No** — not edited |
| ADR 0029 | existing content | **EDIT** — amendment section added for pin advancement |
| `vendor/mlx-lm` inner HEAD | Was `15b522f...` | **YES** — advances to `<NEW_HASH>` |
| `vendor/mlx-lm` outer gitlink | Was `15b522f...` | **YES** — advances to `<NEW_HASH>` |
| Epic 14.3b smoke evidence | `agent-output/cmux-14-3/smoke-report.json`, `smoke-log.txt`, `filtered-dataset-provenance.json` | **No** — remains tracked |
| `agent-output/cmux-13-1/coder-notes.md` | tracked, modified | **No** — stays tracked, modifications kept |

---

## 8. Serial mutation order

All mutations are `git add` / `git rm --cached` / file edits — NO `git commit`,
NO `git push`. Everything stays staged in the working tree.

### Phase A: Unstage BA artifacts that should not be tracked

```
A1. git reset HEAD .cmux-status/ba-14-4.done
    (rollback: git add .cmux-status/ba-14-4.done)
```

### Phase B: `git rm --cached` safe stale files (22 files outside Path A)

```
B1. git rm --cached .cmux-status/ba-14-4.done
B2. git rm --cached .cmux-status/coder.done
B3. git rm --cached custom-handoffs/14-3-filtered/.pipeline-slice.v1
B4. git rm --cached custom-handoffs/14-3-timeout/.pipeline-slice.v1
B5-B22. git rm --cached <each dispatch-epoch in custom-handoffs/*/>
    (rollback: git checkout -- <path> to restore from HEAD)
```

### Phase C: `git add` untracked implementation files (88 files)

```
C1. git add python-envs/mlx/src/
C2. git add scripts/convert_lora_to_ds4.py scripts/fuse_lora_hf.py scripts/make_synth_lora.py scripts/smoke_fuse_serve.sh
C3. git add tests/_fuse_fixture.py tests/ds4_e8m0_ocp_witness.py tests/ds4_f8_e4m3_e8m0_ocp_witness.py tests/fixtures/ tests/helpers/ tests/test_convert_lora_to_ds4.py tests/test_deepseek_v4_forward_parity.py tests/test_deepseek_v4_lora_targets.py tests/test_deepseek_v4_nn_interaction_ablation.py tests/test_deepseek_v4_nn_multilayer_peak.py tests/test_deepseek_v4_validate_real_mode_relaxation.py tests/test_ds4_metal_routed_i8_e8m0_isolation.py tests/test_finetune_ds4_b1_hc_mult_multilayer_readiness.py tests/test_finetune_ds4_b2_real_checkpoint_payload_readiness.py tests/test_finetune_ds4_b2_routed_dequant_trusted_reference_readiness.py tests/test_finetune_ds4_csa_topk_primitive.py tests/test_finetune_ds4_fuse_hf_gate.py tests/test_finetune_ds4_fused_gate_perf.py tests/test_finetune_ds4_i8_dequant_integration.py tests/test_finetune_ds4_stateful_decode_readiness.py tests/test_fuse_lora_hf.py tests/test_make_synth_lora.py tests/test_real_forward_intermediate_dump.py tests/test_shim_ds4_safetensors.py tests/test_smoke_fuse_serve_ac7_template_relative.sh
C4. git add docs/adr/0001-metal-graph-is-production-path.md docs/adr/0002-parity-first-fail-closed-gates.md docs/adr/0003-isolated-fine-tuning-environments.md docs/adr/0004-project-local-pi-agent-scaffolding.md docs/adr/0005-agent-output-is-not-canonical.md docs/adr/0006-canonical-backlog.md docs/adr/0007-expert-block-geometry-shape-authoritative.md docs/adr/0008-two-track-parity-gguf-vs-mlx.md docs/adr/0009-stateful-decode-readiness-diagnostic.md docs/adr/0010-b1-hc-mult-multilayer-readiness-diagnostic.md docs/adr/0011-b2-i8-dequant-realmode-moe-integration-proof.md docs/adr/0012-b2-i8-dequant-realmode-moe-multilayer-integration-proof.md docs/adr/0013-b2-i8-dequant-realmode-moe-topk-multi-expert-integration-proof.md docs/adr/0014-b2-real-checkpoint-payload-readiness-diagnostic.md docs/adr/0015-b2-routed-dequant-trusted-reference-readiness-diagnostic.md docs/adr/0016-b2-ds4-cpu-harness-independence-adjudication.md docs/adr/0017-b2-metal-carry-forward.md docs/adr/0018-test-purity-snapshot-diff.md docs/adr/0019-fusion-primary-adapter-serving.md docs/adr/0020-story-12-3-ac4-hypothesis-retrospective.md docs/adr/0021-loader-paired-tensor-synthesis-i8-e8m0.md docs/adr/0022-key-remap-adapter-fp4-expert-packing-contract.md docs/adr/0022-strategic-pivot-local-mlx-qlora-parity-marker-retired.md docs/adr/0023-post-fuse-generation-coherence-cross-check.md docs/adr/README.md
C5. git add docs/deepseek-v4-architecture-dossier.md docs/deepseek-v4-mtp-policy.md
C6. git add .pi/
    (rollback: git reset HEAD <paths> to unstage)
```

### Phase D: Vendor inner commit + outer gitlink update

```
D1. cd vendor/mlx-lm && git commit -m "..." (see §6 Step 1)
D2. cd .. && git add vendor/mlx-lm
    (rollback: cd vendor/mlx-lm && git reset --soft HEAD~1 to undo inner commit;
     git checkout vendor/mlx-lm in outer to restore old gitlink)
```

### Phase E: Pin reference updates in production source

```
E1. Edit scripts/finetune_ds4.py line 119: MLX_LM_FORK_SHA → <NEW_HASH>
E2. Edit tests/test_mlx_lm_source.py line 50: FORK_SHA → <NEW_HASH>
E3. Compute new blob hash of test_mlx_lm_source.py
E4. Edit tests/test_ds4_segmented_smoke.py TestProtected.test_source_sentinel: expected hash → new blob hash
E5. git add scripts/finetune_ds4.py tests/test_mlx_lm_source.py tests/test_ds4_segmented_smoke.py
    (rollback: git checkout -- <files> to restore from HEAD)
```

### Phase F: Pin reference updates in docs

```
F1. Edit docs/architecture.md line 69: 15b522f... → <NEW_HASH>
F2. Edit docs/backlog.md: 7 occurrences of 15b522f... → <NEW_HASH>
F3. Edit docs/technical-spec.md: 15b522f... reference → <NEW_HASH>
F4. Edit docs/adr/0029-pinned-mlx-lm-source-selection.md: add amendment section
F5. git add docs/architecture.md docs/backlog.md docs/technical-spec.md docs/adr/0029-pinned-mlx-lm-source-selection.md
    (rollback: git checkout -- <files>)
```

### Phase G: Gitignore additions

```
G1. Edit .gitignore: add patterns from §5
G2. git add .gitignore
    (rollback: git checkout -- .gitignore)
```

### Phase H: Stage modified tracked files with legitimate changes

```
H1. git add AGENTS.md (legitimate modification from prior slices)
H2. git add agent-output/cmux-13-1/coder-notes.md (legitimate modification)
H3. git add docs/technical-spec.md (already staged in F3 or re-stage)
    (rollback: git reset HEAD <files>)
```

### Phase I: Produce manifests

```
I1. mkdir -p agent-output/cmux-14-4/
I2. Create agent-output/cmux-14-4/commit-manifest.txt (all git add'd files)
I3. Create agent-output/cmux-14-4/deletion-manifest.txt (all git rm --cached files)
I4. git add -f agent-output/cmux-14-4/commit-manifest.txt agent-output/cmux-14-4/deletion-manifest.txt
    (uses -f because agent-output/ is now gitignored)
```

### Final state criterion

After all phases:
- `git status --porcelain` shows modified/staged/untracked but no new commit
- `git log --oneline -1` shows `af08c25` (no new commit)
- `git diff --cached` shows all staged changes
- Working tree files remain on disk (no deletions from filesystem)

---

## 9. Test and verification strategy

### 9A. Pre-execution baseline

Before any mutations, Test Manager records:
1. `git ls-files | wc -l` → 670
2. `git ls-files --others --exclude-standard | wc -l` → 1,610
3. Protected hash checks (all four):
   ```bash
   python3 -c "import hashlib, pathlib; [...see requirements verification commands...]"
   ```
4. Path A: `git ls-files agent-output/cmux-13-3b | wc -l` → 365
5. Path A digest: `7241924d6f9be2eb0df974fdcb1717728c71b6ee77d41dea3fe8a8af55ebb2af`

### 9B. Post-execution verification

After all phases complete:

1. **Untracked implementation files are tracked:**
   ```bash
   git ls-files python-envs/mlx/src/ | grep __init__
   git ls-files scripts/convert_lora_to_ds4.py
   git ls-files tests/test_convert_lora_to_ds4.py
   git ls-files docs/adr/0001-metal-graph-is-production-path.md
   git ls-files .pi/agents/README.md
   ```

2. **Stale files are untracked (outside Path A):**
   ```bash
   git ls-files '.cmux-status/*'  # → empty
   git ls-files 'custom-handoffs/*/*.dispatch-epoch'  # → empty
   git ls-files 'custom-handoffs/*/.pipeline-slice.v1'  # → empty
   ```

3. **Path A is preserved:**
   ```bash
   git ls-files agent-output/cmux-13-3b/ | wc -l  # → still 365
   ```

4. **Vendor pin updated:**
   ```bash
   git ls-files --stage vendor/mlx-lm  # → mode 160000 at NEW_HASH
   cd vendor/mlx-lm && git rev-parse HEAD  # → NEW_HASH
   ```

5. **Protected hashes (updated as needed):**
   ```bash
   # Provider source unchanged
   python3 -c "import hashlib,pathlib; h=hashlib.sha256(pathlib.Path('python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py').read_bytes()).hexdigest(); assert h=='20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518', h; print('OK')"
   # Provider test unchanged (Path A not touched)
   python3 -c "import hashlib,pathlib; h=hashlib.sha256(pathlib.Path('tests/test_ds4_segmented_loss_and_grad.py').read_bytes()).hexdigest(); assert h=='618a0f22d350ed368e7bf7f782728ad2f6a11486e19a4c97c5e85228aea784c6', h; print('OK')"
   # Source sentinel changed — new hash must match what TestProtected expects
   python3 -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('tests/test_mlx_lm_source.py').read_bytes()).hexdigest())"
   ```

6. **Exact-fork pytest suite:**
   ```bash
   PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" \
     PYTHONDONTWRITEBYTECODE=1 \
     python-envs/mlx/.venv/bin/python -m pytest -q \
     tests/test_ds4_segmented_smoke.py tests/test_finetune_ds4.py \
     tests/test_mlx_lm_source.py tests/test_ds4_segmented_loss_and_grad.py \
     tests/test_ds4_gguf_base_smoke.py
   ```
   Expected: 246 passed, 3 skipped, 1 warning (same as 14.3b r4 baseline).
   **Note:** test_mlx_lm_source.py will have changed (FORK_SHA), so the
   passed count may change — the Coder must fix any test expectations that
   reference the old fork SHA in assertions beyond the `FORK_SHA` constant.

### 9C. Fresh-clone reproducibility check

```bash
# Simulate fresh clone in a temp directory
TEMP=$(mktemp -d)
git clone --recurse-submodules . "$TEMP/fresh-clone"
cd "$TEMP/fresh-clone"
git submodule update --init --recursive

# Verify inner vendor has the trainer seam
grep -c "loss_and_grad" vendor/mlx-lm/mlx_lm/tuner/trainer.py  # → >0

# Verify all Epic 14 imports resolve
PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" \
  python -c "import ds4_ft_mlx; from scripts import finetune_ds4; print('OK')"

# Run the exact-fork pytest suite
PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" \
  python -m pytest -q \
  tests/test_ds4_segmented_smoke.py tests/test_finetune_ds4.py \
  tests/test_mlx_lm_source.py tests/test_ds4_segmented_loss_and_grad.py
```

### 9D. Tracking hygiene verification

Per AGENTS.md HARD RULE, for every test file cited in the 14.3b Test Manager
r4 verdict:

```bash
for f in tests/test_ds4_segmented_smoke.py tests/test_finetune_ds4.py \
         tests/test_mlx_lm_source.py tests/test_ds4_segmented_loss_and_grad.py \
         tests/test_ds4_gguf_base_smoke.py; do
  git ls-files -- "$f" || echo "UNTRACKED: $f"
done
```

All must return the file path (tracked).

---

## 10. STOP-ESCALATE conditions

| # | Condition | Triggered? |
|---|---|---|
| 1 | Orphan untracked implementation file (no consumer) | **No** — all 88 files have verified import/reference chains |
| 2 | Tracked stale file referenced by current source/tests (not actually stale) | **No** — all 22 are dispatch-epoch/pipeline-slice/cmux-status runtime state |
| 3 | Vendor inner cannot be committed (detached HEAD, upstream rejected) | **No** — detached HEAD is safe for submodule commits; no upstream push needed |
| 4 | Deleting tracked file removes evidence referenced by verdict/digest | **YES — Path A (§3): 174 stale files inside `agent-output/cmux-13-3b/` are protected by the 365-file manifest. PARTIAL STOP — operator authorization required for Path A contract amendment** |
| 5 | Untracked ADR contradicts tracked ADR (same number, different content) | **No** — ADRs 0001-0023 are all untracked; two ADR 0022 files co-exist with distinguishable titles; no tracked ADR is contradicted |
| 6 | Scope too large for single slice verification | **No** — 88 commits + 22 deletes + vendor pin is manageable |
| 7 | `agent-output/` and `custom-handoffs/` gitignore decision unresolved | **Resolved — full gitignore (§5)** |

**Severity:** One PARTIAL STOP (Path A stale files). Everything else is
unambiguous and the Coder can proceed.

---

## 11. Risks and caveats

1. **`test_mlx_lm_source.py` is modified but protected.** The cascading
   protected-hash update (`dec2c2b5...` → new hash) must be computed exactly.
   If the Coder miscomputes and records the wrong hash, `TestProtected` fails.
   The Coder should compute the hash programmatically immediately after
   editing, not manually.

2. **`test_mlx_lm_source.py` has 5 references to `FORK_SHA` beyond the
   constant definition.** Lines 84, 280, 283, 328-329, 341 reference it
   via the constant, so updating the constant at line 50 suffices. But if
   any test line hardcodes `15b522f...` instead of `FORK_SHA`, it must be
   found and updated. The grep shows all references go through the
   `FORK_SHA` variable.

3. **Inner commit on detached HEAD has no branch.** This is safe for
   submodules: the outer gitlink pins the exact commit. But if the operator
   later does `git submodule update --init` AND the inner repo's reflog
   is empty (e.g., fresh clone), the commit is reachable only via the
   gitlink. No risk for this design; noted for completeness.

4. **`docs/backlog.md` has historical evidence at lines 4456, 4577, 4637
   that records "inner HEAD and outer gitlink remain `15b522f...`".** After
   the pin update, these become historical statements that no longer match
   the current state. The Coder should annotate them: e.g., "was
   `15b522f...` before Story 14.4 pin advancement to `<NEW_HASH>`" rather
   than silently overwriting. This preserves chain-of-custody transparency.

5. **`agent-output/` gitignore uses `-f` for future evidence.** After this
   slice, future slices that need to track `agent-output/` evidence (like
   the 14.3 slice did for `smoke-report.json`) must use `git add -f
   agent-output/...`. This is the same pattern used for `.pyc` files. It
   is slightly inconvenient but prevents accidental commits of 1,400+
   ephemeral files.

6. **Tests `ds4_lora_test` and `test_q4k_dot` remain untracked.** If they
   are Python scripts, they must be renamed with `.py` extension and added.
   If they are binaries, gitignore them. Coder must inspect.

---

## 12. Coder Phase 1 can proceed now

The design is complete. The single PARTIAL STOP (Path A stale files, §3)
does not block any other phase. Coder should:

1. Execute Phase A through I in serial order (§8).
2. Produce commit-manifest.txt and deletion-manifest.txt.
3. Verify with the full pytest suite (§9B).
4. Verify fresh-clone reproducibility (§9C).
5. Verify tracking hygiene (§9D).
6. `git add` all modified/new test files (AGENTS.md HARD RULE).
7. No `git commit`, no `git push`.

Reviewer must independently verify:
- Every file in the commit manifest is tracked (`git ls-files`).
- No file in the delete manifest is still tracked.
- Path A count and digest are unchanged (365 / `7241924d...`).
- Vendor gitlink matches the inner HEAD.
- All pin references in docs are updated to the new hash.
- Protected hashes are updated as specified.

Test Manager must run the full suite and record the result.