# Story 14.4 — Independent cleanup/reproducibility review

## Verdict: BLOCKED

Exact staged state unsafe and not fresh-clone reproducible.

Reviewed:

- `custom-handoffs/14-4-repo-hygiene/task-reviewer.md`
- `custom-handoffs/14-4-repo-hygiene/requirements.md`
- `custom-handoffs/14-4-repo-hygiene/architecture.md`
- `custom-handoffs/14-4-repo-hygiene/coder-notes.md`
- outer index/worktree, manifests, ignore rules, protected evidence, inner vendor commit, remote reachability, focused baseline

## Blocking findings

### B1 — Generated Mach-O binary staged for commit

`tests/ds4_lora_test`:

- Mach-O 64-bit arm64 executable
- 88,240 bytes
- staged as mode `100755`, status `A`
- simultaneously matched by staged `.gitignore`
- listed under commit-manifest “Staged additions”

Coder notes claim both compiled test binaries were quarantined. False for `tests/ds4_lora_test`; only `tests/test_q4k_dot` remains ignored/untracked.

Required fix:

1. Remove `tests/ds4_lora_test` from index and commit manifest.
2. Keep local binary ignored or delete only under operator-approved generated-artifact cleanup.
3. Re-scan every staged addition for binary/generated content.

Independent staged-addition scan found this sole NUL-bearing staged artifact.

### B2 — New vendor commit unreachable from fresh clone

Observed:

- outer `HEAD`: `af08c255f70566e54c39b2b7e6a644266548a737`
- outer index gitlink: `80fab4e419a57f9465bb9e2f4e90010d645e124c`
- inner `HEAD`: `80fab4e419a57f9465bb9e2f4e90010d645e124c`
- inner parent: `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`
- inner worktree: clean
- inner branch: detached
- `git ls-remote origin`: no advertised ref at `80fab4e...`
- `git branch -r --contains 80fab4e...`: none

Two independent failures:

1. `git clone .` clones committed outer `HEAD`, not staged index; staged 87 additions and gitlink update disappear.
2. Even after outer commit, standard `git submodule update --init --recursive` cannot obtain locally-created `80fab4e...` from `git@github.com:Deviad/mlx-lm.git` until commit becomes remotely reachable.

No-push rule conflicts with fresh-clone gate. Reviewer cannot PASS this state.

Required fix: operator adjudication. Either authorize publishing inner commit to advertised fork ref before outer gitlink commit, or adopt another standard fresh-clone-reachable provenance design. Then prove clone/submodule initialization from remote source, not existing local object database.

### B3 — Remaining untracked inventory not cleaned or deliberately quarantined

Current `git ls-files --others --exclude-standard` count: **119**, not requested/coder-reported 118.

Sorted newline-terminated inventory SHA-256:

`20e6a6a2ae908f76ed05f1ea17a02288c921dccc69ffccb86b9f6d0a72758d68`

Exhaustive partition verified disjoint and complete:

| Classification | Count | Inventory SHA-256 | Disposition |
|---|---:|---|---|
| Required current source/docs | 5 | `4e25339a6aa5cfc98b0c0c52f318eae671a34ca0b5405454cd4fd1104954bda0` | Track, or repair canonical references with explicit replacement |
| Generated binary | 1 | `6980c904f2fb13c33970834c02f079ea2124316cc433381fe9c0ec8b5fab4f38` | Ignore/remove as generated |
| Needs operator/architecture classification | 2 | `04237f716b692840bc0c32113083c0bc8233101ba3d2810ec5d76d5db87093f3` | Do not delete silently |
| Historical root handoff/review scratch | 111 | `ad56145efa73afee1079f0e87089243868f9ae0295d62c0ae53fe566e8ed173a` | Deliberately quarantine; do not leave unexplained |

#### Required current source/docs — all 5

- `python-envs/torch/pyproject.toml` — explicitly required by `AGENTS.md`, ADR 0003, backlog environment contract.
- `python-envs/torch/src/ds4_ft_torch/__init__.py` — package source for declared Torch environment.
- `progress.md` — named by `docs/architecture.md` as repository/current-state document.
- `training-next-status.md` — named by `docs/architecture.md`, ADR README, backlog, technical spec.
- `training-backend-bakeoff-gpt55.md` — cited by staged ADR 0022 as decision evidence.

Fresh clone omits all five. This directly violates untracked implementation/docs and reproducibility gates.

#### Generated binary — all 1

- `ds4_agent_test` — Mach-O 64-bit arm64, 1,294,688 bytes, Makefile output, removed by `make clean`, not ignored by current `.gitignore`.

Add root `/ds4_agent_test` ignore rule; never commit binary.

#### Needs classification — all 2

- `python-envs/legacy-trans/pyproject.toml` — source-like legacy environment definition; no current canonical consumer beyond protected historical log. Quarantine/delete only after explicit adjudication.
- `python-envs/mlx/uv.lock` — 363,958-byte dependency lock. Decide whether lock becomes tracked reproducibility input or generated/local artifact. Do not leave unclassified.

#### Historical root scratch — all 111

Exact set: every remaining untracked path after subtracting eight explicit paths above. Predicate verified: all 111 are root-level `*.md` files. Content/header scan classifies them as historical review, plan, scout, progress-support, or training handoff artifacts (`next-internal-lora-*`, `review-*`, `shim-review-gpt55.md`, `training-local-gates-gpt55.md`, etc.). Tracked-reference scan found no semantic current-source dependency after excluding incidental filenames embedded in protected baseline logs and the three required documents listed above.

These files remain visible and unclassified. Coder quarantined only `context.md` and `adapter-converter-implementation-gpt55.md`; cleanup scope therefore incomplete. Preserve user material. Move under an operator-approved ignored handoff/archive location or add deliberate narrowly-scoped quarantine rules plus manifest. No deletion without confirmation.

### B4 — `git diff --cached --check` fails; coder evidence false

Independent result: exit `2`.

```text
docs/adr/0018-test-purity-snapshot-diff.md:154: trailing whitespace.
+AGENTS.md).
docs/adr/0019-fusion-primary-adapter-serving.md:134: trailing whitespace.
+  marker. `.ds4-gguf-generate-ok` stays scoped to base-only generation;
```

Coder notes claim exit `0`. Fix both staged lines, rerun, record actual exit.

### B5 — Commit manifest incomplete, unsorted, and internally unsafe

Actual outer index with rename detection disabled:

- additions: 87
- modifications: 11
- deletions: 21
- total changed paths: 119

Commit manifest:

- declares 85 additions
- omits its own two staged additions:
  - `agent-output/cmux-14-4/commit-manifest.txt`
  - `agent-output/cmux-14-4/deletion-manifest.txt`
- contains no required per-path deletion section
- includes forbidden generated binary `tests/ds4_lora_test`
- additions list unsorted; first inversion: `.pi/agents/project-overlay.md` before `.pi/agents/README.md`
- modifications list unsorted; first inversion: `agent-output/cmux-13-1/coder-notes.md` before `AGENTS.md`

Deletion manifest itself passes: 21 unique sorted paths, exactly matching staged deletions. Architecture expected 22 because it included `.cmux-status/ba-14-4.done`; that path was only an index addition and disappeared after reset, so no HEAD deletion exists. Final manifests must explicitly reconcile 21 versus planned 22.

R14.4-6 requires complete sorted additions/removals and manifest linkage. Current commit manifest fails.

### B6 — Required current slice handoffs hidden and untracked

Tracking check:

- `custom-handoffs/14-4-repo-hygiene/requirements.md`: tracked
- `custom-handoffs/14-4-repo-hygiene/architecture.md`: untracked/ignored
- `custom-handoffs/14-4-repo-hygiene/coder-notes.md`: untracked/ignored
- `custom-handoffs/14-4-repo-hygiene/task-reviewer.md`: untracked/ignored

Full `custom-handoffs/` ignore is acceptable only with reliable `git add -f` for required verdict/handoff evidence. Current execution did not do that. Fresh clone loses architecture and coder chain-of-custody documents. Force-add required slice evidence or narrow policy; update manifest.

### B7 — Canonical Story 14.4 backlog state stale and contradictory

`docs/backlog.md:4813` still says:

`Status: [ BA REQUIREMENTS — classification only; no deletions/commit/push ]`

Same section still claims:

- only ~68 implementation files
- ~197 stale files will be removed
- only 2 ambiguous files
- current outer/inner pin remains `15b522f...` with staged inner changes
- gitignore gaps remain

Actual state: inner commit exists, outer index pin advanced, 21 non-Path-A stale paths removed, ignore rules added, 119 untracked paths remain, and 174 Path A files intentionally preserved. Update canonical backlog to truthful execution/review state without rewriting historical evidence.

### B8 — Coder notes contain contradictory commit claim

Coder correctly records required inner commit `80fab4e...`, then verification table states “No commit, no push.” Exact truthful statement: no **outer** commit and no push; one required **inner** commit occurred. Requirements/architecture also conflict: R14.4-5/Phase D require inner commit while other text says no commit. Reconcile contract wording before final verdict.

## Independent checks that passed

### Local functional baseline

Exact command scope from Story 14.3b Test Manager verdict:

```text
246 passed, 3 skipped, 1 warning, 2 subtests passed in 21.74s
exit 0
```

All five verdict test files tracked:

- `tests/test_ds4_segmented_smoke.py`
- `tests/test_finetune_ds4.py`
- `tests/test_mlx_lm_source.py`
- `tests/test_ds4_segmented_loss_and_grad.py`
- `tests/test_ds4_gguf_base_smoke.py`

Verdict remains **local only**; fresh-clone reproducibility fails under B2/B3/B6.

### Protected invariants

- Path A filesystem count: 365
- Path A tracked count: 365
- Path A digest: `7241924d6f9be2eb0df974fdcb1717728c71b6ee77d41dea3fe8a8af55ebb2af`
- Path A untracked/missing delta: none
- provider source SHA-256: `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518`
- provider test SHA-256: `618a0f22d350ed368e7bf7f782728ad2f6a11486e19a4c97c5e85228aea784c6`
- updated source test SHA-256: `24325ef35e915b1cc3275d5fac31400c116e1adda6da0254b6158c7a6233dda9`
- source sentinel references exact updated hash
- ADR 0028 SHA-256: `0aa743b731e6393ca84e91bcd3d32bec655177779ee15e21a47ade19e322300b`
- Story 14.3 smoke report/log/provenance: tracked and byte-unchanged versus `HEAD`

### Pin and inner scope

- outer staged gitlink equals inner `HEAD`: `80fab4e...`
- inner worktree clean
- inner commit changes exactly:
  - `mlx_lm/tuner/trainer.py`
  - `tests/test_tuner_trainer.py`
- inner commit `git show --check`: clean
- production/source pin constants and sentinel cascade agree locally
- historical old-pin references mostly preserved with annotations

### Deletion scope

- 21 staged deletions exactly match sorted deletion manifest
- all deleted paths runtime markers/dispatch epochs/pipeline state outside protected Path A
- files preserved on disk

## Exact return-to-Coder gate

1. Unstage `tests/ds4_lora_test`; keep generated binary ignored.
2. Add `/ds4_agent_test` ignore rule.
3. Track five required source/docs paths, or update canonical references with reviewed replacement.
4. Deliberately quarantine all 111 root scratch Markdown files; do not delete user material.
5. Adjudicate `python-envs/legacy-trans/pyproject.toml` and `python-envs/mlx/uv.lock`.
6. Force-add required current-slice handoffs/evidence despite directory ignore.
7. Resolve remote reachability for `80fab4e...`; demonstrate standard fresh clone + recursive submodule init from reachable source. No push without operator authorization.
8. Fix both whitespace errors.
9. Rebuild complete sorted commit/deletion manifests from actual index; remove generated binary; reconcile 21-versus-22 plan delta.
10. Update Story 14.4 canonical backlog/status and coder notes to actual state.
11. Rerun protected checks, tracked-verdict checks, exact 246-test suite, and true fresh-clone gate.

PASS prohibited until all blockers resolved and independently re-reviewed.
