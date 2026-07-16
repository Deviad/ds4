# Story 14.4 — Reviewer final Commit 1 gate

## Verdict: PASS

Exact staged Commit 1 candidate approved.

## Candidate identity

- Reviewed staged tree: `cb7560973525a9445b972592d8e7b79a66ef86c2`
- Cached binary-diff SHA-256: `097c7cc24541207e95bccaa3e6e43b34cf9b0d23065cc74110d72f0b0a654a3d`
- Cached name-status SHA-256: `dedc982613ec40d654fea365d66a9efdd2b796a2e7c3d0917d0519c29357037e`
- Vendor gitlink: `80fab4e419a57f9465bb9e2f4e90010d645e124c`
- Reviewed at: `2026-07-16T19:10:09Z`
- Signed-off-by: `xhigh-reviewer`

## Portable-wrapper gate

PASS.

- Exactly 14 `.pi/agents/bin/*.sh` wrappers.
- All 14 index mode `100755`; working-tree mode `0755`; regular files; no symlinks.
- All 14 pass `bash -n`.
- Wrapper bytes contain no `/Users/spotted` or project-absolute path.
- Runtime root exact: `${AGENT_SKILLS_DIR:-${PI_AGENT_SKILLS_DIR:-${HOME}/.pi/agent/skills}}/role-pipeline/scripts/<same-name>`.
- Deterministic fake-install validation: 14/14 exec wrappers forward two arguments exactly.
- Deterministic sourced-mode validation: 14/14 source same-name targets and return control to caller with positional arguments intact.
- `AGENT_SKILLS_DIR` precedence: PASS.
- `PI_AGENT_SKILLS_DIR` fallback: PASS.
- Non-spotted temporary `HOME` fallback: PASS.
- Missing-target path/FATAL/install/override diagnostics: 14/14 PASS, exit `1`.
- All 14 same-name targets exist in current global role-pipeline installation.
- `.pi/agents/README.md` documents global prerequisite, default `${HOME}` location, both overrides, regular-file behavior, and failure behavior.

## Repository and manifest gates

PASS.

- Exact staged counts: `116 A`, `12 M`, `21 D`, total `149`.
- Commit-manifest addition/modification/deletion sets equal cached index exactly; no duplicates; declared counts exact; lists sorted under repository host locale.
- Deletion manifest contains 21 unique sorted paths and equals cached deletion set exactly.
- `git diff --cached --check`: exit `0`.
- Unstaged tracked changes: `0`.
- Nonignored untracked paths: `0`.
- Staged NUL/Mach-O binary blobs: `0`.
- Index symlinks: `0`.
- `review-final.md` and `test-report-final.md`: present, ignored, absent from Commit 1 index.

## Synthetic fresh-clone gate

PASS.

Procedure: independent `git clone --no-local --no-checkout`, outer `HEAD` checkout, exact cached binary patch applied to index, staged tree comparison, recursive remote submodule initialization, exact five-file suite.

- Synthetic clone tree: `cb7560973525a9445b972592d8e7b79a66ef86c2` — exact match.
- Remote submodule fetch: PASS.
- Synthetic vendor HEAD: `80fab4e419a57f9465bb9e2f4e90010d645e124c`.
- `agent-output/cmux-14-2/` absent before suite.
- All five verdict test files tracked.
- Exact result: `246 passed, 3 skipped, 1 warning, 2 subtests passed in 23.66s`; exit `0`.
- Active-memory evidence parent created; `memory-r4.log` produced.

## Vendor gate

PASS.

- Outer index gitlink equals clean inner HEAD: `80fab4e419a57f9465bb9e2f4e90010d645e124c`.
- Parent: `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`.
- Inner commit changes exactly `mlx_lm/tuner/trainer.py` and `tests/test_tuner_trainer.py`.
- Inner `git show --check`: exit `0`.
- Remote `refs/heads/story-14-loss-and-grad-seam` advertises exact commit.

## Active-memory repair and protected invariants

PASS.

- Active-memory fix limited to `_log_path`, parent `mkdir(parents=True, exist_ok=True)`, and existing `write_text` target.
- Hash cascade updated in protected smoke test.
- Provider source SHA-256: `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518`.
- Provider test SHA-256: `8bf2a19f9e142aab737ac5df54d22902f5bcf631766420e5752c1dc0f114903e`.
- Source-sentinel test SHA-256: `24325ef35e915b1cc3275d5fac31400c116e1adda6da0254b6158c7a6233dda9`.
- Path A: `365` index files; digest `7241924d6f9be2eb0df974fdcb1717728c71b6ee77d41dea3fe8a8af55ebb2af`.
- ADR 0028 SHA-256: `0aa743b731e6393ca84e91bcd3d32bec655177779ee15e21a47ade19e322300b`.
- Story 14.3 smoke report, log, and provenance: index blobs equal `HEAD`.

## Stale deletion and quarantine scope

PASS.

- All 21 index removals preserved locally and ignored.
- All 111 declared historical root Markdown files absent.
- `python-envs/legacy-trans/` absent.
- Ambiguous `context.md` and `adapter-converter-implementation-gpt55.md` preserved locally, untracked, ignored.
- Generated `ds4_agent_test`, `tests/ds4_lora_test`, and `tests/test_q4k_dot` preserved locally, untracked, ignored, absent from index.
- Required Torch environment files, current status documents, bakeoff evidence, and `python-envs/mlx/uv.lock` tracked in candidate.

## Findings

None.

Commit 1 gate: PASS for staged tree `cb7560973525a9445b972592d8e7b79a66ef86c2`.
