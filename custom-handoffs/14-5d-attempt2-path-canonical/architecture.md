# Story 14.5d — minimal canonical attempt-2 path repair architecture

## Verdict

**GO for one TDD path-literal repair and synthetic validation only.**

A3/B3, real assets, provider access, training, cleanup, evidence mutation, commit, and push remain unauthorized.

## Production change

Edit only `scripts/ds4_segmented_pilot.py`.

Replace the first three repo-relative path strings in both independent attempt-2 trust roots with the immutable manifest's exact canonical absolute strings:

- `/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-2/phase-a-log.txt`
- `/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-2/phase-a-report.json`
- `/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-2/pilot-report.json`

Apply this to:

1. `_ATTEMPT2_HISTORICAL_FILES` — import-time private immutable target snapshot.
2. `verify_attempt2_historical_evidence()` local `files_expected` — admission-local independent trust root.

No helper, normalization, fallback, dual relative/absolute acceptance, or evidence rewrite. Keep `_attempt2_expected_path()` unchanged: its existing canonical-root-to-`repo_root` mapping is only the synthetic-test file-access seam; it must not change manifest semantics.

Keep unchanged:

- relative pre-log and runtime-manifest descriptor paths;
- their exact `990`/`cdb6b89d7f7384be84c33abe993311caefe4df3f1c2b94c9ef31a3e65e0fff41` and `3123`/`2cc1d2359017cce2130a9428fd202e8c2f35b5d5810030c95aaac8b464fd3d88` bindings;
- remaining seven absolute `/Volumes/Data NVME/...` entries;
- all ten sizes and SHA-256 values;
- all eight absence facts;
- duplicate-key, exact schema/value, private immutable-target, direct byte/hash, report-identity, and fail-closed checks.

## TDD

Edit only tracked `tests/test_ds4_segmented_pilot.py`.

### RED

Add public-boundary coverage calling `verify_attempt2_historical_evidence(repo_root=tmp_path)` through an exact synthetic fixture whose manifest retains the three canonical absolute repository paths. Do not monkeypatch either trust-root tuple or mask the first-three path conversion. Filesystem seams may redirect only unavailable external `/Volumes/...` targets; existing production semantic/schema/private-root logic must execute.

Before repair, require failure at exact manifest semantic comparison. After repair, require:

- public verifier PASS;
- ten verified targets;
- eight absence bindings retained;
- first three manifest paths equal the three canonical absolute strings.

Add negatives:

- substituting any first-three path with its repo-relative form fails closed;
- substituting a different absolute path fails closed;
- existing path/size/hash/payload/schema/absence mutation tests remain green.

### Command regression

Using `attempt3_phase_specs()` with no `repo_root` or `workspace` override:

- assert Phase A adapter path is `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-a`;
- hash canonical compact JSON of `canonical_attempt3_command("phase-a", spec)`;
- assert `fa4ac09716b820d249030bd93c9fe3dc0817be49059acc437af9ef22a562ad1e`.

Add an explicit workspace-override control proving its adapter path and command hash differ from production default. Synthetic workspace output must never be accepted as production authorization identity.

## Scope and protected bytes

Allowed changes:

- `scripts/ds4_segmented_pilot.py` — six path-string replacements only.
- `tests/test_ds4_segmented_pilot.py` — focused TDD only.
- slice handoff/status artifacts.

Protected byte-for-byte:

- every attempt-1/attempt-2 evidence file and external target;
- `scripts/finetune_ds4.py` and command catalog behavior;
- attempt-3 namespace, paths, authorization separation, collision gates, wrapper order, noclobber, visible `tee`, `PIPESTATUS[0]`, lock/watchdog, publication/rollback, retry/fallback/attempt count;
- model/data/config/provider/vendor/version pins and A3 `2/2700s`, B3 `1/1500s`, total `3/4200s`;
- production inference, Metal, CUDA, SSD, distributed, vendor, environment, and training implementation files.

`docs/backlog.md` already contains Story 14.5d. `docs/technical-spec.md` already records the intended absolute attempt-2 bindings. No ADR or canonical-doc edit is needed because this repairs code drift to the existing contract; `architecture.md` is handoff evidence only.

## Verification gates

1. Capture focused RED before production edit.
2. Focused GREEN for new verifier and command-hash tests.
3. Run tracked canonical Story 14.5 suite only; no real execution.
4. `git ls-files --error-unmatch tests/test_ds4_segmented_pilot.py` must pass.
5. `git diff --check` must pass.
6. Independently verify historical pre-log/manifest bytes and all 10/8 bindings unchanged.
7. Reviewer PASS and Test Manager GREEN must name the same exact revision before fresh external A3 authorization review.

## STOP

Stop on evidence-byte change, broader path acceptance, dynamic normalization/fallback, private-root weakening, command hash drift, workspace-overridden production authorization, attempt-3 semantic drift, untracked verdict test, real asset/training invocation, cleanup, commit, or push.
