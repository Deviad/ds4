# Story 14.5d — canonical attempt-2 evidence path repair requirements

## BA verdict

**GO for TDD repair and synthetic validation only.**

No A3/B3 invocation, real model/data/provider access, training, cleanup, evidence mutation, commit, or push is authorized. A3 remains blocked until this repair is committed, Reviewer PASS and Test Manager GREEN name the same revision, a fresh external A3 authorization binds the exact production-default command, and the operator separately authorizes one visible invocation.

## User story

As a DS4 fine-tuning operator (WHO), I want the production attempt-2 history verifier to compare canonical absolute repository evidence paths (WHAT), so that intact immutable attempt-2 evidence can pass A3 admission without rewriting history or weakening authorization (WHY).

## Requirements

1. `verify_attempt2_historical_evidence()` must construct the first three expected runtime-manifest file paths as canonical absolute paths under the effective repository root, exactly matching the immutable committed manifest. The remaining seven absolute volume paths stay unchanged.
2. Path repair must not modify, normalize, regenerate, replace, or re-hash any attempt-1 or attempt-2 evidence. Preserve the pinned pre-log `990`/`cdb6b89d7f7384be84c33abe993311caefe4df3f1c2b94c9ef31a3e65e0fff41`, runtime manifest `3123`/`2cc1d2359017cce2130a9428fd202e8c2f35b5d5810030c95aaac8b464fd3d88`, all ten target path/size/SHA-256 bindings, and all eight absence facts.
3. Preserve duplicate-key rejection, exact manifest schema/value comparison, private immutable-target comparison, direct target byte/hash verification, report identity checks, and fail-closed behavior for any path, size, hash, payload, schema, or absence mutation.
4. Add tracked TDD coverage through the production verifier using either the committed real evidence read-only or an exact synthetic fixture whose first three manifest paths are absolute. The test must fail before the repair, pass after it, and include at least one relative-path or coordinated-substitution negative proving no compatibility broadening.
5. Add a tracked command-generation regression using production default `attempt3_phase_specs()` with no `workspace` override. Phase A3 must retain adapter path `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-a`, canonical repository log/script paths, and compact canonical command SHA-256 `fa4ac09716b820d249030bd93c9fe3dc0817be49059acc437af9ef22a562ad1e` while command semantics remain unchanged.
6. External authorization candidates and emitted wrappers must derive Phase A3 authorization from production default phase specs only. A synthetic workspace override must never supply the reviewed production command hash. Catalog wrapper ordering, fixed attempt-3 paths, separate A3/B3 authorization, collision gates, noclobber log open, visible `tee`, `PIPESTATUS[0]`, no retry/fallback, and no attempt 4 remain unchanged.
7. Preserve Story 14.5c metadata compatibility, attempt-3 namespace, model/data/config/provider/vendor/version/training pins, A3 `2/2700s`, B3 `1/1500s`, total `3/4200s`, cardinality, lock/watchdog, publication, rollback, and non-claim boundaries.
8. Every verdict-contributing test file must be tracked. Reviewer and Test Manager must independently verify the production verifier, command hash/default-path regression, relevant focused and canonical suites, `git diff --check`, historical evidence byte integrity, and tracking reproducibility on one exact revision.

## Acceptance criteria

1. Production `verify_attempt2_historical_evidence()` passes against the untouched canonical committed attempt-2 evidence.
2. The first three expected manifest paths are canonical absolute repository paths; all ten targets and eight absence facts remain exact.
3. Path/hash/payload/schema/absence mutations still fail closed, including relative-path substitution.
4. Production-default Phase A3 command generation has no workspace override, retains the fixed volume adapter path, and hashes exactly to `fa4ac09716b820d249030bd93c9fe3dc0817be49059acc437af9ef22a562ad1e` unless a separately reviewed command-semantic change explicitly repins it.
5. No historical evidence, runtime identity, attempt-3 namespace, execution budget, cardinality, authorization separation, collision, retry, fallback, or publication contract changes.
6. Reviewer PASS and Test Manager GREEN on the same committed revision are required before fresh external authorization review. A3 and B3 remain unauthorized in this slice.

## STOP/ESCALATE

STOP for any evidence-byte edit; trust-root weakening; acceptance of both relative and canonical forms; dynamic path fallback; workspace-overridden production authorization; command-semantic or runtime-pin drift; attempt-3 collision cleanup; retry/fallback/attempt 4; real asset access or execution; untracked verdict test; commit or push.
