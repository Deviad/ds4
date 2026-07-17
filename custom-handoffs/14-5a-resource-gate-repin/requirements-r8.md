# Story 14.5a — BA r8 requirements: isolated consumer oracles

## Decision and authorization boundary

**GO for tracked synthetic test-only closure.** r8 adds direct, mutation-sensitive oracles for the checkpoint/config consumers left unproven by r7. It does not authorize production changes, real asset access, Phase A2/B2 execution, training, inference, cleanup, deletion, overwrite, commit, or push.

If a direct oracle proves that a required consumer does not exist or that a binding is only an unused catalog assignment, treat that as a real defect and STOP/ESCALATE for Architect/Coder disposition. Do not weaken the oracle or claim coverage from string presence.

Real Phase A2 remains blocked pending independent Reviewer PASS, Test Manager GREEN, and fresh explicit operator authorization. Phase B2 still requires separate authorization after verified Phase A2 completion.

## Evidence reviewed

- Existing canonical requirements: `custom-handoffs/14-5a-resource-gate-repin/requirements.md`.
- Existing design: `custom-handoffs/14-5a-resource-gate-repin/architecture.md`.
- r7 task and implementation evidence: `task-coder-r7.md`, `coder-notes-r7.md`.
- Latest independent verdict: `custom-handoffs/standby/review.md`, r7 **BLOCKED**.
- r7 closed emitted-wrapper quoting and exit propagation.
- r7 canonical verification: `470 passed, 3 skipped, 1 warning, 2 subtests passed`.
- r7 reviewed hashes:
  - `scripts/ds4_segmented_pilot.py`: `02355fa330173951416e0e50652aa40e688f5601f79be3c3cc826d7160a10c31`;
  - `scripts/finetune_ds4.py`: `de303d1166f2ecd2428dc15f63c8be8ed22e9f30274008afd35be458ef4b188b`;
  - `tests/test_ds4_segmented_pilot.py`: `e80baf410b5ec567ae842bbdf520b8e9b3ab6e484543baf0c15bbd25c67226e2`.

## Trackable user story

As a DS4 fine-tuning operator (WHO), I want direct isolated test oracles for every applicable attempt-2 checkpoint/config consumer (WHAT), so that a disconnected, generic, or unused binding cannot pass review and authorize an incorrectly wired Phase A2 or Phase B2 launch (WHY).

## R8 key set

Every r8 matrix must enumerate these keys explicitly.

### Phase A2

1. `phase-a-start-checkpoint`
2. `phase-a-step1-checkpoint`
3. `phase-a-step2-checkpoint`
4. `phase-a-final-checkpoint`
5. `phase-a-config`

### Phase B2

1. `phase-b-resume`
2. `phase-b-start-checkpoint`
3. `phase-b-step1-checkpoint`
4. `phase-b-final-checkpoint`
5. `phase-b-config`

`phase-b-resume` is the exact Phase A2 step-2 dependency and must retain independent report, marker, file-SHA, and canonical-tensor-digest binding proof.

## R14.5a-r8-1 — Direct isolated oracle definition

A verdict-contributing r8 case is valid only when all conditions hold:

1. Start from a fully valid synthetic A2 or B2 fixture.
2. Mutate exactly one named checkpoint/config binding.
3. Keep unrelated outer prerequisites valid, including namespace-map equality, contract digest, serialized report snapshot, report SHA, marker report path/hash, marker contract digest, and resume file/canonical digest when those are upstream of the intended consumer.
4. Invoke the named production consumer or the real publication/launch boundary that contains it.
5. Prove the intended boundary was reached using a specific error message containing the consumer, key, or exact mutated path, or using a positive executable observation of the changed binding.
6. Prove no earlier generic guard supplied the verdict.
7. Prove no unrelated binding changed.

The following do not satisfy r8:

- `with pytest.raises(PilotError)` without a specific message/boundary;
- failure at `canonical A2 namespace path substitution` when a later checkpoint/config consumer is claimed;
- replacing the entire `namespace_paths` map;
- asserting only `str(path) in command` or assignment presence;
- a helper that computes expected and actual values from the same mutated object without an independent observation;
- a test whose setup failure prevents the named consumer from running.

## R14.5a-r8-2 — Explicit consumer matrix

Tests must be parameterized by key within each applicable consumer, not by one broad mutation loop that can stop at a shared guard.

### Canonical report validation

- A2: all five A2 keys.
- B2: all five B2 keys.
- Each case must synchronize outer namespace/digest/report prerequisites, then mutate the report artifact, update-checkpoint, resume-source, effective-config, or other named downstream binding that the validator independently consumes.
- Each case must assert the specific canonical artifact/checkpoint/config/resume boundary reached.

### Success admission and publication/no-write

- A2: all five A2 keys.
- B2: all five B2 keys.
- Each negative case must enter through the real success-admission path before `_write_success_evidence`; direct invocation of `_write_success_evidence` with an already-invalid report is not a validation oracle.
- Each mutation must be rejected before any phase or final report/marker destination is created.

### Phase B dependency admission

- Directly cover all five A2 keys because Phase B admission consumes the canonical A2 report/artifacts.
- Directly cover `phase-b-resume` because Phase B admission consumes its exact path, report binding, marker binding, file SHA, and canonical tensor digest.
- B2-generated start, step-1, final, and config artifacts are not Phase B prerequisites. Mark those matrix cells explicitly `N/A — produced after dependency admission`; do not manufacture a passing dependency test for an unconsumed value.
- Recompute the outer A2 namespace map, contract digest, serialized report, marker report SHA, marker contract digest, and unaffected resume bindings as needed so the named dependency consumer is the first failing boundary.
- Generic namespace substitution or arbitrary `PilotError` does not count.

### Executable catalog

- Enumerate all ten keys and execute generated A2/B2 wrappers in temp-only fixtures with spaces in paths.
- For every applicable key, compare a valid baseline with one-key mutation and prove the changed binding reaches its intended launch-check argument, training argument, resume argument, phase-spec use, or emitted runtime evidence.
- Assert exact top-level parse remains `bash`, `-lc`, one inner script; launch-check precedes training; training status propagates; FD-backed log behavior remains intact.
- Assignment or command-string presence alone is insufficient. The launch/training stub, captured argv, trace, log, or direct called consumer must observe the changed value.
- If any required key has no executable observation beyond an unused shell assignment, report a real missing-consumer defect. Do not mark it covered and do not silently classify it `N/A`.

## R14.5a-r8-3 — Publication safety and positive B2 bindings

Negative A2 cases use fresh destinations and must leave absent:

- `phase-a-report`;
- `phase-a-ok` and `phase-a-fail`;
- `final-report`;
- `final-ok` and `final-fail`.

Negative B2 cases may retain the valid immutable synthetic A2 prerequisites, but must leave absent:

- `phase-b-report`;
- `phase-b-ok` and `phase-b-fail`;
- `final-report`;
- `final-ok` and `final-fail`.

The positive valid B2 case must prove all four success publications:

1. `phase-b-report` exists, parses, and has `status == "ok"`.
2. `phase-b-ok` exists and binds the exact phase report path, SHA-256, output path, contract digest, phase, attempt `2`, and namespace.
3. `final-report` exists, parses, and has `status == "ok"` with the expected A2+B2 aggregate.
4. `final-ok` exists and binds the exact final report path and SHA-256, plus phase, attempt `2`, namespace, output path, and contract digest.

The positive case must also prove `phase-b-fail` and `final-fail` remain absent. Deleting final-report or `final-ok` publication, swapping either report path, or changing either report hash must turn the test RED.

## R14.5a-r8-4 — Independent fixtures and mutation discipline

- Build valid A2 and B2 reports from temp-only artifacts; do not access `/Volumes/Data NVME/...`.
- Use fresh report/marker destinations for no-write tests. A helper that prewrites the destination under test must remove or avoid that write before the oracle begins.
- Snapshot valid A2 prerequisite bytes before B2 negative cases and assert they remain byte-identical afterward.
- One parameter value represents one key and one intended boundary.
- Expected paths and hashes must come from an independent baseline, not from the mutated report field being checked.
- Error regexes must identify the intended boundary and, where production includes it, the exact mutated path/key.

## R14.5a-r8-5 — Test-only scope and protected invariants

Expected r8 changes are limited to tracked test code and r8 handoff evidence. Production changes are prohibited unless a direct RED oracle establishes a real missing/incorrect consumer and the finding is explicitly escalated.

Preserve byte-for-byte unless escalation occurs:

- `scripts/ds4_segmented_pilot.py`;
- `scripts/finetune_ds4.py`;
- Story 14.3 segmented smoke source/tests/evidence;
- provider/vendor code;
- `ds4.c`, `ds4_cli.c`, `ds4_server.c`, `ds4_metal.m`, and all tracked Metal kernels;
- CUDA, distributed, Metal default, CPU reference, SSD streaming, GGUF, and Path A behavior/evidence;
- all attempt-1 evidence.

No real model/dataset/config access, provider call, optimizer update, training, inference, marker staging, cleanup, commit, or push.

## R14.5a-r8-6 — TDD, tracking, and verification

1. Add RED tests first against the r7 state for each missing isolated oracle.
2. Make only the minimum test-fixture/oracle changes needed for GREEN when production already satisfies the contract.
3. If RED exposes a production defect, stop test-only closure and escalate with the exact key, consumer, observed behavior, and smallest required production boundary.
4. Run targeted canonical-validation, publication, dependency, and executable-catalog matrices.
5. Run the canonical exact six-file suite; result must be at least the r7 baseline of `470 passed`, with zero failures.
6. Run `py_compile` for touched Python tests and the two protected scripts.
7. Run `git diff --check` and `git diff --cached --check`.
8. Prove protected production files byte-identical by direct hashes/comparison, not a vacuous zero-line diff.
9. Run `git ls-files -- <every verdict-contributing test file>`; any untracked file blocks the verdict.
10. Perform no marker staging. The role completion marker is workflow status only.

## Acceptance criteria

1. `requirements-r8.md` records the exact ten-key A2/B2 checkpoint/config set and the consumer applicability rules.
2. Every claimed canonical-validation case mutates one key, keeps outer guards valid, reaches a key-specific downstream boundary, and rejects with a specific message.
3. Every claimed publication case enters through real success admission and proves zero phase/final reports or markers on rejection.
4. Dependency tests directly isolate all five A2 keys and `phase-b-resume`; B2-produced artifacts are explicitly marked non-applicable at pre-launch dependency admission.
5. No dependency verdict relies on whole-map mutation, generic namespace substitution, or arbitrary `PilotError`.
6. Executable catalog tests use one-key mutations and observable launch/training/runtime behavior; no verdict relies only on string or assignment presence.
7. Any catalog binding lacking a real executable consumer is reported as a defect rather than covered by a tautological test.
8. Valid B2 publication writes `phase-b-report`, `phase-b-ok`, `final-report`, and `final-ok`, with exact report-path/SHA/contract/namespace/attempt/output bindings.
9. Invalid A2/B2 mutations leave all relevant phase/final report and marker destinations absent and leave valid A2 prerequisites byte-identical.
10. Tests remain temp-only and perform no `/Volumes/Data NVME/...` access.
11. Canonical six-file verification has zero failures and at least `470 passed`; targeted matrices, `py_compile`, whitespace checks, and protected byte comparisons pass.
12. Every verdict-contributing test file is tracked.
13. Protected production/runtime/vendor files and attempt-1 evidence remain unchanged unless a real defect is escalated before further work.
14. No real Phase A2/B2 execution, cleanup, marker staging, commit, or push occurs.
15. Reviewer PASS remains withheld until a fresh independent review confirms these direct isolated oracles; real execution authorization remains blocked.

## STOP/ESCALATE

Stop and report instead of weakening tests if:

- a key cannot reach a consumer after generic guards are kept valid;
- executable catalog evidence ends at an unused assignment;
- satisfying an oracle requires production behavior change;
- a specific boundary cannot be asserted without accepting arbitrary `PilotError`;
- a negative publication case writes any protected phase/final report or marker;
- a test requires real assets or `/Volumes/Data NVME/...` access;
- a verdict-contributing test file is untracked;
- any attempt-1 or protected production evidence changes.
