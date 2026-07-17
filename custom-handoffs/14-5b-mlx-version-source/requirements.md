# Story 14.5b — canonical MLX version source and A2 reauthorization requirements

## BA verdict and authorization boundary

**GO for requirements, design, TDD implementation, and synthetic review only.** This handoff does not authorize real model or dataset access, provider execution, training, inference, cleanup, deletion, overwrite, Phase A2, Phase B2, commit, or push.

A future Phase A2 invocation remains blocked until the minimal repair is implemented, all required tracked tests pass, an independent Reviewer returns PASS, a Test Manager returns GREEN on the exact same revision, and the operator gives fresh explicit authorization for exactly one visible invocation bound to that reviewed revision and exact command. Phase B2 remains separate and unauthorized.

## User story

As a DS4 fine-tuning operator (WHO), I want the Phase A2 launch check to verify the installed MLX distribution through its canonical metadata and preserve one-shot authorization lineage (WHAT), so that a legitimate MLX `0.31.2` installation can proceed without weakening fail-closed version, namespace, or retry controls (WHY).

## Fixed evidence and problem statement

- The only authorized Phase A2 invocation at commit `c910d1ba33912236ee87f3f9bdfb5b31edece6e7` exited `1` before opening its log with `pilot launch check failed closed: MLX version mismatch: None`.
- The canonical interpreter reports `importlib.metadata.version("mlx") == "0.31.2"`; imported `mlx.__version__` is `None`.
- Training/provider calls/optimizer updates were `0/0/0`.
- Every Phase A2 output, checkpoint, config, log, report, OK/fail marker, and final report/OK/fail destination remained absent. No cleanup, overwrite, fallback, or retry occurred.
- That invocation's authorization is consumed. Its failure record and the immutable attempt-1 evidence remain historical evidence and must not be rewritten.

## R14.5b-1 — authoritative MLX version contract

1. `importlib.metadata.version("mlx")` is the sole authoritative source for the installed MLX version used by `scripts/ds4_segmented_pilot.py::_runtime_preflight()`.
2. The accepted value is exact string equality with `0.31.2`. Do not normalize, coerce, compare compatible ranges, accept local/dev suffixes, or treat newer versions as compatible.
3. `mlx.__version__` is non-authoritative. Its absence, `None`, correct value, incorrect value, or spoofed value must not change the decision made from distribution metadata.
4. Importing `mlx` remains mandatory runtime-availability evidence. A module import failure still fails closed; ignoring its version attribute does not make the module optional.
5. Distribution metadata missing for `mlx`, returning a non-string/empty value, or raising any exception fails closed before training. No fallback may read `mlx.__version__`, `mlx_lm.__version__`, package mappings, environment variables, configuration, shell output, `pip`, or another distribution name.
6. The existing immutable identity field `mlx_version` must contain the accepted distribution-metadata value. No alternate field may preserve or publish the module attribute as if authoritative.
7. Failure diagnostics must distinguish at least version mismatch from unavailable/error metadata without claiming the module attribute is authoritative. They must not expose a success path after metadata failure.

## R14.5b-2 — mutation-sensitive TDD matrix

Tracked synthetic tests must isolate the version decision and prove all rows below:

| Distribution metadata for `mlx` | `mlx.__version__` | Required result |
|---|---|---|
| exact `0.31.2` | absent or `None` | version gate passes; identity records `0.31.2` |
| exact `0.31.2` | incorrect/spoofed value | version gate passes; spoof is ignored |
| mismatch, including a nearby version or suffixed value | exact `0.31.2` | fail closed with metadata mismatch |
| mismatch | absent, `None`, or another spoof | fail closed with metadata mismatch |
| missing distribution / `PackageNotFoundError` | any value | fail closed as metadata unavailable |
| generic metadata exception | any value | fail closed as metadata error/unavailable |
| empty or non-string metadata result | any value | fail closed; no coercion or fallback |

Tests must mutate one source at a time, keep unrelated preflight prerequisites valid through temp-only fixtures/stubs, and observe the production preflight boundary. Source-text assertions, a mock helper that reproduces its own expected value, or tests that fail at an earlier unrelated guard do not satisfy this matrix.

## R14.5b-3 — launch-check no-write and no-call contract

For every mismatch, missing, malformed, or error metadata case:

1. Direct `--launch-check` and the generated attempt-2 wrapper boundary must reject before opening the phase log or creating the phase output directory.
2. No training API load, provider call, optimizer update, checkpoint/config write, success/failure publication, or final aggregation may occur.
3. In temp-only wrapper tests, all corresponding Phase A2 destinations must remain absent: phase output/start/step-1/step-2/final checkpoint/config, phase log/report/OK/fail marker, and final report/OK/fail marker.
4. The wrapper must preserve collision checks before the launch check, launch check before log opening, launch check before training, exclusive FD-backed logging after admission, and exact nonzero status propagation.
5. Tests must prove absence from a fresh fixture; they may not create then delete destinations to manufacture a no-write verdict.

## R14.5b-4 — attempt-2 namespace and authorization decision

**Decision: safely reuse the still-empty fixed attempt-2 namespace; do not repin to attempt 3.**

Rationale: the failed invocation stopped before its log and before any namespace mutation; every pinned attempt-2/final destination remains absent; attempt-1 evidence is separate and immutable; existing collision-before-write gates remain fail-closed. A broad path/catalog/report-schema repin would add risk without preventing an overwrite that the unchanged collision gates already forbid.

The audit contract is:

1. `attempt-2` remains the immutable artifact namespace ordinal. It is not rewritten as proof that only one wrapper process was ever launched.
2. The failed invocation at `c910d1ba33912236ee87f3f9bdfb5b31edece6e7`, its exact pre-log error, zero calls/updates, absent destinations, and consumed authorization remain recorded.
3. Any future invocation must be recorded as a newly authorized invocation after Story 14.5b repair, linked to the failed revision/result, the exact repaired revision, Reviewer PASS, Test Manager GREEN, operator authorization, exact command, and resulting exit/evidence.
4. Fresh authorization permits exactly one invocation. It is not a continuation or retry under the consumed authorization, and it creates no automatic retry entitlement. Any exit consumes the new authorization.
5. Immediately before invocation, all fixed Phase A2 and final destinations must be absent and the attempt-1 manifest must verify immutable. Any collision or manifest drift is STOP/ESCALATE: no cleanup, deletion, rename, overwrite, fallback namespace, dynamic suffix, or attempt-3 allocation.
6. Exact fixed attempt-2 A2/B2 paths, identity, parameters, budgets, cardinality, lock, report, marker, and digest contracts remain unchanged. This story authorizes no path repin.
7. A2 remains exactly `2` provider calls/optimizer updates within `2700s` if admitted. B2 remains exactly `1` resumed call/update within `1500s`; total active budget remains `4200s`.
8. B2 remains blocked until the newly authorized A2 succeeds and its exact canonical report, OK marker, checkpoint/config hashes and digests, identity, cardinality, lock release, absent failure evidence, and fixed namespace are independently verified. B2 then requires separate explicit operator authorization for one visible invocation.

## R14.5b-5 — implementation scope and protected invariants

Expected minimal future implementation scope:

- `scripts/ds4_segmented_pilot.py` — replace the MLX module-attribute version decision with authoritative distribution metadata and preserve the accepted value in immutable identity.
- `tests/test_ds4_segmented_pilot.py` — add the mutation-sensitive version and no-write launch-check matrix.
- Canonical documentation/handoff evidence required by the slice.

`scripts/finetune_ds4.py` and its fixed attempt-2 catalog/wrapper must remain byte-for-byte unchanged unless Architect identifies a concrete missing test seam and explicitly repins scope before coding. No provider/vendor, MLX/MLX-LM package, model, dataset, config, attempt-1 evidence, Story 14.3 smoke, Path A, CUDA, distributed, Metal, CPU, SSD streaming, GGUF, or inference code change is permitted.

Use TDD: establish RED tests against the current `mlx.__version__` behavior, implement the smallest GREEN repair, then refactor only while tests remain green. Every verdict-contributing test file must be tracked and verified with `git ls-files` before Coder, Reviewer, or Test Manager cites it.

## Acceptance criteria

1. The user story uses exact WHO/WHAT/WHY form and records the pre-log failure without rewriting historical evidence.
2. `importlib.metadata.version("mlx")` is the sole actual-version source; accepted value is exactly `0.31.2`.
3. Absent, `None`, incorrect, and spoofed `mlx.__version__` values cannot override correct or incorrect distribution metadata.
4. Correct metadata passes the isolated version gate and populates immutable identity `mlx_version == "0.31.2"`.
5. Mismatch, missing distribution, generic metadata exception, and malformed metadata each fail closed with no fallback.
6. Mutation-sensitive tests reach the production preflight boundary; no verdict relies only on source text, tautological mocks, or earlier unrelated failures.
7. Every negative metadata case proves zero training/provider/update activity and no Phase A2 output/log/report/marker/final destination write through direct launch-check and generated-wrapper boundaries.
8. Wrapper collision-before-write, launch-check-before-log/training, FD logging, and exit propagation contracts remain unchanged.
9. The fixed empty attempt-2 namespace is reused only under fresh explicit authorization and complete lineage; no path repin, cleanup, overwrite, fallback, dynamic suffix, attempt-3 allocation, or automatic retry occurs.
10. Any future A2 invocation is bound to one exact reviewed revision and command; any exit consumes that authorization.
11. B2 remains separately blocked and separately authorized only after exact A2 success evidence verification.
12. Existing A2/B2 paths, model/data/config/provider/vendor identity, `2/2700s + 1/1500s = 4200s` budgets, report/marker/digest contracts, and immutable attempt-1 evidence remain unchanged.
13. Targeted tests, the canonical regression suite selected by Architect, `py_compile`, `git diff --check`, protected-file hash checks, and tracked-test checks pass on one exact revision.
14. Independent Reviewer PASS and Test Manager GREEN are mandatory before the operator may authorize one new A2 invocation.
15. No real model/dataset access, provider call, training, inference, cleanup, deletion, overwrite, Phase A2/B2 execution, commit, or push occurs in this BA slice.

## STOP/ESCALATE

STOP for any proposal to trust or fall back to `mlx.__version__`; accept a version other than exact `0.31.2`; normalize metadata; suppress metadata errors; mutate or clean any attempt namespace; change fixed paths, identity, budgets, provider/vendor code, or training semantics; access real assets; run A2/B2; claim authorization from this document; proceed with an untracked verdict file; or treat a failed newly authorized invocation as permission to retry.