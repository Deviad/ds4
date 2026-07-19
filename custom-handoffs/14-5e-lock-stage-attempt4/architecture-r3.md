# Story 14.5e — Architect r3 identity, final-root, and cleanup microrevision

## Verdict

**GO one RED-first correction of Reviewer r4 B1-B7 under this normative delta.**

A4/B4 remain unauthorized. No real model, dataset, provider, training, inference, A4/B4 invocation, cleanup, commit, or push is authorized. Reviewer r4 remains BLOCKED until implementation and tracked tests satisfy this document, canonical docs match implemented truth, and fresh independent Reviewer PASS plus Test Manager GREEN name one exact committed revision.

This document supersedes only prior claims that an external authorization can require equality to a future dynamic-resource snapshot. It does not weaken immutable identity, clean-repository, resource-policy, history, authorization, collision, report, marker, publication, lock, budget, cardinality, retry-none, or fallback-none gates.

## 1. Exact identity trust model

Identity has three deliberately different classes. They must not be hashed or compared as one undifferentiated external snapshot.

### 1.1 Externally authorized deterministic root

Each phase authorization retains one independently reviewed `trusted_identity_sha256`. Its preimage is canonical compact JSON with exact keys and a versioned schema:

```text
schema = attempt4-authorized-identity-v1
immutable = exact deterministic immutable identity
repository_status = []
dynamic_resource_policy = exact attempt4-resource-policy-v1 schema and thresholds
```

The `immutable` object keeps every current exact key and value contract: interpreter, Python version, MLX distribution version `0.31.2`, resolved vendored MLX-LM module, vendor HEAD/gitlink/clean state, provider and smoke hashes, pilot/config/provenance hashes, split hashes, model and dataset manifests, LoRA parameters, and exact Git HEAD. Missing, extra, wrong-type, reordered semantic substitutions, alternate paths, or value drift reject.

`repository_status` is exact clean status, canonicalized as `[]`; a nonempty, malformed, unsorted, alternate-command, or caller-supplied status rejects. External review may therefore bind deterministic immutable identity and exact clean repository state.

`dynamic_resource_policy` binds policy, not future observations. Its canonical contract pins the exact `dynamic_resources` keys, types, ordering/invariants, `PILOT_MEMORY_HEADROOM`, `PILOT_DISK_MIN_FREE`, competing-process limits, allowed skip categories, count/PID consistency rules, and fail-closed treatment of collection errors. Changing policy code, constants, schema, or thresholds changes the root and requires new authorization.

Phase A and Phase B authorizations remain distinct. Neither report, catalog helper, runtime value, nor previous authorization may mint or replace either external root.

### 1.2 Fresh same-process runtime snapshot

`_runtime_preflight()` is the sole production producer of a phase runtime identity snapshot. After collecting it, production must:

1. strict-validate exact top-level and nested schemas;
2. compare `immutable` and `repository_status` exactly to the externally authorized deterministic root;
3. validate `dynamic_resources` against the externally authorized exact policy and thresholds;
4. canonicalize the complete three-part identity once to compact JSON bytes;
5. retain those exact bytes, SHA-256, parsed object, phase, and capture lifecycle state in process until report/final publication or terminal failure.

An arbitrary dictionary supplied by a report, validator caller, prior phase, test hook, or mutable module global is not a trusted fresh snapshot. Production gates consume the retained same-process snapshot object/bytes, not a later reserialization of a mutable dictionary.

Dynamic observations may legitimately differ between A4 and B4, and between external review and execution. A compliant fresh observation must not be rejected merely because available memory, free disk, process observations, or allowed skip evidence differs from an earlier external snapshot. Below-threshold values, malformed schemas, collection errors, impossible counts, disallowed competitors, or post-capture mutation reject.

### 1.3 Report and final lineage

Each phase report records the exact complete identity object plus its canonical byte snapshot/hash lineage. Report validation requires semantic equality and byte/hash equality to that phase's retained same-process snapshot. Final publication preserves both phase snapshots independently:

- deterministic projections must equal the authorized root and each other where the contract requires cross-phase stability;
- Phase A dynamic resources equal Phase A captured bytes only;
- Phase B dynamic resources equal Phase B captured bytes only;
- no cross-phase equality requirement applies to valid dynamic observations.

Any mutation after capture, including a still-policy-compliant resource value, rejects report or final publication because it no longer equals the retained phase bytes.

## 2. Normative requirements delta

Prior R14.5e-7.1 and canonical wording such as “authorization binds exact post-preflight runtime identity” are narrowed as follows:

- **Required:** external authorization equality for exact deterministic `immutable`, exact clean `repository_status`, and exact versioned dynamic-resource policy/schema/thresholds.
- **Required:** fresh preflight validation of actual dynamic values and byte-exact retention through that phase's report and final lineage.
- **Forbidden:** requiring actual dynamic values to equal an earlier external snapshot that could not know future resources.
- **Forbidden:** accepting caller-injected dynamic values without the production fresh-preflight capture lifecycle.
- **Forbidden:** using this delta to relax any threshold, schema, identity, source, history, authorization, collision, report, marker, publication, or lock gate.

Tests and docs must reflect this distinction. A different but policy-valid fresh dynamic snapshot is not an authorization mismatch; mutation of a captured/report snapshot is a lineage mismatch.

## 3. B4 production order

Reviewer r4 B1 is corrected by one explicit order. No strict Phase A validator may run with `{}`, a report-derived trusted identity, or a not-yet-produced runtime identity.

After pre-lock admission, acquisition, and post-lock admission, real B4 `run_phase()` performs:

1. Independently retain current attempt-1, attempt-2, and attempt-3 history roots from their frozen verifiers. Do not derive trusted roots from Phase A report fields.
2. Parse and retain the separately supplied Phase A authorization and Phase B authorization. They must be different, phase-correct, externally rooted values.
3. Read Phase A report bytes and Phase A OK-marker bytes exactly once; compute SHA-256; strict-decode with duplicate-key rejection; retain bytes, hashes, paths, schemas, and parsed snapshots. This is capture, not authorization by report contents.
4. Run fresh B4 `_runtime_preflight()` and create the retained B4 runtime snapshot described in §1.2.
5. Validate B4 authorization against the deterministic projection and resource policy, then validate the B4 dynamic observation against policy.
6. Strict-validate the already captured Phase A report and marker using all separate roots: Phase A authorization, attempt-1/2/3 histories, exact Phase A spec/paths/command/contract/artifacts/budget/cardinality, deterministic identity projection, Phase A's own recorded dynamic snapshot schema/policy, and the exact captured report/marker bytes and hashes.
7. Validate resume source and all Phase B dependency fields against that same captured Phase A evidence.
8. Construct `phase_a_admission_lineage` from those same captured buffers. Do not re-read to create lineage and do not replace any trusted root with a report field.
9. Continue to contract creation, output mutation, API loading, and training only after every preceding gate passes.

A4 uses the same identity capture rules without Phase A dependency. Every repeated post-lock or pre-publication gate consumes the retained snapshot and independent roots.

## 4. Final publication root closure

Reviewer r4 B3 is corrected by eliminating all report-derived fallbacks and by comparing files to retained admission bytes.

For B4 success publication:

1. Retain Phase A report/marker bytes and hashes from §3 for the entire B4 process.
2. Retain Phase A authorization, Phase B authorization, attempt-1/2/3 history roots, B4 runtime identity bytes/hash, phase specs, and resume bindings in separate variables whose values are never sourced from either report.
3. Write Phase B report, immediately read it back as exact bytes, hash it, strict-parse it, and require equality to the in-memory report and retained B4 identity snapshot.
4. Immediately before final report publication, re-read Phase A report and marker and require byte-for-byte and hash equality to the §3 capture. A semantically valid rewrite or a marker rehash is still rejection.
5. Strict-validate Phase A and Phase B from the captured byte snapshots with mandatory explicit trusted arguments. `None`, optional fallback to report fields, or history/auth extraction from Phase A rejects.
6. Build final report only from validated captured Phase A and Phase B snapshots plus separate retained roots. It binds exact phase paths, report hashes, Phase A marker hash, contracts, artifacts, resume digest, budgets, cardinalities, lock lifecycle, both authorizations, all three histories, and both phase identity snapshot hashes.
7. Write final report, read it back, strict-validate exact bytes/hash/schema/value against the in-memory final object and all retained roots, then write and strict-validate final OK marker.
8. Any write, readback, hash, schema, value, retained-root, or marker failure removes no evidence, publishes no success, and follows existing fail-closed failure-evidence rules.

`validate_final_attempt4_report()` and `_write_success_evidence()` must require explicit attempt-1, attempt-2, and attempt-3 trusted history parameters. No default-to-Phase-A behavior is allowed.

## 5. Unlock/close truth table and ownership truth

Reviewer r4 B4 is corrected by separating “cleanup attempted” from “descriptor proven closed.” `_rollback_ft_lock()` and `_release_ft_lock()` must never clear ownership merely because a `finally` block ran.

Owner cleanup records unlock and close outcomes independently and preserves the primary write/attestation error plus every cleanup error. Exact state transitions:

| Unlock result | Close result | Required globals/FD state | Result |
|---|---|---|---|
| success | success | clear all owner globals; FD proven closed; `FLOCKED=False` | success only when no earlier error; otherwise raise earlier error |
| failure | success | clear owner globals because successful close proves FD gone and releases any flock | terminal cleanup failure, preserving earlier error |
| success | failure | retain FD number, path, inode, PID, token, partial state; set `FLOCKED=False`; set cleanup-uncertain terminal state | terminal failure; no further lock operation or publication in process |
| failure | failure | retain FD number and every owner binding; keep last-known `FLOCKED=True`; set cleanup-uncertain terminal state | terminal failure; competitor may still be excluded; no further lock operation or publication in process |

A close exception is not proof the descriptor closed. Retained uncertain descriptors are diagnostic/containment state only: never reused for reads, writes, retry-close, unlock, reacquisition, or publication. Process termination is the only implicit kernel cleanup. Entry to every lock gate rejects while cleanup-uncertain state exists.

Inspection/acquisition candidate descriptors obey the same truth rule. If candidate close is not proven, retain its descriptor and last-known flock state in a process-global cleanup-uncertain registry before raising; never let a local variable disappear while a live flock may remain.

Release/rollback still never unlinks, renames, replaces, quarantines, or repairs the canonical pathname. Unlock/close failure never masks the primary failure. Success evidence is forbidden after any cleanup error.

## 6. A3 executable retirement

Reviewer r4 B5 remains a hard implementation requirement. Attempt 3 may survive only as frozen read-only authorization/history constants and strict historical-verifier code needed to validate recorded A3 failure evidence.

Remove live A3 namespace/spec/command/launch/report/final/marker/preparation functions listed by Reviewer r4, all runnable `attempt == 3` and `attempt in (3, 4)` dispatch branches, and all A3 write/publication behavior. Shared code must dispatch live execution only to attempt 4; historical verifier code must not construct runnable commands, outputs, markers, or phase specs.

AST tests must prove no executable A3 branch or callable live A3 machinery remains outside named read-only historical authorization/verifier boundaries. Parser, catalog, help, emit, and run surfaces continue rejecting attempt 3.

## 7. RED-first verification contract

Before implementation, tracked tests must fail for each defect. Minimum independent coverage:

1. Real B4 `run_phase()` reaches fresh preflight before strict Phase A validation; first strict validation receives retained fresh identity, separate A/B authorizations, and separate history roots.
2. Authorized deterministic immutable or repository-status mutation rejects. Policy/schema/threshold mutation rejects and changes authorization root.
3. Two fresh policy-valid dynamic snapshots may differ without external authorization equality; below-threshold/malformed/caller-injected/post-capture-mutated values reject.
4. Phase A report or marker mutation after B4 admission rejects final publication even when JSON remains canonical and marker hash is recomputed.
5. Every final validator called without explicit attempt-1/2/3 histories, A/B authorizations, captured Phase A bytes/hashes, or phase identity snapshots rejects.
6. Strict `_validate_attempt4_report`, `_validate_attempt4_marker`, Phase B dependency, and `validate_final_attempt4_report` missing/extra/wrong-type/wrong-value matrices exercise production functions.
7. Release and rollback inject prior write/attestation failure plus each unlock/close truth-table combination; competitor subprocess verifies whether flock remains; globals/retained-FD state matches §5.
8. Candidate PRE_LOCK cleanup injects unlock/close combinations and proves no untracked live FD/flock state is lost.
9. Existing persistent-lock PRE/POST, replacement/race, short-write/fsync, crash/stale, same-inode A4-idle-to-B4-reacquire, no-unlink/no-rename matrices remain complete.
10. AST audit proves A3 read-only retirement and no attempt-5/dynamic allocator.
11. Publication write seams for Phase B report, phase marker, final report, and final marker all fail closed with no success marker surviving.
12. Full pilot and exact vendor-first canonical six-file suite pass; every verdict-contributing test file is tracked.

## 8. Canonical documentation truth

Until code and RED matrix satisfy this document, `docs/architecture.md`, `docs/technical-spec.md`, and `docs/backlog.md` must not claim B1-B6 behavior is implemented. Implementation slice must update them together to state:

- deterministic authorization root versus fresh dynamic-policy validation;
- exact B4 capture/preflight/strict-validation order;
- retained Phase A bytes and independent history/auth roots at final publication;
- unlock/close truth table and uncertain-FD retention;
- actual A3 executable retirement;
- exact tests genuinely present.

No ADR required: this narrows feasibility and lineage semantics inside existing Story 14.5e design without changing fixed namespace, training mechanics, budgets, or production inference architecture.

## 9. STOP conditions

STOP on dynamic threshold/schema relaxation; external authorization minted at runtime; dirty repository acceptance; arbitrary caller identity treated fresh; Phase A validation before B4 preflight; report-derived trusted roots; optional final-root fallbacks; Phase A reread accepted without exact admission-byte equality; ownership globals cleared after unproven close; retained uncertain FD reused; any lock-path unlink/rename/replace/quarantine; executable A3 live path; attempt 5/dynamic namespace; untracked verdict test; real execution; cleanup; commit; or push.
