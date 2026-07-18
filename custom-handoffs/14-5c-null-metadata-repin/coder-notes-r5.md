# Story 14.5c — Coder r5

## Scope

Direct RED-first synthetic implementation only. No delegate, cmux, real assets, training, cleanup, commit, or push.

## r5 closure

- `validate_attempt3_authorization()` always validates phase binding and canonical command hash; `verify_sources=False` skips only source, revision, protected-tree, and historical-file I/O.
- Attempt-2 verification now carries its exact pre-log, manifest, ten target files, eight absence facts, and complete semantic schema inside the verifier trust root; mutable compatibility aliases cannot substitute history.
- B3 publication requires explicit retained Phase-A authorization and pre-training admission lineage; final publication compares that lineage and requires exact final report schema with both phase report hashes.
- Added RED-first adversarial tests for authorization phase/hash, canonical-root substitution, final report missing/extra/type/value fields, and coordinated B3 lineage substitution.

## Validation

Focused r5 guards: 4 passed.
Full two-file suite: 391 passed, 1 warning.
Canonical six-file suite: 586 passed, 3 skipped, 1 warning, 2 subtests passed.
Structural compile, diff checks, verdict-test tracking, and protected-byte checks passed.

No real model, dataset, adapter namespace, training, cleanup, commit, push, A3, or B3 execution performed.
