# Story 14.5a — Coder r3 canonical A2 admission closure

## Verdict

Implementation complete for synthetic-only TDD gate. Real Phase A2/B2 remain unauthorized. No model, dataset, adapter, training, inference, cleanup, commit, or push performed.

## Red → green

- Added red tests for the complete canonical A2 validator and full A2/B2 checkpoint/config namespace.
- Added the complete synthetic A2 fixture: exact effective pins/command, immutable identity/resource evidence, provider/update/validation/global evidence, artifact/checkpoint/config bindings, report/marker bindings, and historical manifest.
- Added mutation matrix covering effective pins, immutable identity, provider evidence, validation evidence, artifacts/checkpoints, commands, historical evidence, namespace paths, output/report paths, resume digest, contract digest, and coordinated substitutions.
- Added observer failure matrix for iterator, PID, virtual-memory, disk, and evidence-construction failures plus repeated unknown-PID skip accounting.
- Initial focused RED confirmed missing namespace entries and missing validator.
- Focused GREEN: `20 passed, 120 deselected, 1 warning`.

## Implementation

- `validate_canonical_attempt2_report()` is the single strict validator used by B2 pre-log dependency admission and runtime A2 success publication.
- Validator rejects missing/extra/substituted report, effective, immutable identity/resource, provider/update/validation/global, artifact/checkpoint/config, command, binding, contract, lifecycle, resume, and historical-manifest fields.
- `ATTEMPT2_NAMESPACE` path source now binds every A2/B2 checkpoint/config path and all output/log/report/marker/final destinations.
- Artifact validation reports and verifies adapter config plus exact checkpoint cardinality/schema/progression.
- Historical attempt-1 manifest is returned in canonical manifest form and remains separate from attempt-2 dependencies.
- Tightened provider token-count evidence to equal expected masked tokens.
- Canonical docs updated in `docs/architecture.md`, `docs/technical-spec.md`, and `docs/backlog.md`.

## Verification

- Focused pilot suite: `121 passed, 1 warning` before final mutation additions.
- Exact six-file synthetic suite: `387 passed, 3 skipped, 1 warning, 2 subtests passed in 22.83s`.
- Test files contributing to verdict are tracked by `git ls-files`.
- `git diff --check`: PASS.
- Protected direct source hashes vs HEAD: Story 14.3 smoke, segmented provider, DS4 C/Metal/CLI/server, and Metal MoE files byte-identical.

## Gate boundary

Reviewer r2 was blocked on incomplete A2 admission/schema and missing checkpoint path/mutation coverage. Those blockers are addressed in this r3 revision. Fresh independent Reviewer PASS and Test Manager GREEN remain required before any real Phase A2 authorization; Phase B2 still requires separate authorization after A2 evidence verification.
