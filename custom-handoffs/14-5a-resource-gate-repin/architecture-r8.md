# Story 14.5a — Architect r8 consumer-applicability adjudication

## Verdict

**GO for test-only repin. The three remaining REDs are legitimate executable-catalog `N/A` rows, not production defects.**

The affected keys are:

- `phase-a-step1-checkpoint`;
- `phase-a-step2-checkpoint`;
- `phase-b-step1-checkpoint`.

They name artifacts produced by the pinned training save cadence. They are not launch inputs and must not change launch-check or training argv. Their authoritative consumers are the attempt-2 phase spec, contract digest, post-training artifact validation, canonical report validation, report step/artifact bindings, success-admission boundary, and—only for Phase A artifacts—Phase B dependency admission.

Do not add CLI flags, extra log fields, shell reads, or synthetic uses merely to make these keys observable in the catalog wrapper. Do not edit production code for this disposition.

## Why catalog binding is not applicable

`_pilot_attempt2_command()` correctly binds executable inputs:

- phase output directory through `--adapter-path`;
- Phase B resume source through `--resume-adapter-file`;
- active log through `--log-path` / FD 3;
- immutable model, data, launch config, script, interpreter, and pinned training arguments.

The numbered checkpoint filenames are generated downstream from those inputs:

1. `scripts/ds4_segmented_pilot.py::_execute_training()` supplies `adapter_file=<phase output>/adapters.safetensors` and `steps_per_save=1`.
2. `vendor/mlx-lm/mlx_lm/tuner/trainer.py` writes `Path(adapter_file).parent / f"{it:07d}_adapters.safetensors"` whenever `it % steps_per_save == 0`.
3. Phase A runs two iterations, producing `0000001_adapters.safetensors` and `0000002_adapters.safetensors`.
4. Phase B runs one iteration, producing `0000001_adapters.safetensors`.
5. `_validate_artifacts()` then resolves the expected leaf paths from `phase_spec["namespace_paths"]`, requires every artifact, checks exact numbered-checkpoint cardinality, and validates progression.

Therefore, mutating only one numbered-checkpoint namespace key while leaving phase output, iteration count, and save cadence unchanged must leave launch/training argv unchanged and must fail at the post-training expected-artifact boundary. Making the catalog redirect a generated checkpoint independently would create a second naming authority and weaken the fixed-cadence contract.

The architecture rule that the central namespace feeds all consumers does not require every derived output leaf to become a CLI input. For generated leaves, the namespace is the canonical postcondition registry. Producer authority remains phase output plus pinned save cadence; validator/report/contract authority remains the exact leaf paths.

## Exact ten-key applicability matrix

`phase-a-config` and `phase-b-config` below mean produced `adapter_config.json` artifacts. They are not the immutable launch identity `--config` input.

| Namespace key | Runtime producer binding | Canonical validation / report / contract | Phase B dependency admission | Executable catalog |
|---|---|---|---|---|
| `phase-a-start-checkpoint` | Phase A output plus fixed `phase-a-start.safetensors` name | Applicable: `artifacts.start`, namespace map, contract digest, success admission | Applicable through canonical A2 validation | Applicable only to existing diagnostic namespace log `start=`; not launch/training argv |
| `phase-a-step1-checkpoint` | Phase A output plus save cadence `steps_per_save=1`, iteration 1 | Applicable: `artifacts.checkpoints[0]`, `steps[0].checkpoint`, namespace map, contract digest, success admission | Applicable through canonical A2 validation | **N/A — derived training output; no argv or runtime-log binding required** |
| `phase-a-step2-checkpoint` | Phase A output plus save cadence `steps_per_save=1`, iteration 2 | Applicable: `artifacts.checkpoints[1]`, `steps[1].checkpoint`, Phase A `resume_source`, namespace map, contract digest, success admission | Applicable through canonical A2 validation and canonical `phase-b-resume` dependency | **N/A — derived training output; no argv or runtime-log binding required** |
| `phase-a-final-checkpoint` | Phase A `TrainingArgs.adapter_file` | Applicable: `artifacts.final`, progression, namespace map, contract digest, success admission | Applicable through canonical A2 validation | Applicable only to existing diagnostic namespace log `final=`; not launch/training argv |
| `phase-a-config` | Phase A output plus fixed `adapter_config.json` runtime write | Applicable: `artifacts.config`, namespace map, contract digest, success admission | Applicable through canonical A2 validation | Applicable only to existing diagnostic namespace log `config=`; not launch `--config` |
| `phase-b-resume` | External Phase A step-2 artifact consumed before/during Phase B | Applicable: phase spec, effective command, `resume_source`, namespace map, contract digest, success admission | Applicable directly across report, marker, file SHA-256, canonical tensor digest, and resume continuity | **Applicable:** launch-check argv, training argv, and resume log |
| `phase-b-start-checkpoint` | Phase B output plus fixed `resume-start.safetensors` name | Applicable: `artifacts.start`, resume continuity, namespace map, contract digest, success admission | `N/A — produced after dependency admission` | Applicable only to existing diagnostic namespace log `start=`; not launch/training argv |
| `phase-b-step1-checkpoint` | Phase B output plus save cadence `steps_per_save=1`, iteration 1 | Applicable: `artifacts.checkpoints[0]`, `steps[0].checkpoint`, namespace map, contract digest, success admission | `N/A — produced after dependency admission` | **N/A — derived training output; no argv or runtime-log binding required** |
| `phase-b-final-checkpoint` | Phase B `TrainingArgs.adapter_file` | Applicable: `artifacts.final`, progression, namespace map, contract digest, success admission | `N/A — produced after dependency admission` | Applicable only to existing diagnostic namespace log `final=`; not launch/training argv |
| `phase-b-config` | Phase B output plus fixed `adapter_config.json` runtime write | Applicable: `artifacts.config`, namespace map, contract digest, success admission | `N/A — produced after dependency admission` | Applicable only to existing diagnostic namespace log `config=`; not launch `--config` |

## Required mutation oracles for the three `N/A` catalog rows

For each numbered-checkpoint key, retain direct isolated tests that coherently rebind the report namespace and phase-spec namespace, recompute the contract digest, and avoid generic namespace/digest rejection.

### Canonical and success-admission oracle

- Mutate exactly one numbered-checkpoint key.
- Keep phase output, iters, save cadence, identity, report path, and unrelated namespace entries unchanged.
- Reach `_validate_artifacts()` through `validate_canonical_attempt2_report()`.
- Require exact failure:

```text
missing required pilot artifact: <mutated path>
```

- Require `_write_success_evidence` call count zero.
- Require all protected phase/final report and marker destinations absent.
- For B2 negatives, require valid A2 report and marker bytes unchanged.

### Report and contract oracle

- Require `contract_digest()` to change for a coherent mutation of each key because the complete phase spec is bound into the digest.
- For valid reports, require the matching `artifacts.checkpoints[index]` path and `steps[index].checkpoint` object to equal the validated runtime artifact.
- For Phase A step 2, additionally require `report["resume_source"]` to equal the validated final numbered checkpoint and the canonical `phase-b-resume` path.

### Dependency oracle

- Retain direct Phase B dependency-admission rows for both Phase A numbered checkpoints.
- Rebind report, phase spec, contract, serialized report SHA-256, and marker contract coherently so the case reaches the named artifact boundary.
- Do not add a Phase B step-1 dependency row; that artifact does not exist before Phase B training.

## Executable-catalog test repair

Repair `test_r8_catalog_each_key_changes_observable_consumer` without changing production:

1. Keep executable baseline-versus-mutation rows for the seven applicable catalog observations:
   - `phase-a-start-checkpoint`;
   - `phase-a-final-checkpoint`;
   - `phase-a-config`;
   - `phase-b-resume`;
   - `phase-b-start-checkpoint`;
   - `phase-b-final-checkpoint`;
   - `phase-b-config`.
2. Remove the expectation that the three numbered-checkpoint mutations change launch argv, training argv, order, FD log, or status.
3. Represent those three rows explicitly as `N/A — derived training output` in the shared applicability descriptors or a focused matrix assertion.
4. Do not count assignment text as consumption.
5. Do not add numbered-checkpoint fields to the namespace log merely to make the test green.
6. Do not add unsupported or ignored checkpoint CLI flags.
7. Keep top-level `bash`, `-lc`, one-script parsing, quoting-with-spaces, launch-before-training, FD-backed logging, noclobber, and exact status propagation assertions unchanged.

This narrows the catalog oracle to actual catalog consumers. It does not remove any runtime artifact, report, contract, publication, or dependency oracle.

## Gates and verification

Coder may change tracked test code and r8 handoff notes only.

Required before review:

- targeted canonical, publication/no-write, dependency, report/contract, and applicable-catalog matrices all GREEN;
- canonical exact six-file suite with zero failures and at least the existing `470 passed` floor;
- `py_compile` for touched tests and both protected scripts;
- `git diff --check` and `git diff --cached --check`;
- direct protected-file hashes unchanged;
- every verdict-contributing test file confirmed tracked with `git ls-files`;
- no role marker staged.

Reviewer must independently verify this applicability classification against the runtime producer chain and reject any test that treats assignment presence as consumption. Test Manager must independently verify the exact revision and tracked baseline. Reviewer PASS and Test Manager GREEN remain mandatory.

## Authorization

This adjudication authorizes only the synthetic test repin needed to encode the corrected applicability matrix. No production/runtime/vendor changes, real model or dataset access, provider execution, training, inference, evidence cleanup, commit, or push are authorized.

Real Phase A2 remains blocked pending Reviewer PASS, Test Manager GREEN, and fresh explicit operator authorization. Phase B2 remains separately blocked pending verified A2 completion and separate authorization.
