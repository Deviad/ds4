# Story 14.5c — safetensors null-metadata compatibility + fixed attempt-3 repin requirements

## BA verdict and authorization boundary

**GO for requirements, architecture, TDD implementation, and synthetic validation only.**

This handoff does not authorize real model, dataset, adapter, provider, training, inference, Phase A3, Phase B3, cleanup, deletion, rename, overwrite, commit, or push.

A future Phase A3 invocation remains blocked until the compatibility repair and fixed attempt-3 repin are implemented, all verdict tests are tracked and green, an independent Reviewer returns PASS, a Test Manager returns GREEN on the exact same revision, the repository and immutable historical evidence pass fresh exact-revision gates, and the operator explicitly authorizes exactly one visible A3 invocation bound to that revision and exact emitted command. Phase B3 remains separate and unauthorized until A3 success evidence is independently verified and the operator gives separate explicit authorization.

## Trackable user story

As a DS4 fine-tuning operator (WHO), I want canonical safetensors validation to accept MLX-written null metadata and the consumed pilot to be repinned to one fixed attempt-3 namespace (WHAT), so that valid adapter tensors can pass fail-closed publication without weakening tensor integrity, historical evidence, collision safety, or one-invocation authorization (WHY).

## Fixed failure evidence

- Attempt-2 Phase A ran once at revision `e6d34fa03479316720430f35cc94d4606a45ef96`; its authorization is consumed.
- It completed exactly `2/2` provider calls and optimizer updates in `2444.2739184170496s`, within the `2700s` Phase A budget.
- It saved `phase-a-start.safetensors`, `0000001_adapters.safetensors`, `0000002_adapters.safetensors`, `adapters.safetensors`, and `adapter_config.json`.
- Loss / validation loss: iteration 1 `19.334` / `19.553`; iteration 2 `17.648` / `19.841`.
- Post-training validation exited `1` with exact error `safetensors __metadata__ must be an object`.
- Every one of the four safetensors headers contains JSON `"__metadata__": null`.
- Phase A2 and final failure reports/markers exist; Phase A2 and final OK markers are absent.
- Lock acquired once, released once, and absent after exit; watchdog cancelled; no retry performed.
- Attempt-2 is consumed, contains failed-run evidence, and is immutable. Phase B2 remains blocked permanently.
- These facts establish no convergence, quality, optimizer/RNG/dataset-cursor/scheduler/global-step continuity, resume continuity, or full-training readiness.

Canonical runtime closeout: `custom-handoffs/14-5b-a2-runtime-failure/requirements.md` and commit `d8dc248c31db2a4d81633440034221a79f117649`.

## R14.5c-1 — exact safetensors `__metadata__` compatibility

Apply one narrow rule in the canonical parser/digest boundary `scripts/ds4_segmented_pilot.py::canonical_tensor_digest()`:

1. Header key `__metadata__` absent means no metadata and remains accepted.
2. Header key `__metadata__` present with JSON `null` means no metadata and becomes accepted.
3. Header key `__metadata__` present with a JSON object remains accepted under all existing strict parser rules.
4. Every other JSON value fails closed: string, array, integer, floating-point number, boolean, and any non-object/non-null value accepted by the JSON decoder.
5. Do not coerce falsey values, use truthiness as a type test, convert a value to `{}`, drop an invalid key, or fall back to a permissive parser.
6. Duplicate-key rejection remains exact. Duplicate `__metadata__` keys remain invalid even if one or both values would otherwise be accepted.
7. `__metadata__` remains excluded from tensor entries and from `canonical_tensor_digest_v1`; accepting null does not create a tensor, report field, semantic variant, or new digest version.
8. Object metadata remains ignored by the canonical tensor digest exactly as before. Do not introduce new restrictions on object contents beyond existing strict behavior in this slice.

Equivalent acceptance predicate:

```text
absent OR value is JSON null OR value is JSON object
```

No other value passes.

## R14.5c-2 — tensor and digest invariants

For an accepted file, metadata representation must not change tensor interpretation or loadability:

- tensor names remain exact strings and sort canonically by name;
- dtype allowlist and byte widths remain unchanged;
- shapes, non-negative integer dimensions, and element-count math remain unchanged;
- `data_offsets` type, bounds, overlap, gap, contiguity, and trailing-data checks remain unchanged;
- tensor payload bytes remain read from the same offsets in bounded chunks;
- tensor manifest remains exactly `{name, dtype, shape, nbytes}` per tensor;
- `file_sha256` continues to reflect physical header differences;
- `canonical_tensor_digest_v1` continues to frame sorted tensor name, dtype, shape, byte count, and exact payload bytes only;
- identical tensors encoded with absent, null, or object metadata produce identical tensor manifests and identical canonical tensor digests;
- changing any tensor name, dtype, shape, effective offset/layout, byte count, or payload bytes remains observable through rejection, manifest change, file hash change, or canonical digest change as required by the existing contract.

Do not weaken the 128 MiB header bound, top-level-object requirement, JSON duplicate-key rejection, non-empty tensor requirement, schema checks, exact byte-size check, contiguous layout requirement, bounded read behavior, or OSError fail-closed handling.

## R14.5c-3 — mutation-sensitive parser, artifact, and publication tests

All verdict tests must exercise production boundaries with byte-level fixtures. A source-text assertion, copied helper predicate, or mock that returns its own expected result is insufficient.

### Direct canonical-parser matrix

Use a fixture writer that distinguishes omitted metadata from explicit JSON null. With identical tensor schema and payload, prove:

| `__metadata__` representation | Required result |
|---|---|
| absent | accepted |
| `null` | accepted as no metadata |
| `{}` | accepted |
| non-empty object | accepted under existing rules |
| `""` and non-empty string | `PilotError` |
| `[]` and non-empty array | `PilotError` |
| integer and floating-point number | `PilotError` |
| `false` and `true` | `PilotError` |
| duplicate `__metadata__` key | duplicate-key failure |

For the three accepted representations, assert exact tensor count, tensor manifest, names, dtypes, shapes, byte counts, and equal `canonical_tensor_digest_v1`; assert physical file SHA-256 differs when headers differ. Mutate tensor name, dtype, shape, offsets/layout, and payload independently to prove metadata acceptance cannot mask tensor drift.

### Loadability integration

In the canonical MLX test environment, create a tiny MLX-written safetensors file that exhibits `__metadata__: null`, run `canonical_tensor_digest()`, load it through the normal MLX/safetensors loader, and compare exact tensor name, dtype, shape, and values. This remains tiny synthetic execution only: no model, dataset, adapter namespace, or provider access.

### Artifact-validation boundary

Exercise `_validate_artifacts()` through temp-only Phase A fixtures containing start, step-1, step-2, final, and config artifacts:

- all four null-metadata safetensors pass while retaining exact checkpoint cardinality, names, SHA-256 values, tensor manifests, and canonical digests;
- absent and object metadata remain green;
- each invalid metadata type in any one required checkpoint fails closed;
- failure occurs without rewriting, deleting, or normalizing the offending artifact;
- tensor mutations still fail existing progression/schema/cardinality checks.

### Report/marker publication boundary

Exercise the real success/failure publication seams in temp-only namespaces:

- valid null-metadata artifacts permit a canonical success report and correctly bound phase/final OK markers only after all existing gates pass;
- invalid metadata never leaves any success marker or success report that can satisfy a downstream gate;
- rollback removes partial OK evidence if report/marker publication fails;
- failure evidence remains bound to the exact failure report and does not overwrite historical evidence;
- downstream Phase B3 validation accepts only exact A3 success evidence, never a null-metadata parser shortcut or an A2 report/marker.

## R14.5c-4 — fixed immutable attempt-3 namespace

Attempt 3 is the only new namespace. No attempt 4, dynamic allocator, timestamp, random suffix, alternate path, or fallback namespace may exist.

Canonical namespace field:

```text
ds4-segmented-pilot-attempt-3
```

### Phase A3

```text
phase: phase-a
catalog command: ds4-segmented-pilot-attempt-3-phase-a
output: /Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-a
log: agent-output/cmux-14-5-attempt-3/phase-a-log.txt
report: agent-output/cmux-14-5-attempt-3/phase-a-report.json
OK marker: /Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-3-phase-a-ok
fail marker: /Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-3-phase-a-fail
```

Required Phase A3 artifacts:

```text
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-a/phase-a-start.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-a/0000001_adapters.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-a/0000002_adapters.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-a/adapters.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-a/adapter_config.json
```

### Phase B3

```text
phase: phase-b
catalog command: ds4-segmented-pilot-attempt-3-phase-b
output: /Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-b
resume source: /Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-a/0000002_adapters.safetensors
log: agent-output/cmux-14-5-attempt-3/phase-b-log.txt
report: agent-output/cmux-14-5-attempt-3/phase-b-report.json
OK marker: /Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-3-phase-b-ok
fail marker: /Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-3-phase-b-fail
```

Required Phase B3 artifacts:

```text
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-b/resume-start.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-b/0000001_adapters.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-b/adapters.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-b/adapter_config.json
```

### Final attempt-3 evidence

```text
final report: agent-output/cmux-14-5-attempt-3/pilot-report.json
final OK marker: /Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-3-ok
final fail marker: /Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-3-fail
```

Reports, markers, marker hashes, commands, contract digests, phase dependencies, namespace maps, and resume validation must bind these exact paths.

## R14.5c-5 — catalog retirement and exact operator commands

Remove the consumed attempt-2 commands from every runnable catalog/choice/help/default surface:

```text
ds4-segmented-pilot-attempt-2-phase-a
ds4-segmented-pilot-attempt-2-phase-b
```

They must no longer be emitted or executable through `MLX_STEPS`, `BACKEND_STEPS`, `COMMAND_STEPS`, `emit-commands`, or `run-command`. Retirement must not delete or reinterpret attempt-2 reports, artifacts, markers, parsers, or evidence validators needed for historical verification.

Add only these explicit, non-default runnable commands:

```text
ds4-segmented-pilot-attempt-3-phase-a
ds4-segmented-pilot-attempt-3-phase-b
```

Exclude both from `DEFAULT_BACKEND_STEPS["local-mlx"]`. Preserve byte-exact default `smoke-train`, `smoke-train-2048`, `full-train`, `continue-train`, `ds4-segmented-smoke`, and unrelated backend commands.

Exact emission commands:

```bash
python3 scripts/finetune_ds4.py emit-commands --backend local-mlx --mlx-lm-source fork ds4-segmented-pilot-attempt-3-phase-a
python3 scripts/finetune_ds4.py emit-commands --backend local-mlx --mlx-lm-source fork ds4-segmented-pilot-attempt-3-phase-b
```

Rendered wrappers must use exact `--attempt 3`, phase, log, output, and B3 resume bindings. Preserve wrapper ordering: collision/evidence launch check, parent creation only after admission, noclobber FD log open, exact pilot command, visible `tee /dev/fd/3`, and `PIPESTATUS[0]`. No truncating/append log open, `rm`, `mv`, cleanup, suffixing, fallback, or retry loop.

## R14.5c-6 — collision and no-write semantics

Before A3 opens a log, creates a parent/output, acquires the training lock, or writes evidence:

1. Every A3, B3, and final attempt-3 output/report/log/marker path must be absent.
2. Exact attempt-1 and attempt-2 historical manifests must verify.
3. Repository HEAD, protected-file hashes, pilot/catalog source hashes, provider hash, vendor pin, model/data/config identities, and emitted command must equal the exact independently reviewed revision and authorization record.
4. Any collision, historical drift, missing historical file, hash mismatch, dirty protected path, revision mismatch, or command mismatch blocks with zero attempt-3 writes.

Before B3 opens a log or writes anything:

1. Canonical A3 report and A3 OK marker must exist and pass strict report/marker/hash/identity/cardinality/command/namespace validation.
2. A3 fail marker must be absent.
3. Exact A3 step-2 resume source must match the report SHA-256, canonical tensor digest, tensor manifest, and path binding.
4. Every B3 and final attempt-3 destination must be absent.
5. Attempt-1 and attempt-2 historical manifests must still verify.
6. Any failure blocks with zero B3/final writes.

Never clean, delete, move, rename, truncate, overwrite, or reuse attempt-1 or attempt-2 evidence. Never delete an attempt-3 collision to make admission pass. Any A3 or B3 exit consumes that phase's authorization and namespace; no automatic or manual in-place retry follows.

## R14.5c-7 — immutable attempt-1 and attempt-2 bindings

### Attempt 1

Preserve the exact existing five-entry `ATTEMPT1_EVIDENCE_SHA256` manifest:

| Path | Size | SHA-256 |
|---|---:|---|
| `agent-output/cmux-14-5/phase-a-log.txt` | 29 | `4ac7319d81f9dcd344a185fd57fb3a6342ef981cbd7ca378143d1ea803eb8865` |
| `agent-output/cmux-14-5/phase-a-report.json` | 3771 | `497e5a271df5765e3af7f9795b27961e2b383886aada100d626a5e85f1fc49b1` |
| `agent-output/cmux-14-5/pilot-report.json` | 3874 | `13c8429d8ceaf11dfed24faabb9dcaa2bb9022c02882d055ba7ec0d1b4cc0b02` |
| `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-phase-a-fail` | 458 | `e19872f0d9c1d7fe90fd16aa94bef750c7cd5b9e58e608fc5ff650141ec75f54` |
| `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-fail` | 456 | `201089f5ede43a0fe839325a6feae1347ed6d3c54d020a0e498598628d1677d8` |

All canonical attempt-1 output/report/log/marker names remain reserved. Attempt-1 evidence may satisfy only historical verification, never A3/B3 success or resume dependencies.

### Attempt 2

Bind the attempt-2 runtime evidence to:

- `agent-output/cmux-14-5-attempt-2/phase-a2-prelog-failure.md`: size `990`, SHA-256 `cdb6b89d7f7384be84c33abe993311caefe4df3f1c2b94c9ef31a3e65e0fff41`;
- `agent-output/cmux-14-5-attempt-2/phase-a2-runtime-failure-manifest.json`: size `3123`, SHA-256 `2cc1d2359017cce2130a9428fd202e8c2f35b5d5810030c95aaac8b464fd3d88`;
- all ten exact path/size/SHA-256 entries inside that runtime manifest, including phase log/report/final report, both failure markers, four safetensors artifacts, and `adapter_config.json`;
- revision `e6d34fa03479316720430f35cc94d4606a45ef96`, command SHA-256 `38e6b1aa11d28c41c126fda6dab55cf68a9eb1d9a71fc177eec8875a3114b2de`, namespace `ds4-segmented-pilot-attempt-2`, exit `1`, absent OK markers, released lock, cancelled watchdog, and `retry_performed: false`.

The committed runtime manifest is canonical; implementation must not regenerate it from mutable files and then trust the regenerated result. A private immutable expected snapshot or equivalently mutation-resistant constant must make coordinated in-memory/path/hash substitution fail.

Attempt-2 Phase B names remain retired and reserved. Attempt-2 evidence may satisfy only historical verification, never A3/B3 success, resume, or publication gates.

## R14.5c-8 — unchanged runtime identity, budgets, and cardinality

Only metadata compatibility and attempt ordinal/path/catalog bindings may change. Preserve:

- model `/Volumes/Data NVME/mlx-ft/ds4/model-4bit`;
- dataset `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke`;
- config `/Volumes/Data NVME/mlx-ft/ds4/lora-config.json`;
- DS4 segmented provider SHA-256 `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518`;
- vendor outer gitlink and inner HEAD `80fab4e419a57f9465bb9e2f4e90010d645e124c`;
- exact MLX version `0.31.2` from `importlib.metadata.version("mlx")`;
- sequence length `4096`, batch size `1`, learning rate `1e-5`, mask prompt enabled, gradient checkpointing enabled, segment size `1`, accumulation `1`, seed `0`, LoRA, Adam, `16` layers, `25` validation batches, world size `1`;
- report/save cadence `1`, `report_to=None`, `project_name=None`, `trust_remote_code=False`, `lr_schedule=None`, `clear_cache_threshold=0`;
- `.ds4-ft.lock`, watchdog, identity, resource gate, report/marker ordering, rollback, canonical JSON, and failure-publication semantics;
- retry `none`, fallback `none`.

A3 remains exactly:

```text
2 provider calls
2 optimizer updates
2 completed-update callbacks at local steps [1, 2]
validation callbacks at iterations [0, 1]
checkpoints: start + step 1 + step 2 + final
steps_per_eval: 2
timeout: 2700s
global offset: 0
```

B3 remains exactly:

```text
1 provider call
1 optimizer update
1 completed-update callback at local step [1]
validation callback at iteration [0]
checkpoints: resume-start + step 1 + final
steps_per_eval: 1
timeout: 1500s
global offset: 2
```

Total remains `3` calls/updates and `4200s` active budget. No reduced length, changed validation cadence, alternate backend, changed provider, changed vendor, or cardinality relaxation.

## R14.5c-9 — authorization lineage

Phase A3 authorization requires all of the following on one exact fresh revision:

1. TDD RED evidence against current null rejection.
2. Minimal GREEN compatibility and fixed attempt-3 repin.
3. Tracked focused and canonical regression suites green.
4. Independent Reviewer PASS and Test Manager GREEN on the same revision.
5. Protected-file hashes and immutable attempt-1/attempt-2 manifests green.
6. Exact emitted A3 command captured and bound to that revision.
7. Fresh explicit operator authorization for one visible A3 invocation.

Any A3 exit consumes authorization. Success is not inferred from updates alone; canonical artifacts, report, OK marker, lock release, cardinality, digest, identity, namespace, and absent failure evidence must all verify independently.

Only after verified A3 success may the operator separately authorize exactly one B3 invocation bound to the same implementation revision and exact B3 command. Any B3 exit consumes that authorization. Neither authorization permits retry, fallback, cleanup, attempt 4, or another phase.

## R14.5c-10 — implementation scope and protected paths

Expected minimal implementation/test scope:

- `scripts/ds4_segmented_pilot.py`;
- `scripts/finetune_ds4.py`;
- `tests/test_ds4_segmented_pilot.py`;
- `tests/test_finetune_ds4.py`;
- canonical documentation required by the slice.

Architecture may narrow this scope but may not broaden it to provider/vendor/site-package/model/dataset code without STOP/ESCALATE and a requirements repin.

No model/data/config mutation; no MLX/MLX-LM site-package edit; no provider/vendor edit; no attempt-1/attempt-2 evidence edit; no Path A, Story 14.3 smoke, CUDA, distributed, Metal, CPU, SSD streaming, GGUF, inference, or DS4 runtime edit.

Use TDD: write failing mutation-sensitive tests first, implement the smallest green change, then refactor only while green. Every verdict-contributing test file must be tracked and verified with `git ls-files`; Reviewer and Test Manager must block reproducibility claims for any untracked verdict file.

## Acceptance criteria

1. Exact WHO/WHAT/WHY user story and authorization boundary recorded.
2. Absent and JSON-null `__metadata__` accepted as no metadata; object accepted under existing rules; every other JSON type fails closed.
3. Duplicate-key, header, tensor schema, dtype, shape, offset, byte-size, contiguity, payload, and I/O guards remain strict.
4. Accepted absent/null/object variants preserve exact tensor manifest and `canonical_tensor_digest_v1`; physical file SHA remains physical-file-sensitive.
5. Tiny MLX-written null-metadata safetensors passes canonical digest and normal load with exact tensor values.
6. Mutation-sensitive direct-parser, artifact-validation, report/marker-publication, and downstream-dependency tests exercise production boundaries.
7. Fixed attempt-3 A3/B3/final paths and catalog names match this document exactly.
8. Attempt-2 runnable catalog commands are retired; attempt-3 commands remain explicit and non-default; unrelated/default commands remain byte-identical.
9. A3 admission requires every A3/B3/final destination absent, exact reviewed revision/command, and immutable attempt-1/attempt-2 evidence; any mismatch writes nothing.
10. B3 admission requires exact verified A3 success and absent B3/final destinations; A2 cannot satisfy any dependency.
11. Attempt-1 and attempt-2 files, hashes, paths, markers, reports, artifacts, and reserved names remain immutable; no delete/move/overwrite/reuse.
12. Model/data/config/provider/vendor/version/training pins, A3 `2/2700s`, B3 `1/1500s`, total `3/4200s`, callback/validation/checkpoint cardinality, lock, watchdog, report, digest, and marker contracts remain unchanged.
13. No dynamic suffix, attempt 4, fallback, retry, cleanup, parameter reduction, alternate backend, or identity substitution exists.
14. Reviewer PASS and Test Manager GREEN on one exact revision plus fresh explicit operator authorization are mandatory before one A3 invocation.
15. B3 remains separately blocked until A3 success is independently verified and separately authorized.
16. No real access/execution, cleanup, code implementation, commit, or push occurs in this BA slice.

## STOP/ESCALATE

STOP for any proposal to accept metadata types beyond absent/null/object; weaken duplicate-key/tensor/layout checks; alter digest v1 semantics; normalize or rewrite artifacts; edit MLX/MLX-LM/provider/vendor/model/data/config; mutate historical evidence; reuse attempt 2; introduce attempt 4/dynamic naming/fallback/retry; change runtime pins/budgets/cardinality; allow A2 evidence to satisfy A3/B3 success; run real A3/B3 without exact same-revision double-green and explicit phase authorization; or rely on an untracked verdict test.
