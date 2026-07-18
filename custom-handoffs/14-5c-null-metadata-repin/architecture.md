# Story 14.5c — null-metadata compatibility and fixed attempt-3 architecture

## 1. Verdict and authorization boundary

**GO for TDD implementation and synthetic validation only.**

No real model, dataset, adapter, provider, training, inference, Phase A3, Phase B3, cleanup, deletion, rename, overwrite, commit, or push is authorized. A3 remains blocked until tracked tests are green, Reviewer PASS and Test Manager GREEN name the same exact revision, immutable historical evidence and protected bytes verify, the exact emitted A3 wrapper is captured, and the operator explicitly authorizes one visible invocation. B3 remains separately blocked until verified A3 success and separate explicit authorization.

No ADR is required. This slice changes one safetensors compatibility predicate and repins one operational evidence namespace; trainer/provider/runtime subsystem boundaries remain unchanged.

## 2. Current failure and design boundary

Attempt 2 ran once at revision `e6d34fa03479316720430f35cc94d4606a45ef96`, completed the exact `2/2` calls and updates, and failed during `_validate_artifacts()` because all four MLX-written safetensors headers contained `"__metadata__": null`. Current `canonical_tensor_digest()` uses:

```python
metadata = header.get("__metadata__", {})
if not isinstance(metadata, dict):
    raise PilotError("safetensors __metadata__ must be an object")
```

Tensor parsing, payload reads, manifests, and digest framing had not identified a tensor defect. The repair therefore belongs only at this metadata classification seam. It must not normalize files, change tensor semantics, or create a new digest version.

Attempt 2 is consumed and immutable. Its runnable phase names and `--attempt 2` execution path must be retired. Its reports, failure markers, artifacts, pre-log record, runtime manifest, and reserved names remain historical evidence only.

## 3. Exact safetensors parser contract

### 3.1 Production predicate

Use an identity sentinel, not truthiness or coercion:

```python
_METADATA_ABSENT = object()
metadata = header.get("__metadata__", _METADATA_ABSENT)
if metadata is not _METADATA_ABSENT and metadata is not None and not isinstance(metadata, dict):
    raise PilotError("safetensors __metadata__ must be an object or null")
```

Required behavior:

| Header state | Result |
|---|---|
| key absent | accept as no metadata |
| key present with JSON `null` | accept as no metadata |
| key present with `{}` | accept |
| key present with any JSON object | accept under existing strict JSON behavior |
| string, array, integer, float, `false`, `true` | reject with `PilotError` |

`object_pairs_hook=_reject_duplicate_keys` remains the JSON decoder seam. Duplicate `__metadata__` keys remain invalid before metadata classification, including duplicate null/object combinations. Object metadata contents gain no new schema restrictions.

### 3.2 Tensor and digest invariants

No code below metadata classification changes:

- 128 MiB header bound;
- top-level object requirement;
- non-empty tensor requirement;
- exact tensor names and canonical name sort;
- dtype allowlist and byte widths;
- shape list, exact integer/non-negative dimension checks, and element-count math;
- two-integer `data_offsets`, bounds, overlap, leading-gap, internal-gap, contiguity, and trailing-data checks;
- exact dtype/shape byte-size equality;
- bounded 8 MiB payload reads from the same physical offsets;
- tensor manifest shape `{name, dtype, shape, nbytes}`;
- physical-file `file_sha256`;
- `DS4_CANONICAL_TENSOR_V1` framing over sorted tensor name, dtype, shape rank/dimensions, byte count, and exact payload bytes.

`__metadata__` remains excluded from tensor entries and digest framing. Identical tensor records and payload bytes encoded with absent, null, or object metadata must produce equal tensor manifests and equal `canonical_tensor_digest_v1`, while their physical file hashes remain different.

### 3.3 Fixture and MLX integration seams

Replace the current fixture writer's ambiguous `metadata=None` omission behavior with an explicit sentinel. Omission and explicit JSON null must produce different raw headers.

The canonical MLX integration test must use `mx.save_safetensors()` without hand-editing its output, inspect the raw header and prove `__metadata__` is present with JSON null, run production `canonical_tensor_digest()`, load through normal MLX loading, and compare exact tensor name, dtype, shape, and values. `importorskip` may preserve collection outside the MLX environment, but the canonical MLX verdict run must show this test executed, not skipped.

## 4. Fixed attempt-3 runtime model

### 4.1 Explicit attempt-3 helpers

Convert the live attempt-2 implementation to explicit attempt-3 names and constants:

```text
ATTEMPT3_NAMESPACE
attempt3_launch_identity()
attempt3_namespace()
attempt3_phase_specs()
canonical_attempt3_command()
check_attempt3_launch()
validate_canonical_attempt3_report()
```

Do not add an attempt registry, ordinal allocator, format string accepting arbitrary attempt numbers, timestamp, random suffix, fallback namespace, or attempt 4. Shared low-level helpers may remain generic only where they already express phase-independent behavior, such as canonical JSON, artifact validation, marker writing, digest framing, and file hashing.

`build_parser()` must reject `--attempt 2`. Preserve attempt 1's existing fail-closed historical path to avoid unrelated behavior change, and allow exactly attempts `(1, 3)`. `_phase_spec()`, marker selection, success/failure publication, log-FD attestation, and runtime gates route live repinned work only when `attempt == 3`.

Attempt-2 report parsing needed for immutable history remains read-only and internal. No attempt-2 helper may be reachable from `main()` as a training or launch-check path.

### 4.2 Exact namespace

`attempt3_namespace()` owns every path once:

| Key | Exact path |
|---|---|
| `phase-a-output` | `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-a` |
| `phase-a-start-checkpoint` | `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-a/phase-a-start.safetensors` |
| `phase-a-step1-checkpoint` | `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-a/0000001_adapters.safetensors` |
| `phase-a-step2-checkpoint` | `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-a/0000002_adapters.safetensors` |
| `phase-a-final-checkpoint` | `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-a/adapters.safetensors` |
| `phase-a-config` | `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-a/adapter_config.json` |
| `phase-a-log` | `agent-output/cmux-14-5-attempt-3/phase-a-log.txt` |
| `phase-a-report` | `agent-output/cmux-14-5-attempt-3/phase-a-report.json` |
| `phase-a-ok` | `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-3-phase-a-ok` |
| `phase-a-fail` | `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-3-phase-a-fail` |
| `phase-b-output` | `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-b` |
| `phase-b-start-checkpoint` | `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-b/resume-start.safetensors` |
| `phase-b-step1-checkpoint` | `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-b/0000001_adapters.safetensors` |
| `phase-b-final-checkpoint` | `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-b/adapters.safetensors` |
| `phase-b-config` | `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-b/adapter_config.json` |
| `phase-b-resume` | `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-a/0000002_adapters.safetensors` |
| `phase-b-log` | `agent-output/cmux-14-5-attempt-3/phase-b-log.txt` |
| `phase-b-report` | `agent-output/cmux-14-5-attempt-3/phase-b-report.json` |
| `phase-b-ok` | `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-3-phase-b-ok` |
| `phase-b-fail` | `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-3-phase-b-fail` |
| `final-report` | `agent-output/cmux-14-5-attempt-3/pilot-report.json` |
| `final-ok` | `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-3-ok` |
| `final-fail` | `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-3-fail` |

All specs, commands, reports, markers, artifact paths, contract digests, collision sets, and resume validation consume this exact mapping. Tests may inject temporary `repo_root` and `workspace`; production names remain fixed.

### 4.3 Phase specs and unchanged runtime identity

A3 remains `iters=2`, `steps_per_eval=2`, `timeout=2700`, `global_offset=0`. B3 remains `iters=1`, `steps_per_eval=1`, `timeout=1500`, `global_offset=2`. Preserve all model/data/config, provider/vendor, MLX `0.31.2`, LoRA/training pins, callback/validation/checkpoint cardinality, `.ds4-ft.lock`, watchdog, canonical JSON, report ordering, marker ordering, rollback, retry `none`, and fallback `none` exactly as stated in `requirements.md`.

## 5. Immutable historical evidence

### 5.1 Attempt 1

Keep `ATTEMPT1_EVIDENCE_SHA256` and its private canonical snapshot byte-for-byte unchanged. `verify_attempt1_historical_evidence()` remains a read-only path/size/SHA-256 verifier. Attempt-1 names remain reserved and can satisfy only A3/B3 historical admission, never success or resume dependencies.

### 5.2 Attempt 2

Add one private immutable expected snapshot; never regenerate expected values from current files. It contains:

- pre-log record: `agent-output/cmux-14-5-attempt-2/phase-a2-prelog-failure.md`, size `990`, SHA-256 `cdb6b89d7f7384be84c33abe993311caefe4df3f1c2b94c9ef31a3e65e0fff41`;
- runtime manifest: `agent-output/cmux-14-5-attempt-2/phase-a2-runtime-failure-manifest.json`, size `3123`, SHA-256 `2cc1d2359017cce2130a9428fd202e8c2f35b5d5810030c95aaac8b464fd3d88`;
- exact semantic facts: revision `e6d34fa03479316720430f35cc94d4606a45ef96`, command SHA-256 `38e6b1aa11d28c41c126fda6dab55cf68a9eb1d9a71fc177eec8875a3114b2de`, namespace `ds4-segmented-pilot-attempt-2`, exit `1`, `retry_performed=false`, lock acquired/released once and absent after exit, watchdog cancelled, OK markers absent;
- exact ten runtime entries:

| Path | Size | SHA-256 |
|---|---:|---|
| `/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-2/phase-a-log.txt` | 1432 | `edf1ae2215083aab2b8648403cf53f395ce39c3da2277f80fa7da2e87976734e` |
| `/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-2/phase-a-report.json` | 80454 | `d9cbc1895b62f4f182a25dcf58d851d077c6f92745bd3b0d387c78fa6dc8a500` |
| `/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-2/pilot-report.json` | 80567 | `84e8d26cf658f4ba15eda8346b4e507dc331d65da29b00b21aca1073eb97c0ce` |
| `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-a-fail` | 542 | `079c89c88bf87699562ce8e5866d88ba3e92adec6d2b0c787a481778d85b2918` |
| `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-fail` | 539 | `91b4dd994a3bd0319c5d0edf9d0c49f933cc95f2e49a1b1f359e4a92e609edfe` |
| `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/0000001_adapters.safetensors` | 22292892 | `89363e82325e095bc74f7cdb6e022280438fa1667d52abcf33f1dfb3db1a2d18` |
| `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/0000002_adapters.safetensors` | 22292892 | `2b173d701b09fc9da2bb0a9553cc7b7eb99bacebf9fa6f10e00d7b4305ba3652` |
| `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/adapter_config.json` | 339 | `13620f30b39a0de718f0769830b4b0a2233eaebea62d6fc43da08231d0a0691d` |
| `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/adapters.safetensors` | 22292892 | `2b173d701b09fc9da2bb0a9553cc7b7eb99bacebf9fa6f10e00d7b4305ba3652` |
| `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/phase-a-start.safetensors` | 22292892 | `c6dd9f67be04936c4dce58d7b3643403354516d6167e4e08d76c393890339e51` |

`verify_attempt2_historical_evidence()` must:

1. verify outer pre-log and runtime-manifest path/size/hash;
2. parse the already hash-verified manifest with strict duplicate-key rejection;
3. compare its complete schema and values to the private expected snapshot;
4. verify every one of the ten target files directly against the private tuple, not values copied from the parsed manifest;
5. verify attempt-2 phase/final OK markers, B2 output/log/report/phase-B marker destinations, and `.ds4-ft.lock` remain absent as required by the closeout; the existing A2 step-2 resume source and hashed phase-A/final failure markers remain present;
6. return a canonical historical-evidence object for A3/B3 reports and contract digests.

A coordinated mutation of manifest contents plus target paths/hashes must fail because expected values exist independently in code. Attempt-2 evidence never satisfies A3/B3 status, marker, artifact, or resume checks.

## 6. Authorization identity and pre-log admission

### 6.1 Captured authorization value

The generated A3/B3 wrapper carries one canonical compact authorization JSON value with exact keys:

```text
revision
canonical_command_sha256
pilot_source_sha256
catalog_source_sha256
protected_files_manifest_sha256
attempt2_runtime_manifest_sha256
```

`canonical_command_sha256` hashes canonical compact JSON of `canonical_attempt3_command()` without the authorization value itself, avoiding self-reference. `pilot_source_sha256` and `catalog_source_sha256` bind `scripts/ds4_segmented_pilot.py` and `scripts/finetune_ds4.py`. `protected_files_manifest_sha256` uses sorted rows `<sha256><two spaces><path>\n` over the fixed protected path set. The exact emitted wrapper and authorization JSON are the operator-review unit.

Software validates consistency; human Reviewer/Test Manager/operator lineage supplies authorization. A freshly generated self-consistent value is not, by itself, permission to execute.

### 6.2 A3 pre-log ordering

Before creating a log parent, opening the log, creating output, acquiring the lock, or writing evidence:

1. validate authorization JSON exact keys/types and attempt-3 canonical command hash;
2. require repository HEAD equals authorized revision;
3. require direct pilot/catalog hashes and protected manifest digest equal authorized values;
4. require protected paths clean through direct bytes plus repository-status check;
5. verify provider hash, vendor outer gitlink/inner HEAD, MLX version, model/data/config identities, and immutable runtime pins through the existing preflight identity boundary;
6. compare model/data/config identity to the hash-pinned attempt-2 report snapshot, not mutable current expectations;
7. verify attempt-1 and attempt-2 historical evidence;
8. verify every A3, B3, and final attempt-3 destination absent;
9. return admission evidence without writing.

The catalog wrapper then creates only the log parent, opens the exact log with noclobber FD 3, and starts the training command. Runtime repeats authorization, historical, collision, and FD attestation checks before and after lock acquisition. The active log is the only allowed occupied A3 path in those repeated checks.

### 6.3 B3 pre-log ordering

Before B3 log parent/open or any B3/final write:

1. repeat authorization, revision, source/protected, provider/vendor/model/data/config, and historical checks;
2. require exact A3 output, all four A3 safetensors, config, log, report, and A3 OK marker;
3. require A3 fail marker absent;
4. read one exact A3 report snapshot, verify marker report path/SHA/contract/namespace/attempt bindings, then run `validate_canonical_attempt3_report()`;
5. require A3 `2/2` provider/update cardinality, callbacks `[1,2]`, validations `[0,1]`, exact artifacts, lock release, watchdog cancellation, exit `0`, retry/fallback `none`;
6. require `phase-b-resume` path equals exact A3 step-2 path and matches report file SHA-256, canonical tensor digest, and tensor manifest;
7. verify every B3 and final destination absent.

A2 reports, markers, checkpoints, or namespace values fail exact attempt/namespace/path checks and cannot satisfy any B3 dependency.

## 7. Canonical reports, markers, and publication

### 7.1 A3 report

Retain the strict existing success-report schema and replace only live attempt bindings:

```text
attempt = 3
namespace = ds4-segmented-pilot-attempt-3
attempt_1_historical_evidence = exact verified five-entry manifest
attempt_2_historical_evidence = exact canonical verifier result
```

Effective pins, command, identity, provider evidence, optimizer/update records, validation evidence, artifacts, local/global mapping, lifecycle, output/log/report/marker/final paths, non-claims, retry/fallback, and contract digest remain exact. The contract digest binds both historical evidence objects, every attempt-3 namespace path, active log, authorization value, effective pins, and immutable identity.

`validate_canonical_attempt3_report()` remains the single validator used before A3/B3 success publication and by B3 dependency admission. It rejects missing or extra keys and all attempt/path/identity/cardinality substitutions.

### 7.2 Markers and final report

A3 phase marker binds exact report path/hash, output, contract, attempt `3`, namespace, exit `0`, and resume-source path/file hash/canonical digest. B3 phase marker binds exact B3 report. Final report aggregates only canonical A3 and B3 reports; final marker binds final report path/hash and attempt-3 namespace.

Publication ordering remains:

```text
phase report → final report when applicable → phase OK marker → final OK marker
```

Any publication failure rolls back only attempt-3 OK markers created by that publication sequence, then writes attempt-3 failure evidence under existing fail-atomic ordering. Invalid metadata can never leave an OK marker or success report accepted by downstream validation. Failure publication must not overwrite historical attempt-1/attempt-2 evidence or an existing attempt-3 collision.

## 8. Catalog retirement and wrapper contract

Replace only these `MLX_STEPS`, `BACKEND_STEPS`, `COMMAND_STEPS`, and catalog entries:

```text
remove ds4-segmented-pilot-attempt-2-phase-a
remove ds4-segmented-pilot-attempt-2-phase-b
add    ds4-segmented-pilot-attempt-3-phase-a
add    ds4-segmented-pilot-attempt-3-phase-b
```

Both A3/B3 steps remain excluded from `DEFAULT_BACKEND_STEPS["local-mlx"]`. `emit-commands` and `run-command` must reject both attempt-2 names as unknown/disallowed and accept only explicit attempt-3 names. Help/choice surfaces must contain no runnable attempt-2 name.

Exact emission commands:

```bash
python3 scripts/finetune_ds4.py emit-commands --backend local-mlx --mlx-lm-source fork ds4-segmented-pilot-attempt-3-phase-a
python3 scripts/finetune_ds4.py emit-commands --backend local-mlx --mlx-lm-source fork ds4-segmented-pilot-attempt-3-phase-b
```

Wrapper order remains exact:

```text
set -euo pipefail
→ fixed A3/B3 authorization + collision/history/identity launch check
→ mkdir log parent only after admission
→ set -o noclobber; exec 3>"$LOG"; set +o noclobber
→ exact --attempt 3 training command with --log-path and --log-fd 3
→ visible 2>&1 | tee /dev/fd/3
→ status=${PIPESTATUS[0]}
→ close FD and exit exact status
```

No destination-path `tee`, append/truncating open, `rm`, `mv`, cleanup, retry loop, suffixing, fallback, or alternate command.

Byte-exact protected default strings: `smoke-train`, `smoke-train-2048`, `full-train`, `continue-train`, `ds4-segmented-smoke`, and every unrelated backend command.

## 9. Exact mutation matrix

### 9.1 Parser and digest

| Mutation | Required oracle |
|---|---|
| metadata omitted / `null` / `{}` / non-empty object | accept |
| `""` / non-empty string / `[]` / non-empty array / integer / float / `false` / `true` | `PilotError` |
| duplicate `__metadata__` keys | duplicate-key `PilotError` |
| same tensors under omitted/null/object | equal tensor manifest and canonical digest; unequal physical SHA-256 |
| tensor name | manifest/digest change |
| dtype | rejection or manifest/digest change under valid byte sizing |
| shape | rejection or manifest/digest change |
| leading gap, internal gap, overlap, out-of-bounds, trailing bytes | rejection |
| payload byte | file hash and canonical digest change |

### 9.2 Artifact and publication

For each required A3 checkpoint position, replace only metadata with each invalid type and require `_validate_artifacts()` failure with offending bytes unchanged. Run valid absent/null/object combinations through exact Phase A progression and schema checks. Independently mutate checkpoint name/cardinality/schema/payload/progression and require existing failures.

Inject failure at every report/final-report/phase-marker/final-marker write boundary. Require no success marker survives, failure evidence remains attempt-3-bound, and pre-existing historical/collision bytes remain unchanged.

### 9.3 Namespace, history, identity, and catalog

Parameterize every A3/B3/final namespace key as a collision and as a path substitution. Mutate every attempt-1 and attempt-2 path, size, and hash; mutate attempt-2 manifest semantic fields, one of ten entries, outer manifest bytes, pre-log bytes, OK-marker absence, lock absence, and reserved B2 absence. Every mutation must reject before attempt-3 writes.

Mutate one authorization/revision/command/pilot/catalog/protected/provider/vendor/model/data/config identity at a time. Assert A3 log, parent, output, lock, report, and markers remain absent. For B3, additionally mutate A3 report hash, marker binding, contract, attempt, namespace, command, artifact path/hash/digest/manifest, cardinality, lifecycle, fail-marker absence, and each B3/final collision.

Catalog tests must prove:

- both attempt-2 names absent from all runnable tuples/catalog/help/default surfaces;
- `emit-commands` and `run-command` reject attempt-2 names;
- only fixed A3/B3 names added;
- A3/B3 remain non-default;
- unrelated/default command strings are byte-identical;
- rendered wrapper contains `--attempt 3`, exact paths, authorization, noclobber FD open, `/dev/fd/3`, and `PIPESTATUS[0]` in required order;
- no `attempt-4`, allocator, timestamp, random, cleanup, append, fallback, or retry token/path exists.

## 10. TDD sequence

1. Verify `tests/test_ds4_segmented_pilot.py` and `tests/test_finetune_ds4.py` are tracked with `git ls-files`.
2. Add sentinel fixture writer and direct production parser matrix; capture RED for explicit null.
3. Add accepted-variant manifest/digest/file-hash equality tests and independent tensor mutation oracles.
4. Add MLX-written null integration; capture RED against production parser.
5. Add `_validate_artifacts()` null/absent/object positives, invalid-type matrix, byte-unchanged failures, and tensor progression/schema mutations.
6. Add immutable attempt-2 snapshot/verifier tests, including coordinated manifest/target mutation.
7. Add fixed attempt-3 namespace/spec/command/catalog tests and attempt-2 retirement RED.
8. Add A3/B3 collision, authorization, history, report/marker, resume, publication rollback, and no-write matrices.
9. Implement only metadata sentinel compatibility, fixed attempt-3 live repin, read-only attempt-2 history verifier, and required catalog bindings.
10. Run focused GREEN, then canonical six-file regression, `py_compile`, `git diff --check`, tracking checks, and protected-byte checks.
11. Reviewer and Test Manager independently run the same exact revision and block any untracked verdict-contributing test.

Focused commands:

```bash
PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" PYTHONDONTWRITEBYTECODE=1 \
python-envs/mlx/.venv/bin/python -m pytest -q \
  tests/test_ds4_segmented_pilot.py tests/test_finetune_ds4.py \
  -k 'metadata or attempt3 or attempt2_historical or publication or catalog'
```

Canonical regression:

```bash
PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD" PYTHONDONTWRITEBYTECODE=1 \
python-envs/mlx/.venv/bin/python -m pytest -q \
  tests/test_ds4_segmented_pilot.py \
  tests/test_ds4_segmented_smoke.py \
  tests/test_finetune_ds4.py \
  tests/test_mlx_lm_source.py \
  tests/test_ds4_segmented_loss_and_grad.py \
  tests/test_ds4_gguf_base_smoke.py
```

Structural checks:

```bash
PYTHONDONTWRITEBYTECODE=1 python-envs/mlx/.venv/bin/python -m py_compile \
  scripts/ds4_segmented_pilot.py scripts/finetune_ds4.py \
  tests/test_ds4_segmented_pilot.py tests/test_finetune_ds4.py
git diff --check
git ls-files --error-unmatch -- \
  tests/test_ds4_segmented_pilot.py tests/test_finetune_ds4.py
```

## 11. Scope and protected bytes

Permitted implementation/test/documentation scope:

```text
scripts/ds4_segmented_pilot.py
scripts/finetune_ds4.py
tests/test_ds4_segmented_pilot.py
tests/test_finetune_ds4.py
docs/architecture.md
docs/technical-spec.md
custom-handoffs/14-5c-null-metadata-repin/*
.cmux-status/architect.done
```

Current mutable baselines for scope accounting:

| File | Size | SHA-256 |
|---|---:|---|
| `scripts/ds4_segmented_pilot.py` | 116670 | `c15f5e0f4c02f5923f7a49be0c4efa0db96b55d640f03c597c4c187b11cbb402` |
| `scripts/finetune_ds4.py` | 272056 | `de303d1166f2ecd2428dc15f63c8be8ed22e9f30274008afd35be458ef4b188b` |
| `tests/test_ds4_segmented_pilot.py` | 159541 | `84a251fead49171114c74aff37c6b335a9ee6beb0ccd48b74ecb72b58889e905` |
| `tests/test_finetune_ds4.py` | 66447 | `ee51931e5c838af5c8740c80fa48581a8601938a4198682c52ea9ed750822065` |

Protected byte baselines:

| Protected object | SHA-256 / identity |
|---|---|
| `scripts/ds4_segmented_smoke.py` | `ec17950d985578f3cd97b51734527bfc47e6c399ebe84fad8ffcb12beae18ad8` |
| `python-envs/mlx/src/ds4_ft_mlx/segmented_loss_and_grad.py` | `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518` |
| 22 root `ds4*.{c,h,m,cu}` files, sorted manifest | `07e2e75adeccb4f9b86f12924f41eb1d17d0ffcbda4b1ad0d76e8866b5bd085b` |
| 19 `metal/**` files, sorted manifest | `c67e0ed61758ccd4bd1faca421b697a749fb3a10328f9b2fb4d27b994f0ba13f` |
| vendor outer gitlink and inner HEAD | `80fab4e419a57f9465bb9e2f4e90010d645e124c` |
| attempt-2 pre-log record | `cdb6b89d7f7384be84c33abe993311caefe4df3f1c2b94c9ef31a3e65e0fff41` |
| attempt-2 runtime manifest | `2cc1d2359017cce2130a9428fd202e8c2f35b5d5810030c95aaac8b464fd3d88` |

The five attempt-1 hashes and ten attempt-2 runtime hashes in Sections 5.1–5.2 are immutable. No provider/vendor/site-package/model/dataset/config, Story 14.3, Path A, CUDA, ROCm, distributed, Metal, CPU, SSD streaming, GGUF, CLI/server, or inference source change is permitted.

## 12. Canonical documentation

Update `docs/architecture.md` to record null-as-no-metadata compatibility, consumed attempt 2, fixed attempt-3 namespace, strict historical admission, and unchanged digest/training boundaries.

Update `docs/technical-spec.md` with exact A3/B3 paths, catalog commands, collision/pre-log order, historical verifier, publication contract, authorization lineage, and test gates. Mark the design as implementation-pending until Coder/Reviewer/Test Manager close it; do not imply real A3/B3 authorization.

## 13. STOP/ESCALATE

Stop for any need to accept metadata beyond absent/null/object; weaken duplicate-key/tensor/layout/I/O checks; alter digest v1; normalize or rewrite artifacts; make attempt 2 runnable; introduce attempt 4/dynamic naming; mutate or clean history; permit A2 evidence as A3/B3 success; change model/data/config/provider/vendor/version/training pins, budgets, cardinality, lock, watchdog, logging, retry, fallback, or publication semantics; touch protected runtime paths; access real assets during implementation tests; run A3/B3; or rely on an untracked verdict test.

## 14. Coder handoff

Implement Section 10 in strict RED → minimal GREEN order. Keep historical evidence byte-identical. Produce synthetic evidence only. Do not access real assets, run A3/B3, clean collisions, commit, or push.
