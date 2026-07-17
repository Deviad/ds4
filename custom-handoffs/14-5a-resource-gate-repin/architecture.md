# Story 14.5a — Architecture: resource observer repair and immutable attempt 2

## Verdict

**GO for synthetic implementation only.**

Repair remains limited to `scripts/ds4_segmented_pilot.py`, exact attempt-2 catalog bindings, tracked synthetic tests, and durable documentation. No real asset access, training, inference, smoke, cleanup, deletion, rename, overwrite, commit, or push is authorized.

Real Phase A2 remains blocked until Coder completion, independent Reviewer PASS, Test Manager GREEN, and fresh explicit operator authorization. Phase B2 requires separate authorization after Phase A2 exits and its evidence is verified.

No ADR is required. This is a fail-closed observer repair and a new operational evidence namespace; trainer/provider/runtime architecture remains unchanged.

## 1. Failure classification and repair boundary

Attempt 1 failed before model/dataset loading and before any provider call, optimizer update, or adapter creation. `_resource_gate()` currently evaluates every process RSS inside one list comprehension. `psutil.AccessDenied(pid=0)` therefore aborts the complete scan.

The repair changes only process observation granularity:

- one candidate process is observed at a time;
- current PID is rejected before `memory_info()` access;
- only process-local `psutil.AccessDenied`, `psutil.NoSuchProcess`, and `psutil.ZombieProcess` are skipped;
- scanning continues after those skips;
- every other enumeration, PID, RSS, memory, disk, or evidence-construction error remains terminal;
- no `psutil.Error`, `OSError`, or `Exception` catch may turn an observation failure into success.

The existing strict gates remain:

```text
competitor RSS: fail only when RSS > 50 * 1024**3
available memory: pass only when available >= 32 * 1024**3
workspace/output-parent free disk: pass only when free >= 1 * 1024**3
```

No retry, fallback observer, process termination, cleanup, or threshold change.

## 2. Per-process scan design

### 2.1 Control flow

Keep `psutil.virtual_memory()` and disk observation fail closed. Replace the RSS comprehension with an explicit iterator:

1. Import `psutil`; import failure is terminal.
2. Read and integer-normalize `virtual_memory().available`; any exception is terminal with stage `virtual-memory`.
3. Create `iter(psutil.process_iter(["pid"]))`; creation or advancement failure is terminal with stage `process-enumeration`. An `AccessDenied` raised by iterator creation/advancement is not process-local and is not skipped.
4. For each returned process:
   1. Read and integer-normalize `proc.pid` inside its own boundary.
   2. If PID equals `os.getpid()`, continue without accessing `memory_info()`.
   3. Read and integer-normalize `proc.memory_info().rss` inside its own boundary.
   4. On exactly `AccessDenied`, `NoSuchProcess`, or `ZombieProcess` during PID/RSS access, record the skip and continue.
   5. On any other exception, raise terminal structured observer failure with stage and available PID.
   6. Append `[pid, rss]` to observed candidate evidence.
5. Sort observed candidates by PID then RSS before evidence publication.
6. Evaluate strict `> 50 GiB` over all successfully observed non-current candidates. A later genuine competitor remains terminal even if earlier candidates were skipped.
7. Evaluate memory and disk boundaries with inclusive pass semantics.
8. Construct and validate deterministic resource evidence. Construction/validation failure is terminal with stage `resource-evidence`.

Classify `ZombieProcess` before `NoSuchProcess`, because psutil may model zombie/disappearance classes with inheritance. PID context comes first from the successfully read PID, then from integer `exc.pid` when available. Missing PID remains `null`; it must not be invented.

### 2.2 Exact success schema

Preserve existing top-level resource keys. Add one exact skip object:

```json
{
  "available_memory": 34359738368,
  "disk_free": 1073741824,
  "competing_processes": [[123, 4096], [456, 53687091200]],
  "allowed_process_skips": {
    "total": 3,
    "counts_by_type": {
      "AccessDenied": 1,
      "NoSuchProcess": 1,
      "ZombieProcess": 1
    },
    "pids_by_type": {
      "AccessDenied": [0],
      "NoSuchProcess": [101],
      "ZombieProcess": [202]
    },
    "unknown_pid_counts_by_type": {
      "AccessDenied": 0,
      "NoSuchProcess": 0,
      "ZombieProcess": 0
    }
  }
}
```

Schema invariants:

- all three type keys always exist, including zero-valued cases;
- PID arrays contain integers only and are sorted ascending;
- duplicate skips remain duplicate observations and therefore remain present in PID arrays;
- `counts_by_type[type] == len(pids_by_type[type]) + unknown_pid_counts_by_type[type]`;
- `total == sum(counts_by_type.values())`;
- `competing_processes` contains every successfully observed non-current candidate, not only over-limit candidates, sorted deterministically;
- exception messages are not part of success evidence.

### 2.3 Exact terminal observer schema

Add `ResourceObserverError(PilotError)` carrying immutable stage/type/PID fields. `_phase_failure_report()` receives the exception object, not only `str(error)`, and adds this object only for observer failures:

```json
{
  "resource_observer_error": {
    "stage": "process-rss",
    "exception_type": "RuntimeError",
    "pid": 123
  }
}
```

Allowed stages:

```text
import
virtual-memory
process-enumeration
process-pid
process-rss
resource-evidence
disk-free
```

`pid` is an integer when known and `null` otherwise. The human-readable `error` remains stable:

```text
resource observer failed: stage=process-rss exception=RuntimeError pid=123
```

Do not publish `allowed_process_skips` as success evidence after an unknown failure. Failure evidence may include only the structured terminal observer object above; this prevents partial skips from implying a completed scan.

## 3. Immutable attempt-1 evidence

Attempt-1 paths are historical evidence, never dependencies or collision candidates for attempt 2. They must never be deleted, renamed, truncated, replaced, rewritten, or accepted by Phase B2.

Observed immutable baseline:

| Path | Bytes | SHA-256 |
|---|---:|---|
| `agent-output/cmux-14-5/phase-a-log.txt` | 29 | `4ac7319d81f9dcd344a185fd57fb3a6342ef981cbd7ca378143d1ea803eb8865` |
| `agent-output/cmux-14-5/phase-a-report.json` | 3771 | `497e5a271df5765e3af7f9795b27961e2b383886aada100d626a5e85f1fc49b1` |
| `agent-output/cmux-14-5/pilot-report.json` | 3874 | `13c8429d8ceaf11dfed24faabb9dcaa2bb9022c02882d055ba7ec0d1b4cc0b02` |
| `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-phase-a-fail` | 458 | `e19872f0d9c1d7fe90fd16aa94bef750c7cd5b9e58e608fc5ff650141ec75f54` |
| `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-fail` | 456 | `201089f5ede43a0fe839325a6feae1347ed6d3c54d020a0e498598628d1677d8` |

The implementation owns an exact `ATTEMPT1_EVIDENCE_SHA256` manifest with these paths, sizes, and hashes. Attempt-2 launch preflight chunk-hashes these five historical files and fails before attempt-2 log creation if any entry is missing or mismatched. Reports record the verified manifest under `attempt_1_historical_evidence`; this object is informational and excluded from all Phase A2 success and Phase B2 dependency satisfaction rules.

Other canonical attempt-1 names remain permanently reserved even if absent. No attempt-2 constant, catalog command, report, marker, output, or dependency may contain an attempt-1 destination.

## 4. Attempt-2 namespace model

Use an explicit attempt selector while retaining logical phase names:

```text
--attempt 2 --phase phase-a
--attempt 2 --phase phase-b
```

Reports retain `"phase": "phase-a"` or `"phase": "phase-b"` and add:

```json
{
  "attempt": 2,
  "namespace": "ds4-segmented-pilot-attempt-2"
}
```

No dynamic suffixing or registry allocation. Parser accepts attempt `2` only for new runnable pilot commands. Attempt-1 catalog steps are retired from runnable `MLX_STEPS`/`BACKEND_STEPS`/`COMMAND_STEPS`; their names and paths are reserved historical identities and are not rebound. No attempt-3 path or selector exists.

### 4.1 Exact paths

| Binding | Exact path |
|---|---|
| A2 output | `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a` |
| A2 log | `agent-output/cmux-14-5-attempt-2/phase-a-log.txt` |
| A2 report | `agent-output/cmux-14-5-attempt-2/phase-a-report.json` |
| A2 OK | `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-a-ok` |
| A2 fail | `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-a-fail` |
| B2 output | `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-b` |
| B2 resume | `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/0000002_adapters.safetensors` |
| B2 log | `agent-output/cmux-14-5-attempt-2/phase-b-log.txt` |
| B2 report | `agent-output/cmux-14-5-attempt-2/phase-b-report.json` |
| B2 OK | `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-b-ok` |
| B2 fail | `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-b-fail` |
| final report | `agent-output/cmux-14-5-attempt-2/pilot-report.json` |
| final OK | `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-ok` |
| final fail | `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-fail` |

Required checkpoint names remain exactly those in `requirements.md`; only their output-directory prefix changes.

## 5. Collision ownership and lifecycle

Centralize exact paths in one immutable namespace object. Report writers, marker helpers, collision gates, contract digests, catalog rendering, artifact validation, and Phase B2 dependency validation must consume that object. No independent string substitution.

### 5.1 Phase A2

Before opening A2 log, wrapper launch-check requires absence of all 13 attempt-2 destinations:

- A2/B2 output directories;
- A2/B2 logs;
- A2/B2/final reports;
- A2/B2 OK/fail markers;
- final OK/fail markers.

Attempt-1 paths are not in this set.

After launch-check, wrapper creates A2 log with shell noclobber and keeps its file descriptor open. `tee` writes to `/dev/fd/3`, never opens/truncates the destination pathname. The child receives exact `--log-path` and `--log-fd 3`; before and after lock acquisition it verifies `stat(log_path)` equals `fstat(log_fd)`, the file is regular, and every other attempt-2 destination remains absent. A collision returns terminal status without reports, markers, output creation, cleanup, or overwrite. Existing collision bytes remain untouched.

### 5.2 Phase B2

Before B2 log creation:

Required A2 dependencies:

- exact A2 output directory;
- exact A2 report;
- exact A2 OK marker;
- exact `0000002_adapters.safetensors` resume source;
- no A2 fail marker;
- existing strict marker → single-byte report snapshot → contract/identity/checkpoint file SHA/canonical digest binding.

Required absences:

- B2 output/log/report/OK/fail;
- final report/OK/fail.

A2 log and other required A2 artifacts are historical inputs for B2 verification only. Attempt-1 artifacts are ignored and can neither block nor satisfy B2. The same active-log file-descriptor attestation and pre/post-lock checks apply.

### 5.3 Collision failures versus started-attempt failures

- Collision or dependency rejection before launch writes no attempt-2 evidence.
- A failure after exact namespace admission consumes that phase's one attempt and writes only that phase's attempt-2 fail report/marker plus attempt-2 final failure evidence under existing durable ordering.
- No path is unlinked to make room. Existing `_remove_ok_markers()` behavior must be confined to success markers created by the same admitted attempt-2 publication sequence; it must never address attempt-1 names or pre-existing collision evidence.

## 6. CLI and catalog

Add explicit non-default catalog steps:

```text
ds4-segmented-pilot-attempt-2-phase-a
ds4-segmented-pilot-attempt-2-phase-b
```

They are excluded from `DEFAULT_BACKEND_STEPS["local-mlx"]`. Existing default `smoke-train`, `full-train`, `continue-train`, and `ds4-segmented-smoke` command strings remain byte-identical.

The pilot exposes a non-mutating launch check using the same namespace/dependency helpers as runtime:

```text
--attempt 2 --phase phase-a --launch-check-only --log-path <exact-log>
--attempt 2 --phase phase-b --launch-check-only --log-path <exact-log>
```

`--launch-check-only` performs namespace checks and attempt-1 evidence hash verification only. It does not install watchdogs, acquire the training lock, import MLX, inspect model/dataset/config, create directories, write reports/markers, or run training.

Catalog wrapper order is pinned:

1. `set -euo pipefail`;
2. exact non-mutating launch check;
3. create local report/log parent only after successful check;
4. `set -o noclobber; exec 3>"$LOG"; set +o noclobber`;
5. run exact pilot command with `--log-path "$LOG" --log-fd 3`;
6. visible multiplex through `tee /dev/fd/3`;
7. return `PIPESTATUS[0]`.

No `tee "$LOG"`, `tee -a "$LOG"`, truncating redirect, `rm`, `mv`, suffix increment, or retry loop.

Operator emits exact commands through:

```bash
python3 scripts/finetune_ds4.py emit-commands --backend local-mlx --mlx-lm-source fork ds4-segmented-pilot-attempt-2-phase-a
python3 scripts/finetune_ds4.py emit-commands --backend local-mlx --mlx-lm-source fork ds4-segmented-pilot-attempt-2-phase-b
```

The rendered training arguments remain the Story 14.5 command set, with only:

```text
--attempt 2
attempt-2 adapter path
Phase B2 exact attempt-2 resume source
attempt-2 log binding
```

## 7. Training, identity, timeout, lock, and evidence contracts

Unchanged pins:

- model `/Volumes/Data NVME/mlx-ft/ds4/model-4bit`;
- dataset `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke`;
- config `/Volumes/Data NVME/mlx-ft/ds4/lora-config.json`;
- provider SHA-256 `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518`;
- vendor outer gitlink/inner HEAD `80fab4e419a57f9465bb9e2f4e90010d645e124c`;
- sequence 4096, batch 1, LR `1e-5`, mask prompt, gradient checkpointing, segment 1, accumulation 1, seed 0, LoRA, Adam, 16 layers, 25 validation batches, world size 1;
- report/save cadence 1;
- A2: 2 calls/updates, eval cadence 2, 2700 seconds;
- B2: 1 call/update, eval cadence 1, 1500 seconds;
- total: 3 calls/updates, 4200 active seconds;
- retry `none`, fallback `none`.

Preserve exact Story 14.5:

- `.ds4-ft.lock` O_EXCL ownership token, 60-second acquisition, 2-second polling, release-only-if-owned, retained-partial quarantine behavior;
- watchdog installation/cancellation and terminal cleanup ordering;
- strict JSON duplicate-key rejection and single-byte Phase A report snapshot binding;
- adapter canonical tensor digest, checkpoint progression, callback/provider cardinality, marker/report hash bindings;
- phase report before phase marker; final report before final marker;
- adapter-weight continuity only.

The contract digest includes `attempt=2`, namespace identifier, exact A2/B2/final paths, exact log binding, and attempt-1 historical manifest. Attempt-1 report/marker/contract values can never satisfy the attempt-2 digest.

## 8. Mutation-sensitive synthetic test design

All tests use fake psutil objects and temporary paths. Import and test collection perform no `/Volumes/Data NVME/...` access.

### Resource tests

1. PID 0 `AccessDenied`: fake PID 0 raises from `memory_info`; exact count/PID evidence; later process observed.
2. `NoSuchProcess`: skipped; later process over 50 GiB still fails.
3. `ZombieProcess`: skipped; later process over 50 GiB still fails.
4. Strict RSS boundary: exactly 50 GiB passes; 50 GiB + 1 byte fails.
5. Current PID: `memory_info()` raises `AssertionError` if touched; scan succeeds without touching it, even with a fake advertised RSS above limit.
6. Unknown failures: separate fixtures for iterator creation, iterator advancement, PID property, RSS call, integer normalization, virtual memory, disk, and evidence construction. Every case fails with exact `stage`, `exception_type`, and available PID; no allowed-skip success evidence.
7. Broad-class oracle: unknown `psutil.Error` subclass and `OSError` from `memory_info()` remain terminal.
8. Memory boundary: 32 GiB passes; one byte below fails.
9. Disk boundary: 1 GiB passes; one byte below fails.
10. Determinism: shuffled process order and mixed repeated/unknown PIDs produce exact sorted lists, fixed three-key maps, and count invariants.

Use realistic fake process objects exposing a `pid` property and `memory_info()` returning an object with `.rss`; do not replace the scan with prebuilt tuples.

### Namespace and lifecycle tests

1. Pin every exact A2/B2/final path and every checkpoint path.
2. Parameterize all 13 A2 collision paths; each blocks before log/output/report/marker mutation.
3. Parameterize every B2/final destination; each blocks while valid A2 dependencies are the only accepted existing paths.
4. Substitute attempt-1 report, marker, output, or checkpoint one at a time; B2 rejects each.
5. Substitute any A2 path component, report hash, contract digest, checkpoint file hash, canonical digest, attempt number, namespace, log path, or active log FD; reject.
6. Render catalog commands and assert launch-check precedes parent creation/log FD creation; noclobber precedes training; `tee` targets only `/dev/fd/3`; no truncating destination open exists.
7. Assert old attempt-1 catalog names are not runnable/rebound, new names are non-default, and no attempt-3/dynamic suffix path exists.
8. Snapshot synthetic copies of the five attempt-1 files, run A2/B2 preflight/collision/dependency paths, and assert byte hashes unchanged.
9. Verify the exact attempt-1 baseline manifest itself; mutate path, size, or digest and turn RED.
10. Preserve terminal mutation matrix for parser/config, preflight, start-save, provider, callback, checkpoint validation, reports, markers, timeout, watchdog, lock, and final aggregation under attempt-2 paths.
11. Pin default catalog strings, Story 14.3 smoke source/catalog/report/log/markers, provider/vendor, Path A digest/count, CUDA, distributed, Metal, CPU, SSD streaming, and GGUF protected paths.
12. Run `git ls-files` for every verdict-contributing test before Coder completion and again before Reviewer/Test Manager verdicts.

TDD order: add resource and namespace tests RED; implement minimum GREEN repair; refactor only while GREEN.

## 9. Durable documentation

Coder updates canonical docs in the same slice:

- `docs/architecture.md`: replace stale Story 14.5 pending text with attempt-1 fail-closed disposition, narrow observer rule, immutable attempt-2 namespace, and adapter-only continuity boundary.
- `docs/technical-spec.md`: exact selector/catalog names, exact paths, wrapper collision order, resource success/failure schemas, attempt-1 baseline hashes, synthetic gate, authorization protocol, and non-claims.
- `docs/backlog.md`: verify BA's Story 14.5a exact WHO/WHAT/WHY entry and canonical requirements link remain intact; do not rewrite BA requirements.

No ADR. Handoff `architecture.md` is evidence, not canonical documentation.

## 10. Authorization and visible execution protocol

Synthetic implementation/review only now.

Future Phase A2:

1. Confirm exact repaired revision has Reviewer PASS and Test Manager GREEN.
2. Present emitted A2 command, namespace, identities, 2700-second timeout, and collision result.
3. Obtain fresh explicit operator authorization.
4. Run once in a visible `panel-runner`/cmux pane.
5. Poll to terminal process exit.
6. Verify A2 report/OK marker/report hash, no A2/final fail marker, exact checkpoint file/canonical hashes, two provider calls, two updates, watchdog cancellation, lock release, and attempt-1 baseline hashes.

Future Phase B2:

1. Present exact B2 command and verified A2 resume file/canonical digest.
2. Obtain separate explicit operator authorization after A2 verification.
3. Reuse visible pane only after A2 process has exited.
4. Run B2 once.
5. Verify B2/final reports and markers, one call/update, total three, total active budget, exact continuity, lock release, and non-claims.

No opaque delegation, detached PID/log execution, automatic retry, cleanup, alternate namespace, fallback, reduced parameters, alternate backend, smoke rerun, CUDA, or distributed execution.

## 11. STOP/ESCALATE

Stop and return to BA + Architect if implementation would require:

- any attempt-1 mutation or cleanup;
- broader skipped exception class;
- changed resource thresholds;
- attempt-2 path conflict or attempt-3 allocation;
- changed training identity, cardinality, budget, lock, watchdog, report, marker, or continuity contract;
- Phase B2 dependence on attempt-1 evidence;
- protected Story 14.3, Path A, vendor/provider, CUDA, distributed, Metal, CPU, SSD, GGUF, or runtime changes;
- real asset access for a test;
- untracked verdict-contributing test.

## 12. Coder sequence

1. Write tracked RED tests for resource scan and exact schemas.
2. Write tracked RED tests for attempt-2 namespace, collision order, catalog, and attempt-1 immutability.
3. Implement explicit attempt-2 namespace and resource repair minimally.
4. Preserve existing Story 14.5 terminal/identity/lock behavior under new namespace.
5. Update canonical docs.
6. Run focused synthetic suite, existing six-file gate, source/path/hash guards, `git diff --check`, and test tracking checks.
7. Produce coder handoff only. No real execution, cleanup, commit, or push.
