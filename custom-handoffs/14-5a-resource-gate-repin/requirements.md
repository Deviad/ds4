# Story 14.5a — Resource observer repair + immutable attempt-2 repin

## Authorization boundary

User authorized requirements/design and synthetic repair work after Story 14.5 Phase A attempt 1 failed closed. This BA slice defines the repair and a new immutable attempt-2 namespace only.

No model, dataset, adapter, training, inference, smoke, CUDA, distributed, network, or protected GGUF access/execution is authorized. No cleanup, deletion, rename, overwrite, commit, or push is authorized.

Real Phase A2 remains blocked until this repair has independent Reviewer PASS, Test Manager GREEN, and fresh explicit operator authorization. Real Phase B2 requires separate explicit operator authorization after Phase A2 has exited and its evidence has been verified.

## Failure evidence and attempt-1 disposition

Observed Phase A attempt 1:

- failed closed in `0.04519541701301932` seconds before model loading, dataset loading, provider invocation, optimizer update, or adapter creation;
- error exactly `(pid=0)`, identified as `psutil.AccessDenied(pid=0)` raised while `_resource_gate()` read process memory;
- zero provider calls and zero optimizer updates;
- no adapter output directory or adapter files created;
- lock acquired once, released once, and watchdog cancelled;
- failure reports, log, Phase A fail marker, and final fail marker durably written;
- no retry occurred.

Attempt-1 evidence is immutable historical evidence. Never delete, rename, truncate, replace, rewrite, or otherwise modify:

```text
agent-output/cmux-14-5/phase-a-log.txt
agent-output/cmux-14-5/phase-a-report.json
agent-output/cmux-14-5/pilot-report.json
/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-phase-a-fail
/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-fail
```

All other canonical attempt-1 report, log, output, and marker names remain reserved and must not be reused:

```text
agent-output/cmux-14-5/phase-b-log.txt
agent-output/cmux-14-5/phase-b-report.json
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-a
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-phase-b
/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-phase-a-ok
/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-phase-b-ok
/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-phase-b-fail
/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-ok
```

Attempt-2 preflight must not treat any attempt-1 artifact as an attempt-2 collision. Attempt-1 evidence remains readable only as historical evidence and must not satisfy any attempt-2 dependency or success gate.

## Trackable user story

As a DS4 fine-tuning operator (WHO), I want the process resource observer repaired and the bounded pilot repinned to a new immutable attempt-2 namespace (WHAT), so that inaccessible or disappearing system processes do not cause a false preflight failure while prior failure evidence and one-attempt safety remain intact (WHY).

## R14.5a-1 — Per-process resource observer repair

Repair only the competing-process scan in `scripts/ds4_segmented_pilot.py`.

For each process returned by `psutil.process_iter(...)`:

1. Exclude the current process by PID before reading its RSS.
2. Read that process's RSS independently.
3. Skip only these process-local race/permission exceptions:
   - `psutil.AccessDenied`;
   - `psutil.NoSuchProcess`;
   - `psutil.ZombieProcess`.
4. Continue scanning all remaining processes after an allowed skip.
5. Preserve observable skip evidence with exact total count, count by exception type, and skipped PID values by exception type. Evidence must be deterministic and JSON-serializable; PID lists must be sorted.
6. Any other exception from process enumeration, PID access, RSS observation, virtual-memory observation, or resource evidence construction remains terminal and fail closed. The failure report must identify the exception type and available PID context.

The repair must not convert the scan into best-effort success, catch all `psutil.Error`, catch all `OSError` per process, or silently discard unknown observer failures.

## R14.5a-2 — Resource gates remain exact

Preserve these existing gates and semantics:

- current process excluded from competitor evaluation;
- competing process fails only when observed RSS is strictly greater than `50 GiB`;
- available memory must be at least `32 GiB`;
- free disk at the output/workspace parent must be at least `1 GiB`;
- resource observer unavailable or unknown observation error fails closed;
- successful resource evidence retains available memory, disk free, observed competing-process RSS values, and allowed-skip evidence;
- a genuine process above `50 GiB` remains terminal even when other processes were skipped for allowed exceptions.

No threshold reduction, fallback observer, retry loop, process termination, or resource cleanup is permitted.

## R14.5a-3 — Exact immutable attempt-2 namespace

Attempt 2 uses these exact paths. No canonical attempt-1 path may be reused.

### Phase A2

```text
phase: phase-a
output: /Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a
log: agent-output/cmux-14-5-attempt-2/phase-a-log.txt
report: agent-output/cmux-14-5-attempt-2/phase-a-report.json
OK marker: /Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-a-ok
fail marker: /Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-a-fail
```

Required Phase A2 checkpoints remain:

```text
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/phase-a-start.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/0000001_adapters.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/0000002_adapters.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/adapters.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/adapter_config.json
```

### Phase B2

```text
phase: phase-b
output: /Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-b
resume source: /Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-a/0000002_adapters.safetensors
log: agent-output/cmux-14-5-attempt-2/phase-b-log.txt
report: agent-output/cmux-14-5-attempt-2/phase-b-report.json
OK marker: /Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-b-ok
fail marker: /Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-phase-b-fail
```

Required Phase B2 checkpoints remain:

```text
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-b/resume-start.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-b/0000001_adapters.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-b/adapters.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-2-phase-b/adapter_config.json
```

### Final attempt-2 evidence

```text
final report: agent-output/cmux-14-5-attempt-2/pilot-report.json
final OK marker: /Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-ok
final fail marker: /Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-2-fail
```

Reports, markers, commands, marker bindings, contract digests, catalog commands, and Phase B dependency validation must bind these exact attempt-2 paths. Phase B2 must never resume from an attempt-1 checkpoint or accept an attempt-1 report/marker.

## R14.5a-4 — One-attempt collision semantics

Before Phase A2 opens its log or creates any output/evidence, every Phase A2, Phase B2, and final attempt-2 output/report/log/marker path must be absent. Any existing attempt-2 path is a terminal collision: preserve it, write nothing over it, perform no cleanup, and do not launch.

The command wrapper must check the Phase A2 log path before `tee` opens it. Pilot preflight must independently check all paths it owns before and after lock acquisition.

Before Phase B2 opens its log or creates output/evidence:

- exact Phase A2 output, report, OK marker, and resume source are required dependencies and must pass existing report/marker/hash/identity binding checks;
- Phase A2 fail marker must be absent;
- every Phase B2 and final attempt-2 output/report/log/marker path must be absent;
- any pre-existing Phase B2 or final attempt-2 artifact blocks Phase B2 and cannot be overwritten;
- attempt-1 artifacts neither block Phase B2 nor satisfy any Phase B2 prerequisite.

One real attempt per repinned phase. No automatic retry, manual in-place retry, fallback, alternate namespace allocation, suffix increment, cleanup, or overwrite.

## R14.5a-5 — Training identity and budgets unchanged

Attempt 2 changes only resource-observer behavior and artifact namespace.

Preserve exactly:

- model `/Volumes/Data NVME/mlx-ft/ds4/model-4bit`;
- dataset `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke`;
- config `/Volumes/Data NVME/mlx-ft/ds4/lora-config.json`;
- provider SHA-256 `20572191316e36ce228d5a7b5f1cdd396e4d796c4b620696d10f3c33937f3518`;
- vendor gitlink/inner HEAD `80fab4e419a57f9465bb9e2f4e90010d645e124c`;
- max sequence `4096`, batch `1`, learning rate `1e-5`, mask prompt enabled, gradient checkpointing enabled, segment size `1`, gradient accumulation `1`, seed `0`, LoRA, Adam, 16 layers, 25 validation batches, world size `1`;
- save/report cadence `1`;
- Phase A2: exactly 2 provider calls, 2 optimizer updates, eval cadence `2`, timeout `2700s`;
- Phase B2: exactly 1 provider call, 1 optimizer update, eval cadence `1`, timeout `1500s`;
- total: exactly 3 calls/updates and `4200s` active hard budget;
- one attempt each, retry `none`, fallback `none`.

No budget increase, reduced sequence, changed segment size, alternate dataset/model/backend, smoke rerun, monolithic-loss fallback, or distributed execution.

## R14.5a-6 — Mutation-sensitive synthetic tests

Tracked synthetic tests must turn RED for each prohibited mutation and prove without real asset access:

1. PID `0` raising `psutil.AccessDenied` is skipped, counted under `AccessDenied`, recorded with PID `0`, and later processes are still scanned.
2. A process raising `psutil.NoSuchProcess` is skipped and counted without hiding a later competitor.
3. A process raising `psutil.ZombieProcess` is skipped and counted without hiding a later competitor.
4. A genuine non-current process strictly above `50 GiB` fails the gate.
5. A process at exactly `50 GiB` does not fail that strict-greater-than gate.
6. Current PID is excluded before RSS access and cannot become a competitor, including a fake current process above `50 GiB`.
7. Any unknown exception from iteration or per-process observation fails closed and is not added to allowed-skip evidence.
8. Available memory below `32 GiB` fails; exactly `32 GiB` passes that threshold.
9. Free disk below `1 GiB` fails; exactly `1 GiB` passes that threshold.
10. Skip totals, per-type counts, and sorted PID lists are exact and mutation-sensitive.
11. Attempt-1 paths and byte content remain unchanged and do not collide with attempt 2.
12. Every exact attempt-2 Phase A2/B2/final path is pinned; mutation to any path turns RED.
13. Any existing Phase A2 evidence path blocks Phase A2 before log/output mutation.
14. Any existing Phase B2/final evidence path blocks Phase B2; required valid Phase A2 evidence remains the only exception and dependency.
15. Phase B2 rejects attempt-1 resume/report/marker substitution and accepts only the exact Phase A2 step-2 checkpoint binding.
16. Retry/fallback remains `none`; no generated attempt-3 or dynamic suffix path exists.
17. Story 14.3 smoke source/catalog/report/log/markers, default training catalog strings, provider, vendor, Path A, CUDA, distributed, Metal, CPU, SSD streaming, and GGUF paths remain protected.
18. Test import and execution perform no `/Volumes/Data NVME/...` access.

Every verdict-contributing test file must be tracked before Reviewer or Test Manager adjudication.

## R14.5a-7 — Review and authorization gates

Required sequence:

1. Architect GO on this repin and exact namespace.
2. Coder follows TDD: resource/namespace tests RED, minimal repair GREEN, then refactor while GREEN.
3. Independent Reviewer PASS on the exact implementation revision.
4. Independent Test Manager GREEN on tracked tests and protected invariants.
5. Fresh explicit operator authorization before real Phase A2.
6. Phase A2 runs once in a visible pane and exits completely.
7. Supervisor verifies Phase A2 report, OK marker, checkpoint hashes/digests, call/update cardinality, lock release, and absence of fail evidence.
8. Separate explicit operator authorization before real Phase B2.
9. Phase B2 runs once in the visible execution protocol.

Prior Story 14.5 Reviewer PASS/Tester GREEN validates attempt-1 implementation history but does not authorize repaired attempt 2. Prior operator authorization does not carry forward.

## R14.5a-8 — Protected invariants and non-claims

Preserve:

- Path A permanent STOP and 365-file protected digest evidence;
- frozen Story 14.3 segmented smoke source, command, report, log, provenance, markers, and adapter evidence;
- Story 14.5 attempt-1 evidence byte-for-byte;
- default `smoke-train`, `full-train`, `continue-train`, and `ds4-segmented-smoke` behavior;
- vendor/provider semantics and identities;
- CUDA, distributed, Metal default inference, CPU reference, SSD streaming, GGUF, and C/Objective-C/Metal production paths.

Allowed future success claims remain limited to three bounded updates and exact adapter-weight continuity. No claim of convergence, quality, generalization, throughput improvement, OOM repair, full-training readiness, optimizer/RNG/dataset-cursor/scheduler/global-step continuity, exact interruption recovery, CUDA/distributed readiness, or behavior beyond the bounded pilot.

Resource-observer repair proves only that explicitly allowed inaccessible/disappearing process observations are skipped with auditable evidence while all other resource failures remain fail closed.

## STOP/ESCALATE

Stop and require BA + Architect repin before execution if:

- any attempt-1 evidence would need mutation or cleanup;
- exact attempt-2 paths conflict or require another namespace;
- a broader exception class must be skipped;
- the `>50 GiB`, `32 GiB`, or `1 GiB` gates must change;
- training identity, phase cardinality, or budgets must change;
- any automatic retry/fallback is proposed;
- Phase B2 cannot bind exclusively to Phase A2 evidence;
- protected Story 14.3, Path A, vendor/provider, CUDA, distributed, Metal, CPU, SSD, or GGUF code/evidence would change;
- a required test needs real asset access;
- any verdict-contributing test file is untracked.

## Acceptance criteria

1. `docs/backlog.md` contains Story 14.5a using exact WHO/WHAT/WHY form and points to this canonical requirements file.
2. Attempt-1 failure evidence and reserved namespace are explicitly immutable; no attempt-1 artifact is deleted, renamed, overwritten, or accepted as attempt-2 evidence.
3. Process scanning skips only `AccessDenied`, `NoSuchProcess`, and `ZombieProcess` per process, with deterministic PID/type counts; unknown errors fail closed.
4. Current-process exclusion and strict `>50 GiB` competitor gate remain; `32 GiB` memory and `1 GiB` disk gates remain.
5. Mutation-sensitive tests cover PID 0 access denial, disappearing/zombie processes, a genuine competitor, unknown exceptions, current process, and memory/disk boundaries.
6. Exact Phase A2, Phase B2, final report/log/output/marker paths and exact Phase B2 resume source are pinned in a new immutable namespace.
7. Attempt-1 artifacts do not block attempt 2; any pre-existing artifact in the relevant attempt-2 phase/final namespace blocks without overwrite or cleanup.
8. Same model/dataset/config/provider/vendor/training pins remain; A2 is 2 updates/2700s, B2 is 1 update/1500s, total budget 4200s.
9. One attempt per phase; no automatic retry, fallback, attempt-3 allocation, smoke rerun, alternate backend, or parameter reduction.
10. New Reviewer PASS and Test Manager GREEN plus fresh explicit operator authorization are required before Phase A2; separate authorization is required before Phase B2.
11. Path A, Story 14.3 smoke, vendor/provider, CUDA, distributed, Metal, CPU, SSD streaming, GGUF, and non-claim boundaries remain intact.
12. This BA slice performs no real asset access, training, inference, cleanup, commit, or push.
