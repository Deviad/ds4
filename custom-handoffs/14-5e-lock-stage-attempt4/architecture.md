# Story 14.5e — stage-specific lock verification and fixed attempt-4 architecture

## Verdict

**GO for TDD implementation, canonical documentation updates, and synthetic validation only.**

No A4/B4 invocation, real model/data/provider access, training, inference, cleanup, evidence mutation, commit, or push is authorized. A4 remains blocked until one committed revision receives independent Reviewer PASS and Test Manager GREEN, followed by fresh external authorization review and explicit operator approval. B4 always requires a separate authorization after independently verified A4 success.

## 1. Design boundaries

Use two explicit, fixed lifecycle stages and one fixed live attempt namespace. Do not introduce a permissive lock flag, generic attempt allocator, compatibility fallback, cleanup path, or retry path.

Production files in scope:

- `scripts/ds4_segmented_pilot.py`
- `scripts/finetune_ds4.py`

Tracked test files in scope:

- `tests/test_ds4_segmented_pilot.py`
- `tests/test_finetune_ds4.py`

Canonical documentation in scope:

- `docs/architecture.md`
- `docs/technical-spec.md`

`docs/backlog.md` is BA-owned and already contains Story 14.5e. Preserve that change; Coder must not rewrite the requirements.

No ADR is required. This is a bounded correction and fixed namespace repin within the existing Story 14.5 operator-only pilot architecture. The durable lock lifecycle, A3 historical lineage, and A4 live surface must nevertheless be recorded in `docs/architecture.md` and `docs/technical-spec.md` in the implementation slice.

## 2. Explicit lock-stage API

Define a private closed stage type in `scripts/ds4_segmented_pilot.py` with exactly two values:

```python
class _LockStage(enum.Enum):
    PRE_LOCK = "pre-lock"
    POST_LOCK = "post-lock"
```

Do not accept strings by coercion. Do not provide a default stage. Every caller must pass one enum member explicitly. Identity checks against the enum members make `None`, arbitrary strings, fabricated enum values, and contradictory state fail closed.

Use one narrow dispatcher:

```python
def _verify_ft_lock(stage: _LockStage, workspace: pathlib.Path | str | None = None) -> None:
    ...
```

Semantics:

- `_LockStage.PRE_LOCK`: use non-following inspection of the canonical shared pathname and require it not to exist. A regular file, dangling or live symlink, directory, FIFO, socket, device, unreadable object, or filesystem error rejects admission.
- `_LockStage.POST_LOCK`: call the exact ownership attestation in section 3. Absence is a failure.
- Any other object or value raises `PilotError`.

The helper changes only the current shared-lock predicate. It does not suppress, catch, reorder, or reinterpret any historical, collision, revision, command, authorization, source, identity, phase-dependency, report, marker, or publication gate.

Thread the required `lock_stage` argument through these production boundaries:

- `verify_attempt2_historical_evidence(...)` for the existing shared-lock entry in its immutable absence set;
- new `verify_attempt3_historical_evidence(...)` for R14.5e-4 current-lock validation;
- fixed A4 launch admission;
- `_check_attempt_gates(...)`;
- every direct wrapper/admission call.

Attempt-1, attempt-2, and attempt-3 verifiers remain independent calls. Do not make one verifier's successful result stand in for another.

The returned historical evidence structures and their canonical digests remain stable. Stage selection is an observation rule for the one current shared pathname, not a rewrite of immutable historical facts. In particular, `lock_absent_after_exit: true` in A3 evidence remains an exact recorded post-exit fact at both runtime stages.

## 3. Exact owned-lock attestation

Implement one private ownership predicate and use it unchanged for post-lock admission and normal release. It must prove all R14.5e-2 conditions in one call.

Required state:

1. `_LOCK_OWNED_PATH` equals the canonical shared path returned by this invocation's successful `_acquire_ft_lock()`.
2. `_LOCK_OWNED_TOKEN` is non-empty, equals `str(os.getpid())`, and therefore remains bound to the current process identity.
3. `_LOCK_OWNED_PARTIAL is False`.
4. The pathname itself is a regular file under `os.lstat`; symlinks and all non-regular objects are rejected.
5. Bytes are exactly `b"pid=" + token.encode("utf-8") + b"\n"` with no extra or missing byte.

Make inspection race-aware and descriptor-based:

1. `os.lstat(path)` before open; require regular type.
2. Open read-only with `O_CLOEXEC | O_NOFOLLOW`. Absence of `O_NOFOLLOW` support fails closed; never fall back to following links.
3. `os.fstat(fd)`; require regular type and exact `(st_dev, st_ino)` agreement with the first `lstat` and the inode recorded during acquisition.
4. Read from the descriptor, not `Path.read_text()`. Read through expected length plus one byte and require byte-exact equality.
5. Repeat `fstat(fd)` and `lstat(path)` after the read. Require stable device, inode, type, size, and pathname-to-descriptor identity across all observations.
6. Close the descriptor in `finally`. Any open, stat, read, close, encoding, replacement, or metadata inconsistency raises `PilotError`.

A private immutable `(st_dev, st_ino)` acquisition identity may be added to support this proof. It is internal ownership evidence, not a third stage or public semantic variant. It must be set only for the inode created by `O_CREAT | O_EXCL` and cleared only after successful release/rollback.

## 4. Acquisition, rollback, and release

Preserve the existing 60-second acquisition timeout/polling, watchdog behavior, exclusive create, one-attempt collision semantics, and exact `pid=<token>\n` content.

Acquisition order:

1. Resolve the canonical lock path.
2. Create with `O_CREAT | O_EXCL` and non-following semantics.
3. Immediately record path, current PID token, `_LOCK_OWNED_PARTIAL = True`, and descriptor inode identity.
4. Write the complete byte payload, reject short writes, `fsync`, and close.
5. Run exact ownership attestation.
6. Set `_LOCK_OWNED_PARTIAL = False` only after attestation succeeds.

A pre-existing path is only a collision. Never delete, rename, truncate, replace, quarantine, or inspect it as recoverable stale state.

Acquisition rollback is separate from normal release because a partially written file cannot satisfy the complete-content predicate. Rollback may unlink only when `lstat(path)` proves a regular pathname whose `(st_dev, st_ino)` still equals the descriptor identity created by this invocation. If identity cannot be proven, leave the path and ownership state untouched and fail. Remove the current partial-lock quarantine/rename branch; quarantine is forbidden for this slice.

Normal `_release_ft_lock()` order:

1. Run the exact section 3 ownership predicate.
2. Re-attest pathname device/inode/type immediately before unlink.
3. Unlink once.
4. Verify the pathname is absent without following links.
5. Only then clear `_LOCK_OWNED_PATH`, `_LOCK_OWNED_TOKEN`, `_LOCK_OWNED_PARTIAL`, and the private inode identity.

On foreign, stale, tampered, replaced, symlinked, non-regular, partial, missing, unreadable, or raced state, release raises, performs no unlink or mutation, and retains ownership state. An unlink failure also retains state so a later explicit release attempt can prove ownership again. No release path may rename or quarantine anything.

## 5. Runtime ordering

Preserve the existing wrapper ordering and make stage arguments explicit at each call site.

### Wrapper before any write

The fixed A4 wrapper performs full admission with `_LockStage.PRE_LOCK` before log creation or output mutation. It verifies all three historical lineages, A4 collisions, exact revision/source/command/authorization bindings, and phase dependencies. Only after admission may the wrapper use:

```text
set -o noclobber
exec 3>"$LOG"
set +o noclobber
```

Preserve pinned interpreter selection, `bash -lc`, `set -o pipefail`, unbuffered execution, `--log-fd 3`, visible `tee /dev/fd/3`, and capture of `PIPESTATUS[0]` before any later command.

### In-process `run_phase`

Required order:

1. Parse exact attempt 4 and phase.
2. Run `_check_attempt_gates(..., lock_stage=_LockStage.PRE_LOCK)`.
3. Strict-parse authorization records and independently obtain attempt-1/2/3 historical evidence.
4. Attest active log FD, install watchdog, read config, validate unchanged pins, and construct contract digest.
5. Acquire the shared lock.
6. Immediately run the complete repeated `_check_attempt_gates(..., lock_stage=_LockStage.POST_LOCK)`.
7. Re-attest log FD.
8. Validate Phase B dependency when applicable.
9. Build trusted runtime identity/resource evidence.
10. Run the existing trusted-identity repeated gate with `_LockStage.POST_LOCK`.
11. Execute the unchanged bounded phase only after every gate passes.
12. Any additional revalidation inserted before release must also use `_LockStage.POST_LOCK`.
13. Release only through section 4, then publish failure or success evidence through existing fail-atomic report/marker ordering.

Selecting POST before acquisition necessarily fails because ownership globals/identity are absent or partial. Selecting PRE after acquisition fails because the pathname exists. There is no mutable `allow_lock` bit and no exception-swallowing compatibility path.

## 6. Immutable A3 historical verifier

Add `verify_attempt3_historical_evidence(...)` with its trust root constructed from function-local literals. Module-level convenience constants may exist for display, but replacing them must not alter accepted files, paths, hashes, schemas, or values.

Pin and directly verify these exact regular, non-symlink files:

| File | Size | SHA-256 |
|---|---:|---|
| `agent-output/cmux-14-5-attempt-3/phase-a-authorization.json` | `532` | `543e584621c133e6d8afc1a7b6d0a3bea296a78d2de18f77fd84fcb3bf23b065` |
| `agent-output/cmux-14-5-attempt-3/phase-a-log.txt` | `619` | `77b2dc85dc02735dc1727ffc06f2fb9471230c90f8808d67da7e10bb852ca1bc` |
| `agent-output/cmux-14-5-attempt-3/phase-a3-runtime-preflight-failure.json` | `1075` | `0f19ce8a8147a39dffcb6b3162f67779fd6d26263ed2015285285350c8dde3da` |

Read each file once through a non-following regular-file descriptor, verify stable metadata, exact size, and direct SHA-256. Strict-parse both JSON files with duplicate-key rejection and exact key sets, types, and values.

The authorization record must exactly bind:

- revision `9b906515f5aba3928542cdf099e5248a45030af9`;
- attempt-2 runtime manifest `2cc1d2359017cce2130a9428fd202e8c2f35b5d5810030c95aaac8b464fd3d88`;
- canonical A3 command `fa4ac09716b820d249030bd93c9fe3dc0817be49059acc437af9ef22a562ad1e`;
- catalog source `1258eac5c8c2b412049b18db8e6b28ec4fdd45341c5a99a6041bde10b4a61a97`;
- pilot source `d01a42db49abc0e847251f28683d6521310c50d7ff76960fd28fd93426393127`;
- protected-files manifest `c3b3dbcd4bd0b4b38c5c6c890853e939fb6cc3ef27d981dd05b29ea06ca77ea4`.

The failure manifest must exactly bind Story `14.5d`, the same revision, `phase-a`, attempt `3`, namespace `ds4-segmented-pilot-attempt-3`, status `fail`, exit code `1`, wrapper hash `788083f7d29d596a0acce61f61b9a6c385ccae1ce9d0f44ae34c630a5fa4d3e5`, and exact error:

```text
attempt-2 historical absence fact violated: /Volumes/Data NVME/mlx-ft/ds4/.ds4-ft.lock
```

It must require zero training calls, provider calls, and optimizer updates; `retry_performed: false`; adapter output, report, and failure markers absent; and `lock_absent_after_exit: true`. Its nested authorization and log descriptors must resolve to the two pinned files above with exact canonical paths, sizes, and hashes.

Require all 13 A3/B3/final destinations to remain absent:

- `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-a`
- `agent-output/cmux-14-5-attempt-3/phase-a-report.json`
- `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-3-phase-a-ok`
- `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-3-phase-a-fail`
- `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-3-phase-b`
- `agent-output/cmux-14-5-attempt-3/phase-b-authorization.json`
- `agent-output/cmux-14-5-attempt-3/phase-b-log.txt`
- `agent-output/cmux-14-5-attempt-3/phase-b-report.json`
- `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-3-phase-b-ok`
- `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-3-phase-b-fail`
- `agent-output/cmux-14-5-attempt-3/pilot-report.json`
- `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-3-ok`
- `/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-3-fail`

The shared `.ds4-ft.lock` is not part of that fixed 13-path list: it is validated separately by the required lock stage. Post-lock substitution therefore cannot weaken any A3 destination absence.

Return A3 evidence only as historical-failure lineage. It must never satisfy A4 authorization, A4/B4 phase dependency, resume, report, marker, or publication success.

## 7. Fixed attempt-4 live surface

Implement attempt 4 explicitly. Do not parameterize attempt number or derive names from an integer.

In `scripts/ds4_segmented_pilot.py`, add fixed A4 namespace/spec/authorization/command/report validators and retire A3 from the live parser. Keep only narrow A3 constants/parsers needed by section 6 read-only lineage. `--attempt 3` must be rejected as live execution; `--attempt 4` is the sole pilot attempt.

In `scripts/finetune_ds4.py`:

- remove A3 Phase A/B commands from `MLX_STEPS`, `COMMAND_STEPS`, emit choices, run choices, help, and catalog construction;
- remove live A3 authorization arguments;
- add only `ds4-segmented-pilot-attempt-4-phase-a` and `ds4-segmented-pilot-attempt-4-phase-b` plus exact A4 authorization inputs;
- keep both A4 commands excluded from `DEFAULT_BACKEND_STEPS["local-mlx"]`;
- add explicit `_central_attempt4_*` and `_pilot_attempt4_command` helpers rather than a generic attempt helper;
- leave unrelated command ordering and bytes unchanged.

Exact A4 paths:

- authorization `agent-output/cmux-14-5-attempt-4/phase-a-authorization.json`;
- output `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-a`;
- log/report `agent-output/cmux-14-5-attempt-4/phase-a-log.txt` and `phase-a-report.json`;
- phase markers `.ds4-segmented-pilot-attempt-4-phase-a-{ok,fail}`;
- required artifacts `phase-a-start.safetensors`, `0000001_adapters.safetensors`, `0000002_adapters.safetensors`, `adapters.safetensors`, `adapter_config.json`.

Exact B4 paths:

- authorization `agent-output/cmux-14-5-attempt-4/phase-b-authorization.json`;
- output `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-b`;
- resume source `/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-a/0000002_adapters.safetensors`;
- log/report `agent-output/cmux-14-5-attempt-4/phase-b-log.txt` and `phase-b-report.json`;
- phase markers `.ds4-segmented-pilot-attempt-4-phase-b-{ok,fail}`;
- required artifacts `resume-start.safetensors`, `0000001_adapters.safetensors`, `adapters.safetensors`, `adapter_config.json`.

Final evidence is exactly `agent-output/cmux-14-5-attempt-4/pilot-report.json` and `.ds4-segmented-pilot-attempt-4-{ok,fail}` under the fixed volume workspace.

Every A4/B4 command, authorization, contract digest, active-log binding, report, marker, marker hash, artifact, phase dependency, and resume identity must use these exact names. A4 performs exactly two provider calls and two updates under `2700s`; B4 performs exactly one resumed call and update under `1500s`; total active budget remains `3/4200s`. Preserve all model, dataset, config, interpreter, MLX/MLX-LM, optimizer, seed, LoRA, layer, batching, validation, checkpoint, runtime-identity, resource, and non-claim pins.

## 8. Authorization and collision boundary

A4 authorization is a fresh exact-revision record. Historical A3 authorization is consumed and cannot authorize A4. Before the A4 wrapper creates its log, require:

- independent attempt-1, attempt-2, and attempt-3 history verification;
- all A4, B4, and final output/log/report/authorization/marker destinations required absent for Phase A admission;
- exact current revision and protected-source hashes;
- exact A4 canonical command and fixed namespace;
- one A4 Phase A authorization record.

B4 requires a separate B4 authorization plus independently validated A4 report, OK marker, artifact cardinality, exact step-2 resume file, file SHA-256, canonical tensor digest, command, identity, and historical lineages. B4 and final destinations must be absent before mutation.

Any collision or mismatch is terminal. Never delete or alter A3 evidence, historical destinations, A4/B4 destinations, logs, or markers to make admission pass.

## 9. TDD sequence

All verdict-contributing tests must be tracked before Coder completion.

### RED 1 — lifecycle ordering

Add a production-order test using a synthetic workspace and real production functions:

1. full pre-lock gate passes with lock absent;
2. `_acquire_ft_lock()` creates exact bytes and ownership state;
3. full post-lock gate passes without skipping attempt-1/2/3 or collision checks;
4. trusted-identity repeated post-lock gate passes;
5. `_release_ft_lock()` removes the owned lock once and clears state.

Before implementation, the current repeated verifier must fail at the post-acquisition shared-lock absence check.

Add explicit out-of-order tests: POST before acquisition fails; PRE after acquisition fails; `None`, strings, and unknown stage objects fail.

### RED 2 — ownership and release matrices

Parameterize independent negatives for absent path; wrong/unset path; wrong/unset/empty/stale/foreign token; PID mismatch; partial state; missing/extra bytes; whitespace/newline/encoding changes; symlink; directory; FIFO; non-regular object; unreadable open/read; pre-open replacement; read-time replacement; post-read replacement; inode/type/size drift; and filesystem exceptions.

For every negative, post-lock verification fails. Release tests additionally prove foreign, tampered, symlinked, non-regular, replaced, and partial paths remain untouched and ownership state remains set. Test unlink failure retention and exactly-once successful unlink/state clearing. Test partial-acquisition rollback removes only the inode created by that acquisition; replacement survives untouched. Remove or replace tests that expect quarantine/rename.

### RED 3 — substitution isolation

After real acquisition, mutate each non-lock attempt-1/2/3 trust component independently: file path, type, size, hash, bytes, duplicate key, missing/extra schema key, report identity, authorization binding, command/source/revision binding, and every required absence destination. The full POST gate must reject every mutation. These tests prove only the current shared-lock predicate changes by stage.

### RED 4 — A3 historical evidence

Exercise the public verifier against the real committed files read-only or an exact immutable synthetic snapshot. Assert exact 532/619/1075 sizes and all three hashes, strict schemas/values, nested target bindings, exact failure facts, all 13 destination absences, and stage-specific current-lock handling.

Mutation tests must cover coordinated path/hash/payload substitution, alternate absolute paths, regenerated self-consistent JSON, module-level alias replacement, duplicate keys, symlink/non-regular evidence, and each exact fact. No test may rewrite the real three files.

### RED 5 — fixed A4 surface

Through production parser/catalog/emit boundaries, prove:

- exact A4/B4/final paths and artifact sets;
- emitted wrappers contain exact `--attempt 4`, phase, authorization, output, log, report, marker, and B4 resume bindings;
- A3 commands are absent from choices/help/emit/run and `--attempt 3` is rejected;
- A4 commands are explicit and non-default;
- attempt 5 and dynamic/alternate namespaces are rejected;
- wrapper preflight failure creates no log/output/marker;
- noclobber, log-FD attestation, visible tee, and `PIPESTATUS[0]` ordering remain effective;
- A4/B4 cardinality and `2700/1500/4200` budgets are unchanged.

Use behavioral execution against synthetic fixtures/stubs for ordering and no-mutation claims. Source-text assertions may supplement but never replace production-boundary tests.

### GREEN and regression

Implement the minimum production change after focused RED is captured. Then run focused tests, the canonical tracked Story 14.5 synthetic suite from `docs/technical-spec.md`, syntax compilation, and relevant full regressions. No test may access real model/data/provider/training resources.

Before verdict:

```bash
git ls-files --error-unmatch tests/test_ds4_segmented_pilot.py tests/test_finetune_ds4.py
git diff --check
```

Reviewer and Test Manager must independently repeat tracked-file checks for every test cited in their verdict.

## 10. Protected bytes and STOP conditions

Protected byte-for-byte:

- the three committed A3 evidence files and every attempt-1/attempt-2 evidence file;
- all historical external reports, logs, outputs, checkpoints, authorizations, and markers;
- `ds4_segmented_smoke.py`, smoke evidence, provider implementation, and frozen source manifests;
- model/data/config/vendor/environment/runtime pins;
- production inference, Metal, CUDA, SSD streaming, distributed inference, CPU reference, server, and training implementation outside the two scoped scripts;
- unrelated `finetune_ds4.py` command behavior and default ordering.

STOP on any A3 evidence-byte mutation; trust-root weakening; generic `allow_lock`; symlink following; unlink/rename/quarantine of a lock not proven owned; skipped non-lock post-acquisition gate; A3 live reuse; attempt5/dynamic namespace; collision cleanup; overwrite/truncation; retry/fallback; identity/budget/cardinality drift; untracked verdict test; real asset/provider/training/inference access; commit; or push.
