# Story 14.5e — persistent flock protocol + fixed attempt-4 repin requirements r2

## BA verdict

**GO for requirements, architecture, TDD implementation, and synthetic validation only.**

This BA slice does not authorize code implementation by BA, real model/data/provider access, training, A4/B4 invocation, cleanup, deletion, rename, overwrite, commit, or push. A4 remains blocked until implementation is committed, every verdict-contributing test is tracked, independent Reviewer PASS and Test Manager GREEN name the same exact revision, an external authorization review binds that revision and the exact production A4 command, and the operator explicitly authorizes one visible A4 invocation. B4 always requires separate authorization after independently verified A4 success.

## User story

As a DS4 fine-tuning operator (WHO), I want a persistent flock-owned singleton lock and one fixed attempt-4 namespace (WHAT), so that A4 and B4 can reuse one fail-closed lock inode without weakening historical evidence, collision safety, or one-invocation authorization (WHY).

## Fixed A3 failure evidence

1. Story 14.5d synthetic repair was reviewed and committed at revision `9b906515f5aba3928542cdf099e5248a45030af9`.
2. External review authorized exactly one A3 Phase A invocation for that revision. The authorization was consumed by the invocation, regardless of its exit status.
3. The invocation created only the A3 log, acquired the shared lock, then failed on the repeated post-acquisition gate because `verify_attempt2_historical_evidence()` still required `/Volumes/Data NVME/mlx-ft/ds4/.ds4-ft.lock` to be absent. It rejected the invocation's own valid lock with:

   ```text
   attempt-2 historical absence fact violated: /Volumes/Data NVME/mlx-ft/ds4/.ds4-ft.lock
   ```

4. Exit was `1`. Training calls, provider calls, and optimizer updates were all `0`. No adapter output, phase report, success/failure marker, final report/marker, or B3 artifact was created. No retry occurred. The owned lock was released and was absent after exit. B3 remained blocked.
5. Failure evidence was committed at `aec72af6d4a247271a00fdf22b228ebb57667432` and is immutable:

   | Evidence | Size | SHA-256 |
   |---|---:|---|
   | `agent-output/cmux-14-5-attempt-3/phase-a-authorization.json` | `532` | `543e584621c133e6d8afc1a7b6d0a3bea296a78d2de18f77fd84fcb3bf23b065` |
   | `agent-output/cmux-14-5-attempt-3/phase-a-log.txt` | `619` | `77b2dc85dc02735dc1727ffc06f2fb9471230c90f8808d67da7e10bb852ca1bc` |
   | `agent-output/cmux-14-5-attempt-3/phase-a3-runtime-preflight-failure.json` | `1075` | `0f19ce8a8147a39dffcb6b3162f67779fd6d26263ed2015285285350c8dde3da` |

6. The authorization remains exact six-key canonical JSON for revision `9b906515f5aba3928542cdf099e5248a45030af9`, including canonical command SHA-256 `fa4ac09716b820d249030bd93c9fe3dc0817be49059acc437af9ef22a562ad1e`, pilot source SHA-256 `d01a42db49abc0e847251f28683d6521310c50d7ff76960fd28fd93426393127`, catalog source SHA-256 `1258eac5c8c2b412049b18db8e6b28ec4fdd45341c5a99a6041bde10b4a61a97`, protected-files manifest SHA-256 `c3b3dbcd4bd0b4b38c5c6c890853e939fb6cc3ef27d981dd05b29ea06ca77ea4`, and attempt-2 runtime-manifest SHA-256 `2cc1d2359017cce2130a9428fd202e8c2f35b5d5810030c95aaac8b464fd3d88`.

## R14.5e-1 — explicit persistent lock stages

The production admission path must use an explicit fail-closed lock stage at the one canonical path `/Volumes/Data NVME/mlx-ft/ds4/.ds4-ft.lock`. The only valid stages are:

1. **Pre-lock:** before `_acquire_ft_lock()` succeeds, accept only (a) first-use pathname absence or (b) an existing exact unlocked idle-v1 regular single-link file under stable non-following descriptor/path identity. Existing idle admission must acquire `fcntl.flock(fd, LOCK_EX | LOCK_NB)` before reading, require byte-exact `b"ds4-ft-lock-v1\nstate=idle\n"`, then unlock and close without writing. A busy flock, stale owned bytes, legacy bytes, malformed bytes, symlink, non-regular object, hard link, unreadable path, identity drift, or inspection error fails closed without mutation.
2. **Post-lock:** after `_acquire_ft_lock()` succeeds, every repeated admission/history/identity gate must replace only the current shared-lock admission fact with R14.5e-2 exact live ownership attestation. It must not skip the historical verifier or any other absence, collision, identity, revision, command, authorization, source-hash, protected-file, phase-dependency, or publication check.
3. Unknown, missing, contradictory, or out-of-order stage state fails closed. Post-lock verification cannot be selected before successful acquisition. A permissive `allow_lock`, global bypass, path-exists shortcut, stale takeover, or exception swallowing is forbidden.
4. Wrapper admission and every in-process gate before acquisition use pre-lock semantics. Every repeated gate after acquisition, including trusted-runtime-identity revalidation and any later pre-publication revalidation, uses post-lock semantics until release.
5. The immutable A3 failure manifest field `lock_absent_after_exit: true` remains an exact recorded historical fact and hash input only. Runtime verification must not reinterpret it as perpetual current pathname absence.

## R14.5e-2 — exact flock-owned state and post-lock predicate

1. Acquisition opens or first-use creates the canonical path read/write with non-following and close-on-exec semantics, requires a regular single-link inode, acquires `LOCK_EX | LOCK_NB`, and keeps that same private owning FD and flock live until release or rollback finishes. Existing idle files are reused without inode replacement; a first-use zero-length inode is accepted only inside the creating acquisition call.
2. Owned bytes are exactly, including final newline and key order:

   ```text
   ds4-ft-lock-v1
   state=owned
   pid=<decimal current PID>
   token=<64 lowercase hexadecimal characters>
   ```

   The token is invocation-unique and independent of PID. Legacy `pid=<token>\n`, empty, partial, reordered, extra, malformed-version, malformed-PID, malformed-token, or whitespace variants are rejected.
3. Post-lock attestation requires complete internally consistent ownership state, including the live read/write close-on-exec FD, canonical path, acquisition `(st_dev, st_ino)`, current PID, exact token, `_LOCK_OWNED_PARTIAL is False`, and flock lifecycle state true.
4. Repeated `fstat` of the owning FD and non-following `lstat` of the pathname must prove one stable regular single-link inode equal to the acquisition identity before and after offset-stable reads through the owning FD. Exact owned bytes, size, PID, token, and final newline must match.
5. Successful exclusive flock acquisition plus the still-live private FD and no intervening production unlock/close is the ownership-exclusion proof. A separate process must verify nonblocking flock exclusion in tests; a second same-process descriptor is not authoritative.
6. Missing/closed/wrong FD, absent or contradictory globals, wrong PID/token/path/inode, false flock lifecycle, partial state, modified bytes, pathname replacement, symlink, non-regular object, link-count drift, read/stat error, metadata race, or any filesystem error fails closed. Attestation never closes the owning FD and never mutates the pathname.

## R14.5e-3 — persistent release, rollback, and crash semantics

1. No acquisition, rollback, collision, failure, release, phase completion, or final publication path may unlink, rename, replace, quarantine, or otherwise remove the singleton pathname.
2. Normal release first passes R14.5e-2 attestation while flock remains held, then writes/truncates byte-exact idle v1 through the same owning FD, completes short-write handling, `fsync`, and proves exact idle bytes plus stable same-inode descriptor/path identity while flock remains held. That final same-inode idle attestation is the release linearization point.
3. Release then unlocks and closes in `finally` and clears every live-ownership global. It must not require post-unlock pathname absence or idle bytes because a successor may acquire immediately and write a new owned payload.
4. Acquisition failure may restore exact idle bytes only through the still-flocked owning FD and only when same-inode pathname identity, complete write, `fsync`, exact idle bytes, and stable identity are all proved. Otherwise it performs no pathname mutation. Every rollback/release path unlocks/closes in `finally`, clears live-ownership globals, and raises terminal `PilotError` on failure.
5. Tamper, replacement, write/short-write/fsync/unlock/close failure, or lost identity never permits deletion, rename, replacement-path writes, stale takeover, or automatic cleanup. Stale owned or partial bytes remain fail-closed evidence.
6. A process crash releases flock through kernel FD teardown but may leave the last fsynced owned payload. Future PRE_LOCK rejects those stale owned bytes even when no flock remains. No PID-liveness, age timeout, retry, byte rewrite, deletion, rename, quarantine, or automatic recovery is allowed.
7. Existing 60-second lock acquisition timeout/polling, watchdog cancellation, failure reporting, and one-attempt semantics remain unchanged except for this persistent-file protocol.

## R14.5e-4 — immutable attempt-3 historical verifier

Add an attempt-3 historical verifier with a trust root inside the production verifier. It must:

1. Pin and directly verify the three files, sizes, and SHA-256 values listed under Fixed A3 failure evidence. Each must be an exact regular file, not a symlink or non-regular object.
2. Strict-parse the authorization and failure manifest with duplicate-key rejection and exact schema/type/value comparison. Reject missing/extra keys, alternate paths, coordinated path/hash/payload substitution, regeneration, and mutable module-level alias replacement.
3. Verify the failure manifest's exact runtime facts: Story `14.5d`, revision `9b906515f5aba3928542cdf099e5248a45030af9`, phase `phase-a`, attempt `3`, namespace `ds4-segmented-pilot-attempt-3`, status `fail`, exit `1`, exact lock-absence error, `0` training calls, `0` provider calls, `0` optimizer updates, no retry, absent adapter/report/failure markers, and recorded `lock_absent_after_exit: true`. That absence remains historical evidence only, not a current-filesystem predicate.
4. Verify the authorization record referenced by the failure manifest resolves to the pinned authorization file and exact size/hash; verify the log record resolves to the pinned log and exact size/hash.
5. Require these current attempt-3 destinations to remain absent before A4 acquisition:
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
6. At PRE_LOCK, require the current shared lock to satisfy R14.5e-1 first-use-absence or exact unlocked idle-v1 admission. At POST_LOCK, preserve every immutable A3 file and destination-absence check while substituting only R14.5e-2 live ownership attestation for current lock admission. Never apply the recorded A3 absence as perpetual current pathname absence.
7. Preserve attempt-1 and attempt-2 historical verifiers byte-for-byte in meaning and require attempt-1, attempt-2, and attempt-3 history to pass independently. Attempt-3 evidence may satisfy only historical-failure lineage, never A4/B4 success, resume, or publication gates.

## R14.5e-5 — fixed immutable attempt-4 namespace

Attempt 4 is the only new live namespace. No attempt 3 reuse, attempt 5, dynamic allocator, timestamp, random suffix, fallback namespace, or alternate path is allowed.

Canonical namespace:

```text
ds4-segmented-pilot-attempt-4
```

### Phase A4

```text
phase: phase-a
catalog command: ds4-segmented-pilot-attempt-4-phase-a
authorization: agent-output/cmux-14-5-attempt-4/phase-a-authorization.json
output: /Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-a
log: agent-output/cmux-14-5-attempt-4/phase-a-log.txt
report: agent-output/cmux-14-5-attempt-4/phase-a-report.json
OK marker: /Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-4-phase-a-ok
fail marker: /Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-4-phase-a-fail
```

Required A4 artifacts:

```text
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-a/phase-a-start.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-a/0000001_adapters.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-a/0000002_adapters.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-a/adapters.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-a/adapter_config.json
```

### Phase B4

```text
phase: phase-b
catalog command: ds4-segmented-pilot-attempt-4-phase-b
authorization: agent-output/cmux-14-5-attempt-4/phase-b-authorization.json
output: /Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-b
resume source: /Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-a/0000002_adapters.safetensors
log: agent-output/cmux-14-5-attempt-4/phase-b-log.txt
report: agent-output/cmux-14-5-attempt-4/phase-b-report.json
OK marker: /Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-4-phase-b-ok
fail marker: /Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-4-phase-b-fail
```

Required B4 artifacts:

```text
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-b/resume-start.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-b/0000001_adapters.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-b/adapters.safetensors
/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-b/adapter_config.json
```

### Final attempt-4 evidence

```text
final report: agent-output/cmux-14-5-attempt-4/pilot-report.json
final OK marker: /Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-4-ok
final fail marker: /Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-4-fail
```

Every report, marker, marker hash, contract digest, phase dependency, authorization, command, and resume binding must use these exact paths and namespace.

## R14.5e-6 — catalog retirement and exact live surface

1. Retire consumed A3 commands from every runnable catalog, choice, help, emit, and run surface:

   ```text
   ds4-segmented-pilot-attempt-3-phase-a
   ds4-segmented-pilot-attempt-3-phase-b
   ```

   Historical A3 parsers, evidence constants, and verification remain available only for read-only lineage.
2. Add only these explicit non-default runnable commands:

   ```text
   ds4-segmented-pilot-attempt-4-phase-a
   ds4-segmented-pilot-attempt-4-phase-b
   ```

3. Exclude A4/B4 from `DEFAULT_BACKEND_STEPS["local-mlx"]`. Preserve unrelated backend/default command bytes and behavior.
4. Exact emission surface:

   ```bash
   python3 scripts/finetune_ds4.py emit-commands --backend local-mlx --mlx-lm-source fork ds4-segmented-pilot-attempt-4-phase-a
   python3 scripts/finetune_ds4.py emit-commands --backend local-mlx --mlx-lm-source fork ds4-segmented-pilot-attempt-4-phase-b
   ```

5. Rendered wrappers must use exact `--attempt 4`, phase, authorization, log, output, and B4 resume bindings. Preserve collision/history/revision admission before writes, noclobber log FD, exact pilot invocation, visible `tee /dev/fd/3`, and `PIPESTATUS[0]` ordering.
6. No attempt registry, generic `attempt-N` allocator, dynamic suffix, attempt 5, cleanup, deletion, rename, overwrite, append/truncating log open, retry loop, fallback, or reuse of attempt-3 destinations may be introduced.

## R14.5e-7 — authorization, collision, and execution boundary

1. Before A4 writes anything, all attempt-1, attempt-2, and attempt-3 historical verifiers must pass; all A4, B4, and final attempt-4 destinations must be absent; the shared lock must pass R14.5e-1 PRE_LOCK admission; repository HEAD, protected-file manifest, pilot/catalog sources, provider/vendor identity, effective model/data/config/training identity, and exact emitted A4 command must equal a fresh externally reviewed A4 authorization record.
2. The external authorization record must be canonical, exact-revision, phase-specific, and command-specific. Any A4 invocation exit consumes it. No in-place retry, second invocation, fallback, or namespace reuse is authorized.
3. A4 performs exactly `2` provider calls and `2` completed optimizer updates under a `2700s` hard phase budget. Existing save cadence, artifact validation, report/marker publication, fail-atomic rollback, watchdog, lock, digest, and non-claim contracts remain unchanged.
4. B4 remains blocked until A4 success report, A4 OK marker, step-2 resume source, canonical digest, identity, cardinality, command, authorization, and all historical lineages are independently verified. B4 requires a separate fresh exact-revision authorization and operator invocation approval.
5. B4 performs exactly `1` provider call and `1` resumed optimizer update under a `1500s` hard phase budget. Total active A4+B4 budget remains `4200s`, total provider calls/updates remains `3`, and no larger-readiness or convergence claim follows.
6. Any collision or mismatch blocks before mutation. Never delete or alter historical evidence or a destination to make admission pass.

## R14.5e-8 — mutation-sensitive tracked tests

TDD must establish red before green and exercise production boundaries, not copied predicates or source-text-only assertions.

1. PRE_LOCK matrix: absent first use passes without mutation; exact idle-v1 regular single-link file passes only after a nonblocking exclusive flock proves it unlocked; active-flocked idle, stale owned, legacy, empty, partial, malformed, symlink, hard-link, non-regular, unreadable, and identity-raced states reject without mutation.
2. Production-order lock test: run PRE_LOCK, acquire through `_acquire_ft_lock()`, prove exact owned-v1 bytes and child-process flock exclusion on the same inode, repeat the full POST_LOCK gate, release through the owning FD to exact idle-v1, prove globals cleared and no unlink/rename/replace API called, then prove a separate process or B4-style successor reacquires that same inode.
3. Acquisition and POST_LOCK negative matrix: independently reject competitor flock, missing/closed/wrong FD, unset/wrong path, wrong inode/PID/token, false flock lifecycle, partial state, extra/missing/reordered/malformed bytes, whitespace/newline changes, symlink, directory, FIFO or other non-regular object, link-count drift, unreadable lock, descriptor/path substitution, and stat/read race. Every competitor pathname and payload remains untouched.
4. Prove POST_LOCK stage substitution affects only the current shared-lock fact: mutate every other attempt-1/2/3 historical file, path, size, hash, payload, schema, report identity, authorization binding, and absence destination independently and confirm rejection after lock acquisition.
5. Release/rollback tests must prove exact same-inode idle-v1 write and `fsync` through the flock-owning FD before unlock/close, including short-write handling and final stable attestation. Inject replacement/tamper/write/short-write/fsync/unlock/close failure at every seam; no case may unlink, rename, replace, quarantine, write a replacement inode, or retain false live-ownership globals.
6. Crash test: a subprocess acquires, persists exact owned bytes, and calls `os._exit`; kernel flock release occurs, stale owned bytes remain, and the next PRE_LOCK and B4 admission reject without automatic recovery.
7. Attempt-3 historical tests must verify the real committed evidence read-only or an exact immutable synthetic snapshot, including all three pinned files, strict authorization/failure schemas, direct hashes, manifest target bindings, recorded `lock_absent_after_exit: true` as history only, all listed non-lock absences, and coordinated-substitution negatives.
8. Attempt-4 tests must cover exact A4/B4/final paths, command rendering, catalog retirement of A3, parser rejection of A3 as a live attempt, A4 non-default status, no attempt5/dynamic allocation, pre-log collision/no-write ordering, independent A4/B4 authorizations, B4 dependency validation, successful A4-idle-to-B4-reacquire lifecycle, A4 crash blocking B4, and final publication lineage.
9. Preserve and run relevant Story 14.5c/14.5d focused tests and the canonical six-file suite. Verify `git diff --check` and direct production/vendor protected-file integrity.
10. Every verdict-contributing test file must be tracked. Reviewer and Test Manager must run `git ls-files -- <each cited test file>` before PASS/GREEN and block on any untracked baseline file.

## Acceptance criteria

1. PRE_LOCK accepts only first-use absence or an existing exact idle-v1 unlocked regular single-link file proved under stable non-following FD/path identity; all other states fail closed without mutation.
2. POST_LOCK accepts only exact owned-v1 bytes bound to current PID and independent 64-hex token on the same regular single-link inode through the still-open read/write close-on-exec flock-owning FD and complete in-process ownership state.
3. No code path unlinks, renames, replaces, quarantines, or removes the singleton. Normal release writes and `fsync`s exact idle-v1 through the owning FD, proves stable same-inode idle state while flocked, then unlocks/closes and clears globals. Crash or failed transition may leave stale owned/partial bytes, which block future PRE_LOCK/B4 without automatic recovery.
4. Every non-lock attempt-1/2/3 history, absence, collision, revision, command, authorization, source, identity, phase-dependency, and publication gate remains unchanged and enforced at every applicable stage.
5. The three committed A3 evidence files remain byte-for-byte pinned; recorded `lock_absent_after_exit: true` remains immutable historical fact only; all A3/B3/final non-lock absence facts remain immutable; A3 evidence can never satisfy A4/B4 success.
6. Live runtime is repinned only to fixed `ds4-segmented-pilot-attempt-4`; A3 runnable commands are retired; A4/B4 exact non-default commands and paths are present; no attempt5, dynamic namespace, cleanup, stale takeover, retry, or fallback exists.
7. A4 remains exactly `2` calls/updates within `2700s`, B4 exactly `1` resumed call/update within `1500s`, total `3/4200s`, with unchanged runtime identity and non-claims.
8. Independent Reviewer PASS and Test Manager GREEN on one exact committed revision are required before fresh external A4 authorization review. This slice authorizes no A4/B4 invocation.

## STOP/ESCALATE

STOP for any A3 evidence mutation; attempt-1/2/3 trust-root weakening; generic lock bypass; following symlinks; any lock-path unlink/rename/replace/quarantine/removal; automatic stale-byte repair, PID-liveness or age-based takeover; acceptance without a live owning FD/flock lifecycle; closing the owner FD before release; treating historical A3 absence as perpetual current absence; skipping non-lock gates post-acquisition; A3 live reuse; attempt5/dynamic namespace; collision cleanup; retry/fallback; runtime identity, budget, or cardinality drift; real asset/provider/training/inference access; untracked verdict test; commit; or push.
