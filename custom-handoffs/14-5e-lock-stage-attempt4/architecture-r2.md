# Story 14.5e — feasible persistent-lock microrevision

## Verdict

**GO for one lock-only TDD correction after the requirements delta below is treated as normative.**

This microrevision replaces only `architecture.md` §§2–4, lock-related runtime ordering, and lock-related tests. The fixed attempt-4 namespace, immutable A3 lineage, authorization, publication, budget, protected-byte, and no-real-execution boundaries remain unchanged. Reviewer r2 findings B1, B2, B4, B5, and B6 remain independently blocking and must still be corrected and re-reviewed; this document resolves B3 only.

No A4/B4 invocation, model/data/provider access, training, inference, cleanup, commit, or push is authorized.

## 1. Feasibility boundary

macOS provides no atomic operation equivalent to “unlink this pathname only if it still names the inode held by this open descriptor.” An `lstat(path)` followed by `unlink(path)` always has a substitution window. Therefore no implementation may claim race-safe competitor preservation while deleting the shared-lock pathname.

Use one persistent regular file at the existing canonical path:

```text
/Volumes/Data NVME/mlx-ft/ds4/.ds4-ft.lock
```

Ownership is the conjunction of:

1. one live private read/write file descriptor;
2. an exclusive nonblocking BSD advisory lock acquired with `fcntl.flock(fd, LOCK_EX | LOCK_NB)` and held for the entire owned interval;
3. the acquisition `(st_dev, st_ino)` identity;
4. the canonical pathname still naming that same regular single-link inode at each attestation;
5. exact owned bytes containing the current PID and an invocation-unique token; and
6. complete in-process ownership state.

The pathname is never unlinked or renamed by acquisition, rollback, release, collision handling, failure handling, or normal completion. This guarantees that release cannot delete a replacement inode. Advisory locking provides singleton exclusion among cooperating DS4 invocations. Non-cooperating pathname replacement is tampering: the current invocation detects it at the next attestation and fails closed, but no pathname-based advisory protocol can prevent a privileged external actor from replacing the file between observations.

## 2. Exact persistent states

Define byte constants, not permissive parsers:

```python
_LOCK_IDLE_BYTES = b"ds4-ft-lock-v1\nstate=idle\n"
```

Owned bytes are exactly:

```text
ds4-ft-lock-v1
state=owned
pid=<decimal current PID>
token=<64 lowercase hexadecimal characters>
```

The final newline is mandatory. Generate the token with `secrets.token_hex(32)` after exclusive flock acquisition. No whitespace variants, legacy `pid=<pid>\n`, empty file, partial payload, unknown version, unknown key, reordered key, extra key, or malformed token is accepted.

Private ownership state is narrow and explicit:

```text
_LOCK_OWNED_FD
_LOCK_OWNED_PATH
_LOCK_OWNED_IDENTITY
_LOCK_OWNED_PID
_LOCK_OWNED_TOKEN
_LOCK_OWNED_PARTIAL
_LOCK_OWNED_FLOCKED
```

`_LOCK_OWNED_FD` remains open from successful acquisition until release/rollback finishes. `O_CLOEXEC` is mandatory. No attestation opens a replacement descriptor and closes the owning descriptor. `_LOCK_OWNED_FLOCKED` is lifecycle evidence, not an independent authority: it is set only after successful `flock`, and every production path that unlocks or closes the descriptor clears all ownership state in `finally`.

## 3. Pre-lock verification

`_verify_ft_lock(_LockStage.PRE_LOCK, workspace)` accepts exactly two cases.

### First-use absence

`lstat(path)` returns `ENOENT`. This is admission only. Acquisition must repeat all checks because another process can create the path after pre-lock verification.

### Released idle file

For an existing path:

1. `lstat` must identify a regular file with exactly one link.
2. Open the same pathname `O_RDWR | O_NOFOLLOW | O_CLOEXEC`, without creation.
3. `fstat(fd)` and repeated `lstat(path)` must be stable and identify the same `(st_dev, st_ino)`, regular type, one-link count, and size.
4. Acquire `LOCK_EX | LOCK_NB` before reading. `EWOULDBLOCK`/`EAGAIN` means an active owner and rejects admission. Any other flock error fails closed.
5. Read through the descriptor and require byte-exact `_LOCK_IDLE_BYTES`, with stable `fstat`/`lstat` identity before and after the read.
6. Unlock and close in `finally`. Pre-lock verification never writes.

Symlink, dangling symlink, directory, FIFO, socket, device, hard-linked file, unreadable file, stale owned bytes, legacy bytes, partial/empty bytes, replacement, metadata drift, or any inspection error rejects. No pre-lock failure deletes, renames, truncates, rewrites, quarantines, or chmods anything.

## 4. Acquisition

`_acquire_ft_lock()` preserves the existing canonical path, 60-second timeout/poll interval, watchdog behavior, and one-attempt semantics, but uses the persistent file.

1. Run explicit PRE_LOCK verification.
2. Try first-use creation with `O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC`, mode `0600`. On `EEXIST`, open the existing path with `O_RDWR | O_NOFOLLOW | O_CLOEXEC`; never truncate at open.
3. Require a regular single-link descriptor. Acquire `LOCK_EX | LOCK_NB`. A busy file is a collision; close it and continue only under the existing bounded timeout. No collision path mutates the file.
4. Repeatedly compare `fstat(fd)` and `lstat(path)`. They must identify one stable inode. For an O_EXCL-created first-use inode, initial zero length is accepted only in that same creation call. For an existing inode, exact idle bytes are mandatory while flock is held.
5. Generate the 64-hex token. Record FD, canonical path, identity, PID, token, `PARTIAL=True`, and `FLOCKED=True`.
6. Through the owned descriptor only, seek/truncate, write the complete owned payload with a short-write loop, `fsync`, and verify exact bytes, size, stable descriptor metadata, and pathname identity.
7. Set `PARTIAL=False` only after complete owned attestation succeeds. Keep FD and flock live; return the canonical path.

A pre-lock/acquire race is resolved only by the flock and repeated identity/content checks. Acquisition never trusts the earlier pre-lock observation.

### Acquisition failure

There is no unlink rollback. While the FD and flock remain live, rollback may restore exact idle bytes through that same FD only if the canonical pathname still names the acquisition inode and rollback can complete write, `fsync`, exact-byte verification, and stable identity verification. It then unlocks/closes and clears ownership state.

If identity is lost, bytes are tampered, or idle restoration cannot be proved, rollback performs no pathname mutation. It closes/unlocks in `finally`, clears live-ownership globals because no lock remains held, and raises a terminal `PilotError`. A partial or owned payload may remain; future PRE_LOCK rejects it. No automatic cleanup or retry converts such state to idle.

## 5. Post-lock attestation

`_verify_ft_lock(_LockStage.POST_LOCK, workspace)` requires the canonical path and calls one exact `_owned_lock_attestation()` over the still-open owning FD.

Attestation requires:

1. all ownership globals are present and internally consistent;
2. `PARTIAL` is false and `FLOCKED` is true;
3. stored PID equals `os.getpid()` and token matches exactly 64 lowercase hex characters;
4. FD is valid, read/write, and close-on-exec;
5. descriptor is a regular single-link file whose `(st_dev, st_ino)` equals acquisition identity;
6. repeated `lstat(path)` is regular, single-link, stable, and equals descriptor identity;
7. exact owned bytes, size, PID, token, and final newline are read through the owning FD using offset-stable I/O; and
8. repeated `fstat`/`lstat` after the read match the first observations.

The flock proof is lifecycle-based: successful exclusive flock acquisition plus a live private FD and no intervening production unlock/close. macOS exposes no side-effect-free “does this FD still own flock?” query. Tests must additionally prove exclusion with a separate process attempting `LOCK_EX | LOCK_NB`; do not use a second descriptor in the same process as authoritative proof.

Absent/closed FD, cleared or contradictory state, wrong PID/token/path/inode, unlocked lifecycle state, partial state, modified bytes, replacement, symlink, non-regular path, link-count drift, read/stat error, or metadata race fails. Attestation never closes the owning FD and never mutates the pathname.

## 6. Release and crash semantics

### Normal release

`_release_ft_lock()` has one success path:

1. Exact owned attestation passes while the owning FD still holds flock.
2. Through that FD only, write/truncate exact `_LOCK_IDLE_BYTES`, `fsync`, and verify exact idle bytes plus stable descriptor/path identity while flock remains held.
3. The successful-release linearization point is the final same-inode idle attestation while flock remains held.
4. Unlock and close in `finally`, then clear every ownership global.

The persistent pathname and inode remain. A subsequent process may acquire immediately after unlock, so release must not require post-unlock pathname absence or idle bytes; a valid successor may already have written its owned state.

### Release failure or tamper

If owned attestation or same-path idle transition fails, release does not write to a replacement pathname and never calls unlink, rename, replace, quarantine, or cleanup. It closes/unlocks the owned FD in `finally`, clears live-ownership globals, and raises `PilotError`. Replacement bytes and inode remain untouched. If the original inode became unlinked or hidden, closing the FD lets the kernel reclaim it naturally; code performs no pathname operation.

Failure to write/fsync exact idle may leave stale owned or partial bytes. That state is intentionally non-admissible and requires separately authorized operator investigation. Retaining globals after closing would falsely claim ownership and is forbidden.

### Process crash

Kernel FD teardown releases flock. The last fsynced owned payload can remain. PRE_LOCK treats that payload as stale, not idle, and rejects it even though flock is free. There is no PID-liveness heuristic, age timeout, stale takeover, byte rewrite, deletion, rename, or quarantine. Recovery is outside this slice and requires explicit operator inspection/authorization.

## 7. A3 historical fact and B4 reuse

`lock_absent_after_exit: true` in committed A3 failure evidence remains an exact immutable historical value and hash input. The attempt-3 historical verifier validates that recorded value; it must no longer require the current filesystem pathname to remain absent forever.

First A4 admission may observe the migrated absent path. Successful A4 release leaves exact idle bytes. B4 pre-lock must then accept that exact idle regular file only after proving no active flock, and B4 acquisition reuses the same inode under a new PID/token-owned interval. Every B4 authorization, A4 dependency, report, marker, collision, identity, source, history, and budget gate remains unchanged. A4 crash/stale/tampered state blocks B4. Final B4 release also leaves exact idle state; no phase or final publication path deletes the singleton lock.

## 8. Requirements reconciliation

The following earlier clauses are infeasible and are replaced for this slice:

| Existing clause | Replacement |
|---|---|
| R14.5e-1 PRE_LOCK requires current pathname absent | PRE_LOCK accepts absent first use or exact unlocked idle v1 state under stable non-following FD/path identity. |
| R14.5e-2 owned bytes are only `pid=<token>\n` and token equals PID | Owned payload has fixed v1 header/state, decimal PID, and independent 64-hex invocation token. |
| R14.5e-3 normal release/rollback unlinks and proves pathname absent | No code path unlinks or renames. Normal release writes/fsyncs exact idle bytes through the flock-owning FD, then unlocks/closes. |
| R14.5e-3 retains ownership globals after release failure | Release/rollback always closes/unlocks in `finally` and clears live-ownership globals; terminal failure and stale bytes preserve fail-closed evidence. |
| R14.5e-4/current checks treat A3 `lock_absent_after_exit` as perpetual current absence | Validate it only as immutable recorded A3 history. Current A4/B4 lock state follows PRE/POST persistent-file semantics. |
| R14.5e-8 expects successful release to prove absence | Prove same inode contains exact idle bytes at the release linearization point, no deletion API was called, globals cleared, and a separate process can subsequently acquire. |

All other requirements remain binding. Before Coder r3 completion, BA-owned `docs/backlog.md` wording must match this table; Coder must not silently retain contradictory absence/unlink assertions.

## 9. TDD matrix

Write failing tests before production changes. All verdict-contributing test files must be tracked.

### Pre-lock and migration

- absent first-use passes without mutation;
- exact idle regular unlocked file passes without mutation;
- idle file exclusively flocked by a child process rejects;
- inactive owned/stale and legacy `pid=...` payloads reject;
- empty, partial, extra, reordered, malformed-version/token/newline payloads reject;
- symlink/dangling symlink, directory, FIFO/socket/non-regular, hard link, unreadable path reject;
- substitution at lstat/open/fstat/flock/read/final-stat seams rejects and preserves every competing pathname.

### Acquisition and post-lock

- absent path creates once, writes exact owned v1 payload, keeps FD open, and blocks child-process flock;
- exact idle path reacquires without inode replacement;
- competitor appearing after PRE_LOCK wins flock and causes bounded collision without mutation;
- wrong/cleared FD, PID, token, identity, path, partial/flocked state, bytes, size, type, link count, or close-on-exec state rejects POST_LOCK;
- pathname replacement and descriptor/path content mutation reject without delete/rename;
- full post-lock gate still independently enforces every non-lock attempt-1/2/3, authorization, collision, revision, source, identity, dependency, and publication fact.

### Release, rollback, and crash

- normal release writes/fsyncs exact idle bytes on the same inode, never invokes unlink/rename/replace, closes FD, clears globals, and allows a child/new invocation to acquire;
- inject replacement immediately before every release lstat/write/fsync/final-stat/unlock seam; replacement inode/bytes always survive untouched;
- owned-byte tamper, symlink/non-regular replacement, write failure, short write, fsync failure, unlock failure, and close failure never delete/rename/quarantine and produce terminal failure;
- acquisition failure before and during owned write either proves exact idle restoration on same inode or leaves a rejected stale state; never deletes;
- subprocess `os._exit` after acquisition releases flock but leaves owned bytes, and next PRE_LOCK rejects stale state;
- A4 successful release to idle followed by B4 pre-lock/acquire/post/release succeeds on the persistent singleton; A4 crash state blocks B4.

Use deterministic mutation hooks around file operations. Do not rely only on source-text assertions. Reviewer and Test Manager must independently run the child-process flock tests on macOS.

## 10. Implementation and documentation scope

Lock implementation remains in `scripts/ds4_segmented_pilot.py`; do not add a generic lock library, alternate lock path, directory lock, cleanup command, stale takeover, or compatibility flag. `scripts/finetune_ds4.py` changes are required only if wrapper assertions currently encode lock absence after successful A4/B4 release.

Coder r3 must update:

- `docs/architecture.md`: persistent singleton design, threat/feasibility boundary, lifecycle state machine, and B4 reuse;
- `docs/technical-spec.md`: exact idle/owned bytes, open/flock flags, pre/post checks, release linearization, crash/stale operator behavior, and test commands;
- `docs/backlog.md`: BA-owned acceptance wording reconciled with §8 before final gate.

No ADR is required because this remains the existing Story 14.5 singleton mechanism, corrected to a feasible implementation. Canonical docs must not claim pathname absence after normal A4/B4 release or race-safe conditional unlink.

## 11. STOP conditions

STOP on any pathname unlink/rename/replace/quarantine; auto-repair of stale or malformed bytes; PID-liveness or age-based takeover; acceptance of owned bytes without live in-process FD/flock lifecycle; closing the owner FD before release; skipping any non-lock gate; weakening A3 evidence hashes/values; treating historical A3 absence as current perpetual absence; alternate/dynamic lock path; attempt-3 reuse; attempt-5 allocator; real execution; untracked verdict tests; commit; or push.
