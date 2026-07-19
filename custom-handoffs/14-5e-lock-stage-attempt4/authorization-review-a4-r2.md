# External Phase A4 authorization review r2 — attempt 4

## Verdict

**PASS — authorize exactly one invocation of the unchanged A4 wrapper.**

Authorized artifact only:

- revision: `07fe6c8e13651892e05014b40e78210f0d306462`
- authorization: `/tmp/ds4-phase-a4-authorization.json`
- authorization SHA-256: `6ac22cca3e8a795658f03436bfd9d14a9ba70dc75ceb05c488a1f538afbe54ea`
- wrapper: `agent-output/cmux-14-5-attempt-4/run-phase-a4.sh`
- wrapper SHA-256: `2d9d4e50827eeae759fc6843b9e75a57dd6ffe3d4bada1261dab520352a2e69d`
- canonical command SHA-256: `c4d22b662b6e2cbd585db383675bda39da0c974b5d396e5ced57de0bdffae829`

No retry, B4, fallback, cleanup, suffix, altered authorization, altered wrapper, alternate command, or second A4 invocation authorized.

## Prior denial closure

Prior blocker closed.

Fresh production `_runtime_preflight()` returned canonical `attempt4-authorized-identity-v1` identity with exact keys `schema`, `immutable`, `dynamic_resources`, `repository_status`; repository status `[]`.

Production projection retained exact `schema`, full immutable runtime manifest, clean repository status, and pinned dynamic-resource policy. Computed trusted projection SHA-256:

`3f23e879163cbb99621547061de69befbc42bb25f110dad5514ee52cfb370c93`

Exact authorization carries same root. `canonical_attempt4_authorization(..., trusted_identity=<fresh production preflight>)` regenerated authorization byte-for-byte. `validate_attempt4_authorization(..., verify_sources=True)` passed.

## Independent checks

### Revision, source, protected paths, history, phase, prewrite

PASS:

- HEAD exact: `07fe6c8e13651892e05014b40e78210f0d306462`
- repository status before validation: `[]`
- `git diff --check`: PASS
- vendor HEAD: `80fab4e419a57f9465bb9e2f4e90010d645e124c`
- vendor status: `[]`
- outer vendor gitlink exact
- pilot source root: `021a4c2979bd7f801b9520212b51052ea67abe5d374485d741f2fe484ff374ff`
- catalog source root: `015f7e9c94555c9b3fbf24f3e29d81a723e373e76dba643641554310641e761e`
- protected-files root: `207b43da07d146819d9cd83a5f2711ae4fdaba4de5415f6327edd8add1ae0115`
- attempt-2 manifest root: `2cc1d2359017cce2130a9428fd202e8c2f35b5d5810030c95aaac8b464fd3d88`
- attempt-3 history root: `dc576e1ca032c3246114801fd502cf6e24cdc8916f4292bfdcb63b4f32df8442`
- phase-specific authorization root: `b9063fb2e10e8f5b99816390e642772bca82427a0ae91cac040db8246e09796b`
- prewrite root: `17bf86648c85837117fbf7e45c4deab06100edaee4fa93feae4acc63d93f526c`
- authorization strict 11-key canonical compact JSON schema: PASS
- phase binding: `phase-a`

### Mutation-free production launch check

Exact production-default A4 `--launch-check-only` command from wrapper environment exited `0`.

Result bound attempt `4`, phase `phase-a`, exact revision, supplied authorization, fresh trusted runtime identity, history gates, namespace absence, and pre-lock shared-lock gate.

Before and after launch check:

- all 25 canonical A4 namespace paths absent
- `/Volumes/Data NVME/mlx-ft/ds4/.ds4-ft.lock` absent
- no A4 log
- no A4 output
- no A4 report
- no A4 authorization copy
- no A4 success/failure/final marker
- repository status `[]`
- vendor status `[]`

### A3 immutability

Production `verify_attempt3_historical_evidence(PRE_LOCK)` passed before and after launch check. Thirteen required A3 absence facts remained true. Immutable files remained exact:

- `phase-a-authorization.json`: size `532`, SHA-256 `543e584621c133e6d8afc1a7b6d0a3bea296a78d2de18f77fd84fcb3bf23b065`
- `phase-a-log.txt`: size `619`, SHA-256 `77b2dc85dc02735dc1727ffc06f2fb9471230c90f8808d67da7e10bb852ca1bc`
- `phase-a3-runtime-preflight-failure.json`: size `1075`, SHA-256 `0f19ce8a8147a39dffcb6b3162f67779fd6d26263ed2015285285350c8dde3da`

### Wrapper

PASS:

- `bash -n`
- byte-exact regeneration from production catalog renderer
- exact embedded A4 authorization consumed successfully by launch check
- exactly two pilot subprocesses: one mutation-free launch check, one training invocation
- training invocation only `phase-a`, attempt `4`
- phase spec: namespace `ds4-segmented-pilot-attempt-4`, 2 iterations, 2 steps/eval, 2700-second watchdog, no resume source
- exact train pins: LoRA, 16 layers, batch 1, learning rate `1e-5`, max sequence 4096, prompt masking, gradient checkpointing, accumulation 1, seed 0, Adam, 25 validation batches, report every step, save every step, segment size 1
- filtered dataset exact: `/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke`
- exclusive first-use log via shell `noclobber`
- persistent shared lock owned inside production pilot by canonical `/Volumes/Data NVME/mlx-ft/ds4/.ds4-ft.lock` inode and `flock`
- training pipeline propagates pilot status through `PIPESTATUS[0]`
- no retry, fallback, cleanup, suffix, phase-B training, or loop semantics

## Findings

None.

## Authorization boundary

PASS authorizes exactly one execution of unchanged `agent-output/cmux-14-5-attempt-4/run-phase-a4.sh` with unchanged `/tmp/ds4-phase-a4-authorization.json`, while all reviewed preconditions remain true.

Any byte change, revision/status drift, namespace/lock creation before invocation, failed invocation, or consumed namespace requires stop and fresh independent authorization. No retry or cleanup authorized.

Reviewer ran no wrapper, training, model load, provider call, optimizer update, inference, cleanup, commit, push, or marker-producing validation.
