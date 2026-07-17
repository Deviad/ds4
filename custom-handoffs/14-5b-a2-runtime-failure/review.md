# Story 14.5b A2 runtime-failure closeout — Reviewer

## Verdict: PASS

No blocker, major, or minor findings.

## Direct-evidence review

- Revision: `HEAD=e6d34fa03479316720430f35cc94d4606a45ef96`.
- Scope: tracked working-tree changes limited to `docs/backlog.md` and `training-next-status.md`; staged changes absent; `git diff --check` clean.
- Protected code: `scripts/ds4_segmented_pilot.py` and `scripts/finetune_ds4.py` tracked and byte-equal to `HEAD` (`git hash-object` equals `git rev-parse HEAD:<path>`).
- Run result: phase and pilot reports agree on attempt `2`, namespace `ds4-segmented-pilot-attempt-2`, status `fail`, exit `1`, error `safetensors __metadata__ must be an object`, contract digest `ca3169367b0542c8eb3204158ef6800e02694700b936f460612ffda1b5f2c0d4`, wall time `2444.2739184170496s`, watchdog cancelled, retry `none`, and lock acquired/released with one release attempt.
- Work completed: reviewed command/effective configuration specifies `--iters 2`; log records training `2/2`, iteration losses `19.334` and `17.648`, validation losses `19.553` and `19.841`, and saves of `0000001_adapters.safetensors` and `0000002_adapters.safetensors`. `phase-a-start.safetensors` and final `adapters.safetensors` also exist.
- Header inspection: `phase-a-start.safetensors`, `0000001_adapters.safetensors`, `0000002_adapters.safetensors`, and `adapters.safetensors` each contain `__metadata__: null`.
- Failure bindings: phase failure marker binds `phase-a-report.json` by SHA-256 `d9cbc1895b62f4f182a25dcf58d851d077c6f92745bd3b0d387c78fa6dc8a500`; final failure marker binds `pilot-report.json` by SHA-256 `84e8d26cf658f4ba15eda8346b4e507dc331d65da29b00b21aca1073eb97c0ce`. Pilot report binds the phase-report path. Both markers record attempt `2`, exit `1`, status `fail`, matching namespace and contract digest.
- Current gate: both failure markers present; Phase A2 and final OK markers absent; `/Volumes/Data NVME/mlx-ft/ds4/.ds4-ft.lock` absent; zero live `finetune_ds4.py`, `ds4_segmented_pilot.py`, or attempt-2 training processes; no Phase B2 attempt-2 path observed.
- Disposition: manifest/report `retry_performed=false` / `retry=none`; attempt-2 evidence remains hash-stable, consumed, and non-reusable; B2 remains blocked and unauthorized.
- Documentation: both canonical docs record exact revision, one consumed authorization, `2/2` calls/updates, four saved safetensors, exact losses/validation losses, bounded runtime, exact post-training failure, four null metadata observations, failure/OK-marker state, lock/watchdog/retry state, immutable attempt-2 disposition, future fresh-cycle/new-namespace/fresh-authorization requirements, B2 block, and explicit non-claims for convergence, quality, continuity, resume, and readiness.

## Manifest verification

All 10 manifest entries exist and match declared size and SHA-256:

1. `phase-a-log.txt` — `edf1ae2215083aab2b8648403cf53f395ce39c3da2277f80fa7da2e87976734e`
2. `phase-a-report.json` — `d9cbc1895b62f4f182a25dcf58d851d077c6f92745bd3b0d387c78fa6dc8a500`
3. `pilot-report.json` — `84e8d26cf658f4ba15eda8346b4e507dc331d65da29b00b21aca1073eb97c0ce`
4. `.ds4-segmented-pilot-attempt-2-phase-a-fail` — `079c89c88bf87699562ce8e5866d88ba3e92adec6d2b0c787a481778d85b2918`
5. `.ds4-segmented-pilot-attempt-2-fail` — `91b4dd994a3bd0319c5d0edf9d0c49f933cc95f2e49a1b1f359e4a92e609edfe`
6. `0000001_adapters.safetensors` — `89363e82325e095bc74f7cdb6e022280438fa1667d52abcf33f1dfb3db1a2d18`
7. `0000002_adapters.safetensors` — `2b173d701b09fc9da2bb0a9553cc7b7eb99bacebf9fa6f10e00d7b4305ba3652`
8. `adapter_config.json` — `13620f30b39a0de718f0769830b4b0a2233eaebea62d6fc43da08231d0a0691d`
9. `adapters.safetensors` — `2b173d701b09fc9da2bb0a9553cc7b7eb99bacebf9fa6f10e00d7b4305ba3652`
10. `phase-a-start.safetensors` — `c6dd9f67be04936c4dce58d7b3643403354516d6167e4e08d76c393890339e51`

Review read-only. No test, training, provider invocation, cleanup, commit, push, production edit, or runtime-artifact mutation performed.
