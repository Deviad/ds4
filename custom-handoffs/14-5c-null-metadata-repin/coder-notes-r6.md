# Story 14.5c — Coder r6 exact-validation matrix closure

## Scope

Direct RED-first synthetic implementation only. No delegate, cmux, real assets, training, cleanup, commit, push, or marker.

## r6 closure

- Final attempt-3 validation now requires explicit trusted pre-training Phase-A admission lineage and trusted Phase-A3 authorization. Every final field is derived from and exact-compared with canonical A3/B3 snapshots, including counts, updates, steps, progression, finite wall time, embedded reports, output, contract, non-claims, histories, authorizations, paths, contracts, and report hashes.
- Attempt-3 phase-A/phase-B/final marker validation now enforces exact JSON types, exact values, canonical paths, SHA-256 schemas, lifecycle values, and phase-A resume bindings. Boolean aliases cannot satisfy integer fields.
- Attempt-2 verification retains verifier-local trust-root literals, checks exact ten-file/eight-absence descriptor schemas, uses recursive exact JSON equality for the manifest, and rejects semantic, descriptor/hash, absence, and coordinated substitutions. Added an authentic synthetic fixture exercising the production verifier boundary.
- Canonical tensor parsing now wraps malformed dtype types as `PilotError`; MLX trainable schema extraction rejects missing/invalid/duplicate keys, dtype, and shape metadata while preserving nested MLX flattening.
- Added RED-first r6 mutation tests for final fields, marker fields, attempt-2 trust-root/absence semantics, malformed tensor dtype, and trusted publication inputs. Updated canonical docs to distinguish synthetic r6 completion from pending independent review/test and fresh authorization.

## Validation

- RED-first focused r6 run initially failed on the missing trusted final boundary and malformed dtype handling; implementation changed those boundaries and focused tests turned green.
- `PYTHONPATH=$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD python-envs/mlx/.venv/bin/python -m pytest -q tests/test_ds4_segmented_pilot.py -k 'r6_'`: `38 passed, 340 deselected, 1 warning`.
- Pilot regression: `378 passed, 1 warning`.
- Canonical vendor-first six-file suite: `624 passed, 3 skipped, 1 warning, 2 subtests passed`.
- Pilot source compile passed.
- No real A3/B3 access, training, cleanup, commit, push, or `.cmux-status` marker.
