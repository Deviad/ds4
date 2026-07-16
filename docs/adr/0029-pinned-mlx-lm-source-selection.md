# ADR 0029: Pinned MLX-LM source selection

## Status

Accepted for Story 14.0 bootstrap.

## Context

Path A ended with Story 13.3b-5i's permanent STOP. Epic 14 is separately authorized and does not reopen that evidence. Story 14.0 only prepares a reproducible MLX-LM source boundary so later reviewed work can evaluate trainer-level ideas without editing MLX core, touching real training assets, or modifying the fork.

The authorized fork is `git@github.com:Deviad/mlx-lm.git`. The initial pin was commit `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`. Story 14.4 advanced the pin to `80fab4e419a57f9465bb9e2f4e90010d645e124c` to include the Story 14.1/14.2a trainer seam (`loss_and_grad` passthrough) and test suite. The fork and the released package both report MLX-LM version `0.31.3`, so version alone cannot identify the selected source.

## Decision

- Add `vendor/mlx-lm` as a real git submodule pinned by the outer index gitlink. Do not add a branch-following key.
- Keep the declarative MLX environment on released `mlx-lm==0.31.3` and `mlx==0.31.2`.
- Default `setup-env` and `run-command mlx-lm-source` to release mode.
- Make fork mode an explicit operator choice: `run-command mlx-lm-source --mlx-lm-source fork --execute --yes`.
- Switch sources only inside the isolated MLX virtual environment using the venv's absolute `python -m pip`, `--isolated`, `--require-virtualenv`, and `--no-deps`. Fork mode uses `--no-build-isolation --force-reinstall -e vendor/mlx-lm`; release mode uses `--force-reinstall mlx-lm==0.31.3`.
- Verify identity before mutation and after mutation. Release mode must import from the venv and outside `vendor/mlx-lm`; fork mode must import from `vendor/mlx-lm` and have editable PEP 610 metadata pointing at the fork path.

## Consequences

Story 14.0 remains bootstrap-only. It edits no file below `vendor/mlx-lm`, changes no trainer behavior, accesses no model/data/shard asset, and makes no claim that MLX-LM can fix retained Metal command-buffer lifetime. Any functional fork patch needs a later story with its own requirements, architecture, TDD, review, and test verdict.

Rollback from fork identity is the release selector plus verifier. Removing the submodule itself is a separate source-control rollback and is not part of Story 14.0 implementation.

## Amendment (Story 14.4)

The vendor inner pin advanced from `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe` to `80fab4e419a57f9465bb9e2f4e90010d645e124c` to include the Story 14.1/14.2a trainer seam (`loss_and_grad` passthrough) and test suite. The outer gitlink was updated to match. This is an authorized advancement of the submodule pin, not a source or identity change: the fork URL remains `git@github.com:Deviad/mlx-lm.git`, no branch-following is added, and the fork-identity verification (import path, PEP 610 metadata, module-level hash) remains the verification mechanism.
