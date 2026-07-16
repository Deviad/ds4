<!-- project-overlay.md — injected into every role prompt (overlay slot).
     Keep it SHORT and durable: project purpose, hard rules, docs map.
     Slice-specific detail belongs in the task file, not here. -->

## Project
<!-- One paragraph: what this codebase is, what it is NOT, the core goal. -->

## Always read first
<!-- The canonical docs a role must read before acting, e.g.:
- AGENTS.md
- docs/architecture.md
- docs/backlog.md
-->

## Hard rules
<!-- Project-specific inviolables, e.g.:
- Frozen/off-limits files or symbols and the ADR that freezes them.
- Language bans (e.g. "no C++"), dependency policies.
- Required verification style (source hashes vs git diff, etc.).
-->

## Layout
<!-- Short map of the important directories/files. -->

## Build & test
<!-- The commands: build, unit tests, the baseline count if one is tracked. -->

## Environment & dependencies
<!-- Project environment policy: venv/container/toolchain commands, canonical
     dependency manifests, and forbidden global installs. Example:
- Use `.venv`; never install project packages globally.
- Add dependencies to `pyproject.toml`; do not encode package lists in scripts.
- Unset machine-specific env vars that can break reproducibility.
-->

## Resource safety / exclusive jobs
<!-- Heavy jobs and resources that must not overlap, plus lock/preflight rules.
     Example:
- Only one model process using >100GB RAM at a time; check competing processes
  before launch. The instance lock is intentional — never bypass it.
- Training and quantization are mutually exclusive on this host.
- Resource contention from unrelated work makes a run PENDING-RESOURCE, not a
  code regression; pause/kill only with user authorization, then rerun clean.
-->

## Protected paths (regression matrix)
<!-- Paths/subsystems that NO unrelated change may silently break, each with
     its check command and whether it needs user consent first. Roles run the
     affected checks at every major change; Tester includes them in test-report.
     Example:
     | Path | Check | Consent needed? |
     |---|---|---|
     | Metal inference speed | `make bench` — speed at prior level | no |
     | Distributed inference  | live 2-node run                    | ASK USER (shared hardware) |
-->
