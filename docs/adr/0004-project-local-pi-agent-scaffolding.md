# ADR 0004: Reusable Pi role agents live in `.pi/agents/`

Date: 2026-06-18
Status: Accepted

## Context

The project uses a repeated BA → Architect → Coder → Reviewer + Test Manager workflow for implementation slices. Earlier slices duplicated prompts and launch scripts under per-slice `agent-output/` directories, making workflow improvements hard to preserve and causing inconsistencies in cmux launching, marker handling, and visible terminal behavior.

Pi's standard project-local resource directory is `.pi/` for skills, prompts, extensions, system prompts, and project-local packages. Pi does not define a built-in role-agent folder.

## Decision

Reusable project-local role agent definitions and launchers live under `.pi/agents/`. Per-slice directories under `agent-output/` contain only handoff artifacts, logs, reviews, reports, and archived markers.

By default, cmux role agents are launched in visible interactive Pi mode inside panes: the prompt is passed as a positional argument, without `-p` and without piping through `tee`. Non-interactive `pi -p | tee` is used only when explicitly requested for background/log-only runs.

Completion is verified by `.cmux-status/*.done` markers and required handoff files first; pane text and logs are secondary diagnostics.

## Consequences

- Workflow fixes are made once in `.pi/agents/` and reused across slices.
- Users can watch agent output in cmux panes by default.
- Per-slice history remains clean under `agent-output/<slice>/` without duplicating agent definitions.
- Marker handling remains the authoritative synchronization contract.
