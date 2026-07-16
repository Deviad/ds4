# ADR 0006: Canonical backlog lives in `docs/backlog.md`

Date: 2026-06-18
Status: Accepted

## Context

The project previously used `DS4_Finetuning.md` at the repository root as the epic/story plan. That made the backlog feel separate from architecture docs and encouraged drift between handoff files and canonical planning.

## Decision

The canonical backlog lives at `docs/backlog.md`. It contains Requirements, trackable user stories, and acceptance criteria. BA agents own backlog updates by default for every slice.

User stories use this exact format:

```text
As a [type of user] (WHO), I want [some goal] (WHAT), so that [some reason] (WHY).
```

`.cmux-status/` is only for status marker files (`*.done`). No backlog, prompts, logs, plans, or notes go there.

## Consequences

- Root `DS4_Finetuning.md` is retired.
- Agent handoff requirements may be written under `agent-output/<slice>/requirements.md`, but canonical backlog changes must be reflected in `docs/backlog.md`.
- Reviewers should flag slices where BA changed requirements only in handoff files and not in the canonical backlog.
