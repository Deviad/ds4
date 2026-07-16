# ADR 0005: Agent-output is handoff evidence, not canonical technical documentation

Date: 2026-06-18
Status: Accepted

## Context

The project uses multi-agent handoffs under `agent-output/<slice>/` for requirements, architecture, coder notes, reviews, and test reports. These files are useful evidence, but they drift quickly when a slice is superseded by follow-up work. Letting `agent-output` become the only place where technical decisions live causes stale architecture and duplicated role-agent definitions.

## Decision

Canonical technical documentation lives under `docs/`:

- `docs/architecture.md` for whole-project scaffolding and durable subsystem architecture.
- `docs/adr/` for architecture decisions and workflow policy.
- `docs/technical-spec.md` for build/run/test mechanics and invariants.
- model-specific docs such as `docs/deepseek-v4-architecture-dossier.md` for durable model mapping.

`agent-output/<slice>/` is non-canonical handoff evidence. When an agent creates or changes a durable technical decision, the agent must update the relevant canonical doc in the same slice. If the idea is exploratory, the agent must explicitly say it is not canonical.

## Consequences

- Slice `architecture.md` files may exist, but they must not be the only source for durable design choices.
- Reviewer checks include documentation drift: canonical docs updated or explicit non-canonical deferral recorded.
- Historical `agent-output` docs should be consolidated into `docs/architecture.md` or ADRs when they become durable.
- Per-slice role prompts/launchers stay out of `agent-output`; reusable agent scaffolding lives under `.pi/agents/`.
