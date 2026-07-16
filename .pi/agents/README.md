# Project-local reusable Pi role agents

## Prerequisites

These wrapper scripts require the **global role-pipeline skill** installed under
your Pi agent skills directory. By default:

```bash
ls "${HOME}/.pi/agent/skills/role-pipeline/scripts/"
```

Override the directory via `AGENT_SKILLS_DIR` or `PI_AGENT_SKILLS_DIR`:

```bash
export AGENT_SKILLS_DIR=/custom/path/to/skills
.pi/agents/bin/launch-role.sh ba slice-name "task text"
```

Each wrapper is a portable bash script (mode 755 regular file, not a symlink)
that resolves its same-name target under the role-pipeline skill at runtime.
If the target is absent, the wrapper fails with a clear installation message.

## Role agent rules

- Reuse the existing cmux pane for each role: BA, Architect, Coder, Reviewer, Test Manager.
- If a role pane does not exist, spawn it once, then reuse it for later slices/follow-ups.
- Before assigning a new task to a reused role pane, the supervising parent/operator sends `/new` and waits for the fresh session prompt. Do **not** tell the role agent to run `/new` inside the task prompt; that can wipe its own task context or lose the final result.
- Supervision defaults to the normal inline/foreground completion check unless the user explicitly asks for background monitoring.
- Do not copy role prompts or launchers into `agent-output/<slice>/`.
- `agent-output/<slice>/` is for handoff artifacts only: `requirements.md`, `architecture.md`, `coder-notes.md`, `review.md`, `test-report.md`, follow-up reports, and logs if explicitly requested.
- Canonical backlog lives in `docs/backlog.md`; BA updates it by default with Requirements, trackable user stories, and acceptance criteria.
- User story format: `As a [type of user] (WHO), I want [some goal] (WHAT), so that [some reason] (WHY).`
- `.cmux-status/` is only for `*.done` status markers.
- Each agent must finish with one terminal JSON object **in its own role pane only — never forwarded to the parent pane or any other cmux surface** (forwarding was removed because it interfered with the parent chat): `{"status":"ok","role":"<Role>"}` on success, or `{"status":"error","error":"<message>","role":"<Role>"}` on failure — where `<Role>` is one of `BA`, `Architect`, `Coder`, `Reviewer`, `Test Manager`. The parent supervises completion via `.cmux-status/*.done` markers + required handoff files, which are canonical.
- Launch Pi visibly/interactively by default: no `-p`, no `tee` pipeline.
- Current default role model routing (2026-06-20): BA/Architect use `neuralwatt/glm-5.2-short` (200k ctx) with `anthropic/claude-opus-4-8` as Ctrl+P fallback; Test Manager uses `neuralwatt/qwen3.6-35b` with `openai-codex/gpt-5.4-mini` as Ctrl+P fallback; Coder/Reviewer remain `openai-codex/gpt-5.5`.
- Completion is verified by required handoff files plus `.cmux-status/*.done` markers. The terminal JSON status in the role pane is a secondary diagnostic only, never forwarded to the parent pane.

## Launch

```bash
.pi/agents/bin/launch-role.sh ba cmux-11-15 "Write requirements for Story 11.15a."
.pi/agents/bin/launch-role.sh architect cmux-11-15 "Review requirements and write architecture."
.pi/agents/bin/launch-role.sh coder cmux-11-15 "Implement reviewer follow-up blockers using TDD."
.pi/agents/bin/launch-role.sh reviewer cmux-11-15 "Review coder follow-up; do not edit files."
.pi/agents/bin/launch-role.sh tester cmux-11-15 "Run independent validation."
```

The launcher composes the reusable role prompt with the slice name and task text, then starts interactive Pi in the current pane.
