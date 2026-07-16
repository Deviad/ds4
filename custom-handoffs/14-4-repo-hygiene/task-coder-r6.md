# Story 14.4 — Coder r6 portable agent launchers

Read latest `custom-handoffs/standby/review.md`. Close B1/N1 only.

Replace all 14 absolute `.pi/agents/bin/*.sh` symlinks with regular executable portable wrappers. Each wrapper must:
- derive same-name target from `${AGENT_SKILLS_DIR:-${PI_AGENT_SKILLS_DIR:-${HOME}/.pi/agent/skills}}/role-pipeline/scripts/`;
- fail clearly with path + installation guidance if target absent;
- when sourced (library scripts), `source` target without exec;
- when executed, `exec` target with all arguments;
- contain no `/Users/spotted` or project-absolute path.

Document global role-pipeline skill prerequisite and env override in `.pi/agents/README.md`. Add focused shell tests or deterministic validation that every staged launcher is mode 100755 regular file, resolves under temp/non-spotted HOME or AGENT_SKILLS_DIR override, executable wrappers forward args, source wrappers load successfully, and absent target fails clearly. Avoid committing generated files.

Correct coder-notes-r5 stale count via coder-notes-r6 superseding it. Force-add task/notes. Rebuild Commit 1 manifest LAST and verify exact index equality. Run shell syntax, wrapper portability checks, cached diff, zero unstaged/nonignored untracked, binary scan, synthetic clone exact 246 suite, Path A/protected/vendor gates. Final reports remain ignored outside Commit 1. No model code changes, commit, or push. Marker.