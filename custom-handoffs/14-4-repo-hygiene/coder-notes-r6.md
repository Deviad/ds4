# Story 14.4 — Coder r6 (portable agent launchers)

## r6 change: absolute symlinks → portable wrappers

All 14 `.pi/agents/bin/*.sh` were absolute symlinks pointing to
`/Users/spotted/.pi/agent/skills/role-pipeline/scripts/`. Replaced with
portable bash wrappers (mode 755 regular files) that resolve targets at
runtime via:

```
${AGENT_SKILLS_DIR:-${PI_AGENT_SKILLS_DIR:-${HOME}/.pi/agent/skills}}/role-pipeline/scripts/<name>
```

### Wrapper behavior
- Sourced mode (`${BASH_SOURCE[0]} != $0`): `source`s the target
- Executed mode: `exec`s the target with all arguments
- Target absent: prints clear FATAL message with installation guidance, exits 1
- No `/Users/spotted` or project-absolute paths anywhere in wrappers

### README updated
- Added Prerequisites section documenting global role-pipeline skill requirement
- Documented AGENT_SKILLS_DIR / PI_AGENT_SKILLS_DIR override

## Verification

| Check | Result |
|-------|--------|
| All 14 files mode 100755 regular (not symlinks) | ✅ |
| No absolute paths in wrappers | ✅ |
| Shell syntax (bash -n) | 14/14 OK |
| Exec mode: forwards to real target | ✅ |
| Sourced mode: sources target library | ✅ |
| Absent target: clears error + exits 1 | ✅ |
| AGENT_SKILLS_DIR override works | ✅ |
| 0 staged symlinks | ✅ |
| Full suite | 246/246 GREEN |
| Path A 365 | intact |
| Protected hashes 3/3 | OK |
| Whitespace | exit 0 |
| 0 untracked, 0 unstaged | ✅ |
| Commit 1: 116 add / 12 mod / 21 del / 149 total | ✅ |

## Supersedes r5
- r5 carried stale total count 145 (actual 147)
- r6 manifest and notes both reflect exact 116/12/21/149
