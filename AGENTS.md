# Agent Notes

## Reply style — all panes default (caveman `ultra`)

Every cmux role pane (BA, Architect, Coder, Reviewer, Test Manager) and the
parent/supervisor pane reply in caveman `ultra` by default. See
`~/.pi/agent/skills/caveman/SKILL.md`.

Telegraphic fragments only: drop articles (`a`/`the`), copulas (`is`/`are`),
connectors (`however`/`therefore`); one phrase per line; no preamble, apology, or
filler; open with the answer.

Byte-exact exempt (never compress): code blocks, file paths, URLs, shell
commands, JSON keys, regex, version numbers, exit/error codes, HTTP statuses,
safety + uncertainty markers (`I don't know`, `verify this`, `may be stale`),
user-quoted text, and the final "what's next" line for actionable turns.

Deactivate per session with `normal mode` / `/caveman off` / `talk normal`.

## HARD STARTUP GATE — mandatory for implementation/code changes

Before making any implementation/code edit, spawn the full agent pipeline. Do
not start coding from a fresh session until every required role is assigned:

```text
BA → Architect → Coder → Reviewer + Test Manager
```

Required agents:

- **BA** — requirements, trackable user stories, acceptance criteria; owns `docs/backlog.md` updates.
- **Architect** — design, boundaries, invariants, implementation plan.
- **Coder** — TDD red → green implementation. File-mutating coder work stays serial.
- **Reviewer** — independent fresh-context review; must not edit production code.
- **Tester / Test Manager** — runs and records validation independently.

Use cmux/file-based handoff when possible, following the cmux skill exactly.
Agents must be spawned as separate **panes** in the caller workspace, not as new
workspaces and not as multiple tabs/surfaces in one pane. Required handoff
files/markers: `requirements.md`, `architecture.md`, implementation changes,
`review.md`, `test-report.md`, and
`.cmux-status/{ba,architect,coder,reviewer,test-manager}.done`. Reviewer and
Test Manager start after Coder completes and run in parallel. Any reviewer/tester
finding must be addressed or explicitly deferred before the next coding slice. If
cmux is unavailable, stop and ask the user before using a fallback; do not
silently skip this gate.

> **Architecture docs:** `docs/architecture.md` is the high-level project
> architecture/scaffolding map; `docs/adr/` contains durable architecture
> decisions; `docs/technical-spec.md` is the build/run/test technical
> specification. Read these before architecture or implementation changes.
> `docs/backlog.md` is the canonical backlog: Requirements, trackable user
> stories, and acceptance criteria. BA must update it for every slice. User
> stories use exactly: "As a [type of user] (WHO), I want [some goal] (WHAT), so
> that [some reason] (WHY)."
> `agent-output/<slice>/` is handoff evidence only, not canonical docs. When a
> slice introduces or changes durable architecture, update `docs/architecture.md`
> or an ADR in the same slice; reviewers must flag documentation drift.

`ds4.c` is a DeepSeek V4 Flash specific inference engine. It is not a generic
GGUF runner. The goal is a small, readable, high-performance C codebase with
Objective-C only where Metal requires it and Metal kernels under `metal/`.

## Goals

- Keep the production path as whole-model Metal graph inference.
- Always make sure that the SSD streaming, CUDA, distributed inference, Metal default inference are not affected by fixes to other parts of the code.
- Keep model loading mmap-backed for the Metal default case; do not eagerly copy the full GGUF. Keep the model loading for SSD streaming of routed experts explicit: allocated buffers, fast reads from disk, always try to hide loading of missing routed experts by loading them while performing the inference of the shared expert and routed experts already in RAM. Always try to hide loading of layers for prefill in SSD streaming mode using the inference time of the current layer as the next one is loaded.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV checkpoints.

## Quality Rules

- Always load the `self-improvement-skill` / auto-improvement skill at the start of work so reusable lessons are captured and applied.
- Always run the independent Reviewer in parallel for implementation/code changes. It must not be an inline self-review, and its findings must be addressed or explicitly deferred.
- Use TDD for code changes: write failing/red tests first, then implement the minimal green change, then refactor while keeping tests green.
- Keep code clean, decoupled, and low-complexity; use appropriate design patterns where they reduce coupling or clarify responsibilities.
- Keep the implementation small, sharp, easy to understand. Try to write elegant code in a state of grace. Don't settle for the first thing that comes to mind, try to find the most minimal and better working design. Don't introduce slop: very fragile code that just patches specific cases, dead code, useless code and code ways more complicated of how it should be.
- Comment important inference code where the model mechanics, cache lifetime, memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are fine when they validate the one release path.
- Do not introduce C++.
- In this checkout, many Python/docs files may be untracked, so `git diff` / `git diff -- <path>` can be vacuous as a scope guard. Still run `git diff --check` for whitespace, but verify production byte-intactness with direct checks (AST/source hashes, grep for new symbols in production/vendor files, and targeted reads) rather than citing a zero-line diff as proof.

## Tracking hygiene for test files participating in the baseline (HARD RULE)

This rule closed a real chain-of-custody break from Story 13.3b-3 (`d1f1488`), where Coder modified `tests/test_deepseek_v4_mlx_port.py` to update stale expectations so the suite would pass, but the file was untracked → the "569 passed / 0 RED" verdict was non-reproducible from a fresh clone, and Reviewer's "legitimate, not tautology" verdict was reached without a diff (impossible for an untracked file with no git before-state). Never repeat this:

- **If a test file participates in the pass/fail baseline the slice's verdict depends on, it MUST be tracked.** Before Coder declares done, run `git ls-files -- <test files the slice modified or cites in verdicts>`; any untracked file the slice touched MUST be `git add`ed and included in the slice's commit. Detect-and-comit is mandatory; Coder MUST NOT leave "local expectations updated" uncommitted in notes.
- **Verify Reviewer/Test-Manager verdicts against tracked files only.** Reviewer and Test Manager MUST run `git ls-files` for every test file cited in a legitimacy verdict BEFORE adjudicating (not just check pytest pass counts). If the file is untracked, the verdict is blocked: the diff-based legitimacy check is impossible and the verdict is declared `NEEDS-INFO` flagged for the supervisor — never `GREEN`.
- **When Coder touches an untracked test file, Coder commits it in the same slice.** Either `git add` + commit it (capturing the legit contract update, with a message explaining the change is a scoped contract update NOT a relaxation), OR escalate as a STOP if Coder believes the file should stay untracked (e.g. it's a local-only fixture). The supervisor must never re-discover an uncommitted baseline-modifying test file.
- **Reviewer verifies baseline reproducibility.** On any slice claiming "N passed / 0 RED," Reviewer must confirm that EVERY file contributing to that count is tracked (`git ls-files`) so a fresh clone reproduces the same verdict. If any contributing file is untracked, the "N passed" claim is downgraded to "N passed locally, reproducibility NOT verified" and the slice is returned for tracking.
- **NEVER let a modified untracked file slide via notes.** Coder notes saying "test file X is untracked; local expectations updated" are a RED flag, not an acceptable carry-over. The Coder MUST escalate the tracking question and resolve it (commit) before declaring done.

## cmux agent orchestration

Before any cmux operation, read `/Users/spotted/.pi/agent/skills/cmux/SKILL.md`
and follow it: anchor to the caller workspace, use prefixed refs, pass
`--focus false` where available, and do not suppress cmux stderr.

For multi-agent coding workflows in cmux, use this default flow:

```text
BA → Architect → Coder → Reviewer + Test Manager in parallel
```

Default pane layout when the user does not specify otherwise: keep the parent
assistant in the original top-left pane, spawn BA in a pane to its right, spawn
Architect in a pane to the right of BA, spawn Coder bottom-left, spawn Reviewer
bottom-middle, and spawn Test Manager bottom-right. If the user specifies a
layout, honor it while still using panes (not tabs/workspaces).

Use reusable project-local agent definitions under `.pi/agents/` (the project
`.pi/` directory is Pi's standard place for project-local Pi resources). Do not
create a fresh copy of BA/Architect/Coder/Reviewer/Tester prompts and launchers
for every slice. When a role is needed, reuse the existing spawned role agent in
its pane; if no pane/agent exists for that role, spawn it once and reuse it for
later slices/follow-ups. Before assigning a new task to a reused role pane, the
supervising parent/operator must send `/new` to that pane and wait for the fresh
session prompt; do **not** include instructions telling the role agent to run
`/new` itself, because that can erase the task context or lose the result before
completion. Supervision defaults to the normal inline/foreground completion check
unless the user explicitly asks for background monitoring. Keep reusable role
prompts/scripts in `.pi/agents/` and pass slice-specific arguments/output paths.
Per-slice directories under `agent-output/` are for handoff artifacts and logs
only.

**MANDATORY completion-check protocol (reliable supervision loop).** Never infer
a role agent's state from the status bar + file sizes alone — that signature is
ambiguous (a STOP/ESCALATE writes NO `.done` marker and NO downstream task file
by design, yet the agent is finished, not working). Every poll MUST run the
`cmux-agent-supervision` skill's probe — PREFERRED: `~/.pi/agent/skills/cmux-agent-supervision/scripts/supervisor-poll.sh <surface> <marker> <deliverable-glob> [start-epoch]` (mirrored at `.pi/agents/bin/supervisor-poll.sh`) — which wraps `check-role.sh <surface> <marker> [deliverable-glob]` (also at `.pi/agents/bin/check-role.sh`) and adds **plateau+deliverable-mtime fallback** for the documented glm-5.2 marker-flush hang pattern (role writes deliverable + emits in-pane JSON, then idles into summary view WITHOUT flushing the `.done` marker). The probe reads the PANE BODY + terminal `{"status":...}` JSON + marker together and prints one
unambiguous verdict. Role-agent supervision (spawn/rename/poll/recover) follows
the `cmux-agent-supervision` skill (`~/.pi/agent/skills/cmux-agent-supervision/SKILL.md`); read it before any supervision action that is not a plain `/new` + task dispatch to an existing healthy pane. Verdicts:

- `DONE_ERROR` (exit 3) — `{"status":"error"...}`: the agent ESCALATED/STOPped.
  READ the pane + its handoff doc; do NOT proceed as if it succeeded.
- `DONE_BLOCKED` (exit 4) — `{"status":"blocked"...}`: same, read it.
- `DONE_OK` (exit 0) — terminal `ok` JSON OR `.done` marker present.
- `WORKING` (exit 10) — spinner/`Auto-compacting` active; keep waiting (wins over
  all — never declare DONE mid-run; the echoed prompt's JSON example must not win).
- `IDLE_NO_SIGNAL` (exit 20) — no spinner, no terminal JSON, no marker: AMBIGUOUS,
  the supervisor MUST capture and read the full pane before deciding (this is the
  exact case that was mishandled on 11.50 — treat it as "unknown, go look", never
  as "still working"). **`supervisor-poll.sh` auto-resolves this case when the
  deliverable exists with `mtime ≥ dispatch epoch` AND token count is flat across
  `PLATEAU_POLLS` (default 3 = 60s) — it writes a clearly-marked
  `supervisor_fallback:true` `.done` marker and returns `VERDICT=DONE_OK_FALLBACK`.
  Always pass a start-epoch (4th arg, `$(date +%s)` at dispatch) so this fallback
  can fire. NEVER blind-poll more than 3 plateau cycles (60s) without this fallback firing — that
  IS the documented glm-5.2 hang.

A missing marker is NEVER sufficient evidence of "in progress"; only `WORKING`
(live spinner) means in progress. For slices armed to STOP+ESCALATE, expect and
check for `DONE_ERROR` on every poll. When an agent is `WORKING`, poll in a LOOP
inside one tool call and only return control on a TERMINAL verdict or explicit
max-wait ceiling — never end a turn with "I will poll" while an agent is mid-run.
**NEVER use `cmux respawn-pane` on live/role panes** (known bug destroys without
restarting); use `cmux new-split` + `rename-tab` + `launch-role.sh`, or the
skill's `spawn-role-panes.sh` helper, instead.

Use file-based handoff:

- `requirements.md` from BA.
- `architecture.md` from Architect.
- Implementation changes from Coder.
- `review.md` from Reviewer.
- `test-report.md` from Test Manager.

Use `.cmux-status/` only for status marker files (`*.done`). Do not put prompts,
logs, plans, notes, or backlog content there. Use marker files for reliable
completion instead of terminal text:

- `.cmux-status/ba.done`
- `.cmux-status/architect.done`
- `.cmux-status/coder.done`
- `.cmux-status/reviewer.done`
- `.cmux-status/test-manager.done`

Each agent creates its marker only when fully complete. Reviewer and Test Manager
start after Coder finishes and run in parallel. Reviewer should not edit
production code. In addition to marker files, each role agent must finish by
printing one terminal JSON object on its own line **in its own role pane only —
never forwarded to the parent pane or any other cmux surface** (the
parent/operator supervises completion independently via `.cmux-status/*.done`
markers + required handoff files, which are the canonical completion signals):
`{"status":"ok","role":"<Role>"}` on success, or
`{"status":"error","error":"<message>","role":"<Role>"}` on failure — where
`<Role>` is one of `BA`, `Architect`, `Coder`, `Reviewer`, `Test Manager`.
Do not put these JSON status payloads in `.cmux-status/`; that directory remains
marker-only.
By default, launch Pi agents in visible interactive mode inside each pane (prompt
as positional argument, no `-p`, no `tee` pipeline); use non-interactive `-p | tee`
only when the user explicitly asks for log/background runs. When verifying
completion, trust `.cmux-status/*.done` markers and required handoff files first;
pane text, stale cmux screen captures, and empty tee logs are secondary
diagnostics only.

## Safety

- Avoid large CPU inference runs on macOS; the CPU path has previously exposed kernel VM failures with very large mappings.
- Do not run multiple huge model processes concurrently. The instance lock is intentional.

## Layout

- `ds4.c`: model loading, tokenizer, CPU reference code, Metal graph scheduling,
  sessions, disk-cache payload serialization.
- `ds4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy.
- `ds4_metal.m`: Objective-C Metal runtime and kernel wrappers.
- `metal/*.metal`: compute kernels.
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments, and old planning material.

This list is not complete, check the files for more info.

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model and Metal are available. Use live server tests only when intentionally
testing the API surface.

At every major change where one of the following could be affected, make sure to:

1. Test the normal Metal path and that speed is still at the level it was.
2. Test the SSD streaming path.
3. Test the distributed inference if it could be affected, but ask the user before doing so.
4. Check if CUDA could be broken after the change, and ask the user to give you access to the CUDA machine to actually test if everything is still fine.

## Fine-tuning Python environments

The DS4 fine-tuning workflow uses isolated `venv` environments and declarative
`pyproject.toml` files. Do not install fine-tuning packages into the global
Python or into a single shared environment.

Environment definitions live under `python-envs/`:

- `python-envs/mlx/pyproject.toml` — MLX/MLX-LM stack for Apple Silicon smoke
  tests. Python >= 3.12.
- `python-envs/torch/pyproject.toml` — Torch/PEFT/TRL/Accelerate stack for
  local MPS or remote CUDA training. Python 3.12.

Each `pyproject.toml` declares the full dependency set. Create/update the
environments with `uv venv --seed` + `pip install -e` (`--seed` ensures `pip`
is available inside the uv-managed venv):

```bash
# MLX environment (default on Mac Studio M3 Ultra)
cd "$MLX_WORK"
uv venv --seed .venv
source .venv/bin/activate
pip install -U pip
pip install -e /Users/spotted/projects/ds4-finetuning/python-envs/mlx

# Torch/PEFT environment
uv venv --seed --python 3.12 --clear .venv-torch
source .venv-torch/bin/activate
pip install -U pip
pip install -e /Users/spotted/projects/ds4-finetuning/python-envs/torch
```

The helper `scripts/finetune_ds4.py` emits the same commands via `setup-env`
(`local-mlx`) and `torch-env-create` (`local-torch-mps`); `torch-env-create`
uses `--clear` to replace a stale/non-uv `.venv-torch`. It also unsets
`SSLKEYLOGFILE` before Python commands because the user's shell may point it at
`~/Documents/sslkeylog.log`, which can trigger macOS privacy `PermissionError`
when libraries create SSL contexts. Keep the `pyproject.toml` files in sync
with any new training dependencies; never encode package lists directly in the
helper commands.

Backend policy:

- Local Mac Studio M3 Ultra / MLX is the default.
- `local-torch-mps` is the first non-MLX local candidate.
- `remote-cuda` is an explicit operator choice, not a silent fallback.
- `cpu-check` runs only cheap validation, scans, and DS4 `--inspect` checks.
- `manual` prints backend descriptions.
