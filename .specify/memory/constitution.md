<!--
Sync Impact Report
Version change: template -> 1.0.0
Modified principles:
- Template Principle 1 -> I. Model-Specific Correctness
- Template Principle 2 -> II. Backend-Aware Portability
- Template Principle 3 -> III. Test-First Regression Gates
- Template Principle 4 -> IV. Performance and Memory Discipline
- Template Principle 5 -> V. API, Agent, and Trace Compatibility
Added sections:
- Additional Engineering Constraints
- Development Workflow & Quality Gates
Removed sections:
- None; placeholder comments were replaced with project policy.
Templates requiring updates:
- ✅ .specify/templates/plan-template.md
- ✅ .specify/templates/spec-template.md
- ✅ .specify/templates/tasks-template.md
- ✅ .specify/templates/commands/*.md (directory not present)
Runtime guidance reviewed:
- ✅ README.md
- ✅ AGENT.md
- ✅ AGENTS.md
Follow-up TODOs:
- None
-->
# DwarfStar Constitution

## Core Principles

### I. Model-Specific Correctness
DwarfStar is a narrow DeepSeek V4 Flash/PRO inference engine, not a generic GGUF
runner. Changes that affect model loading, tensor layout, tokenizer behavior,
prompt rendering, attention, KV cache format, quantization, or kernel math MUST
preserve the supported DwarfStar GGUF contract and MUST be validated against the
most relevant official vectors, logprob slices, long-context recall checks, or
quality scorer for the affected path.

Rationale: the project exists to make one supported model family feel finished
end-to-end; unexplained logits, attention, cache, or template drift invalidates
that goal even when the program still runs.

### II. Backend-Aware Portability
Metal on macOS is the primary production path. CUDA/DGX Spark, ROCm/Strix Halo,
SSD streaming, distributed inference, disk KV cache, and CPU diagnostics are
separate compatibility surfaces. Any change touching shared inference, loading,
scheduling, cache, API, or build code MUST name the affected surfaces and verify
the applicable builds or tests. CPU remains a reference/debug path only and MUST
NOT become a production fallback without a separate constitutional amendment.

Rationale: the same model behavior depends on backend-specific memory layouts,
kernels, and hardware limits; silent backend regressions are release blockers.

### III. Test-First Regression Gates
Every planned feature or fix MUST identify the failure mode it can realistically
affect before implementation. If behavior changes, the plan MUST include either
a failing test, an official-vector comparison, a live model validation command,
or a documented manual check that proves the old failure and the new behavior.
Implementation tasks MUST keep tests or validation before dependent code changes
whenever the check can be automated.

Rationale: large-model inference failures are often subtle and expensive to
rediscover; regression evidence must be captured at the boundary that can fail.

### IV. Performance and Memory Discipline
Correctness takes priority over speed, but avoidable speed, memory, disk, or
latency regressions are not acceptable. Performance-sensitive changes MUST record
the hardware, backend, GGUF/quant, context size, command, and before/after result.
A slower path MAY be accepted only when it fixes an important correctness or
safety issue and the trade-off is documented. Code MUST respect resource safety:
do not run multiple huge model processes concurrently, do not hide unbounded disk
or RAM growth, and do not run large CPU inference on macOS.

Rationale: DwarfStar is useful only when high-end local inference is both correct
and practical on the target machines.

### V. API, Agent, and Trace Compatibility
The CLI, OpenAI-compatible chat and responses APIs, Anthropic-compatible
messages API, streaming, tool-call parsing, agent session management, disk KV
reuse, and traces are first-class product surfaces. Changes to these surfaces
MUST preserve documented compatibility or state the breaking change explicitly,
provide a migration path, and update docs/templates in the same change. Traces
MUST be useful for debugging prompt rendering, cache decisions, and parser events
without leaking unrelated state.

Rationale: the inference engine is intended to serve coding agents and long local
sessions; API and trace regressions break the main user workflow even when token
generation remains correct.

## Additional Engineering Constraints

- The codebase MUST remain small, self-contained C with Objective-C only where
  Metal integration requires it, Metal kernels under `metal/`, and no C++.
- Public APIs MUST stay narrow; CLI and server code MUST NOT depend on tensor
  internals beyond documented boundaries.
- Permanent semantic variants behind flags are prohibited. Diagnostic switches
  are allowed when they validate or explain the one intended release path.
- Supported GGUF files MUST be explicit. Arbitrary GGUF compatibility is out of
  scope unless this constitution is amended.
- Inference code comments MUST explain non-obvious model mechanics, shape/order
  dependencies, cache lifetimes, memory policies, or API orchestration.
- Documentation MUST preserve acknowledgements and license obligations for
  llama.cpp/GGML-derived knowledge or source-level adaptations.

## Development Workflow & Quality Gates

- Plans MUST include a Constitution Check listing affected model contracts,
  backends, API/agent surfaces, performance baselines, and resource risks.
- Specifications MUST express testable user stories plus measurable success
  criteria for correctness, compatibility, performance, and safety when relevant.
- Tasks MUST be organized so each user story or backend slice is independently
  testable, with validation tasks before implementation tasks where possible.
- Pull requests or commits MUST list commands run, machine/backend, model quant,
  context size when relevant, and any skipped checks with a reason.
- Release sign-off MUST follow `QA_BEFORE_RELEASES.md`; skipped release gates
  require written rationale.
- `AGENT.md` is the operational runtime guidance file for agents and MUST remain
  consistent with this constitution.

## Governance

This constitution supersedes conflicting local practices for DwarfStar planning,
implementation, testing, and release decisions. Amendments require a written
change to this file, a Sync Impact Report, review of dependent Spec Kit
templates and runtime guidance, and documentation of any migration or skipped
validation work.

Versioning follows semantic versioning:
- MAJOR for removing or redefining a principle in a backward-incompatible way.
- MINOR for adding a principle, adding a required section, or materially
  expanding governance or validation obligations.
- PATCH for clarifications, wording fixes, or non-semantic refinements.

Compliance is reviewed during planning, before implementation tasks are accepted,
and before releases. Any violation MUST be recorded in the plan's Complexity
Tracking section with the reason, the simpler alternative considered, and the
validation evidence required before merge or release.

**Version**: 1.0.0 | **Ratified**: 2026-06-16 | **Last Amended**: 2026-06-16
