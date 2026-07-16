# ADR 0018: Import-purity assertions for readiness-path tests use `sys.modules` snapshot-diff, not absolute membership (the mlx-venv editable install preloads torch/mlx at site-init)

Date: 2026-06-20
Status: Accepted
Related to: ADR 0010 (B1 hc_mult>1 multi-layer readiness diagnostic), ADR 0003 (isolated fine-tuning environments), ADR 0005 (agent-output is not canonical)

## Context

The DS4 fine-tuning project ships an editable `ds4_ft_mlx` install inside the
`python-envs/mlx` venv. Its `.pth`
(`lib/python3.13/site-packages/__editable__.ds4_ft_mlx-0.1.0.pth`) imports
`ds4_ft_mlx.mlx_lm_plugin` at interpreter site-init, which transitively pulls
`mlx` and `torch` into `sys.modules` BEFORE any test code runs.

Story 11.36 added
`tests/test_finetune_ds4_b1_hc_mult_multilayer_readiness.py::B1HcMultMultilayerReadinessTests::test_no_mlx_or_torch_imports_required`,
asserting `assertNotIn("mlx"/"torch", sys.modules)` after exercising the B1
readiness path. That assertion checks an **environment** property the mlx venv
violates by design, so the test has been latent-failing (surfaced as Story 11.44).

The production readiness path itself is import-clean. An `ast` walk of
`_b1_hc_mult_multilayer_readiness` and `build_forward_parity_readiness_report`
in `scripts/finetune_ds4.py` shows **0** `Import`/`ImportFrom` nodes and **0**
`torch`/`mlx` name references in either function body; under `python -S`
(site-init suppressed), importing `scripts.finetune_ds4` and calling both
functions leaves `torch*=0 mlx*=0`. The test's INTENT ("the readiness path does
not NEED to import mlx/torch — it is computable pure-Python/stdlib-only") is
already honored by the production code; only the assertion MECHANISM is broken.

## Decision

Import-purity assertions for the readiness path — and any peer "readiness path
does not import ML-stack module X" assertion in this repo's mlx-venv test suite
— MUST use a **`sys.modules` snapshot-diff**: take a `before` snapshot of the
relevant `torch`/`torch.*`/`mlx`/`mlx.*` keys in `sys.modules`, exercise the
path, then assert the `after - before` set is empty. They MUST NOT use absolute
`assertNotIn("torch"/"mlx", sys.modules)`, because the absolute form couples the
test to the venv's editable-install site-init rather than to the path's own
behavior, producing latent false-failures on every venv that preloads the named
modules.

The predicate matches the test's **NAMED** invariant (`torch`/`mlx` and their
submodules) — it is NOT broadened to `mlx_lm.*`/`transformers.*`. Broader
runtime coverage of "any ML-stack dependency" is already provided structurally
by the production code's 0-Import AST invariant (verifiable by `ast` walk), not
by the runtime predicate; broadening the predicate would be scope creep against
the test's named contract.

## Consequences

- The fixed assertion is **env-independent**: it passes in the mlx venv (where
  torch/mlx are preloaded) and under `python -S` (where they are absent),
  because it measures only what the readiness path itself adds. (Proven:
  snapshot-diff `added == set()` under both regimes — the readiness path adds 0
  torch*/mlx* keys either way.)
- The weakening vs an absolute check — it would not catch a bare `import torch`
  when torch is fully preloaded — is immaterial here because (a) the production
  path is AST-proven import-free, and (b) the realistic regression (a contributor
  pulling a not-yet-preloaded submodule like `import mlx.core` or
  `import torch.distributed`) IS caught, since that adds a new `sys.modules`
  key.
- Strict "the path never even calls `import torch`" enforcement, if ever wanted,
  is a **separate follow-up slice** (a `builtins.__import__` import-guard), not
  this housekeeping slice. That approach is deliberately NOT chosen here: monkey-
  patching `builtins.__import__` is slop-prone under pytest's importer,
  `coverage`/`pytest-cov`, and `importlib` lazy loaders, and the 0-Import AST
  invariant already statically guarantees import-purity — so β's marginal
  precision guards a regression code-review's AST check also catches, for ~2×
  the code and materially higher fragility. AGENTS.md forbids such slop.
- **This ADR governs test ASSERTION MECHANISM only.** It is NOT a license to
  import torch/mlx into the readiness path's production code; the production path
  stays import-clean (enforced by both the AST check and this runtime check).

## Proof (Story 11.44 Architect independent corroboration)

Diagnostic `/tmp/arch-11-44-diag.py` run under both default and `-S`
(`python-envs/mlx/.venv/bin/python3` and `... -S`):

| mode                      | torch* keys after `import finetune_ds4` | mlx* keys | snapshot-diff added by readiness path |
|---------------------------|-----------------------------------------|-----------|---------------------------------------|
| default (site-init on)    | 1050                                    | 32        | `[]` (empty => PASS)                  |
| `-S` (site-init off)      | 0                                       | 0         | `[]` (empty => PASS)                  |

AST purity (independent 3rd corroboration of BA §1.3):
`_b1_hc_mult_multilayer_readiness` and `build_forward_parity_readiness_report`
each show `import_nodes=0 torch_mlx_refs=0` (verified under both modes).

Red reproduction:
`pytest tests/test_finetune_ds4_b1_hc_mult_multilayer_readiness.py -q` fails at
L270 (`assertNotIn("mlx", sys.modules)`); `1 failed, 7 passed`.

## Decision scope: no `docs/architecture.md` edit

This ADR is the durable record; `docs/architecture.md`'s testing-principle line
("Test fixtures should be tiny, deterministic, and shaped to exercise real
semantics without loading large model payloads") is unchanged. The decision is
narrow enough (a test-assertion mechanism for the readiness path) that a
dedicated numbered ADR is the proportional home — discoverable via `docs/adr/`
adjacent to ADR 0010, without bolting a testing sub-section onto
architecture.md's "Data and marker policy" area (which is about markers/data,
not test assertion mechanisms). If a future slice adds a dedicated
`docs/architecture.md` "## Testing" section, it should cross-reference this ADR.

## Amendment (Story 11.45, 2026-06-20): family scope + `csa_topk` static-source corollary

Status note: this Amendment extends — does NOT retract — the Story 11.44
Decision above. The 11.44 record (Context, Decision, Consequences, Proof,
Decision scope) stays byte-intact; the Amendment records the family-scope
broadening that 11.45 EXECUTES + the one genuinely new architectural content of
11.45 (the static-source corollary for siblings whose test exercises NO
path-under-test).

### Family scope (now explicit)

The Story 11.44 Decision already governed "any peer 'readiness path does not
import ML-stack module X' assertion in this repo's mlx-venv test suite" — the
family scope was semi-implicit. Story 11.45 makes it explicit + executes it
across the env-broken import-purity test family in 6 files:

- **4 in-scope env-broken siblings (11.45 fixes):**
  - `tests/test_finetune_ds4_b2_real_checkpoint_payload_readiness.py::B2RealCheckpointPayloadReadinessTests::test_builder_does_not_import_mlx_or_torch`
  - `tests/test_finetune_ds4_stateful_decode_readiness.py::StatefulDecodeReadinessTests::test_no_mlx_or_torch_imports_required`
  - `tests/test_finetune_ds4_i8_dequant_integration.py::I8DequantIntegrationProofTests::test_no_mlx_or_torch_imports_required`
  - `tests/test_finetune_ds4_csa_topk_primitive.py::CsaTopkPrimitiveProofTests::test_no_mlx_or_torch_imports_required_for_unit_tests`
- **2 already-passing snapshot-diff siblings (scope-EXCLUDED — already
  env-independent; touching green tests is scope creep):**
  - `tests/test_finetune_ds4_forward_parity_readiness.py::ForwardParityReadinessTests::test_no_mlx_or_torch_imports_required_for_unit_tests`
  - `tests/test_finetune_ds4_b2_routed_dequant_trusted_reference_readiness.py::B2RoutedDequantTrustedReferenceReadinessTests::test_builder_does_not_import_mlx_or_torch`

The 11.44 reference at `tests/test_finetune_ds4_b1_hc_mult_multilayer_readiness.py`
L256-283 remains the canonical call-bracketing form (byte-intact — hard
constraint). 11.45 #1/#2/#3 inline the SAME predicate per-file (`_is_torch_or_mlx`;)
NOT a shared `tests/_import_predicate.py` helper — see the 11.45
`agent-output/cmux-11-45/architecture.md` §2 Q1 for the DRY-strategy reasoning
and the ~8-file threshold for revisiting).

The predicate stays EXACTLY as Story 11.44 froze it: `name == "torch" or
name.startswith("torch.") or name == "mlx" or name.startswith("mlx.")`. This
ALREADY catches `mlx.core` + `mlx.core.*` via `startswith("mlx.")` (11.45
Architect `-S` probe: `'mlx.core'.startswith("mlx.") = True`). No predicate
broadening to `mlx_lm.*` / `transformers.*` — those are out of the tests' NAMED
`torch`/`mlx` invariant; broader structural coverage of "imports NO ML-stack at
all" is already provided by the production code's 0-Import AST invariant
(verifiable by `ast` walk), not by the runtime predicate.

### Corollary: the `csa_topk` static-source variant (genuine new content of 11.45)

The Story 11.44 call-bracketing template assumes a path-under-test call
(`before` snapshot → call → `added = {after} - before` → `assertFalse(added)`).
A sibling that asserts import-purity but exercises NO path-under-test (a pure
runtime env-assertion) MUST NOT use the call-bracketing template literally — it
would be a TAUTOLOGY: `added` is always empty (nothing ran between `before` and
`after`) → `assertFalse(added)` always passes → dead code = slop (forbidden by
AGENTS.md).

For such siblings, assert import-purity STATICALLY: parse the test module's own
source via `inspect.getsource(sys.modules[__name__])` and assert the module's
own top-level imports pull neither `torch` nor `mlx`/`mlx.core`:

```python
import inspect
source = inspect.getsource(sys.modules[__name__])
self.assertNotIn("import torch", source)
self.assertNotIn("import mlx.core", source)
self.assertNotIn("import mlx" + "\n", source)  # bare `import mlx`, NOT mlx.core / mlx_lm
self.assertNotIn("from torch", source)
self.assertNotIn("from mlx", source)
```

This is the env-independent restatement of the test's named intent ("the
primitive unit tests don't NEED `torch`/`mlx.core`") when there is no
path-under-test to snapshot-diff around. It is honest because the test module's
own top-level imports ARE already torch/mlx/mlx.core-free (Story 11.45 Architect
AST + source-substring probe verified all 5 targets absent). It does NOT weaken
the snapshot-diff invariant for #1/#2/#3 — those siblings ESTABLISH the boundary
(by exercising the path); the static-source variant asserts the boundary for a
sibling that established it elsewhere (at module-load time).

The 5-target substring set is FROZEN for `csa_topk`. If a future sibling hits
this corollary, validate its source programmatically first (substring presence
can collide with unrelated `.py` substrings like `from mlx_utils import ...`
for `from mlx`); if a false-positive substring collision appears, STOP and
escalate — do NOT silently broaden the predicate into a regex/AST-node form to
dodge the collision (that drifts the FROZEN shape and risks masking a genuine
import-purity breach).

This is the SAME governing PRINCIPLE as Story 11.44 ("test the import-purity
invariant env-independently; absolute `assertNotIn(…, sys.modules)` env-broken
in the mlx venv"), with a different assertion MECHANISM (static source-substring
vs runtime call-bracketing) chosen precisely when literal call-bracketing is a
tautology. That single linked decision is why this is an AMENDMENT to ADR 0018
(not a new ADR 0019): splitting it would cargo-cult the record across two files.

### Story 11.45 proof (Architect independent 3rd corroboration)

Diagnostic `/tmp/arch-11-45-diag.py` run under both default and `-S`:

| sibling (#)                      | AST: Import nodes / torch-mlx refs (body) | snapshot-diff `added` (= empty ⇒ green) |
|----------------------------------|------------------------------------------|----------------------------------------|
| #1 `b2_real_checkpoint_payload_readiness` | 0 / 0                                    | `set()` ⇒ PASS (call adds 0 torch*/mlx*) |
| #2 `stateful_decode_readiness`    | 0 / 0                                    | `set()` ⇒ PASS (call adds 0 torch*/mlx*) |
| #3 `_real_mode_proofs_report`     | 0 / 0                                    | `set()` ⇒ PASS (call adds 0 torch*/mlx*) |
| #4 csa_topk (NO path-under-test)  | n/a (static-source corollary)            | source: all 5 substrings ABSENT ⇒ PASS |

Plus `build_forward_parity_readiness_report` (shared by #2 + the 2 already-passing
siblings): `0 / 0`. `scripts.finetune_ds4` module-level import-clean under `-S`;
1082 torch*/mlx* keys preloaded at site-init under default.

Red reproduction: `pytest -v <6 node-ids>` = `4 failed, 2 passed` (the 4 =
env-broken `assertNotIn(…) , sys.modules)`; the 2 = the already-passing
snapshot-diff siblings). Each failure's traceback terminates AT the
`assertNotIn` line (NOT at the path-under-test call) — purely env-broken, NO
second failure mode.

### Amendment scope: still NO `docs/architecture.md` edit; still NO β-import-guard

Same as Story 11.44's "Decision scope": the amendment stays scoped to
`docs/adr/`; `docs/architecture.md` has no "## Testing" section to bolt onto,
and the family-extension + corollary fit naturally in ADR 0018 as a dated
amendment. β (`builtins.__import__` monkey-patch import-guard) STAYS rejected
for the same reason 11.44 recorded: monkey-patching is slop-prone under pytest's
importer + `coverage`/`pytest-cov` + `importlib` lazy loaders, and the 0-Import
AST invariant already structurally guarantees import-purity — so β's marginal
precision guards a regression code-review's AST check ALSO catches, for ~2× the
code and materially higher fragility.

### Story 11.46+ scope boundary (NOT this Amendment's scope)

Across the SAME 6 sibling files, **11** non-env-broken failures exist
(readiness-state drift: fixture changes, real-mode-proof spec drift,
landscape/verdict-count drift, schema/honesty fail-closed assertions). Those
fail at production-readiness assertions INSIDE the test bodies (NOT at an
`assertNotIn(…, sys.modules)` line) — a separate triage problem with a separate
root cause (NOT site-init). They are EXPLICITLY DEFERRED to Story 11.46+; they
are NOT conjoined into 11.45's ADR 0018 application. See 11.45
`agent-output/cmux-11-45/architecture.md` §6 for the per-method enumeration.
