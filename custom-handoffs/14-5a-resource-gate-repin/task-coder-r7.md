# Story 14.5a — Coder r7 executable wrapper closure

Read latest r6 blocked review. Direct TDD only.

1. Rebuild attempt2 catalog wrapper safely: construct complete inner script as one string, then `shlex.quote` it exactly once for `bash -lc`. No nested outer single quote conflicts; printf/path quoting safe.
2. Temp-only executable tests for emitted A2 and B2 commands with spaces: assert top-level parse exactly 3 argv (`bash`,`-lc`,one script), launch-check and training stubs both execute in order, FD-attested log behavior works, child nonzero status propagates, no false-success.
3. Complete every-key/every-applicable-consumer checkpoint/config matrix for A2/B2 artifact validation, dependency admission, canonical report/publication, phase specs, and executable catalog. Add fully valid B2 canonical report/publication positive case then each binding mutation.
4. Track r7 notes/tests; docs pending; no markers staged. Run targeted + canonical 427+ suite, pycompile, executable wrappers, protected/diff. Write coder-notes-r7.md and marker. No delegate/real training/cleanup/commit/push.