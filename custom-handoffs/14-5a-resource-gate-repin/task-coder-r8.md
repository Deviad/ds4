# Story 14.5a — Coder r8 isolated consumer-oracle closure

Read latest r7 blocked review. Test-only TDD unless a real defect appears.

1. Parameterize A2 and B2 checkpoint/config keys across each applicable consumer separately: canonical validation, success publication/no-write, dependency admission, executable catalog.
2. For isolated dependency tests, recompute outer namespace/digest/report/marker bindings so generic namespace guard passes; mutate named consumer binding and assert specific failure message/boundary.
3. Publication tests assert mutation writes zero phase/final reports/markers; positive B2 asserts phase report/OK + final report/OK and exact marker-report hash bindings, including final-ok.
4. Executable catalog per-key tests prove each binding affects intended launch/training arg/use, not unused assignment presence.
5. Track r8 notes/tests; no markers staged. Canonical 470+ suite, targeted matrices, pycompile/diff/protected. Write coder-notes-r8.md and marker. No delegate/real execution/cleanup/commit/push.