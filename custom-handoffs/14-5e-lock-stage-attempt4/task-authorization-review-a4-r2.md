# External Phase A4 authorization review r2 — exactly one invocation

Prior artifact denied without namespace consumption. Review regenerated artifacts at unchanged final revision `07fe6c8e13651892e05014b40e78210f0d306462`:
- `/tmp/ds4-phase-a4-authorization.json` SHA256 `6ac22cca3e8a795658f03436bfd9d14a9ba70dc75ceb05c488a1f538afbe54ea`
- `agent-output/cmux-14-5-attempt-4/run-phase-a4.sh` SHA256 `2d9d4e50827eeae759fc6843b9e75a57dd6ffe3d4bada1261dab520352a2e69d`
- canonical command SHA `c4d22b662b6e2cbd585db383675bda39da0c974b5d396e5ced57de0bdffae829`
- trusted immutable identity SHA `3f23e879163cbb99621547061de69befbc42bb25f110dad5514ee52cfb370c93`, generated from fresh production `_runtime_preflight`; repository status `[]`.

Independently rerun fresh production preflight and validate exact identity projection/auth schema/revision/source/protected/history/phase/prewrite; run mutation-free production launch-check-only and verify no log/output/lock/markers. Verify wrapper byte-exact/pins/one-attempt semantics, all A4 paths/lock absent, A3 immutable. DO NOT run wrapper/training. PASS authorizes exactly one unchanged A4 wrapper invocation only; no retry/B4/fallback/cleanup/modified artifact. Write `authorization-review-a4-r2.md`, marker/status. No edits.