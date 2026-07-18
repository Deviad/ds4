# Story 14.5c — Coder r2 review closure

Read latest `custom-handoffs/standby/review.md` and architecture §§5-10. Direct RED-first only.

Fix all blockers:
1. Implement exact authorization JSON schema/CLI/catalog wrapper binding reviewed revision, canonical A3/B3 command SHA, pilot/catalog/protected/provider/vendor/model/data/config identity; validate before any write. Add coordinated substitution/no-write tests.
2. Fix attempt2 historical verifier to canonical committed manifest nested lock schema, exact path semantics, 10 entries, absence facts. Test every semantic/hash/size/path/target/outer/coordinated mutation.
3. Contract/report lineage uses attempt 3, binds both attempt1 and attempt2 historical objects through reports/markers/dependency/final. Remove stale A2 names/keys where live.
4. Add real mx.save/load null test, artifact matrix per checkpoint, tensor mutation, valid-null publication, invalid no-success/rollback, B3 exact-A3 dependency tests.
5. Stage BA backlog; make architecture/technical spec truthful only after implementation. Keep protected bytes.

No real attempt artifacts/model/dataset/training/cleanup/commit/push/delegate/cmux. Stage exact revision, write coder-notes-r2.md + marker.