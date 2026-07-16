# Story 14.3b — Architect timeout repin

After BA requirements, read them plus docs/architecture.md, docs/technical-spec.md, relevant ADRs, current smoke script/catalog, filtered repin r2 PASS/GREEN handoffs, and timeout failure evidence.

Design minimal implementation: change only the canonical hard timeout constant and corresponding durable requirements/docs/catalog/report expectations from 600s to 1200s. Preserve watchdog installation, abort code, no retry/fallback, lock release, fail marker/report and all other smoke identity. Define tracked synthetic tests proving 1200 constant/report and unchanged 600s lock timeout/other parameters. No real asset reads/smoke and no code edits. Write architecture.md and marker.