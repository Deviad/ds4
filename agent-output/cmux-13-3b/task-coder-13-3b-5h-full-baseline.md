# Story 13.3b-5h — Coder full tracked-baseline closure

Use `openai-codex/gpt-5.5` high. No implementation/report/helper/test edits, real assets, classification, redesign, smoke, or commit.

The existing `coder-13-3b-5h-tracked-baseline.log` ran only two tests. This is not the required full baseline.

Run exactly from canonical project venv:
```bash
python-envs/mlx/.venv/bin/python -m pytest -q $(git ls-files 'tests/test_*.py')
```
Record tracked file count and complete result in a new staged `coder-13-3b-5h-full-tracked-baseline.log`. Do not change test expectations. Also rerun `git diff --check`, `git diff --cached --check`, and verify every contributing test file tracked.

Update `coder-13-3b-5h-notes.md` with full baseline evidence. Rewrite marker/success JSON only if full baseline GREEN; otherwise STOP/error JSON.