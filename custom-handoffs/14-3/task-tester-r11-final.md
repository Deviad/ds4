# Story 14.3 Test Manager r11 final

Validate registered `14-3-coder-r11`, functional hash `8911e91dea22a347e416c48057039d266c79f458332d6f96e191cfa01ce7d1c0`.

Use exact tracked-path Path A calculation: 365 rows and digest `7241924d6f9be2eb0df974fdcb1717728c71b6ee77d41dea3fe8a8af55ebb2af`; verify pyc, `.gitmodules`, vendor gitlink/inner HEAD, protected hashes, ADRs, registry, and four-file hash. Run with exact fork `PYTHONPATH="$PWD/vendor/mlx-lm:$PWD/python-envs/mlx/src:$PWD"`: focused smoke, finetune, source, provider suites, py_compile, direct r11 regressions. Do not use faulty index-blob digest or full staged hash for functional identity.

No real assets, Phase 2, smoke/training/inference/backend/install/network, commit/push. Write only `custom-handoffs/14-3/test-report.md` and marker. GREEN only reproducible evidence; otherwise NO-GO.