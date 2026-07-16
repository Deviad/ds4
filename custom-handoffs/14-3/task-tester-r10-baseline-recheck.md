# Story 14.3 Test Manager r10 baseline recheck

Recheck r10 after supervisor restored protected index state. Registered functional hash remains `d9b4f9e60ae4158423f41dd09de62611355427525c022973faefef077ae2e584`.

Verify current index first: Path A exactly 365 rows and digest `7241924d6f9be2eb0df974fdcb1717728c71b6ee77d41dea3fe8a8af55ebb2af`; `.pyc` exact; vendor gitlink `15b522f593b7ca5fbc0cac6f7572d40859d2d8fe`; inner HEAD/pins/source hashes. Then rerun focused smoke, finetune, exact-fork protected provider/source suites, py_compile and functional hash/registry.

No real assets, Phase 2, smoke/training/inference/backend/install/network, commit/push. Write only `custom-handoffs/14-3/test-report.md` and marker. GREEN only reproducible evidence; otherwise NO-GO.