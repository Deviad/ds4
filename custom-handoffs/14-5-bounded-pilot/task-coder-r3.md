# Story 14.5 — Coder r3 executable-path closure

Read latest blocked review fully. Fix all remaining findings directly via TDD; no delegation/real assets/commit/push.

Required:
1. `_StepObservingProvider` accept real provider `((loss, token_count), gradients)` only; fakes use exact nested shape.
2. Parser namespace exactly supports pinned `load_dataset`: `test`, `hf_dataset` and all accessed fields.
3. Flatten nested trainable tree with pinned `tree_flatten` semantics before every safetensors save; tests use real MLX save probe.
4. Apply `numpy.random.seed(0)` and `mx.random.seed(0)` in each phase before model/training; prove call order.
5. Split immutable common identity from dynamic resource observations. Before Phase B model load/training, recompute/compare exact immutable identity to Phase A. Bind full identity/assets/topology/runtime/command into contract digest. Dynamic resources excluded from equality.
6. Observer validate scalar MLX float32 loss and int32 token before conversion; exact nonempty model trainable schema paths/shapes/dtypes; MLX isfinite; exact mask token count including tensor length. Reject empty/extra/drift.
7. Gate MLX 0.31.2, resolved mlx_lm path, outer gitlink/inner SHA+clean, provenance split hashes, schema invariance, exact checkpoint cardinality, validation callback order/count. Reports include commands and lock lifecycle. Markers include output/timestamp/exit/contract digest.
8. Add terminal mutation tests: real provider shape, real nested save, parser namespace, seeds, Phase B pretraining identity, every injected failure, exactly-once lock release, watchdog cancel, marker/report write failures/order, validation cardinality, full catalog equality/default byte guards.
9. Docs state blocked/implemented truthfully. Track r3 handoff; no markers staged.
10. Run focused + exact suite/probes/protected gates. Write coder-notes-r3.md and marker.