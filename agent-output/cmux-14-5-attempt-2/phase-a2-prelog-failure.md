# Story 14.5a Phase A2 pre-log failure

- Commit: `c910d1ba33912236ee87f3f9bdfb5b31edece6e7`
- Command artifact SHA-256: `38e6b1aa11d28c41c126fda6dab55cf68a9eb1d9a71fc177eec8875a3114b2de`
- Visible panel: `surface:180` (closed after capture)
- Exit: `1`
- Terminal error: `pilot launch check failed closed: MLX version mismatch: None`
- Training calls/updates: `0`
- Adapter output: absent
- Phase A2 log/report/OK/fail paths: absent
- Attempt-2 final OK/fail paths: absent
- Retry: not performed

Read-only diagnosis after failure:

- `mlx.__version__` is `None` in the canonical interpreter.
- `importlib.metadata.version("mlx")` is `0.31.2`.
- `importlib.metadata.version("mlx-lm")` is `0.31.3`.
- `scripts/ds4_segmented_pilot.py::_runtime_preflight()` currently validates `getattr(mlx, "__version__", None)` against `0.31.2`.

The one-attempt authorization is consumed. Any repair and retry requires a fresh reviewed authorization. No attempt-2 namespace cleanup or overwrite occurred.
