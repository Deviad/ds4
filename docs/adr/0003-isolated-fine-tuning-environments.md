# ADR 0003: Fine-tuning Python environments are isolated and declarative

Date: 2026-06-18
Status: Accepted

## Context

The DS4 fine-tuning workflow uses different Python stacks for MLX on Apple Silicon and Torch/PEFT/TRL/Accelerate for local MPS or remote CUDA. Installing everything into a global Python or one shared virtualenv causes dependency drift and makes parity failures hard to reproduce.

## Decision

Fine-tuning dependencies are declared under `python-envs/` and installed into isolated uv-managed environments:

- `python-envs/mlx/pyproject.toml` for MLX/MLX-LM work.
- `python-envs/torch/pyproject.toml` for Torch/PEFT/TRL/Accelerate work.

Helper scripts may emit setup commands, but package lists belong in the environment definitions.

## Consequences

- Local MLX is the default backend on Apple Silicon.
- Torch/MPS and remote CUDA are explicit operator choices, not silent fallbacks.
- Tests can name the environment they require and skip cleanly when it is unavailable.
- New dependencies must update the relevant `pyproject.toml`, not ad hoc install commands.
