# ADR 0001: Whole-model Metal graph inference is the production path

Date: 2026-06-18
Status: Accepted

## Context

DS4 targets DeepSeek V4 Flash on Apple Silicon with a small, readable, high-performance runtime. The project also contains CPU reference/debug code, SSD streaming paths, CUDA/distributed considerations, and fine-tuning utilities. Without a clear production boundary, fixes in one backend can accidentally turn debug or fallback code into an unvalidated release path.

## Decision

The production inference path is whole-model Metal graph execution. Objective-C is used only where Metal requires it, and kernels live under `metal/`. CPU code remains reference/debug only. Backend-specific fixes must preserve isolation for default Metal, SSD streaming, CUDA, and distributed inference.

## Consequences

- Production correctness is validated on the Metal path first.
- CPU output can be used as an oracle/debug reference, but not as a silent fallback.
- CLI/server code should stay narrow and avoid depending on tensor internals.
- Changes that may affect SSD streaming, CUDA, distributed inference, or Metal default inference require explicit validation or a documented deferral.
