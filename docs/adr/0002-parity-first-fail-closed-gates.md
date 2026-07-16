# ADR 0002: Numerical model features are parity-first and fail-closed

Date: 2026-06-18
Status: Accepted

## Context

DeepSeek V4 Flash features such as compressed sparse attention, routed MoE experts, packed FP4/I8 expert payloads, hyperconnections, and KV cache/generation can produce plausible but wrong logits if partially implemented. Silent approximation is more dangerous than an explicit blocker.

## Decision

New model mechanics must be proven by deterministic tiny fixtures and/or trusted references before the corresponding runtime gate is relaxed. Unsupported shapes, ratios, cache states, packing formats, real-scale variants, or missing weights must fail closed with specific errors. Full-forward marker files are written only after their documented parity gates are actually satisfied.

## Consequences

- Tests are added red-first for every gate relaxation.
- Tiny proven subsets may be accepted while real-scale paths remain blocked.
- Marker files such as `.deepseek-v4-forward-parity-ok` are evidence, not aspirations.
- It is acceptable for a correct slice to add explicit `NotImplementedError` guards when a path is not yet proven.
