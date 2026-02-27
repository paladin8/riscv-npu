# Phase 7: Transformer Extension

## Goal
Tiny transformer runs on emulator.

## Before implementing
Evaluate NPU instruction set gaps for transformer workload. Document in .ai/memory.md:
- Is softmax needed as a custom instruction?
- Is layer norm needed?
- Can attention be expressed with existing MACC + GELU?

Propose new instructions if needed. Wait for human approval before adding to ISA.

## Target model
- Embedding dim: 64, heads: 4, layers: 2
- Byte-level vocab (256), context: 32 tokens
- ~200K params, ~200KB int8
- Task: simple pattern (copy, reverse, or character-level LM)

## This phase is exploratory
Expect spec revision after Phase 6 findings. Keep .ai/memory.md updated with what works and what doesn't.
