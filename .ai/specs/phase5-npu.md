# Phase 5: Custom NPU Instructions

## Goal
All NPU instructions execute correctly. C firmware using npu.h intrinsics produces correct results.

## What to build
- NPU state class: acc_lo, acc_hi (uint32), vreg[4] (each 4×int8)
- Decode: opcode 0x0B → dispatch to NPU by funct3/funct7
- Implement all instructions from ISA reference NPU section
- GELU lookup table: precompute for int8 input range, return int8 output
- firmware/common/npu.h (copy from bootstrap doc)
- firmware/npu_test/: C program exercising each instruction, printing PASS/FAIL

## Module split
- npu/instructions.py + npu/engine.py: custom opcode 0x0B execution logic, NPU state, accumulator, GELU table. This is where instruction semantics live.
- devices/npu.py: memory-mapped control/status registers at 0x20000000 (for TUI status reads, DMA-style transfers in later phases). Thin wrapper, delegates to npu/engine.
These are separate concerns. Instruction execution goes in npu/, not devices/.

## Testing
- Unit: each NPU instruction, known inputs → expected output
- MACC chain: accumulate N multiply-adds, compare to Python sum
- QMUL: verify (a*b)>>8 with overflow cases
- CLAMP: values inside range, above 127, below -128
- GELU: compare table output to reference values computed with math.erf from Python standard library
- Integration: npu_test.elf runs, prints all PASS

## Acceptance
uv run pytest tests/npu/ -v — all pass
uv run python -m riscv_npu run firmware/npu_test/npu_test.elf — all PASS
