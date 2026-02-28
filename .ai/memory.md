# Project State

## Current phase
Phase 1 — COMPLETE. All 11 deliverables implemented. 145 tests passing.

## Decisions
- Instruction is a frozen dataclass with opcode, rd, rs1, rs2, imm, funct3, funct7
- RegisterFile is a class (not dataclass) wrapping list[int], x0 hardwired to 0
- RAM has base address + bytearray, bounds-checked, little-endian via int.from_bytes
- Execute takes (Instruction, CPU) → next_pc; CPU reference used for halt signaling
- sign_extend uses XOR trick: ((value ^ sign_bit) - sign_bit) & 0xFFFFFFFF
- to_signed: subtract 0x100000000 if >= 0x80000000
- Decoder dispatches on opcode to determine format, single decode() function
- Execute dispatches on opcode, then funct3/funct7 within each group
- ECALL/EBREAK set cpu.halted = True

## Blockers
None.

## Recent changes
- Implemented all RV32I: decode, execute, registers, RAM, CPU step/run loop
- 145 tests covering all 41 instructions, edge cases, Fibonacci integration
- CLI supports `uv run python -m riscv_npu run <binary>`
- Had to `uv pip install -e .` to fix import issues (uv sync wasn't sufficient)
