# Phase 1: RV32I Core

## Goal
All RV32I instructions decode and execute correctly. Hand-written test programs run to completion.

## Prerequisites
Read docs/isa-reference.md before implementing. It contains the complete instruction tables, bit layouts, and immediate reconstruction formulas. All instruction semantics come from that file.

## What to build
- Instruction dataclass: opcode, rd, rs1, rs2, imm, funct3, funct7
- decode(word: int) -> Instruction: extract fields, reconstruct immediates with sign extension
- RegisterFile: 32 × uint32, x0 returns 0 on read, discards writes
- Memory: bytearray, read8/read16/read32/write8/write16/write32, little-endian
- execute(instruction, registers, memory): implement all RV32I ops per docs/isa-reference.md
- CPU.step(): fetch at PC → decode → execute → PC += 4 (unless branch/jump)
- CLI: uv run python -m riscv_npu run <binary> — loads raw binary at 0x80000000, sets PC, runs until ECALL/EBREAK or 1M cycles

## Immediate decoding (most bug-prone part)
Copy the exact bit-extraction formulas from docs/isa-reference.md. Test each format against a known instruction encoding.
B-type and J-type have scrambled bits. The LSB is implicitly 0 (not stored). Get this right first.

## Test infrastructure
Create tests/cpu/conftest.py with pytest fixtures:
- make_cpu(): returns a fresh CPU + Memory instance with default RAM at 0x80000000
- exec_instruction(cpu, word): decode and execute a single 32-bit instruction word, return the cpu state
- set_regs(cpu, **kwargs): set named registers (e.g., set_regs(cpu, x1=5, x2=10))
These fixtures eliminate setup duplication across 40+ instruction tests.

## Testing requirements
- One test per R-type instruction with at least: positive+positive, negative operand, result overflow
- One test per I-type with: zero imm, positive imm, negative imm (sign extension)
- Shift tests: shift by 0, shift by 31, shift of negative value (SRA vs SRL)
- Load/store: each width, signed vs unsigned extension, address at RAM base
- Branch: taken and not-taken for each, forward and backward offsets
- x0: verify writes are discarded (write to x0, read back, expect 0)
- Fibonacci program: encode as byte array in test, run CPU, check x10 == 55

## Acceptance
uv run pytest tests/cpu/ -v && uv run pytest tests/memory/ -v — all pass

## Files
Create/modify: cpu/decode.py, cpu/execute.py, cpu/registers.py, cpu/cpu.py, memory/ram.py, cli.py, all corresponding test files, tools/assemble.py (optional helper)

## Scope boundary
Do NOT implement: ELF loading, memory bus, UART, syscalls, TUI, NPU, M extension
