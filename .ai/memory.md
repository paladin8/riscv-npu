# Project State

## Current phase
Phase 2 — IN PROGRESS. Spec expanded with 7 deliverables.

## Phase 1 (COMPLETE)
- All RV32I implemented: decode, execute, registers, RAM, CPU step/run loop
- 152 tests passing (was 145, now 152 after post-review fixes)

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

## Phase 2 Decisions
- ELF parser hand-rolled with struct.unpack (no pyelftools)
- M extension: funct7==1 check in _exec_r_type, dispatch to _exec_m_ext
- CSR shim: minimal, tohost (0x51E) stored on CPU, others return 0/discard
- Toolchain: riscv64-unknown-elf-gcc with -march=rv32im -mabi=ilp32

## Blockers
None.

## Recent changes
- Expanded Phase 2 spec with 7 deliverables, implementation details, test requirements
- Implemented ELF parser (parse_elf, ElfProgram, ElfSegment) with 13 tests
- Implemented RAM.load_segment() and load_elf() with 8 more tests
- 173 tests passing total
