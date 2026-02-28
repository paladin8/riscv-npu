# Project State

## Current phase
Phase 2 -- COMPLETE. All 7 deliverables implemented. 268 tests passing.

## Phase 1 (COMPLETE)
- All RV32I implemented: decode, execute, registers, RAM, CPU step/run loop
- 152 tests covering all 41 instructions, edge cases, Fibonacci integration

## Phase 2 (COMPLETE)
- ELF parser: parse_elf, load_elf, find_symbol (hand-rolled, no pyelftools)
- RAM.load_segment for bulk loading
- M extension: 8 instructions (MUL/MULH/MULHSU/MULHU/DIV/DIVU/REM/REMU)
- CSR shim: register dict on CPU, csr_read/csr_write methods
- Machine-mode trap handling: ECALL traps to mtvec, MRET returns to mepc
- Memory-mapped tohost detection for riscv-tests compliance
- CLI: auto-detects ELF by magic, sets PC=entry and SP=0x80FFFFF0
- Firmware: common/ (start.s, linker.ld, Makefile), fibonacci/ (fib(10)=55)
- Compliance: 42 rv32ui-p-* and 8 rv32um-p-* tests all pass

## Decisions
- Instruction is a frozen dataclass with opcode, rd, rs1, rs2, imm, funct3, funct7
- RegisterFile is a class (not dataclass) wrapping list[int], x0 hardwired to 0
- RAM has base address + bytearray, bounds-checked, little-endian via int.from_bytes
- Execute takes (Instruction, CPU) -> next_pc; CPU reference used for halt signaling
- sign_extend uses XOR trick: ((value ^ sign_bit) - sign_bit) & 0xFFFFFFFF
- to_signed: subtract 0x100000000 if >= 0x80000000
- Decoder dispatches on opcode to determine format, single decode() function
- Execute dispatches on opcode, then funct3/funct7 within each group
- M extension: funct7==1 check in _exec_r_type, dispatch to _exec_m_ext
- CSR instructions: funct3 != 0 in OP_SYSTEM, uses cpu.csr_read/csr_write
- ECALL: if mtvec configured, traps (sets mcause/mepc, jumps to mtvec); otherwise halts
- MRET: jumps to mepc
- tohost detection: find_symbol parses ELF symtab, cpu.tohost_addr monitors memory writes
- Toolchain: riscv64-unknown-elf-gcc with -march=rv32im -mabi=ilp32

## Blockers
None.

## Recent changes
- Phase 2 fully implemented: ELF loader, M extension, CSR/trap handling, firmware, compliance
- 268 tests passing total
