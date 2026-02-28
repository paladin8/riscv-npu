# Project State

## Status
Phase 2 COMPLETE. 268 tests passing. Next: Phase 3 (UART + syscalls).

## What's implemented
- RV32IM: all 49 instructions (41 base + 8 M extension)
- ELF loader: parse_elf, load_elf, find_symbol (hand-rolled)
- CSR shim + machine-mode traps (ECALL→mtvec, MRET→mepc)
- CLI: auto-detects ELF, sets PC=entry, SP=0x80FFFFF0
- Compliance: 50 riscv-tests passing (42 rv32ui + 8 rv32um)
- Firmware: fibonacci (59 cycles, a0=55), sort (336 cycles, a0=1)

## Key patterns
- Instruction: frozen dataclass (opcode, rd, rs1, rs2, imm, funct3, funct7)
- Execute takes (Instruction, CPU) → next_pc
- Decoder dispatches on opcode; execute dispatches on opcode then funct3/funct7
- sign_extend: XOR trick. to_signed: subtract 0x100000000 if >= 0x80000000
- ECALL: traps to mtvec if configured, otherwise halts
- CSR dict on CPU with csr_read/csr_write; tohost (0x51E) halts on non-zero write
- Toolchain: riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32

## Blockers
None.
