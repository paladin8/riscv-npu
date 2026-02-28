# Project State

## Status
Phase 3 IN PROGRESS. 278 tests passing. D1+D2 done (MemoryBus + CPU integration). Next: D3 (UART).

## What's implemented
- RV32IM: all 49 instructions (41 base + 8 M extension)
- ELF loader: parse_elf, load_elf, find_symbol (hand-rolled)
- CSR shim + machine-mode traps (ECALL->mtvec, MRET->mepc)
- MemoryBus: routes addr ranges to devices; Device Protocol (read8/write8)
- CPU uses MemoryBus (duck-type replacement for RAM)
- Compliance: 50 riscv-tests passing (42 rv32ui + 8 rv32um)
- Firmware: fibonacci, sort

## Key patterns
- MemoryBus composes multi-byte from device read8/write8 (little-endian)
- RAM satisfies Device protocol without changes (already has read8/write8 with absolute addrs)
- Toolchain: riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32

## Blockers
None.
