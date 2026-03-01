# Project State

## Status
Phase 3 COMPLETE. 303 tests passing. Next: Phase 4 (TUI debugger).

## What's implemented
- RV32IM: all 49 instructions (41 base + 8 M extension)
- ELF loader: parse_elf, load_elf, find_symbol (hand-rolled)
- CSR shim + machine-mode traps (ECALL->mtvec, MRET->mepc)
- MemoryBus: routes addr ranges to devices; Device Protocol (read8/write8)
- UART: 16550-style TX/RX/LSR at 0x10000000, injectable tx_stream + push_rx
- SyscallHandler: write(64), read(63), exit(93), brk(214) via ECALL dispatch
- CLI: MemoryBus + UART + SyscallHandler wired up, exit code propagation
- Compliance: 50 riscv-tests passing (42 rv32ui + 8 rv32um)
- Firmware: fibonacci, sort, hello (syscall I/O), uart-hello (MMIO I/O)

## Key patterns
- MemoryBus composes multi-byte from device read8/write8 (little-endian)
- RAM satisfies Device protocol without changes (already has read8/write8 with absolute addrs)
- UART uses push_rx for RX (external code pushes bytes), no direct stdin reads
- ECALL priority: syscall_handler -> mtvec trap -> halt
- Toolchain: riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32

## Blockers
None.
