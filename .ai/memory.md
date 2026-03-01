# Project State

## Status
Phase 4 COMPLETE. 434 tests passing.

## What's implemented
- RV32IM: all 49 instructions (41 base + 8 M extension)
- ELF loader: parse_elf, load_elf, find_symbol (hand-rolled)
- CSR shim + machine-mode traps (ECALL->mtvec, MRET->mepc)
- MemoryBus: routes addr ranges to devices; Device Protocol (read8/write8)
- UART: 16550-style TX/RX/LSR at 0x10000000, injectable tx_stream + push_rx
- SyscallHandler: write(64), read(63), exit(93), brk(214) via ECALL dispatch
- CLI: run + debug subcommands, MemoryBus + UART + SyscallHandler wired up
- TUI debugger: disasm, registers, memory hex dump, debugger controller, Rich layout
- Compliance: 50 riscv-tests passing (42 rv32ui + 8 rv32um)
- Firmware: fibonacci, sort, hello (syscall I/O), uart-hello (MMIO I/O)

## Key patterns
- MemoryBus composes multi-byte from device read8/write8 (little-endian)
- UART uses push_rx for RX, injectable tx_stream for capture in debug mode
- ECALL priority: syscall_handler -> mtvec trap -> halt
- TUI formatters are pure functions returning Rich-markup strings (testable without terminal)
- DebuggerState is a mutable dataclass (not frozen) holding CPU + breakpoints + prev_regs
- Toolchain: riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32

## Blockers
None.
