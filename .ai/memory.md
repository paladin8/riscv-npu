# Project State

## Status
Phase 5 COMPLETE. 512 tests passing. Next: Phase 6 (MNIST).

## What's implemented
- RV32IM: all 49 instructions (41 base + 8 M extension)
- ELF loader: parse_elf, load_elf, find_symbol (hand-rolled)
- CSR shim + machine-mode traps (ECALL->mtvec, MRET->mepc)
- MemoryBus: routes addr ranges to devices; Device Protocol (read8/write8)
- UART: 16550-style TX/RX/LSR at 0x10000000, injectable tx_stream + push_rx
- SyscallHandler: write(64), read(63), exit(93), brk(214) via ECALL dispatch
- CLI: run + debug subcommands, MemoryBus + UART + SyscallHandler wired up
- TUI debugger: disasm, registers, memory hex dump, debugger controller, Rich layout
- NPU: 8 custom instructions (opcode 0x0B), NpuState (64-bit acc + 4 vregs)
  - MACC, RELU, QMUL, CLAMP, GELU (lookup table), RSTACC, LDVEC, STVEC
  - NpuDevice at 0x20000000 for memory-mapped status reads
- Compliance: 50 riscv-tests passing (42 rv32ui + 8 rv32um)
- Firmware: fibonacci, sort, hello, uart-hello, npu_test (all PASS)

## Key patterns
- NPU instructions dispatched via funct3 on opcode 0x0B
- GELU: precomputed 256-entry int8 lookup table, scale factor 32.0
- NpuState is a mutable dataclass on CPU (cpu.npu_state)
- Accumulator: 64-bit as two uint32 halves (acc_lo, acc_hi)
- Toolchain: riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32

## Blockers
None.
