# Project State

## Status
VMAC instruction COMPLETE. 564 tests passing. MNIST quantized inference works end-to-end with VMAC.

## What's implemented
- RV32IM: all 49 instructions (41 base + 8 M extension)
- ELF loader: parse_elf, load_elf, find_symbol (hand-rolled)
- CSR shim + machine-mode traps (ECALL->mtvec, MRET->mepc)
- MemoryBus: routes addr ranges to devices; Device Protocol (read8/write8)
- UART: 16550-style TX/RX/LSR at 0x10000000, injectable tx_stream + push_rx
- SyscallHandler: write(64), read(63), exit(93), brk(214) via ECALL dispatch
- CLI: run + debug subcommands, MemoryBus + UART + SyscallHandler wired up
- TUI debugger: disasm, registers, memory hex dump, NPU panel, output panel
- NPU: 9 custom instructions (opcode 0x0B), NpuState (64-bit acc + 4 vregs)
  - MACC, VMAC, RELU, QMUL, CLAMP, GELU (lookup table), RSTACC, LDVEC, STVEC
  - VMAC: funct3=0/funct7=1, reads int8 arrays from memory, dot product in one instruction
- Compliance: 50 riscv-tests passing (42 rv32ui + 8 rv32um)
- Firmware: fibonacci, sort, hello, uart-hello, npu_test, mnist (all PASS)
- MNIST: quantized 784->128(ReLU)->10 MLP, ~99% accuracy, uses VMAC for dot products
- Docs: docs/npu-design.md, docs/isa-reference.md

## Key patterns
- NPU instructions dispatched via funct3 on opcode 0x0B; MACC/VMAC share funct3=0, differentiated by funct7
- VMAC firmware: L1 converts uint8 pixels to int8 (subtract 128), biases pre-adjusted at export (b += 128*sum(w))
- Layer 1: VMAC(signed_pixels, weights) -> SRA shift1 -> CLAMP -> RELU
- Layer 2: VMAC(hidden_int8, weights) -> raw int32 argmax (no rescale)
- Toolchain: riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32
- torch/torchvision in optional deps (uv run --extra torch)

## Blockers
None.
