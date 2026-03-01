# Project State

## Status
Phase 6 COMPLETE. 558 tests passing. MNIST quantized inference works end-to-end.

## What's implemented
- RV32IM: all 49 instructions (41 base + 8 M extension)
- ELF loader: parse_elf, load_elf, find_symbol (hand-rolled)
- CSR shim + machine-mode traps (ECALL->mtvec, MRET->mepc)
- MemoryBus: routes addr ranges to devices; Device Protocol (read8/write8)
- UART: 16550-style TX/RX/LSR at 0x10000000, injectable tx_stream + push_rx
- SyscallHandler: write(64), read(63), exit(93), brk(214) via ECALL dispatch
- CLI: run + debug subcommands, MemoryBus + UART + SyscallHandler wired up
- TUI debugger: disasm, registers, memory hex dump, NPU panel, output panel
- NPU: 8 custom instructions (opcode 0x0B), NpuState (64-bit acc + 4 vregs)
  - MACC, RELU, QMUL, CLAMP, GELU (lookup table), RSTACC, LDVEC, STVEC
- Compliance: 50 riscv-tests passing (42 rv32ui + 8 rv32um)
- Firmware: fibonacci, sort, hello, uart-hello, npu_test, mnist (all PASS)
- MNIST: quantized 784->128(ReLU)->10 MLP, ~99% accuracy, ~600K cycles/image
- Docs: docs/npu-design.md, docs/isa-reference.md

## Key patterns
- NPU instructions dispatched via funct3 on opcode 0x0B
- Quantization: per-layer symmetric int8, shift-based rescaling (not QMUL)
- Layer 1: MACC(pixel_uint8, weight_int8) -> SRA shift1 -> CLAMP -> RELU
- Layer 2: MACC(hidden_int8, weight_int8) -> raw int32 argmax (no rescale)
- Toolchain: riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32
- torch/torchvision in optional deps (uv run --extra torch)

## Blockers
None.
