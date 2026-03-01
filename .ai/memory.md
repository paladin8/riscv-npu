# Project State

## Status
Phase 7 (Transformer Extension) COMPLETE. 635 tests collected, 633 passing, 2 integration skipped (need firmware build). Was 564 tests.

## What's implemented
- RV32IM: all 49 instructions (41 base + 8 M extension)
- ELF loader: parse_elf, load_elf, find_symbol (hand-rolled)
- CSR shim + machine-mode traps (ECALL->mtvec, MRET->mepc)
- MemoryBus: routes addr ranges to devices; Device Protocol (read8/write8)
- UART: 16550-style TX/RX/LSR at 0x10000000, injectable tx_stream + push_rx
- SyscallHandler: write(64), read(63), exit(93), brk(214) via ECALL dispatch
- CLI: run + debug subcommands, MemoryBus + UART + SyscallHandler wired up
- TUI debugger: disasm, registers, memory hex dump, NPU panel, output panel
- NPU: 14 custom instructions (opcode 0x0B), NpuState (64-bit acc + 4 vregs)
  - Phase 6: MACC, VMAC, RELU, QMUL, CLAMP, GELU, RSTACC, LDVEC, STVEC
  - Phase 7: VEXP, VRSQRT, VMUL, VREDUCE, VMAX (Q16.16 fixed-point)
- Compliance: 50 riscv-tests passing (42 rv32ui + 8 rv32um)
- Firmware: fibonacci, sort, hello, uart-hello, npu_test, mnist, transformer
- MNIST: quantized 784->128(ReLU)->10 MLP, ~99% accuracy, uses VMAC
- Transformer: char-level LM (dim=64, heads=4, layers=2, vocab=256, ctx=32)
  - Python reference: src/riscv_npu/npu/transformer.py
  - Weight exporter: tools/export_transformer_weights.py
  - C firmware: firmware/transformer/main.c
- Docs: docs/npu-design.md, docs/isa-reference.md (both updated for Phase 7)

## Key patterns
- NPU instructions: opcode 0x0B, funct3 selects operation group
- funct3=0 sub-dispatched by funct7: MACC(0), VMAC(1), VEXP(2), VRSQRT(3), VMUL(4), VREDUCE(5), VMAX(6)
- Q16.16 fixed-point: 1.0 = 65536, used for softmax/RMSNorm intermediates
- VMUL reads scale from acc_lo (avoids needing 4th register operand)
- Toolchain: riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32
- torch/torchvision in optional deps (uv run --extra torch)

## Blockers
None.
