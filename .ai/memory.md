# Project State

## Status
Phase 7 COMPLETE (transformer). 633 passing, 2 skipped. Restructured tools/ into src/riscv_npu/tools/. Quantized transformer has ~22% float/quant top-1 agreement — int8 quantization scheme fundamentally clips in linear layers (MAC overflow after shift). Considering adding RV32F (float extension) to eliminate quantization entirely.

## What's implemented
- RV32IM: all 49 instructions (41 base + 8 M extension)
- ELF loader, CSR shim, machine-mode traps, MemoryBus, UART, SyscallHandler
- CLI: run + debug subcommands; TUI debugger with NPU panel
- NPU: 14 custom instructions (opcode 0x0B), NpuState (64-bit acc + 4 vregs)
  - Phase 6: MACC, VMAC, RELU, QMUL, CLAMP, GELU, RSTACC, LDVEC, STVEC
  - Phase 7: VEXP, VRSQRT, VMUL, VREDUCE, VMAX (Q16.16 fixed-point)
- Firmware: fibonacci, sort, hello, uart-hello, npu_test, mnist, transformer
- MNIST: quantized 784->128(ReLU)->10 MLP, ~99% accuracy
- Transformer: char-level LM (dim=64, heads=4, layers=2, vocab=256, ctx=32)
  - QAT training with custom MultiHeadAttention, activation fake-quantization
  - Calibration-based bias scaling for activation magnitudes
  - Softmax Q16.16 bug fixed (scores must << 16 before exp)

## Key paths
- src/riscv_npu/tools/ — weight exporters, assembler, transformer reference
- src/riscv_npu/npu/ — NPU instruction execution + compute engine
- tests/tools/ — transformer reference tests
- firmware/transformer/main.c — C firmware using NPU instructions

## Key patterns
- NPU: opcode 0x0B, funct3 selects group, funct7 sub-dispatch for funct3=0
- Q16.16 fixed-point: 1.0 = 65536
- Power-of-2 quantization scales (scale = 2^shift) for exact shift cancellation
- Toolchain: riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32
- torch in optional deps (uv run --extra torch)

## Quantization lessons learned
- Single-shift linear layers clip badly: MAC = dim * max(W) * max(x) >> w_shift overflows int8
- Bias must be scaled by weight_scale * activation_scale (not just weight_scale)
- QAT with per-tensor fake-quant doesn't match firmware's integer arithmetic
- RV32F extension would eliminate all quantization issues for transformer
