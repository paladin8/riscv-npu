# Project State

## Status
Phase 7 FP transformer COMPLETE. 812 passing, 0 failures.

## What's implemented
- RV32IMF: 75 instructions (41 base + 8 M extension + 26 F extension)
- ELF loader, CSR shim, machine-mode traps, MemoryBus, UART, SyscallHandler
- CLI: run + debug subcommands; TUI debugger with FPU + NPU panels
- Integer NPU: 14 instructions (opcode 0x0B), NpuState (64-bit acc + 4 vregs)
- FP NPU: 10 instructions (opcode 0x2B), float64 accumulator (facc)
- Float32 transformer: Python reference + C firmware using FP NPU intrinsics
- Firmware: fibonacci, sort, hello, uart-hello, npu_test, mnist, transformer

## Key paths
- src/riscv_npu/tools/transformer.py -- float32 reference (dot_f32, rmsnorm, softmax, linear, gelu, transformer_forward)
- src/riscv_npu/tools/export_transformer_weights.py -- train + export float32 weights (no quantization)
- firmware/transformer/main.c -- float32 firmware using npu_fp.h intrinsics
- src/riscv_npu/npu/fp_instructions.py -- 10 FP NPU instruction handlers
- firmware/common/npu_fp.h -- FP NPU C intrinsics

## Key patterns
- FP NPU: opcode 0x2B, funct3 selects group, funct7 sub-dispatch for funct3=0
- facc is float64 (Python float), rounded to f32 on output via struct.pack
- Transformer: all float32, no int8/Q16.16/shifts/scales/clamp
- Memory float read/write: struct.unpack('<f', struct.pack('<I', bits))[0]
- Toolchain: riscv64-unknown-elf-gcc -march=rv32imf -mabi=ilp32f
