# Project State

## Status
FP NPU instructions COMPLETE. 809 passing, 2 pre-existing failures (transformer integration).

## What's implemented
- RV32IMF: 75 instructions (41 base + 8 M extension + 26 F extension)
- ELF loader, CSR shim, machine-mode traps, MemoryBus, UART, SyscallHandler
- CLI: run + debug subcommands; TUI debugger with FPU + NPU panels
- Integer NPU: 14 instructions (opcode 0x0B), NpuState (64-bit acc + 4 vregs)
- FP NPU: 10 instructions (opcode 0x2B), float64 accumulator (facc)
  - Scalar: FMACC, FRELU, FGELU, FRSTACC
  - Vector: FVMAC, FVEXP, FVRSQRT, FVMUL, FVREDUCE, FVMAX
- Firmware: fibonacci, sort, hello, uart-hello, npu_test, mnist, transformer

## Key paths
- src/riscv_npu/npu/engine.py — NpuState (int acc + FP acc + vregs), fgelu()
- src/riscv_npu/npu/instructions.py — 14 integer NPU instruction handlers
- src/riscv_npu/npu/fp_instructions.py — 10 FP NPU instruction handlers
- src/riscv_npu/cpu/fpu.py — FRegisterFile, FpuState (CSR routing)
- firmware/common/npu.h — int NPU C intrinsics
- firmware/common/npu_fp.h — FP NPU C intrinsics

## Key patterns
- Int NPU: opcode 0x0B, funct3 selects group, funct7 sub-dispatch for funct3=0
- FP NPU: opcode 0x2B, same dispatch pattern, all R-type (no I/S variants)
- FP NPU uses f0-f31 float regs (cpu.fpu_state.fregs), NOT cpu.fpu
- facc is float64 (Python float), rounded to f32 on output via struct.pack
- Memory float read/write: struct.unpack('<f', struct.pack('<I', bits))[0]
- Toolchain: riscv64-unknown-elf-gcc -march=rv32imf -mabi=ilp32f
