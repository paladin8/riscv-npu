# Project State

## Status
Phase 8 polish COMPLETE. 939 tests passing, 0 failures.
All 8 phases complete. Project is fully documented and presentable.

## What's implemented
- RV32IMF: 75 instructions (41 base + 8 M extension + 26 F extension)
- Custom NPU: 24 instructions (14 int opcode 0x0B + 10 FP opcode 0x2B)
- Total: 99 instructions
- ELF loader, CSR shim, machine-mode traps, MemoryBus, UART, SyscallHandler
- CLI: run + debug subcommands; TUI debugger with FPU + NPU + stats panels
- Instruction statistics: per-mnemonic counting via instruction_mnemonic() in decode.py
- Float32 transformer: Python reference + C firmware using FP NPU intrinsics
- Firmware: fibonacci, sort, hello, uart-hello, fpu_test, newton, npu_test, mnist, transformer
- Docs: isa-reference.md, npu-design.md, performance.md

## Key patterns
- FP NPU: opcode 0x2B, funct3 selects group, funct7 sub-dispatch for funct3=0
- facc is float64 (Python float), rounded to f32 on output via struct.pack
- CRITICAL: NPU_FVMAC/FMACC accumulate onto facc without clearing
- Toolchain: riscv64-unknown-elf-gcc -march=rv32imf -mabi=ilp32f
- 32-bit masking (& 0xFFFFFFFF) after every arithmetic op
