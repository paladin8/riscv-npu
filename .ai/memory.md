# Project State

## Status
NPU vector length limit (256 bytes) implemented. 1113 tests passing, 0 failures.

## What's implemented
- RV32IMF: 75 instructions (41 base + 8 M extension + 26 F extension)
- Custom NPU: 30 instructions (14 int opcode 0x0B + 16 FP opcode 0x2B)
- Total: 105 instructions
- ELF loader, CSR shim, machine-mode traps, MemoryBus, UART, SyscallHandler
- Syscalls: read, write, exit, brk, openat, close, lseek (7 total)
- File I/O: fd table, sandbox (--fs-root), O_CREAT/O_TRUNC/O_APPEND/O_RDWR
- CLI: run + debug + gdb subcommands with --fs-root; TUI debugger with FPU + NPU + stats panels
- GDB Remote Stub: RSP over TCP, software breakpoints, step/continue, register/memory R/W
- Library API: Emulator class with load_elf, run, reset, symbol lookup, typed array I/O
- Firmware: fibonacci, sort, hello, uart-hello, fpu_test, newton, npu_test, mnist, transformer, file_demo
- Docs: isa-reference.md, npu-design.md, performance.md, gdb.md
- Cython NPU acceleration: _accel.pyx with 10 vector kernels, try-import fallback
- NPU vector length limit: 256 bytes max per vector op (NpuVectorLengthFault on overflow)

## Key patterns
- FP NPU: opcode 0x2B, funct3 selects group, funct7 sub-dispatch for funct3=0 (0-6 original, 7-12 arrax)
- CRITICAL: NPU_FVMAC/FMACC accumulate onto facc without clearing
- Vector limit: NPU_MAX_VECTOR_BYTES=256 in engine.py, _check_vector_length(n, elem_bytes)
- Firmware must chunk large vectors (MNIST 784 int8 → 4 chunks, transformer FF_DIM 256 f32 → 4 chunks)
- Toolchain: riscv64-unknown-elf-gcc -march=rv32imf -mabi=ilp32f
- 32-bit masking (& 0xFFFFFFFF) after every arithmetic op
- File syscalls: Linux RISC-V ABI numbers (openat=56, close=57, lseek=62)
- Firmware entry: use main() not _start() (start.s calls main)
- Emulator API: src/riscv_npu/emulator.py, numpy is optional (lazy import)
