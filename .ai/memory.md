# Project State

## Status
Phase 11 (arrax NPU vector instructions) COMPLETE. 1079 tests passing, 0 failures.
All 11 phases complete.

## What's implemented
- RV32IMF: 75 instructions (41 base + 8 M extension + 26 F extension)
- Custom NPU: 30 instructions (14 int opcode 0x0B + 16 FP opcode 0x2B)
- Total: 105 instructions
- ELF loader, CSR shim, machine-mode traps, MemoryBus, UART, SyscallHandler
- Syscalls: read, write, exit, brk, openat, close, lseek (7 total)
- File I/O: fd table, sandbox (--fs-root), O_CREAT/O_TRUNC/O_APPEND/O_RDWR
- CLI: run + debug + gdb subcommands with --fs-root; TUI debugger with FPU + NPU + stats panels
- GDB Remote Stub: RSP over TCP, software breakpoints, step/continue, register/memory R/W
  - Target XML for RISC-V CPU+FPU registers (org.gnu.gdb.riscv.cpu/fpu features)
  - No-ack mode, Ctrl+C interrupt, qXfer target description
- Firmware: fibonacci, sort, hello, uart-hello, fpu_test, newton, npu_test, mnist, transformer, file_demo
- Docs: isa-reference.md, npu-design.md, performance.md
- Cython NPU acceleration: _accel.pyx with 10 vector kernels, try-import fallback

## Key patterns
- FP NPU: opcode 0x2B, funct3 selects group, funct7 sub-dispatch for funct3=0 (0-6 original, 7-12 arrax)
- CRITICAL: NPU_FVMAC/FMACC accumulate onto facc without clearing
- Toolchain: riscv64-unknown-elf-gcc -march=rv32imf -mabi=ilp32f
- 32-bit masking (& 0xFFFFFFFF) after every arithmetic op
- GDB stub: src/riscv_npu/gdb/ (protocol.py, commands.py, target_xml.py, server.py)
- GDB register numbering: 0-31 GPR, 32 PC, 33-64 FPR, 65-67 fflags/frm/fcsr
- File syscalls: Linux RISC-V ABI numbers (openat=56, close=57, lseek=62)
- Error returns: -errno as unsigned 32-bit (e.g., -EBADF = 0xFFFFFFF7)
- Firmware entry: use main() not _start() (start.s calls main)
