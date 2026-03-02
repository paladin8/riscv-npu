# Project State

## Status
Phase 9 Cython acceleration COMPLETE. 960 tests passing, 0 failures.
All 9 phases complete.

## What's implemented
- RV32IMF: 75 instructions (41 base + 8 M extension + 26 F extension)
- Custom NPU: 24 instructions (14 int opcode 0x0B + 10 FP opcode 0x2B)
- Total: 99 instructions
- ELF loader, CSR shim, machine-mode traps, MemoryBus, UART, SyscallHandler
- CLI: run + debug subcommands; TUI debugger with FPU + NPU + stats panels
- Firmware: fibonacci, sort, hello, uart-hello, fpu_test, newton, npu_test, mnist, transformer
- Docs: isa-reference.md, npu-design.md, performance.md
- Cython NPU acceleration: _accel.pyx with 10 vector kernels, try-import fallback
- Profiling script: scripts/bench.py (micro-benchmarks + firmware workloads + cProfile)

## Performance optimizations
- Bus multi-byte delegation: read32/write32 delegate to RAM's native methods
- Device lookup cache: _find_device() caches last-hit DeviceMapping
- Instruction dataclass: frozen=True → slots=True
- Cython vector kernels: 7.7x speedup on mnist (116K → 900K ips)

## Key patterns
- FP NPU: opcode 0x2B, funct3 selects group, funct7 sub-dispatch for funct3=0
- CRITICAL: NPU_FVMAC/FMACC accumulate onto facc without clearing
- Toolchain: riscv64-unknown-elf-gcc -march=rv32imf -mabi=ilp32f
- 32-bit masking (& 0xFFFFFFFF) after every arithmetic op
- Cython: `make accel` to compile, `make clean-accel` to remove
- bus.get_device_data(addr) returns (bytearray, base) for direct buffer access
- _USE_ACCEL try-import pattern with Any-typed fallback stubs for basedpyright
