# Project State

## Status
Phase 8 (RV32F) COMPLETE. 755 passing, 2 skipped. All 11 rv32uf compliance tests pass.

## What's implemented
- RV32IMF: 75 instructions (41 base + 8 M extension + 26 F extension)
- RV32F: full single-precision float — 32 float registers, FCSR/FFLAGS/FRM CSR routing
  - Arithmetic: FADD.S, FSUB.S, FMUL.S, FDIV.S, FSQRT.S
  - FMA: FMADD.S, FMSUB.S, FNMSUB.S, FNMADD.S
  - Compare: FEQ.S, FLT.S, FLE.S
  - Convert: FCVT.W.S, FCVT.WU.S, FCVT.S.W, FCVT.S.WU
  - Move: FMV.X.W, FMV.W.X, FCLASS.S
  - Sign injection: FSGNJ.S, FSGNJN.S, FSGNJX.S
  - Min/Max: FMIN.S, FMAX.S
  - Load/Store: FLW, FSW
- ELF loader, CSR shim, machine-mode traps, MemoryBus, UART, SyscallHandler
- CLI: run + debug subcommands; TUI debugger with FPU + NPU panels
- NPU: 14 custom instructions (opcode 0x0B), NpuState (64-bit acc + 4 vregs)
- Firmware: fibonacci, sort, hello, uart-hello, npu_test, mnist, transformer

## Key paths
- src/riscv_npu/cpu/fpu.py — FRegisterFile, FpuState (CSR routing)
- src/riscv_npu/cpu/fpu_execute.py — 26 F-extension instruction execution
- src/riscv_npu/tools/ — weight exporters, assembler, transformer reference
- src/riscv_npu/npu/ — NPU instruction execution + compute engine

## Key patterns
- NPU: opcode 0x0B, funct3 selects group, funct7 sub-dispatch for funct3=0
- FPU: IEEE 754 bits via struct.pack/unpack, canonical NaN = 0x7FC00000
- Inexact detection: compare double-precision result vs single-precision roundtrip
- NaN handling: sNaN sets NV, any-NaN result -> canonical NaN
- Toolchain: riscv64-unknown-elf-gcc -march=rv32imf -mabi=ilp32f (or rv32im for int-only)
