# Phase 2: ELF Loader + M Extension

## Goal
Load and run ELF binaries from riscv32-unknown-elf-gcc. M extension works.

## What to build
- ELF parser: validate header (32-bit LE RISC-V), read program headers, load PT_LOAD segments to their vaddr in memory
- Set PC = entry point from ELF header
- Set SP (x2) = 0x80FFFFF0 (top of RAM, 16-byte aligned) before entry
- M extension: 8 instructions (MUL/MULH/MULHSU/MULHU/DIV/DIVU/REM/REMU), dispatch on funct7=0000001
- firmware/common/start.s: _start sets sp, calls main, calls exit ecall
- firmware/common/linker.ld: .text at 0x80000000, .data after, _stack_top symbol
- firmware/common/Makefile: must use CC=riscv32-unknown-elf-gcc, CFLAGS=-march=rv32im -mabi=ilp32. No exceptions.
- firmware/fibonacci/main.c + Makefile

## M extension edge cases (must test)
- DIV by zero → 0xFFFFFFFF (DIV), 0xFFFFFFFF (DIVU)
- REM by zero → rs1
- 0x80000000 / 0xFFFFFFFF → 0x80000000 (DIV), 0 (REM)

## riscv-tests compliance
Run the official RISC-V test suite (https://github.com/riscv-software-src/riscv-tests) against the emulator. This catches spec-compliance bugs that hand-written tests miss — signed/unsigned edge cases, immediate encoding corners, x0 invariants, etc.

How to integrate:
- Download prebuilt RV32I and RV32M test ELF binaries from the riscv-tests repo (under isa/rv32ui-p-* and isa/rv32um-p-* — the "-p" variants run in bare-metal/physical mode, no virtual memory)
- The test binaries signal pass/fail by writing to a "tohost" CSR or memory-mapped address. Detect this: the test writes 1 to tohost on pass, or a non-1 value encoding the failing test number on failure.
- **CSR shim required:** The test binaries use CSR instructions (csrr, csrw, csrrw — opcode 1110011, funct3 != 000). Your emulator does not implement full CSR support. Add a minimal shim: decode CSR instructions (I-type, opcode 1110011, funct3 001/010/011/101/110/111), intercept writes to tohost (CSR 0x51E), ignore all other CSR reads (return 0) and writes (discard). This is ~20 lines in the decoder/executor, not a full CSR implementation.
- Store test ELFs in tests/fixtures/riscv-tests/ (gitignore the binaries, add a download script or Makefile target)
- tests/integration/test_rv32i_compliance.py runs all rv32ui-p-* tests
- tests/integration/test_rv32m_compliance.py runs all rv32um-p-* tests (or combine into one file)

These tests need the ELF loader to work, so implement the loader first, then run compliance as validation.

## Acceptance
uv run pytest tests/loader/ -v — all pass
uv run pytest tests/cpu/ -v — M extension tests pass
uv run pytest tests/integration/test_rv32i_compliance.py -v — all rv32ui tests pass
uv run pytest tests/integration/test_rv32m_compliance.py -v — all rv32um tests pass
cd firmware/fibonacci && make — compiles
uv run python -m riscv_npu run firmware/fibonacci/fibonacci.elf — correct result

## Scope boundary
Do NOT implement: memory bus (use flat memory with ELF loading), UART, syscalls (ECALL halts), TUI, NPU
Do NOT use pyelftools or any ELF library. Parse the binary format directly.
