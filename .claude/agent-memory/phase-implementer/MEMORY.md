# Phase Implementer Agent Memory

## Project Structure
- `src/riscv_npu/cpu/`: decode.py, execute.py, registers.py, cpu.py
- `src/riscv_npu/memory/`: ram.py (RAM with base+bytearray)
- `src/riscv_npu/loader/`: elf.py (parse_elf, load_elf, find_symbol)
- `src/riscv_npu/cli.py`: CLI entry point, auto-detects ELF vs raw binary
- `firmware/common/`: start.s, linker.ld, Makefile (shared build infra)
- `tests/cpu/conftest.py`: make_cpu, exec_instruction, set_regs fixtures

## Key Patterns
- Instruction encoding helpers in test_execute.py: `_r()`, `_i()`, `_s()`, `_b()`, `_j()`, `_m()`
- Test classes per instruction: `class TestADD`, `class TestMUL`, etc.
- CPU has `tohost_addr` for memory-mapped tohost monitoring (checked every step)
- CPU has `csrs: dict[int, int]` for CSR storage, `csr_read`/`csr_write` methods
- ELF parser validates header strictly: magic, class (32-bit), endian (LE), machine (RISC-V)

## Common Pitfalls
- riscv-tests tohost address varies per binary (e.g., 0x80001000 vs 0x80002000)
  Must use find_symbol() to get correct address per ELF
- Makefile `include` can steal the default target -- put `all:` target BEFORE include
- riscv-tests use machine-mode features (mtvec, mcause, mepc, mret, PMP CSRs)
  Need trap handling for ECALL and MRET to run compliance tests
- ECALL in riscv-tests: traps to mtvec, NOT just halts
- Toolchain is `riscv64-unknown-elf-gcc`, not `riscv32-unknown-elf-gcc`
- RAM needs 4MB for some riscv-tests (e.g., ld_st has segments up to 0x80003000+)
- CSR immediate: `inst.imm & 0xFFF` to get raw 12-bit address (undo sign extension)

## Test Counts (Phase 2 complete)
- 152 Phase 1 tests (RV32I core)
- 19 loader tests (ELF parser + find_symbol + load_elf integration)
- 16 RAM tests (read/write + load_segment)
- 32 M extension tests
- 10 CSR shim tests
- 7 CPU loop tests
- 42 rv32ui compliance tests
- 8 rv32um compliance tests
- Total: 268
