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
- firmware/common/Makefile: must use CC=riscv64-unknown-elf-gcc (the installed toolchain), CFLAGS=-march=rv32im -mabi=ilp32. No exceptions.
- firmware/fibonacci/main.c + Makefile

## Design Decisions

1. **ELF parser is hand-rolled**: The spec forbids pyelftools. Parse ELF32 header and program headers using `struct.unpack` from the standard library. Only need to handle ET_EXEC (statically linked executables), not ET_DYN or ET_REL.

2. **Only PT_LOAD segments matter**: We skip PT_NULL, PT_NOTE, etc. For each PT_LOAD, copy p_filesz bytes from the file to memory at p_vaddr, then zero-fill up to p_memsz (handles .bss).

3. **RAM.load_segment() method**: Add a public method to RAM for bulk-loading bytes at an absolute address. This avoids reaching into `_data` directly and provides proper bounds checking.

4. **M extension lives in the existing R-type dispatch**: All M instructions share opcode 0x33 (OP_R_TYPE) with funct7=0b0000001. Add a check for funct7==1 at the top of `_exec_r_type`, then dispatch to `_exec_m_ext()`.

5. **M extension arithmetic uses Python big ints**: Python handles arbitrary precision natively. For signed multiplication (MULH, MULHSU), convert to signed Python ints, multiply, then extract the upper 32 bits. Always mask results to 32 bits.

6. **CSR shim is minimal**: The riscv-tests binaries use CSRRW/CSRRS/CSRRC and their immediate variants. We handle them in execute's OP_SYSTEM branch (funct3 != 0). For tohost (CSR 0x51E), store the written value on the CPU object. For all other CSRs, reads return 0, writes discard. This is sufficient for compliance tests.

7. **Firmware uses riscv64-unknown-elf-gcc**: The installed toolchain is `riscv64-unknown-elf-gcc`, which supports `-march=rv32im -mabi=ilp32` to produce 32-bit code. All Makefiles must use this.

8. **tohost detection**: The compliance tests write to CSR 0x51E (tohost). The test harness checks `cpu.tohost` after execution. Value 1 = pass, other non-zero = fail (the failing test case number is `(value >> 1)`).

9. **Compliance test ELFs are downloaded, not checked in**: A shell script `tests/fixtures/riscv-tests/download.sh` fetches prebuilt ELFs. The .gitignore already excludes `tests/fixtures/riscv-tests/*.elf`. Tests are skipped if binaries are missing (pytest.importorskip pattern or skipIf).

## Deliverables (ordered by dependency)

### Deliverable 1: ELF Parser
Parse a 32-bit little-endian RISC-V ELF file. Return structured data (entry point + list of segments to load).

**Files:**
- `src/riscv_npu/loader/elf.py` (replace stub)

**Implementation:**
- `@dataclass` `ElfSegment`: `vaddr: int`, `data: bytes`, `memsz: int`
- `@dataclass` `ElfProgram`: `entry: int`, `segments: list[ElfSegment]`
- `def parse_elf(data: bytes) -> ElfProgram`: validates ELF header, reads program headers, extracts PT_LOAD segments
- ELF32 header: 52 bytes. Key fields at known offsets (use `struct.unpack_from`):
  - e_ident[0:4] = b'\x7fELF' (magic)
  - e_ident[4] = 1 (ELFCLASS32)
  - e_ident[5] = 1 (ELFDATA2LSB, little-endian)
  - e_machine (offset 18) = 0xF3 (EM_RISCV)
  - e_entry (offset 24): entry point
  - e_phoff (offset 28): program header table offset
  - e_phentsize (offset 42): size of each program header entry
  - e_phnum (offset 44): number of program header entries
- Program header (32 bytes each): p_type (0=PT_NULL, 1=PT_LOAD), p_offset, p_vaddr, p_filesz, p_memsz
- For each PT_LOAD: extract `data[p_offset : p_offset + p_filesz]` and store with vaddr and memsz
- Raise `ValueError` with descriptive messages on any validation failure

### Deliverable 2: RAM Bulk Loading + ELF Loader Integration
Add RAM.load_segment() and a high-level `load_elf()` function that ties parsing to memory loading.

**Files:**
- `src/riscv_npu/memory/ram.py` (add `load_segment` method)
- `src/riscv_npu/loader/elf.py` (add `load_elf` function)
- `src/riscv_npu/loader/__init__.py` (re-export)

**Implementation:**
- `RAM.load_segment(addr: int, data: bytes) -> None`: bulk-copies `data` into RAM at absolute address `addr`. Bounds-checked.
- `def load_elf(path: str, ram: RAM) -> int`: reads file, calls `parse_elf`, loads each segment via `ram.load_segment(seg.vaddr, seg.data + b'\x00' * (seg.memsz - len(seg.data)))`. Returns entry point.
- Update `src/riscv_npu/loader/__init__.py` to export `load_elf`, `parse_elf`, `ElfProgram`, `ElfSegment`

### Deliverable 3: M Extension (MUL/MULH/MULHSU/MULHU/DIV/DIVU/REM/REMU)
Add the 8 RV32M instructions to the executor.

**Files:**
- `src/riscv_npu/cpu/execute.py` (modify `_exec_r_type`, add `_exec_m_ext`)

**Implementation:**
- In `_exec_r_type`, check `f7 == 0b0000001` before the existing funct3 dispatch. If so, call `_exec_m_ext(inst, regs, pc)`.
- `def _exec_m_ext(inst: Instruction, regs: RegisterFile, pc: int) -> int`:
  - Read rs1, rs2 as unsigned 32-bit values
  - Dispatch on funct3:
    - 0b000 MUL: `(rs1 * rs2) & 0xFFFFFFFF` (lower 32 bits)
    - 0b001 MULH: signed x signed, upper 32 bits. `s1=to_signed(rs1); s2=to_signed(rs2); result = ((s1 * s2) >> 32) & 0xFFFFFFFF`
    - 0b010 MULHSU: signed x unsigned, upper 32 bits. `s1=to_signed(rs1); result = ((s1 * rs2) >> 32) & 0xFFFFFFFF`
    - 0b011 MULHU: unsigned x unsigned, upper 32 bits. `result = ((rs1 * rs2) >> 32) & 0xFFFFFFFF`
    - 0b100 DIV: signed division with rounding toward zero. Edge cases: div-by-zero returns 0xFFFFFFFF; overflow (0x80000000 / -1) returns 0x80000000
    - 0b101 DIVU: unsigned division. Div-by-zero returns 0xFFFFFFFF.
    - 0b110 REM: signed remainder. Div-by-zero returns dividend. Overflow (0x80000000 % -1) returns 0.
    - 0b111 REMU: unsigned remainder. Div-by-zero returns dividend.
  - All results masked to 32 bits before writing to rd

### Deliverable 4: CSR Shim for Compliance Tests
Add minimal CSR instruction handling so riscv-tests binaries can run.

**Files:**
- `src/riscv_npu/cpu/execute.py` (modify OP_SYSTEM handler)
- `src/riscv_npu/cpu/cpu.py` (add `tohost` attribute)

**Implementation:**
- Add `self.tohost: int = 0` to CPU.__init__
- In the OP_SYSTEM handler, when funct3 != 0, handle CSR instructions:
  - CSR address = `inst.imm & 0xFFF` (the raw 12-bit immediate, not sign-extended)
  - funct3 001 = CSRRW: write rs1 to CSR, read old value to rd
  - funct3 010 = CSRRS: read CSR to rd, set bits from rs1
  - funct3 011 = CSRRC: read CSR to rd, clear bits from rs1
  - funct3 101 = CSRRWI: write zimm (rs1 field, zero-extended) to CSR, read old to rd
  - funct3 110 = CSRRSI: read CSR to rd, set bits from zimm
  - funct3 111 = CSRRCI: read CSR to rd, clear bits from zimm
- For CSR 0x51E (tohost): reads return `cpu.tohost`, writes set `cpu.tohost` to new value. When a non-zero value is written, set `cpu.halted = True`.
- For all other CSRs: reads return 0, writes are discarded (no-op)
- Note: the CSR immediate in the instruction word is the raw bits[31:20]. Since OP_SYSTEM uses I-type encoding, `inst.imm` is sign-extended. We need the raw 12-bit value, so mask: `csr_addr = inst.imm & 0xFFF`.

### Deliverable 5: CLI Update for ELF Loading
Update the CLI to detect ELF files and load them properly, setting PC and SP.

**Files:**
- `src/riscv_npu/cli.py` (modify to support ELF)

**Implementation:**
- In `run_binary()`: check if file starts with `b'\x7fELF'`. If so, use `load_elf()` to load, set `cpu.pc = entry`, set `cpu.registers.write(2, 0x80FFFFF0)` (SP). Otherwise, fall back to raw binary loading as before.
- Import `load_elf` from `riscv_npu.loader`

### Deliverable 6: Firmware Scaffolding (common/ and fibonacci/)
Create the firmware build infrastructure and a fibonacci test program.

**Files:**
- `firmware/common/start.s` (new)
- `firmware/common/linker.ld` (new)
- `firmware/common/Makefile` (new)
- `firmware/fibonacci/main.c` (new)
- `firmware/fibonacci/Makefile` (new)

**Implementation:**
- `start.s`:
  ```asm
  .section .text.init
  .globl _start
  _start:
      la sp, _stack_top
      call main
      # Exit: put return value in a0, ecall
      li a7, 93        # exit syscall number (convention)
      ecall
  ```
- `linker.ld`:
  ```ld
  ENTRY(_start)
  MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 1M }
  SECTIONS {
      .text : { *(.text.init) *(.text*) } > RAM
      .rodata : { *(.rodata*) } > RAM
      .data : { *(.data*) } > RAM
      .bss : { *(.bss*) *(COMMON) } > RAM
      . = ALIGN(16);
      _stack_top = ORIGIN(RAM) + LENGTH(RAM) - 16;
  }
  ```
- `firmware/common/Makefile`: defines CC, CFLAGS, LDFLAGS variables for inclusion
- `firmware/fibonacci/main.c`: computes fib(10) = 55, returns it in a0 (return from main)
- `firmware/fibonacci/Makefile`: includes common, builds fibonacci.elf

### Deliverable 7: riscv-tests Download Script + Compliance Tests
Download prebuilt riscv-tests ELFs and write pytest-based compliance test runners.

**Files:**
- `tests/fixtures/riscv-tests/download.sh` (new)
- `tests/integration/test_rv32i_compliance.py` (replace stub)
- `tests/integration/test_rv32m_compliance.py` (replace stub)

**Implementation:**
- `download.sh`: downloads prebuilt rv32ui-p-* and rv32um-p-* ELF binaries from a riscv-tests release. Uses curl/wget.
- Compliance test structure:
  ```python
  import pytest, pathlib
  from riscv_npu.loader.elf import parse_elf
  from riscv_npu.cpu.cpu import CPU
  from riscv_npu.memory.ram import RAM

  FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures" / "riscv-tests"
  BASE = 0x80000000
  RAM_SIZE = 1024 * 1024

  def run_riscv_test(elf_path: pathlib.Path) -> None:
      data = elf_path.read_bytes()
      prog = parse_elf(data)
      ram = RAM(BASE, RAM_SIZE)
      for seg in prog.segments:
          padded = seg.data + b'\x00' * (seg.memsz - len(seg.data))
          ram.load_segment(seg.vaddr, padded)
      cpu = CPU(ram)
      cpu.pc = prog.entry
      cpu.run(max_cycles=100_000)
      assert cpu.tohost == 1, f"Test failed: tohost={cpu.tohost} (test case {cpu.tohost >> 1})"

  # Parametrize over all rv32ui-p-* files
  RV32UI_TESTS = sorted(FIXTURES.glob("rv32ui-p-*"))
  @pytest.mark.parametrize("elf_path", RV32UI_TESTS, ids=lambda p: p.name)
  def test_rv32ui(elf_path):
      if not elf_path.exists():
          pytest.skip("riscv-tests not downloaded")
      run_riscv_test(elf_path)
  ```
- Similar for rv32um-p-* in test_rv32m_compliance.py

## Test Coverage Requirements

### Deliverable 1 Tests (tests/loader/test_elf.py)
- **test_parse_valid_elf**: Construct a minimal valid ELF32 with one PT_LOAD segment, verify parse_elf returns correct entry and segment data
- **test_bad_magic**: Wrong magic bytes -> ValueError
- **test_wrong_class**: ELFCLASS64 -> ValueError
- **test_wrong_endian**: Big-endian -> ValueError
- **test_wrong_machine**: Non-RISC-V -> ValueError
- **test_multiple_segments**: ELF with 2 PT_LOAD segments, verify both are returned
- **test_bss_segment**: Segment where memsz > filesz (uninitialized data)
- **test_non_load_segments_skipped**: PT_NULL and PT_NOTE segments are ignored

### Deliverable 2 Tests (tests/loader/test_elf.py, tests/memory/test_ram.py)
- **test_ram_load_segment**: Load bytes at an address, read them back
- **test_ram_load_segment_bounds_check**: Loading beyond RAM raises MemoryError
- **test_load_elf_integration**: Create a minimal ELF file on disk, load it, verify memory contents and returned entry point

### Deliverable 3 Tests (tests/cpu/test_execute.py)
- One test class per instruction: TestMUL, TestMULH, TestMULHSU, TestMULHU, TestDIV, TestDIVU, TestREM, TestREMU
- Each class tests: basic operation, edge cases (zero operands, max values)
- **Critical edge cases** (from spec):
  - DIV by zero -> 0xFFFFFFFF
  - DIVU by zero -> 0xFFFFFFFF
  - REM by zero -> rs1 (dividend)
  - REMU by zero -> rs1 (dividend)
  - signed overflow: 0x80000000 / 0xFFFFFFFF (i.e., INT_MIN / -1) -> 0x80000000 (DIV), 0 (REM)
  - MULH of -1 * -1 -> 0
  - MULHSU of negative * unsigned
  - MUL of large values (only lower 32 bits)

### Deliverable 4 Tests (tests/cpu/test_execute.py)
- **test_csrrw_tohost**: CSRRW to CSR 0x51E sets cpu.tohost and halts
- **test_csrrs_ignored_csr**: CSRRS to an unknown CSR reads 0
- **test_csrrw_other_csr_discarded**: CSRRW to non-tohost CSR is silently discarded
- **test_csrrwi**: Immediate variant works
- **test_csr_read_to_rd**: Verify rd gets the old CSR value on write

### Deliverable 5 Tests
- Manual testing via CLI (no automated test needed -- the compliance tests validate ELF loading end-to-end)

### Deliverable 6 Tests
- `cd firmware/fibonacci && make` must produce fibonacci.elf
- `uv run python -m riscv_npu run firmware/fibonacci/fibonacci.elf` outputs a0=55

### Deliverable 7 Tests
- The compliance tests themselves ARE the tests
- rv32ui-p-* tests validate all RV32I instructions
- rv32um-p-* tests validate all RV32M instructions

## M Extension Edge Cases (must test)
- DIV by zero -> 0xFFFFFFFF (DIV), 0xFFFFFFFF (DIVU)
- REM by zero -> rs1
- 0x80000000 / 0xFFFFFFFF -> 0x80000000 (DIV), 0 (REM)

## riscv-tests Compliance

Run the official RISC-V test suite (https://github.com/riscv-software-src/riscv-tests) against the emulator. This catches spec-compliance bugs that hand-written tests miss -- signed/unsigned edge cases, immediate encoding corners, x0 invariants, etc.

How to integrate:
- Download prebuilt RV32I and RV32M test ELF binaries from the riscv-tests repo (under isa/rv32ui-p-* and isa/rv32um-p-* -- the "-p" variants run in bare-metal/physical mode, no virtual memory)
- The test binaries signal pass/fail by writing to a "tohost" CSR or memory-mapped address. Detect this: the test writes 1 to tohost on pass, or a non-1 value encoding the failing test number on failure.
- **CSR shim required:** The test binaries use CSR instructions (csrr, csrw, csrrw -- opcode 1110011, funct3 != 000). Your emulator does not implement full CSR support. Add a minimal shim: decode CSR instructions (I-type, opcode 1110011, funct3 001/010/011/101/110/111), intercept writes to tohost (CSR 0x51E), ignore all other CSR reads (return 0) and writes (discard). This is ~20 lines in the decoder/executor, not a full CSR implementation.
- Store test ELFs in tests/fixtures/riscv-tests/ (gitignore the binaries, add a download script or Makefile target)
- tests/integration/test_rv32i_compliance.py runs all rv32ui-p-* tests
- tests/integration/test_rv32m_compliance.py runs all rv32um-p-* tests (or combine into one file)

These tests need the ELF loader to work, so implement the loader first, then run compliance as validation.

## Acceptance Criteria
1. `uv run pytest tests/loader/ -v` -- all pass
2. `uv run pytest tests/cpu/ -v` -- all pass (includes M extension tests)
3. `uv run pytest tests/integration/test_rv32i_compliance.py -v` -- all rv32ui tests pass
4. `uv run pytest tests/integration/test_rv32m_compliance.py -v` -- all rv32um tests pass
5. `cd firmware/fibonacci && make` -- compiles without errors
6. `uv run python -m riscv_npu run firmware/fibonacci/fibonacci.elf` -- halts with a0=55
7. All new functions have type hints and docstrings
8. 32-bit masking applied after every arithmetic operation
9. No external ELF parsing libraries used
10. All 152 existing tests still pass

## Scope Boundary
Do NOT implement: memory bus (use flat memory with ELF loading), UART, syscalls (ECALL halts), TUI, NPU
Do NOT use pyelftools or any ELF library. Parse the binary format directly.
