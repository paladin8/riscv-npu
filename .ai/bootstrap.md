# riscv-npu bootstrap

Read this document. Then execute the bootstrap checklist at the end. That checklist tells you to create CLAUDE.md, .ai/memory.md, phase specs, and project scaffolding. Once those artifacts exist, they — not this document — are your ongoing reference.

---

## Project definition

RISC-V (RV32IM) emulator in Python with custom neural network accelerator instructions. Runs real ELF binaries cross-compiled with riscv32-unknown-elf-gcc. End goal: run quantized MNIST inference and a tiny transformer on the emulator using custom NPU instructions.

## Role boundaries

The human is the architect. They approve specs, review PRs, make design calls. You implement, test, commit. Do not ask "should I continue?" unless the spec is genuinely ambiguous. Check the spec, check CLAUDE.md, make a decision, keep moving.

## Decision protocol

1. Check CLAUDE.md for conventions
2. Check .ai/memory.md for prior decisions
3. Choose the simpler option
4. Log your choice in .ai/memory.md

If the decision is architectural (API shape, significant tradeoffs): describe options in .ai/memory.md with your recommendation. Move to a different task. Do not block.

---

## Repo structure

Create exactly this:

```
riscv-npu/
├── CLAUDE.md
├── pyproject.toml
├── README.md
├── .ai/
│   ├── memory.md
│   ├── specs/
│   │   ├── phase1-rv32i.md
│   │   ├── phase2-elf.md
│   │   ├── phase3-uart-syscalls.md
│   │   ├── phase4-tui.md
│   │   ├── phase5-npu.md
│   │   ├── phase6-mnist.md
│   │   ├── phase7-transformer.md
│   │   └── phase8-polish.md
│   └── completed/
├── src/riscv_npu/
│   ├── __init__.py
│   ├── cli.py
│   ├── cpu/
│   │   ├── __init__.py
│   │   ├── decode.py
│   │   ├── execute.py
│   │   ├── registers.py
│   │   └── cpu.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── bus.py
│   │   ├── ram.py
│   │   └── device.py
│   ├── devices/
│   │   ├── __init__.py
│   │   ├── uart.py
│   │   └── npu.py
│   ├── loader/
│   │   ├── __init__.py
│   │   └── elf.py
│   ├── syscall/
│   │   ├── __init__.py
│   │   └── handler.py
│   ├── npu/
│   │   ├── __init__.py
│   │   ├── instructions.py
│   │   ├── engine.py
│   │   └── quantize.py
│   └── tui/
│       ├── __init__.py
│       ├── app.py
│       ├── registers.py
│       ├── memory.py
│       ├── disasm.py
│       └── npu_status.py
├── tests/
│   ├── cpu/
│   │   ├── test_decode.py
│   │   ├── test_execute.py
│   │   ├── test_registers.py
│   │   └── test_cpu.py
│   ├── memory/
│   │   ├── test_bus.py
│   │   └── test_ram.py
│   ├── loader/
│   │   └── test_elf.py
│   ├── npu/
│   │   ├── test_instructions.py
│   │   └── test_engine.py
│   ├── integration/
│   │   ├── test_rv32i_compliance.py
│   │   ├── test_rv32m_compliance.py
│   │   ├── test_programs.py
│   │   └── test_mnist.py
│   └── fixtures/
│       ├── riscv-tests/
│       ├── programs/
│       └── expected/
├── firmware/
│   ├── common/
│   │   ├── start.s
│   │   ├── syscalls.c
│   │   ├── npu.h
│   │   ├── linker.ld
│   │   └── Makefile
│   ├── hello/
│   ├── fibonacci/
│   ├── sort/
│   ├── npu_test/
│   │   ├── main.c
│   │   └── Makefile
│   └── mnist/
│       ├── main.c
│       ├── nn_runtime.c
│       ├── nn_runtime.h
│       ├── weights.h
│       └── Makefile
├── tools/
│   ├── export_weights.py
│   └── assemble.py
└── docs/
    ├── isa-reference.md
    └── npu-design.md
```

Rules:
- `src/` = emulator (Python). `firmware/` = code that runs ON the emulator (C). Never mix.
- `tests/` mirrors `src/`. One test file per module.
- `.ai/` = your persistent working memory across sessions.
- Files stay under 300 lines. Split if they grow past that.

## Memory map

```
0x00000000 - 0x0000FFFF  Reserved
0x10000000 - 0x100000FF  UART
0x20000000 - 0x200000FF  NPU control registers
0x80000000 - 0x80FFFFFF  RAM (16 MB)
```

Stack starts at top of RAM, grows down. ELF segments load into RAM range.

---

## CLAUDE.md content

Create CLAUDE.md with exactly this content:

```
# riscv-npu

RISC-V (RV32IM) emulator in Python with custom NPU instructions for neural network inference.

## Commands
- uv run pytest                                    — run all tests
- uv run pytest tests/cpu/ -v                      — CPU tests only
- uv run pytest -x                                 — stop on first failure
- uv run python -m riscv_npu run <elf>             — run program headless
- uv run python -m riscv_npu debug <elf>           — run with TUI debugger
- cd firmware/<name> && make                — cross-compile firmware
- uv sync                                      — install/update dependencies
- uv add <pkg>                                 — add new dependency

## Session start
1. Read this file
2. Read .ai/memory.md
3. Read the active phase spec in .ai/specs/

## Session end
1. uv run pytest — confirm passing
2. Commit working changes (atomic, descriptive messages)
3. Update .ai/memory.md: what you did, what works, what's blocked

## Architecture
- src/riscv_npu/cpu/      — decode + execute (the core loop)
- src/riscv_npu/memory/   — bus, RAM, device base class
- src/riscv_npu/devices/  — UART, NPU (memory-mapped I/O)
- src/riscv_npu/loader/   — ELF parser
- src/riscv_npu/syscall/  — ecall dispatch
- src/riscv_npu/npu/      — custom NPU instruction execution + compute engine
- src/riscv_npu/tui/      — Rich-based terminal debugger
- firmware/               — C code that runs ON the emulator (cross-compiled)
- tools/                  — weight export, assembler utilities

## Conventions
- Python 3.14+, type hints on all signatures
- Dataclasses for structured data, functions for stateless ops
- snake_case everywhere
- Docstrings on all public functions
- One test file per module, pytest, descriptive names
- No external deps without logging rationale in .ai/memory.md. Add via uv add <pkg>.

## Testing rules
- One test minimum per instruction
- Run uv run pytest after every few functions, not just end of session
- Never modify tests to make them pass — fix the implementation
- Never skip or disable tests

## Git rules
- main is always passing
- Branch per phase: phase1-rv32i, phase2-elf, etc.
- Atomic commits after each working milestone
- Descriptive messages: "Implement R-type arithmetic (ADD, SUB, SLL...)" not "update file"

## 32-bit masking (critical)
Python ints are arbitrary precision. Mask to 32 bits (& 0xFFFFFFFF) after every arithmetic op.
Signed interpretation: if val >= 0x80000000, val is negative (val - 0x100000000).
This is the #1 source of subtle bugs. Be vigilant.

## Firmware compilation
All firmware MUST be compiled with: -march=rv32im -mabi=ilp32
This emulator does NOT support the A (atomics) or C (compressed) extensions. The C extension is especially dangerous — gcc enables it by default for rv32imac targets, producing 16-bit instructions that the decoder can't handle. Always pass -march=rv32im explicitly.
```

---

## ISA reference

This section is the complete instruction spec. Embed it in the phase specs that need it and in docs/isa-reference.md.

### Registers

32 × 32-bit general purpose: x0-x31. x0 hardwired to 0 (discard writes in register file, not per-instruction). PC is separate.

ABI names for TUI display: x0=zero, x1=ra, x2=sp, x3=gp, x4=tp, x5-x7=t0-t2, x8=s0/fp, x9=s1, x10-x11=a0-a1, x12-x17=a2-a7, x18-x27=s2-s11, x28-x31=t3-t6.

### Instruction formats

All 32-bit, little-endian. Bit layouts:

```
R: funct7[31:25]  rs2[24:20]  rs1[19:15]  funct3[14:12]  rd[11:7]   opcode[6:0]
I: imm[31:20]                 rs1[19:15]  funct3[14:12]  rd[11:7]   opcode[6:0]
S: imm[31:25]    rs2[24:20]  rs1[19:15]  funct3[14:12]  imm[11:7]  opcode[6:0]
B: imm[31:25]    rs2[24:20]  rs1[19:15]  funct3[14:12]  imm[11:7]  opcode[6:0]
U: imm[31:12]                                            rd[11:7]   opcode[6:0]
J: imm[31:12]                                            rd[11:7]   opcode[6:0]
```

Immediate reconstruction (the tricky part):
- **I-type**: imm = sign_extend(inst[31:20])
- **S-type**: imm = sign_extend(inst[31:25] << 5 | inst[11:7])
- **B-type**: imm = sign_extend(inst[31] << 12 | inst[7] << 11 | inst[30:25] << 5 | inst[11:8] << 1). LSB is always 0 (not encoded).
- **U-type**: imm = inst[31:12] << 12
- **J-type**: imm = sign_extend(inst[31] << 20 | inst[19:12] << 12 | inst[20] << 11 | inst[30:21] << 1). LSB is always 0.

B-type and J-type bit scrambling is the #1 decoder bug source. Test against known encodings.

### RV32I instructions

**R-type (opcode 0110011):**

| funct7  | funct3 | name | op |
|---------|--------|------|----|
| 0000000 | 000 | ADD  | rd = rs1 + rs2 |
| 0100000 | 000 | SUB  | rd = rs1 - rs2 |
| 0000000 | 001 | SLL  | rd = rs1 << rs2[4:0] |
| 0000000 | 010 | SLT  | rd = signed(rs1) < signed(rs2) ? 1 : 0 |
| 0000000 | 011 | SLTU | rd = rs1 < rs2 ? 1 : 0 (unsigned) |
| 0000000 | 100 | XOR  | rd = rs1 ^ rs2 |
| 0000000 | 101 | SRL  | rd = rs1 >> rs2[4:0] (logical) |
| 0100000 | 101 | SRA  | rd = signed(rs1) >> rs2[4:0] (arithmetic) |
| 0000000 | 110 | OR   | rd = rs1 \| rs2 |
| 0000000 | 111 | AND  | rd = rs1 & rs2 |

**I-type arithmetic (opcode 0010011):**

| funct3 | name  | op |
|--------|-------|----|
| 000 | ADDI  | rd = rs1 + sext(imm) |
| 010 | SLTI  | rd = signed(rs1) < sext(imm) ? 1 : 0 |
| 011 | SLTIU | rd = rs1 < sext(imm) ? 1 : 0 (note: sext imm, then unsigned compare) |
| 100 | XORI  | rd = rs1 ^ sext(imm) |
| 110 | ORI   | rd = rs1 \| sext(imm) |
| 111 | ANDI  | rd = rs1 & sext(imm) |
| 001 | SLLI  | rd = rs1 << imm[4:0] (imm[11:5] must be 0000000) |
| 101 | SRLI  | rd = rs1 >> imm[4:0] (logical, imm[11:5]=0000000) |
| 101 | SRAI  | rd = signed(rs1) >> imm[4:0] (arithmetic, imm[11:5]=0100000) |

**Loads (opcode 0000011):**

| funct3 | name | op |
|--------|------|----|
| 000 | LB  | rd = sext(mem8[rs1 + sext(imm)]) |
| 001 | LH  | rd = sext(mem16[rs1 + sext(imm)]) |
| 010 | LW  | rd = mem32[rs1 + sext(imm)] |
| 100 | LBU | rd = zext(mem8[rs1 + sext(imm)]) |
| 101 | LHU | rd = zext(mem16[rs1 + sext(imm)]) |

Little-endian. Allow misaligned access (no exceptions).

**Stores (opcode 0100011):**

| funct3 | name | op |
|--------|------|----|
| 000 | SB | mem8[rs1 + sext(imm)] = rs2[7:0] |
| 001 | SH | mem16[rs1 + sext(imm)] = rs2[15:0] |
| 010 | SW | mem32[rs1 + sext(imm)] = rs2[31:0] |

**Branches (opcode 1100011):**

| funct3 | name | condition |
|--------|------|-----------|
| 000 | BEQ  | rs1 == rs2 |
| 001 | BNE  | rs1 != rs2 |
| 100 | BLT  | signed(rs1) < signed(rs2) |
| 101 | BGE  | signed(rs1) >= signed(rs2) |
| 110 | BLTU | rs1 < rs2 (unsigned) |
| 111 | BGEU | rs1 >= rs2 (unsigned) |

If condition true: pc += sext(imm). Offset is relative to branch instruction address, not pc+4.

**Upper immediate:**

| opcode  | name  | op |
|---------|-------|----|
| 0110111 | LUI   | rd = imm << 12 |
| 0010111 | AUIPC | rd = pc + (imm << 12) |

**Jumps:**

| opcode  | name | op |
|---------|------|----|
| 1101111 | JAL  | rd = pc + 4; pc += sext(imm) [J-type] |
| 1100111 | JALR | rd = pc + 4; pc = (rs1 + sext(imm)) & ~1 [I-type] |

**System (opcode 1110011, funct3=000, rd=0, rs1=0):**

| imm[11:0]     | name   |
|----------------|--------|
| 000000000000   | ECALL  |
| 000000000001   | EBREAK |

**Memory ordering (opcode 0001111):**

| funct3 | name  | op |
|--------|-------|----|
| 000    | FENCE | No-op in this emulator (single-core, in-order, no reordering). Must still decode without error — compiled C code emits FENCE instructions. |

### M extension

R-type, opcode 0110011, **funct7 = 0000001**.

| funct3 | name   | op |
|--------|--------|----|
| 000 | MUL    | rd = (rs1 × rs2)[31:0] |
| 001 | MULH   | rd = (signed × signed)[63:32] |
| 010 | MULHSU | rd = (signed × unsigned)[63:32] |
| 011 | MULHU  | rd = (unsigned × unsigned)[63:32] |
| 100 | DIV    | rd = signed(rs1) / signed(rs2) (toward zero) |
| 101 | DIVU   | rd = unsigned(rs1) / unsigned(rs2) |
| 110 | REM    | rd = signed(rs1) % signed(rs2) |
| 111 | REMU   | rd = unsigned(rs1) % unsigned(rs2) |

Edge cases: div by zero → DIV returns 0xFFFFFFFF, DIVU returns 0xFFFFFFFF, REM returns rs1, REMU returns rs1. Signed overflow (0x80000000 / 0xFFFFFFFF) → DIV returns 0x80000000, REM returns 0.

### Custom NPU instructions

Opcode `0x0B` (custom-0 space).

**NPU internal state** (separate from register file):
- acc_lo, acc_hi: 32-bit halves of 64-bit accumulator
- vreg[0..3]: four registers, each 4 × int8

**R-type compute (opcode 0x0B):**

| funct7  | funct3 | name       | op |
|---------|--------|------------|----|
| 0000000 | 000 | NPU.MACC   | {acc_hi,acc_lo} += signed(rs1) × signed(rs2) |
| 0000000 | 001 | NPU.RELU   | rd = max(signed(rs1), 0) |
| 0000000 | 010 | NPU.QMUL   | rd = (signed(rs1) × signed(rs2)) >> 8 |
| 0000000 | 011 | NPU.CLAMP  | rd = clamp(signed(rs1), -128, 127) |
| 0000000 | 100 | NPU.GELU   | rd = gelu_approx(rs1) via lookup table |
| 0000000 | 101 | NPU.RSTACC | rd = acc_lo; acc = 0 |

**I-type data movement (opcode 0x0B):**

| funct3 | name       | op |
|--------|------------|----|
| 110 | NPU.LDVEC | vreg[rd%4] = mem32[rs1 + sext(imm)] as 4×int8 |
| 111 | NPU.STVEC | mem32[rs1 + sext(imm)] = vreg[rs2%4] as 4×int8 |

**C intrinsics header** — write this verbatim to `firmware/common/npu.h`:

```c
#ifndef NPU_H
#define NPU_H
#include <stdint.h>

#define NPU_MACC(a, b) \
    asm volatile(".insn r 0x0B, 0x0, 0x00, x0, %0, %1" :: "r"(a), "r"(b))

static inline int32_t NPU_RSTACC(void) {
    int32_t result;
    asm volatile(".insn r 0x0B, 0x5, 0x00, %0, x0, x0" : "=r"(result));
    return result;
}

static inline int32_t NPU_RELU(int32_t src) {
    int32_t dst;
    asm volatile(".insn r 0x0B, 0x1, 0x00, %0, %1, x0" : "=r"(dst) : "r"(src));
    return dst;
}

static inline int32_t NPU_GELU(int32_t src) {
    int32_t dst;
    asm volatile(".insn r 0x0B, 0x4, 0x00, %0, %1, x0" : "=r"(dst) : "r"(src));
    return dst;
}

static inline int32_t NPU_QMUL(int32_t a, int32_t b) {
    int32_t dst;
    asm volatile(".insn r 0x0B, 0x2, 0x00, %0, %1, %2" : "=r"(dst) : "r"(a), "r"(b));
    return dst;
}

static inline int32_t NPU_CLAMP(int32_t src) {
    int32_t dst;
    asm volatile(".insn r 0x0B, 0x3, 0x00, %0, %1, x0" : "=r"(dst) : "r"(src));
    return dst;
}
#endif
```

---

## Phase specs

Create each as a standalone file in `.ai/specs/`. An agent reads one phase spec and has everything it needs to implement that phase. Do not require reading the bootstrap doc after initial setup.

### .ai/specs/phase1-rv32i.md

```
# Phase 1: RV32I Core

## Goal
All RV32I instructions decode and execute correctly. Hand-written test programs run to completion.

## Prerequisites
Read docs/isa-reference.md before implementing. It contains the complete instruction tables, bit layouts, and immediate reconstruction formulas. All instruction semantics come from that file.

## What to build
- Instruction dataclass: opcode, rd, rs1, rs2, imm, funct3, funct7
- decode(word: int) -> Instruction: extract fields, reconstruct immediates with sign extension
- RegisterFile: 32 × uint32, x0 returns 0 on read, discards writes
- Memory: bytearray, read8/read16/read32/write8/write16/write32, little-endian
- execute(instruction, registers, memory): implement all RV32I ops per docs/isa-reference.md
- CPU.step(): fetch at PC → decode → execute → PC += 4 (unless branch/jump)
- CLI: uv run python -m riscv_npu run <binary> — loads raw binary at 0x80000000, sets PC, runs until ECALL/EBREAK or 1M cycles

## Immediate decoding (most bug-prone part)
Copy the exact bit-extraction formulas from docs/isa-reference.md. Test each format against a known instruction encoding.
B-type and J-type have scrambled bits. The LSB is implicitly 0 (not stored). Get this right first.

## Test infrastructure
Create tests/cpu/conftest.py with pytest fixtures:
- make_cpu(): returns a fresh CPU + Memory instance with default RAM at 0x80000000
- exec_instruction(cpu, word): decode and execute a single 32-bit instruction word, return the cpu state
- set_regs(cpu, **kwargs): set named registers (e.g., set_regs(cpu, x1=5, x2=10))
These fixtures eliminate setup duplication across 40+ instruction tests.

## Testing requirements
- One test per R-type instruction with at least: positive+positive, negative operand, result overflow
- One test per I-type with: zero imm, positive imm, negative imm (sign extension)
- Shift tests: shift by 0, shift by 31, shift of negative value (SRA vs SRL)
- Load/store: each width, signed vs unsigned extension, address at RAM base
- Branch: taken and not-taken for each, forward and backward offsets
- x0: verify writes are discarded (write to x0, read back, expect 0)
- Fibonacci program: encode as byte array in test, run CPU, check x10 == 55

## Acceptance
uv run pytest tests/cpu/ -v && uv run pytest tests/memory/ -v — all pass

## Files
Create/modify: cpu/decode.py, cpu/execute.py, cpu/registers.py, cpu/cpu.py, memory/ram.py, cli.py, all corresponding test files, tools/assemble.py (optional helper)

## Scope boundary
Do NOT implement: ELF loading, memory bus, UART, syscalls, TUI, NPU, M extension
```

### .ai/specs/phase2-elf.md

```
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
```

### .ai/specs/phase3-uart-syscalls.md

```
# Phase 3: UART + Syscalls

## Goal
Programs print to stdout, read from stdin, and exit with a code. Interactive terminal programs (ANSI escape codes, keyboard input) work.

## What to build
- MemoryBus: routes address ranges to devices. Interface: read8/16/32, write8/16/32. Devices register (base_addr, size, device).
- RAM device: wraps existing memory, mapped at 0x80000000
- UART device: mapped at 0x10000000.
  - TX (write): write byte to address → appears on host stdout
  - RX (read): read byte from address → returns next byte from host stdin (non-blocking, returns 0 if no input available)
  - LSR (Line Status Register) at offset 0x05: bit 0 = data ready (input available), bit 5 = THR empty (can write). This is standard 16550 UART behavior.
- SyscallHandler: on ECALL, read a7 for number:
  - 63 (read): a0=fd, a1=buf_ptr, a2=len → read from stdin (fd=0 only), returns bytes read
  - 64 (write): a0=fd, a1=buf_ptr, a2=len → write to stdout (fd=1 only)
  - 93 (exit): a0=code → halt, report code
  - 214 (brk): a0=addr → bump allocator
- firmware/common/syscalls.c: putchar, getchar, puts, exit wrappers using ecall
- firmware/hello/main.c: "Hello, World!" via write syscall

## UART implementation notes
Host stdin must be read non-blocking so the emulator doesn't stall waiting for input. Use select/poll or platform-appropriate non-blocking IO. Buffer incoming bytes — the UART RX holds one byte at a time, the emulated program polls LSR then reads RX. ANSI escape codes from the emulated program pass through the UART TX to the host terminal unmodified — the host terminal interprets them. This means emulated programs can do cursor movement, colors, screen clearing, etc. with raw escape codes.

## Acceptance
uv run pytest tests/memory/test_bus.py -v — pass
uv run pytest tests/devices/ -v — pass
cd firmware/hello && make
uv run python -m riscv_npu run firmware/hello/hello.elf — prints "Hello, World!"
```

### .ai/specs/phase4-tui.md

```
# Phase 4: TUI Debugger

## Goal
Interactive Rich/Textual TUI for stepping through execution.

## Panels
- Registers: x0-x31 with ABI names + hex values, highlight changes since last step
- Disassembly: current PC ± 10 instructions, current line highlighted
- Memory: hex dump at configurable address
- Output: UART capture
- Status bar: PC, instruction count, run/pause state

## Controls
- step: execute one instruction
- continue: run until breakpoint or halt
- breakpoint <addr>: toggle breakpoint
- goto <addr>: move memory inspector

## Run modes
- uv run python -m riscv_npu run <elf> — headless
- uv run python -m riscv_npu debug <elf> — TUI

## Testing
Unit test formatting functions (register display, hex dump, disassembly text). Layout is manually verified. Expect human feedback and iteration on this phase.

## Acceptance
uv run python -m riscv_npu debug firmware/fibonacci/fibonacci.elf — opens, steps work, registers update
```

### .ai/specs/phase5-npu.md

```
# Phase 5: Custom NPU Instructions

## Goal
All NPU instructions execute correctly. C firmware using npu.h intrinsics produces correct results.

## What to build
- NPU state class: acc_lo, acc_hi (uint32), vreg[4] (each 4×int8)
- Decode: opcode 0x0B → dispatch to NPU by funct3/funct7
- Implement all instructions from ISA reference NPU section
- GELU lookup table: precompute for int8 input range, return int8 output
- firmware/common/npu.h (copy from bootstrap doc)
- firmware/npu_test/: C program exercising each instruction, printing PASS/FAIL

## Module split
- npu/instructions.py + npu/engine.py: custom opcode 0x0B execution logic, NPU state, accumulator, GELU table. This is where instruction semantics live.
- devices/npu.py: memory-mapped control/status registers at 0x20000000 (for TUI status reads, DMA-style transfers in later phases). Thin wrapper, delegates to npu/engine.
These are separate concerns. Instruction execution goes in npu/, not devices/.

## Testing
- Unit: each NPU instruction, known inputs → expected output
- MACC chain: accumulate N multiply-adds, compare to Python sum
- QMUL: verify (a*b)>>8 with overflow cases
- CLAMP: values inside range, above 127, below -128
- GELU: compare table output to reference values computed with math.erf from Python standard library
- Integration: npu_test.elf runs, prints all PASS

## Acceptance
uv run pytest tests/npu/ -v — all pass
uv run python -m riscv_npu run firmware/npu_test/npu_test.elf — all PASS
```

### .ai/specs/phase6-mnist.md

```
# Phase 6: MNIST Inference

## Goal
Quantized MLP runs on emulator, correctly classifies digits.

## What to build
- tools/export_weights.py: train MLP (784→128 ReLU→10) on MNIST, quantize int8, export as C arrays
- firmware/mnist/weights.h: generated weight arrays
- firmware/mnist/nn_runtime.c: linear layer + activation using NPU_MACC, NPU_RSTACC, NPU_RELU, NPU_CLAMP
- firmware/mnist/main.c: load test image, run inference, print prediction
- tests/integration/test_mnist.py: run multiple images, compare to PyTorch reference

## Network
- Input: 784 int8 (28×28 flattened)
- Hidden: 128, ReLU
- Output: 10, argmax
- ~100KB weights

## Acceptance
uv run python tools/export_weights.py — generates weights.h
cd firmware/mnist && make
uv run python -m riscv_npu run firmware/mnist/mnist.elf — prints correct digit
uv run pytest tests/integration/test_mnist.py — ≥95% accuracy
```

### .ai/specs/phase7-transformer.md

```
# Phase 7: Transformer Extension

## Goal
Tiny transformer runs on emulator.

## Before implementing
Evaluate NPU instruction set gaps for transformer workload. Document in .ai/memory.md:
- Is softmax needed as a custom instruction?
- Is layer norm needed?
- Can attention be expressed with existing MACC + GELU?

Propose new instructions if needed. Wait for human approval before adding to ISA.

## Target model
- Embedding dim: 64, heads: 4, layers: 2
- Byte-level vocab (256), context: 32 tokens
- ~200K params, ~200KB int8
- Task: simple pattern (copy, reverse, or character-level LM)

## This phase is exploratory
Expect spec revision after Phase 6 findings. Keep .ai/memory.md updated with what works and what doesn't.
```

### .ai/specs/phase8-polish.md

```
# Phase 8: Polish

## Goal
Project is presentable and documented.

## Deliverables
- README: architecture, screenshots, usage, build instructions
- docs/npu-design.md: instruction rationale, what each accelerates
- docs/isa-reference.md: complete quick reference (should already exist, verify/update)
- TUI: NPU status panel, instruction statistics
- Code cleanup: consistent formatting, complete docstrings, no dead code
- Performance profile: where is time spent, what would hardware accelerate
```

---

## .ai/memory.md initial content

```
# Project State

## Current phase
Phase 1 — not started. Scaffold created.

## Decisions
None yet.

## Blockers
None.

## Recent changes
- Initial scaffold created from bootstrap document
```

---

## Dependencies

Package manager: **uv**. All commands use `uv run` (e.g., `uv run pytest`, `uv run python -m riscv_npu run`).

**Always use the latest stable versions of all dependencies.** Do not pin to old versions. When adding a dependency, `uv add <pkg>` will resolve to the latest by default — that's what we want. If a dependency has a major version bump during the project, upgrade to it.

pyproject.toml should include:
- requires-python = ">=3.14"
- Runtime: rich (or textual)
- Dev dependency group: pytest
- Optional dependency group: torch (for phase 6+ weight export only)

Python 3.14+. Type hints on all function signatures. Dataclasses for structured data. snake_case.

---

## Bootstrap checklist

Execute these steps in order:

1. Create repo structure from "Repo structure" section above
2. Create pyproject.toml (use uv as package manager — see Dependencies section)
3. Run `uv sync` to create venv and install all dependencies
4. Create CLAUDE.md from "CLAUDE.md content" section
5. Create .ai/memory.md from "initial content" section
6. Create all 8 phase specs in .ai/specs/ from "Phase specs" section
7. Create docs/isa-reference.md from the ISA reference tables
8. Create firmware/common/npu.h from the C intrinsics in NPU section
9. Create stub files for Phase 1 modules (cpu/decode.py, cpu/execute.py, cpu/registers.py, cpu/cpu.py, memory/ram.py, cli.py) — docstring + pass, no implementation. For non-Phase-1 packages (devices/, loader/, syscall/, npu/, tui/), create only __init__.py files so the package structure exists.
10. Create stub test files for Phase 1 — empty test functions or # TODO markers. Create tests/cpu/conftest.py with fixture stubs.
11. Run `uv run pytest` — should collect and show skipped/no-tests, zero import errors
12. git init, commit: "Initial project scaffold"
13. Update .ai/memory.md: "Scaffold complete. Ready for Phase 1."
14. Stop. Do not start Phase 1 in this session.

**Phase lifecycle:** When a phase is complete and merged to main, move its spec from .ai/specs/ to .ai/completed/.
