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

---

## Design Decisions

### Data structures
- **Instruction**: A frozen dataclass with fields `opcode`, `rd`, `rs1`, `rs2`, `imm`, `funct3`, `funct7`. All fields are `int`. This is the universal decoded representation — every format populates the same struct, unused fields default to 0.
- **RegisterFile**: A class wrapping a `list[int]` of 32 entries. `read(index)` returns 0 for x0. `write(index, value)` is a no-op for x0, otherwise masks value to 32 bits. No dataclass — needs method behavior.
- **RAM**: A class wrapping a `bytearray` with a `base_address` and `size`. Read/write methods translate absolute addresses to bytearray offsets. Little-endian byte order via `int.from_bytes`/`int.to_bytes`.
- **CPU**: A class holding `pc: int`, `registers: RegisterFile`, `memory: RAM`, and a `halted: bool` flag. The `step()` method implements fetch-decode-execute. The `run()` method loops `step()` until halted or cycle limit.

### 32-bit masking strategy
Every arithmetic result written to a register or used as a PC value is masked with `& 0xFFFFFFFF`. Sign extension helper: `sign_extend(value, bits)` that takes a bit width and returns a 32-bit signed representation. Signed comparison helper: `to_signed(value)` that converts unsigned 32-bit to Python signed int (subtract 0x100000000 if >= 0x80000000).

### Decoder design
A single `decode(word: int) -> Instruction` function that:
1. Extracts `opcode = word & 0x7F`
2. Determines format from opcode (R/I/S/B/U/J)
3. Extracts fields and reconstructs the immediate using format-specific logic
4. Returns a populated `Instruction`

The format is determined by opcode value — there's a fixed mapping (e.g., 0x33 → R-type, 0x13 → I-type, etc.). No need for a separate "format" field on Instruction.

### Execute design
A single `execute(inst: Instruction, regs: RegisterFile, memory: RAM, pc: int) -> int` function that returns the next PC. It dispatches on `inst.opcode`, then on `inst.funct3`/`inst.funct7` within each opcode group. Returns `pc + 4` for most instructions, or the branch/jump target when taken.

### Why not a dispatch table
A match/case or if/elif chain is clearer for 41 instructions than a dict of function pointers. It keeps all instruction logic in one file, easily auditable against the ISA reference. We can refactor to a dispatch table later if needed.

### RAM bounds checking
RAM checks that addresses fall within `[base_address, base_address + size)`. Out-of-bounds access raises a `MemoryError` with the faulting address. In Phase 1, there's only one RAM region — the memory bus comes in Phase 3.

### ECALL/EBREAK handling
In Phase 1 (no syscall dispatch), ECALL and EBREAK simply set `cpu.halted = True`. The CPU run loop checks this flag after each step.

---

## Deliverables List

1. **Instruction dataclass and helper functions** — `sign_extend()`, `to_signed()`, `Instruction` dataclass
2. **RegisterFile** — 32-register file with x0 hardwired to 0
3. **RAM** — byte-addressable memory with read/write at 8/16/32-bit widths
4. **Decoder** — `decode(word)` for all 6 instruction formats (R, I, S, B, U, J)
5. **Execute: R-type** — ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND
6. **Execute: I-type arithmetic** — ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI
7. **Execute: Loads and stores** — LB, LH, LW, LBU, LHU, SB, SH, SW
8. **Execute: Branches** — BEQ, BNE, BLT, BGE, BLTU, BGEU
9. **Execute: Upper immediate, jumps, system, fence** — LUI, AUIPC, JAL, JALR, ECALL, EBREAK, FENCE
10. **CPU step loop and test fixtures** — CPU class with step()/run(), conftest fixtures
11. **CLI and Fibonacci integration test** — CLI entry point, Fibonacci program test

---

## Implementation Details

### Deliverable 1: Instruction dataclass and helpers

**Files**: `src/riscv_npu/cpu/decode.py`

```python
from dataclasses import dataclass

def sign_extend(value: int, bits: int) -> int:
    """Sign-extend a `bits`-wide value to 32 bits."""
    sign_bit = 1 << (bits - 1)
    return ((value ^ sign_bit) - sign_bit) & 0xFFFFFFFF

def to_signed(value: int) -> int:
    """Interpret a 32-bit unsigned value as signed."""
    return value - 0x100000000 if value >= 0x80000000 else value

@dataclass(frozen=True)
class Instruction:
    opcode: int
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0
    funct3: int = 0
    funct7: int = 0
```

**Tests**: `tests/cpu/test_decode.py` — test `sign_extend` for 12-bit, 13-bit, 20-bit, 21-bit widths, positive and negative values; test `to_signed` for boundary values.

### Deliverable 2: RegisterFile

**Files**: `src/riscv_npu/cpu/registers.py`

```python
class RegisterFile:
    def __init__(self) -> None:
        self._regs: list[int] = [0] * 32

    def read(self, index: int) -> int:
        if index == 0:
            return 0
        return self._regs[index]

    def write(self, index: int, value: int) -> None:
        if index == 0:
            return
        self._regs[index] = value & 0xFFFFFFFF
```

Replace the existing `create_register_file()` stub with the `RegisterFile` class.

**Tests**: `tests/cpu/test_registers.py` — x0 always returns 0 after write, read/write for x1-x31, 32-bit overflow masking, all 32 registers accessible.

### Deliverable 3: RAM

**Files**: `src/riscv_npu/memory/ram.py`

```python
class RAM:
    def __init__(self, base: int, size: int) -> None:
        self.base = base
        self.size = size
        self._data = bytearray(size)

    def _offset(self, addr: int, width: int) -> int:
        offset = addr - self.base
        if offset < 0 or offset + width > self.size:
            raise MemoryError(f"Access out of bounds: 0x{addr:08X}")
        return offset

    def read8(self, addr: int) -> int: ...
    def read16(self, addr: int) -> int: ...
    def read32(self, addr: int) -> int: ...
    def write8(self, addr: int, value: int) -> None: ...
    def write16(self, addr: int, value: int) -> None: ...
    def write32(self, addr: int, value: int) -> None: ...
```

Uses `int.from_bytes(..., 'little')` and `int.to_bytes(..., 'little')`. Read methods return unsigned values. Misaligned access is allowed per spec.

**Tests**: `tests/memory/test_ram.py` — read/write at each width, little-endian byte ordering, boundary addresses, out-of-bounds raises MemoryError, write-then-read roundtrip.

### Deliverable 4: Decoder

**Files**: `src/riscv_npu/cpu/decode.py` (add `decode()` function)

The `decode(word: int) -> Instruction` function:
1. Extract common fields: `opcode = word & 0x7F`, `rd = (word >> 7) & 0x1F`, `funct3 = (word >> 12) & 0x7`, `rs1 = (word >> 15) & 0x1F`, `rs2 = (word >> 20) & 0x1F`, `funct7 = (word >> 25) & 0x7F`
2. Switch on opcode to determine format:
   - `0x33` (R-type): imm = 0
   - `0x13`, `0x03`, `0x67`, `0x73` (I-type): imm = sign_extend(word >> 20, 12). For SLLI/SRLI/SRAI: imm lower 5 bits are shamt, upper bits encode funct7.
   - `0x23` (S-type): imm = sign_extend(((word >> 25) << 5) | ((word >> 7) & 0x1F), 12)
   - `0x63` (B-type): imm = sign_extend(((word >> 31) << 12) | (((word >> 7) & 1) << 11) | (((word >> 25) & 0x3F) << 5) | (((word >> 8) & 0xF) << 1), 13)
   - `0x37`, `0x17` (U-type): imm = word & 0xFFFFF000
   - `0x6F` (J-type): imm = sign_extend(((word >> 31) << 20) | (((word >> 12) & 0xFF) << 12) | (((word >> 20) & 1) << 11) | (((word >> 21) & 0x3FF) << 1), 21)
   - `0x0F` (FENCE): treated as I-type, effectively a NOP

**Tests**: `tests/cpu/test_decode.py` — One test per format with a hand-assembled instruction word. Verify all fields. Extra tests for B-type and J-type with positive and negative offsets. Test SLLI/SRLI/SRAI immediate encoding.

### Deliverable 5: Execute R-type

**Files**: `src/riscv_npu/cpu/execute.py`

```python
def execute(inst: Instruction, cpu: CPU) -> int:
    """Execute an instruction. Returns next PC."""
```

The `execute` function receives the full `CPU` reference, from which it accesses `cpu.registers`, `cpu.memory`, and `cpu.pc`. This also allows ECALL/EBREAK to set `cpu.halted = True` directly.

R-type (opcode 0x33) dispatch on funct3/funct7:
- ADD: `(rs1_val + rs2_val) & 0xFFFFFFFF`
- SUB: `(rs1_val - rs2_val) & 0xFFFFFFFF`
- SLL: `(rs1_val << (rs2_val & 0x1F)) & 0xFFFFFFFF`
- SLT: `1 if to_signed(rs1_val) < to_signed(rs2_val) else 0`
- SLTU: `1 if rs1_val < rs2_val else 0`
- XOR: `rs1_val ^ rs2_val`
- SRL: `rs1_val >> (rs2_val & 0x1F)`
- SRA: `to_signed(rs1_val) >> (rs2_val & 0x1F)` then mask to 32 bits
- OR: `rs1_val | rs2_val`
- AND: `rs1_val & rs2_val`

All results masked to 32 bits before `regs.write(rd, result)`.

**Tests**: `tests/cpu/test_execute.py` — At least 3 cases per instruction: positive+positive, negative operand, overflow. Shifts tested at 0, 31, and with negative values (SRA vs SRL).

### Deliverable 6: Execute I-type arithmetic

**Files**: `src/riscv_npu/cpu/execute.py` (extend)

I-type arithmetic (opcode 0x13) dispatch on funct3:
- ADDI, SLTI, SLTIU, XORI, ORI, ANDI: same as R-type counterparts but with `inst.imm` instead of rs2
- SLLI: `(rs1_val << (inst.imm & 0x1F)) & 0xFFFFFFFF`
- SRLI: `rs1_val >> (inst.imm & 0x1F)`
- SRAI: `to_signed(rs1_val) >> (inst.imm & 0x1F)` then mask. Distinguished from SRLI by funct7 (bit 30 of instruction word = 1 for SRAI).

Note SLTIU: the immediate is sign-extended, then both operands are compared as unsigned.

**Tests**: `tests/cpu/test_execute.py` — Each instruction with zero/positive/negative immediates. SLTIU with sign-extended negative immediate treated as large unsigned.

### Deliverable 7: Execute Loads and Stores

**Files**: `src/riscv_npu/cpu/execute.py` (extend)

Loads (opcode 0x03): addr = `(rs1_val + to_signed(inst.imm)) & 0xFFFFFFFF`
- LB (funct3=0): `sign_extend(memory.read8(addr), 8)`
- LH (funct3=1): `sign_extend(memory.read16(addr), 16)`
- LW (funct3=2): `memory.read32(addr)`
- LBU (funct3=4): `memory.read8(addr)`
- LHU (funct3=5): `memory.read16(addr)`

Stores (opcode 0x23): addr = `(rs1_val + to_signed(inst.imm)) & 0xFFFFFFFF`
- SB (funct3=0): `memory.write8(addr, rs2_val & 0xFF)`
- SH (funct3=1): `memory.write16(addr, rs2_val & 0xFFFF)`
- SW (funct3=2): `memory.write32(addr, rs2_val & 0xFFFFFFFF)`

**Tests**: `tests/cpu/test_execute.py` — Each load/store width. LB with value 0xFF → sign extends to 0xFFFFFFFF. LBU with 0xFF → stays 0xFF. Store then load roundtrip at each width.

### Deliverable 8: Execute Branches

**Files**: `src/riscv_npu/cpu/execute.py` (extend)

Branches (opcode 0x63): Compare rs1_val and rs2_val per funct3.
- If condition true: return `(pc + to_signed(inst.imm)) & 0xFFFFFFFF` (offset relative to branch PC, NOT pc+4)
- If condition false: return `pc + 4`

Signed comparisons (BLT, BGE) use `to_signed()`. Unsigned comparisons (BLTU, BGEU) use raw values.

**Tests**: `tests/cpu/test_execute.py` — Each branch: taken (forward), taken (backward/negative offset), not-taken. Signed vs unsigned edge cases (e.g., 0x80000000 is less than 0 signed, greater than 0 unsigned).

### Deliverable 9: Execute Upper immediate, Jumps, System, Fence

**Files**: `src/riscv_npu/cpu/execute.py` (extend)

- LUI (opcode 0x37): `regs.write(rd, inst.imm)` — imm is already shifted left by 12 during decode
- AUIPC (opcode 0x17): `regs.write(rd, (pc + inst.imm) & 0xFFFFFFFF)`
- JAL (opcode 0x6F): `regs.write(rd, pc + 4)`, return `(pc + to_signed(inst.imm)) & 0xFFFFFFFF`
- JALR (opcode 0x67): `regs.write(rd, pc + 4)`, return `(rs1_val + to_signed(inst.imm)) & 0xFFFFFFFE` (zero LSB)
- ECALL (opcode 0x73, imm=0): set `halted` flag, return `pc + 4`
- EBREAK (opcode 0x73, imm=1): set `halted` flag, return `pc + 4`
- FENCE (opcode 0x0F): no-op, return `pc + 4`

For ECALL/EBREAK: execute receives the `CPU` reference and sets `cpu.halted = True` directly.

**Tests**: `tests/cpu/test_execute.py` — LUI loads upper 20 bits. AUIPC adds to PC. JAL/JALR save return address and jump. JALR zeros LSB. ECALL/EBREAK signal halt. FENCE is a no-op.

### Deliverable 10: CPU step loop and test fixtures

**Files**: `src/riscv_npu/cpu/cpu.py`, `tests/cpu/conftest.py`

```python
class CPU:
    def __init__(self, memory: RAM) -> None:
        self.pc: int = 0
        self.registers = RegisterFile()
        self.memory = memory
        self.halted: bool = False
        self.cycle_count: int = 0

    def step(self) -> None:
        word = self.memory.read32(self.pc)
        inst = decode(word)
        self.pc = execute(inst, self)
        self.cycle_count += 1

    def run(self, max_cycles: int = 1_000_000) -> None:
        while not self.halted and self.cycle_count < max_cycles:
            self.step()
```

The `execute(inst, cpu)` function receives the CPU reference for halt signaling. ECALL/EBREAK set `cpu.halted = True`.

**Test fixtures** in `tests/cpu/conftest.py`:
- `make_cpu()` → creates RAM(0x80000000, 1MB), CPU with PC at 0x80000000
- `exec_instruction(cpu, word)` → writes word at cpu.pc, calls cpu.step(), returns cpu
- `set_regs(cpu, **kwargs)` → parses `x0`-`x31` keys, calls `cpu.registers.write()`

**Tests**: `tests/cpu/test_cpu.py` — step fetches and executes, PC advances by 4, cycle count increments, run stops on ECALL, run stops at max cycles.

### Deliverable 11: CLI and Fibonacci integration test

**Files**: `src/riscv_npu/cli.py`, `tests/cpu/test_cpu.py`

CLI:
```python
import argparse
from .cpu.cpu import CPU
from .memory.ram import RAM

def main() -> None:
    parser = argparse.ArgumentParser(description="RISC-V NPU Emulator")
    sub = parser.add_subparsers(dest="command")
    run_parser = sub.add_parser("run", help="Run a binary")
    run_parser.add_argument("binary", help="Path to raw binary file")
    args = parser.parse_args()
    if args.command == "run":
        run_binary(args.binary)

def run_binary(path: str) -> None:
    BASE = 0x80000000
    ram = RAM(BASE, 1024 * 1024)  # 1 MB
    with open(path, "rb") as f:
        data = f.read()
    for i, b in enumerate(data):
        ram.write8(BASE + i, b)
    cpu = CPU(ram)
    cpu.pc = BASE
    cpu.run()
    print(f"Halted after {cpu.cycle_count} cycles. x10 (a0) = {cpu.registers.read(10)}")
```

**Fibonacci integration test** in `tests/cpu/test_cpu.py`:
Hand-encode a Fibonacci program computing fib(10) = 55. The program:
```
ADDI x10, x0, 0     # a = 0
ADDI x11, x0, 1     # b = 1
ADDI x12, x0, 10    # n = 10
ADDI x13, x0, 0     # i = 0
loop:
  BEQ x13, x12, done  # if i == n, done
  ADD x14, x10, x11   # t = a + b
  ADD x10, x11, x0    # a = b
  ADD x11, x14, x0    # b = t
  ADDI x13, x13, 1    # i++
  JAL x0, loop         # jump to loop
done:
  ECALL                # halt
```

Encode each instruction as a 32-bit word, load into RAM, run CPU, assert `cpu.registers.read(10) == 55`.

---

## Test Coverage Requirements

### Deliverable 1: Instruction dataclass and helpers
- `test_sign_extend_positive_12bit` — 12-bit value with MSB=0 stays positive
- `test_sign_extend_negative_12bit` — 12-bit value with MSB=1 extends to 32-bit negative
- `test_sign_extend_13bit` — B-type immediate (13-bit)
- `test_sign_extend_21bit` — J-type immediate (21-bit)
- `test_to_signed_positive` — value < 0x80000000 stays positive
- `test_to_signed_negative` — value >= 0x80000000 becomes negative
- `test_to_signed_boundary` — 0x80000000 → -2147483648, 0x7FFFFFFF → 2147483647

### Deliverable 2: RegisterFile
- `test_x0_always_zero` — write to x0, read back 0
- `test_read_write_x1_through_x31` — write and read back each register
- `test_write_masks_to_32_bits` — write 0x1_FFFFFFFF, read back 0xFFFFFFFF
- `test_initial_values_are_zero` — all registers start at 0

### Deliverable 3: RAM
- `test_read_write_8bit` — write8/read8 roundtrip
- `test_read_write_16bit` — write16/read16 roundtrip
- `test_read_write_32bit` — write32/read32 roundtrip
- `test_little_endian` — write32(0x04030201), read8 at each byte offset
- `test_out_of_bounds` — access beyond size raises MemoryError
- `test_base_address_offset` — reads/writes work relative to base address
- `test_signed_byte_read` — write 0xFF, read8 returns 255 (unsigned)

### Deliverable 4: Decoder
- `test_decode_r_type` — hand-encode ADD x1, x2, x3, verify all fields
- `test_decode_i_type` — hand-encode ADDI x1, x2, 100, verify fields + immediate
- `test_decode_i_type_negative_imm` — negative immediate sign-extended correctly
- `test_decode_s_type` — hand-encode SW, verify split immediate recombined
- `test_decode_b_type_positive` — forward branch offset
- `test_decode_b_type_negative` — backward branch offset (negative immediate)
- `test_decode_u_type` — LUI with upper immediate
- `test_decode_j_type_positive` — JAL with forward offset
- `test_decode_j_type_negative` — JAL with backward offset
- `test_decode_slli_srli_srai` — shift immediate encoding with funct7 bits

### Deliverable 5: Execute R-type
For each of ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND:
- Positive + positive operands
- Negative operand (signed interpretation)
- Result overflow / wrapping at 32 bits
Extra:
- `test_sra_vs_srl` — SRA sign-extends, SRL zero-fills
- `test_shift_by_zero` — shifts by 0 return original value
- `test_shift_by_31` — shifts by maximum amount
- `test_write_to_x0_discarded` — R-type with rd=x0 doesn't change x0

### Deliverable 6: Execute I-type arithmetic
For each of ADDI, SLTI, SLTIU, XORI, ORI, ANDI:
- Zero immediate
- Positive immediate
- Negative immediate (sign extension)
Extra:
- `test_sltiu_sign_extended_unsigned` — sign-extended -1 becomes 0xFFFFFFFF for unsigned compare
- `test_slli_srai_srli` — shift immediate variants
- `test_nop_encoding` — ADDI x0, x0, 0 is the NOP encoding

### Deliverable 7: Execute Loads and Stores
- `test_lb_positive` and `test_lb_negative` — byte sign extension
- `test_lh_positive` and `test_lh_negative` — halfword sign extension
- `test_lw` — full word load
- `test_lbu` — byte zero extension
- `test_lhu` — halfword zero extension
- `test_sb_sh_sw` — store at each width, then load back
- `test_store_truncates` — SW stores full 32 bits, SH stores lower 16, SB stores lower 8

### Deliverable 8: Execute Branches
For each of BEQ, BNE, BLT, BGE, BLTU, BGEU:
- `test_taken_forward` — condition true, positive offset
- `test_taken_backward` — condition true, negative offset
- `test_not_taken` — condition false, PC advances by 4
Extra:
- `test_blt_vs_bltu` — 0x80000000 is negative signed but large unsigned
- `test_beq_same_register` — branch always taken when rs1 == rs2

### Deliverable 9: Upper immediate, Jumps, System, Fence
- `test_lui` — loads upper 20 bits, lower 12 are zero
- `test_auipc` — adds upper immediate to PC
- `test_jal_forward` — saves return address, jumps forward
- `test_jal_backward` — saves return address, jumps backward
- `test_jalr` — saves return address, jumps to rs1+imm with LSB cleared
- `test_jalr_clears_lsb` — target address has bit 0 forced to 0
- `test_ecall_halts` — ECALL sets halted flag
- `test_ebreak_halts` — EBREAK sets halted flag
- `test_fence_nop` — FENCE advances PC by 4, no side effects

### Deliverable 10: CPU step loop and test fixtures
- `test_step_advances_pc` — PC increases by 4 after a non-branch instruction
- `test_step_increments_cycle` — cycle_count increases by 1 per step
- `test_run_stops_on_ecall` — run() terminates on ECALL
- `test_run_stops_at_max_cycles` — run() terminates at cycle limit
- `test_fixtures_work` — make_cpu, exec_instruction, set_regs produce expected state

### Deliverable 11: CLI and Fibonacci
- `test_fibonacci` — hand-encoded fib(10) program, run to completion, x10 == 55
- CLI tested manually: `uv run python -m riscv_npu run <test-binary>`

---

## Acceptance Criteria

1. `uv run pytest tests/cpu/ -v` — all tests pass (40+ tests)
2. `uv run pytest tests/memory/ -v` — all tests pass (7+ tests)
3. All 41 RV32I instructions decode correctly from hand-encoded instruction words
4. All 41 RV32I instructions execute with correct results per ISA reference
5. 32-bit masking applied consistently — no arbitrary-precision leakage
6. x0 register always reads as 0, writes are discarded
7. Fibonacci program runs to completion and produces x10 == 55
8. `uv run python -m riscv_npu run <binary>` loads and executes a raw binary file
9. ECALL/EBREAK halt the CPU
10. No modifications to test expectations — all tests written against correct ISA semantics
