# Phase 5: Custom NPU Instructions

## Goal
All NPU instructions execute correctly. C firmware using npu.h intrinsics produces correct results.

## What to build
- NPU state class: acc_lo, acc_hi (uint32), vreg[4] (each 4x int8)
- Decode: opcode 0x0B -> dispatch to NPU by funct3/funct7
- Implement all instructions from ISA reference NPU section
- GELU lookup table: precompute for int8 input range, return int8 output
- firmware/common/npu.h (already exists from bootstrap)
- firmware/npu_test/: C program exercising each instruction, printing PASS/FAIL

## Module split
- npu/instructions.py + npu/engine.py: custom opcode 0x0B execution logic, NPU state, accumulator, GELU table. This is where instruction semantics live.
- devices/npu.py: memory-mapped control/status registers at 0x20000000 (for TUI status reads, DMA-style transfers in later phases). Thin wrapper, delegates to npu/engine.
These are separate concerns. Instruction execution goes in npu/, not devices/.

## Design Decisions

1. **NpuState as dataclass**: Mutable dataclass holding acc_lo, acc_hi (uint32), and vreg (list of 4 lists, each 4x int8). Not frozen since accumulator is mutated by MACC/RSTACC.

2. **GELU lookup table**: Precomputed at module load time for all 256 int8 inputs (-128..127). Uses math.erf for reference: gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2))). Input/output are int8 values (interpreting the register value's low byte as signed int8).

3. **NPU instruction dispatch**: opcode 0x0B decoded as R-type (funct3 0-5) or I-type (funct3 6-7). The decoder handles 0x0B the same as R-type but preserves both rd, rs1, rs2, and imm fields. execute.py dispatches to npu/instructions.py::execute_npu().

4. **32-bit masking**: All NPU arithmetic results masked to 32 bits. Accumulator is a 64-bit value stored as two uint32 halves. MACC adds signed(rs1) * signed(rs2) to the 64-bit accumulator.

5. **Signed interpretation for QMUL/CLAMP/GELU**: Use to_signed() from decode.py for signed register values. QMUL: (signed(rs1) * signed(rs2)) >> 8, result masked to 32 bits. CLAMP: clamp signed(rs1) to [-128, 127], store as uint32 (masked).

6. **LDVEC/STVEC use funct3 to distinguish from R-type**: funct3=6 is LDVEC (I-type: load 4 bytes from mem to vreg[rd%4]), funct3=7 is STVEC (S-type-like: store vreg[rs2%4] to mem). For decoding: both use opcode 0x0B with I-type immediate extraction for the offset.

7. **NPU device (devices/npu.py)**: Read-only status registers at 0x20000000. Offset 0: acc_lo, offset 4: acc_hi, offset 8-23: vreg[0-3] packed. Write to offset 0: reset accumulator. Minimal for now, expanded in later phases.

## Deliverables List

1. **D1: NPU Engine** - `npu/engine.py`: NpuState dataclass, GELU lookup table, accumulator helpers
2. **D2: NPU Instructions** - `npu/instructions.py`: execute_npu() dispatcher + all 8 instruction implementations
3. **D3: CPU Integration** - `cpu/decode.py` + `cpu/execute.py` + `cpu/cpu.py`: opcode 0x0B decode/dispatch, NpuState on CPU
4. **D4: NPU Device** - `devices/npu.py`: memory-mapped status registers
5. **D5: Unit Tests** - `tests/npu/test_engine.py` + `tests/npu/test_instructions.py`: comprehensive tests
6. **D6: Firmware** - `firmware/npu_test/main.c` + `Makefile`: C test program using npu.h intrinsics
7. **D7: Integration Test** - `tests/integration/test_npu_firmware.py`: run npu_test.elf, verify PASS output

## Implementation Details

### D1: npu/engine.py
```python
@dataclass
class NpuState:
    acc_lo: int = 0       # lower 32 bits of 64-bit accumulator
    acc_hi: int = 0       # upper 32 bits of 64-bit accumulator
    vreg: list[list[int]] # 4 vector registers, each [int8, int8, int8, int8]

def build_gelu_table() -> list[int]:
    """Precompute GELU for int8 inputs (-128..127) -> int8 output."""

def acc_add(state: NpuState, value: int) -> None:
    """Add a signed 64-bit value to the accumulator."""

def acc_reset(state: NpuState) -> int:
    """Reset accumulator, return old acc_lo."""
```

### D2: npu/instructions.py
```python
def execute_npu(inst: Instruction, cpu: CPU) -> int:
    """Dispatch NPU instruction by funct3."""
    # funct3=0: MACC, funct3=1: RELU, funct3=2: QMUL,
    # funct3=3: CLAMP, funct3=4: GELU, funct3=5: RSTACC,
    # funct3=6: LDVEC, funct3=7: STVEC
```

### D3: CPU Integration
- `decode.py`: Add `OP_NPU = 0x0B` constant. In decode(), handle opcode 0x0B:
  - funct3 in (0-5): decode as R-type (extract rd, rs1, rs2, funct3, funct7)
  - funct3 == 6 (LDVEC): decode as I-type (rd, rs1, imm)
  - funct3 == 7 (STVEC): decode as S-type (rs1, rs2, imm)
- `execute.py`: Add OP_NPU import, dispatch to npu/instructions.py::execute_npu()
- `cpu.py`: Add `npu_state: NpuState` attribute to CPU.__init__()

### D4: devices/npu.py
```python
class NpuDevice:
    """Memory-mapped NPU status registers at 0x20000000."""
    def __init__(self, npu_state: NpuState): ...
    def read8(self, addr: int) -> int: ...
    def write8(self, addr: int, value: int) -> None: ...
```

### D5: Test Coverage Requirements

**test_engine.py:**
- TestNpuState: initial state zeroed, vreg dimensions correct
- TestGeluTable: table length is 256, specific reference values match math.erf
- TestAccumulator: add positive, add negative, overflow wrapping, reset returns acc_lo

**test_instructions.py:**
- TestMACC: single multiply-add, chain of 10 MACC ops, negative operands, accumulator overflow
- TestRELU: positive passthrough, negative clamped to 0, zero stays 0
- TestQMUL: basic (10*20)>>8, overflow case (127*127)>>8, negative operands
- TestCLAMP: value in range unchanged, above 127 clamped, below -128 clamped, exact boundaries
- TestGELU: input 0 -> ~0, positive input, negative input (should be near 0)
- TestRSTACC: returns acc_lo, resets both halves to 0
- TestLDVEC: load 4 bytes from memory into vreg
- TestSTVEC: store vreg contents to memory as 4 bytes
- TestIntegration: MACC chain then RSTACC, full pipeline through CPU.step()

### D6: firmware/npu_test
Test each instruction with known inputs, compare to expected output, print PASS/FAIL per test.

### D7: Integration test
Run compiled npu_test.elf, capture UART output, verify each line contains "PASS".

## Acceptance Criteria
```
uv run pytest tests/npu/ -v                                    # all pass
uv run pytest tests/integration/test_npu_firmware.py -v        # all pass
uv run python -m riscv_npu run firmware/npu_test/npu_test.elf  # prints PASS for each test
```
