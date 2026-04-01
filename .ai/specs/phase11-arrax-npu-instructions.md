# Phase 11: Arrax NPU Vector Instructions

## Goal
Add 6 new FP NPU vector instructions required by the [arrax](../../../arrax) compiler. These provide elementwise arithmetic, vectorized activations, and scalar-broadcast operations that arrax's `npu` dialect lowers to directly.

## Background
The arrax compiler (MLIR-based, targeting this emulator) needs memory-to-memory vector ops beyond the existing set. The existing FP NPU instructions (opcode 0x2B, funct3=0) use funct7 values 0x00–0x06. The new instructions continue at 0x07–0x0C.

## New Instructions

All use opcode `0x2B`, funct3 `000`, R-type format.
Register convention: `rd` = element count, `rs1` = source address, `rs2` = source/destination address.

| Instruction      | funct7    | Decimal | Operation                                           | Priority     |
|------------------|-----------|---------|-----------------------------------------------------|--------------|
| NPU.FVADD        | 0000111   | 7       | dst[i] = src1[i] + src2[i]                          | Required     |
| NPU.FVSUB        | 0001000   | 8       | dst[i] = src1[i] - src2[i]                          | Required     |
| NPU.FVRELU       | 0001001   | 9       | dst[i] = max(src1[i], 0.0)                          | Required     |
| NPU.FVGELU       | 0001010   | 10      | dst[i] = gelu(src1[i])                              | Required     |
| NPU.FVDIV        | 0001011   | 11      | dst[i] = src1[i] / (float32)facc                    | Nice-to-have |
| NPU.FVSUB_SCALAR | 0001100   | 12      | dst[i] = src1[i] - (float32)facc                    | Required     |

Where `src1 = mem_f32[regs[rs1] + i*4]`, `src2 = mem_f32[regs[rs2] + i*4]`, `dst = mem_f32[regs[rs2] + i*4]`, `n = regs[rd]`, and `(float32)facc` is the FP accumulator rounded to single precision.

### Detailed Semantics

**NPU.FVADD** — Elementwise vector addition. Result stored in-place at rs2. Source/destination overlap is the intended usage pattern (arrax emits `npu.fvadd %A, %B, %C, %n` where the result overwrites the second operand's buffer).

**NPU.FVSUB** — Elementwise vector subtraction: `rs1 - rs2`, result at rs2.

**NPU.FVRELU** — Vectorized ReLU: `max(val, 0.0)`. Source at rs1, result at rs2. May overlap. Edge cases match scalar FRELU: negative zero → +0.0, NaN → NaN.

**NPU.FVGELU** — Vectorized GELU: `0.5 * x * (1 + erf(x / sqrt(2)))`. Source at rs1, result at rs2. May overlap. Uses the same formula as the existing scalar `NPU.FGELU` (funct3=4). Edge cases match scalar FGELU: NaN → NaN, +inf → +inf, -inf → 0.0.

**NPU.FVDIV** — Divide each element by the FP accumulator scalar. Counterpart to existing `NPU.FVMUL` (funct7=0x04). The accumulator is NOT modified. Division by zero follows IEEE 754 (produces ±inf).

**NPU.FVSUB_SCALAR** — Subtract the FP accumulator scalar from each element. Counterpart to FVMUL's scalar multiply. The accumulator is NOT modified.

**Overlap behavior** — For FVADD and FVSUB, rs2 is both a source and the destination. Elements are processed in ascending index order. Partial overlap between the rs1 and rs2 address ranges (where the arrays are offset but overlapping) produces undefined results. The intended usage is either fully overlapping (rs1 == rs2) or non-overlapping buffers.

## What to build

### Module changes

- `npu/fp_instructions.py`: Add 6 new instruction handlers in the funct7 dispatch
- `npu/engine.py`: Add helper functions if needed (e.g., `fgelu_vec` using existing `fgelu`)
- `npu/_accel.pyx`: Add 6 Cython kernels for the new vector operations
- `cpu/decode.py`: No changes needed (already decodes opcode 0x2B as R-type)
- `cpu/execute.py`: No changes needed (already dispatches to fp_instructions)
- `devices/npu.py`: No changes needed
- `firmware/common/npu_fp.h`: Add inline asm intrinsic macros for the 6 new instructions
- `docs/isa-reference.md`: Add entries for all 6 new instructions

## Deliverables

### D1: Instruction Implementations
**File**: `src/riscv_npu/npu/fp_instructions.py`

Add handlers for funct7 values 7–12 in the existing dispatch. Each follows the same pattern as FVEXP/FVMUL:
- Read `n = regs[rd]`
- Read source address from `regs[rs1]` (and `regs[rs2]` for binary ops)
- Loop over `n` elements, reading/writing f32 values from memory
- For FVDIV and FVSUB_SCALAR: convert scalar via `_f64_to_f32_bits(npu.facc)` (same as FVMUL at line 276 — this handles OverflowError, unlike the engine-level `facc_to_f32_bits`)

```python
# FVADD (funct7 = 7) — binary elementwise, same signature as FVEXP
def _exec_fvadd(inst: Instruction, regs: RegisterFile, mem: MemoryBus) -> None:
    n = regs.read(inst.rd)
    addr_src1 = regs.read(inst.rs1)
    addr_src2 = regs.read(inst.rs2)
    for i in range(n):
        s1 = (addr_src1 + i * 4) & 0xFFFFFFFF
        s2 = (addr_src2 + i * 4) & 0xFFFFFFFF
        a = _read_mem_f32(mem, s1)
        b = _read_mem_f32(mem, s2)
        _write_mem_f32(mem, s2, a + b)

# FVSUB (funct7 = 8) — same pattern, a - b
# FVRELU (funct7 = 9) — max(a, 0.0), unary src→dst, NaN → NaN, -0.0 → +0.0
# FVGELU (funct7 = 10) — fgelu(a), unary src→dst, NaN/±inf edge cases

# FVDIV and FVSUB_SCALAR need the npu argument (for facc access):
# def _exec_fvdiv(inst: Instruction, regs: RegisterFile, mem: MemoryBus, npu: NpuState) -> None:
#     — same pattern as _exec_fvmul: scale_bits = _f64_to_f32_bits(npu.facc), then a / scale per element
# def _exec_fvsub_scalar(inst: Instruction, regs: RegisterFile, mem: MemoryBus, npu: NpuState) -> None:
#     — a - scalar per element
```

### D2: Cython Acceleration Kernels
**File**: `src/riscv_npu/npu/_accel.pyx`

Add 6 new kernels following the existing pattern (e.g., `fvexp_f32`, `fvmul_f32`). Use `def` (not `cpdef`), `unsigned char[:]` memoryviews, and `int` offsets — matching existing conventions:

```cython
def fvadd_f32(unsigned char[:] data, int src1, int src2, int n)
def fvsub_f32(unsigned char[:] data, int src1, int src2, int n)
def fvrelu_f32(unsigned char[:] data, int src, int dst, int n)
def fvgelu_f32(unsigned char[:] data, int src, int dst, int n)
def fvdiv_f32(unsigned char[:] data, int src, int dst, int n, unsigned int scale_bits)
def fvsub_scalar_f32(unsigned char[:] data, int src, int dst, int n, unsigned int scalar_bits)
```

Read-only kernels (none in this set) would use `const unsigned char[:]`. FVDIV and FVSUB_SCALAR take the scalar as IEEE 754 bits (same pattern as `fvmul_f32`'s `scale_bits` parameter).

The `fvgelu_f32` kernel must implement GELU inline using C math — it cannot call the Python `fgelu()` function. Add `erf` and `sqrt` to the libc cimport: `from libc.math cimport erf as c_erf, sqrt as c_sqrt`. Handle NaN/inf edge cases (NaN→NaN, +inf→+inf, -inf→0.0) consistently with `fvexp_f32`.

Update `fp_instructions.py` to use the Cython kernels: extend the try-import line to include `fvadd_f32, fvsub_f32, fvrelu_f32, fvgelu_f32, fvdiv_f32, fvsub_scalar_f32` and add corresponding `= None` fallback assignments in the `except ImportError` block (matching lines 14-23 of the existing code).

### D3: Firmware Intrinsics
**File**: `firmware/common/npu_fp.h`

Add inline asm macros following the existing pattern: `do { ... } while (0)` wrapper, explicit register pinning to `a0`/`a1`/`a2`, `"memory"` clobber, and `(src, dst, n)` parameter order. The `.insn` directive uses hard-coded register names (not `%0/%1/%2` operand substitution) because the register pins guarantee placement.

```c
/* FVADD: dst[i] = src1[i] + src2[i], i in 0..n-1 — result in-place at src2 */
#define NPU_FVADD(src1, src2, n) do { \
    register void *_s1 asm("a0") = (void *)(src1); \
    register void *_s2 asm("a1") = (void *)(src2); \
    register int _n asm("a2") = (n); \
    asm volatile(".insn r 0x2B, 0x0, 0x07, a2, a0, a1" \
                 :: "r"(_s1), "r"(_s2), "r"(_n) : "memory"); \
} while (0)

/* FVSUB: dst[i] = src1[i] - src2[i], i in 0..n-1 — result in-place at src2 */
#define NPU_FVSUB(src1, src2, n) do { \
    register void *_s1 asm("a0") = (void *)(src1); \
    register void *_s2 asm("a1") = (void *)(src2); \
    register int _n asm("a2") = (n); \
    asm volatile(".insn r 0x2B, 0x0, 0x08, a2, a0, a1" \
                 :: "r"(_s1), "r"(_s2), "r"(_n) : "memory"); \
} while (0)

/* FVRELU: dst[i] = max(src[i], 0.0), i in 0..n-1 */
#define NPU_FVRELU(src, dst, n) do { \
    register void *_s asm("a0") = (void *)(src); \
    register void *_d asm("a1") = (void *)(dst); \
    register int _n asm("a2") = (n); \
    asm volatile(".insn r 0x2B, 0x0, 0x09, a2, a0, a1" \
                 :: "r"(_s), "r"(_d), "r"(_n) : "memory"); \
} while (0)

/* FVGELU: dst[i] = gelu(src[i]), i in 0..n-1 */
#define NPU_FVGELU(src, dst, n) do { \
    register void *_s asm("a0") = (void *)(src); \
    register void *_d asm("a1") = (void *)(dst); \
    register int _n asm("a2") = (n); \
    asm volatile(".insn r 0x2B, 0x0, 0x0A, a2, a0, a1" \
                 :: "r"(_s), "r"(_d), "r"(_n) : "memory"); \
} while (0)

/* FVDIV: dst[i] = src[i] / (float32)facc, i in 0..n-1 */
#define NPU_FVDIV(src, dst, n) do { \
    register void *_s asm("a0") = (void *)(src); \
    register void *_d asm("a1") = (void *)(dst); \
    register int _n asm("a2") = (n); \
    asm volatile(".insn r 0x2B, 0x0, 0x0B, a2, a0, a1" \
                 :: "r"(_s), "r"(_d), "r"(_n) : "memory"); \
} while (0)

/* FVSUB_SCALAR: dst[i] = src[i] - (float32)facc, i in 0..n-1 */
#define NPU_FVSUB_SCALAR(src, dst, n) do { \
    register void *_s asm("a0") = (void *)(src); \
    register void *_d asm("a1") = (void *)(dst); \
    register int _n asm("a2") = (n); \
    asm volatile(".insn r 0x2B, 0x0, 0x0C, a2, a0, a1" \
                 :: "r"(_s), "r"(_d), "r"(_n) : "memory"); \
} while (0)
```

### D4: Unit Tests
**File**: `tests/npu/test_fp_vector_ops.py`

Separate test file (rather than appending to `test_fp_instructions.py`) to keep the existing 10-instruction test file manageable. One test class per instruction. Each test builds a minimal CPU+memory setup, writes f32 values to memory, executes the instruction, and verifies output.

**Test coverage per instruction:**

- **FVADD**: basic add, negative values, in-place overlap (src1 == src2 address), n=0 (no-op), n=1 (single element)
- **FVSUB**: basic sub, result sign, a - a = 0
- **FVRELU**: positive passthrough, negative → 0.0, zero stays zero, negative zero → +0.0, NaN → NaN
- **FVGELU**: x=0 → 0, positive x ≈ x, large negative x ≈ 0, NaN → NaN, +inf → +inf, -inf → 0.0, matches scalar FGELU reference values
- **FVDIV**: basic division, division by large/small facc, IEEE 754 div-by-zero → ±inf, NaN input → NaN, accumulator unchanged after op
- **FVSUB_SCALAR**: basic subtraction, a - 0.0 = a, NaN input → NaN, accumulator unchanged after op

### D5: Firmware Test Program
**Files**: `firmware/npu_vecops/main.c`, `firmware/npu_vecops/Makefile`

C program that exercises each new instruction with known inputs and prints PASS/FAIL per test, with "ALL PASS" at the end (matching the existing `npu_test` firmware pattern). `main.c` should `#include "../common/npu_fp.h"`. The Makefile should declare `main.o: main.c ../common/npu_fp.h` for header dependency tracking.

Test cases:
1. FVADD: [1.0, 2.0, 3.0] + [4.0, 5.0, 6.0] → [5.0, 7.0, 9.0]
2. FVSUB: [10.0, 20.0] - [3.0, 7.0] → [7.0, 13.0]
3. FVRELU: [-1.0, 0.0, 1.0] → [0.0, 0.0, 1.0]
4. FVGELU: [0.0, 1.0, -1.0] → verify against reference
5. FVDIV: [10.0, 20.0] / 5.0 → [2.0, 4.0]
6. FVSUB_SCALAR: [10.0, 20.0] - 3.0 → [7.0, 17.0]

### D6: ISA Reference Update
**File**: `docs/isa-reference.md`

Add entries for all 6 instructions in the FP NPU section, following the existing format. Update the instruction count summary.

### D7: Integration Test
**File**: `tests/integration/test_npu_vecops_firmware.py`

Run compiled `npu_vecops.elf`, capture UART output, verify "ALL PASS" appears and no lines contain "FAIL" (matching the existing `test_npu_firmware.py` pattern).

## Implementation Order

1. D1 — instruction implementations (the core work)
2. D4 — unit tests (verify correctness immediately)
3. D2 — Cython kernels (acceleration, can be done after correctness is proven)
4. D3 — firmware intrinsics
5. D5 — firmware test program
6. D7 — integration test
7. D6 — ISA reference update

## Post-Implementation Updates

Update `.ai/memory.md` instruction counts in the same commit as the code changes:
- "Custom NPU: 30 instructions (14 int opcode 0x0B + 16 FP opcode 0x2B)"
- "Total: 105 instructions"

## Acceptance Criteria

```
uv run pytest tests/npu/test_fp_vector_ops.py -v                    # all pass
uv run pytest tests/integration/test_npu_vecops_firmware.py -v       # all pass
uv run python -m riscv_npu run firmware/npu_vecops/npu_vecops.elf    # prints PASS for each test
uv run pytest                                                        # full suite still passes (1044+ tests)
```
