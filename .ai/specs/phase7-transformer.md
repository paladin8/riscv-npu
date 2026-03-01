# Phase 7: Transformer Extension

## Goal
Tiny character-level transformer runs on emulator. Adds 5 new NPU vector instructions to accelerate transformer operations (softmax, RMSNorm, elementwise ops).

## Design Decisions

1. **Task**: Character-level language model (predict next character). Byte-level vocab (256 entries), no tokenizer needed.
2. **Normalization**: RMSNorm (not LayerNorm). Simpler -- no mean subtraction, just scale by 1/sqrt(mean(x^2)). Key operation is reciprocal square root (VRSQRT).
3. **New NPU instructions**: 5 vector primitives under funct3=0 with new funct7 values (accelerator-style, extending the MACC/VMAC pattern):
   - VEXP (funct7=2) -- vector exponential approximation (for softmax)
   - VRSQRT (funct7=3) -- vector reciprocal square root (for RMSNorm)
   - VMUL (funct7=4) -- vector elementwise multiply
   - VREDUCE (funct7=5) -- vector sum reduction
   - VMAX (funct7=6) -- vector max reduction
4. **Precision**: int8 weights, int32 intermediate values. VMAC accumulates Q*K^T in int32. Requantize to int8 only after softmax for V projection. Softmax operates on int32 scores, outputs int8 probabilities.
5. **Model**: Embedding dim 64, 4 heads (head_dim=16), 2 layers, byte-level vocab (256), context 32 tokens, ~200K params, ~200KB int8.
6. **Fixed-point format**: New vector ops use Q16.16 fixed-point for intermediate precision. VEXP input/output are Q16.16. VRSQRT input is Q16.16 (mean of squares), output is Q16.16 scale factor. VMUL multiplies int8 * Q16.16 scale -> int8 result (with shift).
7. **Softmax strategy**: Compute max (VMAX), subtract max (prevents overflow), exponentiate (VEXP), sum (VREDUCE), divide by sum (multiply by reciprocal via VMUL variant). All in int32/Q16.16 domain. Final output quantized to uint8 (0-255 representing 0.0-1.0).

## Instruction Encoding

All new instructions: opcode=0x0B, funct3=0, R-type format.

| funct7 | Name       | Semantics                                                        | Operands                    |
|--------|------------|------------------------------------------------------------------|-----------------------------|
| 0      | NPU.MACC   | acc += signed(rs1) * signed(rs2) (existing)                      | rs1=val, rs2=val            |
| 1      | NPU.VMAC   | acc += dot(mem_i8[rs1..+rd], mem_i8[rs2..+rd]) (existing)        | rd=count, rs1=addr, rs2=addr|
| 2      | NPU.VEXP   | mem_i32[rs2+i*4] = exp_approx(mem_i32[rs1+i*4]) for i in 0..rd-1 | rd=count, rs1=src, rs2=dst  |
| 3      | NPU.VRSQRT | rd = rsqrt_approx(mem_i32[rs1]) (scalar, Q16.16)                 | rd=result, rs1=addr         |
| 4      | NPU.VMUL   | mem_i8[rs2+i] = clamp((mem_i8[rs1+i] * acc_lo) >> 16, i<rd)     | rd=count, rs1=src, rs2=dst  |
| 5      | NPU.VREDUCE| rd = sum(mem_i32[rs1+i*4]) for i in 0..rs2-1                     | rd=result, rs1=addr, rs2=count|
| 6      | NPU.VMAX   | rd = max(mem_i32[rs1+i*4]) for i in 0..rs2-1                    | rd=result, rs1=addr, rs2=count|

### Detail: VEXP (funct7=2)
- Reads rd as element count, rs1 as source address, rs2 as destination address.
- For each element: reads int32 from mem[rs1+i*4] as Q16.16 fixed-point input.
- Computes exp(x) in Q16.16 fixed-point using polynomial approximation.
- Input range: approximately [-8.0, 0.0] in Q16.16 (values after max subtraction for softmax).
- Writes int32 Q16.16 result to mem[rs2+i*4].
- Approximation: exp(x) ~ polynomial or lookup-based, sufficient for softmax normalization.

### Detail: VRSQRT (funct7=3)
- Scalar operation. Reads one int32 from mem[rs1] as Q16.16 (the mean of squared values).
- Computes 1/sqrt(x) in Q16.16 fixed-point.
- Result written to register rd as Q16.16 int32.
- Used in RMSNorm: scale = rsqrt(mean(x^2) + eps).

### Detail: VMUL (funct7=4)
- Reads rd as element count (from register), rs1 as source int8 array address, rs2 as destination int8 array address.
- Reads a Q16.16 scale factor from the accumulator low word (acc_lo).
- For each element i in 0..n-1: dst[i] = clamp((src[i] * acc_lo) >> 16, -128, 127).
- src[i] is read as signed int8 from memory. Result is written as unsigned byte.
- The accumulator is NOT modified by this instruction.
- Used for RMSNorm scaling and attention score normalization.

### Detail: VREDUCE (funct7=5)
- Reads rs2 as element count (from register), rs1 as source address.
- Sums all int32 values from mem[rs1+i*4] for i in 0..rs2-1.
- Result (int32) written to register rd.
- Used in softmax denominator and RMSNorm mean computation.

### Detail: VMAX (funct7=6)
- Reads rs2 as element count (from register), rs1 as source address.
- Finds maximum int32 value from mem[rs1+i*4] for i in 0..rs2-1.
- Result (int32) written to register rd.
- Used in softmax numerically-stable computation (subtract max before exp).

## Deliverables List

1. **NPU instruction implementations** (engine + instructions + decode + disasm)
   - VEXP, VRSQRT, VMUL, VREDUCE, VMAX
   - Unit tests for each instruction
2. **C intrinsics** (firmware/common/npu.h)
   - Inline assembly wrappers for 5 new instructions
3. **Python quantized transformer inference** (src/riscv_npu/npu/transformer.py)
   - Pure-Python reference implementation matching firmware behavior
   - Unit tests
4. **Transformer model trainer and weight exporter** (tools/export_transformer_weights.py)
   - Train tiny char-level transformer on simple text corpus
   - Quantize weights to int8
   - Export C header and Python test data
5. **C firmware** (firmware/transformer/)
   - main.c implementing full transformer forward pass
   - Makefile for cross-compilation
6. **Integration tests** (tests/integration/test_transformer.py)
   - Run firmware on emulator, verify output matches Python reference
7. **Documentation updates**
   - docs/npu-design.md: new instructions
   - docs/isa-reference.md: new instruction table entries

## Implementation Details

### Deliverable 1: NPU Instructions

**Files modified:**
- `src/riscv_npu/cpu/decode.py`: Extend OP_NPU funct3==0 case to pass through all funct7 values (already works -- R-type decode captures funct7).
- `src/riscv_npu/npu/engine.py`: Add `build_exp_table()` for VEXP lookup, `fixed_rsqrt()` for VRSQRT.
- `src/riscv_npu/npu/instructions.py`: Add `_exec_vexp()`, `_exec_vrsqrt()`, `_exec_vmul()`, `_exec_vreduce()`, `_exec_vmax()`. Dispatch from funct7 values 2-6 within funct3==0 handler.
- `src/riscv_npu/tui/disasm.py`: Add disassembly mnemonics for new instructions.

**Key functions:**

```python
# engine.py
def exp_q16_16(x: int) -> int:
    """Compute exp(x) where x is Q16.16 fixed-point. Returns Q16.16."""

def rsqrt_q16_16(x: int) -> int:
    """Compute 1/sqrt(x) where x is Q16.16 fixed-point. Returns Q16.16."""

# instructions.py
def _exec_vexp(inst, regs, mem, npu) -> None:
    """VEXP: vectorized exp over int32 Q16.16 array."""

def _exec_vrsqrt(inst, regs, mem, npu) -> None:
    """VRSQRT: scalar reciprocal sqrt, Q16.16."""

def _exec_vmul(inst, regs, mem, npu) -> None:
    """VMUL: scale int8 vector by Q16.16 factor from accumulator."""

def _exec_vreduce(inst, regs, mem) -> None:
    """VREDUCE: sum int32 array into rd."""

def _exec_vmax(inst, regs, mem) -> None:
    """VMAX: max of int32 array into rd."""
```

**Q16.16 format**: 32-bit signed integer where bits [31:16] are the integer part and bits [15:0] are the fractional part. 1.0 = 0x00010000 = 65536. Range: approximately [-32768, 32767.99998].

**VEXP approximation**: Use piece-wise polynomial or lookup table. For softmax, inputs are in [-8, 0] range (after max subtraction). exp(-8) ~ 0.000335 in Q16.16 = ~22. exp(0) = 1.0 = 65536. A 256-entry lookup table with linear interpolation gives sufficient precision.

**VRSQRT approximation**: Newton-Raphson with a good initial guess. For RMSNorm, input is always positive. rsqrt(x) = 1/sqrt(x). One or two Newton iterations from a lookup-based initial estimate.

### Deliverable 2: C Intrinsics

**File modified:** `firmware/common/npu.h`

```c
// VEXP: dst[i] = exp(src[i]) for i in 0..n-1, Q16.16 fixed-point
#define NPU_VEXP(src, dst, n) ...

// VRSQRT: returns 1/sqrt(mem[addr]) in Q16.16
static inline int32_t NPU_VRSQRT(void *addr) ...

// VMUL: dst[i] = clamp((src[i] * acc_lo) >> 16, -128, 127) for i in 0..n-1
#define NPU_VMUL(src, dst, n) ...

// VREDUCE: returns sum of int32 array
static inline int32_t NPU_VREDUCE(void *addr, int n) ...

// VMAX: returns max of int32 array
static inline int32_t NPU_VMAX(void *addr, int n) ...
```

### Deliverable 3: Python Transformer Reference

**File:** `src/riscv_npu/npu/transformer.py`

Pure-Python quantized transformer forward pass. All operations match what the firmware will do: int8 weights, int32 intermediates, Q16.16 for softmax/norm.

Key functions:
```python
def rmsnorm_q(x_i8: list[int], gamma_i8: list[int], dim: int) -> list[int]:
    """RMSNorm in quantized domain."""

def attention_q(q_i8, k_i8, v_i8, n_tokens, head_dim) -> list[int]:
    """Single-head attention in quantized domain."""

def transformer_block_q(x_i8, weights, n_tokens, dim, n_heads) -> list[int]:
    """One transformer block: attn + ffn with residual."""

def transformer_forward_q(tokens, weights, config) -> list[int]:
    """Full transformer: embedding -> blocks -> output logits."""
```

### Deliverable 4: Weight Exporter

**File:** `tools/export_transformer_weights.py`

Trains on a simple text corpus (e.g., Shakespeare or generated patterns). Exports:
- `firmware/transformer/weights.h`: C arrays for all weights
- `firmware/transformer/test_data.py`: test input sequences + expected outputs

### Deliverable 5: C Firmware

**File:** `firmware/transformer/main.c`

Transformer forward pass in C using NPU instructions. Reads input tokens from a buffer, runs one forward pass, outputs predicted next token via UART/stdout.

### Deliverable 6: Integration Tests

**File:** `tests/integration/test_transformer.py`

Same pattern as test_mnist.py: load ELF, inject test data, run CPU, compare output to Python reference.

## Test Coverage Requirements

### Deliverable 1 tests (in tests/npu/test_instructions.py and tests/npu/test_engine.py):

**VEXP tests:**
- `test_vexp_zero`: exp(0) = 1.0 in Q16.16 = 65536
- `test_vexp_negative`: exp(-1.0) ~ 0.368 in Q16.16 ~ 24109
- `test_vexp_large_negative`: exp(-8.0) ~ 0.000335, should be small positive
- `test_vexp_multiple_elements`: vector of 4 elements, verify each
- `test_vexp_zero_count`: n=0 does nothing

**VRSQRT tests:**
- `test_vrsqrt_one`: rsqrt(1.0) = 1.0 in Q16.16
- `test_vrsqrt_four`: rsqrt(4.0) = 0.5 in Q16.16
- `test_vrsqrt_quarter`: rsqrt(0.25) = 2.0 in Q16.16
- `test_vrsqrt_large`: rsqrt(100.0) = 0.1 in Q16.16

**VMUL tests:**
- `test_vmul_identity`: scale=1.0 (acc_lo=65536), values pass through
- `test_vmul_half`: scale=0.5, values halved
- `test_vmul_negative_scale`: negative scale factor
- `test_vmul_clamps_to_int8`: overflow clamped to [-128, 127]
- `test_vmul_zero_count`: n=0 does nothing

**VREDUCE tests:**
- `test_vreduce_basic`: sum of [1, 2, 3, 4] = 10
- `test_vreduce_negative`: sum including negatives
- `test_vreduce_single`: single element
- `test_vreduce_zero_count`: n=0 returns 0

**VMAX tests:**
- `test_vmax_basic`: max of [1, 5, 3, 2] = 5
- `test_vmax_all_negative`: max of [-3, -1, -5] = -1
- `test_vmax_single`: single element
- `test_vmax_zero_count`: n=0 returns minimum int32

### Deliverable 3 tests (in tests/npu/test_transformer.py):
- `test_rmsnorm_identity`: all-ones input, gamma=1
- `test_rmsnorm_scaling`: verify correct normalization
- `test_attention_single_head`: small known Q, K, V matrices
- `test_softmax_q`: verify softmax sums to ~255 (uint8 scale)
- `test_transformer_forward_deterministic`: same input -> same output

## Acceptance Criteria

1. `uv run pytest` passes with all existing + new tests
2. `uv run pytest --co -q | tail -1` shows increased test count (>564)
3. New NPU instructions work correctly through the CPU step pipeline
4. Python transformer reference produces deterministic output for fixed weights
5. (Stretch) Firmware compiles and runs on emulator, matching Python reference
