# NPU Design

The Neural Processing Unit (NPU) is a custom coprocessor embedded in the RISC-V pipeline. It adds 9 instructions under opcode `0x0B` (the RISC-V custom-0 encoding space) that accelerate quantized int8 neural network inference.

## Motivation

Neural network inference on a bare RV32IM core requires many general-purpose instructions to perform operations that have predictable, repetitive structure: multiply-accumulate loops, activation functions, and quantization math. The NPU instructions collapse these hot paths into single-cycle operations.

| Bottleneck               | Without NPU                                            | With NPU            |
|--------------------------|--------------------------------------------------------|---------------------|
| Dot product (N elements) | N multiplies + N adds + bookkeeping                    | 1 × VMAC + RSTACC   |
| ReLU activation          | branch + move                                          | RELU                |
| GELU activation          | floating-point approximation (not available on RV32IM) | GELU (table lookup) |
| Quantized multiply       | multiply + shift + sign handling                       | QMUL                |
| Clamp to int8            | two comparisons + two branches                         | CLAMP               |

## NPU State

The NPU has internal state separate from the RISC-V register file:

```
64-bit accumulator:   acc_hi[31:0]  acc_lo[31:0]
Vector registers:     vreg[0..3], each 4 × int8 (-128..127)
```

- The **accumulator** is a 64-bit signed register used by MACC/VMAC/RSTACC. It prevents overflow during long multiply-accumulate chains (e.g., a 784-element dot product).
- The **vector registers** hold packed int8 quartets for bulk load/store between memory and the NPU. They enable efficient weight/activation transfer in later phases (DMA-style).

NPU state is initialized to zero and persists across instructions. It is *not* saved/restored by ECALL or MRET (there is no NPU context-switch support).

## Instruction Reference

All NPU instructions use opcode `0x0B`. The `funct3` field selects the operation.

### Encoding Formats

```
R-type (funct3 0-5):  funct7[31:25]  rs2[24:20]   rs1[19:15]  funct3[14:12]  rd[11:7]   0x0B[6:0]
I-type (funct3 6):    imm[31:20]                  rs1[19:15]  110[14:12]     rd[11:7]   0x0B[6:0]
S-type (funct3 7):    imm[31:25]     rs2[24:20]   rs1[19:15]  111[14:12]     imm[11:7]  0x0B[6:0]
```

### NPU.MACC — Multiply-Accumulate

```
funct3 = 000    Format: R-type
Syntax: NPU.MACC rs1, rs2
```

`{acc_hi, acc_lo} += signed(rs1) * signed(rs2)`

Multiplies two signed 32-bit register values, producing a 64-bit product, and adds it to the 64-bit accumulator. The `rd` field is ignored (convention: x0). No general-purpose register is written.

**Use case**: Scalar multiply-accumulate for small or irregular dot products.

### NPU.VMAC — Vector Multiply-Accumulate

```
funct3 = 000, funct7 = 0000001    Format: R-type
Syntax: NPU.VMAC rd, rs1, rs2
```

```
n = regs[rd]
for i in 0..n-1:
    a = sign_extend_8(mem[regs[rs1] + i])
    b = sign_extend_8(mem[regs[rs2] + i])
    {acc_hi, acc_lo} += a * b
```

Reads `n` (from rd) pairs of signed int8 bytes from memory at addresses `rs1` and `rs2`, multiplies each pair, and adds all products to the 64-bit accumulator. Does **not** reset the accumulator, allowing chaining or bias pre-loading.

Both arrays are treated as signed int8. For unsigned uint8 inputs (e.g., pixel values), the caller should pre-convert to signed and adjust biases: `bias' = bias + 128 * sum(weights_row)`.

**Use case**: Entire dot product in one instruction. Replaces the scalar MACC loop, reducing a 784-iteration inner loop from ~3,920 instructions to ~5.

### NPU.RELU — Rectified Linear Unit

```
funct3 = 001    Format: R-type
Syntax: NPU.RELU rd, rs1
```

`rd = max(signed(rs1), 0)`

If the signed interpretation of rs1 is negative, writes 0 to rd. Otherwise writes rs1 unchanged. The `rs2` field is ignored (convention: x0).

**Use case**: ReLU activation function applied element-wise after a linear layer.

### NPU.QMUL — Quantized Multiply

```
funct3 = 010    Format: R-type
Syntax: NPU.QMUL rd, rs1, rs2
```

`rd = (signed(rs1) * signed(rs2)) >> 8`

Signed multiply followed by an arithmetic right shift by 8 bits. This is the standard fixed-point multiply for int8 quantized inference where values are scaled by 2^8.

**Use case**: Scaling operations in quantized networks — multiply an activation by a scale factor, then shift back to int8 range.

### NPU.CLAMP — Clamp to Int8

```
funct3 = 011    Format: R-type
Syntax: NPU.CLAMP rd, rs1
```

`rd = clamp(signed(rs1), -128, 127)`

Clamps the signed 32-bit value to the int8 range [-128, 127]. The result is stored as a 32-bit value (sign-extended). The `rs2` field is ignored (convention: x0).

**Use case**: Re-quantization after accumulation or scaling — ensures the result fits in int8 before storing to a weight/activation buffer.

### NPU.GELU — Gaussian Error Linear Unit

```
funct3 = 100    Format: R-type
Syntax: NPU.GELU rd, rs1
```

`rd = gelu_table[rs1[7:0]]`

Reads the low 8 bits of rs1 as a signed int8 value, looks it up in a precomputed 256-entry GELU table, and writes the int8 result (sign-extended to 32 bits) to rd. The `rs2` field is ignored (convention: x0).

The table is computed from the exact GELU formula:

```
gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

Input is scaled by 1/32 (int8 range [-128,127] maps to approximately [-4, +4] in float), GELU is applied, and the result is scaled back and clamped to int8.

**Use case**: GELU activation in transformer models (used by GPT, BERT, etc.). A lookup table avoids the need for floating-point math entirely.

### NPU.RSTACC — Reset Accumulator

```
funct3 = 101    Format: R-type
Syntax: NPU.RSTACC rd
```

`rd = acc_lo; {acc_hi, acc_lo} = 0`

Reads the lower 32 bits of the accumulator into rd, then zeroes the entire 64-bit accumulator. Both `rs1` and `rs2` are ignored (convention: x0).

**Use case**: End of a dot-product chain. After N MACC operations, RSTACC retrieves the result and resets for the next vector.

### NPU.LDVEC — Load Vector Register

```
funct3 = 110    Format: I-type
Syntax: NPU.LDVEC vrd, offset(rs1)
```

`vreg[rd % 4] = mem[rs1 + sext(imm)..+3] as 4 × int8`

Loads 4 consecutive bytes from memory into a vector register. Bytes are interpreted as signed int8 values. The destination vector register is selected by `rd % 4`.

**Use case**: Loading a quartet of int8 weights or activations for vectorized processing in later phases.

### NPU.STVEC — Store Vector Register

```
funct3 = 111    Format: S-type
Syntax: NPU.STVEC vrs2, offset(rs1)
```

`mem[rs1 + sext(imm)..+3] = vreg[rs2 % 4] as 4 × int8`

Stores 4 int8 values from a vector register to consecutive memory bytes. The source vector register is selected by `rs2 % 4`. Each int8 is written as a single unsigned byte.

**Use case**: Writing back computed activations to memory.

## Memory-Mapped Registers

The NPU exposes read-only status registers at base address `0x20000000` for debugger inspection and diagnostic firmware:

| Offset | Size | Name    | Access               | Description                       |
|--------|------|---------|----------------------|-----------------------------------|
| 0x00   | 4    | acc_lo  | R (write resets acc) | Accumulator low 32 bits           |
| 0x04   | 4    | acc_hi  | R                    | Accumulator high 32 bits          |
| 0x08   | 4    | vreg[0] | R                    | Vector register 0 (4 packed int8) |
| 0x0C   | 4    | vreg[1] | R                    | Vector register 1                 |
| 0x10   | 4    | vreg[2] | R                    | Vector register 2                 |
| 0x14   | 4    | vreg[3] | R                    | Vector register 3                 |

Writing any value to offset 0x00 resets the accumulator to zero. All other writes are ignored.

## C Intrinsics

The header `firmware/common/npu.h` provides inline assembly wrappers using the `.insn` directive:

```c
#include "npu.h"

// Dot product of two arrays
int32_t dot(int32_t *a, int32_t *b, int n) {
    for (int i = 0; i < n; i++)
        NPU_MACC(a[i], b[i]);
    return NPU_RSTACC();
}

// Activations
int32_t x = NPU_RELU(val);
int32_t y = NPU_GELU(val);
int32_t z = NPU_QMUL(a, scale);
int32_t c = NPU_CLAMP(wide_val);
```

Available macros/functions:

| Intrinsic                   | Signature                            | Instruction |
|-----------------------------|--------------------------------------|-------------|
| `NPU_MACC(a, b)`            | macro, no return                     | NPU.MACC    |
| `NPU_VMAC(a, b, n)`         | macro, no return                     | NPU.VMAC    |
| `NPU_RSTACC()`              | `int32_t NPU_RSTACC(void)`           | NPU.RSTACC  |
| `NPU_RELU(src)`             | `int32_t NPU_RELU(int32_t)`          | NPU.RELU    |
| `NPU_GELU(src)`             | `int32_t NPU_GELU(int32_t)`          | NPU.GELU    |
| `NPU_QMUL(a, b)`            | `int32_t NPU_QMUL(int32_t, int32_t)` | NPU.QMUL    |
| `NPU_CLAMP(src)`            | `int32_t NPU_CLAMP(int32_t)`         | NPU.CLAMP   |

LDVEC/STVEC do not have C intrinsics yet (planned for vectorized firmware in later phases).

## Typical Inference Pipeline

A quantized linear layer `y = relu(W @ x + b)` using NPU instructions:

```
for each output neuron i:
    NPU_VMAC(&W[i][0], &x[0], N)   // entire dot product in one instruction
    sum = NPU_RSTACC()             // read result, reset for next neuron
    sum = sum + bias[i]            // add bias (plain ADD)
    sum = sum >> shift             // re-quantize (arithmetic right shift)
    sum = NPU_CLAMP(sum)           // clamp to int8
    y[i] = NPU_RELU(sum)           // activation
```

## Implementation

| Component                            | File                                | Role                         |
|--------------------------------------|-------------------------------------|------------------------------|
| NPU state + accumulator + GELU table | `src/riscv_npu/npu/engine.py`       | Dataclass and pure functions |
| Instruction dispatch + semantics     | `src/riscv_npu/npu/instructions.py` | Called from CPU execute loop |
| Memory-mapped device                 | `src/riscv_npu/devices/npu.py`      | Bus-attached register reads  |
| Decoder (opcode 0x0B)                | `src/riscv_npu/cpu/decode.py`       | R/I/S-type format selection  |
| C intrinsics                         | `firmware/common/npu.h`             | `.insn` inline assembly      |
