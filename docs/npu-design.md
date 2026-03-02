# NPU Design

The Neural Processing Unit (NPU) is a custom coprocessor embedded in the RISC-V pipeline. It provides two instruction sets:

- **Integer NPU** (opcode `0x0B`, custom-0): 14 instructions for quantized int8 inference
- **Floating-point NPU** (opcode `0x2B`, custom-1): 10 instructions for FP32 inference

Both share the same architectural pattern — scalar accumulator operations, activation functions, and vectorized memory operations — but target different precision domains.

## Motivation

Neural network inference on a bare RV32IM core requires many general-purpose instructions to perform operations that have predictable, repetitive structure: multiply-accumulate loops, activation functions, and quantization math. The NPU instructions collapse these hot paths into single-cycle operations.

### Integer NPU (quantized int8)

| Bottleneck               | Without NPU                                            | With NPU              |
|--------------------------|--------------------------------------------------------|-----------------------|
| Dot product (N elements) | N multiplies + N adds + bookkeeping                    | 1 × VMAC + RSTACC     |
| ReLU activation          | branch + move                                          | RELU                  |
| GELU activation          | floating-point approximation (not available on RV32IM) | GELU (table lookup)   |
| Quantized multiply       | multiply + shift + sign handling                       | QMUL                  |
| Clamp to int8            | two comparisons + two branches                         | CLAMP                 |
| Softmax (N elements)     | N exp + sum + N div (all float, not available)         | VMAX + VEXP + VREDUCE |
| RMSNorm                  | N sq + sum + sqrt + N div (all float)                  | VREDUCE + VRSQRT      |
| Vector scale             | N multiply + N shift + N clamp                         | VMUL                  |

### Floating-Point NPU (FP32)

| Bottleneck               | Without FP NPU                                 | With FP NPU                    |
|--------------------------|-------------------------------------------------|--------------------------------|
| FP dot product (N elems) | N FMADD.S + loop overhead                       | 1 × FVMAC + FRSTACC            |
| FP ReLU                  | FMAX.S with zero-loaded register                | FRELU                          |
| FP GELU                  | ~20 FP instructions (erf approximation + FMA)   | FGELU                          |
| FP softmax (N elements)  | N exp + sum + N div (scalar loop)               | FVMAX + FVEXP + FVREDUCE       |
| FP RMSNorm               | N sq + sum + sqrt + N mul (scalar loop)         | FVREDUCE + FVRSQRT + FVMUL     |
| FP vector scale          | N FMUL.S in a loop                              | FVMUL                          |

## NPU State

The NPU has internal state separate from the RISC-V register file:

```
Integer accumulator:  acc_hi[31:0]  acc_lo[31:0]    (64-bit signed)
FP accumulator:       facc                           (float64 / double)
Vector registers:     vreg[0..3], each 4 × int8 (-128..127)
```

- The **integer accumulator** is a 64-bit signed register used by MACC/VMAC/RSTACC. It prevents overflow during long multiply-accumulate chains (e.g., a 784-element dot product).
- The **FP accumulator** is a float64 (double-precision) register used by FMACC/FVMAC/FRSTACC. Double precision prevents precision loss when summing many FP32 products; the result is rounded to float32 on readout via FRSTACC.
- The **vector registers** hold packed int8 quartets for bulk load/store between memory and the NPU. They enable efficient weight/activation transfer (DMA-style). FP NPU instructions do not use these registers — they operate on float32 arrays in memory and the f0-f31 float register file directly.

NPU state is initialized to zero and persists across instructions. It is *not* saved/restored by ECALL or MRET (there is no NPU context-switch support).

## Integer Instruction Reference (opcode 0x0B)

All integer NPU instructions use opcode `0x0B`. The `funct3` field selects the operation.

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

### NPU.VEXP -- Vector Exponential (Phase 7)

```
funct3 = 000, funct7 = 0000010    Format: R-type
Syntax: NPU.VEXP rd, rs1, rs2
```

```
n = regs[rd]
for i in 0..n-1:
    mem32[regs[rs2] + i*4] = exp_q16_16(mem32[regs[rs1] + i*4])
```

Reads `n` (from rd) int32 values from memory at address `rs1`, computes the exponential of each Q16.16 fixed-point value, and writes the Q16.16 results to memory at address `rs2`. Used in softmax computation.

**Use case**: Softmax numerator: after subtracting the max, exponentiate each score.

### NPU.VRSQRT -- Vector Reciprocal Square Root (Phase 7)

```
funct3 = 000, funct7 = 0000011    Format: R-type
Syntax: NPU.VRSQRT rd, rs1
```

`rd = rsqrt_q16_16(mem32[regs[rs1]])`

Scalar operation. Reads one int32 from memory at address `rs1`, computes 1/sqrt(x) in Q16.16 fixed-point, and writes the result to register `rd`. Input must be positive; zero or negative inputs saturate to 0x7FFFFFFF.

**Use case**: RMSNorm: compute the scaling factor `1/sqrt(mean(x^2) + eps)`.

### NPU.VMUL -- Vector Scale (Phase 7)

```
funct3 = 000, funct7 = 0000100    Format: R-type
Syntax: NPU.VMUL rd, rs1, rs2
```

```
n = regs[rd]
scale = signed(acc_lo)   // Q16.16 from accumulator
for i in 0..n-1:
    val = sign_extend_8(mem8[regs[rs1] + i])
    result = (val * scale) >> 16
    mem8[regs[rs2] + i] = clamp(result, -128, 127)
```

Scales each int8 element from the source array by a Q16.16 factor taken from the accumulator low word (`acc_lo`), writing clamped int8 results to the destination array. The accumulator is NOT modified.

**Use case**: RMSNorm scaling, attention score normalization.

### NPU.VREDUCE -- Vector Sum Reduction (Phase 7)

```
funct3 = 000, funct7 = 0000101    Format: R-type
Syntax: NPU.VREDUCE rd, rs1, rs2
```

`rd = sum(mem32[regs[rs1] + i*4]) for i in 0..regs[rs2]-1`

Sums `n` signed int32 values from memory. Count is read from register `rs2`, source address from `rs1`. The 32-bit sum is written to `rd`.

**Use case**: Softmax denominator (sum of exp values), RMSNorm mean computation.

### NPU.VMAX -- Vector Max Reduction (Phase 7)

```
funct3 = 000, funct7 = 0000110    Format: R-type
Syntax: NPU.VMAX rd, rs1, rs2
```

`rd = max(mem32[regs[rs1] + i*4]) for i in 0..regs[rs2]-1`

Finds the maximum signed int32 value from `n` elements in memory. Returns 0x80000000 if count is 0.

**Use case**: Softmax numerical stability: subtract the max score before exponentiation.

## Q16.16 Fixed-Point Format (Phase 7)

The Phase 7 vector instructions use Q16.16 signed fixed-point for intermediate precision:

- **Representation**: 32-bit signed integer. Bits [31:16] = integer part, bits [15:0] = fractional part.
- **1.0** = 0x00010000 = 65536.
- **Range**: approximately [-32768.0, +32767.99998].
- **Resolution**: 1/65536 ~ 0.0000153.

This format avoids floating-point hardware while providing sufficient precision for softmax and RMSNorm computations.

## Floating-Point Instruction Reference (opcode 0x2B)

All FP NPU instructions use opcode `0x2B` (RISC-V custom-1 encoding space). The encoding layout mirrors the integer NPU: `funct3` selects the operation group, `funct7` sub-dispatches within `funct3=000`.

FP NPU instructions read/write the **f0-f31 float register file** (from RV32F) and **float32 arrays in memory**. They do not use the integer vector registers (vreg[0..3]).

### Encoding Formats

```
R-type (funct3 0-5):  funct7[31:25]  rs2[24:20]   rs1[19:15]  funct3[14:12]  rd[11:7]   0x2B[6:0]
```

All FP NPU instructions use R-type encoding.

### NPU.FMACC — FP Multiply-Accumulate

```
funct3 = 000, funct7 = 0000000    Format: R-type
Syntax: NPU.FMACC rs1, rs2
```

`facc += f[rs1] × f[rs2]`

Multiplies two single-precision values from float registers, producing a double-precision product, and adds it to the float64 accumulator. The `rd` field is ignored (convention: f0). No register is written.

**Use case**: Scalar FP multiply-accumulate for small or irregular dot products.

### NPU.FVMAC — FP Vector Multiply-Accumulate

```
funct3 = 000, funct7 = 0000001    Format: R-type
Syntax: NPU.FVMAC rd, rs1, rs2
```

```
n = regs[rd]
for i in 0..n-1:
    a = mem_f32[regs[rs1] + i*4]
    b = mem_f32[regs[rs2] + i*4]
    facc += (float64)(a) * (float64)(b)
```

Reads `n` (from integer register rd) pairs of float32 values from memory, multiplies each pair in double precision, and adds all products to the float64 accumulator. Does **not** reset the accumulator, allowing chaining or bias pre-loading.

**Use case**: Entire FP dot product in one instruction. The double-precision accumulator prevents catastrophic cancellation when summing many small products.

### NPU.FRELU — FP Rectified Linear Unit

```
funct3 = 001, funct7 = 0000000    Format: R-type
Syntax: NPU.FRELU rd, rs1
```

`f[rd] = max(f[rs1], +0.0)`

If f[rs1] is negative (including -0.0), writes +0.0 to f[rd]. Otherwise writes f[rs1] unchanged. NaN input produces NaN. The `rs2` field is ignored (convention: f0).

**Use case**: ReLU activation function. While achievable with FMAX.S, FRELU provides semantic clarity and avoids needing a zero-valued register.

### NPU.FGELU — FP Gaussian Error Linear Unit

```
funct3 = 100, funct7 = 0000000    Format: R-type
Syntax: NPU.FGELU rd, rs1
```

`f[rd] = 0.5 × f[rs1] × (1 + erf(f[rs1] / sqrt(2)))`

Computes the exact GELU activation function on a single-precision value. Unlike the integer GELU (table lookup on int8), this operates at full FP32 precision. The `rs2` field is ignored (convention: f0).

**Use case**: GELU activation in transformer models at full precision.

### NPU.FRSTACC — FP Reset Accumulator

```
funct3 = 101, funct7 = 0000000    Format: R-type
Syntax: NPU.FRSTACC rd
```

`f[rd] = (float32)facc; facc = 0.0`

Reads the float64 accumulator, rounds to single precision, writes the result to float register f[rd], then zeroes the accumulator. Both `rs1` and `rs2` are ignored (convention: f0).

**Use case**: End of an FP dot-product chain. After N FMACC or FVMAC operations, FRSTACC retrieves the float32 result and resets for the next vector.

### NPU.FVEXP — FP Vector Exponential

```
funct3 = 000, funct7 = 0000010    Format: R-type
Syntax: NPU.FVEXP rd, rs1, rs2
```

```
n = regs[rd]
for i in 0..n-1:
    mem_f32[regs[rs2] + i*4] = exp(mem_f32[regs[rs1] + i*4])
```

Reads `n` float32 values from memory at address `rs1`, computes exp(x) for each, and writes float32 results to memory at address `rs2`. Source and destination may overlap.

**Use case**: Softmax numerator: after subtracting the max, exponentiate each score.

### NPU.FVRSQRT — FP Reciprocal Square Root

```
funct3 = 000, funct7 = 0000011    Format: R-type
Syntax: NPU.FVRSQRT rd, rs1
```

`f[rd] = 1.0 / sqrt(mem_f32[regs[rs1]])`

Reads one float32 from memory at address `rs1`, computes 1/sqrt(x), and writes the float32 result to f[rd]. Negative input produces NaN. Zero input produces +inf.

**Use case**: RMSNorm: compute the scaling factor `1/sqrt(mean(x²) + eps)`.

### NPU.FVMUL — FP Vector Scale

```
funct3 = 000, funct7 = 0000100    Format: R-type
Syntax: NPU.FVMUL rd, rs1, rs2
```

```
n = regs[rd]
scale = (float32)facc
for i in 0..n-1:
    mem_f32[regs[rs2] + i*4] = mem_f32[regs[rs1] + i*4] × scale
```

Scales each float32 element from the source array by the float32 value of the accumulator (rounded from float64), writing float32 results to the destination array. The accumulator is NOT modified. Source and destination may overlap.

**Use case**: RMSNorm scaling, attention score normalization, softmax division.

### NPU.FVREDUCE — FP Vector Sum Reduction

```
funct3 = 000, funct7 = 0000101    Format: R-type
Syntax: NPU.FVREDUCE rd, rs1, rs2
```

`f[rd] = sum(mem_f32[regs[rs1] + i*4]) for i in 0..regs[rs2]-1`

Sums `n` float32 values from memory using double-precision accumulation internally, then rounds the final result to float32 and writes to f[rd]. Count is read from integer register `rs2`, source address from integer register `rs1`.

**Use case**: Softmax denominator (sum of exp values), RMSNorm mean computation.

### NPU.FVMAX — FP Vector Max Reduction

```
funct3 = 000, funct7 = 0000110    Format: R-type
Syntax: NPU.FVMAX rd, rs1, rs2
```

`f[rd] = max(mem_f32[regs[rs1] + i*4]) for i in 0..regs[rs2]-1`

Finds the maximum float32 value from `n` elements in memory. Returns -inf if count is 0. NaN elements are propagated (any NaN in the input produces NaN output).

**Use case**: Softmax numerical stability: subtract the max score before exponentiation.

## Memory-Mapped Registers

The NPU exposes read-only status registers at base address `0x20000000` for debugger inspection and diagnostic firmware:

| Offset | Size | Name    | Access               | Description                          |
|--------|------|---------|----------------------|--------------------------------------|
| 0x00   | 4    | acc_lo  | R (write resets acc) | Integer accumulator low 32 bits      |
| 0x04   | 4    | acc_hi  | R                    | Integer accumulator high 32 bits     |
| 0x08   | 4    | vreg[0] | R                    | Vector register 0 (4 packed int8)    |
| 0x0C   | 4    | vreg[1] | R                    | Vector register 1                    |
| 0x10   | 4    | vreg[2] | R                    | Vector register 2                    |
| 0x14   | 4    | vreg[3] | R                    | Vector register 3                    |
| 0x18   | 4    | facc_lo | R (write resets)     | FP accumulator low 32 bits (IEEE)    |
| 0x1C   | 4    | facc_hi | R                    | FP accumulator high 32 bits (IEEE)   |

Writing any value to offset 0x00 resets the integer accumulator to zero. Writing any value to offset 0x18 resets the FP accumulator to 0.0. All other writes are ignored.

The FP accumulator is exposed as two 32-bit words containing the raw IEEE 754 double-precision bits (little-endian: facc_lo at lower address).

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

| Intrinsic                    | Signature                              | Instruction  |
|------------------------------|----------------------------------------|--------------|
| `NPU_MACC(a, b)`             | macro, no return                       | NPU.MACC     |
| `NPU_VMAC(a, b, n)`          | macro, no return                       | NPU.VMAC     |
| `NPU_RSTACC()`               | `int32_t NPU_RSTACC(void)`             | NPU.RSTACC   |
| `NPU_RELU(src)`              | `int32_t NPU_RELU(int32_t)`            | NPU.RELU     |
| `NPU_GELU(src)`              | `int32_t NPU_GELU(int32_t)`            | NPU.GELU     |
| `NPU_QMUL(a, b)`             | `int32_t NPU_QMUL(int32_t, int32_t)`   | NPU.QMUL     |
| `NPU_CLAMP(src)`             | `int32_t NPU_CLAMP(int32_t)`           | NPU.CLAMP    |
| `NPU_VEXP(src, dst, n)`      | macro, no return                       | NPU.VEXP     |
| `NPU_VRSQRT(addr)`           | `int32_t NPU_VRSQRT(void *)`           | NPU.VRSQRT   |
| `NPU_VMUL(src, dst, n)`      | macro, no return                       | NPU.VMUL     |
| `NPU_VREDUCE(addr, n)`       | `int32_t NPU_VREDUCE(void *, int)`     | NPU.VREDUCE  |
| `NPU_VMAX(addr, n)`          | `int32_t NPU_VMAX(void *, int)`        | NPU.VMAX     |

LDVEC/STVEC do not have C intrinsics yet.

### FP NPU Intrinsics

FP NPU intrinsics use the `.insn r` directive with opcode `0x2B`. Scalar results go to float registers.

| Intrinsic                       | Signature                              | Instruction  |
|---------------------------------|----------------------------------------|--------------|
| `NPU_FMACC(a, b)`               | macro, no return (float args)          | NPU.FMACC    |
| `NPU_FVMAC(a, b, n)`            | macro, no return                       | NPU.FVMAC    |
| `NPU_FRSTACC()`                 | `float NPU_FRSTACC(void)`              | NPU.FRSTACC  |
| `NPU_FRELU(src)`                | `float NPU_FRELU(float)`               | NPU.FRELU    |
| `NPU_FGELU(src)`                | `float NPU_FGELU(float)`               | NPU.FGELU    |
| `NPU_FVEXP(src, dst, n)`        | macro, no return                       | NPU.FVEXP    |
| `NPU_FVRSQRT(addr)`             | `float NPU_FVRSQRT(void *)`            | NPU.FVRSQRT  |
| `NPU_FVMUL(src, dst, n)`        | macro, no return                       | NPU.FVMUL    |
| `NPU_FVREDUCE(addr, n)`         | `float NPU_FVREDUCE(void *, int)`      | NPU.FVREDUCE |
| `NPU_FVMAX(addr, n)`            | `float NPU_FVMAX(void *, int)`         | NPU.FVMAX    |

## Typical Inference Pipelines

### Quantized (int8) linear layer

`y = relu(W @ x + b)` using integer NPU instructions:

```
for each output neuron i:
    NPU_VMAC(&W[i][0], &x[0], N)   // entire dot product in one instruction
    sum = NPU_RSTACC()             // read result, reset for next neuron
    sum = sum + bias[i]            // add bias (plain ADD)
    sum = sum >> shift             // re-quantize (arithmetic right shift)
    sum = NPU_CLAMP(sum)           // clamp to int8
    y[i] = NPU_RELU(sum)           // activation
```

### FP32 linear layer

`y = gelu(W @ x + b)` using FP NPU instructions:

```
for each output neuron i:
    NPU_FVMAC(&W[i][0], &x[0], N)  // entire FP dot product in one instruction
    sum = NPU_FRSTACC()            // read float32 result, reset accumulator
    sum = sum + bias[i]            // add bias (FADD.S)
    y[i] = NPU_FGELU(sum)          // GELU activation at full precision
```

### FP32 softmax

```
max_val = NPU_FVMAX(scores, N)      // find max for numerical stability
for i in 0..N-1:
    scores[i] -= max_val             // subtract max (FSUB.S loop or prep)
NPU_FVEXP(scores, exp_buf, N)       // vectorized exp()
denom = NPU_FVREDUCE(exp_buf, N)    // sum of exponentials
// store 1/denom in accumulator, then scale:
// facc = 1.0 / denom via FDIV.S + FMACC, then FVMUL
NPU_FVMUL(exp_buf, output, N)       // divide by sum (multiply by 1/sum)
```

### FP32 RMSNorm

```
// sum of squares
NPU_FVMAC(x, x, N)              // facc = sum(x[i]^2)
sum_sq = NPU_FRSTACC()           // read sum, reset accumulator
mean_sq = sum_sq / N + eps       // compute mean + epsilon
scale = NPU_FVRSQRT(&mean_sq)   // 1/sqrt(mean_sq)
// output[i] = input[i] * gamma[i] * scale
for i in 0..N-1:
    output[i] = input[i] * gamma[i] * scale
```

### FP32 Transformer Pipeline

The character-level transformer uses all 10 FP NPU instructions for a complete float32 inference pipeline:

- **Embedding**: Simple float addition (token_embed + pos_embed)
- **Linear layers**: FVMAC + FRSTACC for dot products, FADD.S for bias
- **RMSNorm**: FVMAC (sum of squares), FRSTACC, FVRSQRT (1/sqrt scale)
- **Attention**: FVMAC for Q.K dot products, softmax for attention weights, weighted sum of V
- **Softmax**: FVMAX (numerical stability), FVEXP (exponentiate), FVREDUCE (sum), FMACC + FVMUL (normalize)
- **FFN**: Linear + FGELU activation + Linear
- **Residual**: Simple float addition (no clamping needed)

Model dimensions: vocab=256, embed_dim=64, heads=4, layers=2, context=32, ff_dim=256 (~135K params, ~527KB float32).

Implementation files:
- Python reference: `src/riscv_npu/tools/transformer.py`
- Weight export: `src/riscv_npu/tools/export_transformer_weights.py`
- C firmware: `firmware/transformer/main.c`

## Implementation

| Component                            | File                                                  | Role                         |
|--------------------------------------|-------------------------------------------------------|------------------------------|
| NPU state + accumulator + GELU table | `src/riscv_npu/npu/engine.py`                         | Dataclass and pure functions |
| Integer instruction dispatch         | `src/riscv_npu/npu/instructions.py`                   | Called from CPU execute loop |
| FP instruction dispatch              | `src/riscv_npu/npu/fp_instructions.py`                | Called from CPU execute loop |
| Memory-mapped device                 | `src/riscv_npu/devices/npu.py`                        | Bus-attached register reads  |
| Decoder (opcode 0x0B, 0x2B)          | `src/riscv_npu/cpu/decode.py`                         | R/I/S-type format selection  |
| Integer C intrinsics                 | `firmware/common/npu.h`                               | `.insn` inline assembly      |
| FP C intrinsics                      | `firmware/common/npu_fp.h`                            | `.insn` inline assembly      |
| Float transformer reference          | `src/riscv_npu/tools/transformer.py`                  | Python validation reference  |
| Float weight export                  | `src/riscv_npu/tools/export_transformer_weights.py`   | Train + export float32       |
| Transformer firmware                 | `firmware/transformer/main.c`                         | C firmware using FP NPU      |
