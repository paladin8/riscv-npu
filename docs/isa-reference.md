# ISA Reference

## Registers

### Integer registers

32 × 32-bit general purpose: x0-x31. x0 hardwired to 0 (discard writes in register file, not per-instruction). PC is separate.

ABI names for TUI display: x0=zero, x1=ra, x2=sp, x3=gp, x4=tp, x5-x7=t0-t2, x8=s0/fp, x9=s1, x10-x11=a0-a1, x12-x17=a2-a7, x18-x27=s2-s11, x28-x31=t3-t6.

### Floating-point registers (F extension)

32 × 32-bit floating-point: f0-f31. IEEE 754 single-precision. Plus `fcsr` (FP control/status register) with rounding mode and exception flags (NV, DZ, OF, UF, NX).

ABI names: f0-f7=ft0-ft7, f8-f9=fs0-fs1, f10-f11=fa0-fa1, f12-f17=fa2-fa7, f18-f27=fs2-fs11, f28-f31=ft8-ft11.

## Instruction formats

All 32-bit, little-endian. Bit layouts:

```
R: funct7[31:25]  rs2[24:20]  rs1[19:15]  funct3[14:12]  rd[11:7]   opcode[6:0]
I: imm[31:20]                 rs1[19:15]  funct3[14:12]  rd[11:7]   opcode[6:0]
S: imm[31:25]    rs2[24:20]  rs1[19:15]  funct3[14:12]  imm[11:7]  opcode[6:0]
B: imm[31:25]    rs2[24:20]  rs1[19:15]  funct3[14:12]  imm[11:7]  opcode[6:0]
U: imm[31:12]                                            rd[11:7]   opcode[6:0]
J: imm[31:12]                                            rd[11:7]   opcode[6:0]
```

## Immediate reconstruction

- **I-type**: imm = sign_extend(inst[31:20])
- **S-type**: imm = sign_extend(inst[31:25] << 5 | inst[11:7])
- **B-type**: imm = sign_extend(inst[31] << 12 | inst[7] << 11 | inst[30:25] << 5 | inst[11:8] << 1). LSB is always 0 (not encoded).
- **U-type**: imm = inst[31:12] << 12
- **J-type**: imm = sign_extend(inst[31] << 20 | inst[19:12] << 12 | inst[20] << 11 | inst[30:21] << 1). LSB is always 0.

B-type and J-type bit scrambling is the #1 decoder bug source. Test against known encodings.

## RV32I instructions

### R-type (opcode 0110011)

| funct7  | funct3 | name | op                                        |
|---------|--------|------|-------------------------------------------|
| 0000000 | 000    | ADD  | rd = rs1 + rs2                            |
| 0100000 | 000    | SUB  | rd = rs1 - rs2                            |
| 0000000 | 001    | SLL  | rd = rs1 << rs2[4:0]                      |
| 0000000 | 010    | SLT  | rd = signed(rs1) < signed(rs2) ? 1 : 0    |
| 0000000 | 011    | SLTU | rd = rs1 < rs2 ? 1 : 0 (unsigned)         |
| 0000000 | 100    | XOR  | rd = rs1 ^ rs2                            |
| 0000000 | 101    | SRL  | rd = rs1 >> rs2[4:0] (logical)            |
| 0100000 | 101    | SRA  | rd = signed(rs1) >> rs2[4:0] (arithmetic) |
| 0000000 | 110    | OR   | rd = rs1 \| rs2                           |
| 0000000 | 111    | AND  | rd = rs1 & rs2                            |

### I-type arithmetic (opcode 0010011)

| funct3 | name  | op                                                           |
|--------|-------|--------------------------------------------------------------|
| 000    | ADDI  | rd = rs1 + sext(imm)                                         |
| 010    | SLTI  | rd = signed(rs1) < sext(imm) ? 1 : 0                         |
| 011    | SLTIU | rd = rs1 < sext(imm) ? 1 : 0 (note: sext imm, then unsigned) |
| 100    | XORI  | rd = rs1 ^ sext(imm)                                         |
| 110    | ORI   | rd = rs1 \| sext(imm)                                        |
| 111    | ANDI  | rd = rs1 & sext(imm)                                         |
| 001    | SLLI  | rd = rs1 << imm[4:0] (imm[11:5] must be 0000000)             |
| 101    | SRLI  | rd = rs1 >> imm[4:0] (logical, imm[11:5]=0000000)            |
| 101    | SRAI  | rd = signed(rs1) >> imm[4:0] (arithmetic, imm[11:5]=0100000) |

### Loads (opcode 0000011)

| funct3 | name | op                                |
|--------|------|-----------------------------------|
| 000    | LB   | rd = sext(mem8[rs1 + sext(imm)])  |
| 001    | LH   | rd = sext(mem16[rs1 + sext(imm)]) |
| 010    | LW   | rd = mem32[rs1 + sext(imm)]       |
| 100    | LBU  | rd = zext(mem8[rs1 + sext(imm)])  |
| 101    | LHU  | rd = zext(mem16[rs1 + sext(imm)]) |

Little-endian. Allow misaligned access (no exceptions).

### Stores (opcode 0100011)

| funct3 | name | op                                 |
|--------|------|------------------------------------|
| 000    | SB   | mem8[rs1 + sext(imm)] = rs2[7:0]   |
| 001    | SH   | mem16[rs1 + sext(imm)] = rs2[15:0] |
| 010    | SW   | mem32[rs1 + sext(imm)] = rs2[31:0] |

### Branches (opcode 1100011)

| funct3 | name | condition                  |
|--------|------|----------------------------|
| 000    | BEQ  | rs1 == rs2                 |
| 001    | BNE  | rs1 != rs2                 |
| 100    | BLT  | signed(rs1) < signed(rs2)  |
| 101    | BGE  | signed(rs1) >= signed(rs2) |
| 110    | BLTU | rs1 < rs2 (unsigned)       |
| 111    | BGEU | rs1 >= rs2 (unsigned)      |

If condition true: pc += sext(imm). Offset is relative to branch instruction address, not pc+4.

### Upper immediate

| opcode  | name  | op                    |
|---------|-------|-----------------------|
| 0110111 | LUI   | rd = imm << 12        |
| 0010111 | AUIPC | rd = pc + (imm << 12) |

### Jumps

| opcode  | name | op                                                |
|---------|------|---------------------------------------------------|
| 1101111 | JAL  | rd = pc + 4; pc += sext(imm) [J-type]             |
| 1100111 | JALR | rd = pc + 4; pc = (rs1 + sext(imm)) & ~1 [I-type] |

### System (opcode 1110011, funct3=000, rd=0, rs1=0)

| imm[11:0]    | name   |
|--------------|--------|
| 000000000000 | ECALL  |
| 000000000001 | EBREAK |

### Memory ordering (opcode 0001111)

| funct3 | name  | op                                                                                                                                         |
|--------|-------|--------------------------------------------------------------------------------------------------------------------------------------------|
| 000    | FENCE | No-op in this emulator (single-core, in-order, no reordering). Must still decode without error — compiled C code emits FENCE instructions. |

## M extension

R-type, opcode 0110011, **funct7 = 0000001**.

| funct3 | name   | op                                           |
|--------|--------|----------------------------------------------|
| 000    | MUL    | rd = (rs1 × rs2)[31:0]                       |
| 001    | MULH   | rd = (signed × signed)[63:32]                |
| 010    | MULHSU | rd = (signed × unsigned)[63:32]              |
| 011    | MULHU  | rd = (unsigned × unsigned)[63:32]            |
| 100    | DIV    | rd = signed(rs1) / signed(rs2) (toward zero) |
| 101    | DIVU   | rd = unsigned(rs1) / unsigned(rs2)           |
| 110    | REM    | rd = signed(rs1) % signed(rs2)               |
| 111    | REMU   | rd = unsigned(rs1) % unsigned(rs2)           |

Edge cases: div by zero → DIV returns 0xFFFFFFFF, DIVU returns 0xFFFFFFFF, REM returns rs1, REMU returns rs1. Signed overflow (0x80000000 / 0xFFFFFFFF) → DIV returns 0x80000000, REM returns 0.

## F extension (single-precision float)

All F instructions operate on 32-bit IEEE 754 single-precision values in the f0-f31 register file. NaN handling follows RISC-V spec: signaling NaNs raise the NV (invalid) flag and results are canonicalized to the quiet NaN 0x7FC00000.

### FP load/store

| opcode  | funct3 | name | op                              |
|---------|--------|------|---------------------------------|
| 0000111 | 010    | FLW  | f[rd] = mem32[rs1 + sext(imm)]  |
| 0100111 | 010    | FSW  | mem32[rs1 + sext(imm)] = f[rs2] |

FLW is I-type, FSW is S-type. Both use width=010 (word).

### FP arithmetic (opcode 1010011)

| funct7  | name    | op                           |
|---------|---------|------------------------------|
| 0000000 | FADD.S  | f[rd] = f[rs1] + f[rs2]      |
| 0000100 | FSUB.S  | f[rd] = f[rs1] - f[rs2]      |
| 0001000 | FMUL.S  | f[rd] = f[rs1] × f[rs2]      |
| 0001100 | FDIV.S  | f[rd] = f[rs1] / f[rs2]      |
| 0101100 | FSQRT.S | f[rd] = sqrt(f[rs1]) (rs2=0) |

Division by zero: x/0 = ±inf (DZ flag), 0/0 = NaN (NV flag).

### FP fused multiply-add (R4-type)

| opcode  | name     | op                                  |
|---------|----------|-------------------------------------|
| 1000011 | FMADD.S  | f[rd] = f[rs1] × f[rs2] + f[rs3]    |
| 1000111 | FMSUB.S  | f[rd] = f[rs1] × f[rs2] - f[rs3]    |
| 1001011 | FNMSUB.S | f[rd] = -(f[rs1] × f[rs2]) + f[rs3] |
| 1001111 | FNMADD.S | f[rd] = -(f[rs1] × f[rs2]) - f[rs3] |

R4-type format: rs3 is encoded in bits [31:27] (funct7[6:2]).

### FP sign injection (opcode 1010011, funct7 = 0010000)

| funct3 | name     | op                                      |
|--------|----------|-----------------------------------------|
| 000    | FSGNJ.S  | f[rd] = {sign(rs2), magnitude(rs1)}     |
| 001    | FSGNJN.S | f[rd] = {~sign(rs2), magnitude(rs1)}    |
| 010    | FSGNJX.S | f[rd] = {sign(rs1)^sign(rs2), mag(rs1)} |

Pseudo-instructions: `fmv.s rd,rs` = `fsgnj.s rd,rs,rs`; `fneg.s rd,rs` = `fsgnjn.s rd,rs,rs`; `fabs.s rd,rs` = `fsgnjx.s rd,rs,rs`.

### FP min/max (opcode 1010011, funct7 = 0010100)

| funct3 | name   | op                                       |
|--------|--------|------------------------------------------|
| 000    | FMIN.S | f[rd] = min(f[rs1], f[rs2]); -0.0 < +0.0 |
| 001    | FMAX.S | f[rd] = max(f[rs1], f[rs2]); -0.0 < +0.0 |

If one operand is NaN, the result is the non-NaN operand. Both NaN → canonical NaN.

### FP comparison (opcode 1010011, funct7 = 1010000)

| funct3 | name  | op                                 |
|--------|-------|------------------------------------|
| 010    | FEQ.S | x[rd] = (f[rs1] == f[rs2]) ? 1 : 0 |
| 001    | FLT.S | x[rd] = (f[rs1] < f[rs2]) ? 1 : 0  |
| 000    | FLE.S | x[rd] = (f[rs1] <= f[rs2]) ? 1 : 0 |

Results go to integer register x[rd]. FEQ signals NV only on signaling NaN; FLT/FLE signal NV on any NaN.

### FP conversion (opcode 1010011)

| funct7  | rs2 | name      | op                                |
|---------|-----|-----------|-----------------------------------|
| 1100000 | 0   | FCVT.W.S  | x[rd] = (int32)f[rs1] (truncate)  |
| 1100000 | 1   | FCVT.WU.S | x[rd] = (uint32)f[rs1] (truncate) |
| 1101000 | 0   | FCVT.S.W  | f[rd] = (float)(int32)x[rs1]      |
| 1101000 | 1   | FCVT.S.WU | f[rd] = (float)(uint32)x[rs1]     |

Out-of-range conversions saturate: NaN → INT_MAX (signed) or UINT_MAX (unsigned), ±inf → clamped.

### FP move and classify (opcode 1010011)

| funct7  | funct3 | name     | op                                           |
|---------|--------|----------|----------------------------------------------|
| 1110000 | 000    | FMV.X.W  | x[rd] = f[rs1] (bitwise, float→int reg)      |
| 1111000 | 000    | FMV.W.X  | f[rd] = x[rs1] (bitwise, int→float reg)      |
| 1110000 | 001    | FCLASS.S | x[rd] = 10-bit classification mask of f[rs1] |

FCLASS.S bit assignments: 0=neg inf, 1=neg normal, 2=neg subnormal, 3=neg zero, 4=pos zero, 5=pos subnormal, 6=pos normal, 7=pos inf, 8=signaling NaN, 9=quiet NaN.

## Custom NPU instructions

Opcode `0x0B` (custom-0 space).

### NPU internal state (separate from register file)
- acc_lo, acc_hi: 32-bit halves of 64-bit accumulator
- vreg[0..3]: four registers, each 4 × int8

### R-type compute (opcode 0x0B)

| funct7  | funct3 | name        | op                                                    |
|---------|--------|-------------|-------------------------------------------------------|
| 0000000 | 000    | NPU.MACC    | {acc_hi,acc_lo} += signed(rs1) × signed(rs2)          |
| 0000001 | 000    | NPU.VMAC    | acc += dot(mem_i8[rs1..+rd], mem_i8[rs2..+rd])        |
| 0000010 | 000    | NPU.VEXP    | mem_i32[rs2+i*4] = exp(mem_i32[rs1+i*4]), Q16.16      |
| 0000011 | 000    | NPU.VRSQRT  | rd = 1/sqrt(mem_i32[rs1]), Q16.16                     |
| 0000100 | 000    | NPU.VMUL    | mem_i8[rs2+i] = clamp((mem_i8[rs1+i] * acc_lo) >> 16) |
| 0000101 | 000    | NPU.VREDUCE | rd = sum(mem_i32[rs1+i*4]), i in 0..rs2-1             |
| 0000110 | 000    | NPU.VMAX    | rd = max(mem_i32[rs1+i*4]), i in 0..rs2-1             |
| 0000000 | 001    | NPU.RELU    | rd = max(signed(rs1), 0)                              |
| 0000000 | 010    | NPU.QMUL    | rd = (signed(rs1) × signed(rs2)) >> 8                 |
| 0000000 | 011    | NPU.CLAMP   | rd = clamp(signed(rs1), -128, 127)                    |
| 0000000 | 100    | NPU.GELU    | rd = gelu_approx(rs1) via lookup table                |
| 0000000 | 101    | NPU.RSTACC  | rd = acc_lo; acc = 0                                  |

### I-type data movement (opcode 0x0B)

| funct3 | name      | op                                             |
|--------|-----------|------------------------------------------------|
| 110    | NPU.LDVEC | vreg[rd%4] = mem32[rs1 + sext(imm)] as 4×int8  |
| 111    | NPU.STVEC | mem32[rs1 + sext(imm)] = vreg[rs2%4] as 4×int8 |
