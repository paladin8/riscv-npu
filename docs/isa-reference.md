# ISA Reference

## Registers

32 × 32-bit general purpose: x0-x31. x0 hardwired to 0 (discard writes in register file, not per-instruction). PC is separate.

ABI names for TUI display: x0=zero, x1=ra, x2=sp, x3=gp, x4=tp, x5-x7=t0-t2, x8=s0/fp, x9=s1, x10-x11=a0-a1, x12-x17=a2-a7, x18-x27=s2-s11, x28-x31=t3-t6.

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

## Custom NPU instructions

Opcode `0x0B` (custom-0 space).

### NPU internal state (separate from register file)
- acc_lo, acc_hi: 32-bit halves of 64-bit accumulator
- vreg[0..3]: four registers, each 4 × int8

### R-type compute (opcode 0x0B)

| funct7  | funct3 | name       | op                                                      |
|---------|--------|------------|---------------------------------------------------------|
| 0000000 | 000    | NPU.MACC   | {acc_hi,acc_lo} += signed(rs1) × signed(rs2)            |
| 0000001 | 000    | NPU.VMAC   | acc += dot(mem_int8[rs1..+rd], mem_int8[rs2..+rd])      |
| 0000000 | 001    | NPU.RELU   | rd = max(signed(rs1), 0)                                |
| 0000000 | 010    | NPU.QMUL   | rd = (signed(rs1) × signed(rs2)) >> 8                   |
| 0000000 | 011    | NPU.CLAMP  | rd = clamp(signed(rs1), -128, 127)                      |
| 0000000 | 100    | NPU.GELU   | rd = gelu_approx(rs1) via lookup table                  |
| 0000000 | 101    | NPU.RSTACC | rd = acc_lo; acc = 0                                    |

### I-type data movement (opcode 0x0B)

| funct3 | name      | op                                             |
|--------|-----------|------------------------------------------------|
| 110    | NPU.LDVEC | vreg[rd%4] = mem32[rs1 + sext(imm)] as 4×int8  |
| 111    | NPU.STVEC | mem32[rs1 + sext(imm)] = vreg[rs2%4] as 4×int8 |
