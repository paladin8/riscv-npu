"""Instruction decoder: extracts fields and reconstructs immediates."""

from dataclasses import dataclass


def sign_extend(value: int, bits: int) -> int:
    """Sign-extend a `bits`-wide value to 32 bits."""
    sign_bit = 1 << (bits - 1)
    return ((value ^ sign_bit) - sign_bit) & 0xFFFFFFFF


def to_signed(value: int) -> int:
    """Interpret a 32-bit unsigned value as signed Python int."""
    return value - 0x100000000 if value >= 0x80000000 else value


@dataclass(frozen=True)
class Instruction:
    """Decoded RISC-V instruction."""

    opcode: int
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0
    funct3: int = 0
    funct7: int = 0


# Opcode constants
OP_R_TYPE = 0x33
OP_I_ARITH = 0x13
OP_LOAD = 0x03
OP_STORE = 0x23
OP_BRANCH = 0x63
OP_LUI = 0x37
OP_AUIPC = 0x17
OP_JAL = 0x6F
OP_JALR = 0x67
OP_SYSTEM = 0x73
OP_FENCE = 0x0F
OP_NPU = 0x0B


def decode(word: int) -> Instruction:
    """Decode a 32-bit instruction word into an Instruction."""
    opcode = word & 0x7F
    rd = (word >> 7) & 0x1F
    funct3 = (word >> 12) & 0x7
    rs1 = (word >> 15) & 0x1F
    rs2 = (word >> 20) & 0x1F
    funct7 = (word >> 25) & 0x7F

    if opcode == OP_R_TYPE:
        # R-type: no immediate
        return Instruction(opcode=opcode, rd=rd, rs1=rs1, rs2=rs2,
                           funct3=funct3, funct7=funct7)

    elif opcode == OP_NPU:
        # Custom NPU instructions (opcode 0x0B)
        # funct3 0-5: R-type compute (rd, rs1, rs2)
        # funct3 6: LDVEC - I-type (rd, rs1, imm)
        # funct3 7: STVEC - S-type like (rs1, rs2, imm)
        if funct3 <= 5:
            return Instruction(opcode=opcode, rd=rd, rs1=rs1, rs2=rs2,
                               funct3=funct3, funct7=funct7)
        elif funct3 == 6:  # LDVEC: I-type
            imm = sign_extend(word >> 20, 12)
            return Instruction(opcode=opcode, rd=rd, rs1=rs1, imm=imm,
                               funct3=funct3, funct7=funct7)
        else:  # funct3 == 7, STVEC: S-type
            imm = sign_extend((funct7 << 5) | rd, 12)
            return Instruction(opcode=opcode, rs1=rs1, rs2=rs2, imm=imm,
                               funct3=funct3, funct7=funct7)

    elif opcode in (OP_I_ARITH, OP_LOAD, OP_JALR, OP_SYSTEM, OP_FENCE):
        # I-type: imm = sign_extend(inst[31:20], 12)
        imm = sign_extend(word >> 20, 12)
        return Instruction(opcode=opcode, rd=rd, rs1=rs1, imm=imm,
                           funct3=funct3, funct7=funct7)

    elif opcode == OP_STORE:
        # S-type: imm = sign_extend(inst[31:25] << 5 | inst[11:7], 12)
        imm = sign_extend((funct7 << 5) | rd, 12)
        return Instruction(opcode=opcode, rs1=rs1, rs2=rs2, imm=imm,
                           funct3=funct3, funct7=funct7)

    elif opcode == OP_BRANCH:
        # B-type: scrambled bits, LSB always 0
        imm = sign_extend(
            ((word >> 31) & 1) << 12
            | ((word >> 7) & 1) << 11
            | ((word >> 25) & 0x3F) << 5
            | ((word >> 8) & 0xF) << 1,
            13,
        )
        return Instruction(opcode=opcode, rs1=rs1, rs2=rs2, imm=imm,
                           funct3=funct3, funct7=funct7)

    elif opcode in (OP_LUI, OP_AUIPC):
        # U-type: imm = inst[31:12] << 12 (already in upper position)
        imm = word & 0xFFFFF000
        return Instruction(opcode=opcode, rd=rd, imm=imm)

    elif opcode == OP_JAL:
        # J-type: scrambled bits, LSB always 0
        imm = sign_extend(
            ((word >> 31) & 1) << 20
            | ((word >> 12) & 0xFF) << 12
            | ((word >> 20) & 1) << 11
            | ((word >> 21) & 0x3FF) << 1,
            21,
        )
        return Instruction(opcode=opcode, rd=rd, imm=imm)

    else:
        raise ValueError(f"Unknown opcode: 0x{opcode:02X} (word=0x{word:08X})")
