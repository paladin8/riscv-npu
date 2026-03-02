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
    rs3: int = 0
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
OP_FP_NPU = 0x2B
OP_LOAD_FP = 0x07
OP_STORE_FP = 0x27
OP_FMADD = 0x43
OP_FMSUB = 0x47
OP_FNMSUB = 0x4B
OP_FNMADD = 0x4F
OP_OP_FP = 0x53


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

    elif opcode == OP_FP_NPU:
        # FP NPU instructions (opcode 0x2B): all R-type
        return Instruction(opcode=opcode, rd=rd, rs1=rs1, rs2=rs2,
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

    elif opcode == OP_LOAD_FP:
        # FLW: I-type (same layout as integer loads)
        imm = sign_extend(word >> 20, 12)
        return Instruction(opcode=opcode, rd=rd, rs1=rs1, imm=imm,
                           funct3=funct3, funct7=funct7)

    elif opcode == OP_STORE_FP:
        # FSW: S-type (same layout as integer stores)
        imm = sign_extend((funct7 << 5) | rd, 12)
        return Instruction(opcode=opcode, rs1=rs1, rs2=rs2, imm=imm,
                           funct3=funct3, funct7=funct7)

    elif opcode in (OP_FMADD, OP_FMSUB, OP_FNMSUB, OP_FNMADD):
        # R4-type: rs3 in bits[31:27], fmt in bits[26:25]
        rs3 = (word >> 27) & 0x1F
        return Instruction(opcode=opcode, rd=rd, rs1=rs1, rs2=rs2, rs3=rs3,
                           funct3=funct3, funct7=funct7)

    elif opcode == OP_OP_FP:
        # R-type floating-point (FADD.S, FSUB.S, etc.)
        return Instruction(opcode=opcode, rd=rd, rs1=rs1, rs2=rs2,
                           funct3=funct3, funct7=funct7)

    else:
        raise ValueError(f"Unknown opcode: 0x{opcode:02X} (word=0x{word:08X})")


# ---------------------------------------------------------------------------
# Instruction mnemonic classifier (lightweight, for stats tracking)
# ---------------------------------------------------------------------------

# R-type mnemonics: (funct3, funct7) -> name
_R_MNEMONICS: dict[tuple[int, int], str] = {
    (0b000, 0b0000000): "ADD",
    (0b000, 0b0100000): "SUB",
    (0b001, 0b0000000): "SLL",
    (0b010, 0b0000000): "SLT",
    (0b011, 0b0000000): "SLTU",
    (0b100, 0b0000000): "XOR",
    (0b101, 0b0000000): "SRL",
    (0b101, 0b0100000): "SRA",
    (0b110, 0b0000000): "OR",
    (0b111, 0b0000000): "AND",
}

# M-extension mnemonics: funct3 -> name (all have funct7 = 0b0000001)
_M_MNEMONICS: dict[int, str] = {
    0b000: "MUL", 0b001: "MULH", 0b010: "MULHSU", 0b011: "MULHU",
    0b100: "DIV", 0b101: "DIVU", 0b110: "REM", 0b111: "REMU",
}

# I-type arithmetic mnemonics: funct3 -> name
_I_MNEMONICS: dict[int, str] = {
    0b000: "ADDI", 0b010: "SLTI", 0b011: "SLTIU",
    0b100: "XORI", 0b110: "ORI", 0b111: "ANDI",
}

# Load mnemonics: funct3 -> name
_LD_MNEMONICS: dict[int, str] = {
    0b000: "LB", 0b001: "LH", 0b010: "LW", 0b100: "LBU", 0b101: "LHU",
}

# Store mnemonics: funct3 -> name
_ST_MNEMONICS: dict[int, str] = {
    0b000: "SB", 0b001: "SH", 0b010: "SW",
}

# Branch mnemonics: funct3 -> name
_BR_MNEMONICS: dict[int, str] = {
    0b000: "BEQ", 0b001: "BNE", 0b100: "BLT",
    0b101: "BGE", 0b110: "BLTU", 0b111: "BGEU",
}

# NPU integer funct3=0 sub-dispatch by funct7
_NPU_F7_MNEMONICS: dict[int, str] = {
    0: "NPU.MACC", 1: "NPU.VMAC", 2: "NPU.VEXP", 3: "NPU.VRSQRT",
    4: "NPU.VMUL", 5: "NPU.VREDUCE", 6: "NPU.VMAX",
}

# NPU integer funct3 -> name (for funct3 != 0)
_NPU_F3_MNEMONICS: dict[int, str] = {
    0b001: "NPU.RELU", 0b010: "NPU.QMUL", 0b011: "NPU.CLAMP",
    0b100: "NPU.GELU", 0b101: "NPU.RSTACC", 0b110: "NPU.LDVEC",
    0b111: "NPU.STVEC",
}

# FP NPU funct3=0 sub-dispatch by funct7
_FP_NPU_F7_MNEMONICS: dict[int, str] = {
    0: "NPU.FMACC", 1: "NPU.FVMAC", 2: "NPU.FVEXP", 3: "NPU.FVRSQRT",
    4: "NPU.FVMUL", 5: "NPU.FVREDUCE", 6: "NPU.FVMAX",
}

# FP NPU funct3 -> name (for funct3 != 0)
_FP_NPU_F3_MNEMONICS: dict[int, str] = {
    0b001: "NPU.FRELU", 0b100: "NPU.FGELU", 0b101: "NPU.FRSTACC",
}

# OP-FP funct7 -> name
_OP_FP_MNEMONICS: dict[int, str] = {
    0x00: "fadd.s", 0x04: "fsub.s", 0x08: "fmul.s", 0x0C: "fdiv.s",
    0x2C: "fsqrt.s",
}

# OP-FP funct7=0x10 sign-injection by funct3
_FSGNJ_MNEMONICS: dict[int, str] = {
    0: "fsgnj.s", 1: "fsgnjn.s", 2: "fsgnjx.s",
}

# OP-FP funct7=0x50 compare by funct3
_FCMP_MNEMONICS: dict[int, str] = {
    0: "fle.s", 1: "flt.s", 2: "feq.s",
}


def instruction_mnemonic(inst: Instruction) -> str:
    """Return the short mnemonic name for a decoded instruction.

    This is a lightweight classifier intended for instruction statistics
    tracking. It does not format operands -- just the instruction name.

    Args:
        inst: A decoded Instruction dataclass.

    Returns:
        A short string like "ADD", "ADDI", "NPU.FMACC", "flw", etc.
    """
    op = inst.opcode

    if op == OP_R_TYPE:
        if inst.funct7 == 0b0000001:
            return _M_MNEMONICS.get(inst.funct3, "M_EXT?")
        return _R_MNEMONICS.get((inst.funct3, inst.funct7), "R_TYPE?")

    if op == OP_I_ARITH:
        f3 = inst.funct3
        if f3 == 0b001:
            return "SLLI"
        if f3 == 0b101:
            return "SRAI" if inst.funct7 == 0b0100000 else "SRLI"
        return _I_MNEMONICS.get(f3, "I_ARITH?")

    if op == OP_LOAD:
        return _LD_MNEMONICS.get(inst.funct3, "LOAD?")

    if op == OP_STORE:
        return _ST_MNEMONICS.get(inst.funct3, "STORE?")

    if op == OP_BRANCH:
        return _BR_MNEMONICS.get(inst.funct3, "BRANCH?")

    if op == OP_LUI:
        return "LUI"

    if op == OP_AUIPC:
        return "AUIPC"

    if op == OP_JAL:
        return "JAL"

    if op == OP_JALR:
        return "JALR"

    if op == OP_SYSTEM:
        if inst.funct3 == 0:
            imm_val = inst.imm & 0xFFF
            if imm_val == 0:
                return "ECALL"
            if imm_val == 1:
                return "EBREAK"
            if imm_val == 0x302:
                return "MRET"
            return "SYSTEM?"
        csr_names: dict[int, str] = {
            0b001: "CSRRW", 0b010: "CSRRS", 0b011: "CSRRC",
            0b101: "CSRRWI", 0b110: "CSRRSI", 0b111: "CSRRCI",
        }
        return csr_names.get(inst.funct3, "CSR?")

    if op == OP_FENCE:
        return "FENCE"

    if op == OP_NPU:
        if inst.funct3 == 0:
            return _NPU_F7_MNEMONICS.get(inst.funct7, "NPU.F7?")
        return _NPU_F3_MNEMONICS.get(inst.funct3, "NPU?")

    if op == OP_FP_NPU:
        if inst.funct3 == 0:
            return _FP_NPU_F7_MNEMONICS.get(inst.funct7, "NPU.FF7?")
        return _FP_NPU_F3_MNEMONICS.get(inst.funct3, "FP_NPU?")

    if op == OP_LOAD_FP:
        return "flw"

    if op == OP_STORE_FP:
        return "fsw"

    if op == OP_FMADD:
        return "fmadd.s"

    if op == OP_FMSUB:
        return "fmsub.s"

    if op == OP_FNMSUB:
        return "fnmsub.s"

    if op == OP_FNMADD:
        return "fnmadd.s"

    if op == OP_OP_FP:
        f7 = inst.funct7
        if f7 in _OP_FP_MNEMONICS:
            return _OP_FP_MNEMONICS[f7]
        if f7 == 0x10:
            return _FSGNJ_MNEMONICS.get(inst.funct3, "fsgnj?")
        if f7 == 0x14:
            return "fmin.s" if inst.funct3 == 0 else "fmax.s"
        if f7 == 0x50:
            return _FCMP_MNEMONICS.get(inst.funct3, "fcmp?")
        if f7 == 0x60:
            return "fcvt.w.s" if inst.rs2 == 0 else "fcvt.wu.s"
        if f7 == 0x68:
            return "fcvt.s.w" if inst.rs2 == 0 else "fcvt.s.wu"
        if f7 == 0x70:
            return "fmv.x.w" if inst.funct3 == 0 else "fclass.s"
        if f7 == 0x78:
            return "fmv.w.x"
        return "OP-FP?"

    return "UNKNOWN"
