"""TUI disassembly panel: converts decoded instructions to human-readable text."""

from __future__ import annotations

from dataclasses import dataclass

from ..cpu.decode import (
    Instruction,
    decode,
    to_signed,
    OP_R_TYPE,
    OP_I_ARITH,
    OP_LOAD,
    OP_STORE,
    OP_BRANCH,
    OP_LUI,
    OP_AUIPC,
    OP_JAL,
    OP_JALR,
    OP_SYSTEM,
    OP_FENCE,
    OP_NPU,
    OP_FP_NPU,
    OP_LOAD_FP,
    OP_STORE_FP,
    OP_FMADD,
    OP_FMSUB,
    OP_FNMSUB,
    OP_FNMADD,
    OP_OP_FP,
)
from ..memory.bus import MemoryBus


@dataclass(frozen=True)
class DisassemblyLine:
    """A single line of disassembly output."""

    addr: int
    word: int
    text: str
    is_current: bool


# R-type mnemonics: (funct3, funct7) -> name
_R_TYPE_MNEMONICS: dict[tuple[int, int], str] = {
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
_M_EXT_MNEMONICS: dict[int, str] = {
    0b000: "MUL",
    0b001: "MULH",
    0b010: "MULHSU",
    0b011: "MULHU",
    0b100: "DIV",
    0b101: "DIVU",
    0b110: "REM",
    0b111: "REMU",
}

# I-type arithmetic mnemonics: funct3 -> name
_I_ARITH_MNEMONICS: dict[int, str] = {
    0b000: "ADDI",
    0b010: "SLTI",
    0b011: "SLTIU",
    0b100: "XORI",
    0b110: "ORI",
    0b111: "ANDI",
}

# Load mnemonics: funct3 -> name
_LOAD_MNEMONICS: dict[int, str] = {
    0b000: "LB",
    0b001: "LH",
    0b010: "LW",
    0b100: "LBU",
    0b101: "LHU",
}

# Store mnemonics: funct3 -> name
_STORE_MNEMONICS: dict[int, str] = {
    0b000: "SB",
    0b001: "SH",
    0b010: "SW",
}

# Branch mnemonics: funct3 -> name
_BRANCH_MNEMONICS: dict[int, str] = {
    0b000: "BEQ",
    0b001: "BNE",
    0b100: "BLT",
    0b101: "BGE",
    0b110: "BLTU",
    0b111: "BGEU",
}

# CSR mnemonics: funct3 -> name
_CSR_MNEMONICS: dict[int, str] = {
    0b001: "CSRRW",
    0b010: "CSRRS",
    0b011: "CSRRC",
    0b101: "CSRRWI",
    0b110: "CSRRSI",
    0b111: "CSRRCI",
}

# NPU mnemonics: funct3 -> name
_NPU_MNEMONICS: dict[int, str] = {
    0b000: "NPU.MACC",
    0b001: "NPU.RELU",
    0b010: "NPU.QMUL",
    0b011: "NPU.CLAMP",
    0b100: "NPU.GELU",
    0b101: "NPU.RSTACC",
    0b110: "NPU.LDVEC",
    0b111: "NPU.STVEC",
}


# FP NPU mnemonics: funct3 -> name (for non-funct7 instructions)
_FP_NPU_MNEMONICS: dict[int, str] = {
    0b001: "NPU.FRELU",
    0b100: "NPU.FGELU",
    0b101: "NPU.FRSTACC",
}

# FP NPU funct3=0 sub-dispatch by funct7
_FP_NPU_F3_0_MNEMONICS: dict[int, str] = {
    0: "NPU.FMACC",
    1: "NPU.FVMAC",
    2: "NPU.FVEXP",
    3: "NPU.FVRSQRT",
    4: "NPU.FVMUL",
    5: "NPU.FVREDUCE",
    6: "NPU.FVMAX",
}


# Float ABI register names for disassembly
_FLOAT_ABI: list[str] = [
    "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
    "fs0", "fs1",
    "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7",
    "fs2", "fs3", "fs4", "fs5", "fs6", "fs7", "fs8", "fs9", "fs10", "fs11",
    "ft8", "ft9", "ft10", "ft11",
]


def _reg(index: int) -> str:
    """Format a register reference as 'x<N>'."""
    return f"x{index}"


def _freg(index: int) -> str:
    """Format a float register reference with ABI name."""
    return _FLOAT_ABI[index]


def _imm_signed(imm: int) -> int:
    """Interpret a 32-bit immediate as a signed value for display."""
    return to_signed(imm)


def disassemble_instruction(inst: Instruction) -> str:
    """Convert a decoded Instruction into a human-readable mnemonic string.

    Args:
        inst: A decoded Instruction dataclass.

    Returns:
        A string like "ADD x1, x2, x3" or "ADDI x5, x0, 42".
    """
    if inst.opcode == OP_R_TYPE:
        return _disasm_r_type(inst)
    elif inst.opcode == OP_I_ARITH:
        return _disasm_i_arith(inst)
    elif inst.opcode == OP_LOAD:
        return _disasm_load(inst)
    elif inst.opcode == OP_STORE:
        return _disasm_store(inst)
    elif inst.opcode == OP_BRANCH:
        return _disasm_branch(inst)
    elif inst.opcode == OP_LUI:
        return f"LUI {_reg(inst.rd)}, 0x{inst.imm >> 12:X}"
    elif inst.opcode == OP_AUIPC:
        return f"AUIPC {_reg(inst.rd)}, 0x{inst.imm >> 12:X}"
    elif inst.opcode == OP_JAL:
        return f"JAL {_reg(inst.rd)}, {_imm_signed(inst.imm)}"
    elif inst.opcode == OP_JALR:
        return f"JALR {_reg(inst.rd)}, {_reg(inst.rs1)}, {_imm_signed(inst.imm)}"
    elif inst.opcode == OP_SYSTEM:
        return _disasm_system(inst)
    elif inst.opcode == OP_NPU:
        return _disasm_npu(inst)
    elif inst.opcode == OP_FP_NPU:
        return _disasm_fp_npu(inst)
    elif inst.opcode == OP_FENCE:
        return "FENCE"
    elif inst.opcode == OP_LOAD_FP:
        return _disasm_flw(inst)
    elif inst.opcode == OP_STORE_FP:
        return _disasm_fsw(inst)
    elif inst.opcode in (OP_FMADD, OP_FMSUB, OP_FNMSUB, OP_FNMADD):
        return _disasm_fma(inst)
    elif inst.opcode == OP_OP_FP:
        return _disasm_op_fp(inst)
    else:
        return f"UNKNOWN (opcode=0x{inst.opcode:02X})"


def _disasm_r_type(inst: Instruction) -> str:
    """Disassemble an R-type instruction."""
    # Check M-extension first (funct7 == 0b0000001)
    if inst.funct7 == 0b0000001:
        name = _M_EXT_MNEMONICS.get(inst.funct3, f"M_EXT?{inst.funct3}")
        return f"{name} {_reg(inst.rd)}, {_reg(inst.rs1)}, {_reg(inst.rs2)}"

    key = (inst.funct3, inst.funct7)
    name = _R_TYPE_MNEMONICS.get(key, f"R_TYPE?{inst.funct3},{inst.funct7}")
    return f"{name} {_reg(inst.rd)}, {_reg(inst.rs1)}, {_reg(inst.rs2)}"


def _disasm_i_arith(inst: Instruction) -> str:
    """Disassemble an I-type arithmetic instruction."""
    f3 = inst.funct3

    # Shift instructions use shamt (lower 5 bits of imm)
    if f3 == 0b001:  # SLLI
        shamt = inst.imm & 0x1F
        return f"SLLI {_reg(inst.rd)}, {_reg(inst.rs1)}, {shamt}"
    elif f3 == 0b101:
        shamt = inst.imm & 0x1F
        if inst.funct7 == 0b0100000:
            return f"SRAI {_reg(inst.rd)}, {_reg(inst.rs1)}, {shamt}"
        else:
            return f"SRLI {_reg(inst.rd)}, {_reg(inst.rs1)}, {shamt}"

    name = _I_ARITH_MNEMONICS.get(f3, f"I_ARITH?{f3}")
    imm_val = _imm_signed(inst.imm)
    return f"{name} {_reg(inst.rd)}, {_reg(inst.rs1)}, {imm_val}"


def _disasm_load(inst: Instruction) -> str:
    """Disassemble a load instruction."""
    name = _LOAD_MNEMONICS.get(inst.funct3, f"LOAD?{inst.funct3}")
    offset = _imm_signed(inst.imm)
    return f"{name} {_reg(inst.rd)}, {offset}({_reg(inst.rs1)})"


def _disasm_store(inst: Instruction) -> str:
    """Disassemble a store instruction."""
    name = _STORE_MNEMONICS.get(inst.funct3, f"STORE?{inst.funct3}")
    offset = _imm_signed(inst.imm)
    return f"{name} {_reg(inst.rs2)}, {offset}({_reg(inst.rs1)})"


def _disasm_branch(inst: Instruction) -> str:
    """Disassemble a branch instruction."""
    name = _BRANCH_MNEMONICS.get(inst.funct3, f"BRANCH?{inst.funct3}")
    offset = _imm_signed(inst.imm)
    return f"{name} {_reg(inst.rs1)}, {_reg(inst.rs2)}, {offset}"


def _disasm_system(inst: Instruction) -> str:
    """Disassemble a SYSTEM instruction (ECALL, EBREAK, MRET, CSR ops)."""
    if inst.funct3 == 0:
        imm_val = inst.imm & 0xFFF
        if imm_val == 0:
            return "ECALL"
        elif imm_val == 1:
            return "EBREAK"
        elif imm_val == 0x302:
            return "MRET"
        else:
            return f"SYSTEM (imm=0x{imm_val:03X})"
    else:
        # CSR instruction
        csr_addr = inst.imm & 0xFFF
        name = _CSR_MNEMONICS.get(inst.funct3, f"CSR?{inst.funct3}")
        if inst.funct3 in (0b101, 0b110, 0b111):
            # Immediate variants: rs1 field is the zimm value
            return f"{name} {_reg(inst.rd)}, 0x{csr_addr:03X}, {inst.rs1}"
        else:
            return f"{name} {_reg(inst.rd)}, 0x{csr_addr:03X}, {_reg(inst.rs1)}"


def _disasm_npu(inst: Instruction) -> str:
    """Disassemble an NPU instruction (opcode 0x0B)."""
    f3 = inst.funct3
    if f3 == 0b000:
        return _disasm_npu_f3_0(inst)
    name = _NPU_MNEMONICS.get(f3, f"NPU?{f3}")
    if f3 in (0b001, 0b011, 0b100):  # RELU, CLAMP, GELU: rd, rs1
        return f"{name} {_reg(inst.rd)}, {_reg(inst.rs1)}"
    elif f3 == 0b010:  # QMUL: rd, rs1, rs2
        return f"{name} {_reg(inst.rd)}, {_reg(inst.rs1)}, {_reg(inst.rs2)}"
    elif f3 == 0b101:  # RSTACC: rd only
        return f"{name} {_reg(inst.rd)}"
    elif f3 == 0b110:  # LDVEC: I-type load
        offset = _imm_signed(inst.imm)
        return f"{name} v{inst.rd % 4}, {offset}({_reg(inst.rs1)})"
    elif f3 == 0b111:  # STVEC: S-type store
        offset = _imm_signed(inst.imm)
        return f"{name} v{inst.rs2 % 4}, {offset}({_reg(inst.rs1)})"
    else:
        return name


# funct3=0 sub-dispatch by funct7
_NPU_F3_0_MNEMONICS: dict[int, str] = {
    0: "NPU.MACC",
    1: "NPU.VMAC",
    2: "NPU.VEXP",
    3: "NPU.VRSQRT",
    4: "NPU.VMUL",
    5: "NPU.VREDUCE",
    6: "NPU.VMAX",
}


def _disasm_npu_f3_0(inst: Instruction) -> str:
    """Disassemble funct3=0 NPU instructions (dispatched by funct7)."""
    f7 = inst.funct7
    name = _NPU_F3_0_MNEMONICS.get(f7, f"NPU.F7?{f7}")
    if f7 == 0:  # MACC: rs1, rs2 (no rd output)
        return f"{name} {_reg(inst.rs1)}, {_reg(inst.rs2)}"
    elif f7 == 1:  # VMAC: rd=count, rs1=addr_a, rs2=addr_b
        return f"{name} {_reg(inst.rd)}, {_reg(inst.rs1)}, {_reg(inst.rs2)}"
    elif f7 == 2:  # VEXP: rd=count, rs1=src, rs2=dst
        return f"{name} {_reg(inst.rd)}, {_reg(inst.rs1)}, {_reg(inst.rs2)}"
    elif f7 == 3:  # VRSQRT: rd=result, rs1=addr (scalar)
        return f"{name} {_reg(inst.rd)}, {_reg(inst.rs1)}"
    elif f7 == 4:  # VMUL: rd=count, rs1=src, rs2=dst
        return f"{name} {_reg(inst.rd)}, {_reg(inst.rs1)}, {_reg(inst.rs2)}"
    elif f7 == 5:  # VREDUCE: rd=result, rs1=addr, rs2=count
        return f"{name} {_reg(inst.rd)}, {_reg(inst.rs1)}, {_reg(inst.rs2)}"
    elif f7 == 6:  # VMAX: rd=result, rs1=addr, rs2=count
        return f"{name} {_reg(inst.rd)}, {_reg(inst.rs1)}, {_reg(inst.rs2)}"
    else:
        return name


def _disasm_fp_npu(inst: Instruction) -> str:
    """Disassemble an FP NPU instruction (opcode 0x2B)."""
    f3 = inst.funct3
    if f3 == 0b000:
        return _disasm_fp_npu_f3_0(inst)
    name = _FP_NPU_MNEMONICS.get(f3, f"FP_NPU?{f3}")
    if f3 == 0b001:  # FRELU: f[rd], f[rs1]
        return f"{name} {_freg(inst.rd)}, {_freg(inst.rs1)}"
    elif f3 == 0b100:  # FGELU: f[rd], f[rs1]
        return f"{name} {_freg(inst.rd)}, {_freg(inst.rs1)}"
    elif f3 == 0b101:  # FRSTACC: f[rd]
        return f"{name} {_freg(inst.rd)}"
    else:
        return name


def _disasm_fp_npu_f3_0(inst: Instruction) -> str:
    """Disassemble funct3=0 FP NPU instructions (dispatched by funct7)."""
    f7 = inst.funct7
    name = _FP_NPU_F3_0_MNEMONICS.get(f7, f"NPU.FF7?{f7}")
    if f7 == 0:  # FMACC: f[rs1], f[rs2] (no rd output)
        return f"{name} {_freg(inst.rs1)}, {_freg(inst.rs2)}"
    elif f7 == 1:  # FVMAC: x[rd]=count, x[rs1]=addr_a, x[rs2]=addr_b
        return f"{name} {_reg(inst.rd)}, {_reg(inst.rs1)}, {_reg(inst.rs2)}"
    elif f7 == 2:  # FVEXP: x[rd]=count, x[rs1]=src, x[rs2]=dst
        return f"{name} {_reg(inst.rd)}, {_reg(inst.rs1)}, {_reg(inst.rs2)}"
    elif f7 == 3:  # FVRSQRT: f[rd]=result, x[rs1]=addr
        return f"{name} {_freg(inst.rd)}, {_reg(inst.rs1)}"
    elif f7 == 4:  # FVMUL: x[rd]=count, x[rs1]=src, x[rs2]=dst
        return f"{name} {_reg(inst.rd)}, {_reg(inst.rs1)}, {_reg(inst.rs2)}"
    elif f7 == 5:  # FVREDUCE: f[rd]=result, x[rs1]=addr, x[rs2]=count
        return f"{name} {_freg(inst.rd)}, {_reg(inst.rs1)}, {_reg(inst.rs2)}"
    elif f7 == 6:  # FVMAX: f[rd]=result, x[rs1]=addr, x[rs2]=count
        return f"{name} {_freg(inst.rd)}, {_reg(inst.rs1)}, {_reg(inst.rs2)}"
    else:
        return name


# OP-FP mnemonics: funct7 -> name (for simple R-type floats)
_OP_FP_MNEMONICS: dict[int, str] = {
    0x00: "fadd.s",
    0x04: "fsub.s",
    0x08: "fmul.s",
    0x0C: "fdiv.s",
    0x2C: "fsqrt.s",
}

# Sign injection mnemonics: funct3 -> name
_FSGNJ_MNEMONICS: dict[int, str] = {
    0: "fsgnj.s",
    1: "fsgnjn.s",
    2: "fsgnjx.s",
}

# Compare mnemonics: funct3 -> name
_FCMP_MNEMONICS: dict[int, str] = {
    0: "fle.s",
    1: "flt.s",
    2: "feq.s",
}

# Fused multiply-add mnemonics: opcode -> name
_FMA_MNEMONICS: dict[int, str] = {
    OP_FMADD: "fmadd.s",
    OP_FMSUB: "fmsub.s",
    OP_FNMSUB: "fnmsub.s",
    OP_FNMADD: "fnmadd.s",
}


def _disasm_flw(inst: Instruction) -> str:
    """Disassemble FLW instruction."""
    offset = _imm_signed(inst.imm)
    return f"flw {_freg(inst.rd)}, {offset}({_reg(inst.rs1)})"


def _disasm_fsw(inst: Instruction) -> str:
    """Disassemble FSW instruction."""
    offset = _imm_signed(inst.imm)
    return f"fsw {_freg(inst.rs2)}, {offset}({_reg(inst.rs1)})"


def _disasm_fma(inst: Instruction) -> str:
    """Disassemble R4-type fused multiply-add instructions."""
    name = _FMA_MNEMONICS.get(inst.opcode, f"fma?0x{inst.opcode:02X}")
    return f"{name} {_freg(inst.rd)}, {_freg(inst.rs1)}, {_freg(inst.rs2)}, {_freg(inst.rs3)}"


def _disasm_op_fp(inst: Instruction) -> str:
    """Disassemble OP-FP instructions (opcode 0x53)."""
    f7 = inst.funct7
    f3 = inst.funct3

    # Simple arithmetic (fadd, fsub, fmul, fdiv)
    if f7 in _OP_FP_MNEMONICS and f7 != 0x2C:
        name = _OP_FP_MNEMONICS[f7]
        return f"{name} {_freg(inst.rd)}, {_freg(inst.rs1)}, {_freg(inst.rs2)}"

    # fsqrt.s (only rs1)
    if f7 == 0x2C:
        return f"fsqrt.s {_freg(inst.rd)}, {_freg(inst.rs1)}"

    # Sign injection
    if f7 == 0x10:
        name = _FSGNJ_MNEMONICS.get(f3, f"fsgnj?{f3}")
        return f"{name} {_freg(inst.rd)}, {_freg(inst.rs1)}, {_freg(inst.rs2)}"

    # Min/max
    if f7 == 0x14:
        name = "fmin.s" if f3 == 0 else "fmax.s"
        return f"{name} {_freg(inst.rd)}, {_freg(inst.rs1)}, {_freg(inst.rs2)}"

    # Compare (result to int reg)
    if f7 == 0x50:
        name = _FCMP_MNEMONICS.get(f3, f"fcmp?{f3}")
        return f"{name} {_reg(inst.rd)}, {_freg(inst.rs1)}, {_freg(inst.rs2)}"

    # Convert float->int
    if f7 == 0x60:
        if inst.rs2 == 0:
            return f"fcvt.w.s {_reg(inst.rd)}, {_freg(inst.rs1)}"
        else:
            return f"fcvt.wu.s {_reg(inst.rd)}, {_freg(inst.rs1)}"

    # Convert int->float
    if f7 == 0x68:
        if inst.rs2 == 0:
            return f"fcvt.s.w {_freg(inst.rd)}, {_reg(inst.rs1)}"
        else:
            return f"fcvt.s.wu {_freg(inst.rd)}, {_reg(inst.rs1)}"

    # FMV.X.W / FCLASS.S
    if f7 == 0x70:
        if f3 == 0:
            return f"fmv.x.w {_reg(inst.rd)}, {_freg(inst.rs1)}"
        elif f3 == 1:
            return f"fclass.s {_reg(inst.rd)}, {_freg(inst.rs1)}"

    # FMV.W.X
    if f7 == 0x78:
        return f"fmv.w.x {_freg(inst.rd)}, {_reg(inst.rs1)}"

    return f"OP-FP? (funct7=0x{f7:02X}, funct3={f3})"


def disassemble_region(
    memory: MemoryBus, center_pc: int, count: int
) -> list[DisassemblyLine]:
    """Disassemble a region of memory centered on center_pc.

    Reads `count` instruction words from memory, centered on `center_pc`.
    Each instruction is decoded and disassembled into a human-readable string.
    Instructions at addresses that cannot be read (unmapped memory) are shown
    as "???".

    Args:
        memory: The memory bus to read from.
        center_pc: The PC address to center the disassembly on.
        count: Total number of instructions to disassemble.

    Returns:
        A list of DisassemblyLine objects, one per instruction.
    """
    half = count // 2
    start_addr = (center_pc - half * 4) & 0xFFFFFFFF
    lines: list[DisassemblyLine] = []

    for i in range(count):
        addr = (start_addr + i * 4) & 0xFFFFFFFF
        is_current = addr == center_pc
        try:
            word = memory.read32(addr)
            inst = decode(word)
            text = disassemble_instruction(inst)
        except (MemoryError, ValueError):
            word = 0
            text = "???"
        lines.append(DisassemblyLine(addr=addr, word=word, text=text, is_current=is_current))

    return lines
