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


def _reg(index: int) -> str:
    """Format a register reference as 'x<N>'."""
    return f"x{index}"


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
    elif inst.opcode == OP_FENCE:
        return "FENCE"
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
    imm_val = _imm_signed(inst.imm)

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
