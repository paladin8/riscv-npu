"""Instruction execution: implements all RV32I operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .decode import (
    Instruction,
    sign_extend,
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

from .registers import RegisterFile
from ..memory.ram import RAM

if TYPE_CHECKING:
    from .cpu import CPU


def execute(inst: Instruction, cpu: CPU) -> int:
    """Execute a decoded instruction. Returns the next PC value."""
    regs = cpu.registers
    mem = cpu.memory
    pc = cpu.pc

    if inst.opcode == OP_R_TYPE:
        return _exec_r_type(inst, regs, pc)

    elif inst.opcode == OP_I_ARITH:
        return _exec_i_arith(inst, regs, pc)

    elif inst.opcode == OP_LOAD:
        return _exec_load(inst, regs, mem, pc)

    elif inst.opcode == OP_STORE:
        return _exec_store(inst, regs, mem, pc)

    elif inst.opcode == OP_BRANCH:
        return _exec_branch(inst, regs, pc)

    elif inst.opcode == OP_LUI:
        regs.write(inst.rd, inst.imm)
        return pc + 4

    elif inst.opcode == OP_AUIPC:
        regs.write(inst.rd, (pc + inst.imm) & 0xFFFFFFFF)
        return pc + 4

    elif inst.opcode == OP_JAL:
        regs.write(inst.rd, (pc + 4) & 0xFFFFFFFF)
        return (pc + to_signed(inst.imm)) & 0xFFFFFFFF

    elif inst.opcode == OP_JALR:
        rs1_val = regs.read(inst.rs1)
        regs.write(inst.rd, (pc + 4) & 0xFFFFFFFF)
        return (rs1_val + to_signed(inst.imm)) & 0xFFFFFFFE

    elif inst.opcode == OP_SYSTEM:
        if inst.imm == 0:  # ECALL
            cpu.halted = True
        elif inst.imm == 1:  # EBREAK
            cpu.halted = True
        return pc + 4

    elif inst.opcode == OP_FENCE:
        return pc + 4

    else:
        raise ValueError(f"Unimplemented opcode: 0x{inst.opcode:02X}")


def _exec_r_type(inst: Instruction, regs: RegisterFile, pc: int) -> int:
    """Execute R-type instructions (opcode 0x33)."""
    rs1 = regs.read(inst.rs1)
    rs2 = regs.read(inst.rs2)
    f3 = inst.funct3
    f7 = inst.funct7

    if f3 == 0b000:
        if f7 == 0b0000000:  # ADD
            result = (rs1 + rs2) & 0xFFFFFFFF
        elif f7 == 0b0100000:  # SUB
            result = (rs1 - rs2) & 0xFFFFFFFF
        else:
            raise ValueError(f"Unknown R-type funct7: {f7:#09b}")
    elif f3 == 0b001:  # SLL
        result = (rs1 << (rs2 & 0x1F)) & 0xFFFFFFFF
    elif f3 == 0b010:  # SLT
        result = 1 if to_signed(rs1) < to_signed(rs2) else 0
    elif f3 == 0b011:  # SLTU
        result = 1 if rs1 < rs2 else 0
    elif f3 == 0b100:  # XOR
        result = rs1 ^ rs2
    elif f3 == 0b101:
        if f7 == 0b0000000:  # SRL
            result = rs1 >> (rs2 & 0x1F)
        elif f7 == 0b0100000:  # SRA
            result = (to_signed(rs1) >> (rs2 & 0x1F)) & 0xFFFFFFFF
        else:
            raise ValueError(f"Unknown R-type funct7 for shift: {f7:#09b}")
    elif f3 == 0b110:  # OR
        result = rs1 | rs2
    elif f3 == 0b111:  # AND
        result = rs1 & rs2
    else:
        raise ValueError(f"Unknown R-type funct3: {f3:#05b}")

    regs.write(inst.rd, result)
    return pc + 4


def _exec_i_arith(inst: Instruction, regs: RegisterFile, pc: int) -> int:
    """Execute I-type arithmetic instructions (opcode 0x13)."""
    rs1 = regs.read(inst.rs1)
    imm = inst.imm
    imm_signed = to_signed(imm)
    f3 = inst.funct3

    if f3 == 0b000:  # ADDI
        result = (rs1 + imm_signed) & 0xFFFFFFFF
    elif f3 == 0b010:  # SLTI
        result = 1 if to_signed(rs1) < imm_signed else 0
    elif f3 == 0b011:  # SLTIU
        # Note: immediate is sign-extended, then unsigned comparison
        result = 1 if rs1 < imm else 0
    elif f3 == 0b100:  # XORI
        result = rs1 ^ imm
    elif f3 == 0b110:  # ORI
        result = rs1 | imm
    elif f3 == 0b111:  # ANDI
        result = rs1 & imm
    elif f3 == 0b001:  # SLLI
        shamt = imm & 0x1F
        result = (rs1 << shamt) & 0xFFFFFFFF
    elif f3 == 0b101:
        shamt = imm & 0x1F
        if inst.funct7 == 0b0000000:  # SRLI
            result = rs1 >> shamt
        elif inst.funct7 == 0b0100000:  # SRAI
            result = (to_signed(rs1) >> shamt) & 0xFFFFFFFF
        else:
            raise ValueError(f"Unknown I-type shift funct7: {inst.funct7:#09b}")
    else:
        raise ValueError(f"Unknown I-type funct3: {f3:#05b}")

    regs.write(inst.rd, result)
    return pc + 4


def _exec_load(inst: Instruction, regs: RegisterFile, mem: RAM, pc: int) -> int:
    """Execute load instructions (opcode 0x03)."""
    addr = (regs.read(inst.rs1) + to_signed(inst.imm)) & 0xFFFFFFFF
    f3 = inst.funct3

    if f3 == 0b000:  # LB
        result = sign_extend(mem.read8(addr), 8)
    elif f3 == 0b001:  # LH
        result = sign_extend(mem.read16(addr), 16)
    elif f3 == 0b010:  # LW
        result = mem.read32(addr)
    elif f3 == 0b100:  # LBU
        result = mem.read8(addr)
    elif f3 == 0b101:  # LHU
        result = mem.read16(addr)
    else:
        raise ValueError(f"Unknown load funct3: {f3:#05b}")

    regs.write(inst.rd, result)
    return pc + 4


def _exec_store(inst: Instruction, regs: RegisterFile, mem: RAM, pc: int) -> int:
    """Execute store instructions (opcode 0x23)."""
    addr = (regs.read(inst.rs1) + to_signed(inst.imm)) & 0xFFFFFFFF
    rs2_val = regs.read(inst.rs2)
    f3 = inst.funct3

    if f3 == 0b000:  # SB
        mem.write8(addr, rs2_val & 0xFF)
    elif f3 == 0b001:  # SH
        mem.write16(addr, rs2_val & 0xFFFF)
    elif f3 == 0b010:  # SW
        mem.write32(addr, rs2_val)
    else:
        raise ValueError(f"Unknown store funct3: {f3:#05b}")

    return pc + 4


def _exec_branch(inst: Instruction, regs: RegisterFile, pc: int) -> int:
    """Execute branch instructions (opcode 0x63)."""
    rs1 = regs.read(inst.rs1)
    rs2 = regs.read(inst.rs2)
    f3 = inst.funct3

    if f3 == 0b000:  # BEQ
        taken = rs1 == rs2
    elif f3 == 0b001:  # BNE
        taken = rs1 != rs2
    elif f3 == 0b100:  # BLT
        taken = to_signed(rs1) < to_signed(rs2)
    elif f3 == 0b101:  # BGE
        taken = to_signed(rs1) >= to_signed(rs2)
    elif f3 == 0b110:  # BLTU
        taken = rs1 < rs2
    elif f3 == 0b111:  # BGEU
        taken = rs1 >= rs2
    else:
        raise ValueError(f"Unknown branch funct3: {f3:#05b}")

    if taken:
        return (pc + to_signed(inst.imm)) & 0xFFFFFFFF
    return pc + 4
