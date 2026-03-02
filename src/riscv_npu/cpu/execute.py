"""Instruction execution: implements all RV32I and RV32M operations."""

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

from .registers import RegisterFile
from ..memory.bus import MemoryBus
from ..npu.instructions import execute_npu
from ..npu.fp_instructions import execute_fp_npu
from .fpu_execute import execute_fpu

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
        return _exec_system(inst, cpu, pc)

    elif inst.opcode == OP_FENCE:
        return pc + 4

    elif inst.opcode == OP_NPU:
        return execute_npu(inst, cpu)

    elif inst.opcode == OP_FP_NPU:
        return execute_fp_npu(inst, cpu)

    elif inst.opcode in (
        OP_LOAD_FP, OP_STORE_FP, OP_FMADD, OP_FMSUB,
        OP_FNMSUB, OP_FNMADD, OP_OP_FP,
    ):
        return execute_fpu(inst, cpu)

    else:
        raise ValueError(f"Unimplemented opcode: 0x{inst.opcode:02X}")


def _exec_r_type(inst: Instruction, regs: RegisterFile, pc: int) -> int:
    """Execute R-type instructions (opcode 0x33), including M extension."""
    rs1 = regs.read(inst.rs1)
    rs2 = regs.read(inst.rs2)
    f3 = inst.funct3
    f7 = inst.funct7

    # M extension: funct7 == 0b0000001
    if f7 == 0b0000001:
        return _exec_m_ext(inst, regs, pc, rs1, rs2)

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


def _exec_m_ext(
    inst: Instruction, regs: RegisterFile, pc: int, rs1: int, rs2: int
) -> int:
    """Execute M extension instructions (MUL/MULH/MULHSU/MULHU/DIV/DIVU/REM/REMU).

    All M instructions use opcode 0x33 with funct7=0b0000001.
    rs1 and rs2 are unsigned 32-bit register values.
    """
    f3 = inst.funct3

    if f3 == 0b000:  # MUL: lower 32 bits of rs1 * rs2
        result = (rs1 * rs2) & 0xFFFFFFFF

    elif f3 == 0b001:  # MULH: upper 32 bits of signed * signed
        s1 = to_signed(rs1)
        s2 = to_signed(rs2)
        result = ((s1 * s2) >> 32) & 0xFFFFFFFF

    elif f3 == 0b010:  # MULHSU: upper 32 bits of signed * unsigned
        s1 = to_signed(rs1)
        result = ((s1 * rs2) >> 32) & 0xFFFFFFFF

    elif f3 == 0b011:  # MULHU: upper 32 bits of unsigned * unsigned
        result = ((rs1 * rs2) >> 32) & 0xFFFFFFFF

    elif f3 == 0b100:  # DIV: signed division, round toward zero
        if rs2 == 0:
            result = 0xFFFFFFFF
        else:
            s1 = to_signed(rs1)
            s2 = to_signed(rs2)
            # Overflow: INT_MIN / -1
            if s1 == -0x80000000 and s2 == -1:
                result = 0x80000000
            else:
                # Python // rounds toward negative infinity; RISC-V needs
                # truncation toward zero, so use abs division + sign fix
                q = abs(s1) // abs(s2)
                if (s1 < 0) != (s2 < 0):
                    q = -q
                result = q & 0xFFFFFFFF

    elif f3 == 0b101:  # DIVU: unsigned division
        if rs2 == 0:
            result = 0xFFFFFFFF
        else:
            result = (rs1 // rs2) & 0xFFFFFFFF

    elif f3 == 0b110:  # REM: signed remainder
        if rs2 == 0:
            result = rs1
        else:
            s1 = to_signed(rs1)
            s2 = to_signed(rs2)
            # Overflow: INT_MIN % -1
            if s1 == -0x80000000 and s2 == -1:
                result = 0
            else:
                # Python % has sign of divisor; RISC-V rem has sign of dividend
                # Use: rem = dividend - (truncated_quotient * divisor)
                q = abs(s1) // abs(s2)
                if (s1 < 0) != (s2 < 0):
                    q = -q
                rem = s1 - q * s2
                result = rem & 0xFFFFFFFF

    elif f3 == 0b111:  # REMU: unsigned remainder
        if rs2 == 0:
            result = rs1
        else:
            result = (rs1 % rs2) & 0xFFFFFFFF

    else:
        raise ValueError(f"Unknown M extension funct3: {f3:#05b}")

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


def _exec_load(inst: Instruction, regs: RegisterFile, mem: MemoryBus, pc: int) -> int:
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


def _exec_store(inst: Instruction, regs: RegisterFile, mem: MemoryBus, pc: int) -> int:
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


# MRET instruction imm field value
_IMM_MRET = 0x302
# ECALL cause codes
_CAUSE_MACHINE_ECALL = 11
# CSR addresses used by trap handling (imported from cpu module constants)
_CSR_MTVEC = 0x305
_CSR_MEPC = 0x341
_CSR_MCAUSE = 0x342


def _exec_system(inst: Instruction, cpu: CPU, pc: int) -> int:
    """Execute SYSTEM instructions (ecall, ebreak, mret, CSR ops).

    Dispatches based on funct3:
    - funct3==0: ECALL, EBREAK, MRET (distinguished by imm)
    - funct3!=0: CSR instructions (CSRRW, CSRRS, CSRRC and I variants)
    """
    if inst.funct3 == 0:
        if inst.imm == 0:  # ECALL
            return _exec_ecall(cpu, pc)
        elif inst.imm == 1:  # EBREAK
            cpu.halted = True
            return pc + 4
        elif (inst.imm & 0xFFF) == _IMM_MRET:  # MRET
            return _exec_mret(cpu)
        else:
            # Unknown system instruction, treat as NOP
            return pc + 4
    else:
        _exec_csr(inst, cpu)
        return pc + 4


def _exec_ecall(cpu: CPU, pc: int) -> int:
    """Handle ECALL: try syscall handler, then trap to mtvec, then halt.

    Priority order:
    1. If a syscall_handler is installed, try it first. If it handles
       the syscall (returns True), advance PC by 4 (no trap).
    2. If mtvec is configured (non-zero), trigger a machine-mode trap.
    3. Otherwise, halt the CPU (simple mode).
    """
    # Try syscall handler first
    if cpu.syscall_handler is not None:
        if cpu.syscall_handler.handle(cpu):
            return (pc + 4) & 0xFFFFFFFF

    # Fall through to trap/halt
    mtvec = cpu.csr_read(_CSR_MTVEC)
    if mtvec != 0:
        # Trap: set cause and return address, jump to trap vector
        cpu.csr_write(_CSR_MCAUSE, _CAUSE_MACHINE_ECALL)
        cpu.csr_write(_CSR_MEPC, pc)
        return mtvec & 0xFFFFFFFF
    else:
        # No trap vector configured: just halt (simple mode)
        cpu.halted = True
        return pc + 4


def _exec_mret(cpu: CPU) -> int:
    """Handle MRET: return from machine-mode trap handler.

    Jumps to the address stored in mepc.
    """
    mepc = cpu.csr_read(_CSR_MEPC)
    return mepc & 0xFFFFFFFF


def _exec_csr(inst: Instruction, cpu: CPU) -> None:
    """Execute CSR instructions.

    Handles CSRRW, CSRRS, CSRRC and their immediate variants (CSRRWI,
    CSRRSI, CSRRCI). Uses the CPU's csr_read/csr_write methods which
    route to the CSR register storage.
    """
    regs = cpu.registers
    f3 = inst.funct3
    # The CSR address is the raw 12-bit immediate (not sign-extended)
    csr_addr = inst.imm & 0xFFF

    # Read the current CSR value
    old_val = cpu.csr_read(csr_addr)

    # Determine the source value
    if f3 in (0b001, 0b010, 0b011):
        # CSRRW (001), CSRRS (010), CSRRC (011): source is rs1
        src = regs.read(inst.rs1)
    elif f3 in (0b101, 0b110, 0b111):
        # CSRRWI (101), CSRRSI (110), CSRRCI (111): source is zimm (rs1 field)
        src = inst.rs1  # 5-bit zero-extended immediate
    else:
        return  # Unknown CSR funct3, ignore

    # Compute new CSR value
    if f3 in (0b001, 0b101):  # CSRRW / CSRRWI
        new_val = src
    elif f3 in (0b010, 0b110):  # CSRRS / CSRRSI
        new_val = old_val | src
    elif f3 in (0b011, 0b111):  # CSRRC / CSRRCI
        new_val = old_val & ~src
    else:
        new_val = old_val

    new_val = new_val & 0xFFFFFFFF

    # Write old value to rd
    regs.write(inst.rd, old_val)

    # Write new value to CSR
    cpu.csr_write(csr_addr, new_val)
