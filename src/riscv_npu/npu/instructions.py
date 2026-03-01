"""NPU custom instruction execution: opcode 0x0B dispatch and semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..cpu.decode import Instruction, to_signed
from .engine import NpuState, acc_add, acc_reset, GELU_TABLE

if TYPE_CHECKING:
    from ..cpu.cpu import CPU


def execute_npu(inst: Instruction, cpu: CPU) -> int:
    """Execute an NPU instruction (opcode 0x0B).

    Dispatches by funct3:
        0: MACC  - multiply-accumulate
        1: RELU  - rectified linear unit
        2: QMUL  - quantized multiply (arithmetic right shift by 8)
        3: CLAMP - clamp to int8 range [-128, 127]
        4: GELU  - GELU activation via lookup table
        5: RSTACC - reset accumulator, return old acc_lo
        6: LDVEC - load 4 bytes from memory into vector register
        7: STVEC - store vector register to memory as 4 bytes

    Args:
        inst: Decoded instruction with opcode 0x0B.
        cpu: CPU instance (provides registers, memory, npu_state).

    Returns:
        Next PC value (always pc + 4).
    """
    f3 = inst.funct3
    npu = cpu.npu_state
    regs = cpu.registers
    pc = cpu.pc

    if f3 == 0:
        _exec_macc(inst, regs, npu)
    elif f3 == 1:
        _exec_relu(inst, regs)
    elif f3 == 2:
        _exec_qmul(inst, regs)
    elif f3 == 3:
        _exec_clamp(inst, regs)
    elif f3 == 4:
        _exec_gelu(inst, regs)
    elif f3 == 5:
        _exec_rstacc(inst, regs, npu)
    elif f3 == 6:
        _exec_ldvec(inst, regs, cpu.memory, npu)
    elif f3 == 7:
        _exec_stvec(inst, regs, cpu.memory, npu)
    else:
        raise ValueError(f"Unknown NPU funct3: {f3:#05b}")

    return (pc + 4) & 0xFFFFFFFF


def _exec_macc(inst: Instruction, regs: 'RegisterFile', npu: NpuState) -> None:
    """NPU.MACC: {acc_hi, acc_lo} += signed(rs1) * signed(rs2).

    Multiplies two signed 32-bit register values and adds the 64-bit
    product to the accumulator.
    """
    rs1_val = to_signed(regs.read(inst.rs1))
    rs2_val = to_signed(regs.read(inst.rs2))
    product = rs1_val * rs2_val
    acc_add(npu, product)


def _exec_relu(inst: Instruction, regs: 'RegisterFile') -> None:
    """NPU.RELU: rd = max(signed(rs1), 0).

    Applies ReLU activation: if the signed value of rs1 is negative,
    writes 0 to rd; otherwise writes the original (unsigned) value.
    """
    rs1_val = regs.read(inst.rs1)
    signed_val = to_signed(rs1_val)
    result = rs1_val if signed_val >= 0 else 0
    regs.write(inst.rd, result)


def _exec_qmul(inst: Instruction, regs: 'RegisterFile') -> None:
    """NPU.QMUL: rd = (signed(rs1) * signed(rs2)) >> 8.

    Quantized multiply: signed multiply then arithmetic right shift by 8.
    The result is a 32-bit value stored in rd.
    """
    rs1_val = to_signed(regs.read(inst.rs1))
    rs2_val = to_signed(regs.read(inst.rs2))
    product = rs1_val * rs2_val
    # Arithmetic right shift by 8 (Python >> on negative is arithmetic)
    result = product >> 8
    regs.write(inst.rd, result & 0xFFFFFFFF)


def _exec_clamp(inst: Instruction, regs: 'RegisterFile') -> None:
    """NPU.CLAMP: rd = clamp(signed(rs1), -128, 127).

    Clamps the signed 32-bit value to the int8 range [-128, 127].
    Result stored as uint32 (masked to 32 bits).
    """
    signed_val = to_signed(regs.read(inst.rs1))
    clamped = max(-128, min(127, signed_val))
    regs.write(inst.rd, clamped & 0xFFFFFFFF)


def _exec_gelu(inst: Instruction, regs: 'RegisterFile') -> None:
    """NPU.GELU: rd = gelu_approx(rs1) via lookup table.

    Interprets the low 8 bits of rs1 as a signed int8 value,
    looks up the GELU result in the precomputed table, and
    writes it to rd (sign-extended to 32 bits).
    """
    rs1_val = regs.read(inst.rs1)
    # Extract low byte as signed int8
    byte_val = rs1_val & 0xFF
    if byte_val >= 128:
        signed_byte = byte_val - 256
    else:
        signed_byte = byte_val
    # Look up in table: index = signed_byte + 128
    table_idx = signed_byte + 128
    result = GELU_TABLE[table_idx]
    # Store as uint32 (sign extend from int8)
    regs.write(inst.rd, result & 0xFFFFFFFF)


def _exec_rstacc(inst: Instruction, regs: 'RegisterFile', npu: NpuState) -> None:
    """NPU.RSTACC: rd = acc_lo; acc = 0.

    Returns the lower 32 bits of the accumulator in rd,
    then resets the entire 64-bit accumulator to zero.
    """
    old_lo = acc_reset(npu)
    regs.write(inst.rd, old_lo)


def _exec_ldvec(
    inst: Instruction,
    regs: 'RegisterFile',
    mem: 'MemoryBus',
    npu: NpuState,
) -> None:
    """NPU.LDVEC: vreg[rd%4] = mem32[rs1 + sext(imm)] as 4x int8.

    Loads 4 bytes from memory at the computed address into a vector
    register. Bytes are stored as signed int8 values.
    """
    addr = (regs.read(inst.rs1) + to_signed(inst.imm)) & 0xFFFFFFFF
    vreg_idx = inst.rd % 4
    for i in range(4):
        byte_val = mem.read8((addr + i) & 0xFFFFFFFF)
        # Convert to signed int8
        if byte_val >= 128:
            npu.vreg[vreg_idx][i] = byte_val - 256
        else:
            npu.vreg[vreg_idx][i] = byte_val


def _exec_stvec(
    inst: Instruction,
    regs: 'RegisterFile',
    mem: 'MemoryBus',
    npu: NpuState,
) -> None:
    """NPU.STVEC: mem32[rs1 + sext(imm)] = vreg[rs2%4] as 4x int8.

    Stores 4 int8 values from a vector register to memory at the
    computed address. Each int8 is written as a single byte.
    """
    addr = (regs.read(inst.rs1) + to_signed(inst.imm)) & 0xFFFFFFFF
    vreg_idx = inst.rs2 % 4
    for i in range(4):
        val = npu.vreg[vreg_idx][i]
        # Convert signed int8 to unsigned byte
        byte_val = val & 0xFF
        mem.write8((addr + i) & 0xFFFFFFFF, byte_val)
