"""NPU custom instruction execution: opcode 0x0B dispatch and semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..cpu.decode import Instruction, to_signed
from .engine import NpuState, acc_add, acc_reset, GELU_TABLE, exp_q16_16, rsqrt_q16_16

if TYPE_CHECKING:
    from ..cpu.cpu import CPU
    from ..cpu.registers import RegisterFile
    from ..memory.bus import MemoryBus


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
        f7 = inst.funct7
        if f7 == 0:
            _exec_macc(inst, regs, npu)
        elif f7 == 1:
            _exec_vmac(inst, regs, cpu.memory, npu)
        elif f7 == 2:
            _exec_vexp(inst, regs, cpu.memory)
        elif f7 == 3:
            _exec_vrsqrt(inst, regs, cpu.memory)
        elif f7 == 4:
            _exec_vmul(inst, regs, cpu.memory, npu)
        elif f7 == 5:
            _exec_vreduce(inst, regs, cpu.memory)
        elif f7 == 6:
            _exec_vmax(inst, regs, cpu.memory)
        else:
            raise ValueError(f"Unknown NPU funct7: {f7:#09b} for funct3=0")
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


def _exec_vmac(
    inst: Instruction,
    regs: 'RegisterFile',
    mem: 'MemoryBus',
    npu: NpuState,
) -> None:
    """NPU.VMAC: acc += dot(mem_int8[rs1..+n], mem_int8[rs2..+n]).

    Reads rd as element count, rs1/rs2 as base addresses of int8 arrays.
    For each pair of bytes, sign-extends to int8, multiplies, and adds
    the product to the 64-bit accumulator. Does NOT reset the accumulator.
    """
    n = regs.read(inst.rd)
    addr_a = regs.read(inst.rs1)
    addr_b = regs.read(inst.rs2)
    for i in range(n):
        byte_a = mem.read8((addr_a + i) & 0xFFFFFFFF)
        byte_b = mem.read8((addr_b + i) & 0xFFFFFFFF)
        # Sign-extend bytes to int8
        a = byte_a - 256 if byte_a >= 128 else byte_a
        b = byte_b - 256 if byte_b >= 128 else byte_b
        acc_add(npu, a * b)


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


# ==================== Phase 7: Vector instructions ====================


def _exec_vexp(
    inst: Instruction,
    regs: 'RegisterFile',
    mem: 'MemoryBus',
) -> None:
    """NPU.VEXP: vectorized exp over int32 Q16.16 array.

    For each element i in 0..n-1:
        dst[i] = exp_q16_16(src[i])

    Reads rd as element count, rs1 as source address, rs2 as destination
    address. Elements are int32 (4 bytes each) in Q16.16 fixed-point format.
    """
    n = regs.read(inst.rd)
    addr_src = regs.read(inst.rs1)
    addr_dst = regs.read(inst.rs2)
    for i in range(n):
        src_addr = (addr_src + i * 4) & 0xFFFFFFFF
        dst_addr = (addr_dst + i * 4) & 0xFFFFFFFF
        val = mem.read32(src_addr)
        result = exp_q16_16(val)
        mem.write32(dst_addr, result & 0xFFFFFFFF)


def _exec_vrsqrt(
    inst: Instruction,
    regs: 'RegisterFile',
    mem: 'MemoryBus',
) -> None:
    """NPU.VRSQRT: scalar reciprocal square root in Q16.16.

    Reads one int32 from mem[rs1] as Q16.16 fixed-point input.
    Computes 1/sqrt(x) in Q16.16 and writes the result to register rd.
    """
    addr = regs.read(inst.rs1)
    val = mem.read32(addr)
    result = rsqrt_q16_16(val)
    regs.write(inst.rd, result & 0xFFFFFFFF)


def _exec_vmul(
    inst: Instruction,
    regs: 'RegisterFile',
    mem: 'MemoryBus',
    npu: NpuState,
) -> None:
    """NPU.VMUL: scale int8 vector by Q16.16 factor from accumulator.

    For each element i in 0..n-1:
        dst[i] = clamp((src_int8[i] * acc_lo_signed) >> 16, -128, 127)

    Reads rd as element count, rs1 as source int8 address, rs2 as
    destination int8 address. The scale factor is taken from acc_lo,
    interpreted as a signed 32-bit Q16.16 value.
    The accumulator is NOT modified.
    """
    n = regs.read(inst.rd)
    addr_src = regs.read(inst.rs1)
    addr_dst = regs.read(inst.rs2)
    # Scale factor from accumulator low word, interpreted as signed
    scale = to_signed(npu.acc_lo)
    for i in range(n):
        byte_val = mem.read8((addr_src + i) & 0xFFFFFFFF)
        # Sign-extend byte to int8
        src_signed = byte_val - 256 if byte_val >= 128 else byte_val
        # Multiply and arithmetic right shift by 16
        product = src_signed * scale
        result = product >> 16  # Python >> on negative is arithmetic
        # Clamp to int8 range
        result = max(-128, min(127, result))
        # Write as unsigned byte
        mem.write8((addr_dst + i) & 0xFFFFFFFF, result & 0xFF)


def _exec_vreduce(
    inst: Instruction,
    regs: 'RegisterFile',
    mem: 'MemoryBus',
) -> None:
    """NPU.VREDUCE: sum int32 array and write result to rd.

    Reads rs2 as element count, rs1 as source address of int32 array.
    Sums all elements (signed int32) and writes the 32-bit result to rd.
    If count is 0, rd is set to 0.
    """
    n = regs.read(inst.rs2)
    addr_src = regs.read(inst.rs1)
    total = 0
    for i in range(n):
        val = mem.read32((addr_src + i * 4) & 0xFFFFFFFF)
        total += to_signed(val)
    regs.write(inst.rd, total & 0xFFFFFFFF)


def _exec_vmax(
    inst: Instruction,
    regs: 'RegisterFile',
    mem: 'MemoryBus',
) -> None:
    """NPU.VMAX: find maximum of int32 array and write result to rd.

    Reads rs2 as element count, rs1 as source address of int32 array.
    Finds the maximum signed int32 value and writes it to rd.
    If count is 0, rd is set to 0x80000000 (minimum int32).
    """
    n = regs.read(inst.rs2)
    addr_src = regs.read(inst.rs1)
    if n == 0:
        regs.write(inst.rd, 0x80000000)
        return
    max_val = -0x80000000  # Start with minimum int32
    for i in range(n):
        val = mem.read32((addr_src + i * 4) & 0xFFFFFFFF)
        signed_val = to_signed(val)
        if signed_val > max_val:
            max_val = signed_val
    regs.write(inst.rd, max_val & 0xFFFFFFFF)
