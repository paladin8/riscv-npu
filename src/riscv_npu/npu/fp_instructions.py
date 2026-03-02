"""FP NPU custom instruction execution: opcode 0x2B dispatch and semantics."""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING

from ..cpu.decode import Instruction
from .engine import NpuState, facc_add, facc_reset, fgelu

if TYPE_CHECKING:
    from ..cpu.cpu import CPU
    from ..cpu.fpu import FRegisterFile
    from ..cpu.registers import RegisterFile
    from ..memory.bus import MemoryBus


def _read_mem_f32(mem: MemoryBus, addr: int) -> float:
    """Read a float32 value from memory at the given address."""
    bits = mem.read32(addr)
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def _f64_to_f32_bits(value: float) -> int:
    """Convert a float64 value to IEEE 754 single-precision bits.

    Handles overflow: values exceeding float32 range become +/-inf.

    Args:
        value: The float64 value to convert.

    Returns:
        32-bit unsigned integer with IEEE 754 single-precision bits.
    """
    try:
        return struct.unpack('<I', struct.pack('<f', value))[0]
    except OverflowError:
        # Value exceeds float32 range -> +inf or -inf
        if value > 0:
            return 0x7F800000  # +inf
        else:
            return 0xFF800000  # -inf


def _write_mem_f32(mem: MemoryBus, addr: int, value: float) -> None:
    """Write a float32 value to memory at the given address."""
    mem.write32(addr, _f64_to_f32_bits(value))


def execute_fp_npu(inst: Instruction, cpu: CPU) -> int:
    """Execute an FP NPU instruction (opcode 0x2B).

    Dispatches by funct3:
        0: sub-dispatch by funct7 (FMACC, FVMAC, FVEXP, FVRSQRT, FVMUL, FVREDUCE, FVMAX)
        1: FRELU  - FP rectified linear unit
        4: FGELU  - FP GELU activation
        5: FRSTACC - FP reset accumulator

    Args:
        inst: Decoded instruction with opcode 0x2B.
        cpu: CPU instance (provides registers, memory, fpu, npu_state).

    Returns:
        Next PC value (always pc + 4).
    """
    f3 = inst.funct3
    npu = cpu.npu_state
    regs = cpu.registers
    fregs = cpu.fpu_state.fregs
    mem = cpu.memory
    pc = cpu.pc

    if f3 == 0:
        f7 = inst.funct7
        if f7 == 0:
            _exec_fmacc(inst, fregs, npu)
        elif f7 == 1:
            _exec_fvmac(inst, regs, mem, npu)
        elif f7 == 2:
            _exec_fvexp(inst, regs, mem)
        elif f7 == 3:
            _exec_fvrsqrt(inst, regs, fregs, mem)
        elif f7 == 4:
            _exec_fvmul(inst, regs, mem, npu)
        elif f7 == 5:
            _exec_fvreduce(inst, regs, fregs, mem)
        elif f7 == 6:
            _exec_fvmax(inst, regs, fregs, mem)
        else:
            raise ValueError(f"Unknown FP NPU funct7: {f7:#09b} for funct3=0")
    elif f3 == 1:
        _exec_frelu(inst, fregs)
    elif f3 == 4:
        _exec_fgelu(inst, fregs)
    elif f3 == 5:
        _exec_frstacc(inst, fregs, npu)
    else:
        raise ValueError(f"Unknown FP NPU funct3: {f3:#05b}")

    return (pc + 4) & 0xFFFFFFFF


def _exec_fmacc(inst: Instruction, fregs: FRegisterFile, npu: NpuState) -> None:
    """NPU.FMACC: facc += f[rs1] * f[rs2].

    Multiplies two single-precision float register values in double precision
    and adds the product to the float64 accumulator.
    """
    a = float(fregs.read_float(inst.rs1))
    b = float(fregs.read_float(inst.rs2))
    facc_add(npu, a * b)


def _exec_fvmac(
    inst: Instruction,
    regs: RegisterFile,
    mem: MemoryBus,
    npu: NpuState,
) -> None:
    """NPU.FVMAC: facc += dot(mem_f32[rs1..+n], mem_f32[rs2..+n]).

    Reads rd as element count (integer register), rs1/rs2 as base addresses
    of float32 arrays. For each pair, multiplies in double precision and
    adds to the float64 accumulator. Does NOT reset the accumulator.
    """
    n = regs.read(inst.rd)
    addr_a = regs.read(inst.rs1)
    addr_b = regs.read(inst.rs2)
    for i in range(n):
        a = _read_mem_f32(mem, (addr_a + i * 4) & 0xFFFFFFFF)
        b = _read_mem_f32(mem, (addr_b + i * 4) & 0xFFFFFFFF)
        facc_add(npu, float(a) * float(b))


def _exec_frelu(inst: Instruction, fregs: FRegisterFile) -> None:
    """NPU.FRELU: f[rd] = max(f[rs1], +0.0).

    If f[rs1] is negative (including -0.0), writes +0.0.
    NaN input produces NaN.
    """
    val = fregs.read_float(inst.rs1)
    if math.isnan(val):
        result = val
    elif val < 0.0 or (val == 0.0 and math.copysign(1.0, val) < 0.0):
        result = 0.0
    else:
        result = val
    fregs.write_float(inst.rd, result)


def _exec_fgelu(inst: Instruction, fregs: FRegisterFile) -> None:
    """NPU.FGELU: f[rd] = gelu(f[rs1]).

    Computes the exact GELU activation at full FP32 precision.
    """
    val = fregs.read_float(inst.rs1)
    if math.isnan(val) or math.isinf(val):
        # NaN → NaN, +inf → +inf, -inf → -0.0 (gelu(-inf) = 0)
        if math.isinf(val) and val < 0:
            result = 0.0
        else:
            result = val
    else:
        result = fgelu(val)
    fregs.write_float(inst.rd, result)


def _exec_frstacc(inst: Instruction, fregs: FRegisterFile, npu: NpuState) -> None:
    """NPU.FRSTACC: f[rd] = (float32)facc; facc = 0.0.

    Reads the float64 accumulator, rounds to single precision,
    writes to float register f[rd], then zeroes the accumulator.
    Handles overflow: facc values exceeding float32 range produce +/-inf.
    """
    old_val = facc_reset(npu)
    fregs.write_bits(inst.rd, _f64_to_f32_bits(old_val))


def _exec_fvexp(
    inst: Instruction,
    regs: RegisterFile,
    mem: MemoryBus,
) -> None:
    """NPU.FVEXP: vectorized exp over float32 array.

    For each element i in 0..n-1:
        dst[i] = exp(src[i])

    Reads rd as element count (integer), rs1 as source address,
    rs2 as destination address. Elements are float32 (4 bytes each).
    """
    n = regs.read(inst.rd)
    addr_src = regs.read(inst.rs1)
    addr_dst = regs.read(inst.rs2)
    for i in range(n):
        src_addr = (addr_src + i * 4) & 0xFFFFFFFF
        dst_addr = (addr_dst + i * 4) & 0xFFFFFFFF
        val = _read_mem_f32(mem, src_addr)
        if math.isnan(val):
            result = val
        elif math.isinf(val):
            result = 0.0 if val < 0 else val
        else:
            try:
                result = math.exp(val)
            except OverflowError:
                result = float('inf')
        _write_mem_f32(mem, dst_addr, result)


def _exec_fvrsqrt(
    inst: Instruction,
    regs: RegisterFile,
    fregs: FRegisterFile,
    mem: MemoryBus,
) -> None:
    """NPU.FVRSQRT: f[rd] = 1/sqrt(mem_f32[rs1]).

    Reads one float32 from memory at address rs1, computes 1/sqrt(x),
    and writes the float32 result to f[rd].
    Negative → NaN. Zero → +inf.
    """
    addr = regs.read(inst.rs1)
    val = _read_mem_f32(mem, addr)
    if math.isnan(val) or val < 0.0:
        result = float('nan')
    elif val == 0.0:
        result = float('inf')
    else:
        result = 1.0 / math.sqrt(val)
    fregs.write_float(inst.rd, result)


def _exec_fvmul(
    inst: Instruction,
    regs: RegisterFile,
    mem: MemoryBus,
    npu: NpuState,
) -> None:
    """NPU.FVMUL: scale float32 array by accumulator value.

    For each element i in 0..n-1:
        dst[i] = src[i] * (float32)facc

    Reads rd as element count (integer), rs1 as source address,
    rs2 as destination address. Scale factor is facc rounded to float32.
    The accumulator is NOT modified.
    """
    n = regs.read(inst.rd)
    addr_src = regs.read(inst.rs1)
    addr_dst = regs.read(inst.rs2)
    # Round facc to float32 for scaling (handles overflow to +/-inf)
    scale_bits = _f64_to_f32_bits(npu.facc)
    scale = struct.unpack('<f', struct.pack('<I', scale_bits))[0]
    for i in range(n):
        src_addr = (addr_src + i * 4) & 0xFFFFFFFF
        dst_addr = (addr_dst + i * 4) & 0xFFFFFFFF
        val = _read_mem_f32(mem, src_addr)
        _write_mem_f32(mem, dst_addr, val * scale)


def _exec_fvreduce(
    inst: Instruction,
    regs: RegisterFile,
    fregs: FRegisterFile,
    mem: MemoryBus,
) -> None:
    """NPU.FVREDUCE: f[rd] = sum(mem_f32[rs1..+n]).

    Sums n float32 values from memory using double-precision accumulation,
    then rounds to float32 and writes to f[rd]. Count from integer register rs2,
    source address from integer register rs1.
    """
    n = regs.read(inst.rs2)
    addr_src = regs.read(inst.rs1)
    total = 0.0  # double-precision accumulation
    for i in range(n):
        val = _read_mem_f32(mem, (addr_src + i * 4) & 0xFFFFFFFF)
        total += float(val)
    fregs.write_bits(inst.rd, _f64_to_f32_bits(total))


def _exec_fvmax(
    inst: Instruction,
    regs: RegisterFile,
    fregs: FRegisterFile,
    mem: MemoryBus,
) -> None:
    """NPU.FVMAX: f[rd] = max(mem_f32[rs1..+n]).

    Finds the maximum float32 value from n elements in memory.
    Returns -inf if count is 0. NaN elements are propagated.
    Count from integer register rs2, source address from integer register rs1.
    """
    n = regs.read(inst.rs2)
    addr_src = regs.read(inst.rs1)
    if n == 0:
        fregs.write_float(inst.rd, float('-inf'))
        return
    max_val = float('-inf')
    for i in range(n):
        val = _read_mem_f32(mem, (addr_src + i * 4) & 0xFFFFFFFF)
        if math.isnan(val):
            fregs.write_float(inst.rd, float('nan'))
            return
        if val > max_val:
            max_val = val
    fregs.write_float(inst.rd, max_val)
