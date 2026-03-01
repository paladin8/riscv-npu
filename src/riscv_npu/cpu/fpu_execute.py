"""FPU instruction execution: implements all RV32F (single-precision float) operations."""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING

from .decode import (
    Instruction,
    to_signed,
    OP_LOAD_FP,
    OP_STORE_FP,
    OP_FMADD,
    OP_FMSUB,
    OP_FNMSUB,
    OP_FNMADD,
    OP_OP_FP,
)

if TYPE_CHECKING:
    from .cpu import CPU

# Canonical NaN (quiet NaN with payload 0)
CANONICAL_NAN_BITS = 0x7FC00000

# Sign bit mask
SIGN_BIT = 0x80000000


def _float_to_bits(f: float) -> int:
    """Convert Python float to IEEE 754 single-precision bits."""
    return struct.unpack('<I', struct.pack('<f', f))[0]


def _bits_to_float(bits: int) -> float:
    """Convert IEEE 754 single-precision bits to Python float."""
    return struct.unpack('<f', struct.pack('<I', bits & 0xFFFFFFFF))[0]


def _round_to_single(value: float) -> float:
    """Round a double-precision value to single-precision."""
    return _bits_to_float(_float_to_bits(value))


def _round_to_single_with_flags(value: float, fpu: "FpuState") -> float:
    """Round to single-precision and set NX flag if inexact."""
    from .fpu import FpuState  # noqa: F811

    result = _round_to_single(value)
    # Inexact if the double-precision value differs from single-precision
    if not math.isnan(value) and not math.isinf(value) and value != result:
        fpu.set_flags(nx=True)
    return result


def _is_snan(bits: int) -> bool:
    """Check if bits represent a signaling NaN (exponent=0xFF, MSB of mantissa=0, mantissa!=0)."""
    exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    return exp == 0xFF and mantissa != 0 and (mantissa & 0x400000) == 0


def _is_nan(bits: int) -> bool:
    """Check if bits represent any NaN."""
    exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    return exp == 0xFF and mantissa != 0


def _classify(bits: int) -> int:
    """Classify a float value into one of 10 categories (FCLASS.S result).

    Returns a 10-bit mask:
      bit 0: negative infinity
      bit 1: negative normal
      bit 2: negative subnormal
      bit 3: negative zero
      bit 4: positive zero
      bit 5: positive subnormal
      bit 6: positive normal
      bit 7: positive infinity
      bit 8: signaling NaN
      bit 9: quiet NaN
    """
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF

    if exp == 0xFF:
        if mantissa == 0:
            # Infinity
            return 1 << 0 if sign else 1 << 7
        elif mantissa & 0x400000:
            # Quiet NaN
            return 1 << 9
        else:
            # Signaling NaN
            return 1 << 8
    elif exp == 0:
        if mantissa == 0:
            # Zero
            return 1 << 3 if sign else 1 << 4
        else:
            # Subnormal
            return 1 << 2 if sign else 1 << 5
    else:
        # Normal
        return 1 << 1 if sign else 1 << 6


def execute_fpu(inst: Instruction, cpu: CPU) -> int:
    """Execute a floating-point instruction. Returns the next PC value."""
    pc = cpu.pc

    if inst.opcode == OP_LOAD_FP:
        return _exec_flw(inst, cpu, pc)
    elif inst.opcode == OP_STORE_FP:
        return _exec_fsw(inst, cpu, pc)
    elif inst.opcode == OP_FMADD:
        return _exec_fmadd(inst, cpu, pc)
    elif inst.opcode == OP_FMSUB:
        return _exec_fmsub(inst, cpu, pc)
    elif inst.opcode == OP_FNMSUB:
        return _exec_fnmsub(inst, cpu, pc)
    elif inst.opcode == OP_FNMADD:
        return _exec_fnmadd(inst, cpu, pc)
    elif inst.opcode == OP_OP_FP:
        return _exec_op_fp(inst, cpu, pc)
    else:
        raise ValueError(f"Unknown FPU opcode: 0x{inst.opcode:02X}")


def _exec_flw(inst: Instruction, cpu: CPU, pc: int) -> int:
    """FLW: f[rd] = mem[x[rs1] + imm]."""
    addr = (cpu.registers.read(inst.rs1) + to_signed(inst.imm)) & 0xFFFFFFFF
    bits = cpu.memory.read32(addr)
    cpu.fpu_state.fregs.write_bits(inst.rd, bits)
    return pc + 4


def _exec_fsw(inst: Instruction, cpu: CPU, pc: int) -> int:
    """FSW: mem[x[rs1] + imm] = f[rs2]."""
    addr = (cpu.registers.read(inst.rs1) + to_signed(inst.imm)) & 0xFFFFFFFF
    bits = cpu.fpu_state.fregs.read_bits(inst.rs2)
    cpu.memory.write32(addr, bits)
    return pc + 4


def _exec_fmadd(inst: Instruction, cpu: CPU, pc: int) -> int:
    """FMADD.S: f[rd] = f[rs1] * f[rs2] + f[rs3]."""
    fpu = cpu.fpu_state
    a = fpu.fregs.read_float(inst.rs1)
    b = fpu.fregs.read_float(inst.rs2)
    c = fpu.fregs.read_float(inst.rs3)
    if _check_nan_3(inst, fpu):
        return pc + 4
    result = _round_to_single_with_flags(a * b + c, fpu)
    _write_float_result(fpu, inst.rd, result)
    return pc + 4


def _exec_fmsub(inst: Instruction, cpu: CPU, pc: int) -> int:
    """FMSUB.S: f[rd] = f[rs1] * f[rs2] - f[rs3]."""
    fpu = cpu.fpu_state
    a = fpu.fregs.read_float(inst.rs1)
    b = fpu.fregs.read_float(inst.rs2)
    c = fpu.fregs.read_float(inst.rs3)
    if _check_nan_3(inst, fpu):
        return pc + 4
    result = _round_to_single_with_flags(a * b - c, fpu)
    _write_float_result(fpu, inst.rd, result)
    return pc + 4


def _exec_fnmsub(inst: Instruction, cpu: CPU, pc: int) -> int:
    """FNMSUB.S: f[rd] = -(f[rs1] * f[rs2]) + f[rs3]."""
    fpu = cpu.fpu_state
    a = fpu.fregs.read_float(inst.rs1)
    b = fpu.fregs.read_float(inst.rs2)
    c = fpu.fregs.read_float(inst.rs3)
    if _check_nan_3(inst, fpu):
        return pc + 4
    result = _round_to_single_with_flags(-(a * b) + c, fpu)
    _write_float_result(fpu, inst.rd, result)
    return pc + 4


def _exec_fnmadd(inst: Instruction, cpu: CPU, pc: int) -> int:
    """FNMADD.S: f[rd] = -(f[rs1] * f[rs2]) - f[rs3]."""
    fpu = cpu.fpu_state
    a = fpu.fregs.read_float(inst.rs1)
    b = fpu.fregs.read_float(inst.rs2)
    c = fpu.fregs.read_float(inst.rs3)
    if _check_nan_3(inst, fpu):
        return pc + 4
    result = _round_to_single_with_flags(-(a * b) - c, fpu)
    _write_float_result(fpu, inst.rd, result)
    return pc + 4


def _check_nan_3(inst: Instruction, fpu: "FpuState") -> bool:
    """Check for NaN inputs in a 3-operand instruction. Sets NV flag for sNaN.

    Returns True if any input is NaN (result written as canonical NaN).
    """
    from .fpu import FpuState  # noqa: F811

    a_bits = fpu.fregs.read_bits(inst.rs1)
    b_bits = fpu.fregs.read_bits(inst.rs2)
    c_bits = fpu.fregs.read_bits(inst.rs3)

    has_snan = _is_snan(a_bits) or _is_snan(b_bits) or _is_snan(c_bits)
    has_nan = _is_nan(a_bits) or _is_nan(b_bits) or _is_nan(c_bits)

    if has_snan:
        fpu.set_flags(nv=True)
    if has_nan:
        fpu.fregs.write_bits(inst.rd, CANONICAL_NAN_BITS)
        return True
    return False


def _write_float_result(fpu: "FpuState", rd: int, result: float) -> None:
    """Write a float result, handling NaN canonicalization and NV flag for invalid ops."""
    if math.isnan(result):
        fpu.set_flags(nv=True)
        fpu.fregs.write_bits(rd, CANONICAL_NAN_BITS)
    else:
        fpu.fregs.write_float(rd, result)


def _exec_op_fp(inst: Instruction, cpu: CPU, pc: int) -> int:
    """Execute OP-FP instructions (opcode 0x53), dispatched by funct7."""
    fpu = cpu.fpu_state
    f7 = inst.funct7
    f3 = inst.funct3

    if f7 == 0x00:
        _exec_fadd(inst, fpu)
    elif f7 == 0x04:
        _exec_fsub(inst, fpu)
    elif f7 == 0x08:
        _exec_fmul(inst, fpu)
    elif f7 == 0x0C:
        _exec_fdiv(inst, fpu)
    elif f7 == 0x2C:
        _exec_fsqrt(inst, fpu)
    elif f7 == 0x10:
        _exec_fsgnj(inst, fpu, f3)
    elif f7 == 0x14:
        _exec_fminmax(inst, fpu, f3)
    elif f7 == 0x50:
        _exec_fcmp(inst, fpu, cpu, f3)
    elif f7 == 0x60:
        _exec_fcvt_w(inst, fpu, cpu)
    elif f7 == 0x68:
        _exec_fcvt_s(inst, fpu, cpu)
    elif f7 == 0x70:
        if f3 == 0:
            _exec_fmv_x_w(inst, fpu, cpu)
        elif f3 == 1:
            _exec_fclass(inst, fpu, cpu)
        else:
            raise ValueError(f"Unknown OP-FP funct3 for funct7=0x70: {f3}")
    elif f7 == 0x78:
        _exec_fmv_w_x(inst, fpu, cpu)
    else:
        raise ValueError(f"Unknown OP-FP funct7: 0x{f7:02X}")

    return pc + 4


def _exec_fadd(inst: Instruction, fpu: "FpuState") -> None:
    """FADD.S: f[rd] = f[rs1] + f[rs2]."""
    a_bits = fpu.fregs.read_bits(inst.rs1)
    b_bits = fpu.fregs.read_bits(inst.rs2)
    if _is_snan(a_bits) or _is_snan(b_bits):
        fpu.set_flags(nv=True)
    if _is_nan(a_bits) or _is_nan(b_bits):
        fpu.fregs.write_bits(inst.rd, CANONICAL_NAN_BITS)
        return
    a = fpu.fregs.read_float(inst.rs1)
    b = fpu.fregs.read_float(inst.rs2)
    result = _round_to_single_with_flags(a + b, fpu)
    _write_float_result(fpu, inst.rd, result)


def _exec_fsub(inst: Instruction, fpu: "FpuState") -> None:
    """FSUB.S: f[rd] = f[rs1] - f[rs2]."""
    a_bits = fpu.fregs.read_bits(inst.rs1)
    b_bits = fpu.fregs.read_bits(inst.rs2)
    if _is_snan(a_bits) or _is_snan(b_bits):
        fpu.set_flags(nv=True)
    if _is_nan(a_bits) or _is_nan(b_bits):
        fpu.fregs.write_bits(inst.rd, CANONICAL_NAN_BITS)
        return
    a = fpu.fregs.read_float(inst.rs1)
    b = fpu.fregs.read_float(inst.rs2)
    result = _round_to_single_with_flags(a - b, fpu)
    _write_float_result(fpu, inst.rd, result)


def _exec_fmul(inst: Instruction, fpu: "FpuState") -> None:
    """FMUL.S: f[rd] = f[rs1] * f[rs2]."""
    a_bits = fpu.fregs.read_bits(inst.rs1)
    b_bits = fpu.fregs.read_bits(inst.rs2)
    if _is_snan(a_bits) or _is_snan(b_bits):
        fpu.set_flags(nv=True)
    if _is_nan(a_bits) or _is_nan(b_bits):
        fpu.fregs.write_bits(inst.rd, CANONICAL_NAN_BITS)
        return
    a = fpu.fregs.read_float(inst.rs1)
    b = fpu.fregs.read_float(inst.rs2)
    result = _round_to_single_with_flags(a * b, fpu)
    _write_float_result(fpu, inst.rd, result)


def _exec_fdiv(inst: Instruction, fpu: "FpuState") -> None:
    """FDIV.S: f[rd] = f[rs1] / f[rs2]."""
    a_bits = fpu.fregs.read_bits(inst.rs1)
    b_bits = fpu.fregs.read_bits(inst.rs2)
    if _is_snan(a_bits) or _is_snan(b_bits):
        fpu.set_flags(nv=True)
    if _is_nan(a_bits) or _is_nan(b_bits):
        fpu.fregs.write_bits(inst.rd, CANONICAL_NAN_BITS)
        return

    a = fpu.fregs.read_float(inst.rs1)
    b = fpu.fregs.read_float(inst.rs2)

    if b == 0.0:
        if a == 0.0:
            # 0/0 = NaN, invalid
            fpu.set_flags(nv=True)
            fpu.fregs.write_bits(inst.rd, CANONICAL_NAN_BITS)
        else:
            # x/0 = +-inf, divide by zero
            fpu.set_flags(dz=True)
            sign = (a_bits ^ b_bits) & SIGN_BIT
            fpu.fregs.write_bits(inst.rd, sign | 0x7F800000)
        return

    result = _round_to_single_with_flags(a / b, fpu)
    _write_float_result(fpu, inst.rd, result)


def _exec_fsqrt(inst: Instruction, fpu: "FpuState") -> None:
    """FSQRT.S: f[rd] = sqrt(f[rs1])."""
    a_bits = fpu.fregs.read_bits(inst.rs1)
    if _is_snan(a_bits):
        fpu.set_flags(nv=True)
    if _is_nan(a_bits):
        fpu.fregs.write_bits(inst.rd, CANONICAL_NAN_BITS)
        return

    a = fpu.fregs.read_float(inst.rs1)

    if a < 0.0:
        fpu.set_flags(nv=True)
        fpu.fregs.write_bits(inst.rd, CANONICAL_NAN_BITS)
        return

    # -0.0 -> -0.0
    if a == 0.0:
        fpu.fregs.write_bits(inst.rd, a_bits)
        return

    result = _round_to_single_with_flags(math.sqrt(a), fpu)
    _write_float_result(fpu, inst.rd, result)


def _exec_fsgnj(inst: Instruction, fpu: "FpuState", funct3: int) -> None:
    """Sign injection: FSGNJ.S, FSGNJN.S, FSGNJX.S."""
    rs1_bits = fpu.fregs.read_bits(inst.rs1)
    rs2_bits = fpu.fregs.read_bits(inst.rs2)

    body = rs1_bits & 0x7FFFFFFF  # magnitude of rs1

    if funct3 == 0:  # FSGNJ: sign from rs2
        sign = rs2_bits & SIGN_BIT
    elif funct3 == 1:  # FSGNJN: negated sign from rs2
        sign = (~rs2_bits) & SIGN_BIT
    elif funct3 == 2:  # FSGNJX: XOR signs
        sign = (rs1_bits ^ rs2_bits) & SIGN_BIT
    else:
        raise ValueError(f"Unknown FSGNJ funct3: {funct3}")

    fpu.fregs.write_bits(inst.rd, sign | body)


def _exec_fminmax(inst: Instruction, fpu: "FpuState", funct3: int) -> None:
    """FMIN.S / FMAX.S."""
    a_bits = fpu.fregs.read_bits(inst.rs1)
    b_bits = fpu.fregs.read_bits(inst.rs2)

    a_is_nan = _is_nan(a_bits)
    b_is_nan = _is_nan(b_bits)

    # Signal on sNaN
    if _is_snan(a_bits) or _is_snan(b_bits):
        fpu.set_flags(nv=True)

    # If both NaN, return canonical NaN
    if a_is_nan and b_is_nan:
        fpu.fregs.write_bits(inst.rd, CANONICAL_NAN_BITS)
        return

    # If one is NaN, return the other
    if a_is_nan:
        fpu.fregs.write_bits(inst.rd, b_bits)
        return
    if b_is_nan:
        fpu.fregs.write_bits(inst.rd, a_bits)
        return

    a = fpu.fregs.read_float(inst.rs1)
    b = fpu.fregs.read_float(inst.rs2)

    # RISC-V spec: -0.0 < +0.0 for min/max purposes
    if a == 0.0 and b == 0.0:
        a_neg = bool(a_bits & SIGN_BIT)
        b_neg = bool(b_bits & SIGN_BIT)
        if funct3 == 0:  # FMIN
            fpu.fregs.write_bits(inst.rd, a_bits if a_neg else b_bits)
        else:  # FMAX
            fpu.fregs.write_bits(inst.rd, b_bits if a_neg else a_bits)
        return

    if funct3 == 0:  # FMIN
        fpu.fregs.write_bits(inst.rd, a_bits if a <= b else b_bits)
    elif funct3 == 1:  # FMAX
        fpu.fregs.write_bits(inst.rd, a_bits if a >= b else b_bits)
    else:
        raise ValueError(f"Unknown FMIN/FMAX funct3: {funct3}")


def _exec_fcmp(
    inst: Instruction, fpu: "FpuState", cpu: "CPU", funct3: int
) -> None:
    """FEQ.S, FLT.S, FLE.S: result goes to integer register x[rd]."""
    a_bits = fpu.fregs.read_bits(inst.rs1)
    b_bits = fpu.fregs.read_bits(inst.rs2)

    a_is_nan = _is_nan(a_bits)
    b_is_nan = _is_nan(b_bits)

    if funct3 == 2:  # FEQ
        # FEQ: only signals NV on signaling NaN
        if _is_snan(a_bits) or _is_snan(b_bits):
            fpu.set_flags(nv=True)
        if a_is_nan or b_is_nan:
            cpu.registers.write(inst.rd, 0)
            return
        a = fpu.fregs.read_float(inst.rs1)
        b = fpu.fregs.read_float(inst.rs2)
        cpu.registers.write(inst.rd, 1 if a == b else 0)

    elif funct3 == 1:  # FLT
        # FLT: signals NV on any NaN
        if a_is_nan or b_is_nan:
            fpu.set_flags(nv=True)
            cpu.registers.write(inst.rd, 0)
            return
        a = fpu.fregs.read_float(inst.rs1)
        b = fpu.fregs.read_float(inst.rs2)
        cpu.registers.write(inst.rd, 1 if a < b else 0)

    elif funct3 == 0:  # FLE
        # FLE: signals NV on any NaN
        if a_is_nan or b_is_nan:
            fpu.set_flags(nv=True)
            cpu.registers.write(inst.rd, 0)
            return
        a = fpu.fregs.read_float(inst.rs1)
        b = fpu.fregs.read_float(inst.rs2)
        cpu.registers.write(inst.rd, 1 if a <= b else 0)

    else:
        raise ValueError(f"Unknown FCMP funct3: {funct3}")


def _exec_fcvt_w(inst: Instruction, fpu: "FpuState", cpu: "CPU") -> None:
    """FCVT.W.S / FCVT.WU.S: convert float to integer."""
    a_bits = fpu.fregs.read_bits(inst.rs1)

    if _is_nan(a_bits):
        fpu.set_flags(nv=True)
        if inst.rs2 == 0:  # FCVT.W.S: NaN -> INT_MAX
            cpu.registers.write(inst.rd, 0x7FFFFFFF)
        else:  # FCVT.WU.S: NaN -> UINT_MAX
            cpu.registers.write(inst.rd, 0xFFFFFFFF)
        return

    a = fpu.fregs.read_float(inst.rs1)

    if inst.rs2 == 0:  # FCVT.W.S (signed)
        if math.isinf(a):
            fpu.set_flags(nv=True)
            cpu.registers.write(inst.rd, 0x7FFFFFFF if a > 0 else 0x80000000)
            return
        # Truncate toward zero
        ival = int(a)
        if ival > 0x7FFFFFFF:
            fpu.set_flags(nv=True)
            ival = 0x7FFFFFFF
        elif ival < -0x80000000:
            fpu.set_flags(nv=True)
            ival = -0x80000000
        elif a != float(ival):
            fpu.set_flags(nx=True)
        cpu.registers.write(inst.rd, ival & 0xFFFFFFFF)

    elif inst.rs2 == 1:  # FCVT.WU.S (unsigned)
        if math.isinf(a):
            fpu.set_flags(nv=True)
            cpu.registers.write(inst.rd, 0xFFFFFFFF if a > 0 else 0)
            return
        # Truncate toward zero
        ival = int(a)
        if ival < 0:
            fpu.set_flags(nv=True)
            cpu.registers.write(inst.rd, 0)
            return
        if ival > 0xFFFFFFFF:
            fpu.set_flags(nv=True)
            cpu.registers.write(inst.rd, 0xFFFFFFFF)
            return
        if a != float(ival):
            fpu.set_flags(nx=True)
        cpu.registers.write(inst.rd, ival & 0xFFFFFFFF)
    else:
        raise ValueError(f"Unknown FCVT.W rs2: {inst.rs2}")


def _exec_fcvt_s(inst: Instruction, fpu: "FpuState", cpu: "CPU") -> None:
    """FCVT.S.W / FCVT.S.WU: convert integer to float."""
    x_val = cpu.registers.read(inst.rs1)

    if inst.rs2 == 0:  # FCVT.S.W (signed)
        signed_val = to_signed(x_val)
        result = _round_to_single(float(signed_val))
        fpu.fregs.write_float(inst.rd, result)
    elif inst.rs2 == 1:  # FCVT.S.WU (unsigned)
        result = _round_to_single(float(x_val))
        fpu.fregs.write_float(inst.rd, result)
    else:
        raise ValueError(f"Unknown FCVT.S rs2: {inst.rs2}")


def _exec_fmv_x_w(inst: Instruction, fpu: "FpuState", cpu: "CPU") -> None:
    """FMV.X.W: x[rd] = f[rs1] (bitwise move, float reg -> int reg)."""
    bits = fpu.fregs.read_bits(inst.rs1)
    cpu.registers.write(inst.rd, bits)


def _exec_fmv_w_x(inst: Instruction, fpu: "FpuState", cpu: "CPU") -> None:
    """FMV.W.X: f[rd] = x[rs1] (bitwise move, int reg -> float reg)."""
    bits = cpu.registers.read(inst.rs1)
    fpu.fregs.write_bits(inst.rd, bits)


def _exec_fclass(inst: Instruction, fpu: "FpuState", cpu: "CPU") -> None:
    """FCLASS.S: x[rd] = classify(f[rs1])."""
    bits = fpu.fregs.read_bits(inst.rs1)
    cpu.registers.write(inst.rd, _classify(bits))
