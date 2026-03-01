"""Tests for RV32F (single-precision floating-point) instructions."""

import math
import struct

import pytest

from riscv_npu.cpu.cpu import CPU
from riscv_npu.cpu.fpu import FRegisterFile, FpuState, CSR_FFLAGS, CSR_FRM, CSR_FCSR
from riscv_npu.cpu.decode import (
    OP_LOAD_FP, OP_STORE_FP, OP_FMADD, OP_FMSUB,
    OP_FNMSUB, OP_FNMADD, OP_OP_FP,
)
from riscv_npu.memory.bus import MemoryBus
from riscv_npu.memory.ram import RAM

BASE = 0x80000000
RAM_SIZE = 1024 * 1024


def _make_cpu() -> CPU:
    """Create a fresh CPU with 1MB RAM."""
    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    bus.register(BASE, RAM_SIZE, ram)
    cpu = CPU(bus)
    cpu.pc = BASE
    return cpu


def _float_bits(f: float) -> int:
    """Convert float to IEEE 754 single-precision bits."""
    return struct.unpack('<I', struct.pack('<f', f))[0]


def _bits_float(bits: int) -> float:
    """Convert IEEE 754 single-precision bits to float."""
    return struct.unpack('<f', struct.pack('<I', bits & 0xFFFFFFFF))[0]


# Instruction encoders

def _r_type(opcode: int, rd: int, funct3: int, rs1: int, rs2: int, funct7: int) -> int:
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def _r4_type(opcode: int, rd: int, funct3: int, rs1: int, rs2: int, rs3: int, fmt: int = 0) -> int:
    return (rs3 << 27) | (fmt << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def _i_type(opcode: int, rd: int, funct3: int, rs1: int, imm: int) -> int:
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def _s_type(opcode: int, funct3: int, rs1: int, rs2: int, imm: int) -> int:
    imm_11_5 = (imm >> 5) & 0x7F
    imm_4_0 = imm & 0x1F
    return (imm_11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_4_0 << 7) | opcode


def _exec(cpu: CPU, word: int) -> None:
    """Write instruction at PC and step."""
    cpu.memory.write32(cpu.pc, word)
    cpu.step()


# ============================================================
# FRegisterFile tests
# ============================================================

class TestFRegisterFile:
    """Test float register file read/write operations."""

    def test_initial_zero(self) -> None:
        fregs = FRegisterFile()
        for i in range(32):
            assert fregs.read_bits(i) == 0

    def test_write_read_bits(self) -> None:
        fregs = FRegisterFile()
        fregs.write_bits(0, 0x3F800000)  # 1.0
        assert fregs.read_bits(0) == 0x3F800000

    def test_write_read_float(self) -> None:
        fregs = FRegisterFile()
        fregs.write_float(5, 3.14)
        val = fregs.read_float(5)
        assert abs(val - 3.14) < 1e-6

    def test_f0_is_writable(self) -> None:
        """Unlike x0, f0 is NOT hardwired to zero."""
        fregs = FRegisterFile()
        fregs.write_float(0, 42.0)
        assert fregs.read_float(0) == 42.0

    def test_all_32_registers(self) -> None:
        fregs = FRegisterFile()
        for i in range(32):
            fregs.write_float(i, float(i))
        for i in range(32):
            assert fregs.read_float(i) == float(i)

    def test_nan_roundtrip(self) -> None:
        fregs = FRegisterFile()
        fregs.write_bits(1, 0x7FC00000)  # canonical NaN
        assert math.isnan(fregs.read_float(1))

    def test_negative_zero(self) -> None:
        fregs = FRegisterFile()
        fregs.write_bits(2, 0x80000000)  # -0.0
        val = fregs.read_float(2)
        assert val == 0.0
        assert math.copysign(1, val) == -1.0

    def test_32bit_mask(self) -> None:
        fregs = FRegisterFile()
        fregs.write_bits(3, 0x1_FFFFFFFF)
        assert fregs.read_bits(3) == 0xFFFFFFFF


# ============================================================
# FpuState tests
# ============================================================

class TestFpuState:
    """Test FPU control/status register operations."""

    def test_initial_state(self) -> None:
        fpu = FpuState()
        assert fpu.fcsr == 0
        assert fpu.fflags == 0
        assert fpu.frm == 0

    def test_fflags_set_get(self) -> None:
        fpu = FpuState()
        fpu.fflags = 0x1F
        assert fpu.fflags == 0x1F
        assert fpu.fcsr == 0x1F

    def test_frm_set_get(self) -> None:
        fpu = FpuState()
        fpu.frm = 3
        assert fpu.frm == 3
        assert fpu.fcsr == (3 << 5)

    def test_fflags_frm_independent(self) -> None:
        fpu = FpuState()
        fpu.fflags = 0x0A
        fpu.frm = 5
        assert fpu.fflags == 0x0A
        assert fpu.frm == 5

    def test_set_flags_sticky(self) -> None:
        fpu = FpuState()
        fpu.set_flags(nv=True)
        assert fpu.fflags & 0x10
        fpu.set_flags(nx=True)
        assert fpu.fflags & 0x10  # NV still set (sticky)
        assert fpu.fflags & 0x01  # NX now set

    def test_set_flags_all(self) -> None:
        fpu = FpuState()
        fpu.set_flags(nv=True, dz=True, of=True, uf=True, nx=True)
        assert fpu.fflags == 0x1F

    def test_csr_routing_fflags(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.set_flags(nv=True, nx=True)
        assert cpu.csr_read(CSR_FFLAGS) == 0x11

    def test_csr_routing_frm(self) -> None:
        cpu = _make_cpu()
        cpu.csr_write(CSR_FRM, 4)
        assert cpu.fpu_state.frm == 4

    def test_csr_routing_fcsr(self) -> None:
        cpu = _make_cpu()
        cpu.csr_write(CSR_FCSR, 0xAB)
        assert cpu.fpu_state.fcsr == 0xAB


# ============================================================
# Arithmetic instructions
# ============================================================

class TestFaddS:
    """FADD.S tests."""

    def test_basic_add(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 1.5)
        cpu.fpu_state.fregs.write_float(2, 2.5)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x00))
        assert cpu.fpu_state.fregs.read_float(3) == 4.0

    def test_add_negative(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, -1.0)
        cpu.fpu_state.fregs.write_float(2, 3.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x00))
        assert cpu.fpu_state.fregs.read_float(3) == 2.0

    def test_add_inf(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, float('inf'))
        cpu.fpu_state.fregs.write_float(2, 1.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x00))
        assert math.isinf(cpu.fpu_state.fregs.read_float(3))

    def test_add_nan_produces_canonical_nan(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_bits(1, 0x7FC00000)  # NaN
        cpu.fpu_state.fregs.write_float(2, 1.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x00))
        assert cpu.fpu_state.fregs.read_bits(3) == 0x7FC00000


class TestFsubS:
    """FSUB.S tests."""

    def test_basic_sub(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 5.0)
        cpu.fpu_state.fregs.write_float(2, 3.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x04))
        assert cpu.fpu_state.fregs.read_float(3) == 2.0

    def test_sub_to_zero(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 7.0)
        cpu.fpu_state.fregs.write_float(2, 7.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x04))
        assert cpu.fpu_state.fregs.read_float(3) == 0.0


class TestFmulS:
    """FMUL.S tests."""

    def test_basic_mul(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 3.0)
        cpu.fpu_state.fregs.write_float(2, 4.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x08))
        assert cpu.fpu_state.fregs.read_float(3) == 12.0

    def test_mul_by_zero(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 100.0)
        cpu.fpu_state.fregs.write_float(2, 0.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x08))
        assert cpu.fpu_state.fregs.read_float(3) == 0.0


class TestFdivS:
    """FDIV.S tests."""

    def test_basic_div(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 10.0)
        cpu.fpu_state.fregs.write_float(2, 4.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x0C))
        assert cpu.fpu_state.fregs.read_float(3) == 2.5

    def test_div_by_zero_positive(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 1.0)
        cpu.fpu_state.fregs.write_float(2, 0.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x0C))
        assert math.isinf(cpu.fpu_state.fregs.read_float(3))
        assert cpu.fpu_state.fregs.read_float(3) > 0
        assert cpu.fpu_state.fflags & 0x08  # DZ flag

    def test_div_zero_by_zero(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 0.0)
        cpu.fpu_state.fregs.write_float(2, 0.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x0C))
        assert math.isnan(cpu.fpu_state.fregs.read_float(3))
        assert cpu.fpu_state.fflags & 0x10  # NV flag


class TestFsqrtS:
    """FSQRT.S tests."""

    def test_basic_sqrt(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 4.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=2, funct3=0, rs1=1, rs2=0, funct7=0x2C))
        assert cpu.fpu_state.fregs.read_float(2) == 2.0

    def test_sqrt_negative(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, -1.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=2, funct3=0, rs1=1, rs2=0, funct7=0x2C))
        assert math.isnan(cpu.fpu_state.fregs.read_float(2))
        assert cpu.fpu_state.fflags & 0x10  # NV flag

    def test_sqrt_zero(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 0.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=2, funct3=0, rs1=1, rs2=0, funct7=0x2C))
        assert cpu.fpu_state.fregs.read_float(2) == 0.0

    def test_sqrt_neg_zero(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_bits(1, 0x80000000)  # -0.0
        _exec(cpu, _r_type(OP_OP_FP, rd=2, funct3=0, rs1=1, rs2=0, funct7=0x2C))
        assert cpu.fpu_state.fregs.read_bits(2) == 0x80000000


# ============================================================
# Sign injection
# ============================================================

class TestFsgnjS:
    """FSGNJ.S / FSGNJN.S / FSGNJX.S tests."""

    def test_fsgnj_positive(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, -3.0)
        cpu.fpu_state.fregs.write_float(2, 1.0)  # positive sign
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x10))
        assert cpu.fpu_state.fregs.read_float(3) == 3.0

    def test_fsgnj_negative(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 3.0)
        cpu.fpu_state.fregs.write_float(2, -1.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x10))
        assert cpu.fpu_state.fregs.read_float(3) == -3.0

    def test_fsgnjn(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 5.0)
        cpu.fpu_state.fregs.write_float(2, 1.0)  # positive -> negated = negative
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=1, rs1=1, rs2=2, funct7=0x10))
        assert cpu.fpu_state.fregs.read_float(3) == -5.0

    def test_fsgnjx_same_sign(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, -2.0)
        cpu.fpu_state.fregs.write_float(2, -3.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=2, rs1=1, rs2=2, funct7=0x10))
        # XOR of two negative signs = positive
        assert cpu.fpu_state.fregs.read_float(3) == 2.0

    def test_fsgnjx_diff_sign(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 2.0)
        cpu.fpu_state.fregs.write_float(2, -3.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=2, rs1=1, rs2=2, funct7=0x10))
        assert cpu.fpu_state.fregs.read_float(3) == -2.0


# ============================================================
# Min / Max
# ============================================================

class TestFminmaxS:
    """FMIN.S / FMAX.S tests."""

    def test_fmin_basic(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 2.0)
        cpu.fpu_state.fregs.write_float(2, 3.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x14))
        assert cpu.fpu_state.fregs.read_float(3) == 2.0

    def test_fmax_basic(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 2.0)
        cpu.fpu_state.fregs.write_float(2, 3.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=1, rs1=1, rs2=2, funct7=0x14))
        assert cpu.fpu_state.fregs.read_float(3) == 3.0

    def test_fmin_nan_returns_other(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_bits(1, 0x7FC00000)  # NaN
        cpu.fpu_state.fregs.write_float(2, 5.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x14))
        assert cpu.fpu_state.fregs.read_float(3) == 5.0

    def test_fmax_both_nan(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_bits(1, 0x7FC00000)
        cpu.fpu_state.fregs.write_bits(2, 0x7FC00000)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=1, rs1=1, rs2=2, funct7=0x14))
        assert cpu.fpu_state.fregs.read_bits(3) == 0x7FC00000

    def test_fmin_negative_zero(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_bits(1, 0x80000000)  # -0.0
        cpu.fpu_state.fregs.write_bits(2, 0x00000000)  # +0.0
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x14))
        assert cpu.fpu_state.fregs.read_bits(3) == 0x80000000  # -0.0

    def test_fmax_negative_zero(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_bits(1, 0x80000000)  # -0.0
        cpu.fpu_state.fregs.write_bits(2, 0x00000000)  # +0.0
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=1, rs1=1, rs2=2, funct7=0x14))
        assert cpu.fpu_state.fregs.read_bits(3) == 0x00000000  # +0.0


# ============================================================
# Comparisons
# ============================================================

class TestFcmpS:
    """FEQ.S / FLT.S / FLE.S tests."""

    def test_feq_equal(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 3.0)
        cpu.fpu_state.fregs.write_float(2, 3.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=2, rs1=1, rs2=2, funct7=0x50))
        assert cpu.registers.read(3) == 1

    def test_feq_not_equal(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 3.0)
        cpu.fpu_state.fregs.write_float(2, 4.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=2, rs1=1, rs2=2, funct7=0x50))
        assert cpu.registers.read(3) == 0

    def test_feq_nan(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_bits(1, 0x7FC00000)
        cpu.fpu_state.fregs.write_float(2, 3.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=2, rs1=1, rs2=2, funct7=0x50))
        assert cpu.registers.read(3) == 0

    def test_flt_less(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 2.0)
        cpu.fpu_state.fregs.write_float(2, 3.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=1, rs1=1, rs2=2, funct7=0x50))
        assert cpu.registers.read(3) == 1

    def test_flt_not_less(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 3.0)
        cpu.fpu_state.fregs.write_float(2, 3.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=1, rs1=1, rs2=2, funct7=0x50))
        assert cpu.registers.read(3) == 0

    def test_flt_nan_sets_nv(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_bits(1, 0x7FC00000)
        cpu.fpu_state.fregs.write_float(2, 3.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=1, rs1=1, rs2=2, funct7=0x50))
        assert cpu.registers.read(3) == 0
        assert cpu.fpu_state.fflags & 0x10  # NV

    def test_fle_equal(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 3.0)
        cpu.fpu_state.fregs.write_float(2, 3.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x50))
        assert cpu.registers.read(3) == 1

    def test_fle_less(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 2.0)
        cpu.fpu_state.fregs.write_float(2, 3.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x50))
        assert cpu.registers.read(3) == 1

    def test_fle_greater(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 5.0)
        cpu.fpu_state.fregs.write_float(2, 3.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x50))
        assert cpu.registers.read(3) == 0


# ============================================================
# Convert float <-> int
# ============================================================

class TestFcvtWS:
    """FCVT.W.S / FCVT.WU.S tests."""

    def test_fcvt_w_s_positive(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 42.7)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=0, funct7=0x60))
        assert cpu.registers.read(3) == 42  # truncated

    def test_fcvt_w_s_negative(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, -3.9)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=0, funct7=0x60))
        val = cpu.registers.read(3)
        # Should be -3 in two's complement
        assert val == (-3) & 0xFFFFFFFF

    def test_fcvt_w_s_overflow(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 3e9)  # > INT_MAX
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=0, funct7=0x60))
        assert cpu.registers.read(3) == 0x7FFFFFFF
        assert cpu.fpu_state.fflags & 0x10  # NV

    def test_fcvt_w_s_nan(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_bits(1, 0x7FC00000)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=0, funct7=0x60))
        assert cpu.registers.read(3) == 0x7FFFFFFF
        assert cpu.fpu_state.fflags & 0x10  # NV

    def test_fcvt_wu_s_positive(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 42.3)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=1, funct7=0x60))
        assert cpu.registers.read(3) == 42

    def test_fcvt_wu_s_negative(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, -1.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=1, funct7=0x60))
        assert cpu.registers.read(3) == 0
        assert cpu.fpu_state.fflags & 0x10  # NV

    def test_fcvt_wu_s_nan(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_bits(1, 0x7FC00000)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=1, funct7=0x60))
        assert cpu.registers.read(3) == 0xFFFFFFFF


class TestFcvtSW:
    """FCVT.S.W / FCVT.S.WU tests."""

    def test_fcvt_s_w_positive(self) -> None:
        cpu = _make_cpu()
        cpu.registers.write(1, 42)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=0, funct7=0x68))
        assert cpu.fpu_state.fregs.read_float(3) == 42.0

    def test_fcvt_s_w_negative(self) -> None:
        cpu = _make_cpu()
        cpu.registers.write(1, (-10) & 0xFFFFFFFF)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=0, funct7=0x68))
        assert cpu.fpu_state.fregs.read_float(3) == -10.0

    def test_fcvt_s_wu(self) -> None:
        cpu = _make_cpu()
        cpu.registers.write(1, 0xFFFFFFFF)  # 4294967295 as unsigned
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=1, funct7=0x68))
        # Should be approximately 4294967296.0 (rounded in single-precision)
        val = cpu.fpu_state.fregs.read_float(3)
        assert val > 4e9


# ============================================================
# Move (bitwise)
# ============================================================

class TestFmvS:
    """FMV.X.W / FMV.W.X tests."""

    def test_fmv_x_w(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_bits(1, 0x40490FDB)  # pi
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=0, funct7=0x70))
        assert cpu.registers.read(3) == 0x40490FDB

    def test_fmv_w_x(self) -> None:
        cpu = _make_cpu()
        cpu.registers.write(1, 0x3F800000)  # 1.0
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=0, funct7=0x78))
        assert cpu.fpu_state.fregs.read_float(3) == 1.0

    def test_fmv_roundtrip(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, -2.5)
        bits = cpu.fpu_state.fregs.read_bits(1)
        # FMV.X.W
        _exec(cpu, _r_type(OP_OP_FP, rd=5, funct3=0, rs1=1, rs2=0, funct7=0x70))
        # FMV.W.X
        cpu.memory.write32(cpu.pc, _r_type(OP_OP_FP, rd=2, funct3=0, rs1=5, rs2=0, funct7=0x78))
        cpu.step()
        assert cpu.fpu_state.fregs.read_float(2) == -2.5


# ============================================================
# FCLASS.S
# ============================================================

class TestFclassS:
    """FCLASS.S tests for all 10 categories."""

    def _fclass(self, bits: int) -> int:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_bits(1, bits)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=1, rs1=1, rs2=0, funct7=0x70))
        return cpu.registers.read(3)

    def test_neg_inf(self) -> None:
        assert self._fclass(0xFF800000) == (1 << 0)

    def test_neg_normal(self) -> None:
        assert self._fclass(_float_bits(-1.0)) == (1 << 1)

    def test_neg_subnormal(self) -> None:
        assert self._fclass(0x80000001) == (1 << 2)

    def test_neg_zero(self) -> None:
        assert self._fclass(0x80000000) == (1 << 3)

    def test_pos_zero(self) -> None:
        assert self._fclass(0x00000000) == (1 << 4)

    def test_pos_subnormal(self) -> None:
        assert self._fclass(0x00000001) == (1 << 5)

    def test_pos_normal(self) -> None:
        assert self._fclass(_float_bits(1.0)) == (1 << 6)

    def test_pos_inf(self) -> None:
        assert self._fclass(0x7F800000) == (1 << 7)

    def test_snan(self) -> None:
        assert self._fclass(0x7F800001) == (1 << 8)

    def test_qnan(self) -> None:
        assert self._fclass(0x7FC00000) == (1 << 9)


# ============================================================
# Fused multiply-add
# ============================================================

class TestFmaddS:
    """FMADD.S tests."""

    def test_basic(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 2.0)
        cpu.fpu_state.fregs.write_float(2, 3.0)
        cpu.fpu_state.fregs.write_float(3, 1.0)
        _exec(cpu, _r4_type(OP_FMADD, rd=4, funct3=0, rs1=1, rs2=2, rs3=3))
        assert cpu.fpu_state.fregs.read_float(4) == 7.0  # 2*3 + 1

    def test_negative(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, -1.0)
        cpu.fpu_state.fregs.write_float(2, 5.0)
        cpu.fpu_state.fregs.write_float(3, 10.0)
        _exec(cpu, _r4_type(OP_FMADD, rd=4, funct3=0, rs1=1, rs2=2, rs3=3))
        assert cpu.fpu_state.fregs.read_float(4) == 5.0  # -1*5 + 10


class TestFmsubS:
    """FMSUB.S tests."""

    def test_basic(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 2.0)
        cpu.fpu_state.fregs.write_float(2, 3.0)
        cpu.fpu_state.fregs.write_float(3, 1.0)
        _exec(cpu, _r4_type(OP_FMSUB, rd=4, funct3=0, rs1=1, rs2=2, rs3=3))
        assert cpu.fpu_state.fregs.read_float(4) == 5.0  # 2*3 - 1


class TestFnmsubS:
    """FNMSUB.S tests."""

    def test_basic(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 2.0)
        cpu.fpu_state.fregs.write_float(2, 3.0)
        cpu.fpu_state.fregs.write_float(3, 10.0)
        _exec(cpu, _r4_type(OP_FNMSUB, rd=4, funct3=0, rs1=1, rs2=2, rs3=3))
        assert cpu.fpu_state.fregs.read_float(4) == 4.0  # -(2*3) + 10


class TestFnmaddS:
    """FNMADD.S tests."""

    def test_basic(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 2.0)
        cpu.fpu_state.fregs.write_float(2, 3.0)
        cpu.fpu_state.fregs.write_float(3, 1.0)
        _exec(cpu, _r4_type(OP_FNMADD, rd=4, funct3=0, rs1=1, rs2=2, rs3=3))
        assert cpu.fpu_state.fregs.read_float(4) == -7.0  # -(2*3) - 1


# ============================================================
# Load / Store
# ============================================================

class TestFLW:
    """FLW (float load word) tests."""

    def test_basic_load(self) -> None:
        cpu = _make_cpu()
        # Write float bits to memory
        data_addr = BASE + 0x100
        cpu.memory.write32(data_addr, _float_bits(3.14))
        cpu.registers.write(1, data_addr)
        _exec(cpu, _i_type(OP_LOAD_FP, rd=2, funct3=2, rs1=1, imm=0))
        assert abs(cpu.fpu_state.fregs.read_float(2) - 3.14) < 1e-6

    def test_load_with_offset(self) -> None:
        cpu = _make_cpu()
        data_addr = BASE + 0x200
        cpu.memory.write32(data_addr + 8, _float_bits(2.5))
        cpu.registers.write(1, data_addr)
        _exec(cpu, _i_type(OP_LOAD_FP, rd=3, funct3=2, rs1=1, imm=8))
        assert cpu.fpu_state.fregs.read_float(3) == 2.5


class TestFSW:
    """FSW (float store word) tests."""

    def test_basic_store(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(2, 7.5)
        data_addr = BASE + 0x100
        cpu.registers.write(1, data_addr)
        _exec(cpu, _s_type(OP_STORE_FP, funct3=2, rs1=1, rs2=2, imm=0))
        stored = cpu.memory.read32(data_addr)
        assert stored == _float_bits(7.5)

    def test_store_with_offset(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(5, -1.25)
        data_addr = BASE + 0x200
        cpu.registers.write(1, data_addr)
        _exec(cpu, _s_type(OP_STORE_FP, funct3=2, rs1=1, rs2=5, imm=12))
        stored = cpu.memory.read32(data_addr + 12)
        assert stored == _float_bits(-1.25)

    def test_load_store_roundtrip(self) -> None:
        cpu = _make_cpu()
        data_addr = BASE + 0x300
        cpu.fpu_state.fregs.write_float(1, 123.456)
        cpu.registers.write(2, data_addr)
        # Store
        _exec(cpu, _s_type(OP_STORE_FP, funct3=2, rs1=2, rs2=1, imm=0))
        # Load back
        cpu.memory.write32(cpu.pc, _i_type(OP_LOAD_FP, rd=3, funct3=2, rs1=2, imm=0))
        cpu.step()
        assert abs(cpu.fpu_state.fregs.read_float(3) - 123.456) < 0.01


# ============================================================
# FCSR flag tests
# ============================================================

class TestFcsrFlags:
    """Test that instructions set appropriate FCSR flags."""

    def test_snan_input_sets_nv(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_bits(1, 0x7F800001)  # sNaN
        cpu.fpu_state.fregs.write_float(2, 1.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x00))  # FADD
        assert cpu.fpu_state.fflags & 0x10  # NV

    def test_div_by_zero_sets_dz(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 5.0)
        cpu.fpu_state.fregs.write_float(2, 0.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x0C))  # FDIV
        assert cpu.fpu_state.fflags & 0x08  # DZ

    def test_fcvt_inexact_sets_nx(self) -> None:
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 1.5)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=0, funct7=0x60))  # FCVT.W.S
        assert cpu.fpu_state.fflags & 0x01  # NX

    def test_flags_are_sticky(self) -> None:
        cpu = _make_cpu()
        # Cause NV
        cpu.fpu_state.fregs.write_float(1, 0.0)
        cpu.fpu_state.fregs.write_float(2, 0.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x0C))  # 0/0
        assert cpu.fpu_state.fflags & 0x10  # NV set
        # Do a normal add - NV should still be set
        cpu.fpu_state.fregs.write_float(1, 1.0)
        cpu.fpu_state.fregs.write_float(2, 2.0)
        cpu.memory.write32(cpu.pc, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x00))
        cpu.step()
        assert cpu.fpu_state.fflags & 0x10  # NV still sticky


# ============================================================
# PC advancement
# ============================================================

class TestPcAdvancement:
    """Ensure all FPU instructions advance PC by 4."""

    def test_fadd_advances_pc(self) -> None:
        cpu = _make_cpu()
        start_pc = cpu.pc
        cpu.fpu_state.fregs.write_float(1, 1.0)
        cpu.fpu_state.fregs.write_float(2, 2.0)
        _exec(cpu, _r_type(OP_OP_FP, rd=3, funct3=0, rs1=1, rs2=2, funct7=0x00))
        assert cpu.pc == start_pc + 4

    def test_flw_advances_pc(self) -> None:
        cpu = _make_cpu()
        start_pc = cpu.pc
        cpu.registers.write(1, BASE + 0x100)
        cpu.memory.write32(BASE + 0x100, 0)
        _exec(cpu, _i_type(OP_LOAD_FP, rd=2, funct3=2, rs1=1, imm=0))
        assert cpu.pc == start_pc + 4

    def test_fsw_advances_pc(self) -> None:
        cpu = _make_cpu()
        start_pc = cpu.pc
        cpu.registers.write(1, BASE + 0x100)
        _exec(cpu, _s_type(OP_STORE_FP, funct3=2, rs1=1, rs2=2, imm=0))
        assert cpu.pc == start_pc + 4

    def test_fmadd_advances_pc(self) -> None:
        cpu = _make_cpu()
        start_pc = cpu.pc
        _exec(cpu, _r4_type(OP_FMADD, rd=4, funct3=0, rs1=1, rs2=2, rs3=3))
        assert cpu.pc == start_pc + 4
