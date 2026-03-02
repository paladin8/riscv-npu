"""Tests for FP NPU instructions: all 10 FP NPU ops via CPU step."""

import math
import struct

from riscv_npu.cpu.cpu import CPU
from riscv_npu.memory.bus import MemoryBus
from riscv_npu.memory.ram import RAM

BASE = 0x80000000
RAM_SIZE = 1024 * 1024
OP_FP_NPU = 0x2B


def _make_cpu() -> CPU:
    """Create a fresh CPU with RAM."""
    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    bus.register(BASE, RAM_SIZE, ram)
    cpu = CPU(bus)
    cpu.pc = BASE
    return cpu


def _fp_npu_r(funct7: int, rs2: int, rs1: int, funct3: int, rd: int) -> int:
    """Encode an R-type FP NPU instruction (opcode 0x2B)."""
    return ((funct7 << 25) | (rs2 << 20) | (rs1 << 15)
            | (funct3 << 12) | (rd << 7) | OP_FP_NPU)


def _exec(cpu: CPU, word: int) -> None:
    """Write instruction word at PC and step the CPU."""
    cpu.memory.write32(cpu.pc, word)
    cpu.step()


def _f32_bits(val: float) -> int:
    """Convert a Python float to IEEE 754 single-precision bits."""
    return struct.unpack('<I', struct.pack('<f', val))[0]


def _write_f32_array(cpu: CPU, addr: int, values: list[float]) -> None:
    """Write a list of float32 values to memory."""
    for i, v in enumerate(values):
        cpu.memory.write32(addr + i * 4, _f32_bits(v))


def _read_f32_array(cpu: CPU, addr: int, count: int) -> list[float]:
    """Read a list of float32 values from memory."""
    result = []
    for i in range(count):
        bits = cpu.memory.read32(addr + i * 4)
        val = struct.unpack('<f', struct.pack('<I', bits))[0]
        result.append(val)
    return result


# ==================== FMACC tests ====================


class TestFMACC:
    """NPU.FMACC: facc += f[rs1] * f[rs2]."""

    def test_single_multiply(self) -> None:
        """FMACC: 3.0 * 4.0 = 12.0 in facc."""
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 3.0)
        cpu.fpu_state.fregs.write_float(2, 4.0)
        # FMACC: funct7=0, rs2=f2, rs1=f1, funct3=0, rd=f0
        _exec(cpu, _fp_npu_r(0, 2, 1, 0, 0))
        assert cpu.npu_state.facc == 12.0

    def test_accumulate(self) -> None:
        """FMACC accumulates: 2*3 + 4*5 = 26."""
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 2.0)
        cpu.fpu_state.fregs.write_float(2, 3.0)
        _exec(cpu, _fp_npu_r(0, 2, 1, 0, 0))
        cpu.fpu_state.fregs.write_float(1, 4.0)
        cpu.fpu_state.fregs.write_float(2, 5.0)
        _exec(cpu, _fp_npu_r(0, 2, 1, 0, 0))
        assert cpu.npu_state.facc == 26.0

    def test_negative_operands(self) -> None:
        """FMACC with negative: -2.5 * 4.0 = -10.0."""
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, -2.5)
        cpu.fpu_state.fregs.write_float(2, 4.0)
        _exec(cpu, _fp_npu_r(0, 2, 1, 0, 0))
        assert cpu.npu_state.facc == -10.0


# ==================== FVMAC tests ====================


class TestFVMAC:
    """NPU.FVMAC: facc += dot(mem_f32[rs1], mem_f32[rs2])."""

    def test_dot_product(self) -> None:
        """FVMAC: [1,2,3] . [4,5,6] = 32."""
        cpu = _make_cpu()
        data_a = BASE + 0x1000
        data_b = BASE + 0x2000
        _write_f32_array(cpu, data_a, [1.0, 2.0, 3.0])
        _write_f32_array(cpu, data_b, [4.0, 5.0, 6.0])
        # rd=x10 (count=3), rs1=x11 (addr_a), rs2=x12 (addr_b)
        cpu.registers.write(10, 3)
        cpu.registers.write(11, data_a)
        cpu.registers.write(12, data_b)
        _exec(cpu, _fp_npu_r(1, 12, 11, 0, 10))
        assert abs(cpu.npu_state.facc - 32.0) < 1e-6

    def test_empty_dot_product(self) -> None:
        """FVMAC with n=0 adds nothing."""
        cpu = _make_cpu()
        cpu.registers.write(10, 0)
        cpu.registers.write(11, BASE + 0x1000)
        cpu.registers.write(12, BASE + 0x2000)
        _exec(cpu, _fp_npu_r(1, 12, 11, 0, 10))
        assert cpu.npu_state.facc == 0.0

    def test_does_not_reset(self) -> None:
        """FVMAC accumulates without reset."""
        cpu = _make_cpu()
        cpu.npu_state.facc = 100.0
        data_a = BASE + 0x1000
        data_b = BASE + 0x2000
        _write_f32_array(cpu, data_a, [1.0])
        _write_f32_array(cpu, data_b, [2.0])
        cpu.registers.write(10, 1)
        cpu.registers.write(11, data_a)
        cpu.registers.write(12, data_b)
        _exec(cpu, _fp_npu_r(1, 12, 11, 0, 10))
        assert abs(cpu.npu_state.facc - 102.0) < 1e-6


# ==================== FRELU tests ====================


class TestFRELU:
    """NPU.FRELU: f[rd] = max(f[rs1], +0.0)."""

    def test_positive_passthrough(self) -> None:
        """Positive values pass through unchanged."""
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 5.5)
        # FRELU: funct7=0, rs2=0, rs1=f1, funct3=1, rd=f2
        _exec(cpu, _fp_npu_r(0, 0, 1, 1, 2))
        assert cpu.fpu_state.fregs.read_float(2) == 5.5

    def test_negative_to_zero(self) -> None:
        """Negative values become 0."""
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, -3.7)
        _exec(cpu, _fp_npu_r(0, 0, 1, 1, 2))
        assert cpu.fpu_state.fregs.read_float(2) == 0.0

    def test_zero_passthrough(self) -> None:
        """Zero passes through."""
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 0.0)
        _exec(cpu, _fp_npu_r(0, 0, 1, 1, 2))
        assert cpu.fpu_state.fregs.read_float(2) == 0.0

    def test_negative_zero_to_positive_zero(self) -> None:
        """-0.0 becomes +0.0."""
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_bits(1, 0x80000000)  # -0.0
        _exec(cpu, _fp_npu_r(0, 0, 1, 1, 2))
        result = cpu.fpu_state.fregs.read_float(2)
        assert result == 0.0
        assert math.copysign(1.0, result) > 0  # positive zero

    def test_nan_propagates(self) -> None:
        """NaN input produces NaN output."""
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_bits(1, 0x7FC00000)  # canonical NaN
        _exec(cpu, _fp_npu_r(0, 0, 1, 1, 2))
        assert math.isnan(cpu.fpu_state.fregs.read_float(2))


# ==================== FGELU tests ====================


class TestFGELU:
    """NPU.FGELU: f[rd] = gelu(f[rs1])."""

    def test_zero(self) -> None:
        """GELU(0) = 0."""
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 0.0)
        # FGELU: funct7=0, rs2=0, rs1=f1, funct3=4, rd=f2
        _exec(cpu, _fp_npu_r(0, 0, 1, 4, 2))
        assert cpu.fpu_state.fregs.read_float(2) == 0.0

    def test_positive(self) -> None:
        """GELU(2.0) ≈ 1.9545."""
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, 2.0)
        _exec(cpu, _fp_npu_r(0, 0, 1, 4, 2))
        result = cpu.fpu_state.fregs.read_float(2)
        assert abs(result - 1.9545) < 0.01

    def test_negative(self) -> None:
        """GELU(-2.0) ≈ -0.0455."""
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, -2.0)
        _exec(cpu, _fp_npu_r(0, 0, 1, 4, 2))
        result = cpu.fpu_state.fregs.read_float(2)
        assert abs(result - (-0.0455)) < 0.01

    def test_nan_propagates(self) -> None:
        """NaN input produces NaN output."""
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_bits(1, 0x7FC00000)
        _exec(cpu, _fp_npu_r(0, 0, 1, 4, 2))
        assert math.isnan(cpu.fpu_state.fregs.read_float(2))

    def test_negative_inf(self) -> None:
        """GELU(-inf) = 0."""
        cpu = _make_cpu()
        cpu.fpu_state.fregs.write_float(1, float('-inf'))
        _exec(cpu, _fp_npu_r(0, 0, 1, 4, 2))
        assert cpu.fpu_state.fregs.read_float(2) == 0.0


# ==================== FRSTACC tests ====================


class TestFRSTACC:
    """NPU.FRSTACC: f[rd] = (float32)facc; facc = 0."""

    def test_reads_and_resets(self) -> None:
        """FRSTACC reads facc into f[rd] and zeroes facc."""
        cpu = _make_cpu()
        cpu.npu_state.facc = 42.5
        # FRSTACC: funct7=0, rs2=0, rs1=0, funct3=5, rd=f3
        _exec(cpu, _fp_npu_r(0, 0, 0, 5, 3))
        assert cpu.fpu_state.fregs.read_float(3) == 42.5
        assert cpu.npu_state.facc == 0.0

    def test_zero_accumulator(self) -> None:
        """FRSTACC with facc=0 writes 0.0."""
        cpu = _make_cpu()
        _exec(cpu, _fp_npu_r(0, 0, 0, 5, 3))
        assert cpu.fpu_state.fregs.read_float(3) == 0.0

    def test_rounds_to_f32(self) -> None:
        """FRSTACC rounds double to float32."""
        cpu = _make_cpu()
        # Use a value that differs between f32 and f64
        cpu.npu_state.facc = 1.0000001192092896  # slightly more than 1.0 in f32
        _exec(cpu, _fp_npu_r(0, 0, 0, 5, 3))
        # write_float rounds to f32
        result_bits = cpu.fpu_state.fregs.read_bits(3)
        # Should be 1.0 or very close in f32 representation
        assert result_bits in (0x3F800000, 0x3F800001)


# ==================== FVEXP tests ====================


class TestFVEXP:
    """NPU.FVEXP: dst[i] = exp(src[i])."""

    def test_basic(self) -> None:
        """FVEXP: exp(0)=1, exp(1)≈2.718."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [0.0, 1.0])
        cpu.registers.write(10, 2)   # count
        cpu.registers.write(11, src)  # source
        cpu.registers.write(12, dst)  # destination
        # FVEXP: funct7=2, rs2=x12, rs1=x11, funct3=0, rd=x10
        _exec(cpu, _fp_npu_r(2, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 2)
        assert abs(results[0] - 1.0) < 1e-5
        assert abs(results[1] - math.e) < 0.01

    def test_negative_input(self) -> None:
        """FVEXP: exp(-1) ≈ 0.368."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [-1.0])
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(2, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 1)
        assert abs(results[0] - math.exp(-1.0)) < 0.01

    def test_in_place(self) -> None:
        """FVEXP: source and destination can overlap."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        _write_f32_array(cpu, addr, [0.0])
        cpu.registers.write(10, 1)
        cpu.registers.write(11, addr)
        cpu.registers.write(12, addr)
        _exec(cpu, _fp_npu_r(2, 12, 11, 0, 10))
        results = _read_f32_array(cpu, addr, 1)
        assert abs(results[0] - 1.0) < 1e-5


# ==================== FVRSQRT tests ====================


class TestFVRSQRT:
    """NPU.FVRSQRT: f[rd] = 1/sqrt(mem_f32[rs1])."""

    def test_basic(self) -> None:
        """FVRSQRT: 1/sqrt(4.0) = 0.5."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        _write_f32_array(cpu, addr, [4.0])
        cpu.registers.write(11, addr)
        # FVRSQRT: funct7=3, rs2=0, rs1=x11, funct3=0, rd=f5
        _exec(cpu, _fp_npu_r(3, 0, 11, 0, 5))
        assert abs(cpu.fpu_state.fregs.read_float(5) - 0.5) < 1e-5

    def test_one(self) -> None:
        """FVRSQRT: 1/sqrt(1.0) = 1.0."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        _write_f32_array(cpu, addr, [1.0])
        cpu.registers.write(11, addr)
        _exec(cpu, _fp_npu_r(3, 0, 11, 0, 5))
        assert abs(cpu.fpu_state.fregs.read_float(5) - 1.0) < 1e-5

    def test_zero_gives_inf(self) -> None:
        """FVRSQRT: 1/sqrt(0) = +inf."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        _write_f32_array(cpu, addr, [0.0])
        cpu.registers.write(11, addr)
        _exec(cpu, _fp_npu_r(3, 0, 11, 0, 5))
        result = cpu.fpu_state.fregs.read_float(5)
        assert math.isinf(result) and result > 0

    def test_negative_gives_nan(self) -> None:
        """FVRSQRT: 1/sqrt(negative) = NaN."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        _write_f32_array(cpu, addr, [-4.0])
        cpu.registers.write(11, addr)
        _exec(cpu, _fp_npu_r(3, 0, 11, 0, 5))
        assert math.isnan(cpu.fpu_state.fregs.read_float(5))


# ==================== FVMUL tests ====================


class TestFVMUL:
    """NPU.FVMUL: dst[i] = src[i] * (float32)facc."""

    def test_scale_array(self) -> None:
        """FVMUL: [1, 2, 3] * 2.0 = [2, 4, 6]."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [1.0, 2.0, 3.0])
        cpu.npu_state.facc = 2.0
        cpu.registers.write(10, 3)   # count
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        # FVMUL: funct7=4, rs2=x12, rs1=x11, funct3=0, rd=x10
        _exec(cpu, _fp_npu_r(4, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 3)
        assert abs(results[0] - 2.0) < 1e-5
        assert abs(results[1] - 4.0) < 1e-5
        assert abs(results[2] - 6.0) < 1e-5

    def test_does_not_modify_facc(self) -> None:
        """FVMUL does not modify the accumulator."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [1.0])
        cpu.npu_state.facc = 3.0
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(4, 12, 11, 0, 10))
        assert cpu.npu_state.facc == 3.0

    def test_scale_by_zero(self) -> None:
        """FVMUL with facc=0 zeros the array."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [5.0, 10.0])
        cpu.npu_state.facc = 0.0
        cpu.registers.write(10, 2)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(4, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 2)
        assert results[0] == 0.0
        assert results[1] == 0.0


# ==================== FVREDUCE tests ====================


class TestFVREDUCE:
    """NPU.FVREDUCE: f[rd] = sum(mem_f32[rs1..+n])."""

    def test_sum_array(self) -> None:
        """FVREDUCE: sum([1, 2, 3]) = 6."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        _write_f32_array(cpu, addr, [1.0, 2.0, 3.0])
        cpu.registers.write(11, addr)  # source
        cpu.registers.write(12, 3)     # count
        # FVREDUCE: funct7=5, rs2=x12, rs1=x11, funct3=0, rd=f5
        _exec(cpu, _fp_npu_r(5, 12, 11, 0, 5))
        assert abs(cpu.fpu_state.fregs.read_float(5) - 6.0) < 1e-5

    def test_empty_sum(self) -> None:
        """FVREDUCE with n=0 returns 0."""
        cpu = _make_cpu()
        cpu.registers.write(11, BASE + 0x1000)
        cpu.registers.write(12, 0)
        _exec(cpu, _fp_npu_r(5, 12, 11, 0, 5))
        assert cpu.fpu_state.fregs.read_float(5) == 0.0

    def test_negative_values(self) -> None:
        """FVREDUCE with mixed signs."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        _write_f32_array(cpu, addr, [10.0, -3.0, -2.0])
        cpu.registers.write(11, addr)
        cpu.registers.write(12, 3)
        _exec(cpu, _fp_npu_r(5, 12, 11, 0, 5))
        assert abs(cpu.fpu_state.fregs.read_float(5) - 5.0) < 1e-5


# ==================== FVMAX tests ====================


class TestFVMAX:
    """NPU.FVMAX: f[rd] = max(mem_f32[rs1..+n])."""

    def test_max_array(self) -> None:
        """FVMAX: max([5, -3, 10]) = 10."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        _write_f32_array(cpu, addr, [5.0, -3.0, 10.0])
        cpu.registers.write(11, addr)
        cpu.registers.write(12, 3)
        # FVMAX: funct7=6, rs2=x12, rs1=x11, funct3=0, rd=f5
        _exec(cpu, _fp_npu_r(6, 12, 11, 0, 5))
        assert cpu.fpu_state.fregs.read_float(5) == 10.0

    def test_all_negative(self) -> None:
        """FVMAX: max([-5, -3, -10]) = -3."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        _write_f32_array(cpu, addr, [-5.0, -3.0, -10.0])
        cpu.registers.write(11, addr)
        cpu.registers.write(12, 3)
        _exec(cpu, _fp_npu_r(6, 12, 11, 0, 5))
        assert cpu.fpu_state.fregs.read_float(5) == -3.0

    def test_empty_returns_neg_inf(self) -> None:
        """FVMAX with n=0 returns -inf."""
        cpu = _make_cpu()
        cpu.registers.write(11, BASE + 0x1000)
        cpu.registers.write(12, 0)
        _exec(cpu, _fp_npu_r(6, 12, 11, 0, 5))
        result = cpu.fpu_state.fregs.read_float(5)
        assert math.isinf(result) and result < 0

    def test_nan_propagates(self) -> None:
        """FVMAX with NaN element returns NaN."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        _write_f32_array(cpu, addr, [1.0])
        # Write NaN as second element
        cpu.memory.write32(addr + 4, 0x7FC00000)
        cpu.registers.write(11, addr)
        cpu.registers.write(12, 2)
        _exec(cpu, _fp_npu_r(6, 12, 11, 0, 5))
        assert math.isnan(cpu.fpu_state.fregs.read_float(5))
