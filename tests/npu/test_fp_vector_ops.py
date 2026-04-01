"""Tests for Phase 11 FP NPU vector instructions (funct7 7-12).

FVADD, FVSUB, FVRELU, FVGELU, FVDIV, FVSUB_SCALAR.
"""

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


# ==================== FVADD tests (funct7=7) ====================


class TestFVADD:
    """NPU.FVADD: dst[i] = src1[i] + src2[i], result at rs2."""

    def test_basic_add(self) -> None:
        """FVADD: [1, 2, 3] + [4, 5, 6] = [5, 7, 9]."""
        cpu = _make_cpu()
        src1 = BASE + 0x1000
        src2 = BASE + 0x2000
        _write_f32_array(cpu, src1, [1.0, 2.0, 3.0])
        _write_f32_array(cpu, src2, [4.0, 5.0, 6.0])
        cpu.registers.write(10, 3)    # count
        cpu.registers.write(11, src1)  # rs1 = source 1
        cpu.registers.write(12, src2)  # rs2 = source 2 / destination
        _exec(cpu, _fp_npu_r(7, 12, 11, 0, 10))
        results = _read_f32_array(cpu, src2, 3)
        assert abs(results[0] - 5.0) < 1e-5
        assert abs(results[1] - 7.0) < 1e-5
        assert abs(results[2] - 9.0) < 1e-5

    def test_negative_values(self) -> None:
        """FVADD with negative operands."""
        cpu = _make_cpu()
        src1 = BASE + 0x1000
        src2 = BASE + 0x2000
        _write_f32_array(cpu, src1, [-1.0, 2.5])
        _write_f32_array(cpu, src2, [3.0, -4.5])
        cpu.registers.write(10, 2)
        cpu.registers.write(11, src1)
        cpu.registers.write(12, src2)
        _exec(cpu, _fp_npu_r(7, 12, 11, 0, 10))
        results = _read_f32_array(cpu, src2, 2)
        assert abs(results[0] - 2.0) < 1e-5
        assert abs(results[1] - (-2.0)) < 1e-5

    def test_in_place_overlap(self) -> None:
        """FVADD: src1 == src2 means a[i] + a[i] = 2*a[i]."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        _write_f32_array(cpu, addr, [3.0, 5.0])
        cpu.registers.write(10, 2)
        cpu.registers.write(11, addr)
        cpu.registers.write(12, addr)
        _exec(cpu, _fp_npu_r(7, 12, 11, 0, 10))
        results = _read_f32_array(cpu, addr, 2)
        assert abs(results[0] - 6.0) < 1e-5
        assert abs(results[1] - 10.0) < 1e-5

    def test_n_zero(self) -> None:
        """FVADD with n=0 is a no-op."""
        cpu = _make_cpu()
        addr = BASE + 0x2000
        _write_f32_array(cpu, addr, [42.0])
        cpu.registers.write(10, 0)
        cpu.registers.write(11, BASE + 0x1000)
        cpu.registers.write(12, addr)
        _exec(cpu, _fp_npu_r(7, 12, 11, 0, 10))
        results = _read_f32_array(cpu, addr, 1)
        assert results[0] == 42.0

    def test_single_element(self) -> None:
        """FVADD with n=1."""
        cpu = _make_cpu()
        src1 = BASE + 0x1000
        src2 = BASE + 0x2000
        _write_f32_array(cpu, src1, [10.0])
        _write_f32_array(cpu, src2, [20.0])
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src1)
        cpu.registers.write(12, src2)
        _exec(cpu, _fp_npu_r(7, 12, 11, 0, 10))
        results = _read_f32_array(cpu, src2, 1)
        assert abs(results[0] - 30.0) < 1e-5


# ==================== FVSUB tests (funct7=8) ====================


class TestFVSUB:
    """NPU.FVSUB: dst[i] = src1[i] - src2[i], result at rs2."""

    def test_basic_sub(self) -> None:
        """FVSUB: [10, 20] - [3, 7] = [7, 13]."""
        cpu = _make_cpu()
        src1 = BASE + 0x1000
        src2 = BASE + 0x2000
        _write_f32_array(cpu, src1, [10.0, 20.0])
        _write_f32_array(cpu, src2, [3.0, 7.0])
        cpu.registers.write(10, 2)
        cpu.registers.write(11, src1)
        cpu.registers.write(12, src2)
        _exec(cpu, _fp_npu_r(8, 12, 11, 0, 10))
        results = _read_f32_array(cpu, src2, 2)
        assert abs(results[0] - 7.0) < 1e-5
        assert abs(results[1] - 13.0) < 1e-5

    def test_result_sign(self) -> None:
        """FVSUB: 1.0 - 5.0 = -4.0."""
        cpu = _make_cpu()
        src1 = BASE + 0x1000
        src2 = BASE + 0x2000
        _write_f32_array(cpu, src1, [1.0])
        _write_f32_array(cpu, src2, [5.0])
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src1)
        cpu.registers.write(12, src2)
        _exec(cpu, _fp_npu_r(8, 12, 11, 0, 10))
        results = _read_f32_array(cpu, src2, 1)
        assert abs(results[0] - (-4.0)) < 1e-5

    def test_self_sub_is_zero(self) -> None:
        """FVSUB: a - a = 0 when src1 == src2."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        _write_f32_array(cpu, addr, [7.0, -3.0, 100.0])
        cpu.registers.write(10, 3)
        cpu.registers.write(11, addr)
        cpu.registers.write(12, addr)
        _exec(cpu, _fp_npu_r(8, 12, 11, 0, 10))
        results = _read_f32_array(cpu, addr, 3)
        for r in results:
            assert r == 0.0


# ==================== FVRELU tests (funct7=9) ====================


class TestFVRELU:
    """NPU.FVRELU: dst[i] = max(src[i], 0.0)."""

    def test_positive_passthrough(self) -> None:
        """FVRELU: positive values unchanged."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [1.0, 5.5, 100.0])
        cpu.registers.write(10, 3)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(9, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 3)
        assert abs(results[0] - 1.0) < 1e-5
        assert abs(results[1] - 5.5) < 1e-5
        assert abs(results[2] - 100.0) < 1e-5

    def test_negative_to_zero(self) -> None:
        """FVRELU: negative values become 0.0."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [-1.0, -100.0, -0.001])
        cpu.registers.write(10, 3)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(9, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 3)
        for r in results:
            assert r == 0.0

    def test_zero_stays_zero(self) -> None:
        """FVRELU: +0.0 stays +0.0."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [0.0])
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(9, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 1)
        assert results[0] == 0.0
        # Verify it's +0.0 (sign bit = 0)
        bits = cpu.memory.read32(dst)
        assert bits == 0x00000000

    def test_negative_zero_to_positive_zero(self) -> None:
        """FVRELU: -0.0 becomes +0.0."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        cpu.memory.write32(src, 0x80000000)  # -0.0
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(9, 12, 11, 0, 10))
        bits = cpu.memory.read32(dst)
        assert bits == 0x00000000  # +0.0

    def test_nan_propagation(self) -> None:
        """FVRELU: NaN input produces NaN output."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        cpu.memory.write32(src, 0x7FC00000)  # quiet NaN
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(9, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 1)
        assert math.isnan(results[0])


# ==================== FVGELU tests (funct7=10) ====================


class TestFVGELU:
    """NPU.FVGELU: dst[i] = gelu(src[i])."""

    def test_zero(self) -> None:
        """FVGELU: gelu(0) = 0."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [0.0])
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(10, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 1)
        assert abs(results[0]) < 1e-5

    def test_positive(self) -> None:
        """FVGELU: gelu(1.0) ≈ 0.8413."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [1.0])
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(10, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 1)
        expected = 0.5 * 1.0 * (1 + math.erf(1.0 / math.sqrt(2)))
        assert abs(results[0] - expected) < 1e-4

    def test_large_negative(self) -> None:
        """FVGELU: gelu(-5.0) ≈ 0."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [-5.0])
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(10, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 1)
        assert abs(results[0]) < 0.001

    def test_matches_scalar_fgelu(self) -> None:
        """FVGELU results should match the scalar NPU.FGELU formula."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        inputs = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
        _write_f32_array(cpu, src, inputs)
        cpu.registers.write(10, len(inputs))
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(10, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, len(inputs))
        for x, r in zip(inputs, results):
            expected = 0.5 * x * (1 + math.erf(x / math.sqrt(2)))
            assert abs(r - expected) < 1e-4, f"gelu({x}): got {r}, expected {expected}"

    def test_nan(self) -> None:
        """FVGELU: NaN input produces NaN output."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        cpu.memory.write32(src, 0x7FC00000)  # quiet NaN
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(10, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 1)
        assert math.isnan(results[0])

    def test_positive_inf(self) -> None:
        """FVGELU: +inf → +inf."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        cpu.memory.write32(src, 0x7F800000)  # +inf
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(10, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 1)
        assert math.isinf(results[0]) and results[0] > 0

    def test_negative_inf(self) -> None:
        """FVGELU: -inf → 0.0."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        cpu.memory.write32(src, 0xFF800000)  # -inf
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(10, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 1)
        assert results[0] == 0.0


# ==================== FVDIV tests (funct7=11) ====================


class TestFVDIV:
    """NPU.FVDIV: dst[i] = src[i] / (float32)facc."""

    def test_basic_division(self) -> None:
        """FVDIV: [10, 20] / 5.0 = [2, 4]."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [10.0, 20.0])
        cpu.npu_state.facc = 5.0
        cpu.registers.write(10, 2)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(11, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 2)
        assert abs(results[0] - 2.0) < 1e-5
        assert abs(results[1] - 4.0) < 1e-5

    def test_divide_by_zero_inf(self) -> None:
        """FVDIV: division by zero produces ±inf (IEEE 754)."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [1.0, -1.0])
        cpu.npu_state.facc = 0.0
        cpu.registers.write(10, 2)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(11, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 2)
        assert math.isinf(results[0]) and results[0] > 0
        assert math.isinf(results[1]) and results[1] < 0

    def test_zero_divided_by_zero_nan(self) -> None:
        """FVDIV: 0.0 / 0.0 produces NaN (IEEE 754)."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [0.0])
        cpu.npu_state.facc = 0.0
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(11, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 1)
        assert math.isnan(results[0])

    def test_negative_zero_divisor_sign(self) -> None:
        """FVDIV: 1.0 / -0.0 = -inf (IEEE 754 sign rule)."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [1.0])
        cpu.npu_state.facc = -0.0
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(11, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 1)
        assert math.isinf(results[0]) and results[0] < 0

    def test_accumulator_unchanged(self) -> None:
        """FVDIV does not modify the accumulator."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [6.0])
        cpu.npu_state.facc = 3.0
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(11, 12, 11, 0, 10))
        assert cpu.npu_state.facc == 3.0

    def test_nan_input(self) -> None:
        """FVDIV: NaN input produces NaN output."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        cpu.memory.write32(src, 0x7FC00000)  # quiet NaN
        cpu.npu_state.facc = 2.0
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(11, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 1)
        assert math.isnan(results[0])

    def test_divide_by_large_facc(self) -> None:
        """FVDIV: dividing by a large accumulator produces small results."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [1.0])
        cpu.npu_state.facc = 1e10
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(11, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 1)
        assert abs(results[0] - 1e-10) < 1e-15


# ==================== FVSUB_SCALAR tests (funct7=12) ====================


class TestFVSUB_SCALAR:
    """NPU.FVSUB_SCALAR: dst[i] = src[i] - (float32)facc."""

    def test_basic_subtraction(self) -> None:
        """FVSUB_SCALAR: [10, 20] - 3.0 = [7, 17]."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [10.0, 20.0])
        cpu.npu_state.facc = 3.0
        cpu.registers.write(10, 2)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(12, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 2)
        assert abs(results[0] - 7.0) < 1e-5
        assert abs(results[1] - 17.0) < 1e-5

    def test_subtract_zero(self) -> None:
        """FVSUB_SCALAR: a - 0.0 = a."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [5.0, -3.0])
        cpu.npu_state.facc = 0.0
        cpu.registers.write(10, 2)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(12, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 2)
        assert abs(results[0] - 5.0) < 1e-5
        assert abs(results[1] - (-3.0)) < 1e-5

    def test_accumulator_unchanged(self) -> None:
        """FVSUB_SCALAR does not modify the accumulator."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        _write_f32_array(cpu, src, [10.0])
        cpu.npu_state.facc = 7.0
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(12, 12, 11, 0, 10))
        assert cpu.npu_state.facc == 7.0

    def test_nan_input(self) -> None:
        """FVSUB_SCALAR: NaN input produces NaN output."""
        cpu = _make_cpu()
        src = BASE + 0x1000
        dst = BASE + 0x2000
        cpu.memory.write32(src, 0x7FC00000)  # quiet NaN
        cpu.npu_state.facc = 1.0
        cpu.registers.write(10, 1)
        cpu.registers.write(11, src)
        cpu.registers.write(12, dst)
        _exec(cpu, _fp_npu_r(12, 12, 11, 0, 10))
        results = _read_f32_array(cpu, dst, 1)
        assert math.isnan(results[0])

    def test_in_place(self) -> None:
        """FVSUB_SCALAR: source and destination can overlap."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        _write_f32_array(cpu, addr, [10.0, 20.0])
        cpu.npu_state.facc = 5.0
        cpu.registers.write(10, 2)
        cpu.registers.write(11, addr)
        cpu.registers.write(12, addr)
        _exec(cpu, _fp_npu_r(12, 12, 11, 0, 10))
        results = _read_f32_array(cpu, addr, 2)
        assert abs(results[0] - 5.0) < 1e-5
        assert abs(results[1] - 15.0) < 1e-5
