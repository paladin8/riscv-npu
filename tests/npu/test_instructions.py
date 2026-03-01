"""Tests for NPU instructions: all 8 NPU ops via CPU step."""

import math

from riscv_npu.cpu.cpu import CPU
from riscv_npu.cpu.decode import to_signed
from riscv_npu.memory.bus import MemoryBus
from riscv_npu.memory.ram import RAM
from riscv_npu.npu.engine import GELU_TABLE, acc_get64, acc_set64

BASE = 0x80000000
RAM_SIZE = 1024 * 1024
OP_NPU = 0x0B


def _make_cpu() -> CPU:
    """Create a fresh CPU with RAM."""
    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    bus.register(BASE, RAM_SIZE, ram)
    cpu = CPU(bus)
    cpu.pc = BASE
    return cpu


def _npu_r(funct7: int, rs2: int, rs1: int, funct3: int, rd: int) -> int:
    """Encode an R-type NPU instruction (opcode 0x0B)."""
    return ((funct7 << 25) | (rs2 << 20) | (rs1 << 15)
            | (funct3 << 12) | (rd << 7) | OP_NPU)


def _npu_i(imm12: int, rs1: int, funct3: int, rd: int) -> int:
    """Encode an I-type NPU instruction (opcode 0x0B) for LDVEC."""
    return (((imm12 & 0xFFF) << 20) | (rs1 << 15)
            | (funct3 << 12) | (rd << 7) | OP_NPU)


def _npu_s(imm12: int, rs2: int, rs1: int, funct3: int) -> int:
    """Encode an S-type NPU instruction (opcode 0x0B) for STVEC."""
    imm_11_5 = (imm12 >> 5) & 0x7F
    imm_4_0 = imm12 & 0x1F
    return ((imm_11_5 << 25) | (rs2 << 20) | (rs1 << 15)
            | (funct3 << 12) | (imm_4_0 << 7) | OP_NPU)


def _exec(cpu: CPU, word: int) -> None:
    """Write instruction word at PC and step the CPU."""
    cpu.memory.write32(cpu.pc, word)
    cpu.step()


# ==================== MACC tests ====================

class TestMACC:
    """NPU.MACC: {acc_hi,acc_lo} += signed(rs1) * signed(rs2)."""

    def test_single_multiply(self) -> None:
        """Single MACC: 3 * 7 = 21."""
        cpu = _make_cpu()
        cpu.registers.write(1, 3)
        cpu.registers.write(2, 7)
        # MACC: funct7=0, rs2=x2, rs1=x1, funct3=0, rd=x0
        _exec(cpu, _npu_r(0, 2, 1, 0, 0))
        assert cpu.npu_state.acc_lo == 21
        assert cpu.npu_state.acc_hi == 0

    def test_negative_operands(self) -> None:
        """MACC with negative operands: (-3) * 7 = -21."""
        cpu = _make_cpu()
        cpu.registers.write(1, (-3) & 0xFFFFFFFF)
        cpu.registers.write(2, 7)
        _exec(cpu, _npu_r(0, 2, 1, 0, 0))
        assert acc_get64(cpu.npu_state) == -21

    def test_both_negative(self) -> None:
        """MACC with both negative: (-3) * (-7) = 21."""
        cpu = _make_cpu()
        cpu.registers.write(1, (-3) & 0xFFFFFFFF)
        cpu.registers.write(2, (-7) & 0xFFFFFFFF)
        _exec(cpu, _npu_r(0, 2, 1, 0, 0))
        assert acc_get64(cpu.npu_state) == 21

    def test_chain_10(self) -> None:
        """Chain of 10 MACCs: sum of 5*10 = 500."""
        cpu = _make_cpu()
        cpu.registers.write(1, 5)
        cpu.registers.write(2, 10)
        for i in range(10):
            cpu.pc = BASE + i * 4
            _exec(cpu, _npu_r(0, 2, 1, 0, 0))
        assert acc_get64(cpu.npu_state) == 500

    def test_accumulator_overflow_64bit(self) -> None:
        """MACC chain that crosses 32-bit boundary in accumulator."""
        cpu = _make_cpu()
        cpu.registers.write(1, 0x10000)  # 65536
        cpu.registers.write(2, 0x10000)  # 65536
        # 65536 * 65536 = 4294967296 = 2^32
        _exec(cpu, _npu_r(0, 2, 1, 0, 0))
        assert cpu.npu_state.acc_lo == 0
        assert cpu.npu_state.acc_hi == 1
        assert acc_get64(cpu.npu_state) == 0x100000000

    def test_large_signed_multiply(self) -> None:
        """MACC with large values: 0x7FFFFFFF * 2."""
        cpu = _make_cpu()
        cpu.registers.write(1, 0x7FFFFFFF)
        cpu.registers.write(2, 2)
        _exec(cpu, _npu_r(0, 2, 1, 0, 0))
        assert acc_get64(cpu.npu_state) == 0x7FFFFFFF * 2


# ==================== VMAC tests ====================

class TestVMAC:
    """NPU.VMAC: acc += dot(mem_int8[rs1..+n], mem_int8[rs2..+n])."""

    def _write_bytes(self, cpu: CPU, addr: int, values: list[int]) -> None:
        """Write a list of int8 values to memory as unsigned bytes."""
        for i, v in enumerate(values):
            cpu.memory.write8(addr + i, v & 0xFF)

    def test_basic_dot_product(self) -> None:
        """VMAC: [1,2,3,4] . [5,6,7,8] = 5+12+21+32 = 70."""
        cpu = _make_cpu()
        data_base = BASE + 0x1000
        a_vals = [1, 2, 3, 4]
        b_vals = [5, 6, 7, 8]
        self._write_bytes(cpu, data_base, a_vals)
        self._write_bytes(cpu, data_base + 0x100, b_vals)
        cpu.registers.write(10, data_base)       # rs1 = addr_a
        cpu.registers.write(11, data_base + 0x100)  # rs2 = addr_b
        cpu.registers.write(12, 4)               # rd = count
        # VMAC: funct7=1, rs2=x11, rs1=x10, funct3=0, rd=x12
        _exec(cpu, _npu_r(1, 11, 10, 0, 12))
        assert acc_get64(cpu.npu_state) == 70

    def test_accumulates_without_reset(self) -> None:
        """VMAC adds to existing accumulator value."""
        cpu = _make_cpu()
        data_base = BASE + 0x1000
        self._write_bytes(cpu, data_base, [10, 20])
        self._write_bytes(cpu, data_base + 0x100, [3, 4])
        # Pre-load accumulator with 100
        acc_set64(cpu.npu_state, 100)
        cpu.registers.write(10, data_base)
        cpu.registers.write(11, data_base + 0x100)
        cpu.registers.write(12, 2)
        _exec(cpu, _npu_r(1, 11, 10, 0, 12))
        # 100 + 10*3 + 20*4 = 100 + 30 + 80 = 210
        assert acc_get64(cpu.npu_state) == 210

    def test_negative_values(self) -> None:
        """VMAC with signed int8 values: (-1)*2 + (-3)*4 = -2 + -12 = -14."""
        cpu = _make_cpu()
        data_base = BASE + 0x1000
        self._write_bytes(cpu, data_base, [-1, -3])       # 0xFF, 0xFD
        self._write_bytes(cpu, data_base + 0x100, [2, 4])
        cpu.registers.write(10, data_base)
        cpu.registers.write(11, data_base + 0x100)
        cpu.registers.write(12, 2)
        _exec(cpu, _npu_r(1, 11, 10, 0, 12))
        assert acc_get64(cpu.npu_state) == -14

    def test_large_dot_product(self) -> None:
        """VMAC with 784 elements (MNIST-scale), verify against Python reference."""
        cpu = _make_cpu()
        n = 784
        data_base = BASE + 0x1000
        addr_a = data_base
        addr_b = data_base + 0x1000
        # Generate test data: a[i] = (i*7) % 256 as uint8 -> int8
        # b[i] = ((i*13 + 50) % 256) as uint8 -> int8
        expected = 0
        for i in range(n):
            a_byte = (i * 7) % 256
            b_byte = ((i * 13 + 50) % 256)
            cpu.memory.write8(addr_a + i, a_byte)
            cpu.memory.write8(addr_b + i, b_byte)
            a_signed = a_byte - 256 if a_byte >= 128 else a_byte
            b_signed = b_byte - 256 if b_byte >= 128 else b_byte
            expected += a_signed * b_signed

        cpu.registers.write(10, addr_a)
        cpu.registers.write(11, addr_b)
        cpu.registers.write(12, n)
        _exec(cpu, _npu_r(1, 11, 10, 0, 12))
        assert acc_get64(cpu.npu_state) == expected

    def test_zero_length(self) -> None:
        """VMAC with n=0 does nothing to accumulator."""
        cpu = _make_cpu()
        acc_set64(cpu.npu_state, 42)
        cpu.registers.write(10, BASE + 0x1000)
        cpu.registers.write(11, BASE + 0x2000)
        cpu.registers.write(12, 0)
        _exec(cpu, _npu_r(1, 11, 10, 0, 12))
        assert acc_get64(cpu.npu_state) == 42

    def test_single_element(self) -> None:
        """VMAC with n=1, equivalent to scalar MACC on int8 values."""
        cpu = _make_cpu()
        data_base = BASE + 0x1000
        self._write_bytes(cpu, data_base, [7])
        self._write_bytes(cpu, data_base + 0x100, [9])
        cpu.registers.write(10, data_base)
        cpu.registers.write(11, data_base + 0x100)
        cpu.registers.write(12, 1)
        _exec(cpu, _npu_r(1, 11, 10, 0, 12))
        assert acc_get64(cpu.npu_state) == 63


# ==================== RELU tests ====================

class TestRELU:
    """NPU.RELU: rd = max(signed(rs1), 0)."""

    def test_positive_passthrough(self) -> None:
        """Positive value passes through unchanged."""
        cpu = _make_cpu()
        cpu.registers.write(1, 42)
        # RELU: funct7=0, rs2=0, rs1=x1, funct3=1, rd=x3
        _exec(cpu, _npu_r(0, 0, 1, 1, 3))
        assert cpu.registers.read(3) == 42

    def test_negative_clamped(self) -> None:
        """Negative value clamped to 0."""
        cpu = _make_cpu()
        cpu.registers.write(1, (-10) & 0xFFFFFFFF)
        _exec(cpu, _npu_r(0, 0, 1, 1, 3))
        assert cpu.registers.read(3) == 0

    def test_zero_stays_zero(self) -> None:
        """Zero stays zero."""
        cpu = _make_cpu()
        cpu.registers.write(1, 0)
        _exec(cpu, _npu_r(0, 0, 1, 1, 3))
        assert cpu.registers.read(3) == 0

    def test_large_positive(self) -> None:
        """Large positive value (but < 0x80000000) passes through."""
        cpu = _make_cpu()
        cpu.registers.write(1, 0x7FFFFFFF)
        _exec(cpu, _npu_r(0, 0, 1, 1, 3))
        assert cpu.registers.read(3) == 0x7FFFFFFF

    def test_min_negative(self) -> None:
        """Most negative value (-2^31) clamped to 0."""
        cpu = _make_cpu()
        cpu.registers.write(1, 0x80000000)
        _exec(cpu, _npu_r(0, 0, 1, 1, 3))
        assert cpu.registers.read(3) == 0


# ==================== QMUL tests ====================

class TestQMUL:
    """NPU.QMUL: rd = (signed(rs1) * signed(rs2)) >> 8."""

    def test_basic(self) -> None:
        """Basic QMUL: (10 * 20) >> 8 = 200 >> 8 = 0 (integer)."""
        cpu = _make_cpu()
        cpu.registers.write(1, 10)
        cpu.registers.write(2, 20)
        # QMUL: funct7=0, rs2=x2, rs1=x1, funct3=2, rd=x3
        _exec(cpu, _npu_r(0, 2, 1, 2, 3))
        assert cpu.registers.read(3) == (10 * 20) >> 8

    def test_larger_values(self) -> None:
        """QMUL with larger values: (256 * 256) >> 8 = 256."""
        cpu = _make_cpu()
        cpu.registers.write(1, 256)
        cpu.registers.write(2, 256)
        _exec(cpu, _npu_r(0, 2, 1, 2, 3))
        assert cpu.registers.read(3) == (256 * 256) >> 8

    def test_max_positive(self) -> None:
        """QMUL: (127 * 127) >> 8 = 16129 >> 8 = 63."""
        cpu = _make_cpu()
        cpu.registers.write(1, 127)
        cpu.registers.write(2, 127)
        _exec(cpu, _npu_r(0, 2, 1, 2, 3))
        assert cpu.registers.read(3) == (127 * 127) >> 8

    def test_negative_operands(self) -> None:
        """QMUL with one negative: (-128 * 64) >> 8 = -8192 >> 8 = -32."""
        cpu = _make_cpu()
        cpu.registers.write(1, (-128) & 0xFFFFFFFF)
        cpu.registers.write(2, 64)
        _exec(cpu, _npu_r(0, 2, 1, 2, 3))
        result = to_signed(cpu.registers.read(3))
        assert result == ((-128) * 64) >> 8

    def test_both_negative(self) -> None:
        """QMUL with both negative: (-10 * -20) >> 8 = 200 >> 8 = 0."""
        cpu = _make_cpu()
        cpu.registers.write(1, (-10) & 0xFFFFFFFF)
        cpu.registers.write(2, (-20) & 0xFFFFFFFF)
        _exec(cpu, _npu_r(0, 2, 1, 2, 3))
        assert cpu.registers.read(3) == ((-10) * (-20)) >> 8

    def test_zero(self) -> None:
        """QMUL with zero: (0 * anything) >> 8 = 0."""
        cpu = _make_cpu()
        cpu.registers.write(1, 0)
        cpu.registers.write(2, 999)
        _exec(cpu, _npu_r(0, 2, 1, 2, 3))
        assert cpu.registers.read(3) == 0


# ==================== CLAMP tests ====================

class TestCLAMP:
    """NPU.CLAMP: rd = clamp(signed(rs1), -128, 127)."""

    def test_in_range(self) -> None:
        """Value within [-128, 127] passes through unchanged."""
        cpu = _make_cpu()
        cpu.registers.write(1, 42)
        # CLAMP: funct7=0, rs2=0, rs1=x1, funct3=3, rd=x3
        _exec(cpu, _npu_r(0, 0, 1, 3, 3))
        assert cpu.registers.read(3) == 42

    def test_above_max(self) -> None:
        """Value above 127 clamped to 127."""
        cpu = _make_cpu()
        cpu.registers.write(1, 200)
        _exec(cpu, _npu_r(0, 0, 1, 3, 3))
        assert cpu.registers.read(3) == 127

    def test_below_min(self) -> None:
        """Value below -128 clamped to -128."""
        cpu = _make_cpu()
        cpu.registers.write(1, (-200) & 0xFFFFFFFF)
        _exec(cpu, _npu_r(0, 0, 1, 3, 3))
        # -128 as uint32
        assert cpu.registers.read(3) == (-128) & 0xFFFFFFFF

    def test_exact_max_boundary(self) -> None:
        """Value exactly 127 passes through."""
        cpu = _make_cpu()
        cpu.registers.write(1, 127)
        _exec(cpu, _npu_r(0, 0, 1, 3, 3))
        assert cpu.registers.read(3) == 127

    def test_exact_min_boundary(self) -> None:
        """Value exactly -128 passes through."""
        cpu = _make_cpu()
        cpu.registers.write(1, (-128) & 0xFFFFFFFF)
        _exec(cpu, _npu_r(0, 0, 1, 3, 3))
        assert cpu.registers.read(3) == (-128) & 0xFFFFFFFF

    def test_zero(self) -> None:
        """Zero passes through."""
        cpu = _make_cpu()
        cpu.registers.write(1, 0)
        _exec(cpu, _npu_r(0, 0, 1, 3, 3))
        assert cpu.registers.read(3) == 0

    def test_large_positive(self) -> None:
        """Large positive value clamped to 127."""
        cpu = _make_cpu()
        cpu.registers.write(1, 10000)
        _exec(cpu, _npu_r(0, 0, 1, 3, 3))
        assert cpu.registers.read(3) == 127

    def test_large_negative(self) -> None:
        """Large negative value clamped to -128."""
        cpu = _make_cpu()
        cpu.registers.write(1, (-10000) & 0xFFFFFFFF)
        _exec(cpu, _npu_r(0, 0, 1, 3, 3))
        assert cpu.registers.read(3) == (-128) & 0xFFFFFFFF


# ==================== GELU tests ====================

class TestGELU:
    """NPU.GELU: rd = gelu_approx(rs1) via lookup table."""

    def test_zero(self) -> None:
        """GELU(0) should be 0."""
        cpu = _make_cpu()
        cpu.registers.write(1, 0)
        # GELU: funct7=0, rs2=0, rs1=x1, funct3=4, rd=x3
        _exec(cpu, _npu_r(0, 0, 1, 4, 3))
        assert cpu.registers.read(3) == 0

    def test_positive_input(self) -> None:
        """GELU of a positive input should be positive."""
        cpu = _make_cpu()
        cpu.registers.write(1, 64)  # Interpreted as int8 = 64
        _exec(cpu, _npu_r(0, 0, 1, 4, 3))
        result = cpu.registers.read(3)
        # GELU(64) should be close to 64 (large positive)
        signed_result = to_signed(result)
        assert signed_result > 0

    def test_negative_input(self) -> None:
        """GELU of a large negative input should be near 0."""
        cpu = _make_cpu()
        # Store -128 as full 32-bit value. Low byte = 0x80 -> signed -128
        cpu.registers.write(1, (-128) & 0xFFFFFFFF)
        _exec(cpu, _npu_r(0, 0, 1, 4, 3))
        result = cpu.registers.read(3)
        signed_result = to_signed(result)
        assert signed_result == 0  # GELU(-128/32) ~ 0

    def test_matches_table(self) -> None:
        """GELU instruction result matches precomputed table for several values."""
        for x in [0, 1, 10, 50, 100, 127]:
            cpu = _make_cpu()
            cpu.registers.write(1, x)
            _exec(cpu, _npu_r(0, 0, 1, 4, 3))
            result = cpu.registers.read(3)
            expected = GELU_TABLE[x + 128]
            # Result is stored as uint32 (sign-extended from int8)
            assert to_signed(result) == expected, (
                f"GELU mismatch at x={x}: got {to_signed(result)}, expected {expected}"
            )

    def test_negative_matches_table(self) -> None:
        """GELU instruction matches table for negative int8 values."""
        for x in [-1, -10, -50, -100, -128]:
            cpu = _make_cpu()
            # Low byte of the register is the int8 value
            cpu.registers.write(1, x & 0xFFFFFFFF)
            _exec(cpu, _npu_r(0, 0, 1, 4, 3))
            result = cpu.registers.read(3)
            expected = GELU_TABLE[x + 128]
            assert to_signed(result) == expected, (
                f"GELU mismatch at x={x}: got {to_signed(result)}, expected {expected}"
            )


# ==================== RSTACC tests ====================

class TestRSTACC:
    """NPU.RSTACC: rd = acc_lo; acc = 0."""

    def test_returns_acc_lo(self) -> None:
        """RSTACC returns the lower 32 bits of the accumulator."""
        cpu = _make_cpu()
        acc_set64(cpu.npu_state, 0x12345678)
        # RSTACC: funct7=0, rs2=0, rs1=0, funct3=5, rd=x3
        _exec(cpu, _npu_r(0, 0, 0, 5, 3))
        assert cpu.registers.read(3) == 0x12345678

    def test_resets_accumulator(self) -> None:
        """RSTACC zeroes both halves of the accumulator."""
        cpu = _make_cpu()
        acc_set64(cpu.npu_state, 0x0000000ADEADBEEF)
        _exec(cpu, _npu_r(0, 0, 0, 5, 3))
        assert cpu.npu_state.acc_lo == 0
        assert cpu.npu_state.acc_hi == 0

    def test_returns_acc_lo_when_hi_set(self) -> None:
        """RSTACC returns acc_lo even when acc_hi is non-zero."""
        cpu = _make_cpu()
        acc_set64(cpu.npu_state, 0xFFFFFFFF00000042)
        _exec(cpu, _npu_r(0, 0, 0, 5, 3))
        assert cpu.registers.read(3) == 0x00000042

    def test_macc_then_rstacc(self) -> None:
        """MACC chain followed by RSTACC returns correct sum."""
        cpu = _make_cpu()
        cpu.registers.write(1, 10)
        cpu.registers.write(2, 20)
        # MACC x 5: 10*20 = 200 each, total 1000
        for i in range(5):
            cpu.pc = BASE + i * 4
            _exec(cpu, _npu_r(0, 2, 1, 0, 0))
        # RSTACC into x3
        cpu.pc = BASE + 5 * 4
        _exec(cpu, _npu_r(0, 0, 0, 5, 3))
        assert cpu.registers.read(3) == 1000
        assert cpu.npu_state.acc_lo == 0
        assert cpu.npu_state.acc_hi == 0


# ==================== LDVEC tests ====================

class TestLDVEC:
    """NPU.LDVEC: vreg[rd%4] = mem32[rs1 + sext(imm)] as 4x int8."""

    def test_load_positive_bytes(self) -> None:
        """Load 4 positive bytes from memory into a vreg."""
        cpu = _make_cpu()
        addr = BASE + 0x100
        cpu.memory.write8(addr, 10)
        cpu.memory.write8(addr + 1, 20)
        cpu.memory.write8(addr + 2, 30)
        cpu.memory.write8(addr + 3, 40)
        cpu.registers.write(1, addr)
        # LDVEC: imm=0, rs1=x1, funct3=6, rd=0 -> vreg[0]
        _exec(cpu, _npu_i(0, 1, 6, 0))
        assert cpu.npu_state.vreg[0] == [10, 20, 30, 40]

    def test_load_negative_bytes(self) -> None:
        """Load bytes with values > 127 (interpreted as negative int8)."""
        cpu = _make_cpu()
        addr = BASE + 0x100
        cpu.memory.write8(addr, 0xFF)      # -1
        cpu.memory.write8(addr + 1, 0x80)  # -128
        cpu.memory.write8(addr + 2, 0x7F)  # 127
        cpu.memory.write8(addr + 3, 0x00)  # 0
        cpu.registers.write(1, addr)
        _exec(cpu, _npu_i(0, 1, 6, 0))
        assert cpu.npu_state.vreg[0] == [-1, -128, 127, 0]

    def test_load_with_offset(self) -> None:
        """LDVEC with non-zero immediate offset."""
        cpu = _make_cpu()
        addr = BASE + 0x200
        cpu.memory.write8(addr + 8, 1)
        cpu.memory.write8(addr + 9, 2)
        cpu.memory.write8(addr + 10, 3)
        cpu.memory.write8(addr + 11, 4)
        cpu.registers.write(1, addr)
        # LDVEC: imm=8, rs1=x1, funct3=6, rd=1 -> vreg[1]
        _exec(cpu, _npu_i(8, 1, 6, 1))
        assert cpu.npu_state.vreg[1] == [1, 2, 3, 4]

    def test_load_vreg_index_modulo(self) -> None:
        """LDVEC with rd=5 loads into vreg[5%4] = vreg[1]."""
        cpu = _make_cpu()
        addr = BASE + 0x100
        for i in range(4):
            cpu.memory.write8(addr + i, i + 10)
        cpu.registers.write(1, addr)
        # rd=5 -> vreg[5%4] = vreg[1]
        _exec(cpu, _npu_i(0, 1, 6, 5))
        assert cpu.npu_state.vreg[1] == [10, 11, 12, 13]


# ==================== STVEC tests ====================

class TestSTVEC:
    """NPU.STVEC: mem32[rs1 + sext(imm)] = vreg[rs2%4] as 4x int8."""

    def test_store_positive_bytes(self) -> None:
        """Store 4 positive int8 values from vreg to memory."""
        cpu = _make_cpu()
        cpu.npu_state.vreg[0] = [10, 20, 30, 40]
        addr = BASE + 0x100
        cpu.registers.write(1, addr)
        # STVEC: imm=0, rs2=0 (vreg[0]), rs1=x1, funct3=7
        _exec(cpu, _npu_s(0, 0, 1, 7))
        assert cpu.memory.read8(addr) == 10
        assert cpu.memory.read8(addr + 1) == 20
        assert cpu.memory.read8(addr + 2) == 30
        assert cpu.memory.read8(addr + 3) == 40

    def test_store_negative_bytes(self) -> None:
        """Store negative int8 values (written as unsigned bytes)."""
        cpu = _make_cpu()
        cpu.npu_state.vreg[1] = [-1, -128, 127, 0]
        addr = BASE + 0x100
        cpu.registers.write(1, addr)
        # STVEC: rs2=1 (vreg[1])
        _exec(cpu, _npu_s(0, 1, 1, 7))
        assert cpu.memory.read8(addr) == 0xFF      # -1
        assert cpu.memory.read8(addr + 1) == 0x80  # -128
        assert cpu.memory.read8(addr + 2) == 0x7F  # 127
        assert cpu.memory.read8(addr + 3) == 0x00  # 0

    def test_store_with_offset(self) -> None:
        """STVEC with non-zero immediate offset."""
        cpu = _make_cpu()
        cpu.npu_state.vreg[2] = [5, 6, 7, 8]
        addr = BASE + 0x200
        cpu.registers.write(1, addr)
        # STVEC: imm=16, rs2=2 (vreg[2])
        _exec(cpu, _npu_s(16, 2, 1, 7))
        for i in range(4):
            assert cpu.memory.read8(addr + 16 + i) == 5 + i

    def test_store_vreg_index_modulo(self) -> None:
        """STVEC with rs2=6 stores vreg[6%4] = vreg[2]."""
        cpu = _make_cpu()
        cpu.npu_state.vreg[2] = [99, 98, 97, 96]
        addr = BASE + 0x100
        cpu.registers.write(1, addr)
        # rs2=6 -> vreg[6%4] = vreg[2]
        _exec(cpu, _npu_s(0, 6, 1, 7))
        assert cpu.memory.read8(addr) == 99
        assert cpu.memory.read8(addr + 1) == 98

    def test_ldvec_stvec_roundtrip(self) -> None:
        """Load 4 bytes, store them to a different address, verify match."""
        cpu = _make_cpu()
        src_addr = BASE + 0x100
        dst_addr = BASE + 0x200
        for i in range(4):
            cpu.memory.write8(src_addr + i, 0x10 + i)

        # LDVEC into vreg[0]
        cpu.registers.write(1, src_addr)
        _exec(cpu, _npu_i(0, 1, 6, 0))

        # STVEC from vreg[0]
        cpu.registers.write(2, dst_addr)
        cpu.pc = BASE + 4
        _exec(cpu, _npu_s(0, 0, 2, 7))

        for i in range(4):
            assert cpu.memory.read8(dst_addr + i) == 0x10 + i


# ==================== Integration tests ====================

class TestNPUIntegration:
    """Integration tests combining multiple NPU instructions."""

    def test_macc_rstacc_pipeline(self) -> None:
        """Full MACC -> RSTACC pipeline through CPU step."""
        cpu = _make_cpu()
        # dot product: [1,2,3,4] . [5,6,7,8] = 5+12+21+32 = 70
        pairs = [(1, 5), (2, 6), (3, 7), (4, 8)]
        for idx, (a, b) in enumerate(pairs):
            cpu.registers.write(1, a)
            cpu.registers.write(2, b)
            cpu.pc = BASE + idx * 4
            _exec(cpu, _npu_r(0, 2, 1, 0, 0))
        # RSTACC
        cpu.pc = BASE + len(pairs) * 4
        _exec(cpu, _npu_r(0, 0, 0, 5, 3))
        assert cpu.registers.read(3) == 70

    def test_relu_then_clamp(self) -> None:
        """RELU followed by CLAMP: relu(200) = 200, clamp(200) = 127."""
        cpu = _make_cpu()
        cpu.registers.write(1, 200)
        _exec(cpu, _npu_r(0, 0, 1, 1, 3))  # RELU -> x3
        assert cpu.registers.read(3) == 200

        cpu.registers.write(1, cpu.registers.read(3))
        cpu.pc = BASE + 4
        _exec(cpu, _npu_r(0, 0, 1, 3, 3))  # CLAMP -> x3
        assert cpu.registers.read(3) == 127

    def test_qmul_then_clamp(self) -> None:
        """QMUL followed by CLAMP: quantized multiply then clamp to int8."""
        cpu = _make_cpu()
        cpu.registers.write(1, 1000)
        cpu.registers.write(2, 1000)
        _exec(cpu, _npu_r(0, 2, 1, 2, 3))  # QMUL -> x3
        result = to_signed(cpu.registers.read(3))
        assert result == (1000 * 1000) >> 8  # = 3906

        # CLAMP the QMUL result
        cpu.registers.write(1, cpu.registers.read(3))
        cpu.pc = BASE + 4
        _exec(cpu, _npu_r(0, 0, 1, 3, 3))  # CLAMP -> x3
        assert cpu.registers.read(3) == 127  # 3906 > 127

    def test_pc_advances(self) -> None:
        """All NPU instructions advance PC by 4."""
        cpu = _make_cpu()
        initial_pc = cpu.pc
        _exec(cpu, _npu_r(0, 0, 0, 1, 0))  # RELU
        assert cpu.pc == initial_pc + 4


# ==================== VEXP tests ====================

class TestVEXP:
    """NPU.VEXP: vectorized exp over Q16.16 int32 array."""

    Q16_ONE = 1 << 16  # 65536 = 1.0 in Q16.16

    def _write_q16_array(self, cpu: CPU, addr: int, values: list[int]) -> None:
        """Write a list of Q16.16 int32 values to memory."""
        for i, v in enumerate(values):
            cpu.memory.write32(addr + i * 4, v & 0xFFFFFFFF)

    def _read_q16_array(self, cpu: CPU, addr: int, count: int) -> list[int]:
        """Read Q16.16 int32 values from memory."""
        return [cpu.memory.read32(addr + i * 4) for i in range(count)]

    def test_vexp_zero(self) -> None:
        """exp(0) = 1.0 in Q16.16 = 65536."""
        cpu = _make_cpu()
        data_base = BASE + 0x1000
        dst_base = BASE + 0x2000
        self._write_q16_array(cpu, data_base, [0])  # 0.0 in Q16.16
        cpu.registers.write(10, data_base)   # rs1 = src
        cpu.registers.write(11, dst_base)    # rs2 = dst
        cpu.registers.write(12, 1)           # rd = count
        # VEXP: funct7=2, rs2=x11, rs1=x10, funct3=0, rd=x12
        _exec(cpu, _npu_r(2, 11, 10, 0, 12))
        result = self._read_q16_array(cpu, dst_base, 1)
        assert result[0] == self.Q16_ONE  # exp(0) = 1.0

    def test_vexp_negative_one(self) -> None:
        """exp(-1.0) ~ 0.3679 in Q16.16 ~ 24109."""
        cpu = _make_cpu()
        data_base = BASE + 0x1000
        dst_base = BASE + 0x2000
        neg_one = (-self.Q16_ONE) & 0xFFFFFFFF  # -1.0 in Q16.16
        self._write_q16_array(cpu, data_base, [neg_one])
        cpu.registers.write(10, data_base)
        cpu.registers.write(11, dst_base)
        cpu.registers.write(12, 1)
        _exec(cpu, _npu_r(2, 11, 10, 0, 12))
        result = self._read_q16_array(cpu, dst_base, 1)
        expected = round(0.367879441 * self.Q16_ONE)  # ~24109
        assert abs(result[0] - expected) <= 1

    def test_vexp_large_negative(self) -> None:
        """exp(-8.0) ~ 0.000335, small positive in Q16.16."""
        cpu = _make_cpu()
        data_base = BASE + 0x1000
        dst_base = BASE + 0x2000
        neg_eight = (-8 * self.Q16_ONE) & 0xFFFFFFFF
        self._write_q16_array(cpu, data_base, [neg_eight])
        cpu.registers.write(10, data_base)
        cpu.registers.write(11, dst_base)
        cpu.registers.write(12, 1)
        _exec(cpu, _npu_r(2, 11, 10, 0, 12))
        result = self._read_q16_array(cpu, dst_base, 1)
        expected = round(0.000335463 * self.Q16_ONE)  # ~22
        assert result[0] > 0
        assert abs(result[0] - expected) <= 1

    def test_vexp_multiple_elements(self) -> None:
        """VEXP on 4 elements: verify each output."""
        import math as _math
        cpu = _make_cpu()
        data_base = BASE + 0x1000
        dst_base = BASE + 0x2000
        inputs = [0, -self.Q16_ONE, -2 * self.Q16_ONE, -4 * self.Q16_ONE]
        self._write_q16_array(cpu, data_base, inputs)
        cpu.registers.write(10, data_base)
        cpu.registers.write(11, dst_base)
        cpu.registers.write(12, 4)
        _exec(cpu, _npu_r(2, 11, 10, 0, 12))
        results = self._read_q16_array(cpu, dst_base, 4)
        for i, x in enumerate(inputs):
            x_float = to_signed(x & 0xFFFFFFFF) / self.Q16_ONE
            expected = round(_math.exp(x_float) * self.Q16_ONE)
            assert abs(results[i] - expected) <= 1, (
                f"VEXP mismatch at index {i}: got {results[i]}, expected {expected}"
            )

    def test_vexp_zero_count(self) -> None:
        """VEXP with n=0 does nothing."""
        cpu = _make_cpu()
        data_base = BASE + 0x1000
        dst_base = BASE + 0x2000
        cpu.memory.write32(dst_base, 0xDEADBEEF)  # sentinel
        cpu.registers.write(10, data_base)
        cpu.registers.write(11, dst_base)
        cpu.registers.write(12, 0)
        _exec(cpu, _npu_r(2, 11, 10, 0, 12))
        assert cpu.memory.read32(dst_base) == 0xDEADBEEF  # untouched


# ==================== VRSQRT tests ====================

class TestVRSQRT:
    """NPU.VRSQRT: scalar 1/sqrt(x) in Q16.16."""

    Q16_ONE = 1 << 16

    def test_vrsqrt_one(self) -> None:
        """rsqrt(1.0) = 1.0 in Q16.16."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        cpu.memory.write32(addr, self.Q16_ONE)  # 1.0
        cpu.registers.write(10, addr)  # rs1 = addr
        # VRSQRT: funct7=3, rs2=0, rs1=x10, funct3=0, rd=x3
        _exec(cpu, _npu_r(3, 0, 10, 0, 3))
        result = cpu.registers.read(3)
        assert result == self.Q16_ONE  # 1/sqrt(1) = 1.0

    def test_vrsqrt_four(self) -> None:
        """rsqrt(4.0) = 0.5 in Q16.16 = 32768."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        cpu.memory.write32(addr, 4 * self.Q16_ONE)  # 4.0
        cpu.registers.write(10, addr)
        _exec(cpu, _npu_r(3, 0, 10, 0, 3))
        result = cpu.registers.read(3)
        expected = self.Q16_ONE // 2  # 0.5 = 32768
        assert result == expected

    def test_vrsqrt_quarter(self) -> None:
        """rsqrt(0.25) = 2.0 in Q16.16 = 131072."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        cpu.memory.write32(addr, self.Q16_ONE // 4)  # 0.25
        cpu.registers.write(10, addr)
        _exec(cpu, _npu_r(3, 0, 10, 0, 3))
        result = cpu.registers.read(3)
        expected = 2 * self.Q16_ONE  # 2.0 = 131072
        assert result == expected

    def test_vrsqrt_large(self) -> None:
        """rsqrt(100.0) = 0.1 in Q16.16 ~ 6554."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        cpu.memory.write32(addr, 100 * self.Q16_ONE)  # 100.0
        cpu.registers.write(10, addr)
        _exec(cpu, _npu_r(3, 0, 10, 0, 3))
        result = cpu.registers.read(3)
        expected = round(0.1 * self.Q16_ONE)  # ~6554
        assert abs(to_signed(result) - expected) <= 1

    def test_vrsqrt_zero_saturates(self) -> None:
        """rsqrt(0) should saturate to max value."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        cpu.memory.write32(addr, 0)  # 0.0
        cpu.registers.write(10, addr)
        _exec(cpu, _npu_r(3, 0, 10, 0, 3))
        result = cpu.registers.read(3)
        assert result == 0x7FFFFFFF  # Saturated max


# ==================== VMUL tests ====================

class TestVMUL:
    """NPU.VMUL: scale int8 vector by Q16.16 factor from accumulator."""

    Q16_ONE = 1 << 16

    def _write_bytes(self, cpu: CPU, addr: int, values: list[int]) -> None:
        """Write a list of int8 values to memory as unsigned bytes."""
        for i, v in enumerate(values):
            cpu.memory.write8(addr + i, v & 0xFF)

    def _read_bytes_signed(self, cpu: CPU, addr: int, count: int) -> list[int]:
        """Read bytes from memory as signed int8 values."""
        result = []
        for i in range(count):
            b = cpu.memory.read8(addr + i)
            result.append(b - 256 if b >= 128 else b)
        return result

    def test_vmul_identity(self) -> None:
        """Scale=1.0 (acc_lo=65536), values pass through unchanged."""
        cpu = _make_cpu()
        src_addr = BASE + 0x1000
        dst_addr = BASE + 0x2000
        values = [10, -20, 50, -100, 127, -128]
        self._write_bytes(cpu, src_addr, values)
        # Set accumulator to 1.0 in Q16.16
        acc_set64(cpu.npu_state, self.Q16_ONE)
        cpu.registers.write(10, src_addr)  # rs1 = src
        cpu.registers.write(11, dst_addr)  # rs2 = dst
        cpu.registers.write(12, 6)         # rd = count
        # VMUL: funct7=4, rs2=x11, rs1=x10, funct3=0, rd=x12
        _exec(cpu, _npu_r(4, 11, 10, 0, 12))
        result = self._read_bytes_signed(cpu, dst_addr, 6)
        assert result == values

    def test_vmul_half(self) -> None:
        """Scale=0.5, values halved."""
        cpu = _make_cpu()
        src_addr = BASE + 0x1000
        dst_addr = BASE + 0x2000
        values = [100, -100, 50, -50]
        self._write_bytes(cpu, src_addr, values)
        acc_set64(cpu.npu_state, self.Q16_ONE // 2)  # 0.5
        cpu.registers.write(10, src_addr)
        cpu.registers.write(11, dst_addr)
        cpu.registers.write(12, 4)
        _exec(cpu, _npu_r(4, 11, 10, 0, 12))
        result = self._read_bytes_signed(cpu, dst_addr, 4)
        # 100*0.5=50, -100*0.5=-50, 50*0.5=25, -50*0.5=-25
        assert result == [50, -50, 25, -25]

    def test_vmul_negative_scale(self) -> None:
        """Negative scale factor inverts sign."""
        cpu = _make_cpu()
        src_addr = BASE + 0x1000
        dst_addr = BASE + 0x2000
        values = [10, -10]
        self._write_bytes(cpu, src_addr, values)
        # -1.0 in Q16.16 as acc_lo (signed interpretation of 0xFFFF0000)
        neg_one = (-self.Q16_ONE) & 0xFFFFFFFF
        acc_set64(cpu.npu_state, neg_one)
        cpu.registers.write(10, src_addr)
        cpu.registers.write(11, dst_addr)
        cpu.registers.write(12, 2)
        _exec(cpu, _npu_r(4, 11, 10, 0, 12))
        result = self._read_bytes_signed(cpu, dst_addr, 2)
        assert result == [-10, 10]

    def test_vmul_clamps_to_int8(self) -> None:
        """Overflow is clamped to [-128, 127]."""
        cpu = _make_cpu()
        src_addr = BASE + 0x1000
        dst_addr = BASE + 0x2000
        values = [100, -100]
        self._write_bytes(cpu, src_addr, values)
        # Scale = 2.0 -> 100*2=200 > 127, should clamp to 127
        acc_set64(cpu.npu_state, 2 * self.Q16_ONE)
        cpu.registers.write(10, src_addr)
        cpu.registers.write(11, dst_addr)
        cpu.registers.write(12, 2)
        _exec(cpu, _npu_r(4, 11, 10, 0, 12))
        result = self._read_bytes_signed(cpu, dst_addr, 2)
        assert result == [127, -128]

    def test_vmul_zero_count(self) -> None:
        """VMUL with n=0 does nothing."""
        cpu = _make_cpu()
        dst_addr = BASE + 0x2000
        cpu.memory.write8(dst_addr, 0xAB)  # sentinel
        acc_set64(cpu.npu_state, self.Q16_ONE)
        cpu.registers.write(10, BASE + 0x1000)
        cpu.registers.write(11, dst_addr)
        cpu.registers.write(12, 0)
        _exec(cpu, _npu_r(4, 11, 10, 0, 12))
        assert cpu.memory.read8(dst_addr) == 0xAB  # untouched

    def test_vmul_preserves_accumulator(self) -> None:
        """VMUL does not modify the accumulator."""
        cpu = _make_cpu()
        src_addr = BASE + 0x1000
        dst_addr = BASE + 0x2000
        cpu.memory.write8(src_addr, 10)
        acc_set64(cpu.npu_state, self.Q16_ONE)
        cpu.registers.write(10, src_addr)
        cpu.registers.write(11, dst_addr)
        cpu.registers.write(12, 1)
        _exec(cpu, _npu_r(4, 11, 10, 0, 12))
        assert acc_get64(cpu.npu_state) == self.Q16_ONE


# ==================== VREDUCE tests ====================

class TestVREDUCE:
    """NPU.VREDUCE: sum int32 array into rd."""

    def _write_int32_array(self, cpu: CPU, addr: int, values: list[int]) -> None:
        """Write int32 values to memory."""
        for i, v in enumerate(values):
            cpu.memory.write32(addr + i * 4, v & 0xFFFFFFFF)

    def test_vreduce_basic(self) -> None:
        """Sum of [1, 2, 3, 4] = 10."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        self._write_int32_array(cpu, addr, [1, 2, 3, 4])
        cpu.registers.write(10, addr)  # rs1 = addr
        cpu.registers.write(11, 4)     # rs2 = count
        # VREDUCE: funct7=5, rs2=x11, rs1=x10, funct3=0, rd=x3
        _exec(cpu, _npu_r(5, 11, 10, 0, 3))
        assert cpu.registers.read(3) == 10

    def test_vreduce_negative(self) -> None:
        """Sum including negatives: [10, -3, 5, -7] = 5."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        self._write_int32_array(cpu, addr, [10, -3, 5, -7])
        cpu.registers.write(10, addr)
        cpu.registers.write(11, 4)
        _exec(cpu, _npu_r(5, 11, 10, 0, 3))
        assert to_signed(cpu.registers.read(3)) == 5

    def test_vreduce_single(self) -> None:
        """Single element: [42] -> 42."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        self._write_int32_array(cpu, addr, [42])
        cpu.registers.write(10, addr)
        cpu.registers.write(11, 1)
        _exec(cpu, _npu_r(5, 11, 10, 0, 3))
        assert cpu.registers.read(3) == 42

    def test_vreduce_zero_count(self) -> None:
        """VREDUCE with n=0 returns 0."""
        cpu = _make_cpu()
        cpu.registers.write(10, BASE + 0x1000)
        cpu.registers.write(11, 0)
        _exec(cpu, _npu_r(5, 11, 10, 0, 3))
        assert cpu.registers.read(3) == 0

    def test_vreduce_large_values(self) -> None:
        """VREDUCE with large values to test 32-bit wrapping."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        self._write_int32_array(cpu, addr, [0x7FFFFFFF, 1])
        cpu.registers.write(10, addr)
        cpu.registers.write(11, 2)
        _exec(cpu, _npu_r(5, 11, 10, 0, 3))
        # 0x7FFFFFFF + 1 = 0x80000000 (wraps to negative in int32)
        result = cpu.registers.read(3)
        assert result == 0x80000000


# ==================== VMAX tests ====================

class TestVMAX:
    """NPU.VMAX: find maximum of int32 array."""

    def _write_int32_array(self, cpu: CPU, addr: int, values: list[int]) -> None:
        """Write int32 values to memory."""
        for i, v in enumerate(values):
            cpu.memory.write32(addr + i * 4, v & 0xFFFFFFFF)

    def test_vmax_basic(self) -> None:
        """Max of [1, 5, 3, 2] = 5."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        self._write_int32_array(cpu, addr, [1, 5, 3, 2])
        cpu.registers.write(10, addr)  # rs1 = addr
        cpu.registers.write(11, 4)     # rs2 = count
        # VMAX: funct7=6, rs2=x11, rs1=x10, funct3=0, rd=x3
        _exec(cpu, _npu_r(6, 11, 10, 0, 3))
        assert cpu.registers.read(3) == 5

    def test_vmax_all_negative(self) -> None:
        """Max of [-3, -1, -5] = -1."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        self._write_int32_array(cpu, addr, [-3, -1, -5])
        cpu.registers.write(10, addr)
        cpu.registers.write(11, 3)
        _exec(cpu, _npu_r(6, 11, 10, 0, 3))
        result = to_signed(cpu.registers.read(3))
        assert result == -1

    def test_vmax_single(self) -> None:
        """Single element: [42] -> 42."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        self._write_int32_array(cpu, addr, [42])
        cpu.registers.write(10, addr)
        cpu.registers.write(11, 1)
        _exec(cpu, _npu_r(6, 11, 10, 0, 3))
        assert cpu.registers.read(3) == 42

    def test_vmax_zero_count(self) -> None:
        """VMAX with n=0 returns 0x80000000 (minimum int32)."""
        cpu = _make_cpu()
        cpu.registers.write(10, BASE + 0x1000)
        cpu.registers.write(11, 0)
        _exec(cpu, _npu_r(6, 11, 10, 0, 3))
        assert cpu.registers.read(3) == 0x80000000

    def test_vmax_mixed(self) -> None:
        """Max of mixed positive and negative: [-100, 0, 100, -50] = 100."""
        cpu = _make_cpu()
        addr = BASE + 0x1000
        self._write_int32_array(cpu, addr, [-100, 0, 100, -50])
        cpu.registers.write(10, addr)
        cpu.registers.write(11, 4)
        _exec(cpu, _npu_r(6, 11, 10, 0, 3))
        assert cpu.registers.read(3) == 100
