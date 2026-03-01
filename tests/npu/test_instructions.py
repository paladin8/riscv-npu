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
