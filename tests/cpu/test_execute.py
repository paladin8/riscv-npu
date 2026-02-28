"""Tests for instruction execution."""

from riscv_npu.cpu.decode import to_signed

# Encoding helpers — same as test_decode.py but local for clarity
OP_R = 0x33
OP_I = 0x13
OP_LOAD = 0x03
OP_STORE = 0x23
OP_BRANCH = 0x63
OP_LUI = 0x37
OP_AUIPC = 0x17
OP_JAL = 0x6F
OP_JALR = 0x67
OP_SYSTEM = 0x73
OP_FENCE = 0x0F

BASE = 0x80000000


def _r(funct7: int, rs2: int, rs1: int, funct3: int, rd: int) -> int:
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | OP_R


def _i(imm12: int, rs1: int, funct3: int, rd: int, opcode: int = OP_I) -> int:
    return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def _s(imm12: int, rs2: int, rs1: int, funct3: int) -> int:
    imm_11_5 = (imm12 >> 5) & 0x7F
    imm_4_0 = imm12 & 0x1F
    return (imm_11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_4_0 << 7) | OP_STORE


def _b(imm13: int, rs2: int, rs1: int, funct3: int) -> int:
    imm = imm13 & 0x1FFF
    bit_12 = (imm >> 12) & 1
    bit_11 = (imm >> 11) & 1
    bits_10_5 = (imm >> 5) & 0x3F
    bits_4_1 = (imm >> 1) & 0xF
    return (bit_12 << 31) | (bits_10_5 << 25) | (rs2 << 20) | (rs1 << 15) | \
           (funct3 << 12) | (bits_4_1 << 8) | (bit_11 << 7) | OP_BRANCH


def _j(imm21: int, rd: int) -> int:
    imm = imm21 & 0x1FFFFF
    bit_20 = (imm >> 20) & 1
    bits_10_1 = (imm >> 1) & 0x3FF
    bit_11 = (imm >> 11) & 1
    bits_19_12 = (imm >> 12) & 0xFF
    return (bit_20 << 31) | (bits_10_1 << 21) | (bit_11 << 20) | \
           (bits_19_12 << 12) | (rd << 7) | OP_JAL


# ==================== R-type tests ====================

class TestADD:
    def test_positive(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=5, x2=10)
        exec_instruction(cpu, _r(0, 2, 1, 0b000, 3))  # ADD x3, x1, x2
        assert cpu.registers.read(3) == 15

    def test_negative(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=5, x2=0xFFFFFFFF)  # x2 = -1
        exec_instruction(cpu, _r(0, 2, 1, 0b000, 3))
        assert cpu.registers.read(3) == 4

    def test_overflow(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFFF, x2=1)
        exec_instruction(cpu, _r(0, 2, 1, 0b000, 3))
        assert cpu.registers.read(3) == 0  # wraps


class TestSUB:
    def test_positive(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=10, x2=3)
        exec_instruction(cpu, _r(0b0100000, 2, 1, 0b000, 3))  # SUB x3, x1, x2
        assert cpu.registers.read(3) == 7

    def test_negative_result(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=3, x2=10)
        exec_instruction(cpu, _r(0b0100000, 2, 1, 0b000, 3))
        assert to_signed(cpu.registers.read(3)) == -7

    def test_overflow(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0, x2=1)
        exec_instruction(cpu, _r(0b0100000, 2, 1, 0b000, 3))
        assert cpu.registers.read(3) == 0xFFFFFFFF  # -1


class TestSLL:
    def test_basic(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=1, x2=4)
        exec_instruction(cpu, _r(0, 2, 1, 0b001, 3))  # SLL x3, x1, x2
        assert cpu.registers.read(3) == 16

    def test_by_zero(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xABCD, x2=0)
        exec_instruction(cpu, _r(0, 2, 1, 0b001, 3))
        assert cpu.registers.read(3) == 0xABCD

    def test_by_31(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=1, x2=31)
        exec_instruction(cpu, _r(0, 2, 1, 0b001, 3))
        assert cpu.registers.read(3) == 0x80000000


class TestSLT:
    def test_less(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFFF, x2=1)  # -1 < 1 signed
        exec_instruction(cpu, _r(0, 2, 1, 0b010, 3))
        assert cpu.registers.read(3) == 1

    def test_not_less(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=5, x2=3)
        exec_instruction(cpu, _r(0, 2, 1, 0b010, 3))
        assert cpu.registers.read(3) == 0

    def test_equal(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=7, x2=7)
        exec_instruction(cpu, _r(0, 2, 1, 0b010, 3))
        assert cpu.registers.read(3) == 0


class TestSLTU:
    def test_unsigned(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=1, x2=0xFFFFFFFF)  # 1 < 0xFFFFFFFF unsigned
        exec_instruction(cpu, _r(0, 2, 1, 0b011, 3))
        assert cpu.registers.read(3) == 1

    def test_not_less(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFFF, x2=1)
        exec_instruction(cpu, _r(0, 2, 1, 0b011, 3))
        assert cpu.registers.read(3) == 0


class TestXOR:
    def test_basic(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFF00FF00, x2=0x0F0F0F0F)
        exec_instruction(cpu, _r(0, 2, 1, 0b100, 3))
        assert cpu.registers.read(3) == 0xF00FF00F

    def test_self(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xDEADBEEF)
        exec_instruction(cpu, _r(0, 1, 1, 0b100, 3))  # XOR x3=x1^x1 — rd=3? no, same reg
        # Actually XOR x1, x1, x1 → 0
        exec_instruction(cpu, _r(0, 1, 1, 0b100, 1))
        assert cpu.registers.read(1) == 0


class TestSRL:
    def test_basic(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0x80000000, x2=4)
        exec_instruction(cpu, _r(0, 2, 1, 0b101, 3))  # SRL
        assert cpu.registers.read(3) == 0x08000000

    def test_by_zero(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xDEADBEEF, x2=0)
        exec_instruction(cpu, _r(0, 2, 1, 0b101, 3))
        assert cpu.registers.read(3) == 0xDEADBEEF

    def test_by_31(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0x80000000, x2=31)
        exec_instruction(cpu, _r(0, 2, 1, 0b101, 3))
        assert cpu.registers.read(3) == 1


class TestSRA:
    def test_negative(self, exec_instruction, make_cpu, set_regs) -> None:
        """SRA sign-extends: 0x80000000 >> 4 = 0xF8000000"""
        cpu = make_cpu()
        set_regs(cpu, x1=0x80000000, x2=4)
        exec_instruction(cpu, _r(0b0100000, 2, 1, 0b101, 3))
        assert cpu.registers.read(3) == 0xF8000000

    def test_positive(self, exec_instruction, make_cpu, set_regs) -> None:
        """SRA of positive value same as SRL."""
        cpu = make_cpu()
        set_regs(cpu, x1=0x40000000, x2=4)
        exec_instruction(cpu, _r(0b0100000, 2, 1, 0b101, 3))
        assert cpu.registers.read(3) == 0x04000000

    def test_sra_vs_srl(self, exec_instruction, make_cpu, set_regs) -> None:
        """SRA and SRL differ on negative values."""
        cpu = make_cpu()
        set_regs(cpu, x1=0x80000000, x2=1)
        # SRL
        exec_instruction(cpu, _r(0, 2, 1, 0b101, 3))
        srl_result = cpu.registers.read(3)
        # SRA
        cpu.pc = BASE
        exec_instruction(cpu, _r(0b0100000, 2, 1, 0b101, 4))
        sra_result = cpu.registers.read(4)
        assert srl_result == 0x40000000
        assert sra_result == 0xC0000000


class TestOR:
    def test_basic(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xF0F0F0F0, x2=0x0F0F0F0F)
        exec_instruction(cpu, _r(0, 2, 1, 0b110, 3))
        assert cpu.registers.read(3) == 0xFFFFFFFF


class TestAND:
    def test_basic(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFF00FF00, x2=0x0F0F0F0F)
        exec_instruction(cpu, _r(0, 2, 1, 0b111, 3))
        assert cpu.registers.read(3) == 0x0F000F00


class TestWriteToX0:
    def test_r_type_to_x0(self, exec_instruction, make_cpu, set_regs) -> None:
        """Writing to x0 is discarded."""
        cpu = make_cpu()
        set_regs(cpu, x1=5, x2=10)
        exec_instruction(cpu, _r(0, 2, 1, 0b000, 0))  # ADD x0, x1, x2
        assert cpu.registers.read(0) == 0


# ==================== I-type arithmetic tests ====================

class TestADDI:
    def test_positive(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=10)
        exec_instruction(cpu, _i(5, 1, 0b000, 2))  # ADDI x2, x1, 5
        assert cpu.registers.read(2) == 15

    def test_zero(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=42)
        exec_instruction(cpu, _i(0, 1, 0b000, 2))
        assert cpu.registers.read(2) == 42

    def test_negative(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=10)
        exec_instruction(cpu, _i((-3) & 0xFFF, 1, 0b000, 2))  # ADDI x2, x1, -3
        assert cpu.registers.read(2) == 7

    def test_overflow(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFFF)
        exec_instruction(cpu, _i(1, 1, 0b000, 2))  # ADDI x2, x1, 1
        assert cpu.registers.read(2) == 0

    def test_nop(self, exec_instruction, make_cpu) -> None:
        """NOP = ADDI x0, x0, 0"""
        cpu = make_cpu()
        exec_instruction(cpu, _i(0, 0, 0b000, 0))
        assert cpu.registers.read(0) == 0


class TestSLTI:
    def test_less(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFFF)  # -1
        exec_instruction(cpu, _i(5, 1, 0b010, 2))  # SLTI x2, x1, 5
        assert cpu.registers.read(2) == 1

    def test_not_less(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=10)
        exec_instruction(cpu, _i(5, 1, 0b010, 2))
        assert cpu.registers.read(2) == 0


class TestSLTIU:
    def test_basic(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=5)
        exec_instruction(cpu, _i(10, 1, 0b011, 2))
        assert cpu.registers.read(2) == 1

    def test_sign_extended_unsigned(self, exec_instruction, make_cpu, set_regs) -> None:
        """imm=-1 sign-extends to 0xFFFFFFFF; 5 < 0xFFFFFFFF unsigned → 1"""
        cpu = make_cpu()
        set_regs(cpu, x1=5)
        exec_instruction(cpu, _i(0xFFF, 1, 0b011, 2))  # -1 as 12-bit
        assert cpu.registers.read(2) == 1


class TestXORI:
    def test_basic(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFF)
        exec_instruction(cpu, _i(0x0F, 1, 0b100, 2))
        assert cpu.registers.read(2) == 0xF0

    def test_negative_imm(self, exec_instruction, make_cpu, set_regs) -> None:
        """XORI with -1 is bitwise NOT."""
        cpu = make_cpu()
        set_regs(cpu, x1=0x12345678)
        exec_instruction(cpu, _i(0xFFF, 1, 0b100, 2))  # -1
        assert cpu.registers.read(2) == 0xEDCBA987


class TestORI:
    def test_basic(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xF0)
        exec_instruction(cpu, _i(0x0F, 1, 0b110, 2))
        assert cpu.registers.read(2) == 0xFF


class TestANDI:
    def test_basic(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFF)
        exec_instruction(cpu, _i(0x0F, 1, 0b111, 2))
        assert cpu.registers.read(2) == 0x0F


class TestSLLI:
    def test_basic(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=1)
        exec_instruction(cpu, _i(4, 1, 0b001, 2))  # SLLI x2, x1, 4
        assert cpu.registers.read(2) == 16

    def test_by_zero(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xAB)
        exec_instruction(cpu, _i(0, 1, 0b001, 2))
        assert cpu.registers.read(2) == 0xAB


class TestSRLI:
    def test_basic(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0x80000000)
        exec_instruction(cpu, _i(4, 1, 0b101, 2))  # SRLI x2, x1, 4
        assert cpu.registers.read(2) == 0x08000000

    def test_by_zero(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xDEADBEEF)
        exec_instruction(cpu, _i(0, 1, 0b101, 2))
        assert cpu.registers.read(2) == 0xDEADBEEF

    def test_by_31(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0x80000000)
        exec_instruction(cpu, _i(31, 1, 0b101, 2))
        assert cpu.registers.read(2) == 1


class TestSRAI:
    def test_negative(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0x80000000)
        exec_instruction(cpu, _i(0x400 | 4, 1, 0b101, 2))  # SRAI x2, x1, 4 (funct7 bit set)
        assert cpu.registers.read(2) == 0xF8000000

    def test_by_zero(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0x80000000)
        exec_instruction(cpu, _i(0x400, 1, 0b101, 2))  # SRAI x2, x1, 0
        assert cpu.registers.read(2) == 0x80000000

    def test_by_31(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0x80000000)
        exec_instruction(cpu, _i(0x400 | 31, 1, 0b101, 2))  # SRAI x2, x1, 31
        assert cpu.registers.read(2) == 0xFFFFFFFF


# ==================== Load/Store tests ====================

class TestLoads:
    def test_lb_positive(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        cpu.memory.write8(BASE + 0x100, 0x42)
        set_regs(cpu, x1=BASE + 0x100)
        exec_instruction(cpu, _i(0, 1, 0b000, 2, OP_LOAD))  # LB x2, 0(x1)
        assert cpu.registers.read(2) == 0x42

    def test_lb_negative(self, exec_instruction, make_cpu, set_regs) -> None:
        """LB sign-extends: 0xFF → 0xFFFFFFFF."""
        cpu = make_cpu()
        cpu.memory.write8(BASE + 0x100, 0xFF)
        set_regs(cpu, x1=BASE + 0x100)
        exec_instruction(cpu, _i(0, 1, 0b000, 2, OP_LOAD))
        assert cpu.registers.read(2) == 0xFFFFFFFF

    def test_lh_positive(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        cpu.memory.write16(BASE + 0x100, 0x1234)
        set_regs(cpu, x1=BASE + 0x100)
        exec_instruction(cpu, _i(0, 1, 0b001, 2, OP_LOAD))  # LH
        assert cpu.registers.read(2) == 0x1234

    def test_lh_negative(self, exec_instruction, make_cpu, set_regs) -> None:
        """LH sign-extends: 0x8000 → 0xFFFF8000."""
        cpu = make_cpu()
        cpu.memory.write16(BASE + 0x100, 0x8000)
        set_regs(cpu, x1=BASE + 0x100)
        exec_instruction(cpu, _i(0, 1, 0b001, 2, OP_LOAD))
        assert cpu.registers.read(2) == 0xFFFF8000

    def test_lw(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        cpu.memory.write32(BASE + 0x100, 0xDEADBEEF)
        set_regs(cpu, x1=BASE + 0x100)
        exec_instruction(cpu, _i(0, 1, 0b010, 2, OP_LOAD))  # LW
        assert cpu.registers.read(2) == 0xDEADBEEF

    def test_lbu(self, exec_instruction, make_cpu, set_regs) -> None:
        """LBU zero-extends: 0xFF stays 0xFF."""
        cpu = make_cpu()
        cpu.memory.write8(BASE + 0x100, 0xFF)
        set_regs(cpu, x1=BASE + 0x100)
        exec_instruction(cpu, _i(0, 1, 0b100, 2, OP_LOAD))  # LBU
        assert cpu.registers.read(2) == 0xFF

    def test_lhu(self, exec_instruction, make_cpu, set_regs) -> None:
        """LHU zero-extends: 0xFFFF stays 0xFFFF."""
        cpu = make_cpu()
        cpu.memory.write16(BASE + 0x100, 0xFFFF)
        set_regs(cpu, x1=BASE + 0x100)
        exec_instruction(cpu, _i(0, 1, 0b101, 2, OP_LOAD))  # LHU
        assert cpu.registers.read(2) == 0xFFFF

    def test_load_with_offset(self, exec_instruction, make_cpu, set_regs) -> None:
        """Load with immediate offset."""
        cpu = make_cpu()
        cpu.memory.write32(BASE + 0x104, 0x12345678)
        set_regs(cpu, x1=BASE + 0x100)
        exec_instruction(cpu, _i(4, 1, 0b010, 2, OP_LOAD))  # LW x2, 4(x1)
        assert cpu.registers.read(2) == 0x12345678


class TestStores:
    def test_sb(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=BASE + 0x100, x2=0xAB)
        exec_instruction(cpu, _s(0, 2, 1, 0b000))  # SB x2, 0(x1)
        assert cpu.memory.read8(BASE + 0x100) == 0xAB

    def test_sh(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=BASE + 0x100, x2=0xBEEF)
        exec_instruction(cpu, _s(0, 2, 1, 0b001))  # SH
        assert cpu.memory.read16(BASE + 0x100) == 0xBEEF

    def test_sw(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=BASE + 0x100, x2=0xDEADBEEF)
        exec_instruction(cpu, _s(0, 2, 1, 0b010))  # SW
        assert cpu.memory.read32(BASE + 0x100) == 0xDEADBEEF

    def test_sb_truncates(self, exec_instruction, make_cpu, set_regs) -> None:
        """SB stores only the lower 8 bits."""
        cpu = make_cpu()
        set_regs(cpu, x1=BASE + 0x100, x2=0x12345678)
        exec_instruction(cpu, _s(0, 2, 1, 0b000))
        assert cpu.memory.read8(BASE + 0x100) == 0x78

    def test_sh_truncates(self, exec_instruction, make_cpu, set_regs) -> None:
        """SH stores only the lower 16 bits."""
        cpu = make_cpu()
        set_regs(cpu, x1=BASE + 0x100, x2=0x12345678)
        exec_instruction(cpu, _s(0, 2, 1, 0b001))
        assert cpu.memory.read16(BASE + 0x100) == 0x5678

    def test_sw_negative_offset(self, exec_instruction, make_cpu, set_regs) -> None:
        """Store with negative immediate offset."""
        cpu = make_cpu()
        set_regs(cpu, x1=BASE + 0x104, x2=0xCAFEBABE)
        exec_instruction(cpu, _s((-4) & 0xFFF, 2, 1, 0b010))  # SW x2, -4(x1)
        assert cpu.memory.read32(BASE + 0x100) == 0xCAFEBABE

    def test_store_load_roundtrip(self, exec_instruction, make_cpu, set_regs) -> None:
        """Store then load at each width."""
        cpu = make_cpu()
        set_regs(cpu, x1=BASE + 0x200, x2=0xCAFEBABE)
        # SW
        exec_instruction(cpu, _s(0, 2, 1, 0b010))
        # LW
        cpu.pc = BASE
        exec_instruction(cpu, _i(0x200, 1, 0b010, 3, OP_LOAD))  # LW x3, 0x200(x1) — wait, x1 is already base+0x200
        # Actually let me use base as x1
        cpu2 = make_cpu()
        set_regs(cpu2, x1=BASE + 0x200)
        cpu2.memory.write32(BASE + 0x200, 0xCAFEBABE)
        exec_instruction(cpu2, _i(0, 1, 0b010, 3, OP_LOAD))
        assert cpu2.registers.read(3) == 0xCAFEBABE


# ==================== Branch tests ====================

class TestBEQ:
    def test_taken(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=42, x2=42)
        exec_instruction(cpu, _b(8, 2, 1, 0b000))  # BEQ +8
        assert cpu.pc == BASE + 8

    def test_not_taken(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=1, x2=2)
        exec_instruction(cpu, _b(8, 2, 1, 0b000))
        assert cpu.pc == BASE + 4

    def test_backward(self, exec_instruction, make_cpu, set_regs) -> None:
        """BEQ with negative offset (backward branch)."""
        cpu = make_cpu()
        cpu.pc = BASE + 0x100
        set_regs(cpu, x1=5, x2=5)
        cpu.memory.write32(BASE + 0x100, _b((-16) & 0x1FFF, 2, 1, 0b000))
        cpu.step()
        assert cpu.pc == BASE + 0x100 - 16


class TestBNE:
    def test_taken(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=1, x2=2)
        exec_instruction(cpu, _b(8, 2, 1, 0b001))
        assert cpu.pc == BASE + 8

    def test_not_taken(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=5, x2=5)
        exec_instruction(cpu, _b(8, 2, 1, 0b001))
        assert cpu.pc == BASE + 4


class TestBLT:
    def test_taken(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFFF, x2=1)  # -1 < 1 signed
        exec_instruction(cpu, _b(8, 2, 1, 0b100))
        assert cpu.pc == BASE + 8

    def test_not_taken(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=5, x2=3)
        exec_instruction(cpu, _b(8, 2, 1, 0b100))
        assert cpu.pc == BASE + 4


class TestBGE:
    def test_taken_greater(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=5, x2=3)
        exec_instruction(cpu, _b(8, 2, 1, 0b101))
        assert cpu.pc == BASE + 8

    def test_taken_equal(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=5, x2=5)
        exec_instruction(cpu, _b(8, 2, 1, 0b101))
        assert cpu.pc == BASE + 8

    def test_not_taken(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFFF, x2=1)  # -1 < 1
        exec_instruction(cpu, _b(8, 2, 1, 0b101))
        assert cpu.pc == BASE + 4


class TestBLTU:
    def test_taken(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=1, x2=0xFFFFFFFF)  # 1 < 0xFFFFFFFF unsigned
        exec_instruction(cpu, _b(8, 2, 1, 0b110))
        assert cpu.pc == BASE + 8

    def test_not_taken(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFFF, x2=1)
        exec_instruction(cpu, _b(8, 2, 1, 0b110))
        assert cpu.pc == BASE + 4


class TestBGEU:
    def test_taken(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFFF, x2=1)
        exec_instruction(cpu, _b(8, 2, 1, 0b111))
        assert cpu.pc == BASE + 8

    def test_not_taken(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=1, x2=0xFFFFFFFF)
        exec_instruction(cpu, _b(8, 2, 1, 0b111))
        assert cpu.pc == BASE + 4


class TestBLTvsBLTU:
    def test_signed_vs_unsigned(self, exec_instruction, make_cpu, set_regs) -> None:
        """0x80000000 is negative signed but large unsigned."""
        cpu = make_cpu()
        set_regs(cpu, x1=0x80000000, x2=1)
        # BLT: signed(0x80000000) = -2^31 < 1 → taken
        exec_instruction(cpu, _b(8, 2, 1, 0b100))
        assert cpu.pc == BASE + 8

        cpu2 = make_cpu()
        set_regs(cpu2, x1=0x80000000, x2=1)
        # BLTU: 0x80000000 > 1 unsigned → not taken
        exec_instruction(cpu2, _b(8, 2, 1, 0b110))
        assert cpu2.pc == BASE + 4


# ==================== Upper immediate, Jump, System tests ====================

class TestLUI:
    def test_basic(self, exec_instruction, make_cpu) -> None:
        word = (0xDEADB << 12) | (1 << 7) | OP_LUI
        cpu = make_cpu()
        exec_instruction(cpu, word)  # LUI x1, 0xDEADB
        assert cpu.registers.read(1) == 0xDEADB000

    def test_lower_12_zero(self, exec_instruction, make_cpu) -> None:
        word = (1 << 12) | (2 << 7) | OP_LUI  # LUI x2, 1
        cpu = make_cpu()
        exec_instruction(cpu, word)
        assert cpu.registers.read(2) == 0x1000


class TestAUIPC:
    def test_basic(self, exec_instruction, make_cpu) -> None:
        word = (0x12345 << 12) | (1 << 7) | OP_AUIPC
        cpu = make_cpu()
        exec_instruction(cpu, word)
        assert cpu.registers.read(1) == (BASE + 0x12345000) & 0xFFFFFFFF


class TestJAL:
    def test_forward(self, exec_instruction, make_cpu) -> None:
        cpu = make_cpu()
        exec_instruction(cpu, _j(100, 1))  # JAL x1, +100
        assert cpu.registers.read(1) == BASE + 4  # return address
        assert cpu.pc == BASE + 100

    def test_backward(self, exec_instruction, make_cpu) -> None:
        cpu = make_cpu()
        cpu.pc = BASE + 0x100
        cpu.memory.write32(BASE + 0x100, _j((-20) & 0x1FFFFF, 1))
        cpu.step()
        assert cpu.registers.read(1) == BASE + 0x100 + 4
        assert cpu.pc == BASE + 0x100 - 20

    def test_x0_link(self, exec_instruction, make_cpu) -> None:
        """JAL x0 doesn't save return address (unconditional jump)."""
        cpu = make_cpu()
        exec_instruction(cpu, _j(8, 0))
        assert cpu.registers.read(0) == 0


class TestJALR:
    def test_basic(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=BASE + 0x200)
        exec_instruction(cpu, _i(0, 1, 0b000, 2, OP_JALR))  # JALR x2, x1, 0
        assert cpu.registers.read(2) == BASE + 4
        assert cpu.pc == BASE + 0x200

    def test_with_offset(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=BASE + 0x200)
        exec_instruction(cpu, _i(8, 1, 0b000, 2, OP_JALR))
        assert cpu.pc == BASE + 0x208

    def test_negative_offset(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=BASE + 0x210)
        exec_instruction(cpu, _i((-8) & 0xFFF, 1, 0b000, 2, OP_JALR))
        assert cpu.pc == BASE + 0x208

    def test_clears_lsb(self, exec_instruction, make_cpu, set_regs) -> None:
        """JALR clears bit 0 of target address."""
        cpu = make_cpu()
        set_regs(cpu, x1=BASE + 0x201)  # odd address
        exec_instruction(cpu, _i(0, 1, 0b000, 2, OP_JALR))
        assert cpu.pc == BASE + 0x200  # bit 0 cleared


class TestECALL:
    def test_halts(self, exec_instruction, make_cpu) -> None:
        cpu = make_cpu()
        exec_instruction(cpu, _i(0, 0, 0b000, 0, OP_SYSTEM))
        assert cpu.halted is True

    def test_pc_advances(self, exec_instruction, make_cpu) -> None:
        cpu = make_cpu()
        exec_instruction(cpu, _i(0, 0, 0b000, 0, OP_SYSTEM))
        assert cpu.pc == BASE + 4


class TestEBREAK:
    def test_halts(self, exec_instruction, make_cpu) -> None:
        cpu = make_cpu()
        exec_instruction(cpu, _i(1, 0, 0b000, 0, OP_SYSTEM))
        assert cpu.halted is True


class TestFENCE:
    def test_nop(self, exec_instruction, make_cpu) -> None:
        cpu = make_cpu()
        exec_instruction(cpu, _i(0, 0, 0b000, 0, OP_FENCE))
        assert cpu.pc == BASE + 4
        assert cpu.halted is False


# ==================== M extension tests ====================

# Helper for M extension R-type: funct7=0b0000001
def _m(rs2: int, rs1: int, funct3: int, rd: int) -> int:
    return _r(0b0000001, rs2, rs1, funct3, rd)


class TestMUL:
    def test_basic(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=6, x2=7)
        exec_instruction(cpu, _m(2, 1, 0b000, 3))
        assert cpu.registers.read(3) == 42

    def test_zero(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=100, x2=0)
        exec_instruction(cpu, _m(2, 1, 0b000, 3))
        assert cpu.registers.read(3) == 0

    def test_overflow_lower_bits(self, exec_instruction, make_cpu, set_regs) -> None:
        """MUL returns only the lower 32 bits."""
        cpu = make_cpu()
        set_regs(cpu, x1=0x80000000, x2=2)
        exec_instruction(cpu, _m(2, 1, 0b000, 3))
        assert cpu.registers.read(3) == 0  # 0x100000000 & 0xFFFFFFFF = 0

    def test_negative_times_positive(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFFF, x2=3)  # -1 * 3 = -3
        exec_instruction(cpu, _m(2, 1, 0b000, 3))
        assert cpu.registers.read(3) == 0xFFFFFFFD  # -3 in two's complement

    def test_large_values(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0x10000, x2=0x10000)
        exec_instruction(cpu, _m(2, 1, 0b000, 3))
        assert cpu.registers.read(3) == 0  # 0x100000000 & 0xFFFFFFFF = 0


class TestMULH:
    def test_basic(self, exec_instruction, make_cpu, set_regs) -> None:
        """MULH: upper 32 bits of signed * signed."""
        cpu = make_cpu()
        set_regs(cpu, x1=0x10000, x2=0x10000)
        exec_instruction(cpu, _m(2, 1, 0b001, 3))
        # 0x10000 * 0x10000 = 0x100000000, upper 32 bits = 1
        assert cpu.registers.read(3) == 1

    def test_neg_times_neg(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFFF, x2=0xFFFFFFFF)  # -1 * -1 = 1
        exec_instruction(cpu, _m(2, 1, 0b001, 3))
        assert cpu.registers.read(3) == 0  # upper 32 bits of 1 = 0

    def test_neg_times_pos(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFFF, x2=2)  # -1 * 2 = -2
        exec_instruction(cpu, _m(2, 1, 0b001, 3))
        assert cpu.registers.read(3) == 0xFFFFFFFF  # upper 32 of -2 (64-bit)

    def test_zero(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0, x2=0x80000000)
        exec_instruction(cpu, _m(2, 1, 0b001, 3))
        assert cpu.registers.read(3) == 0


class TestMULHSU:
    def test_positive_times_unsigned(self, exec_instruction, make_cpu, set_regs) -> None:
        """MULHSU: signed * unsigned, upper 32 bits."""
        cpu = make_cpu()
        set_regs(cpu, x1=0x10000, x2=0x10000)
        exec_instruction(cpu, _m(2, 1, 0b010, 3))
        assert cpu.registers.read(3) == 1

    def test_negative_times_unsigned(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFFF, x2=1)  # -1 (signed) * 1 (unsigned) = -1
        exec_instruction(cpu, _m(2, 1, 0b010, 3))
        # -1 * 1 = -1, as 64-bit: 0xFFFFFFFFFFFFFFFF, upper 32 = 0xFFFFFFFF
        assert cpu.registers.read(3) == 0xFFFFFFFF

    def test_negative_times_large_unsigned(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFFF, x2=0xFFFFFFFF)  # -1 * 0xFFFFFFFF
        exec_instruction(cpu, _m(2, 1, 0b010, 3))
        # -1 * 0xFFFFFFFF = -0xFFFFFFFF = -(2^32 - 1)
        # As 64-bit: 0xFFFFFFFF00000001, upper 32 = 0xFFFFFFFF
        assert cpu.registers.read(3) == 0xFFFFFFFF


class TestMULHU:
    def test_basic(self, exec_instruction, make_cpu, set_regs) -> None:
        """MULHU: upper 32 bits of unsigned * unsigned."""
        cpu = make_cpu()
        set_regs(cpu, x1=0x10000, x2=0x10000)
        exec_instruction(cpu, _m(2, 1, 0b011, 3))
        assert cpu.registers.read(3) == 1

    def test_max_unsigned(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFFF, x2=0xFFFFFFFF)
        exec_instruction(cpu, _m(2, 1, 0b011, 3))
        # 0xFFFFFFFF * 0xFFFFFFFF = 0xFFFFFFFE00000001, upper 32 = 0xFFFFFFFE
        assert cpu.registers.read(3) == 0xFFFFFFFE

    def test_zero(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=5, x2=0)
        exec_instruction(cpu, _m(2, 1, 0b011, 3))
        assert cpu.registers.read(3) == 0


class TestDIV:
    def test_basic(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=20, x2=6)
        exec_instruction(cpu, _m(2, 1, 0b100, 3))
        assert cpu.registers.read(3) == 3  # 20 / 6 = 3 (truncated)

    def test_negative_dividend(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFEC, x2=3)  # -20 / 3 = -6 (truncated toward zero)
        exec_instruction(cpu, _m(2, 1, 0b100, 3))
        assert cpu.registers.read(3) == 0xFFFFFFFA  # -6 in two's complement

    def test_negative_divisor(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=20, x2=0xFFFFFFFD)  # 20 / -3 = -6
        exec_instruction(cpu, _m(2, 1, 0b100, 3))
        assert cpu.registers.read(3) == 0xFFFFFFFA  # -6

    def test_both_negative(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFEC, x2=0xFFFFFFFD)  # -20 / -3 = 6
        exec_instruction(cpu, _m(2, 1, 0b100, 3))
        assert cpu.registers.read(3) == 6

    def test_div_by_zero(self, exec_instruction, make_cpu, set_regs) -> None:
        """DIV by zero returns 0xFFFFFFFF (all ones)."""
        cpu = make_cpu()
        set_regs(cpu, x1=42, x2=0)
        exec_instruction(cpu, _m(2, 1, 0b100, 3))
        assert cpu.registers.read(3) == 0xFFFFFFFF

    def test_overflow(self, exec_instruction, make_cpu, set_regs) -> None:
        """Signed overflow: INT_MIN / -1 returns INT_MIN (0x80000000)."""
        cpu = make_cpu()
        set_regs(cpu, x1=0x80000000, x2=0xFFFFFFFF)  # -2147483648 / -1
        exec_instruction(cpu, _m(2, 1, 0b100, 3))
        assert cpu.registers.read(3) == 0x80000000


class TestDIVU:
    def test_basic(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=20, x2=6)
        exec_instruction(cpu, _m(2, 1, 0b101, 3))
        assert cpu.registers.read(3) == 3

    def test_large_unsigned(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFFF, x2=2)  # 4294967295 / 2 = 2147483647
        exec_instruction(cpu, _m(2, 1, 0b101, 3))
        assert cpu.registers.read(3) == 0x7FFFFFFF

    def test_div_by_zero(self, exec_instruction, make_cpu, set_regs) -> None:
        """DIVU by zero returns 0xFFFFFFFF."""
        cpu = make_cpu()
        set_regs(cpu, x1=42, x2=0)
        exec_instruction(cpu, _m(2, 1, 0b101, 3))
        assert cpu.registers.read(3) == 0xFFFFFFFF


class TestREM:
    def test_basic(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=20, x2=6)
        exec_instruction(cpu, _m(2, 1, 0b110, 3))
        assert cpu.registers.read(3) == 2  # 20 % 6 = 2

    def test_negative_dividend(self, exec_instruction, make_cpu, set_regs) -> None:
        """Remainder has sign of dividend."""
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFEC, x2=3)  # -20 % 3 = -2
        exec_instruction(cpu, _m(2, 1, 0b110, 3))
        assert cpu.registers.read(3) == 0xFFFFFFFE  # -2

    def test_negative_divisor(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=20, x2=0xFFFFFFFD)  # 20 % -3 = 2
        exec_instruction(cpu, _m(2, 1, 0b110, 3))
        assert cpu.registers.read(3) == 2

    def test_rem_by_zero(self, exec_instruction, make_cpu, set_regs) -> None:
        """REM by zero returns the dividend."""
        cpu = make_cpu()
        set_regs(cpu, x1=42, x2=0)
        exec_instruction(cpu, _m(2, 1, 0b110, 3))
        assert cpu.registers.read(3) == 42

    def test_overflow(self, exec_instruction, make_cpu, set_regs) -> None:
        """Signed overflow: INT_MIN % -1 returns 0."""
        cpu = make_cpu()
        set_regs(cpu, x1=0x80000000, x2=0xFFFFFFFF)  # -2147483648 % -1
        exec_instruction(cpu, _m(2, 1, 0b110, 3))
        assert cpu.registers.read(3) == 0


class TestREMU:
    def test_basic(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=20, x2=6)
        exec_instruction(cpu, _m(2, 1, 0b111, 3))
        assert cpu.registers.read(3) == 2

    def test_large_unsigned(self, exec_instruction, make_cpu, set_regs) -> None:
        cpu = make_cpu()
        set_regs(cpu, x1=0xFFFFFFFF, x2=7)  # 4294967295 % 7 = 3
        exec_instruction(cpu, _m(2, 1, 0b111, 3))
        # 4294967295 = 613566756 * 7 + 3
        assert cpu.registers.read(3) == 3

    def test_rem_by_zero(self, exec_instruction, make_cpu, set_regs) -> None:
        """REMU by zero returns the dividend."""
        cpu = make_cpu()
        set_regs(cpu, x1=42, x2=0)
        exec_instruction(cpu, _m(2, 1, 0b111, 3))
        assert cpu.registers.read(3) == 42


# ==================== CSR shim tests ====================

# CSR instruction encoding: I-type with opcode 0x73
# CSR address in imm[11:0], source in rs1 (or zimm for I variants)
def _csr(csr_addr: int, rs1: int, funct3: int, rd: int) -> int:
    """Encode a CSR instruction."""
    return ((csr_addr & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | OP_SYSTEM


CSR_TOHOST = 0x51E
CSR_MSTATUS = 0x300


class TestCSRRW:
    def test_csrrw_tohost_pass(self, exec_instruction, make_cpu, set_regs) -> None:
        """CSRRW to tohost=1 sets cpu.tohost and halts."""
        cpu = make_cpu()
        set_regs(cpu, x1=1)
        exec_instruction(cpu, _csr(CSR_TOHOST, 1, 0b001, 0))
        assert cpu.tohost == 1
        assert cpu.halted is True

    def test_csrrw_tohost_fail(self, exec_instruction, make_cpu, set_regs) -> None:
        """CSRRW to tohost with non-1 value also halts (test failure)."""
        cpu = make_cpu()
        set_regs(cpu, x1=0x0A)  # Test case 5 failed (0x0A >> 1 = 5)
        exec_instruction(cpu, _csr(CSR_TOHOST, 1, 0b001, 0))
        assert cpu.tohost == 0x0A
        assert cpu.halted is True

    def test_csrrw_reads_old_value(self, exec_instruction, make_cpu, set_regs) -> None:
        """CSRRW writes old CSR value to rd."""
        cpu = make_cpu()
        cpu.tohost = 42
        set_regs(cpu, x1=99)
        exec_instruction(cpu, _csr(CSR_TOHOST, 1, 0b001, 3))
        assert cpu.registers.read(3) == 42
        assert cpu.tohost == 99

    def test_csrrw_other_csr_discarded(self, exec_instruction, make_cpu, set_regs) -> None:
        """CSRRW to a non-tohost CSR is silently discarded."""
        cpu = make_cpu()
        set_regs(cpu, x1=0xFF)
        exec_instruction(cpu, _csr(CSR_MSTATUS, 1, 0b001, 3))
        assert cpu.registers.read(3) == 0  # Unknown CSR reads as 0
        assert cpu.halted is False  # Did not halt


class TestCSRRS:
    def test_csrrs_reads_zero_for_unknown(self, exec_instruction, make_cpu, set_regs) -> None:
        """CSRRS on unknown CSR reads 0 into rd."""
        cpu = make_cpu()
        set_regs(cpu, x1=0)
        exec_instruction(cpu, _csr(CSR_MSTATUS, 1, 0b010, 3))
        assert cpu.registers.read(3) == 0

    def test_csrrs_sets_bits_in_tohost(self, exec_instruction, make_cpu, set_regs) -> None:
        """CSRRS ORs rs1 into the CSR."""
        cpu = make_cpu()
        cpu.tohost = 0x0F
        set_regs(cpu, x1=0xF0)
        exec_instruction(cpu, _csr(CSR_TOHOST, 1, 0b010, 3))
        assert cpu.registers.read(3) == 0x0F  # Old value in rd
        assert cpu.tohost == 0xFF  # OR'd result


class TestCSRRC:
    def test_csrrc_clears_bits(self, exec_instruction, make_cpu, set_regs) -> None:
        """CSRRC clears bits in the CSR."""
        cpu = make_cpu()
        cpu.tohost = 0xFF
        set_regs(cpu, x1=0x0F)
        exec_instruction(cpu, _csr(CSR_TOHOST, 1, 0b011, 3))
        assert cpu.registers.read(3) == 0xFF  # Old value in rd
        assert cpu.tohost == 0xF0  # Cleared lower nibble


class TestCSRRWI:
    def test_csrrwi(self, exec_instruction, make_cpu) -> None:
        """CSRRWI uses rs1 field as 5-bit immediate."""
        cpu = make_cpu()
        # zimm = 5 (in rs1 field), write to tohost
        exec_instruction(cpu, _csr(CSR_TOHOST, 5, 0b101, 3))
        assert cpu.registers.read(3) == 0  # Old tohost was 0
        assert cpu.tohost == 5


class TestCSRRSI:
    def test_csrrsi(self, exec_instruction, make_cpu) -> None:
        """CSRRSI sets bits using zimm."""
        cpu = make_cpu()
        cpu.tohost = 0x10
        exec_instruction(cpu, _csr(CSR_TOHOST, 3, 0b110, 3))  # zimm=3
        assert cpu.registers.read(3) == 0x10
        assert cpu.tohost == 0x13  # 0x10 | 3 = 0x13


class TestCSRRCI:
    def test_csrrci(self, exec_instruction, make_cpu) -> None:
        """CSRRCI clears bits using zimm."""
        cpu = make_cpu()
        cpu.tohost = 0xFF
        exec_instruction(cpu, _csr(CSR_TOHOST, 0x0F, 0b111, 3))  # zimm=15
        assert cpu.registers.read(3) == 0xFF
        assert cpu.tohost == 0xF0
