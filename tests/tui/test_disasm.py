"""Tests for the TUI disassembly module."""

import pytest

from riscv_npu.cpu.decode import Instruction, decode
from riscv_npu.memory.bus import MemoryBus
from riscv_npu.memory.ram import RAM
from riscv_npu.tui.disasm import (
    DisassemblyLine,
    disassemble_instruction,
    disassemble_region,
)

# --- Helper: encode instruction words ---
# These helpers build 32-bit instruction words from fields.


def _r_type(rd: int, rs1: int, rs2: int, funct3: int, funct7: int) -> int:
    """Encode an R-type instruction word (opcode=0x33)."""
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0x33


def _i_type(rd: int, rs1: int, imm12: int, funct3: int, opcode: int = 0x13) -> int:
    """Encode an I-type instruction word."""
    return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def _s_type(rs1: int, rs2: int, imm12: int, funct3: int) -> int:
    """Encode an S-type (store) instruction word (opcode=0x23)."""
    imm_hi = (imm12 >> 5) & 0x7F
    imm_lo = imm12 & 0x1F
    return (imm_hi << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_lo << 7) | 0x23


def _b_type(rs1: int, rs2: int, imm13: int, funct3: int) -> int:
    """Encode a B-type (branch) instruction word (opcode=0x63)."""
    # imm[12|10:5|4:1|11]
    bit12 = (imm13 >> 12) & 1
    bit11 = (imm13 >> 11) & 1
    bits_10_5 = (imm13 >> 5) & 0x3F
    bits_4_1 = (imm13 >> 1) & 0xF
    return (
        (bit12 << 31) | (bits_10_5 << 25) | (rs2 << 20) | (rs1 << 15)
        | (funct3 << 12) | (bits_4_1 << 8) | (bit11 << 7) | 0x63
    )


def _u_type(rd: int, imm20: int, opcode: int) -> int:
    """Encode a U-type instruction word."""
    return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | opcode


def _jal(rd: int, imm21: int) -> int:
    """Encode a JAL instruction word (opcode=0x6F)."""
    bit20 = (imm21 >> 20) & 1
    bits_10_1 = (imm21 >> 1) & 0x3FF
    bit11 = (imm21 >> 11) & 1
    bits_19_12 = (imm21 >> 12) & 0xFF
    return (
        (bit20 << 31) | (bits_10_1 << 21) | (bit11 << 20) | (bits_19_12 << 12)
        | (rd << 7) | 0x6F
    )


# --- R-type tests ---


class TestRTypeDisassembly:
    """Tests for R-type instruction disassembly."""

    def test_add(self) -> None:
        word = _r_type(rd=1, rs1=2, rs2=3, funct3=0b000, funct7=0b0000000)
        assert disassemble_instruction(decode(word)) == "ADD x1, x2, x3"

    def test_sub(self) -> None:
        word = _r_type(rd=5, rs1=6, rs2=7, funct3=0b000, funct7=0b0100000)
        assert disassemble_instruction(decode(word)) == "SUB x5, x6, x7"

    def test_sll(self) -> None:
        word = _r_type(rd=1, rs1=2, rs2=3, funct3=0b001, funct7=0b0000000)
        assert disassemble_instruction(decode(word)) == "SLL x1, x2, x3"

    def test_slt(self) -> None:
        word = _r_type(rd=1, rs1=2, rs2=3, funct3=0b010, funct7=0b0000000)
        assert disassemble_instruction(decode(word)) == "SLT x1, x2, x3"

    def test_sltu(self) -> None:
        word = _r_type(rd=1, rs1=2, rs2=3, funct3=0b011, funct7=0b0000000)
        assert disassemble_instruction(decode(word)) == "SLTU x1, x2, x3"

    def test_xor(self) -> None:
        word = _r_type(rd=1, rs1=2, rs2=3, funct3=0b100, funct7=0b0000000)
        assert disassemble_instruction(decode(word)) == "XOR x1, x2, x3"

    def test_srl(self) -> None:
        word = _r_type(rd=1, rs1=2, rs2=3, funct3=0b101, funct7=0b0000000)
        assert disassemble_instruction(decode(word)) == "SRL x1, x2, x3"

    def test_sra(self) -> None:
        word = _r_type(rd=1, rs1=2, rs2=3, funct3=0b101, funct7=0b0100000)
        assert disassemble_instruction(decode(word)) == "SRA x1, x2, x3"

    def test_or(self) -> None:
        word = _r_type(rd=1, rs1=2, rs2=3, funct3=0b110, funct7=0b0000000)
        assert disassemble_instruction(decode(word)) == "OR x1, x2, x3"

    def test_and(self) -> None:
        word = _r_type(rd=1, rs1=2, rs2=3, funct3=0b111, funct7=0b0000000)
        assert disassemble_instruction(decode(word)) == "AND x1, x2, x3"


# --- M-extension tests ---


class TestMExtDisassembly:
    """Tests for M-extension instruction disassembly."""

    def test_mul(self) -> None:
        word = _r_type(rd=1, rs1=2, rs2=3, funct3=0b000, funct7=0b0000001)
        assert disassemble_instruction(decode(word)) == "MUL x1, x2, x3"

    def test_mulh(self) -> None:
        word = _r_type(rd=1, rs1=2, rs2=3, funct3=0b001, funct7=0b0000001)
        assert disassemble_instruction(decode(word)) == "MULH x1, x2, x3"

    def test_mulhsu(self) -> None:
        word = _r_type(rd=1, rs1=2, rs2=3, funct3=0b010, funct7=0b0000001)
        assert disassemble_instruction(decode(word)) == "MULHSU x1, x2, x3"

    def test_mulhu(self) -> None:
        word = _r_type(rd=1, rs1=2, rs2=3, funct3=0b011, funct7=0b0000001)
        assert disassemble_instruction(decode(word)) == "MULHU x1, x2, x3"

    def test_div(self) -> None:
        word = _r_type(rd=1, rs1=2, rs2=3, funct3=0b100, funct7=0b0000001)
        assert disassemble_instruction(decode(word)) == "DIV x1, x2, x3"

    def test_divu(self) -> None:
        word = _r_type(rd=1, rs1=2, rs2=3, funct3=0b101, funct7=0b0000001)
        assert disassemble_instruction(decode(word)) == "DIVU x1, x2, x3"

    def test_rem(self) -> None:
        word = _r_type(rd=1, rs1=2, rs2=3, funct3=0b110, funct7=0b0000001)
        assert disassemble_instruction(decode(word)) == "REM x1, x2, x3"

    def test_remu(self) -> None:
        word = _r_type(rd=1, rs1=2, rs2=3, funct3=0b111, funct7=0b0000001)
        assert disassemble_instruction(decode(word)) == "REMU x1, x2, x3"


# --- I-type arithmetic tests ---


class TestIArithDisassembly:
    """Tests for I-type arithmetic instruction disassembly."""

    def test_addi_positive(self) -> None:
        word = _i_type(rd=5, rs1=0, imm12=42, funct3=0b000)
        assert disassemble_instruction(decode(word)) == "ADDI x5, x0, 42"

    def test_addi_negative(self) -> None:
        # -1 in 12-bit two's complement = 0xFFF
        word = _i_type(rd=5, rs1=1, imm12=0xFFF, funct3=0b000)
        assert disassemble_instruction(decode(word)) == "ADDI x5, x1, -1"

    def test_slti(self) -> None:
        word = _i_type(rd=1, rs1=2, imm12=10, funct3=0b010)
        assert disassemble_instruction(decode(word)) == "SLTI x1, x2, 10"

    def test_sltiu(self) -> None:
        word = _i_type(rd=1, rs1=2, imm12=10, funct3=0b011)
        assert disassemble_instruction(decode(word)) == "SLTIU x1, x2, 10"

    def test_xori(self) -> None:
        word = _i_type(rd=1, rs1=2, imm12=0xFF, funct3=0b100)
        assert disassemble_instruction(decode(word)) == "XORI x1, x2, 255"

    def test_ori(self) -> None:
        word = _i_type(rd=1, rs1=2, imm12=0xFF, funct3=0b110)
        assert disassemble_instruction(decode(word)) == "ORI x1, x2, 255"

    def test_andi(self) -> None:
        word = _i_type(rd=1, rs1=2, imm12=0xFF, funct3=0b111)
        assert disassemble_instruction(decode(word)) == "ANDI x1, x2, 255"

    def test_slli(self) -> None:
        word = _i_type(rd=1, rs1=2, imm12=5, funct3=0b001)
        assert disassemble_instruction(decode(word)) == "SLLI x1, x2, 5"

    def test_srli(self) -> None:
        word = _i_type(rd=1, rs1=2, imm12=5, funct3=0b101)
        assert disassemble_instruction(decode(word)) == "SRLI x1, x2, 5"

    def test_srai(self) -> None:
        # SRAI: imm[11:5] = 0b0100000, imm[4:0] = shamt
        word = _i_type(rd=1, rs1=2, imm12=(0b0100000 << 5) | 5, funct3=0b101)
        assert disassemble_instruction(decode(word)) == "SRAI x1, x2, 5"


# --- Load tests ---


class TestLoadDisassembly:
    """Tests for load instruction disassembly."""

    def test_lb(self) -> None:
        word = _i_type(rd=1, rs1=2, imm12=4, funct3=0b000, opcode=0x03)
        assert disassemble_instruction(decode(word)) == "LB x1, 4(x2)"

    def test_lh(self) -> None:
        word = _i_type(rd=1, rs1=2, imm12=8, funct3=0b001, opcode=0x03)
        assert disassemble_instruction(decode(word)) == "LH x1, 8(x2)"

    def test_lw(self) -> None:
        word = _i_type(rd=1, rs1=2, imm12=12, funct3=0b010, opcode=0x03)
        assert disassemble_instruction(decode(word)) == "LW x1, 12(x2)"

    def test_lbu(self) -> None:
        word = _i_type(rd=1, rs1=2, imm12=0, funct3=0b100, opcode=0x03)
        assert disassemble_instruction(decode(word)) == "LBU x1, 0(x2)"

    def test_lhu(self) -> None:
        word = _i_type(rd=1, rs1=2, imm12=0, funct3=0b101, opcode=0x03)
        assert disassemble_instruction(decode(word)) == "LHU x1, 0(x2)"

    def test_lw_negative_offset(self) -> None:
        # -4 in 12-bit two's complement = 0xFFC
        word = _i_type(rd=1, rs1=2, imm12=0xFFC, funct3=0b010, opcode=0x03)
        assert disassemble_instruction(decode(word)) == "LW x1, -4(x2)"


# --- Store tests ---


class TestStoreDisassembly:
    """Tests for store instruction disassembly."""

    def test_sb(self) -> None:
        word = _s_type(rs1=2, rs2=3, imm12=4, funct3=0b000)
        assert disassemble_instruction(decode(word)) == "SB x3, 4(x2)"

    def test_sh(self) -> None:
        word = _s_type(rs1=2, rs2=3, imm12=8, funct3=0b001)
        assert disassemble_instruction(decode(word)) == "SH x3, 8(x2)"

    def test_sw(self) -> None:
        word = _s_type(rs1=2, rs2=3, imm12=12, funct3=0b010)
        assert disassemble_instruction(decode(word)) == "SW x3, 12(x2)"

    def test_sw_negative_offset(self) -> None:
        # -4 in 12-bit two's complement = 0xFFC
        word = _s_type(rs1=2, rs2=3, imm12=0xFFC, funct3=0b010)
        assert disassemble_instruction(decode(word)) == "SW x3, -4(x2)"


# --- Branch tests ---


class TestBranchDisassembly:
    """Tests for branch instruction disassembly."""

    def test_beq(self) -> None:
        word = _b_type(rs1=1, rs2=2, imm13=8, funct3=0b000)
        assert disassemble_instruction(decode(word)) == "BEQ x1, x2, 8"

    def test_bne(self) -> None:
        word = _b_type(rs1=1, rs2=2, imm13=12, funct3=0b001)
        assert disassemble_instruction(decode(word)) == "BNE x1, x2, 12"

    def test_blt(self) -> None:
        word = _b_type(rs1=1, rs2=2, imm13=16, funct3=0b100)
        assert disassemble_instruction(decode(word)) == "BLT x1, x2, 16"

    def test_bge(self) -> None:
        word = _b_type(rs1=1, rs2=2, imm13=20, funct3=0b101)
        assert disassemble_instruction(decode(word)) == "BGE x1, x2, 20"

    def test_bltu(self) -> None:
        word = _b_type(rs1=1, rs2=2, imm13=24, funct3=0b110)
        assert disassemble_instruction(decode(word)) == "BLTU x1, x2, 24"

    def test_bgeu(self) -> None:
        word = _b_type(rs1=1, rs2=2, imm13=28, funct3=0b111)
        assert disassemble_instruction(decode(word)) == "BGEU x1, x2, 28"

    def test_beq_negative_offset(self) -> None:
        # -8 in 13-bit two's complement: 0x1FF8
        word = _b_type(rs1=1, rs2=2, imm13=0x1FF8, funct3=0b000)
        assert disassemble_instruction(decode(word)) == "BEQ x1, x2, -8"


# --- U-type tests ---


class TestUTypeDisassembly:
    """Tests for U-type instruction disassembly."""

    def test_lui(self) -> None:
        word = _u_type(rd=1, imm20=0x12345, opcode=0x37)
        assert disassemble_instruction(decode(word)) == "LUI x1, 0x12345"

    def test_auipc(self) -> None:
        word = _u_type(rd=1, imm20=0xABCDE, opcode=0x17)
        assert disassemble_instruction(decode(word)) == "AUIPC x1, 0xABCDE"


# --- J-type tests ---


class TestJTypeDisassembly:
    """Tests for J-type instruction disassembly."""

    def test_jal_positive(self) -> None:
        word = _jal(rd=1, imm21=100)
        assert disassemble_instruction(decode(word)) == "JAL x1, 100"

    def test_jal_negative(self) -> None:
        # -4 in 21-bit two's complement: 0x1FFFFC
        word = _jal(rd=0, imm21=0x1FFFFC)
        assert disassemble_instruction(decode(word)) == "JAL x0, -4"

    def test_jalr(self) -> None:
        word = _i_type(rd=1, rs1=2, imm12=0, funct3=0b000, opcode=0x67)
        assert disassemble_instruction(decode(word)) == "JALR x1, x2, 0"


# --- System instruction tests ---


class TestSystemDisassembly:
    """Tests for SYSTEM instruction disassembly."""

    def test_ecall(self) -> None:
        word = 0x00000073  # ECALL: opcode=0x73, all other fields zero
        assert disassemble_instruction(decode(word)) == "ECALL"

    def test_ebreak(self) -> None:
        word = 0x00100073  # EBREAK: imm=1
        assert disassemble_instruction(decode(word)) == "EBREAK"

    def test_mret(self) -> None:
        word = 0x30200073  # MRET: imm=0x302
        assert disassemble_instruction(decode(word)) == "MRET"

    def test_csrrw(self) -> None:
        # CSRRW x1, 0x300, x2: funct3=001, csr=0x300
        word = (0x300 << 20) | (2 << 15) | (0b001 << 12) | (1 << 7) | 0x73
        assert disassemble_instruction(decode(word)) == "CSRRW x1, 0x300, x2"

    def test_csrrs(self) -> None:
        word = (0x300 << 20) | (2 << 15) | (0b010 << 12) | (1 << 7) | 0x73
        assert disassemble_instruction(decode(word)) == "CSRRS x1, 0x300, x2"

    def test_csrrc(self) -> None:
        word = (0x300 << 20) | (2 << 15) | (0b011 << 12) | (1 << 7) | 0x73
        assert disassemble_instruction(decode(word)) == "CSRRC x1, 0x300, x2"

    def test_csrrwi(self) -> None:
        # CSRRWI x1, 0x300, 3: funct3=101, zimm=3 (in rs1 field)
        word = (0x300 << 20) | (3 << 15) | (0b101 << 12) | (1 << 7) | 0x73
        assert disassemble_instruction(decode(word)) == "CSRRWI x1, 0x300, 3"

    def test_csrrsi(self) -> None:
        word = (0x300 << 20) | (3 << 15) | (0b110 << 12) | (1 << 7) | 0x73
        assert disassemble_instruction(decode(word)) == "CSRRSI x1, 0x300, 3"

    def test_csrrci(self) -> None:
        word = (0x300 << 20) | (3 << 15) | (0b111 << 12) | (1 << 7) | 0x73
        assert disassemble_instruction(decode(word)) == "CSRRCI x1, 0x300, 3"


# --- FENCE test ---


class TestFenceDisassembly:
    """Tests for FENCE instruction disassembly."""

    def test_fence(self) -> None:
        word = 0x0000000F  # FENCE: opcode=0x0F
        assert disassemble_instruction(decode(word)) == "FENCE"


# --- disassemble_region tests ---


class TestDisassembleRegion:
    """Tests for disassemble_region with memory."""

    def _make_memory(self) -> tuple[MemoryBus, RAM]:
        """Create a MemoryBus with 1 MB of RAM at 0x80000000."""
        bus = MemoryBus()
        ram = RAM(0x80000000, 1024 * 1024)
        bus.register(0x80000000, 1024 * 1024, ram)
        return bus, ram

    def test_single_instruction(self) -> None:
        bus, ram = self._make_memory()
        # ADDI x1, x0, 42
        word = _i_type(rd=1, rs1=0, imm12=42, funct3=0b000)
        bus.write32(0x80000000, word)
        lines = disassemble_region(bus, 0x80000000, 1)
        assert len(lines) == 1
        assert lines[0].addr == 0x80000000
        assert lines[0].is_current is True
        assert "ADDI" in lines[0].text

    def test_multiple_instructions_centered(self) -> None:
        bus, ram = self._make_memory()
        # Write 5 NOP-like instructions (ADDI x0, x0, 0)
        nop = _i_type(rd=0, rs1=0, imm12=0, funct3=0b000)
        for i in range(5):
            bus.write32(0x80000000 + i * 4, nop)
        lines = disassemble_region(bus, 0x80000008, 5)
        assert len(lines) == 5
        # The center instruction should be marked as current
        current_lines = [l for l in lines if l.is_current]
        assert len(current_lines) == 1
        assert current_lines[0].addr == 0x80000008

    def test_unmapped_memory_shows_question_marks(self) -> None:
        bus, ram = self._make_memory()
        # Address 0x00000000 is unmapped, but center on a mapped address
        # near the start of RAM so some instructions before it are unmapped
        nop = _i_type(rd=0, rs1=0, imm12=0, funct3=0b000)
        bus.write32(0x80000000, nop)
        lines = disassemble_region(bus, 0x80000000, 5)
        # Instructions before 0x80000000 are unmapped
        unmapped = [l for l in lines if l.text == "???"]
        assert len(unmapped) >= 1

    def test_disassembly_line_has_correct_word(self) -> None:
        bus, ram = self._make_memory()
        # ADD x1, x2, x3
        word = _r_type(rd=1, rs1=2, rs2=3, funct3=0b000, funct7=0b0000000)
        bus.write32(0x80000000, word)
        lines = disassemble_region(bus, 0x80000000, 1)
        assert lines[0].word == word
