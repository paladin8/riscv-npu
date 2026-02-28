"""Tests for instruction decoder."""

from riscv_npu.cpu.decode import (
    Instruction,
    decode,
    sign_extend,
    to_signed,
    OP_R_TYPE,
    OP_I_ARITH,
    OP_LOAD,
    OP_STORE,
    OP_BRANCH,
    OP_LUI,
    OP_AUIPC,
    OP_JAL,
    OP_JALR,
    OP_SYSTEM,
    OP_FENCE,
)


# ---------- sign_extend tests ----------

class TestSignExtend:
    def test_positive_12bit(self) -> None:
        """12-bit value 0x7FF (max positive) stays positive."""
        assert sign_extend(0x7FF, 12) == 0x7FF

    def test_negative_12bit(self) -> None:
        """12-bit value 0x800 (MSB set) sign-extends to 0xFFFFF800."""
        assert sign_extend(0x800, 12) == 0xFFFFF800

    def test_negative_12bit_all_ones(self) -> None:
        """12-bit value 0xFFF (-1) sign-extends to 0xFFFFFFFF."""
        assert sign_extend(0xFFF, 12) == 0xFFFFFFFF

    def test_positive_13bit(self) -> None:
        """B-type: 13-bit value 0x100 stays positive."""
        assert sign_extend(0x100, 13) == 0x100

    def test_negative_13bit(self) -> None:
        """B-type: 13-bit value 0x1000 (MSB set) sign-extends."""
        assert sign_extend(0x1000, 13) == 0xFFFFF000

    def test_positive_21bit(self) -> None:
        """J-type: 21-bit value 0x0FFFFF stays positive."""
        assert sign_extend(0x0FFFFF, 21) == 0x0FFFFF

    def test_negative_21bit(self) -> None:
        """J-type: 21-bit value 0x100000 (MSB set) sign-extends."""
        assert sign_extend(0x100000, 21) == 0xFFF00000

    def test_zero(self) -> None:
        assert sign_extend(0, 12) == 0


# ---------- to_signed tests ----------

class TestToSigned:
    def test_positive(self) -> None:
        assert to_signed(42) == 42

    def test_zero(self) -> None:
        assert to_signed(0) == 0

    def test_max_positive(self) -> None:
        assert to_signed(0x7FFFFFFF) == 2147483647

    def test_min_negative(self) -> None:
        assert to_signed(0x80000000) == -2147483648

    def test_minus_one(self) -> None:
        assert to_signed(0xFFFFFFFF) == -1


# ---------- Decode R-type tests ----------

def _encode_r(funct7: int, rs2: int, rs1: int, funct3: int, rd: int) -> int:
    """Encode an R-type instruction word."""
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | OP_R_TYPE


class TestDecodeRType:
    def test_add(self) -> None:
        """ADD x1, x2, x3 → funct7=0, funct3=0"""
        word = _encode_r(0b0000000, 3, 2, 0b000, 1)
        inst = decode(word)
        assert inst.opcode == OP_R_TYPE
        assert inst.rd == 1
        assert inst.rs1 == 2
        assert inst.rs2 == 3
        assert inst.funct3 == 0
        assert inst.funct7 == 0

    def test_sub(self) -> None:
        """SUB x5, x6, x7 → funct7=0x20, funct3=0"""
        word = _encode_r(0b0100000, 7, 6, 0b000, 5)
        inst = decode(word)
        assert inst.funct7 == 0b0100000
        assert inst.rd == 5
        assert inst.rs1 == 6
        assert inst.rs2 == 7

    def test_sra(self) -> None:
        """SRA x10, x11, x12 → funct7=0x20, funct3=0x5"""
        word = _encode_r(0b0100000, 12, 11, 0b101, 10)
        inst = decode(word)
        assert inst.funct7 == 0b0100000
        assert inst.funct3 == 0b101


# ---------- Decode I-type tests ----------

def _encode_i(imm12: int, rs1: int, funct3: int, rd: int, opcode: int = OP_I_ARITH) -> int:
    """Encode an I-type instruction word."""
    return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


class TestDecodeIType:
    def test_addi_positive(self) -> None:
        """ADDI x1, x2, 100"""
        word = _encode_i(100, 2, 0b000, 1)
        inst = decode(word)
        assert inst.opcode == OP_I_ARITH
        assert inst.rd == 1
        assert inst.rs1 == 2
        assert inst.imm == 100
        assert inst.funct3 == 0

    def test_addi_negative(self) -> None:
        """ADDI x1, x2, -1 (imm=0xFFF)"""
        word = _encode_i(0xFFF, 2, 0b000, 1)
        inst = decode(word)
        assert inst.imm == 0xFFFFFFFF  # -1 in 32-bit unsigned

    def test_addi_zero(self) -> None:
        """NOP = ADDI x0, x0, 0"""
        word = _encode_i(0, 0, 0b000, 0)
        inst = decode(word)
        assert inst.imm == 0
        assert inst.rd == 0
        assert inst.rs1 == 0

    def test_load_lw(self) -> None:
        """LW x5, 8(x10) — load opcode"""
        word = _encode_i(8, 10, 0b010, 5, OP_LOAD)
        inst = decode(word)
        assert inst.opcode == OP_LOAD
        assert inst.imm == 8
        assert inst.rs1 == 10
        assert inst.rd == 5

    def test_slli(self) -> None:
        """SLLI x1, x2, 5 — funct7=0, shamt=5"""
        word = _encode_i(5, 2, 0b001, 1)
        inst = decode(word)
        assert inst.imm == 5
        assert inst.funct3 == 0b001
        assert inst.funct7 == 0

    def test_srai(self) -> None:
        """SRAI x1, x2, 5 — funct7=0x20, shamt=5"""
        word = _encode_i(0x400 | 5, 2, 0b101, 1)  # bit 10 set = funct7 bit 5
        inst = decode(word)
        assert inst.funct3 == 0b101
        assert inst.funct7 == 0b0100000
        # imm is sign_extend of full 12-bit field, but execute uses lower 5 bits as shamt
        assert inst.imm & 0x1F == 5


# ---------- Decode S-type tests ----------

def _encode_s(imm12: int, rs2: int, rs1: int, funct3: int) -> int:
    """Encode an S-type instruction word."""
    imm_11_5 = (imm12 >> 5) & 0x7F
    imm_4_0 = imm12 & 0x1F
    return (imm_11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_4_0 << 7) | OP_STORE


class TestDecodeSType:
    def test_sw_positive(self) -> None:
        """SW x5, 16(x10) — offset 16"""
        word = _encode_s(16, 5, 10, 0b010)
        inst = decode(word)
        assert inst.opcode == OP_STORE
        assert inst.rs1 == 10
        assert inst.rs2 == 5
        assert inst.imm == 16

    def test_sw_negative(self) -> None:
        """SW x5, -4(x10)"""
        word = _encode_s((-4) & 0xFFF, 5, 10, 0b010)
        inst = decode(word)
        assert inst.imm == 0xFFFFFFFC  # -4 as 32-bit unsigned


# ---------- Decode B-type tests ----------

def _encode_b(imm13: int, rs2: int, rs1: int, funct3: int) -> int:
    """Encode a B-type instruction word. imm13 has bit 0 always 0."""
    imm = imm13 & 0x1FFE  # mask to 13 bits, bit 0 unused
    if imm13 < 0:
        imm = imm13 & 0x1FFF
    bit_12 = (imm >> 12) & 1
    bit_11 = (imm >> 11) & 1
    bits_10_5 = (imm >> 5) & 0x3F
    bits_4_1 = (imm >> 1) & 0xF
    return (bit_12 << 31) | (bits_10_5 << 25) | (rs2 << 20) | (rs1 << 15) | \
           (funct3 << 12) | (bits_4_1 << 8) | (bit_11 << 7) | OP_BRANCH


class TestDecodeBType:
    def test_positive_offset(self) -> None:
        """BEQ x1, x2, +8"""
        word = _encode_b(8, 2, 1, 0b000)
        inst = decode(word)
        assert inst.opcode == OP_BRANCH
        assert inst.rs1 == 1
        assert inst.rs2 == 2
        assert inst.imm == 8

    def test_negative_offset(self) -> None:
        """BEQ x1, x2, -16"""
        # -16 in 13-bit signed: 0x1FF0
        neg16 = (-16) & 0x1FFF
        word = _encode_b(neg16, 2, 1, 0b000)
        inst = decode(word)
        assert to_signed(inst.imm) == -16

    def test_large_positive(self) -> None:
        """BNE with offset +4094 (max positive B-type)"""
        word = _encode_b(4094, 0, 0, 0b001)
        inst = decode(word)
        assert inst.imm == 4094

    def test_large_negative(self) -> None:
        """BLT with offset -4096 (most negative B-type)"""
        neg4096 = (-4096) & 0x1FFF
        word = _encode_b(neg4096, 0, 0, 0b100)
        inst = decode(word)
        assert to_signed(inst.imm) == -4096


# ---------- Decode U-type tests ----------

class TestDecodeUType:
    def test_lui(self) -> None:
        """LUI x1, 0xDEADB (upper 20 bits)"""
        word = (0xDEADB << 12) | (1 << 7) | OP_LUI
        inst = decode(word)
        assert inst.opcode == OP_LUI
        assert inst.rd == 1
        assert inst.imm == 0xDEADB000

    def test_auipc(self) -> None:
        """AUIPC x2, 0x12345"""
        word = (0x12345 << 12) | (2 << 7) | OP_AUIPC
        inst = decode(word)
        assert inst.opcode == OP_AUIPC
        assert inst.rd == 2
        assert inst.imm == 0x12345000


# ---------- Decode J-type tests ----------

def _encode_j(imm21: int, rd: int) -> int:
    """Encode a J-type instruction word. imm21 has bit 0 always 0."""
    imm = imm21 & 0x1FFFFF
    bit_20 = (imm >> 20) & 1
    bits_10_1 = (imm >> 1) & 0x3FF
    bit_11 = (imm >> 11) & 1
    bits_19_12 = (imm >> 12) & 0xFF
    return (bit_20 << 31) | (bits_10_1 << 21) | (bit_11 << 20) | \
           (bits_19_12 << 12) | (rd << 7) | OP_JAL


class TestDecodeJType:
    def test_positive_offset(self) -> None:
        """JAL x1, +100"""
        word = _encode_j(100, 1)
        inst = decode(word)
        assert inst.opcode == OP_JAL
        assert inst.rd == 1
        assert inst.imm == 100

    def test_negative_offset(self) -> None:
        """JAL x1, -20"""
        neg20 = (-20) & 0x1FFFFF
        word = _encode_j(neg20, 1)
        inst = decode(word)
        assert to_signed(inst.imm) == -20

    def test_large_positive(self) -> None:
        """JAL x0, +1048574 (near max)"""
        word = _encode_j(1048574, 0)
        inst = decode(word)
        assert inst.imm == 1048574

    def test_large_negative(self) -> None:
        """JAL x0, -1048576 (most negative J-type)"""
        neg = (-1048576) & 0x1FFFFF
        word = _encode_j(neg, 0)
        inst = decode(word)
        assert to_signed(inst.imm) == -1048576


# ---------- System/Fence ----------

class TestDecodeSystem:
    def test_ecall(self) -> None:
        word = _encode_i(0, 0, 0b000, 0, OP_SYSTEM)
        inst = decode(word)
        assert inst.opcode == OP_SYSTEM
        assert inst.imm == 0

    def test_ebreak(self) -> None:
        word = _encode_i(1, 0, 0b000, 0, OP_SYSTEM)
        inst = decode(word)
        assert inst.opcode == OP_SYSTEM
        assert inst.imm == 1

    def test_fence(self) -> None:
        word = _encode_i(0, 0, 0b000, 0, OP_FENCE)
        inst = decode(word)
        assert inst.opcode == OP_FENCE
