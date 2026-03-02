"""Tests for instruction decoder."""

from riscv_npu.cpu.decode import (
    Instruction,
    decode,
    instruction_mnemonic,
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
    OP_NPU,
    OP_FP_NPU,
    OP_LOAD_FP,
    OP_STORE_FP,
    OP_FMADD,
    OP_FMSUB,
    OP_FNMSUB,
    OP_FNMADD,
    OP_OP_FP,
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


# ---------- instruction_mnemonic tests ----------

def _mnemonic_r(funct7: int, rs2: int, rs1: int, funct3: int, rd: int) -> str:
    """Encode R-type and return its mnemonic."""
    word = (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | OP_R_TYPE
    return instruction_mnemonic(decode(word))


def _mnemonic_i(imm12: int, rs1: int, funct3: int, rd: int, opcode: int = OP_I_ARITH) -> str:
    """Encode I-type and return its mnemonic."""
    word = ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode
    return instruction_mnemonic(decode(word))


class TestInstructionMnemonicRType:
    """Test instruction_mnemonic for R-type instructions."""

    def test_add(self) -> None:
        assert _mnemonic_r(0b0000000, 3, 2, 0b000, 1) == "ADD"

    def test_sub(self) -> None:
        assert _mnemonic_r(0b0100000, 3, 2, 0b000, 1) == "SUB"

    def test_sll(self) -> None:
        assert _mnemonic_r(0b0000000, 3, 2, 0b001, 1) == "SLL"

    def test_slt(self) -> None:
        assert _mnemonic_r(0b0000000, 3, 2, 0b010, 1) == "SLT"

    def test_sltu(self) -> None:
        assert _mnemonic_r(0b0000000, 3, 2, 0b011, 1) == "SLTU"

    def test_xor(self) -> None:
        assert _mnemonic_r(0b0000000, 3, 2, 0b100, 1) == "XOR"

    def test_srl(self) -> None:
        assert _mnemonic_r(0b0000000, 3, 2, 0b101, 1) == "SRL"

    def test_sra(self) -> None:
        assert _mnemonic_r(0b0100000, 3, 2, 0b101, 1) == "SRA"

    def test_or(self) -> None:
        assert _mnemonic_r(0b0000000, 3, 2, 0b110, 1) == "OR"

    def test_and(self) -> None:
        assert _mnemonic_r(0b0000000, 3, 2, 0b111, 1) == "AND"


class TestInstructionMnemonicIType:
    """Test instruction_mnemonic for I-type arithmetic instructions."""

    def test_addi(self) -> None:
        assert _mnemonic_i(42, 2, 0b000, 1) == "ADDI"

    def test_slti(self) -> None:
        assert _mnemonic_i(42, 2, 0b010, 1) == "SLTI"

    def test_sltiu(self) -> None:
        assert _mnemonic_i(42, 2, 0b011, 1) == "SLTIU"

    def test_xori(self) -> None:
        assert _mnemonic_i(42, 2, 0b100, 1) == "XORI"

    def test_ori(self) -> None:
        assert _mnemonic_i(42, 2, 0b110, 1) == "ORI"

    def test_andi(self) -> None:
        assert _mnemonic_i(42, 2, 0b111, 1) == "ANDI"

    def test_slli(self) -> None:
        assert _mnemonic_i(5, 2, 0b001, 1) == "SLLI"

    def test_srli(self) -> None:
        assert _mnemonic_i(5, 2, 0b101, 1) == "SRLI"

    def test_srai(self) -> None:
        assert _mnemonic_i(0x400 | 5, 2, 0b101, 1) == "SRAI"


class TestInstructionMnemonicLoadStore:
    """Test instruction_mnemonic for load/store instructions."""

    def test_lb(self) -> None:
        assert _mnemonic_i(0, 2, 0b000, 1, OP_LOAD) == "LB"

    def test_lh(self) -> None:
        assert _mnemonic_i(0, 2, 0b001, 1, OP_LOAD) == "LH"

    def test_lw(self) -> None:
        assert _mnemonic_i(0, 2, 0b010, 1, OP_LOAD) == "LW"

    def test_lbu(self) -> None:
        assert _mnemonic_i(0, 2, 0b100, 1, OP_LOAD) == "LBU"

    def test_lhu(self) -> None:
        assert _mnemonic_i(0, 2, 0b101, 1, OP_LOAD) == "LHU"

    def test_sb(self) -> None:
        word = (5 << 20) | (2 << 15) | (0b000 << 12) | (0 << 7) | OP_STORE
        assert instruction_mnemonic(decode(word)) == "SB"

    def test_sh(self) -> None:
        word = (5 << 20) | (2 << 15) | (0b001 << 12) | (0 << 7) | OP_STORE
        assert instruction_mnemonic(decode(word)) == "SH"

    def test_sw(self) -> None:
        word = (5 << 20) | (2 << 15) | (0b010 << 12) | (0 << 7) | OP_STORE
        assert instruction_mnemonic(decode(word)) == "SW"


class TestInstructionMnemonicBranch:
    """Test instruction_mnemonic for branch instructions."""

    def test_beq(self) -> None:
        word = _encode_b(8, 2, 1, 0b000)
        assert instruction_mnemonic(decode(word)) == "BEQ"

    def test_bne(self) -> None:
        word = _encode_b(8, 2, 1, 0b001)
        assert instruction_mnemonic(decode(word)) == "BNE"

    def test_blt(self) -> None:
        word = _encode_b(8, 2, 1, 0b100)
        assert instruction_mnemonic(decode(word)) == "BLT"

    def test_bge(self) -> None:
        word = _encode_b(8, 2, 1, 0b101)
        assert instruction_mnemonic(decode(word)) == "BGE"

    def test_bltu(self) -> None:
        word = _encode_b(8, 2, 1, 0b110)
        assert instruction_mnemonic(decode(word)) == "BLTU"

    def test_bgeu(self) -> None:
        word = _encode_b(8, 2, 1, 0b111)
        assert instruction_mnemonic(decode(word)) == "BGEU"


class TestInstructionMnemonicUpperJump:
    """Test instruction_mnemonic for upper-immediate and jump instructions."""

    def test_lui(self) -> None:
        word = (0xDEADB << 12) | (1 << 7) | OP_LUI
        assert instruction_mnemonic(decode(word)) == "LUI"

    def test_auipc(self) -> None:
        word = (0x12345 << 12) | (2 << 7) | OP_AUIPC
        assert instruction_mnemonic(decode(word)) == "AUIPC"

    def test_jal(self) -> None:
        word = _encode_j(100, 1)
        assert instruction_mnemonic(decode(word)) == "JAL"

    def test_jalr(self) -> None:
        word = _encode_i(0, 1, 0b000, 2, OP_JALR)
        assert instruction_mnemonic(decode(word)) == "JALR"


class TestInstructionMnemonicSystem:
    """Test instruction_mnemonic for system instructions."""

    def test_ecall(self) -> None:
        word = _encode_i(0, 0, 0b000, 0, OP_SYSTEM)
        assert instruction_mnemonic(decode(word)) == "ECALL"

    def test_ebreak(self) -> None:
        word = _encode_i(1, 0, 0b000, 0, OP_SYSTEM)
        assert instruction_mnemonic(decode(word)) == "EBREAK"

    def test_mret(self) -> None:
        word = _encode_i(0x302, 0, 0b000, 0, OP_SYSTEM)
        assert instruction_mnemonic(decode(word)) == "MRET"

    def test_csrrw(self) -> None:
        word = _encode_i(0x300, 1, 0b001, 5, OP_SYSTEM)
        assert instruction_mnemonic(decode(word)) == "CSRRW"

    def test_csrrs(self) -> None:
        word = _encode_i(0x300, 1, 0b010, 5, OP_SYSTEM)
        assert instruction_mnemonic(decode(word)) == "CSRRS"

    def test_csrrc(self) -> None:
        word = _encode_i(0x300, 1, 0b011, 5, OP_SYSTEM)
        assert instruction_mnemonic(decode(word)) == "CSRRC"

    def test_csrrwi(self) -> None:
        word = _encode_i(0x300, 1, 0b101, 5, OP_SYSTEM)
        assert instruction_mnemonic(decode(word)) == "CSRRWI"

    def test_fence(self) -> None:
        word = _encode_i(0, 0, 0b000, 0, OP_FENCE)
        assert instruction_mnemonic(decode(word)) == "FENCE"


class TestInstructionMnemonicMExt:
    """Test instruction_mnemonic for M extension instructions."""

    def test_mul(self) -> None:
        assert _mnemonic_r(0b0000001, 3, 2, 0b000, 1) == "MUL"

    def test_mulh(self) -> None:
        assert _mnemonic_r(0b0000001, 3, 2, 0b001, 1) == "MULH"

    def test_mulhsu(self) -> None:
        assert _mnemonic_r(0b0000001, 3, 2, 0b010, 1) == "MULHSU"

    def test_mulhu(self) -> None:
        assert _mnemonic_r(0b0000001, 3, 2, 0b011, 1) == "MULHU"

    def test_div(self) -> None:
        assert _mnemonic_r(0b0000001, 3, 2, 0b100, 1) == "DIV"

    def test_divu(self) -> None:
        assert _mnemonic_r(0b0000001, 3, 2, 0b101, 1) == "DIVU"

    def test_rem(self) -> None:
        assert _mnemonic_r(0b0000001, 3, 2, 0b110, 1) == "REM"

    def test_remu(self) -> None:
        assert _mnemonic_r(0b0000001, 3, 2, 0b111, 1) == "REMU"


def _encode_fp_r(funct7: int, rs2: int, rs1: int, funct3: int, rd: int, opcode: int = OP_OP_FP) -> int:
    """Encode an R-type FP instruction word."""
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


class TestInstructionMnemonicFpu:
    """Test instruction_mnemonic for F extension instructions."""

    def test_flw(self) -> None:
        word = _encode_i(0, 2, 0b010, 1, OP_LOAD_FP)
        assert instruction_mnemonic(decode(word)) == "flw"

    def test_fsw(self) -> None:
        word = (5 << 20) | (2 << 15) | (0b010 << 12) | (0 << 7) | OP_STORE_FP
        assert instruction_mnemonic(decode(word)) == "fsw"

    def test_fadd(self) -> None:
        word = _encode_fp_r(0x00, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "fadd.s"

    def test_fsub(self) -> None:
        word = _encode_fp_r(0x04, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "fsub.s"

    def test_fmul(self) -> None:
        word = _encode_fp_r(0x08, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "fmul.s"

    def test_fdiv(self) -> None:
        word = _encode_fp_r(0x0C, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "fdiv.s"

    def test_fsqrt(self) -> None:
        word = _encode_fp_r(0x2C, 0, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "fsqrt.s"

    def test_fmadd(self) -> None:
        # R4-type: rs3 in bits[31:27]
        word = (1 << 27) | (3 << 20) | (2 << 15) | (0 << 12) | (1 << 7) | OP_FMADD
        assert instruction_mnemonic(decode(word)) == "fmadd.s"

    def test_fmsub(self) -> None:
        word = (1 << 27) | (3 << 20) | (2 << 15) | (0 << 12) | (1 << 7) | OP_FMSUB
        assert instruction_mnemonic(decode(word)) == "fmsub.s"

    def test_fnmsub(self) -> None:
        word = (1 << 27) | (3 << 20) | (2 << 15) | (0 << 12) | (1 << 7) | OP_FNMSUB
        assert instruction_mnemonic(decode(word)) == "fnmsub.s"

    def test_fnmadd(self) -> None:
        word = (1 << 27) | (3 << 20) | (2 << 15) | (0 << 12) | (1 << 7) | OP_FNMADD
        assert instruction_mnemonic(decode(word)) == "fnmadd.s"

    def test_fsgnj(self) -> None:
        word = _encode_fp_r(0x10, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "fsgnj.s"

    def test_fsgnjn(self) -> None:
        word = _encode_fp_r(0x10, 3, 2, 0b001, 1)
        assert instruction_mnemonic(decode(word)) == "fsgnjn.s"

    def test_fsgnjx(self) -> None:
        word = _encode_fp_r(0x10, 3, 2, 0b010, 1)
        assert instruction_mnemonic(decode(word)) == "fsgnjx.s"

    def test_fmin(self) -> None:
        word = _encode_fp_r(0x14, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "fmin.s"

    def test_fmax(self) -> None:
        word = _encode_fp_r(0x14, 3, 2, 0b001, 1)
        assert instruction_mnemonic(decode(word)) == "fmax.s"

    def test_feq(self) -> None:
        word = _encode_fp_r(0x50, 3, 2, 0b010, 1)
        assert instruction_mnemonic(decode(word)) == "feq.s"

    def test_flt(self) -> None:
        word = _encode_fp_r(0x50, 3, 2, 0b001, 1)
        assert instruction_mnemonic(decode(word)) == "flt.s"

    def test_fle(self) -> None:
        word = _encode_fp_r(0x50, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "fle.s"

    def test_fcvt_w_s(self) -> None:
        word = _encode_fp_r(0x60, 0, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "fcvt.w.s"

    def test_fcvt_wu_s(self) -> None:
        word = _encode_fp_r(0x60, 1, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "fcvt.wu.s"

    def test_fcvt_s_w(self) -> None:
        word = _encode_fp_r(0x68, 0, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "fcvt.s.w"

    def test_fcvt_s_wu(self) -> None:
        word = _encode_fp_r(0x68, 1, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "fcvt.s.wu"

    def test_fmv_x_w(self) -> None:
        word = _encode_fp_r(0x70, 0, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "fmv.x.w"

    def test_fclass(self) -> None:
        word = _encode_fp_r(0x70, 0, 2, 0b001, 1)
        assert instruction_mnemonic(decode(word)) == "fclass.s"

    def test_fmv_w_x(self) -> None:
        word = _encode_fp_r(0x78, 0, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "fmv.w.x"


def _encode_npu_r(funct7: int, rs2: int, rs1: int, funct3: int, rd: int) -> int:
    """Encode an R-type NPU instruction word (opcode 0x0B)."""
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | OP_NPU


class TestInstructionMnemonicNpu:
    """Test instruction_mnemonic for integer NPU instructions."""

    def test_macc(self) -> None:
        word = _encode_npu_r(0, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.MACC"

    def test_vmac(self) -> None:
        word = _encode_npu_r(1, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.VMAC"

    def test_vexp(self) -> None:
        word = _encode_npu_r(2, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.VEXP"

    def test_vrsqrt(self) -> None:
        word = _encode_npu_r(3, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.VRSQRT"

    def test_vmul(self) -> None:
        word = _encode_npu_r(4, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.VMUL"

    def test_vreduce(self) -> None:
        word = _encode_npu_r(5, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.VREDUCE"

    def test_vmax(self) -> None:
        word = _encode_npu_r(6, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.VMAX"

    def test_relu(self) -> None:
        word = _encode_npu_r(0, 0, 2, 0b001, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.RELU"

    def test_qmul(self) -> None:
        word = _encode_npu_r(0, 3, 2, 0b010, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.QMUL"

    def test_clamp(self) -> None:
        word = _encode_npu_r(0, 0, 2, 0b011, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.CLAMP"

    def test_gelu(self) -> None:
        word = _encode_npu_r(0, 0, 2, 0b100, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.GELU"

    def test_rstacc(self) -> None:
        word = _encode_npu_r(0, 0, 0, 0b101, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.RSTACC"

    def test_ldvec(self) -> None:
        # LDVEC is I-type (funct3=6)
        word = (0 << 20) | (2 << 15) | (0b110 << 12) | (1 << 7) | OP_NPU
        assert instruction_mnemonic(decode(word)) == "NPU.LDVEC"

    def test_stvec(self) -> None:
        # STVEC is S-type like (funct3=7)
        word = (3 << 20) | (2 << 15) | (0b111 << 12) | (0 << 7) | OP_NPU
        assert instruction_mnemonic(decode(word)) == "NPU.STVEC"


def _encode_fp_npu_r(funct7: int, rs2: int, rs1: int, funct3: int, rd: int) -> int:
    """Encode an R-type FP NPU instruction word (opcode 0x2B)."""
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | OP_FP_NPU


class TestInstructionMnemonicFpNpu:
    """Test instruction_mnemonic for FP NPU instructions."""

    def test_fmacc(self) -> None:
        word = _encode_fp_npu_r(0, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.FMACC"

    def test_fvmac(self) -> None:
        word = _encode_fp_npu_r(1, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.FVMAC"

    def test_fvexp(self) -> None:
        word = _encode_fp_npu_r(2, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.FVEXP"

    def test_fvrsqrt(self) -> None:
        word = _encode_fp_npu_r(3, 0, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.FVRSQRT"

    def test_fvmul(self) -> None:
        word = _encode_fp_npu_r(4, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.FVMUL"

    def test_fvreduce(self) -> None:
        word = _encode_fp_npu_r(5, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.FVREDUCE"

    def test_fvmax(self) -> None:
        word = _encode_fp_npu_r(6, 3, 2, 0b000, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.FVMAX"

    def test_frelu(self) -> None:
        word = _encode_fp_npu_r(0, 0, 2, 0b001, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.FRELU"

    def test_fgelu(self) -> None:
        word = _encode_fp_npu_r(0, 0, 2, 0b100, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.FGELU"

    def test_frstacc(self) -> None:
        word = _encode_fp_npu_r(0, 0, 0, 0b101, 1)
        assert instruction_mnemonic(decode(word)) == "NPU.FRSTACC"
