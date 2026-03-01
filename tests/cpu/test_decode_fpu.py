"""Tests for F-extension instruction decoding."""

import pytest

from riscv_npu.cpu.decode import (
    Instruction,
    decode,
    OP_LOAD_FP,
    OP_STORE_FP,
    OP_FMADD,
    OP_FMSUB,
    OP_FNMSUB,
    OP_FNMADD,
    OP_OP_FP,
)


def _encode_r_type(opcode: int, rd: int, funct3: int, rs1: int, rs2: int, funct7: int) -> int:
    """Encode an R-type instruction word."""
    return (
        (funct7 << 25) | (rs2 << 20) | (rs1 << 15) |
        (funct3 << 12) | (rd << 7) | opcode
    )


def _encode_r4_type(opcode: int, rd: int, funct3: int, rs1: int, rs2: int, rs3: int, fmt: int = 0) -> int:
    """Encode an R4-type instruction word (fused multiply-add)."""
    return (
        (rs3 << 27) | (fmt << 25) | (rs2 << 20) | (rs1 << 15) |
        (funct3 << 12) | (rd << 7) | opcode
    )


def _encode_i_type(opcode: int, rd: int, funct3: int, rs1: int, imm: int) -> int:
    """Encode an I-type instruction word."""
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def _encode_s_type(opcode: int, funct3: int, rs1: int, rs2: int, imm: int) -> int:
    """Encode an S-type instruction word."""
    imm_11_5 = (imm >> 5) & 0x7F
    imm_4_0 = imm & 0x1F
    return (imm_11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_4_0 << 7) | opcode


class TestDecodeFLW:
    """Test FLW (float load word) decoding."""

    def test_flw_basic(self) -> None:
        word = _encode_i_type(OP_LOAD_FP, rd=1, funct3=2, rs1=2, imm=16)
        inst = decode(word)
        assert inst.opcode == OP_LOAD_FP
        assert inst.rd == 1
        assert inst.rs1 == 2
        assert inst.funct3 == 2

    def test_flw_negative_offset(self) -> None:
        word = _encode_i_type(OP_LOAD_FP, rd=5, funct3=2, rs1=10, imm=0xFFC)  # -4
        inst = decode(word)
        assert inst.opcode == OP_LOAD_FP
        assert inst.rd == 5
        assert inst.rs1 == 10


class TestDecodeFSW:
    """Test FSW (float store word) decoding."""

    def test_fsw_basic(self) -> None:
        word = _encode_s_type(OP_STORE_FP, funct3=2, rs1=2, rs2=1, imm=8)
        inst = decode(word)
        assert inst.opcode == OP_STORE_FP
        assert inst.rs1 == 2
        assert inst.rs2 == 1

    def test_fsw_negative_offset(self) -> None:
        word = _encode_s_type(OP_STORE_FP, funct3=2, rs1=3, rs2=4, imm=0xFFC)
        inst = decode(word)
        assert inst.opcode == OP_STORE_FP
        assert inst.rs1 == 3
        assert inst.rs2 == 4


class TestDecodeR4Type:
    """Test R4-type (fused multiply-add) decoding."""

    def test_fmadd_rs3_extraction(self) -> None:
        word = _encode_r4_type(OP_FMADD, rd=1, funct3=0, rs1=2, rs2=3, rs3=4)
        inst = decode(word)
        assert inst.opcode == OP_FMADD
        assert inst.rd == 1
        assert inst.rs1 == 2
        assert inst.rs2 == 3
        assert inst.rs3 == 4

    def test_fmsub_rs3_extraction(self) -> None:
        word = _encode_r4_type(OP_FMSUB, rd=5, funct3=0, rs1=6, rs2=7, rs3=8)
        inst = decode(word)
        assert inst.opcode == OP_FMSUB
        assert inst.rs3 == 8

    def test_fnmsub_rs3_extraction(self) -> None:
        word = _encode_r4_type(OP_FNMSUB, rd=10, funct3=0, rs1=11, rs2=12, rs3=31)
        inst = decode(word)
        assert inst.opcode == OP_FNMSUB
        assert inst.rs3 == 31

    def test_fnmadd_rs3_extraction(self) -> None:
        word = _encode_r4_type(OP_FNMADD, rd=0, funct3=0, rs1=1, rs2=2, rs3=15)
        inst = decode(word)
        assert inst.opcode == OP_FNMADD
        assert inst.rs3 == 15

    def test_r4_all_registers_max(self) -> None:
        word = _encode_r4_type(OP_FMADD, rd=31, funct3=7, rs1=31, rs2=31, rs3=31)
        inst = decode(word)
        assert inst.rd == 31
        assert inst.rs1 == 31
        assert inst.rs2 == 31
        assert inst.rs3 == 31


class TestDecodeOpFP:
    """Test OP-FP (floating-point arithmetic) decoding."""

    def test_fadd_s(self) -> None:
        word = _encode_r_type(OP_OP_FP, rd=1, funct3=0, rs1=2, rs2=3, funct7=0x00)
        inst = decode(word)
        assert inst.opcode == OP_OP_FP
        assert inst.funct7 == 0x00
        assert inst.rd == 1
        assert inst.rs1 == 2
        assert inst.rs2 == 3

    def test_fsub_s(self) -> None:
        word = _encode_r_type(OP_OP_FP, rd=4, funct3=0, rs1=5, rs2=6, funct7=0x04)
        inst = decode(word)
        assert inst.funct7 == 0x04

    def test_fmul_s(self) -> None:
        word = _encode_r_type(OP_OP_FP, rd=7, funct3=0, rs1=8, rs2=9, funct7=0x08)
        inst = decode(word)
        assert inst.funct7 == 0x08

    def test_fdiv_s(self) -> None:
        word = _encode_r_type(OP_OP_FP, rd=10, funct3=0, rs1=11, rs2=12, funct7=0x0C)
        inst = decode(word)
        assert inst.funct7 == 0x0C

    def test_fsqrt_s(self) -> None:
        word = _encode_r_type(OP_OP_FP, rd=1, funct3=0, rs1=2, rs2=0, funct7=0x2C)
        inst = decode(word)
        assert inst.funct7 == 0x2C
        assert inst.rs2 == 0

    def test_fcmp_feq(self) -> None:
        word = _encode_r_type(OP_OP_FP, rd=1, funct3=2, rs1=2, rs2=3, funct7=0x50)
        inst = decode(word)
        assert inst.funct7 == 0x50
        assert inst.funct3 == 2

    def test_fclass(self) -> None:
        word = _encode_r_type(OP_OP_FP, rd=1, funct3=1, rs1=2, rs2=0, funct7=0x70)
        inst = decode(word)
        assert inst.funct7 == 0x70
        assert inst.funct3 == 1

    def test_fmv_x_w(self) -> None:
        word = _encode_r_type(OP_OP_FP, rd=1, funct3=0, rs1=2, rs2=0, funct7=0x70)
        inst = decode(word)
        assert inst.funct7 == 0x70
        assert inst.funct3 == 0

    def test_fmv_w_x(self) -> None:
        word = _encode_r_type(OP_OP_FP, rd=1, funct3=0, rs1=2, rs2=0, funct7=0x78)
        inst = decode(word)
        assert inst.funct7 == 0x78
