"""Tests for RAM read/write operations."""

import pytest

from riscv_npu.memory.ram import RAM

BASE = 0x80000000
SIZE = 4096


class TestRAM:
    def test_read_write_8bit(self) -> None:
        ram = RAM(BASE, SIZE)
        ram.write8(BASE, 0xAB)
        assert ram.read8(BASE) == 0xAB

    def test_read_write_16bit(self) -> None:
        ram = RAM(BASE, SIZE)
        ram.write16(BASE, 0x1234)
        assert ram.read16(BASE) == 0x1234

    def test_read_write_32bit(self) -> None:
        ram = RAM(BASE, SIZE)
        ram.write32(BASE, 0xDEADBEEF)
        assert ram.read32(BASE) == 0xDEADBEEF

    def test_little_endian(self) -> None:
        """write32 stores in little-endian: LSB at lowest address."""
        ram = RAM(BASE, SIZE)
        ram.write32(BASE, 0x04030201)
        assert ram.read8(BASE) == 0x01
        assert ram.read8(BASE + 1) == 0x02
        assert ram.read8(BASE + 2) == 0x03
        assert ram.read8(BASE + 3) == 0x04

    def test_little_endian_16(self) -> None:
        """write16 stores in little-endian."""
        ram = RAM(BASE, SIZE)
        ram.write16(BASE, 0xBEEF)
        assert ram.read8(BASE) == 0xEF
        assert ram.read8(BASE + 1) == 0xBE

    def test_out_of_bounds_read(self) -> None:
        ram = RAM(BASE, SIZE)
        with pytest.raises(MemoryError):
            ram.read8(BASE + SIZE)

    def test_out_of_bounds_below_base(self) -> None:
        ram = RAM(BASE, SIZE)
        with pytest.raises(MemoryError):
            ram.read8(BASE - 1)

    def test_out_of_bounds_write(self) -> None:
        ram = RAM(BASE, SIZE)
        with pytest.raises(MemoryError):
            ram.write32(BASE + SIZE - 3, 0)  # needs 4 bytes, only 3 remain

    def test_base_address_offset(self) -> None:
        """Reads/writes at base+offset work correctly."""
        ram = RAM(BASE, SIZE)
        ram.write8(BASE + 100, 0x42)
        assert ram.read8(BASE + 100) == 0x42

    def test_unsigned_byte(self) -> None:
        """read8 returns unsigned value (0-255)."""
        ram = RAM(BASE, SIZE)
        ram.write8(BASE, 0xFF)
        assert ram.read8(BASE) == 0xFF  # 255, not -1

    def test_write_masks_value(self) -> None:
        """write8 masks to 8 bits, write16 to 16 bits."""
        ram = RAM(BASE, SIZE)
        ram.write8(BASE, 0x1FF)  # should store 0xFF
        assert ram.read8(BASE) == 0xFF
        ram.write16(BASE, 0x1FFFF)  # should store 0xFFFF
        assert ram.read16(BASE) == 0xFFFF

    def test_initial_values_zero(self) -> None:
        ram = RAM(BASE, SIZE)
        assert ram.read32(BASE) == 0


class TestLoadSegment:
    def test_load_and_readback(self) -> None:
        """load_segment copies bytes, which can be read back with read8/32."""
        ram = RAM(BASE, SIZE)
        data = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        ram.load_segment(BASE, data)
        assert ram.read8(BASE) == 0x01
        assert ram.read8(BASE + 7) == 0x08
        # Little-endian 32-bit read of first 4 bytes
        assert ram.read32(BASE) == 0x04030201

    def test_load_at_offset(self) -> None:
        """load_segment works at an address offset from base."""
        ram = RAM(BASE, SIZE)
        data = b"\xAA\xBB\xCC\xDD"
        ram.load_segment(BASE + 256, data)
        assert ram.read32(BASE + 256) == 0xDDCCBBAA

    def test_load_bounds_check(self) -> None:
        """load_segment raises MemoryError if segment exceeds RAM."""
        ram = RAM(BASE, SIZE)
        with pytest.raises(MemoryError):
            ram.load_segment(BASE + SIZE - 3, b"\x00" * 4)

    def test_load_below_base(self) -> None:
        """load_segment raises MemoryError for address below RAM base."""
        ram = RAM(BASE, SIZE)
        with pytest.raises(MemoryError):
            ram.load_segment(BASE - 1, b"\x00")

    def test_load_empty(self) -> None:
        """load_segment with empty data is a no-op."""
        ram = RAM(BASE, SIZE)
        ram.load_segment(BASE, b"")
        assert ram.read32(BASE) == 0  # Still zero
