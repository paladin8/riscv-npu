"""Tests for register file."""

from riscv_npu.cpu.registers import RegisterFile


class TestRegisterFile:
    def test_x0_always_zero(self) -> None:
        """Writing to x0 is discarded, read always returns 0."""
        rf = RegisterFile()
        rf.write(0, 12345)
        assert rf.read(0) == 0

    def test_read_write_x1(self) -> None:
        rf = RegisterFile()
        rf.write(1, 42)
        assert rf.read(1) == 42

    def test_read_write_x31(self) -> None:
        rf = RegisterFile()
        rf.write(31, 0xDEADBEEF)
        assert rf.read(31) == 0xDEADBEEF

    def test_write_masks_to_32_bits(self) -> None:
        """Values larger than 32 bits are masked."""
        rf = RegisterFile()
        rf.write(1, 0x1_FFFFFFFF)
        assert rf.read(1) == 0xFFFFFFFF

    def test_initial_values_are_zero(self) -> None:
        rf = RegisterFile()
        for i in range(32):
            assert rf.read(i) == 0

    def test_independent_registers(self) -> None:
        """Writing one register doesn't affect others."""
        rf = RegisterFile()
        rf.write(1, 111)
        rf.write(2, 222)
        assert rf.read(1) == 111
        assert rf.read(2) == 222
        assert rf.read(3) == 0

    def test_overwrite(self) -> None:
        """Writing to the same register overwrites the previous value."""
        rf = RegisterFile()
        rf.write(5, 100)
        rf.write(5, 200)
        assert rf.read(5) == 200
