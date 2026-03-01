"""Tests for the TUI memory hex dump module."""

from riscv_npu.memory.bus import MemoryBus
from riscv_npu.memory.ram import RAM
from riscv_npu.tui.memory import format_hex_dump


def _make_memory(base: int = 0x80000000, size: int = 256) -> MemoryBus:
    """Create a MemoryBus with RAM at the given base address."""
    bus = MemoryBus()
    ram = RAM(base, size)
    bus.register(base, size, ram)
    return bus


class TestFormatHexDump:
    """Tests for format_hex_dump function."""

    def test_single_row_all_zeros(self) -> None:
        bus = _make_memory()
        output = format_hex_dump(bus, 0x80000000, num_rows=1)
        assert "0x80000000:" in output
        assert "00 00 00 00 00 00 00 00" in output

    def test_hex_byte_values(self) -> None:
        bus = _make_memory()
        bus.write8(0x80000000, 0xDE)
        bus.write8(0x80000001, 0xAD)
        bus.write8(0x80000002, 0xBE)
        bus.write8(0x80000003, 0xEF)
        output = format_hex_dump(bus, 0x80000000, num_rows=1)
        assert "DE AD BE EF" in output

    def test_ascii_printable_characters(self) -> None:
        bus = _make_memory()
        # Write "Hello" at the start
        for i, ch in enumerate(b"Hello"):
            bus.write8(0x80000000 + i, ch)
        output = format_hex_dump(bus, 0x80000000, num_rows=1)
        assert "Hello" in output

    def test_ascii_non_printable_as_dot(self) -> None:
        bus = _make_memory()
        bus.write8(0x80000000, 0x01)  # Non-printable
        bus.write8(0x80000001, 0x7F)  # DEL, non-printable
        output = format_hex_dump(bus, 0x80000000, num_rows=1)
        # Both non-printable bytes should show as '.'
        # The ASCII column starts after the pipe
        ascii_section = output.split("|")[1]
        assert ascii_section.startswith("..")

    def test_address_alignment(self) -> None:
        bus = _make_memory()
        # Start at unaligned address, should align down to 16-byte boundary
        output = format_hex_dump(bus, 0x80000005, num_rows=1)
        assert "0x80000000:" in output  # Aligned down

    def test_multiple_rows(self) -> None:
        bus = _make_memory()
        output = format_hex_dump(bus, 0x80000000, num_rows=3)
        lines = output.strip().split("\n")
        assert len(lines) == 3
        assert "0x80000000:" in lines[0]
        assert "0x80000010:" in lines[1]
        assert "0x80000020:" in lines[2]

    def test_unmapped_address_shows_question_marks(self) -> None:
        bus = MemoryBus()
        # No devices registered, everything unmapped
        output = format_hex_dump(bus, 0x00000000, num_rows=1)
        assert "??" in output
        # All 16 bytes should be '??'
        assert output.count("??") == 16

    def test_row_contains_pipe_delimited_ascii(self) -> None:
        bus = _make_memory()
        output = format_hex_dump(bus, 0x80000000, num_rows=1)
        # Should have |...| at the end
        assert output.count("|") == 2  # Opening and closing pipe

    def test_group_separation_between_byte_8_and_9(self) -> None:
        bus = _make_memory()
        for i in range(16):
            bus.write8(0x80000000 + i, 0xAA)
        output = format_hex_dump(bus, 0x80000000, num_rows=1)
        # There should be a double space between the two groups of 8
        # The hex section should look like "AA AA ... AA  AA AA ... AA"
        hex_section = output.split(": ", 1)[1].split("  |")[0]
        # The groups should be separated by a double space (from the extra "" join)
        assert "AA AA AA AA AA AA AA AA  AA AA AA AA AA AA AA AA" in hex_section

    def test_default_num_rows(self) -> None:
        bus = _make_memory(size=1024)
        output = format_hex_dump(bus, 0x80000000)
        lines = output.strip().split("\n")
        assert len(lines) == 16  # Default is 16 rows
