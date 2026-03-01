"""Tests for the TUI register display module."""

from riscv_npu.cpu.registers import RegisterFile
from riscv_npu.tui.registers import (
    ABI_NAMES,
    format_registers,
    snapshot_registers,
)


class TestABINames:
    """Tests for ABI name constants."""

    def test_abi_names_has_32_entries(self) -> None:
        assert len(ABI_NAMES) == 32

    def test_abi_names_starts_with_zero(self) -> None:
        assert ABI_NAMES[0] == "zero"

    def test_abi_names_ra(self) -> None:
        assert ABI_NAMES[1] == "ra"

    def test_abi_names_sp(self) -> None:
        assert ABI_NAMES[2] == "sp"

    def test_abi_names_ends_with_t6(self) -> None:
        assert ABI_NAMES[31] == "t6"


class TestFormatRegisters:
    """Tests for format_registers function."""

    def test_all_32_registers_present(self) -> None:
        regs = RegisterFile()
        output = format_registers(regs)
        for i in range(32):
            assert f"x{i}" in output

    def test_all_abi_names_present(self) -> None:
        regs = RegisterFile()
        output = format_registers(regs)
        for name in ABI_NAMES:
            assert name in output

    def test_x0_always_shows_zero(self) -> None:
        regs = RegisterFile()
        output = format_registers(regs)
        assert "0x00000000" in output

    def test_register_value_displayed_in_hex(self) -> None:
        regs = RegisterFile()
        regs.write(1, 0xDEADBEEF)
        output = format_registers(regs)
        assert "0xDEADBEEF" in output

    def test_no_highlighting_without_prev_values(self) -> None:
        regs = RegisterFile()
        regs.write(1, 0x42)
        output = format_registers(regs)
        assert "[bold yellow]" not in output

    def test_no_highlighting_when_values_unchanged(self) -> None:
        regs = RegisterFile()
        regs.write(1, 0x42)
        prev = snapshot_registers(regs)
        output = format_registers(regs, prev)
        assert "[bold yellow]" not in output

    def test_highlighting_when_value_changed(self) -> None:
        regs = RegisterFile()
        regs.write(1, 0x42)
        prev = snapshot_registers(regs)
        regs.write(1, 0x99)
        output = format_registers(regs, prev)
        assert "[bold yellow]" in output
        assert "0x00000099" in output

    def test_x0_never_highlighted(self) -> None:
        regs = RegisterFile()
        # x0 is always 0, and prev[0] would also be 0
        prev = [0] * 32
        prev[0] = 999  # Fake a "change" for x0
        output = format_registers(regs, prev)
        # x0 line should NOT be highlighted since x0 is hardwired to 0
        lines = output.split("\n")
        # x0 appears in the first column of the first line
        first_line = lines[0]
        # The x0 entry should not be wrapped in bold yellow
        x0_entry = first_line.split("  ")[0]
        assert "[bold yellow]" not in x0_entry

    def test_multiple_registers_changed(self) -> None:
        regs = RegisterFile()
        prev = snapshot_registers(regs)
        regs.write(5, 0xAAAA)
        regs.write(10, 0xBBBB)
        output = format_registers(regs, prev)
        assert output.count("[bold yellow]") == 2

    def test_output_has_8_lines(self) -> None:
        regs = RegisterFile()
        output = format_registers(regs)
        lines = output.strip().split("\n")
        assert len(lines) == 8


class TestSnapshotRegisters:
    """Tests for snapshot_registers function."""

    def test_snapshot_returns_32_values(self) -> None:
        regs = RegisterFile()
        snap = snapshot_registers(regs)
        assert len(snap) == 32

    def test_snapshot_captures_current_values(self) -> None:
        regs = RegisterFile()
        regs.write(1, 0x42)
        regs.write(5, 0xABCD)
        snap = snapshot_registers(regs)
        assert snap[0] == 0  # x0
        assert snap[1] == 0x42
        assert snap[5] == 0xABCD

    def test_snapshot_is_independent_copy(self) -> None:
        regs = RegisterFile()
        regs.write(1, 0x42)
        snap = snapshot_registers(regs)
        regs.write(1, 0x99)
        assert snap[1] == 0x42  # Snapshot should not change
