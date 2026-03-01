"""Tests for the TUI NPU state formatter."""

from riscv_npu.npu.engine import NpuState, acc_set64
from riscv_npu.tui.npu import format_npu_state


class TestFormatNpuState:
    """Tests for format_npu_state."""

    def test_default_state_all_zeros(self) -> None:
        npu = NpuState()
        result = format_npu_state(npu)
        assert "0x00000000_00000000" in result
        assert "(0)" in result
        assert "v0:" in result
        assert "v3:" in result

    def test_default_state_no_highlighting(self) -> None:
        npu = NpuState()
        result = format_npu_state(npu)
        assert "[bold yellow]" not in result

    def test_nonzero_accumulator_highlighted(self) -> None:
        npu = NpuState(acc_lo=0xFF)
        result = format_npu_state(npu)
        assert "[bold yellow]Accumulator[/bold yellow]" in result
        assert "0x00000000_000000FF" in result
        assert "(255)" in result

    def test_accumulator_hi_nonzero(self) -> None:
        npu = NpuState(acc_hi=0x00000003, acc_lo=0x000000FF)
        result = format_npu_state(npu)
        assert "0x00000003_000000FF" in result
        assert "[bold yellow]" in result

    def test_negative_accumulator(self) -> None:
        npu = NpuState()
        acc_set64(npu, -1)
        result = format_npu_state(npu)
        assert "0xFFFFFFFF_FFFFFFFF" in result
        assert "(-1)" in result
        assert "[bold yellow]" in result

    def test_large_positive_accumulator(self) -> None:
        npu = NpuState()
        acc_set64(npu, 2**40)
        result = format_npu_state(npu)
        assert "0x00000100_00000000" in result
        assert f"({2**40})" in result

    def test_nonzero_vreg_highlighted(self) -> None:
        npu = NpuState()
        npu.vreg[1] = [10, -20, 127, -128]
        result = format_npu_state(npu)
        lines = result.split("\n")
        # v0 should NOT be highlighted (all zeros)
        v0_line = [l for l in lines if "v0:" in l][0]
        assert "[bold yellow]" not in v0_line
        # v1 SHOULD be highlighted
        v1_line = [l for l in lines if "v1:" in l][0]
        assert "[bold yellow]" in v1_line

    def test_vreg_format_int8_values(self) -> None:
        npu = NpuState()
        npu.vreg[2] = [-128, -1, 0, 127]
        result = format_npu_state(npu)
        assert "-128" in result
        assert "  -1" in result
        assert " 127" in result

    def test_all_vregs_present(self) -> None:
        npu = NpuState()
        result = format_npu_state(npu)
        for i in range(4):
            assert f"v{i}:" in result

    def test_accumulator_zero_vreg_nonzero(self) -> None:
        npu = NpuState()
        npu.vreg[0] = [1, 2, 3, 4]
        result = format_npu_state(npu)
        # Accumulator section should NOT be highlighted
        lines = result.split("\n")
        assert lines[0] == "Accumulator"
        # v0 should be highlighted
        v0_line = [l for l in lines if "v0:" in l][0]
        assert "[bold yellow]" in v0_line

    def test_accumulator_min_int64(self) -> None:
        npu = NpuState(acc_hi=0x80000000, acc_lo=0)
        result = format_npu_state(npu)
        assert "0x80000000_00000000" in result
        assert f"({-(2**63)})" in result

    def test_accumulator_max_int64(self) -> None:
        npu = NpuState(acc_hi=0x7FFFFFFF, acc_lo=0xFFFFFFFF)
        result = format_npu_state(npu)
        assert "0x7FFFFFFF_FFFFFFFF" in result
        assert f"({2**63 - 1})" in result

    def test_sections_present(self) -> None:
        npu = NpuState()
        result = format_npu_state(npu)
        assert "Accumulator" in result
        assert "Vector Registers" in result
