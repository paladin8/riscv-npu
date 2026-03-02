"""Tests for TUI instruction statistics formatting."""

from riscv_npu.tui.stats import format_instruction_stats, _categorize


class TestCategorize:
    """Test _categorize classifies mnemonics correctly."""

    def test_rv32i(self) -> None:
        assert _categorize("ADD") == "RV32I"
        assert _categorize("ADDI") == "RV32I"
        assert _categorize("LW") == "RV32I"
        assert _categorize("BEQ") == "RV32I"
        assert _categorize("JAL") == "RV32I"
        assert _categorize("ECALL") == "RV32I"

    def test_m_ext(self) -> None:
        assert _categorize("MUL") == "M-ext"
        assert _categorize("DIV") == "M-ext"
        assert _categorize("REMU") == "M-ext"

    def test_f_ext(self) -> None:
        assert _categorize("fadd.s") == "F-ext"
        assert _categorize("flw") == "F-ext"
        assert _categorize("fsw") == "F-ext"
        assert _categorize("fmadd.s") == "F-ext"

    def test_npu_int(self) -> None:
        assert _categorize("NPU.MACC") == "NPU-int"
        assert _categorize("NPU.VMAC") == "NPU-int"
        assert _categorize("NPU.RELU") == "NPU-int"

    def test_npu_fp(self) -> None:
        assert _categorize("NPU.FMACC") == "NPU-fp"
        assert _categorize("NPU.FVMAC") == "NPU-fp"
        assert _categorize("NPU.FRELU") == "NPU-fp"

    def test_unknown(self) -> None:
        assert _categorize("UNKNOWN_INSTR") == "Other"


class TestFormatInstructionStatsEmpty:
    """Test format_instruction_stats with empty input."""

    def test_empty_dict(self) -> None:
        result = format_instruction_stats({})
        assert result == "No instructions executed."


class TestFormatInstructionStatsTopN:
    """Test that top_n parameter limits individual entries."""

    def test_shows_top_n_entries(self) -> None:
        stats = {f"INSTR_{i}": (100 - i) for i in range(20)}
        result = format_instruction_stats(stats, top_n=5)
        # Should have INSTR_0 through INSTR_4 (top 5 by count)
        assert "INSTR_0" in result
        assert "INSTR_4" in result
        # INSTR_5 should not appear as an individual entry (in "others")
        assert "INSTR_5" not in result
        assert "... others" in result

    def test_shows_all_when_fewer_than_top_n(self) -> None:
        stats = {"ADD": 10, "SUB": 5}
        result = format_instruction_stats(stats, top_n=15)
        assert "ADD" in result
        assert "SUB" in result
        assert "... others" not in result


class TestFormatInstructionStatsCategories:
    """Test category grouping in the output."""

    def test_categories_appear(self) -> None:
        stats = {
            "ADD": 100,
            "MUL": 50,
            "fadd.s": 30,
            "NPU.MACC": 20,
            "NPU.FMACC": 10,
        }
        result = format_instruction_stats(stats)
        assert "By category:" in result
        assert "RV32I" in result
        assert "M-ext" in result
        assert "F-ext" in result
        assert "NPU-int" in result
        assert "NPU-fp" in result


class TestFormatInstructionStatsPercentages:
    """Test that percentages are computed correctly."""

    def test_single_instruction_is_100_pct(self) -> None:
        stats = {"ADD": 100}
        result = format_instruction_stats(stats)
        assert "100.0%" in result

    def test_two_equal_instructions_are_50_pct(self) -> None:
        stats = {"ADD": 50, "SUB": 50}
        result = format_instruction_stats(stats)
        assert "50.0%" in result

    def test_total_count_shown(self) -> None:
        stats = {"ADD": 1234, "SUB": 5678}
        result = format_instruction_stats(stats)
        assert "6,912" in result  # 1234 + 5678
