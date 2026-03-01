"""Tests for NPU compute engine: NpuState, GELU table, accumulator helpers."""

import math

from riscv_npu.npu.engine import (
    NpuState,
    acc_add,
    acc_get64,
    acc_reset,
    acc_set64,
    build_gelu_table,
    GELU_TABLE,
    _gelu_int8,
)


class TestNpuState:
    """Tests for NpuState initialization."""

    def test_initial_accumulator_zero(self) -> None:
        """Accumulator starts at 0."""
        state = NpuState()
        assert state.acc_lo == 0
        assert state.acc_hi == 0

    def test_initial_vregs_zero(self) -> None:
        """All 4 vector registers start at zero."""
        state = NpuState()
        assert len(state.vreg) == 4
        for v in state.vreg:
            assert v == [0, 0, 0, 0]

    def test_vregs_independent(self) -> None:
        """Modifying one vreg does not affect others."""
        state = NpuState()
        state.vreg[0][0] = 42
        assert state.vreg[1][0] == 0
        assert state.vreg[2][0] == 0
        assert state.vreg[3][0] == 0

    def test_two_states_independent(self) -> None:
        """Two NpuState instances do not share mutable state."""
        s1 = NpuState()
        s2 = NpuState()
        s1.vreg[0][0] = 99
        s1.acc_lo = 123
        assert s2.vreg[0][0] == 0
        assert s2.acc_lo == 0


class TestAccumulator:
    """Tests for accumulator helper functions."""

    def test_acc_set_get_positive(self) -> None:
        """Set and get a positive 64-bit value."""
        state = NpuState()
        acc_set64(state, 0x0000000100000002)
        assert state.acc_lo == 0x00000002
        assert state.acc_hi == 0x00000001
        assert acc_get64(state) == 0x0000000100000002

    def test_acc_set_get_negative(self) -> None:
        """Set and get a negative 64-bit value."""
        state = NpuState()
        acc_set64(state, -1)
        assert state.acc_lo == 0xFFFFFFFF
        assert state.acc_hi == 0xFFFFFFFF
        assert acc_get64(state) == -1

    def test_acc_add_positive(self) -> None:
        """Add a positive value to the accumulator."""
        state = NpuState()
        acc_add(state, 100)
        assert acc_get64(state) == 100

    def test_acc_add_negative(self) -> None:
        """Add a negative value to the accumulator."""
        state = NpuState()
        acc_add(state, -50)
        assert acc_get64(state) == -50

    def test_acc_add_chain(self) -> None:
        """Multiple adds accumulate correctly."""
        state = NpuState()
        for i in range(10):
            acc_add(state, 1000)
        assert acc_get64(state) == 10000

    def test_acc_add_crosses_32bit(self) -> None:
        """Accumulator correctly handles values > 32 bits."""
        state = NpuState()
        acc_add(state, 0x100000000)  # 2^32
        assert state.acc_lo == 0
        assert state.acc_hi == 1
        assert acc_get64(state) == 0x100000000

    def test_acc_add_overflow_wraps(self) -> None:
        """64-bit overflow wraps around."""
        state = NpuState()
        acc_set64(state, 0x7FFFFFFFFFFFFFFF)  # max positive
        acc_add(state, 1)
        # Should wrap to most negative 64-bit value
        assert acc_get64(state) == -0x8000000000000000

    def test_acc_reset_returns_lo(self) -> None:
        """acc_reset returns old acc_lo and zeroes both halves."""
        state = NpuState()
        acc_set64(state, 0x0000000ADEADBEEF)
        old_lo = acc_reset(state)
        assert old_lo == 0xDEADBEEF
        assert state.acc_lo == 0
        assert state.acc_hi == 0

    def test_acc_reset_when_zero(self) -> None:
        """acc_reset on zeroed accumulator returns 0."""
        state = NpuState()
        assert acc_reset(state) == 0
        assert state.acc_lo == 0
        assert state.acc_hi == 0


class TestGeluTable:
    """Tests for GELU lookup table."""

    def test_table_length(self) -> None:
        """Table has exactly 256 entries (one per int8 value)."""
        assert len(GELU_TABLE) == 256

    def test_gelu_zero(self) -> None:
        """GELU(0) = 0."""
        assert GELU_TABLE[128] == 0  # index 128 = input 0

    def test_gelu_positive_passthrough(self) -> None:
        """Large positive inputs pass through approximately unchanged."""
        # For x >> 0, gelu(x) ~ x
        # Input 127 (index 255): should be close to 127
        assert GELU_TABLE[255] == 127

    def test_gelu_negative_suppressed(self) -> None:
        """Large negative inputs are suppressed toward 0."""
        # For x << 0, gelu(x) ~ 0
        # Input -128 (index 0): should be close to 0
        assert GELU_TABLE[0] == 0

    def test_gelu_reference_values(self) -> None:
        """Compare several table entries to Python math.erf reference."""
        for x in [-64, -32, -16, -8, 0, 8, 16, 32, 64]:
            expected = _gelu_int8(x)
            table_idx = x + 128
            assert GELU_TABLE[table_idx] == expected, (
                f"GELU mismatch at x={x}: table={GELU_TABLE[table_idx]}, "
                f"expected={expected}"
            )

    def test_gelu_monotonic_positive(self) -> None:
        """GELU is monotonically non-decreasing for positive inputs."""
        for i in range(129, 256):
            assert GELU_TABLE[i] >= GELU_TABLE[i - 1], (
                f"GELU not monotonic at index {i}: "
                f"{GELU_TABLE[i-1]} > {GELU_TABLE[i]}"
            )

    def test_gelu_all_in_int8_range(self) -> None:
        """All table entries are valid int8 values."""
        for i, val in enumerate(GELU_TABLE):
            assert -128 <= val <= 127, (
                f"GELU out of int8 range at index {i}: {val}"
            )

    def test_build_gelu_table_matches_module(self) -> None:
        """build_gelu_table() produces the same result as GELU_TABLE."""
        fresh = build_gelu_table()
        assert fresh == GELU_TABLE
