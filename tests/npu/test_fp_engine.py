"""Tests for FP NPU engine: facc accumulator and fgelu function."""

import math

from riscv_npu.npu.engine import NpuState, facc_add, facc_reset, facc_to_f32_bits, fgelu


class TestFaccAccumulator:
    """FP accumulator add/reset semantics."""

    def test_initial_zero(self) -> None:
        """FP accumulator starts at 0.0."""
        npu = NpuState()
        assert npu.facc == 0.0

    def test_add_positive(self) -> None:
        """Add a positive value to facc."""
        npu = NpuState()
        facc_add(npu, 3.14)
        assert npu.facc == 3.14

    def test_add_multiple(self) -> None:
        """Accumulate multiple values."""
        npu = NpuState()
        facc_add(npu, 1.0)
        facc_add(npu, 2.0)
        facc_add(npu, 3.0)
        assert npu.facc == 6.0

    def test_add_negative(self) -> None:
        """Add a negative value."""
        npu = NpuState()
        facc_add(npu, -5.5)
        assert npu.facc == -5.5

    def test_reset_returns_old_value(self) -> None:
        """Reset returns the previous accumulator value."""
        npu = NpuState()
        facc_add(npu, 42.0)
        old = facc_reset(npu)
        assert old == 42.0

    def test_reset_zeroes_accumulator(self) -> None:
        """Reset sets accumulator to 0.0."""
        npu = NpuState()
        facc_add(npu, 42.0)
        facc_reset(npu)
        assert npu.facc == 0.0

    def test_to_f32_bits_one(self) -> None:
        """facc_to_f32_bits for 1.0 = 0x3F800000."""
        npu = NpuState()
        npu.facc = 1.0
        assert facc_to_f32_bits(npu) == 0x3F800000

    def test_to_f32_bits_zero(self) -> None:
        """facc_to_f32_bits for 0.0 = 0x00000000."""
        npu = NpuState()
        assert facc_to_f32_bits(npu) == 0x00000000

    def test_double_precision_accumulation(self) -> None:
        """FP accumulator should maintain double precision internally."""
        npu = NpuState()
        # Add many small values that would lose precision in float32
        for _ in range(1000000):
            facc_add(npu, 1e-7)
        # float32 would give ~0.1 with significant error
        # float64 should give ~0.1 with much less error
        assert abs(npu.facc - 0.1) < 1e-5


class TestFgelu:
    """fgelu function accuracy tests."""

    def test_zero(self) -> None:
        """GELU(0) = 0."""
        assert fgelu(0.0) == 0.0

    def test_positive(self) -> None:
        """GELU(2.0) ≈ 1.9545."""
        result = fgelu(2.0)
        assert abs(result - 1.9545) < 0.001

    def test_negative(self) -> None:
        """GELU(-2.0) ≈ -0.0455."""
        result = fgelu(-2.0)
        assert abs(result - (-0.0455)) < 0.001

    def test_large_positive(self) -> None:
        """GELU(x) ≈ x for large positive x."""
        result = fgelu(10.0)
        assert abs(result - 10.0) < 0.001

    def test_large_negative(self) -> None:
        """GELU(x) ≈ 0 for large negative x."""
        result = fgelu(-10.0)
        assert abs(result) < 0.001

    def test_one(self) -> None:
        """GELU(1.0) ≈ 0.8413."""
        result = fgelu(1.0)
        expected = 0.5 * 1.0 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))
        assert abs(result - expected) < 1e-10
