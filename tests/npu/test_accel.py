"""Tests for Cython-accelerated NPU vector kernels.

All tests are skipped if _accel.so is not compiled.
Each kernel is tested against known input/output values that match
the pure-Python implementations in instructions.py / fp_instructions.py.
"""

import math
import struct

import pytest

accel = pytest.importorskip("riscv_npu.npu._accel")


def _make_buf(size: int = 4096) -> bytearray:
    """Create a zeroed bytearray for use as fake RAM."""
    return bytearray(size)


def _write_int8(buf: bytearray, offset: int, values: list[int]) -> None:
    """Write signed int8 values into a buffer at the given offset."""
    for i, v in enumerate(values):
        buf[offset + i] = v & 0xFF


def _write_int32_le(buf: bytearray, offset: int, values: list[int]) -> None:
    """Write signed int32 values into a buffer in little-endian format."""
    for i, v in enumerate(values):
        struct.pack_into("<i", buf, offset + i * 4, v)


def _read_int32_le(buf: bytearray, offset: int) -> int:
    """Read a signed int32 from a buffer in little-endian format."""
    return struct.unpack_from("<i", buf, offset)[0]


def _read_uint32_le(buf: bytearray, offset: int) -> int:
    """Read an unsigned int32 from a buffer in little-endian format."""
    return struct.unpack_from("<I", buf, offset)[0]


def _write_f32_le(buf: bytearray, offset: int, values: list[float]) -> None:
    """Write float32 values into a buffer in little-endian format."""
    for i, v in enumerate(values):
        struct.pack_into("<f", buf, offset + i * 4, v)


def _read_f32_le(buf: bytearray, offset: int) -> float:
    """Read a float32 from a buffer in little-endian format."""
    return struct.unpack_from("<f", buf, offset)[0]


# ---------------------------------------------------------------------------
# Integer kernel tests
# ---------------------------------------------------------------------------


class TestVmacInt8:
    """Tests for vmac_int8 kernel."""

    def test_vmac_int8_basic(self) -> None:
        """Dot product of small known int8 arrays."""
        buf = _make_buf()
        # a = [1, 2, 3, 4], b = [5, 6, 7, 8]
        _write_int8(buf, 0, [1, 2, 3, 4])
        _write_int8(buf, 100, [5, 6, 7, 8])
        result = accel.vmac_int8(buf, 0, 100, 4)
        # 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert result == 70

    def test_vmac_int8_negative(self) -> None:
        """Dot product with signed int8 values (negative numbers)."""
        buf = _make_buf()
        # a = [-1, -2], b = [3, -4] => (-1)*3 + (-2)*(-4) = -3 + 8 = 5
        _write_int8(buf, 0, [-1, -2])
        _write_int8(buf, 100, [3, -4])
        result = accel.vmac_int8(buf, 0, 100, 2)
        assert result == 5

    def test_vmac_int8_empty(self) -> None:
        """Dot product with n=0 returns 0."""
        buf = _make_buf()
        result = accel.vmac_int8(buf, 0, 100, 0)
        assert result == 0


class TestVmulInt8:
    """Tests for vmul_int8 kernel."""

    def test_vmul_int8_basic(self) -> None:
        """Scale int8 vector and verify clamping."""
        buf = _make_buf()
        # src = [10, -10, 100, -100]
        _write_int8(buf, 0, [10, -10, 100, -100])
        # scale = 2.0 in Q16.16 = 2 * 65536 = 131072
        scale = 2 * 65536
        accel.vmul_int8(buf, 0, 100, 4, scale)
        # Expected: [20, -20, 127 (clamped), -128 (clamped)]
        assert (buf[100] - 256 if buf[100] >= 128 else buf[100]) == 20
        assert (buf[101] - 256 if buf[101] >= 128 else buf[101]) == -20
        assert (buf[102] - 256 if buf[102] >= 128 else buf[102]) == 127
        assert (buf[103] - 256 if buf[103] >= 128 else buf[103]) == -128


class TestVreduceInt32:
    """Tests for vreduce_int32 kernel."""

    def test_vreduce_int32_basic(self) -> None:
        """Sum of known int32 values."""
        buf = _make_buf()
        _write_int32_le(buf, 0, [10, -20, 30, -40])
        result = accel.vreduce_int32(buf, 0, 4)
        assert result == -20

    def test_vreduce_int32_empty(self) -> None:
        """Sum with n=0 returns 0."""
        buf = _make_buf()
        result = accel.vreduce_int32(buf, 0, 0)
        assert result == 0


class TestVmaxInt32:
    """Tests for vmax_int32 kernel."""

    def test_vmax_int32_basic(self) -> None:
        """Max of known int32 values."""
        buf = _make_buf()
        _write_int32_le(buf, 0, [-100, 50, 200, -1])
        result = accel.vmax_int32(buf, 0, 4)
        assert result == 200

    def test_vmax_int32_empty(self) -> None:
        """Max with n=0 returns minimum int32."""
        buf = _make_buf()
        result = accel.vmax_int32(buf, 0, 0)
        assert result == -2147483648


class TestVexpInt32:
    """Tests for vexp_int32 kernel."""

    def test_vexp_int32_basic(self) -> None:
        """Exp of known Q16.16 values matches Python reference."""
        from riscv_npu.npu.engine import exp_q16_16

        buf = _make_buf()
        # Test values: 0 (exp(0)=1.0), -65536 (exp(-1)), 65536 (exp(1))
        test_vals = [0, -65536, 65536]
        _write_int32_le(buf, 0, test_vals)
        accel.vexp_int32(buf, 0, 100, 3)

        for i, v in enumerate(test_vals):
            expected = exp_q16_16(v & 0xFFFFFFFF)
            actual = _read_uint32_le(buf, 100 + i * 4)
            assert actual == expected, f"vexp_int32({v}): {actual} != {expected}"


# ---------------------------------------------------------------------------
# Float kernel tests
# ---------------------------------------------------------------------------


class TestFvmacF32:
    """Tests for fvmac_f32 kernel."""

    def test_fvmac_f32_basic(self) -> None:
        """Dot product of known f32 arrays."""
        buf = _make_buf()
        _write_f32_le(buf, 0, [1.0, 2.0, 3.0])
        _write_f32_le(buf, 100, [4.0, 5.0, 6.0])
        result = accel.fvmac_f32(buf, 0, 100, 3)
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert abs(result - 32.0) < 1e-6

    def test_fvmac_f32_empty(self) -> None:
        """Dot product with n=0 returns 0.0."""
        buf = _make_buf()
        result = accel.fvmac_f32(buf, 0, 100, 0)
        assert result == 0.0


class TestFvmulF32:
    """Tests for fvmul_f32 kernel."""

    def test_fvmul_f32_basic(self) -> None:
        """Scale f32 array by a known factor."""
        buf = _make_buf()
        _write_f32_le(buf, 0, [1.0, 2.0, 3.0])
        # scale = 2.0, convert to IEEE bits
        scale_bits = struct.unpack("<I", struct.pack("<f", 2.0))[0]
        accel.fvmul_f32(buf, 0, 100, 3, scale_bits)
        assert abs(_read_f32_le(buf, 100) - 2.0) < 1e-6
        assert abs(_read_f32_le(buf, 104) - 4.0) < 1e-6
        assert abs(_read_f32_le(buf, 108) - 6.0) < 1e-6


class TestFvexpF32:
    """Tests for fvexp_f32 kernel."""

    def test_fvexp_f32_basic(self) -> None:
        """Exp of known float values."""
        buf = _make_buf()
        _write_f32_le(buf, 0, [0.0, 1.0, -1.0])
        accel.fvexp_f32(buf, 0, 100, 3)
        assert abs(_read_f32_le(buf, 100) - 1.0) < 1e-6
        assert abs(_read_f32_le(buf, 104) - math.e) < 1e-5
        assert abs(_read_f32_le(buf, 108) - (1.0 / math.e)) < 1e-6

    def test_fvexp_f32_nan(self) -> None:
        """NaN input produces NaN output."""
        buf = _make_buf()
        _write_f32_le(buf, 0, [float("nan")])
        accel.fvexp_f32(buf, 0, 100, 1)
        assert math.isnan(_read_f32_le(buf, 100))

    def test_fvexp_f32_inf(self) -> None:
        """Inf/-inf handled correctly."""
        buf = _make_buf()
        _write_f32_le(buf, 0, [float("-inf"), float("inf")])
        accel.fvexp_f32(buf, 0, 100, 2)
        assert _read_f32_le(buf, 100) == 0.0
        assert _read_f32_le(buf, 104) == float("inf")


class TestFvreduceF32:
    """Tests for fvreduce_f32 kernel."""

    def test_fvreduce_f32_basic(self) -> None:
        """Sum of known f32 values."""
        buf = _make_buf()
        _write_f32_le(buf, 0, [1.5, 2.5, 3.0, -1.0])
        result = accel.fvreduce_f32(buf, 0, 4)
        assert abs(result - 6.0) < 1e-6


class TestFvmaxF32:
    """Tests for fvmax_f32 kernel."""

    def test_fvmax_f32_basic(self) -> None:
        """Max of known f32 values."""
        buf = _make_buf()
        _write_f32_le(buf, 0, [-1.0, 3.5, 2.0, 0.5])
        result = accel.fvmax_f32(buf, 0, 4)
        assert abs(result - 3.5) < 1e-6

    def test_fvmax_f32_nan(self) -> None:
        """NaN element propagated as max."""
        buf = _make_buf()
        _write_f32_le(buf, 0, [1.0, float("nan"), 3.0])
        result = accel.fvmax_f32(buf, 0, 3)
        assert math.isnan(result)

    def test_fvmax_f32_empty(self) -> None:
        """Max with n=0 returns -inf."""
        buf = _make_buf()
        result = accel.fvmax_f32(buf, 0, 0)
        assert result == float("-inf")
