# cython: language_level=3
"""Cython-accelerated NPU vector kernels.

Operates directly on a RAM bytearray via typed memoryviews, bypassing
the per-element bus dispatch. Each function takes the raw buffer plus
integer offsets, reads/writes data in place, and returns the result.

Build with: cythonize -i src/riscv_npu/npu/_accel.pyx
"""

cimport cython
from libc.math cimport exp as c_exp, erf as c_erf, INFINITY, NAN
from libc.string cimport memcpy
from libc.stdint cimport int8_t, int32_t, uint32_t


# ---------------------------------------------------------------------------
# Integer NPU kernels
# ---------------------------------------------------------------------------


@cython.boundscheck(False)
@cython.wraparound(False)
def vmac_int8(const unsigned char[:] data, int off_a, int off_b, int n):
    """Dot product of two int8 arrays in memory.

    Args:
        data: Raw RAM buffer.
        off_a: Byte offset of first int8 array.
        off_b: Byte offset of second int8 array.
        n: Number of elements.

    Returns:
        Sum of int8[a_i] * int8[b_i] as a 64-bit integer.
    """
    cdef long long acc = 0
    cdef int i
    cdef signed char a, b
    for i in range(n):
        a = <signed char>data[off_a + i]
        b = <signed char>data[off_b + i]
        acc += <long long>a * <long long>b
    return acc


@cython.boundscheck(False)
@cython.wraparound(False)
def vmul_int8(unsigned char[:] data, int src, int dst, int n, int scale):
    """Scale int8 vector by Q16.16 factor, writing results in place.

    For each element i:
        dst[i] = clamp((src_int8[i] * scale) >> 16, -128, 127)

    Args:
        data: Raw RAM buffer (writable).
        src: Byte offset of source int8 array.
        dst: Byte offset of destination int8 array.
        n: Number of elements.
        scale: Q16.16 signed scale factor.
    """
    cdef int i
    cdef signed char src_val
    cdef long long product
    cdef int result
    for i in range(n):
        src_val = <signed char>data[src + i]
        product = <long long>src_val * <long long>scale
        result = <int>(product >> 16)
        if result < -128:
            result = -128
        elif result > 127:
            result = 127
        data[dst + i] = <unsigned char>(<signed char>result)


@cython.boundscheck(False)
@cython.wraparound(False)
def vreduce_int32(const unsigned char[:] data, int src, int n):
    """Sum of n signed int32 values from memory.

    Args:
        data: Raw RAM buffer.
        src: Byte offset of int32 array.
        n: Number of int32 elements.

    Returns:
        Sum as a 64-bit integer.
    """
    cdef long long total = 0
    cdef int i
    cdef int32_t val
    cdef int off
    for i in range(n):
        off = src + i * 4
        # Read little-endian int32 byte-by-byte to avoid alignment issues
        val = (<int32_t>data[off]
               | (<int32_t>data[off + 1] << 8)
               | (<int32_t>data[off + 2] << 16)
               | (<int32_t>data[off + 3] << 24))
        total += val
    return total


@cython.boundscheck(False)
@cython.wraparound(False)
def vmax_int32(const unsigned char[:] data, int src, int n):
    """Maximum of n signed int32 values from memory.

    Args:
        data: Raw RAM buffer.
        src: Byte offset of int32 array.
        n: Number of int32 elements.

    Returns:
        Maximum signed int32 value. Returns -2147483648 if n == 0.
    """
    cdef int32_t max_val = <int32_t>(-2147483648)
    cdef int i
    cdef int32_t val
    cdef int off
    if n == 0:
        return <int>max_val
    for i in range(n):
        off = src + i * 4
        val = (<int32_t>data[off]
               | (<int32_t>data[off + 1] << 8)
               | (<int32_t>data[off + 2] << 16)
               | (<int32_t>data[off + 3] << 24))
        if val > max_val:
            max_val = val
    return <int>max_val


@cython.boundscheck(False)
@cython.wraparound(False)
def vexp_int32(unsigned char[:] data, int src, int dst, int n):
    """Vectorized exp over int32 Q16.16 fixed-point array.

    For each element i:
        dst[i] = exp_q16_16(src[i])

    Uses the same algorithm as engine.py:exp_q16_16():
    - Sign-extend 32-bit input
    - Convert to float: x_float = x_signed / 65536.0
    - Clamp to [-20.0, 20.0]
    - Compute math.exp(x_float)
    - result = round(exp_val * 65536)
    - Clamp to [0, 0x7FFFFFFF]

    Args:
        data: Raw RAM buffer (writable).
        src: Byte offset of source int32 array.
        dst: Byte offset of destination int32 array.
        n: Number of int32 elements.
    """
    cdef int i
    cdef int32_t val
    cdef int src_off, dst_off
    cdef double x_float, result_float
    cdef long long result_int
    cdef uint32_t result
    for i in range(n):
        src_off = src + i * 4
        dst_off = dst + i * 4
        # Read little-endian signed int32
        val = (<int32_t>data[src_off]
               | (<int32_t>data[src_off + 1] << 8)
               | (<int32_t>data[src_off + 2] << 16)
               | (<int32_t>data[src_off + 3] << 24))
        # Convert Q16.16 to float
        x_float = <double>val / 65536.0
        # Clamp input
        if x_float < -20.0:
            x_float = -20.0
        elif x_float > 20.0:
            x_float = 20.0
        result_float = c_exp(x_float)
        # Convert back to Q16.16
        result_int = <long long>(result_float * 65536.0 + 0.5)
        # Clamp to [0, 0x7FFFFFFF]
        if result_int < 0:
            result_int = 0
        elif result_int > 0x7FFFFFFF:
            result_int = 0x7FFFFFFF
        result = <uint32_t>result_int
        # Write little-endian uint32
        data[dst_off] = <unsigned char>(result & 0xFF)
        data[dst_off + 1] = <unsigned char>((result >> 8) & 0xFF)
        data[dst_off + 2] = <unsigned char>((result >> 16) & 0xFF)
        data[dst_off + 3] = <unsigned char>((result >> 24) & 0xFF)


# ---------------------------------------------------------------------------
# Float NPU kernels
# ---------------------------------------------------------------------------

cdef inline float _read_f32(const unsigned char[:] data, int off):
    """Read a float32 from memory at the given byte offset (little-endian)."""
    cdef uint32_t bits
    cdef float result
    bits = (<uint32_t>data[off]
            | (<uint32_t>data[off + 1] << 8)
            | (<uint32_t>data[off + 2] << 16)
            | (<uint32_t>data[off + 3] << 24))
    memcpy(&result, &bits, 4)
    return result


cdef inline void _write_f32(unsigned char[:] data, int off, float val):
    """Write a float32 to memory at the given byte offset (little-endian)."""
    cdef uint32_t bits
    memcpy(&bits, &val, 4)
    data[off] = <unsigned char>(bits & 0xFF)
    data[off + 1] = <unsigned char>((bits >> 8) & 0xFF)
    data[off + 2] = <unsigned char>((bits >> 16) & 0xFF)
    data[off + 3] = <unsigned char>((bits >> 24) & 0xFF)


@cython.boundscheck(False)
@cython.wraparound(False)
def fvmac_f32(const unsigned char[:] data, int off_a, int off_b, int n):
    """Dot product of two f32 arrays, with double-precision accumulation.

    Args:
        data: Raw RAM buffer.
        off_a: Byte offset of first f32 array.
        off_b: Byte offset of second f32 array.
        n: Number of elements.

    Returns:
        Dot product as a double.
    """
    cdef double acc = 0.0
    cdef int i
    cdef float a, b
    for i in range(n):
        a = _read_f32(data, off_a + i * 4)
        b = _read_f32(data, off_b + i * 4)
        acc += <double>a * <double>b
    return acc


@cython.boundscheck(False)
@cython.wraparound(False)
def fvmul_f32(unsigned char[:] data, int src, int dst, int n, unsigned int scale_bits):
    """Scale float32 array by a float32 factor given as IEEE bits.

    For each element i:
        dst[i] = src[i] * scale

    The scale factor is provided as uint32 IEEE 754 bits (to match the
    pure-Python path which rounds facc to float32 before scaling).

    Args:
        data: Raw RAM buffer (writable).
        src: Byte offset of source f32 array.
        dst: Byte offset of destination f32 array.
        n: Number of elements.
        scale_bits: IEEE 754 float32 bits for the scale factor.
    """
    cdef float scale
    cdef uint32_t sb = scale_bits
    memcpy(&scale, &sb, 4)
    cdef int i
    cdef float val, result
    cdef uint32_t result_bits
    for i in range(n):
        val = _read_f32(data, src + i * 4)
        result = val * scale
        _write_f32(data, dst + i * 4, result)


@cython.boundscheck(False)
@cython.wraparound(False)
def fvexp_f32(unsigned char[:] data, int src, int dst, int n):
    """Vectorized exp over float32 array.

    For each element i:
        dst[i] = exp(src[i])

    Handles NaN, +inf, -inf correctly:
    - NaN -> NaN
    - -inf -> 0.0
    - +inf -> +inf
    - overflow -> +inf

    Args:
        data: Raw RAM buffer (writable).
        src: Byte offset of source f32 array.
        dst: Byte offset of destination f32 array.
        n: Number of elements.
    """
    cdef int i
    cdef float val
    cdef double result_d
    cdef float result
    for i in range(n):
        val = _read_f32(data, src + i * 4)
        # NaN check: NaN != NaN
        if val != val:
            result = val  # NaN propagation
        elif val == -INFINITY:
            result = 0.0
        elif val == INFINITY:
            result = <float>INFINITY
        else:
            result_d = c_exp(<double>val)
            if result_d == INFINITY:
                result = <float>INFINITY
            else:
                result = <float>result_d
        _write_f32(data, dst + i * 4, result)


@cython.boundscheck(False)
@cython.wraparound(False)
def fvreduce_f32(const unsigned char[:] data, int src, int n):
    """Sum of n float32 values with double-precision accumulation.

    Args:
        data: Raw RAM buffer.
        src: Byte offset of f32 array.
        n: Number of elements.

    Returns:
        Sum as a double.
    """
    cdef double total = 0.0
    cdef int i
    cdef float val
    for i in range(n):
        val = _read_f32(data, src + i * 4)
        total += <double>val
    return total


@cython.boundscheck(False)
@cython.wraparound(False)
def fvmax_f32(const unsigned char[:] data, int src, int n):
    """Maximum of n float32 values.

    Returns -inf if n == 0. NaN elements are propagated (returns NaN).

    Args:
        data: Raw RAM buffer.
        src: Byte offset of f32 array.
        n: Number of elements.

    Returns:
        Maximum float32 value.
    """
    cdef float max_val = <float>(-INFINITY)
    cdef int i
    cdef float val
    if n == 0:
        return max_val
    for i in range(n):
        val = _read_f32(data, src + i * 4)
        # NaN check
        if val != val:
            return val
        if val > max_val:
            max_val = val
    return max_val


@cython.boundscheck(False)
@cython.wraparound(False)
def fvadd_f32(unsigned char[:] data, int src1, int src2, int n):
    """Elementwise add: dst[i] = src1[i] + src2[i], result at src2.

    Args:
        data: Raw RAM buffer (writable).
        src1: Byte offset of first f32 array.
        src2: Byte offset of second f32 array (also destination).
        n: Number of elements.
    """
    cdef int i
    cdef float a, b
    for i in range(n):
        a = _read_f32(data, src1 + i * 4)
        b = _read_f32(data, src2 + i * 4)
        _write_f32(data, src2 + i * 4, a + b)


@cython.boundscheck(False)
@cython.wraparound(False)
def fvsub_f32(unsigned char[:] data, int src1, int src2, int n):
    """Elementwise subtract: dst[i] = src1[i] - src2[i], result at src2.

    Args:
        data: Raw RAM buffer (writable).
        src1: Byte offset of first f32 array.
        src2: Byte offset of second f32 array (also destination).
        n: Number of elements.
    """
    cdef int i
    cdef float a, b
    for i in range(n):
        a = _read_f32(data, src1 + i * 4)
        b = _read_f32(data, src2 + i * 4)
        _write_f32(data, src2 + i * 4, a - b)


@cython.boundscheck(False)
@cython.wraparound(False)
def fvrelu_f32(unsigned char[:] data, int src, int dst, int n):
    """Vectorized ReLU: dst[i] = max(src[i], 0.0).

    Handles -0.0 -> +0.0 and NaN -> NaN.

    Args:
        data: Raw RAM buffer (writable).
        src: Byte offset of source f32 array.
        dst: Byte offset of destination f32 array.
        n: Number of elements.
    """
    cdef int i
    cdef float val
    cdef uint32_t bits
    for i in range(n):
        val = _read_f32(data, src + i * 4)
        # NaN check: NaN != NaN
        if val != val:
            _write_f32(data, dst + i * 4, val)
        elif val < 0.0:
            _write_f32(data, dst + i * 4, 0.0)
        else:
            # Check for -0.0: bit 31 set but value == 0.0
            memcpy(&bits, &val, 4)
            if bits == 0x80000000:
                _write_f32(data, dst + i * 4, 0.0)
            else:
                _write_f32(data, dst + i * 4, val)


cdef float _M_SQRT1_2 = 0.7071067811865475  # 1/sqrt(2)


@cython.boundscheck(False)
@cython.wraparound(False)
def fvgelu_f32(unsigned char[:] data, int src, int dst, int n):
    """Vectorized GELU: dst[i] = 0.5 * x * (1 + erf(x / sqrt(2))).

    Handles NaN -> NaN, +inf -> +inf, -inf -> 0.0.

    Args:
        data: Raw RAM buffer (writable).
        src: Byte offset of source f32 array.
        dst: Byte offset of destination f32 array.
        n: Number of elements.
    """
    cdef int i
    cdef float val
    cdef double x_d, result_d
    cdef float result
    for i in range(n):
        val = _read_f32(data, src + i * 4)
        # NaN check: NaN != NaN
        if val != val:
            result = val
        elif val == -INFINITY:
            result = 0.0
        elif val == INFINITY:
            result = <float>INFINITY
        else:
            x_d = <double>val
            result_d = 0.5 * x_d * (1.0 + c_erf(x_d * _M_SQRT1_2))
            result = <float>result_d
        _write_f32(data, dst + i * 4, result)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def fvdiv_f32(unsigned char[:] data, int src, int dst, int n, unsigned int scale_bits):
    """Divide float32 array by a float32 divisor given as IEEE bits.

    For each element i:
        dst[i] = src[i] / divisor

    The divisor is provided as uint32 IEEE 754 bits.

    Args:
        data: Raw RAM buffer (writable).
        src: Byte offset of source f32 array.
        dst: Byte offset of destination f32 array.
        n: Number of elements.
        scale_bits: IEEE 754 float32 bits for the divisor.
    """
    cdef float divisor
    cdef uint32_t sb = scale_bits
    memcpy(&divisor, &sb, 4)
    cdef int i
    cdef float val, result
    for i in range(n):
        val = _read_f32(data, src + i * 4)
        # C float division handles IEEE 754 correctly (inf, NaN, div-by-zero)
        result = val / divisor
        _write_f32(data, dst + i * 4, result)


@cython.boundscheck(False)
@cython.wraparound(False)
def fvsub_scalar_f32(unsigned char[:] data, int src, int dst, int n, unsigned int scalar_bits):
    """Subtract a float32 scalar from each element of a float32 array.

    For each element i:
        dst[i] = src[i] - scalar

    The scalar is provided as uint32 IEEE 754 bits.

    Args:
        data: Raw RAM buffer (writable).
        src: Byte offset of source f32 array.
        dst: Byte offset of destination f32 array.
        n: Number of elements.
        scalar_bits: IEEE 754 float32 bits for the scalar.
    """
    cdef float scalar
    cdef uint32_t sb = scalar_bits
    memcpy(&scalar, &sb, 4)
    cdef int i
    cdef float val, result
    for i in range(n):
        val = _read_f32(data, src + i * 4)
        result = val - scalar
        _write_f32(data, dst + i * 4, result)
