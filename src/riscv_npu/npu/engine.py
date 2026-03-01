"""NPU compute engine: accumulator, GELU table, vector registers."""

import math
from dataclasses import dataclass, field


def _make_vregs() -> list[list[int]]:
    """Create 4 vector registers, each containing 4 zeroed int8 values."""
    return [[0, 0, 0, 0] for _ in range(4)]


@dataclass
class NpuState:
    """NPU internal state: 64-bit accumulator and 4 vector registers.

    The accumulator is stored as two 32-bit halves (acc_lo, acc_hi).
    Vector registers each hold 4 int8 values (-128..127).
    """

    acc_lo: int = 0
    acc_hi: int = 0
    vreg: list[list[int]] = field(default_factory=_make_vregs)


def acc_get64(state: NpuState) -> int:
    """Read the full 64-bit accumulator value (signed).

    Combines acc_hi and acc_lo into a single Python int.
    The result is sign-extended from 64 bits.
    """
    val = (state.acc_hi << 32) | state.acc_lo
    if val >= (1 << 63):
        val -= 1 << 64
    return val


def acc_set64(state: NpuState, value: int) -> None:
    """Write a 64-bit value into the accumulator halves.

    Masks to 64 bits, then splits into acc_lo and acc_hi.
    """
    value = value & 0xFFFFFFFFFFFFFFFF
    state.acc_lo = value & 0xFFFFFFFF
    state.acc_hi = (value >> 32) & 0xFFFFFFFF


def acc_add(state: NpuState, value: int) -> None:
    """Add a signed value to the 64-bit accumulator.

    Reads current accumulator, adds value, writes back.
    Wraps on overflow (64-bit modular arithmetic).
    """
    current = acc_get64(state)
    acc_set64(state, current + value)


def acc_reset(state: NpuState) -> int:
    """Reset the accumulator to 0 and return the old acc_lo value.

    This implements the RSTACC instruction semantics:
    rd = acc_lo, then acc = 0.
    """
    old_lo = state.acc_lo
    state.acc_lo = 0
    state.acc_hi = 0
    return old_lo


def _gelu_int8(x: int) -> int:
    """Compute GELU for a single int8 value.

    Uses the exact formula: gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2))).
    Input and output are int8 scale values. We treat the input as a
    fixed-point value scaled by some factor; for a lookup table mapping
    int8 -> int8, we normalize to [-1, 1] range approximately.

    The input x is in [-128, 127]. We scale it to a reasonable float
    range: x_float = x / 32.0 (so int8 range maps to roughly [-4, 4]).
    Then apply GELU and scale back.
    """
    x_float = x / 32.0
    gelu_val = 0.5 * x_float * (1.0 + math.erf(x_float / math.sqrt(2.0)))
    # Scale back and clamp to int8
    result = round(gelu_val * 32.0)
    return max(-128, min(127, result))


def build_gelu_table() -> list[int]:
    """Precompute GELU lookup table for all 256 int8 inputs.

    Index i maps to int8 value (i - 128), so index 0 = -128, index 255 = 127.
    Each entry is the int8 GELU output.

    Returns:
        List of 256 int8 values.
    """
    table = []
    for i in range(256):
        x = i - 128  # Convert index to signed int8
        table.append(_gelu_int8(x))
    return table


# Module-level precomputed table
GELU_TABLE: list[int] = build_gelu_table()


# ---------------------------------------------------------------------------
# Q16.16 fixed-point helpers
# ---------------------------------------------------------------------------

# Q16.16: 1.0 is represented as 65536 (0x00010000)
Q16_ONE: int = 1 << 16


def _to_signed32(val: int) -> int:
    """Interpret a 32-bit unsigned value as a signed Python int."""
    val = val & 0xFFFFFFFF
    return val - 0x100000000 if val >= 0x80000000 else val


def exp_q16_16(x: int) -> int:
    """Compute exp(x) where x is Q16.16 fixed-point signed. Returns Q16.16.

    Uses Python's math.exp on the float equivalent and converts back.
    Input is sign-extended from 32 bits. The result is clamped to
    the positive Q16.16 range [0, 0x7FFFFFFF].

    For softmax, inputs are in roughly [-8.0, 0.0] after max subtraction.
    exp(0) = 1.0 = 65536. exp(-8) ~ 22.

    Args:
        x: Q16.16 fixed-point value (signed 32-bit).

    Returns:
        Q16.16 fixed-point result, always >= 0.
    """
    x_signed = _to_signed32(x)
    x_float = x_signed / Q16_ONE
    # Clamp input to prevent overflow in exp()
    x_float = max(-20.0, min(20.0, x_float))
    result_float = math.exp(x_float)
    result = round(result_float * Q16_ONE)
    # Clamp to positive 32-bit range
    return max(0, min(0x7FFFFFFF, result)) & 0xFFFFFFFF


def rsqrt_q16_16(x: int) -> int:
    """Compute 1/sqrt(x) where x is Q16.16 fixed-point. Returns Q16.16.

    Input must be positive. If x <= 0, returns a large value (saturate).
    Uses Python's math.sqrt for reference accuracy.

    Args:
        x: Q16.16 fixed-point value (positive).

    Returns:
        Q16.16 fixed-point result representing 1/sqrt(x).
    """
    x_signed = _to_signed32(x)
    if x_signed <= 0:
        # Return max Q16.16 value as saturation
        return 0x7FFFFFFF
    x_float = x_signed / Q16_ONE
    result_float = 1.0 / math.sqrt(x_float)
    result = round(result_float * Q16_ONE)
    return max(0, min(0x7FFFFFFF, result)) & 0xFFFFFFFF
