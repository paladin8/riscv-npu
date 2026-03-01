"""Float register file and FPU state for RV32F (single-precision floating-point)."""

import struct
from dataclasses import dataclass, field

# CSR addresses for floating-point state
CSR_FFLAGS = 0x001
CSR_FRM = 0x002
CSR_FCSR = 0x003


class FRegisterFile:
    """32 single-precision float registers (f0-f31), stored as raw IEEE 754 bits.

    Unlike the integer register file, f0 is NOT hardwired to zero.
    """

    def __init__(self) -> None:
        self._regs: list[int] = [0] * 32

    def read_bits(self, index: int) -> int:
        """Read raw 32-bit IEEE 754 representation."""
        return self._regs[index]

    def write_bits(self, index: int, value: int) -> None:
        """Write raw 32-bit IEEE 754 representation."""
        self._regs[index] = value & 0xFFFFFFFF

    def read_float(self, index: int) -> float:
        """Read register as a Python float."""
        bits = self._regs[index]
        return struct.unpack('<f', struct.pack('<I', bits))[0]

    def write_float(self, index: int, value: float) -> None:
        """Write a Python float to a register (rounded to single-precision)."""
        bits = struct.unpack('<I', struct.pack('<f', value))[0]
        self._regs[index] = bits


@dataclass
class FpuState:
    """Floating-point unit state: registers + control/status register."""

    fregs: FRegisterFile = field(default_factory=FRegisterFile)
    fcsr: int = 0  # fflags[4:0] + frm[7:5]

    @property
    def fflags(self) -> int:
        """Extract exception flags (bits 4:0 of fcsr)."""
        return self.fcsr & 0x1F

    @fflags.setter
    def fflags(self, value: int) -> None:
        """Set exception flags (bits 4:0 of fcsr)."""
        self.fcsr = (self.fcsr & ~0x1F) | (value & 0x1F)

    @property
    def frm(self) -> int:
        """Extract rounding mode (bits 7:5 of fcsr)."""
        return (self.fcsr >> 5) & 0x7

    @frm.setter
    def frm(self, value: int) -> None:
        """Set rounding mode (bits 7:5 of fcsr)."""
        self.fcsr = (self.fcsr & ~0xE0) | ((value & 0x7) << 5)

    def set_flags(
        self,
        nv: bool = False,
        dz: bool = False,
        of: bool = False,
        uf: bool = False,
        nx: bool = False,
    ) -> None:
        """OR sticky exception flags into fflags.

        Flags are sticky — once set, they remain set until explicitly cleared.

        Args:
            nv: Invalid operation.
            dz: Divide by zero.
            of: Overflow.
            uf: Underflow.
            nx: Inexact.
        """
        bits = 0
        if nv:
            bits |= 0x10
        if dz:
            bits |= 0x08
        if of:
            bits |= 0x04
        if uf:
            bits |= 0x02
        if nx:
            bits |= 0x01
        self.fcsr |= bits
