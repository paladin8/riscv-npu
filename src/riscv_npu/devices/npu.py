"""NPU device: memory-mapped control/status registers at 0x20000000.

Provides read-only access to NPU state for the TUI debugger and
diagnostic firmware. Writing to offset 0 resets the integer accumulator.
Writing to offset 0x18 resets the FP accumulator.

Register map:
    0x00-0x03: acc_lo (32-bit, read-only; write resets int accumulator)
    0x04-0x07: acc_hi (32-bit, read-only)
    0x08-0x0B: vreg[0] (4 bytes, packed int8)
    0x0C-0x0F: vreg[1] (4 bytes, packed int8)
    0x10-0x13: vreg[2] (4 bytes, packed int8)
    0x14-0x17: vreg[3] (4 bytes, packed int8)
    0x18-0x1B: facc_lo (32-bit, FP acc low bits; write resets FP accumulator)
    0x1C-0x1F: facc_hi (32-bit, FP acc high bits)
"""

from __future__ import annotations

import struct

from ..npu.engine import NpuState, acc_reset

NPU_BASE = 0x20000000
NPU_SIZE = 0x100  # 256 bytes reserved


class NpuDevice:
    """Memory-mapped NPU status registers.

    Thin wrapper around NpuState for bus-accessible reads.
    Delegates all state to the NpuState instance shared with
    the CPU's NPU instruction executor.
    """

    def __init__(self, npu_state: NpuState, base: int = NPU_BASE) -> None:
        self._state = npu_state
        self._base = base

    def read8(self, addr: int) -> int:
        """Read a byte from NPU status registers.

        Args:
            addr: Absolute address.

        Returns:
            Byte value at the given register offset.
        """
        offset = addr - self._base
        if 0 <= offset < 4:
            # acc_lo: little-endian byte
            return (self._state.acc_lo >> (offset * 8)) & 0xFF
        elif 4 <= offset < 8:
            # acc_hi: little-endian byte
            return (self._state.acc_hi >> ((offset - 4) * 8)) & 0xFF
        elif 8 <= offset < 24:
            # vreg[0..3]: 4 bytes each, packed as unsigned bytes
            vreg_offset = offset - 8
            vreg_idx = vreg_offset // 4
            byte_idx = vreg_offset % 4
            val = self._state.vreg[vreg_idx][byte_idx]
            return val & 0xFF
        elif 24 <= offset < 32:
            # facc: float64 as two little-endian 32-bit words (IEEE 754 double)
            facc_bytes = struct.pack('<d', self._state.facc)
            byte_idx = offset - 24
            return facc_bytes[byte_idx]
        return 0

    def write8(self, addr: int, value: int) -> None:
        """Write a byte to NPU control registers.

        Writing any value to offset 0x00 resets the accumulator.
        All other writes are ignored.

        Args:
            addr: Absolute address.
            value: Byte value to write.
        """
        offset = addr - self._base
        if offset == 0:
            acc_reset(self._state)
        elif offset == 0x18:
            self._state.facc = 0.0
