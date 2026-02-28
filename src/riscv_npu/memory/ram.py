"""RAM: bytearray-backed memory with read/write operations."""


class RAM:
    """Byte-addressable RAM with a base address and fixed size."""

    def __init__(self, base: int, size: int) -> None:
        self.base = base
        self.size = size
        self._data = bytearray(size)

    def _offset(self, addr: int, width: int) -> int:
        """Translate absolute address to internal offset, checking bounds."""
        offset = addr - self.base
        if offset < 0 or offset + width > self.size:
            raise MemoryError(f"Access out of bounds: 0x{addr:08X}")
        return offset

    def read8(self, addr: int) -> int:
        """Read an unsigned byte."""
        off = self._offset(addr, 1)
        return self._data[off]

    def read16(self, addr: int) -> int:
        """Read an unsigned 16-bit halfword (little-endian)."""
        off = self._offset(addr, 2)
        return int.from_bytes(self._data[off:off + 2], "little")

    def read32(self, addr: int) -> int:
        """Read an unsigned 32-bit word (little-endian)."""
        off = self._offset(addr, 4)
        return int.from_bytes(self._data[off:off + 4], "little")

    def write8(self, addr: int, value: int) -> None:
        """Write a byte."""
        off = self._offset(addr, 1)
        self._data[off] = value & 0xFF

    def write16(self, addr: int, value: int) -> None:
        """Write a 16-bit halfword (little-endian)."""
        off = self._offset(addr, 2)
        self._data[off:off + 2] = (value & 0xFFFF).to_bytes(2, "little")

    def write32(self, addr: int, value: int) -> None:
        """Write a 32-bit word (little-endian)."""
        off = self._offset(addr, 4)
        self._data[off:off + 4] = (value & 0xFFFFFFFF).to_bytes(4, "little")
