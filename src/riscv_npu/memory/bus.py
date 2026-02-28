"""Memory bus: routes address ranges to devices."""

from dataclasses import dataclass

from .device import Device


@dataclass
class DeviceMapping:
    """A device registered on the bus with its address range."""

    base: int
    size: int
    device: Device


class MemoryBus:
    """Routes memory accesses to registered devices by address range.

    Devices are registered with a base address and size. The bus
    dispatches read/write operations to the device that owns the
    target address. Multi-byte operations are composed from byte-level
    device reads/writes in little-endian order.
    """

    def __init__(self) -> None:
        self._devices: list[DeviceMapping] = []

    def register(self, base: int, size: int, device: Device) -> None:
        """Register a device at the given address range.

        Args:
            base: Start address of the device's address space.
            size: Number of bytes the device occupies.
            device: The device to register.

        Raises:
            ValueError: If the new range overlaps an existing device.
        """
        new_end = base + size
        for mapping in self._devices:
            existing_end = mapping.base + mapping.size
            if base < existing_end and new_end > mapping.base:
                raise ValueError(
                    f"Address range [0x{base:08X}, 0x{new_end:08X}) overlaps "
                    f"existing device at [0x{mapping.base:08X}, 0x{existing_end:08X})"
                )
        self._devices.append(DeviceMapping(base=base, size=size, device=device))

    def _find_device(self, addr: int, width: int) -> DeviceMapping:
        """Find the device that owns the address range [addr, addr+width).

        Args:
            addr: Start address of the access.
            width: Number of bytes being accessed.

        Returns:
            The DeviceMapping that covers the entire range.

        Raises:
            MemoryError: If no device covers the address range.
        """
        for mapping in self._devices:
            if (addr >= mapping.base
                    and addr + width <= mapping.base + mapping.size):
                return mapping
        raise MemoryError(f"Unmapped address: 0x{addr:08X} (width={width})")

    def read8(self, addr: int) -> int:
        """Read an unsigned byte from the device at addr."""
        mapping = self._find_device(addr, 1)
        return mapping.device.read8(addr)

    def read16(self, addr: int) -> int:
        """Read an unsigned 16-bit halfword (little-endian)."""
        mapping = self._find_device(addr, 2)
        b0 = mapping.device.read8(addr)
        b1 = mapping.device.read8(addr + 1)
        return b0 | (b1 << 8)

    def read32(self, addr: int) -> int:
        """Read an unsigned 32-bit word (little-endian)."""
        mapping = self._find_device(addr, 4)
        b0 = mapping.device.read8(addr)
        b1 = mapping.device.read8(addr + 1)
        b2 = mapping.device.read8(addr + 2)
        b3 = mapping.device.read8(addr + 3)
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)

    def write8(self, addr: int, value: int) -> None:
        """Write a byte to the device at addr."""
        mapping = self._find_device(addr, 1)
        mapping.device.write8(addr, value)

    def write16(self, addr: int, value: int) -> None:
        """Write a 16-bit halfword (little-endian)."""
        mapping = self._find_device(addr, 2)
        mapping.device.write8(addr, value & 0xFF)
        mapping.device.write8(addr + 1, (value >> 8) & 0xFF)

    def write32(self, addr: int, value: int) -> None:
        """Write a 32-bit word (little-endian)."""
        mapping = self._find_device(addr, 4)
        mapping.device.write8(addr, value & 0xFF)
        mapping.device.write8(addr + 1, (value >> 8) & 0xFF)
        mapping.device.write8(addr + 2, (value >> 16) & 0xFF)
        mapping.device.write8(addr + 3, (value >> 24) & 0xFF)

    def load_segment(self, addr: int, data: bytes) -> None:
        """Bulk-load bytes into memory at an absolute address.

        Writes each byte individually via write8, routing through
        the bus to the appropriate device.

        Args:
            addr: Absolute start address for the load.
            data: Raw bytes to copy into memory.
        """
        for i, byte in enumerate(data):
            self.write8(addr + i, byte)
