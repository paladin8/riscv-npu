"""Memory bus: routes address ranges to devices."""

from __future__ import annotations

from typing import Any

from .device import Device


class DeviceMapping:
    """A device registered on the bus with its address range and capabilities."""

    __slots__ = ("base", "size", "device",
                 "has_read16", "has_read32",
                 "has_write16", "has_write32",
                 "has_load_segment")

    def __init__(self, base: int, size: int, device: Device) -> None:
        self.base = base
        self.size = size
        self.device: Any = device
        self.has_read16: bool = hasattr(device, "read16")
        self.has_read32: bool = hasattr(device, "read32")
        self.has_write16: bool = hasattr(device, "write16")
        self.has_write32: bool = hasattr(device, "write32")
        self.has_load_segment: bool = hasattr(device, "load_segment")


class MemoryBus:
    """Routes memory accesses to registered devices by address range.

    Devices are registered with a base address and size. The bus
    dispatches read/write operations to the device that owns the
    target address. When a device provides native multi-byte methods
    (read16, read32, etc.), the bus delegates directly; otherwise it
    composes multi-byte operations from byte-level reads/writes.
    """

    def __init__(self) -> None:
        self._devices: list[DeviceMapping] = []
        self._last_hit: DeviceMapping | None = None

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

        Uses a last-hit cache to skip the linear scan when consecutive
        accesses target the same device (the common case for RAM).

        Args:
            addr: Start address of the access.
            width: Number of bytes being accessed.

        Returns:
            The DeviceMapping that covers the entire range.

        Raises:
            MemoryError: If no device covers the address range.
        """
        hit = self._last_hit
        if (hit is not None
                and addr >= hit.base
                and addr + width <= hit.base + hit.size):
            return hit
        for mapping in self._devices:
            if (addr >= mapping.base
                    and addr + width <= mapping.base + mapping.size):
                self._last_hit = mapping
                return mapping
        raise MemoryError(f"Unmapped address: 0x{addr:08X} (width={width})")

    def read8(self, addr: int) -> int:
        """Read an unsigned byte from the device at addr."""
        mapping = self._find_device(addr, 1)
        return mapping.device.read8(addr)

    def read16(self, addr: int) -> int:
        """Read an unsigned 16-bit halfword (little-endian)."""
        mapping = self._find_device(addr, 2)
        if mapping.has_read16:
            return mapping.device.read16(addr)
        b0 = mapping.device.read8(addr)
        b1 = mapping.device.read8(addr + 1)
        return b0 | (b1 << 8)

    def read32(self, addr: int) -> int:
        """Read an unsigned 32-bit word (little-endian)."""
        mapping = self._find_device(addr, 4)
        if mapping.has_read32:
            return mapping.device.read32(addr)
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
        if mapping.has_write16:
            mapping.device.write16(addr, value)
        else:
            mapping.device.write8(addr, value & 0xFF)
            mapping.device.write8(addr + 1, (value >> 8) & 0xFF)

    def write32(self, addr: int, value: int) -> None:
        """Write a 32-bit word (little-endian)."""
        mapping = self._find_device(addr, 4)
        if mapping.has_write32:
            mapping.device.write32(addr, value)
        else:
            mapping.device.write8(addr, value & 0xFF)
            mapping.device.write8(addr + 1, (value >> 8) & 0xFF)
            mapping.device.write8(addr + 2, (value >> 16) & 0xFF)
            mapping.device.write8(addr + 3, (value >> 24) & 0xFF)

    def get_device_data(self, addr: int) -> tuple[bytearray, int]:
        """Get (raw_buffer, device_base_addr) for the device at addr.

        Enables direct buffer access for bulk operations like NPU vector
        instructions, bypassing per-element bus dispatch.

        Args:
            addr: Any address within the target device's range.

        Returns:
            Tuple of (device's backing bytearray, device base address).

        Raises:
            MemoryError: If no device covers the address.
            AttributeError: If the device has no _data buffer.
        """
        mapping = self._find_device(addr, 1)
        return mapping.device._data, mapping.base

    def load_segment(self, addr: int, data: bytes) -> None:
        """Bulk-load bytes into memory at an absolute address.

        Delegates to the device's load_segment() if available,
        otherwise writes each byte individually via write8.

        Args:
            addr: Absolute start address for the load.
            data: Raw bytes to copy into memory.
        """
        mapping = self._find_device(addr, len(data))
        if mapping.has_load_segment:
            mapping.device.load_segment(addr, data)
        else:
            for i, byte in enumerate(data):
                self.write8(addr + i, byte)
