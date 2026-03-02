"""Tests for memory bus routing."""

import pytest

from riscv_npu.memory.bus import MemoryBus


class MockDevice:
    """Simple mock device that stores bytes in a dict keyed by address."""

    def __init__(self) -> None:
        self.data: dict[int, int] = {}

    def read8(self, addr: int) -> int:
        return self.data.get(addr, 0)

    def write8(self, addr: int, value: int) -> None:
        self.data[addr] = value & 0xFF


class TestMemoryBus:
    """Tests for MemoryBus device registration and address routing."""

    def test_register_and_read8(self) -> None:
        """Registering a device and reading from it returns device data."""
        bus = MemoryBus()
        dev = MockDevice()
        dev.data[0x1000] = 0xAB
        bus.register(0x1000, 16, dev)
        assert bus.read8(0x1000) == 0xAB

    def test_register_and_write8(self) -> None:
        """Writing through the bus reaches the registered device."""
        bus = MemoryBus()
        dev = MockDevice()
        bus.register(0x2000, 16, dev)
        bus.write8(0x2000, 0x42)
        assert dev.data[0x2000] == 0x42

    def test_read16_little_endian(self) -> None:
        """read16 composes two byte reads in little-endian order."""
        bus = MemoryBus()
        dev = MockDevice()
        dev.data[0x1000] = 0x34  # LSB
        dev.data[0x1001] = 0x12  # MSB
        bus.register(0x1000, 16, dev)
        assert bus.read16(0x1000) == 0x1234

    def test_read32_little_endian(self) -> None:
        """read32 composes four byte reads in little-endian order."""
        bus = MemoryBus()
        dev = MockDevice()
        dev.data[0x1000] = 0x78
        dev.data[0x1001] = 0x56
        dev.data[0x1002] = 0x34
        dev.data[0x1003] = 0x12
        bus.register(0x1000, 16, dev)
        assert bus.read32(0x1000) == 0x12345678

    def test_write16_little_endian(self) -> None:
        """write16 decomposes into two byte writes in little-endian order."""
        bus = MemoryBus()
        dev = MockDevice()
        bus.register(0x1000, 16, dev)
        bus.write16(0x1000, 0xBEEF)
        assert dev.data[0x1000] == 0xEF  # LSB
        assert dev.data[0x1001] == 0xBE  # MSB

    def test_write32_little_endian(self) -> None:
        """write32 decomposes into four byte writes in little-endian order."""
        bus = MemoryBus()
        dev = MockDevice()
        bus.register(0x1000, 16, dev)
        bus.write32(0x1000, 0xDEADBEEF)
        assert dev.data[0x1000] == 0xEF
        assert dev.data[0x1001] == 0xBE
        assert dev.data[0x1002] == 0xAD
        assert dev.data[0x1003] == 0xDE

    def test_unmapped_address_raises(self) -> None:
        """Accessing an unregistered address raises MemoryError."""
        bus = MemoryBus()
        with pytest.raises(MemoryError, match="Unmapped address"):
            bus.read8(0x9999)

    def test_multiple_devices(self) -> None:
        """Two devices at different ranges both route correctly."""
        bus = MemoryBus()
        dev_a = MockDevice()
        dev_b = MockDevice()
        bus.register(0x1000, 16, dev_a)
        bus.register(0x2000, 16, dev_b)

        bus.write8(0x1000, 0xAA)
        bus.write8(0x2000, 0xBB)

        assert bus.read8(0x1000) == 0xAA
        assert bus.read8(0x2000) == 0xBB
        # Verify isolation -- dev_a didn't get dev_b's write
        assert dev_a.data.get(0x2000) is None

    def test_load_segment(self) -> None:
        """load_segment writes bytes sequentially via write8."""
        bus = MemoryBus()
        dev = MockDevice()
        bus.register(0x1000, 256, dev)
        bus.load_segment(0x1000, b"\x01\x02\x03\x04")
        assert dev.data[0x1000] == 0x01
        assert dev.data[0x1001] == 0x02
        assert dev.data[0x1002] == 0x03
        assert dev.data[0x1003] == 0x04

    def test_overlapping_registration_raises(self) -> None:
        """Registering overlapping address ranges raises ValueError."""
        bus = MemoryBus()
        dev_a = MockDevice()
        dev_b = MockDevice()
        bus.register(0x1000, 16, dev_a)
        with pytest.raises(ValueError, match="overlaps"):
            bus.register(0x1008, 16, dev_b)

    def test_get_device_data_returns_buffer_and_base(self) -> None:
        """get_device_data returns the device's raw buffer and base address."""
        from riscv_npu.memory.ram import RAM

        bus = MemoryBus()
        ram = RAM(0x8000_0000, 1024)
        bus.register(0x8000_0000, 1024, ram)

        data, base = bus.get_device_data(0x8000_0000)
        assert data is ram._data
        assert base == 0x8000_0000

        # Works for addresses within the device range, not just the base
        data2, base2 = bus.get_device_data(0x8000_0100)
        assert data2 is ram._data
        assert base2 == 0x8000_0000

    def test_get_device_data_unmapped_address_raises(self) -> None:
        """get_device_data raises MemoryError for unmapped addresses."""
        bus = MemoryBus()
        with pytest.raises(MemoryError, match="Unmapped address"):
            bus.get_device_data(0x9999)
