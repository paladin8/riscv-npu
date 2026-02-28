"""Base protocol for memory-mapped devices."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Device(Protocol):
    """Protocol for memory-mapped devices.

    Devices implement byte-level read/write. The MemoryBus composes
    multi-byte accesses from these byte operations (little-endian).

    Addresses are absolute -- the device is responsible for translating
    to its internal offset.
    """

    def read8(self, addr: int) -> int:
        """Read a single byte at the given absolute address."""
        ...

    def write8(self, addr: int, value: int) -> None:
        """Write a single byte at the given absolute address."""
        ...
