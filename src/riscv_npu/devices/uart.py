"""UART device: 16550-style memory-mapped serial I/O."""

import sys
from collections import deque
from typing import BinaryIO

# Default base address and size for the UART
UART_BASE = 0x10000000
UART_SIZE = 8  # 8 byte-wide registers

# Register offsets (relative to base)
_RBR = 0  # Receiver Buffer Register (read)
_THR = 0  # Transmitter Holding Register (write)
_LSR = 5  # Line Status Register (read)

# LSR bit masks
_LSR_DATA_READY = 0x01  # bit 0: data available in RBR
_LSR_THR_EMPTY = 0x20   # bit 5: THR is empty (ready to write)


class UART:
    """16550-style UART device for memory-mapped serial I/O.

    TX: write a byte to THR (offset 0) and it appears on the host
    output stream. RX: push bytes via push_rx(), then the emulated
    program reads them from RBR (offset 0). LSR (offset 5) reports
    status: bit 0 = data ready, bit 5 = THR empty (always set).

    ANSI escape codes pass through TX unmodified -- the host terminal
    interprets them.
    """

    def __init__(
        self,
        base: int = UART_BASE,
        tx_stream: BinaryIO | None = None,
    ) -> None:
        self._base = base
        self._tx = tx_stream if tx_stream is not None else sys.stdout.buffer
        self._rx_buf: deque[int] = deque()

    def read8(self, addr: int) -> int:
        """Read a byte from a UART register.

        - Offset 0 (RBR): returns next byte from RX buffer, or 0 if empty.
        - Offset 5 (LSR): returns line status bits.
        - Other offsets: returns 0.
        """
        offset = addr - self._base
        if offset == _RBR:
            if self._rx_buf:
                return self._rx_buf.popleft()
            return 0
        elif offset == _LSR:
            status = _LSR_THR_EMPTY  # THR is always ready
            if self._rx_buf:
                status |= _LSR_DATA_READY
            return status
        return 0

    def write8(self, addr: int, value: int) -> None:
        """Write a byte to a UART register.

        - Offset 0 (THR): sends the byte to the TX output stream.
        - Other offsets: no-op.
        """
        offset = addr - self._base
        if offset == _THR:
            self._tx.write(bytes([value & 0xFF]))
            self._tx.flush()

    def push_rx(self, data: bytes) -> None:
        """Push bytes into the RX buffer for the emulated program to read.

        Called by external code (CLI stdin reader or test harness).

        Args:
            data: Raw bytes to enqueue.
        """
        self._rx_buf.extend(data)
