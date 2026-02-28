"""Tests for the 16550-style UART device."""

import io

from riscv_npu.devices.uart import UART, UART_BASE


class TestUART:
    """Tests for UART TX, RX, and LSR register behavior."""

    def test_tx_write_byte(self) -> None:
        """Writing to THR sends the byte to the TX stream."""
        tx = io.BytesIO()
        uart = UART(tx_stream=tx)
        uart.write8(UART_BASE + 0, ord("A"))
        assert tx.getvalue() == b"A"

    def test_tx_multiple_bytes(self) -> None:
        """Multiple writes to THR appear in order on the TX stream."""
        tx = io.BytesIO()
        uart = UART(tx_stream=tx)
        for ch in b"Hello":
            uart.write8(UART_BASE + 0, ch)
        assert tx.getvalue() == b"Hello"

    def test_rx_empty_returns_zero(self) -> None:
        """Reading RBR with an empty RX buffer returns 0."""
        uart = UART(tx_stream=io.BytesIO())
        assert uart.read8(UART_BASE + 0) == 0

    def test_rx_push_and_read(self) -> None:
        """Bytes pushed via push_rx are read back in FIFO order from RBR."""
        uart = UART(tx_stream=io.BytesIO())
        uart.push_rx(b"\x41\x42\x43")
        assert uart.read8(UART_BASE + 0) == 0x41
        assert uart.read8(UART_BASE + 0) == 0x42
        assert uart.read8(UART_BASE + 0) == 0x43
        assert uart.read8(UART_BASE + 0) == 0  # empty

    def test_lsr_thr_empty(self) -> None:
        """LSR always reports bit 5 set (THR empty / ready to write)."""
        uart = UART(tx_stream=io.BytesIO())
        lsr = uart.read8(UART_BASE + 5)
        assert lsr & 0x20 == 0x20

    def test_lsr_data_ready_empty(self) -> None:
        """LSR bit 0 is clear when the RX buffer is empty."""
        uart = UART(tx_stream=io.BytesIO())
        lsr = uart.read8(UART_BASE + 5)
        assert lsr & 0x01 == 0

    def test_lsr_data_ready_has_data(self) -> None:
        """LSR bit 0 is set when the RX buffer has data."""
        uart = UART(tx_stream=io.BytesIO())
        uart.push_rx(b"X")
        lsr = uart.read8(UART_BASE + 5)
        assert lsr & 0x01 == 0x01

    def test_read_unknown_register(self) -> None:
        """Reading an unmapped UART register offset returns 0."""
        uart = UART(tx_stream=io.BytesIO())
        assert uart.read8(UART_BASE + 1) == 0
        assert uart.read8(UART_BASE + 3) == 0
        assert uart.read8(UART_BASE + 7) == 0

    def test_write_unknown_register(self) -> None:
        """Writing to an unmapped UART register offset is a no-op."""
        tx = io.BytesIO()
        uart = UART(tx_stream=tx)
        uart.write8(UART_BASE + 1, 0xFF)
        uart.write8(UART_BASE + 5, 0xFF)
        # No bytes should appear on TX
        assert tx.getvalue() == b""

    def test_ansi_passthrough(self) -> None:
        """ANSI escape sequences pass through TX unmodified."""
        tx = io.BytesIO()
        uart = UART(tx_stream=tx)
        # ESC[31m = red text
        escape = b"\x1b[31m"
        for byte in escape:
            uart.write8(UART_BASE + 0, byte)
        assert tx.getvalue() == escape
