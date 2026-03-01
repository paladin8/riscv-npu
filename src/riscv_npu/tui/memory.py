"""TUI memory hex dump panel: formats memory regions as hex dumps."""

from __future__ import annotations

from ..memory.bus import MemoryBus


def format_hex_dump(memory: MemoryBus, start_addr: int, num_rows: int = 16) -> str:
    """Format a memory region as a hex dump with addresses, hex bytes, and ASCII.

    Each row displays 16 bytes in the format:
        ADDR: HH HH HH HH HH HH HH HH  HH HH HH HH HH HH HH HH  |ASCII...........|

    Unmapped addresses show '??' for each byte and '.' in the ASCII column.

    Args:
        memory: The memory bus to read from.
        start_addr: The starting address of the hex dump (will be aligned
            down to a 16-byte boundary).
        num_rows: Number of 16-byte rows to display.

    Returns:
        A multi-line string suitable for display in a Rich Panel.
    """
    # Align start address down to 16-byte boundary
    aligned_addr = start_addr & ~0xF
    lines: list[str] = []

    for row in range(num_rows):
        row_addr = (aligned_addr + row * 16) & 0xFFFFFFFF
        hex_parts: list[str] = []
        ascii_parts: list[str] = []

        for col in range(16):
            byte_addr = (row_addr + col) & 0xFFFFFFFF
            try:
                byte_val = memory.read8(byte_addr)
                hex_parts.append(f"{byte_val:02X}")
                # Printable ASCII range: 0x20-0x7E
                if 0x20 <= byte_val <= 0x7E:
                    ascii_parts.append(chr(byte_val))
                else:
                    ascii_parts.append(".")
            except MemoryError:
                hex_parts.append("??")
                ascii_parts.append(".")

            # Add extra space between groups of 8
            if col == 7:
                hex_parts.append("")

        hex_str = " ".join(hex_parts)
        ascii_str = "".join(ascii_parts)
        lines.append(f"0x{row_addr:08X}: {hex_str}  |{ascii_str}|")

    return "\n".join(lines)
