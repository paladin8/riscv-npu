"""TUI register display: formats all 32 registers with ABI names and change highlighting."""

from __future__ import annotations

from ..cpu.registers import RegisterFile

# RISC-V ABI register names (x0-x31)
ABI_NAMES: list[str] = [
    "zero", "ra", "sp", "gp", "tp",
    "t0", "t1", "t2",
    "s0", "s1",
    "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
    "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
    "t3", "t4", "t5", "t6",
]


def format_registers(regs: RegisterFile, prev_values: list[int] | None = None) -> str:
    """Format all 32 registers for display with ABI names and change highlighting.

    Produces a multi-line string with 4 columns of registers. Each register
    shows its x-number, ABI name, and hex value. Registers whose values have
    changed since the previous snapshot are highlighted with Rich markup
    ``[bold yellow]...[/bold yellow]``.

    Args:
        regs: The current register file.
        prev_values: Optional list of 32 previous register values. If None,
            no highlighting is applied.

    Returns:
        A string with Rich markup suitable for display in a Rich Panel.
    """
    lines: list[str] = []
    cols = 4
    rows = 32 // cols  # 8 rows

    for row in range(rows):
        parts: list[str] = []
        for col in range(cols):
            idx = row + col * rows
            val = regs.read(idx)
            abi = ABI_NAMES[idx]
            label = f"x{idx:<2d} {abi:<4s}"
            hex_val = f"0x{val:08X}"

            changed = (
                prev_values is not None
                and idx != 0  # x0 never changes
                and val != prev_values[idx]
            )

            if changed:
                entry = f"[bold yellow]{label} {hex_val}[/bold yellow]"
            else:
                entry = f"{label} {hex_val}"

            parts.append(entry)
        lines.append("  ".join(parts))

    return "\n".join(lines)


def snapshot_registers(regs: RegisterFile) -> list[int]:
    """Capture a snapshot of all 32 register values.

    Args:
        regs: The register file to snapshot.

    Returns:
        A list of 32 unsigned 32-bit register values.
    """
    return [regs.read(i) for i in range(32)]
