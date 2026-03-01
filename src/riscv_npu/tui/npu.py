"""TUI NPU display: formats accumulator and vector registers with change highlighting."""

from __future__ import annotations

from ..npu.engine import NpuState, acc_get64


def format_npu_state(npu: NpuState) -> str:
    """Format NPU state (accumulator + vector registers) for display.

    Shows the 64-bit accumulator as hex and signed decimal, and all 4 vector
    registers as int8 arrays. Non-zero values are highlighted with Rich markup
    ``[bold yellow]...[/bold yellow]``.

    Args:
        npu: The current NPU state.

    Returns:
        A string with Rich markup suitable for display in a Rich Panel.
    """
    lines: list[str] = []

    # Accumulator
    acc64 = acc_get64(npu)
    acc_hex = f"0x{npu.acc_hi:08X}_{npu.acc_lo:08X}"
    acc_line = f"  {acc_hex}  ({acc64})"
    if npu.acc_lo != 0 or npu.acc_hi != 0:
        lines.append("[bold yellow]Accumulator[/bold yellow]")
        lines.append(f"[bold yellow]{acc_line}[/bold yellow]")
    else:
        lines.append("Accumulator")
        lines.append(acc_line)

    lines.append("")

    # Vector registers
    lines.append("Vector Registers")
    for i in range(4):
        elems = ", ".join(f"{v:4d}" for v in npu.vreg[i])
        vreg_line = f"  v{i}: [{elems}]"
        if any(v != 0 for v in npu.vreg[i]):
            lines.append(f"[bold yellow]{vreg_line}[/bold yellow]")
        else:
            lines.append(vreg_line)

    return "\n".join(lines)
