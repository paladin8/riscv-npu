"""TUI NPU display: formats accumulator and vector registers with change highlighting."""

from __future__ import annotations

import struct

from ..npu.engine import NpuState, acc_get64


def format_npu_state(npu: NpuState) -> str:
    """Format NPU state (accumulators + vector registers) for display.

    Shows the integer and float accumulators side-by-side on one line,
    followed by the 4 vector registers. Non-zero values are highlighted
    with Rich markup ``[bold yellow]...[/bold yellow]``.

    Args:
        npu: The current NPU state.

    Returns:
        A string with Rich markup suitable for display in a Rich Panel.
    """
    lines: list[str] = []

    # Accumulators — side by side
    acc64 = acc_get64(npu)
    acc_hex = f"0x{npu.acc_hi:08X}_{npu.acc_lo:08X}"
    int_val = f"int: {acc_hex} ({acc64})"

    try:
        f32 = struct.unpack('<f', struct.pack('<f', npu.facc))[0]
    except OverflowError:
        f32 = float('inf') if npu.facc > 0 else float('-inf')
    fp_val = f"facc: {npu.facc:.8g} (f32: {f32:.6g})"

    if npu.acc_lo != 0 or npu.acc_hi != 0:
        int_part = f"[bold yellow]{int_val}[/bold yellow]"
    else:
        int_part = int_val

    if npu.facc != 0.0:
        fp_part = f"[bold yellow]{fp_val}[/bold yellow]"
    else:
        fp_part = fp_val

    lines.append("Accumulators")
    lines.append(f"  {int_part}   {fp_part}")

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
