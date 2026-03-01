"""TUI register display: formats all 32 registers with ABI names and change highlighting."""

from __future__ import annotations

import struct

from ..cpu.fpu import FpuState
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


# RISC-V ABI float register names (f0-f31)
FLOAT_ABI_NAMES: list[str] = [
    "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
    "fs0", "fs1",
    "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7",
    "fs2", "fs3", "fs4", "fs5", "fs6", "fs7", "fs8", "fs9", "fs10", "fs11",
    "ft8", "ft9", "ft10", "ft11",
]


def format_float_registers(
    fpu_state: FpuState, prev_values: list[int] | None = None
) -> str:
    """Format all 32 float registers for display with ABI names and change highlighting.

    Displays f0-f31 in a 4x8 grid showing hex bits and float value.
    Includes FCSR flags at the bottom.

    Args:
        fpu_state: The current FPU state.
        prev_values: Optional list of 32 previous float register bit values.

    Returns:
        A string with Rich markup suitable for display in a Rich Panel.
    """
    lines: list[str] = []
    cols = 4
    rows = 32 // cols  # 8 rows
    fregs = fpu_state.fregs

    for row in range(rows):
        parts: list[str] = []
        for col in range(cols):
            idx = row + col * rows
            bits = fregs.read_bits(idx)
            fval = struct.unpack('<f', struct.pack('<I', bits))[0]
            abi = FLOAT_ABI_NAMES[idx]
            label = f"f{idx:<2d} {abi:<4s}"
            val_str = f"{fval:>10.4g}"

            changed = prev_values is not None and bits != prev_values[idx]

            if changed:
                entry = f"[bold yellow]{label} {val_str}[/bold yellow]"
            else:
                entry = f"{label} {val_str}"

            parts.append(entry)
        lines.append("  ".join(parts))

    # FCSR flags line
    flags = fpu_state.fflags
    flag_strs = []
    if flags & 0x10:
        flag_strs.append("NV")
    if flags & 0x08:
        flag_strs.append("DZ")
    if flags & 0x04:
        flag_strs.append("OF")
    if flags & 0x02:
        flag_strs.append("UF")
    if flags & 0x01:
        flag_strs.append("NX")
    rm_names = ["RNE", "RTZ", "RDN", "RUP", "RMM", "?5", "?6", "DYN"]
    lines.append(f"FCSR: frm={rm_names[fpu_state.frm]}  fflags={','.join(flag_strs) or 'none'}")

    return "\n".join(lines)


def snapshot_float_registers(fpu_state: FpuState) -> list[int]:
    """Capture a snapshot of all 32 float register bit values.

    Args:
        fpu_state: The FPU state to snapshot.

    Returns:
        A list of 32 unsigned 32-bit IEEE 754 bit patterns.
    """
    return [fpu_state.fregs.read_bits(i) for i in range(32)]
