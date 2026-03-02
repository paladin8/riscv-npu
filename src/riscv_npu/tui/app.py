"""TUI application: main debugger interface using Rich Live display."""

from __future__ import annotations

import io
import readline  # noqa: F401  # pyright: ignore[reportUnusedImport]
import sys

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from ..cpu.cpu import CPU
from ..devices.uart import UART, UART_BASE, UART_SIZE
from ..loader.elf import find_symbol, load_elf, parse_elf
from ..memory.bus import MemoryBus
from ..memory.ram import RAM
from ..syscall.handler import SyscallHandler
from .debugger import DebuggerState, process_command
from .disasm import disassemble_region
from .memory import format_hex_dump
from .npu import format_npu_state
from .registers import format_float_registers, format_registers
from .stats import format_instruction_stats

# System constants (match cli.py)
_BASE = 0x80000000
_RAM_SIZE = 4 * 1024 * 1024  # 4 MB
_STACK_TOP = _BASE + _RAM_SIZE - 16  # Top of RAM, 16-byte aligned


def render_debugger(state: DebuggerState) -> Layout:
    """Build the Rich Layout with all debugger panels.

    Layout structure (top gets 3x vertical space, bottom 1x):
        +------------------+------------------+
        |   Registers      |                  |
        +------------------+                  |
        |   FPU Registers  |   Disassembly    |
        +------------------+                  |
        |   NPU            |                  |
        +------------------+------------------+
        |   Memory         |   Output (UART)  |
        +------------------+------------------+
        |   Instruction Statistics            |
        +-------------------------------------+
        |   Status bar (full width)           |
        +-------------------------------------+

    Args:
        state: The current debugger state.

    Returns:
        A Rich Layout object ready for display.
    """
    layout = Layout()

    # Status panel height: 2 (border) + content lines
    # Content = 1 info line + message lines
    msg_lines = state.message.count("\n") + 1
    status_height = 2 + 1 + msg_lines

    # Top half: registers/NPU + disassembly
    layout.split_column(
        Layout(name="top", ratio=3),
        Layout(name="bottom", ratio=1),
        Layout(name="stats", ratio=1),
        Layout(name="status", size=status_height),
    )

    layout["top"].split_row(
        Layout(name="left_col", ratio=1),
        Layout(name="disassembly", ratio=1),
    )

    layout["left_col"].split_column(
        Layout(name="registers", ratio=2),
        Layout(name="fpu", ratio=2),
        Layout(name="npu", ratio=1),
    )

    layout["bottom"].split_row(
        Layout(name="memory", ratio=1),
        Layout(name="output", ratio=1),
    )

    # Registers panel
    reg_text = format_registers(state.cpu.registers, state.prev_regs)
    layout["registers"].update(Panel(reg_text, title="Registers"))

    # FPU panel
    fpu_text = format_float_registers(state.cpu.fpu_state, state.prev_fregs)
    layout["fpu"].update(Panel(fpu_text, title="FPU Registers"))

    # NPU panel
    npu_text = format_npu_state(state.cpu.npu_state)
    layout["npu"].update(Panel(npu_text, title="NPU"))

    # Disassembly panel
    disasm_lines = disassemble_region(state.cpu.memory, state.cpu.pc, 21)
    disasm_text = Text()
    for line in disasm_lines:
        marker = ">>>" if line.is_current else "   "
        bp_marker = " *" if line.addr in state.breakpoints else "  "
        entry = f"{marker}{bp_marker} 0x{line.addr:08X}: {line.text}"
        if line.is_current:
            disasm_text.append(entry + "\n", style="bold green")
        elif line.addr in state.breakpoints:
            disasm_text.append(entry + "\n", style="bold red")
        else:
            disasm_text.append(entry + "\n")
    layout["disassembly"].update(Panel(disasm_text, title="Disassembly"))

    # Memory panel
    mem_text = format_hex_dump(state.cpu.memory, state.mem_view_addr, num_rows=8)
    layout["memory"].update(Panel(mem_text, title=f"Memory @ 0x{state.mem_view_addr:08X}"))

    # Output panel (UART + syscall stdout)
    uart_content = state.uart_capture.getvalue()
    try:
        uart_text = uart_content.decode("utf-8", errors="replace")
    except Exception:
        uart_text = repr(uart_content)
    # Show last 8 lines
    uart_lines = uart_text.split("\n")
    if len(uart_lines) > 8:
        uart_lines = uart_lines[-8:]
    layout["output"].update(Panel("\n".join(uart_lines), title="Output"))

    # Instruction statistics panel
    stats_text = format_instruction_stats(state.cpu.instruction_stats)
    layout["stats"].update(Panel(stats_text, title="Instruction Statistics"))

    # Status bar
    halted_str = "HALTED" if state.cpu.halted else "RUNNING"
    bp_str = ", ".join(f"0x{a:08X}" for a in sorted(state.breakpoints))
    status_text = (
        f"PC: 0x{state.cpu.pc:08X}  |  "
        f"Cycles: {state.cpu.cycle_count}  |  "
        f"State: {halted_str}  |  "
        f"Breakpoints: {bp_str if bp_str else 'none'}\n"
        f"{state.message}"
    )
    layout["status"].update(Panel(status_text, title="Status"))

    return layout


def run_debugger(
    elf_path: str, writes: list[tuple[str, str]] | None = None,
) -> None:
    """Launch the TUI debugger for an ELF binary.

    Loads the ELF file, creates the CPU/bus/UART/syscall infrastructure,
    and enters the command loop with Rich console output.

    Args:
        elf_path: Path to the ELF file to debug.
        writes: Optional list of (symbol, file_path) pairs to write into RAM.
    """
    # Set up memory bus, RAM, UART with capture buffer
    bus = MemoryBus()
    ram = RAM(_BASE, _RAM_SIZE)
    uart_capture = io.BytesIO()
    uart = UART(tx_stream=uart_capture)
    bus.register(_BASE, _RAM_SIZE, ram)
    bus.register(UART_BASE, UART_SIZE, uart)

    # Set up CPU with syscall handler
    cpu = CPU(bus)
    handler = SyscallHandler(stdout=uart_capture)
    cpu.syscall_handler = handler

    # Load ELF
    entry = load_elf(elf_path, ram)
    cpu.pc = entry
    cpu.registers.write(2, _STACK_TOP)  # SP = x2

    # Set initial program break
    with open(elf_path, "rb") as f:
        elf_data = f.read()
    prog = parse_elf(elf_data)
    if prog.segments:
        end = max(s.vaddr + s.memsz for s in prog.segments)
        handler.brk = (end + 15) & ~15

    # Apply --write arguments
    if writes:
        for symbol, file_path in writes:
            addr = find_symbol(elf_data, symbol)
            if addr is None:
                print(f"Error: symbol '{symbol}' not found in ELF")
                sys.exit(1)
            with open(file_path, "rb") as f:
                data = f.read()
            ram.load_segment(addr, data)
            print(
                f"Loaded {len(data)} bytes from {file_path} "
                f"at {symbol} (0x{addr:08X})"
            )

    # Create debugger state
    state = DebuggerState(
        cpu=cpu,
        uart_capture=uart_capture,
        mem_view_addr=cpu.pc,
    )

    console = Console()

    def _render(st: DebuggerState) -> None:
        console.clear()
        lay = render_debugger(st)
        console.print(lay)

    state.render_fn = _render

    # Initial display
    _render(state)

    # Command loop
    while True:
        try:
            cmd = input("dbg> ")
        except (EOFError, KeyboardInterrupt):
            console.print("\nExiting debugger.")
            break

        should_continue = process_command(state, cmd)
        if not should_continue:
            console.print("Exiting debugger.")
            break

        _render(state)

    sys.exit(0)
