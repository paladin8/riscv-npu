"""TUI debugger controller: state management and command processing."""

from __future__ import annotations

import io
from dataclasses import dataclass, field

from ..cpu.cpu import CPU
from .registers import snapshot_registers


@dataclass
class DebuggerState:
    """Mutable state for the TUI debugger session.

    Holds the CPU, breakpoint set, previous register snapshot for change
    tracking, memory view address, UART capture buffer, and status message.
    """

    cpu: CPU
    breakpoints: set[int] = field(default_factory=set)
    prev_regs: list[int] = field(default_factory=lambda: [0] * 32)
    mem_view_addr: int = 0x80000000
    uart_capture: io.BytesIO = field(default_factory=io.BytesIO)
    running: bool = False
    message: str = "Ready. Type 's' to step, 'c' to continue, 'q' to quit."


def debugger_step(state: DebuggerState) -> None:
    """Execute one instruction and update register snapshot.

    Takes a snapshot of the current registers before stepping, so the
    display can highlight which registers changed.

    Args:
        state: The current debugger state (modified in place).
    """
    if state.cpu.halted:
        state.message = "CPU is halted."
        return

    state.prev_regs = snapshot_registers(state.cpu.registers)
    state.cpu.step()
    state.message = f"Stepped to 0x{state.cpu.pc:08X} (cycle {state.cpu.cycle_count})"


def debugger_continue(state: DebuggerState, max_cycles: int = 10000) -> None:
    """Run until breakpoint, halt, or cycle limit.

    Takes a register snapshot before running, then executes instructions
    until one of the stop conditions is met.

    Args:
        state: The current debugger state (modified in place).
        max_cycles: Maximum number of cycles to execute before stopping.
    """
    if state.cpu.halted:
        state.message = "CPU is halted."
        return

    state.prev_regs = snapshot_registers(state.cpu.registers)
    start_cycle = state.cpu.cycle_count
    cycles_run = 0

    while not state.cpu.halted and cycles_run < max_cycles:
        state.cpu.step()
        cycles_run += 1

        if state.cpu.pc in state.breakpoints:
            state.message = (
                f"Breakpoint hit at 0x{state.cpu.pc:08X} "
                f"(ran {cycles_run} cycles)"
            )
            return

    if state.cpu.halted:
        state.message = f"CPU halted after {cycles_run} cycles."
    else:
        state.message = f"Stopped after {max_cycles} cycles (limit reached)."


def process_command(state: DebuggerState, cmd: str) -> bool:
    """Parse and execute a debugger command.

    Supported commands:
        s, step         -- execute one instruction
        c, continue, r, run  -- run until breakpoint/halt/limit
        b <addr>        -- toggle breakpoint at hex address
        g <addr>        -- set memory view address
        q, quit         -- exit the debugger

    Args:
        state: The current debugger state (modified in place).
        cmd: The raw command string from the user.

    Returns:
        True to continue the debugger loop, False to quit.
    """
    parts = cmd.strip().split()
    if not parts:
        return True

    verb = parts[0].lower()

    if verb in ("s", "step"):
        debugger_step(state)

    elif verb in ("c", "continue", "r", "run"):
        debugger_continue(state)

    elif verb in ("b", "breakpoint"):
        if len(parts) < 2:
            state.message = "Usage: b <hex_address>"
            return True
        try:
            addr = int(parts[1], 16)
            addr = addr & 0xFFFFFFFF
        except ValueError:
            state.message = f"Invalid address: {parts[1]}"
            return True

        if addr in state.breakpoints:
            state.breakpoints.discard(addr)
            state.message = f"Breakpoint removed at 0x{addr:08X}"
        else:
            state.breakpoints.add(addr)
            state.message = f"Breakpoint set at 0x{addr:08X}"

    elif verb in ("g", "goto"):
        if len(parts) < 2:
            state.message = "Usage: g <hex_address>"
            return True
        try:
            addr = int(parts[1], 16)
            addr = addr & 0xFFFFFFFF
        except ValueError:
            state.message = f"Invalid address: {parts[1]}"
            return True
        state.mem_view_addr = addr
        state.message = f"Memory view set to 0x{addr:08X}"

    elif verb in ("q", "quit"):
        return False

    else:
        state.message = f"Unknown command: {verb}"

    return True
