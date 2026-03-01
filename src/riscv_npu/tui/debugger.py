"""TUI debugger controller: state management and command processing."""

from __future__ import annotations

import io
import time
from collections.abc import Callable
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
    render_fn: Callable[[DebuggerState], None] | None = None


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


def debugger_run_at_speed(
    state: DebuggerState,
    hz: int,
    max_steps: int | None = None,
) -> None:
    """Run at a fixed speed with live display updates.

    Executes instructions at the given rate (steps per second), calling
    ``state.render_fn`` to redraw the display between frames. For rates
    above 30 Hz, batches multiple steps per frame to cap redraws at ~30 fps.

    Stops on breakpoint, CPU halt, max_steps limit, or KeyboardInterrupt.

    Args:
        state: The current debugger state (modified in place).
        hz: Target steps per second (must be >= 1).
        max_steps: Optional maximum number of steps before stopping.
    """
    if state.cpu.halted:
        state.message = "CPU is halted."
        return

    steps_per_frame = max(1, hz // 30)
    frame_interval = steps_per_frame / hz

    total_steps = 0

    try:
        while True:
            frame_start = time.monotonic()
            state.prev_regs = snapshot_registers(state.cpu.registers)

            for _ in range(steps_per_frame):
                if state.cpu.halted:
                    break
                if max_steps is not None and total_steps >= max_steps:
                    break

                state.cpu.step()
                total_steps += 1

                if state.cpu.pc in state.breakpoints:
                    state.message = (
                        f"Breakpoint hit at 0x{state.cpu.pc:08X} "
                        f"(ran {total_steps} steps at {hz} Hz)"
                    )
                    if state.render_fn is not None:
                        state.render_fn(state)
                    return

            state.message = (
                f"Running at {hz} Hz — "
                f"step {total_steps}, "
                f"PC 0x{state.cpu.pc:08X} "
                f"(Ctrl+C to stop)"
            )

            if state.render_fn is not None:
                state.render_fn(state)

            if state.cpu.halted:
                state.message = f"CPU halted after {total_steps} steps."
                if state.render_fn is not None:
                    state.render_fn(state)
                return

            if max_steps is not None and total_steps >= max_steps:
                state.message = f"Stopped after {total_steps} steps (limit reached)."
                if state.render_fn is not None:
                    state.render_fn(state)
                return

            elapsed = time.monotonic() - frame_start
            remaining = frame_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        state.message = f"Paused after {total_steps} steps at {hz} Hz."


HELP_TEXT = (
    "s, step                  — step one instruction\n"
    "c, continue              — run until breakpoint/halt\n"
    "r, run <hz> [max_steps]  — run at fixed speed (Ctrl+C to pause)\n"
    "b <addr>                 — toggle breakpoint (hex address)\n"
    "g <addr>                 — set memory view address (hex)\n"
    "h, help                  — show this help\n"
    "q, quit                  — exit debugger"
)


def process_command(state: DebuggerState, cmd: str) -> bool:
    """Parse and execute a debugger command.

    Supported commands:
        s, step                  -- execute one instruction
        c, continue              -- run until breakpoint/halt/limit
        r, run <hz> [max_steps]  -- run at fixed speed (steps/sec)
        b <addr>                 -- toggle breakpoint at hex address
        g <addr>                 -- set memory view address
        h, help                  -- show command help
        q, quit                  -- exit the debugger

    Args:
        state: The current debugger state (modified in place).
        cmd: The raw command string from the user.

    Returns:
        True to continue the debugger loop, False to quit.
    """
    parts = cmd.strip().split()
    if not parts:
        state.message = HELP_TEXT
        return True

    verb = parts[0].lower()

    if verb in ("s", "step"):
        debugger_step(state)

    elif verb in ("c", "continue"):
        debugger_continue(state)

    elif verb in ("r", "run"):
        if len(parts) < 2:
            state.message = "Usage: run <hz> [max_steps]"
            return True
        try:
            hz = int(parts[1])
        except ValueError:
            state.message = f"Invalid hz: {parts[1]}"
            return True
        if hz < 1:
            state.message = "Hz must be >= 1."
            return True

        max_steps: int | None = None
        if len(parts) >= 3:
            try:
                max_steps = int(parts[2])
            except ValueError:
                state.message = f"Invalid max_steps: {parts[2]}"
                return True
            if max_steps < 1:
                state.message = "max_steps must be >= 1."
                return True

        debugger_run_at_speed(state, hz, max_steps)

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

    elif verb in ("h", "help"):
        state.message = HELP_TEXT

    elif verb in ("q", "quit"):
        return False

    else:
        state.message = f"Unknown command: {verb}\n\n{HELP_TEXT}"

    return True
