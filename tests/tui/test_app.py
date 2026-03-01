"""Smoke tests for the TUI app module."""

from rich.layout import Layout

from riscv_npu.cpu.cpu import CPU
from riscv_npu.memory.bus import MemoryBus
from riscv_npu.memory.ram import RAM
from riscv_npu.tui.app import render_debugger
from riscv_npu.tui.debugger import DebuggerState

BASE = 0x80000000
RAM_SIZE = 1024 * 1024


def _make_state() -> DebuggerState:
    """Create a DebuggerState with a fresh CPU and memory."""
    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    bus.register(BASE, RAM_SIZE, ram)
    cpu = CPU(bus)
    cpu.pc = BASE
    # Write a NOP so disassembly has something to decode
    nop = ((0 & 0xFFF) << 20) | (0 << 15) | (0b000 << 12) | (0 << 7) | 0x13  # ADDI x0, x0, 0
    for i in range(21):
        bus.write32(BASE + i * 4, nop)
    return DebuggerState(cpu=cpu)


class TestRenderDebugger:
    """Smoke tests for render_debugger."""

    def test_returns_layout(self) -> None:
        state = _make_state()
        result = render_debugger(state)
        assert isinstance(result, Layout)

    def test_layout_renders_without_error(self) -> None:
        state = _make_state()
        layout = render_debugger(state)
        # Just confirm it can be converted to a renderable without crashing
        from rich.console import Console
        import io
        console = Console(file=io.StringIO(), width=120, height=40)
        console.print(layout)
        output = console.file.getvalue()
        assert len(output) > 0

    def test_layout_contains_panel_titles(self) -> None:
        state = _make_state()
        layout = render_debugger(state)
        from rich.console import Console
        import io
        console = Console(file=io.StringIO(), width=120, height=60)
        console.print(layout)
        output = console.file.getvalue()
        assert "Registers" in output
        assert "Disassembly" in output
        assert "Memory" in output
        assert "Output" in output  # panel title: "Output"
        assert "Status" in output
        assert "NPU" in output

    def test_layout_shows_pc(self) -> None:
        state = _make_state()
        layout = render_debugger(state)
        from rich.console import Console
        import io
        console = Console(file=io.StringIO(), width=120, height=60)
        console.print(layout)
        output = console.file.getvalue()
        assert "80000000" in output

    def test_layout_with_breakpoint(self) -> None:
        state = _make_state()
        state.breakpoints.add(BASE + 8)
        layout = render_debugger(state)
        from rich.console import Console
        import io
        console = Console(file=io.StringIO(), width=120, height=60)
        console.print(layout)
        output = console.file.getvalue()
        assert "80000008" in output

    def test_layout_with_uart_output(self) -> None:
        state = _make_state()
        state.uart_capture.write(b"Hello from UART\n")
        layout = render_debugger(state)
        from rich.console import Console
        import io
        console = Console(file=io.StringIO(), width=120, height=60)
        console.print(layout)
        output = console.file.getvalue()
        assert "Hello from UART" in output
