"""Tests for the TUI debugger controller."""

import io

from riscv_npu.cpu.cpu import CPU
from riscv_npu.memory.bus import MemoryBus
from riscv_npu.memory.ram import RAM
from riscv_npu.tui.debugger import (
    DebuggerState,
    debugger_continue,
    debugger_step,
    process_command,
)

BASE = 0x80000000
RAM_SIZE = 1024 * 1024


def _make_state() -> DebuggerState:
    """Create a DebuggerState with a fresh CPU and memory."""
    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    bus.register(BASE, RAM_SIZE, ram)
    cpu = CPU(bus)
    cpu.pc = BASE
    return DebuggerState(cpu=cpu)


def _addi_word(rd: int, rs1: int, imm12: int) -> int:
    """Encode ADDI rd, rs1, imm12 as a 32-bit instruction word."""
    return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (0b000 << 12) | (rd << 7) | 0x13


def _ebreak_word() -> int:
    """Encode EBREAK as a 32-bit instruction word."""
    return 0x00100073


class TestDebuggerStep:
    """Tests for debugger_step function."""

    def test_step_advances_pc(self) -> None:
        state = _make_state()
        # ADDI x1, x0, 42
        state.cpu.memory.write32(BASE, _addi_word(1, 0, 42))
        debugger_step(state)
        assert state.cpu.pc == BASE + 4

    def test_step_updates_prev_regs(self) -> None:
        state = _make_state()
        state.cpu.registers.write(1, 0x99)
        state.cpu.memory.write32(BASE, _addi_word(1, 0, 42))
        debugger_step(state)
        # prev_regs should have the value BEFORE the step
        assert state.prev_regs[1] == 0x99

    def test_step_increments_cycle_count(self) -> None:
        state = _make_state()
        state.cpu.memory.write32(BASE, _addi_word(1, 0, 1))
        debugger_step(state)
        assert state.cpu.cycle_count == 1

    def test_step_on_halted_cpu_does_nothing(self) -> None:
        state = _make_state()
        state.cpu.halted = True
        old_pc = state.cpu.pc
        debugger_step(state)
        assert state.cpu.pc == old_pc
        assert "halted" in state.message.lower()

    def test_step_updates_message(self) -> None:
        state = _make_state()
        state.cpu.memory.write32(BASE, _addi_word(1, 0, 42))
        debugger_step(state)
        assert "Stepped" in state.message


class TestDebuggerContinue:
    """Tests for debugger_continue function."""

    def test_continue_stops_at_breakpoint(self) -> None:
        state = _make_state()
        # Write 10 ADDI instructions
        for i in range(10):
            state.cpu.memory.write32(BASE + i * 4, _addi_word(1, 1, 1))
        # Set breakpoint at the 5th instruction
        bp_addr = BASE + 5 * 4
        state.breakpoints.add(bp_addr)
        debugger_continue(state)
        assert state.cpu.pc == bp_addr
        assert "Breakpoint" in state.message

    def test_continue_stops_at_halt(self) -> None:
        state = _make_state()
        # ADDI then EBREAK
        state.cpu.memory.write32(BASE, _addi_word(1, 0, 1))
        state.cpu.memory.write32(BASE + 4, _ebreak_word())
        debugger_continue(state)
        assert state.cpu.halted is True
        assert "halted" in state.message.lower()

    def test_continue_stops_at_max_cycles(self) -> None:
        state = _make_state()
        # Write a long loop of NOPs (ADDI x0, x0, 0)
        nop = _addi_word(0, 0, 0)
        for i in range(200):
            state.cpu.memory.write32(BASE + i * 4, nop)
        debugger_continue(state, max_cycles=100)
        assert state.cpu.cycle_count == 100
        assert "limit" in state.message.lower()

    def test_continue_on_halted_cpu_does_nothing(self) -> None:
        state = _make_state()
        state.cpu.halted = True
        debugger_continue(state)
        assert "halted" in state.message.lower()

    def test_continue_updates_prev_regs(self) -> None:
        state = _make_state()
        state.cpu.registers.write(5, 0xAA)
        state.cpu.memory.write32(BASE, _ebreak_word())
        debugger_continue(state)
        # prev_regs should have the value BEFORE running
        assert state.prev_regs[5] == 0xAA


class TestProcessCommand:
    """Tests for process_command function."""

    def test_step_command(self) -> None:
        state = _make_state()
        state.cpu.memory.write32(BASE, _addi_word(1, 0, 42))
        result = process_command(state, "s")
        assert result is True
        assert state.cpu.pc == BASE + 4

    def test_step_full_word(self) -> None:
        state = _make_state()
        state.cpu.memory.write32(BASE, _addi_word(1, 0, 42))
        result = process_command(state, "step")
        assert result is True
        assert state.cpu.pc == BASE + 4

    def test_continue_command(self) -> None:
        state = _make_state()
        state.cpu.memory.write32(BASE, _ebreak_word())
        result = process_command(state, "c")
        assert result is True
        assert state.cpu.halted is True

    def test_continue_full_word(self) -> None:
        state = _make_state()
        state.cpu.memory.write32(BASE, _ebreak_word())
        result = process_command(state, "continue")
        assert result is True

    def test_run_alias(self) -> None:
        state = _make_state()
        state.cpu.memory.write32(BASE, _ebreak_word())
        result = process_command(state, "r")
        assert result is True
        assert state.cpu.halted is True

    def test_breakpoint_set(self) -> None:
        state = _make_state()
        result = process_command(state, "b 80000010")
        assert result is True
        assert 0x80000010 in state.breakpoints
        assert "set" in state.message.lower()

    def test_breakpoint_toggle_remove(self) -> None:
        state = _make_state()
        process_command(state, "b 80000010")
        assert 0x80000010 in state.breakpoints
        process_command(state, "b 80000010")
        assert 0x80000010 not in state.breakpoints
        assert "removed" in state.message.lower()

    def test_breakpoint_missing_address(self) -> None:
        state = _make_state()
        result = process_command(state, "b")
        assert result is True
        assert "Usage" in state.message

    def test_breakpoint_invalid_address(self) -> None:
        state = _make_state()
        result = process_command(state, "b not_hex")
        assert result is True
        assert "Invalid" in state.message

    def test_goto_command(self) -> None:
        state = _make_state()
        result = process_command(state, "g 80000100")
        assert result is True
        assert state.mem_view_addr == 0x80000100

    def test_goto_missing_address(self) -> None:
        state = _make_state()
        result = process_command(state, "g")
        assert result is True
        assert "Usage" in state.message

    def test_goto_invalid_address(self) -> None:
        state = _make_state()
        result = process_command(state, "g xyz")
        assert result is True
        assert "Invalid" in state.message

    def test_quit_command(self) -> None:
        state = _make_state()
        result = process_command(state, "q")
        assert result is False

    def test_quit_full_word(self) -> None:
        state = _make_state()
        result = process_command(state, "quit")
        assert result is False

    def test_unknown_command(self) -> None:
        state = _make_state()
        result = process_command(state, "blah")
        assert result is True
        assert "Unknown" in state.message

    def test_empty_command(self) -> None:
        state = _make_state()
        result = process_command(state, "")
        assert result is True

    def test_whitespace_command(self) -> None:
        state = _make_state()
        result = process_command(state, "   ")
        assert result is True

    def test_case_insensitive(self) -> None:
        state = _make_state()
        state.cpu.memory.write32(BASE, _addi_word(1, 0, 42))
        result = process_command(state, "S")
        assert result is True
        assert state.cpu.pc == BASE + 4


class TestDebuggerState:
    """Tests for DebuggerState defaults."""

    def test_default_breakpoints_empty(self) -> None:
        state = _make_state()
        assert len(state.breakpoints) == 0

    def test_default_mem_view_addr(self) -> None:
        state = _make_state()
        assert state.mem_view_addr == 0x80000000

    def test_default_running_false(self) -> None:
        state = _make_state()
        assert state.running is False

    def test_default_message(self) -> None:
        state = _make_state()
        assert "Ready" in state.message

    def test_uart_capture_is_bytesio(self) -> None:
        state = _make_state()
        assert isinstance(state.uart_capture, io.BytesIO)
