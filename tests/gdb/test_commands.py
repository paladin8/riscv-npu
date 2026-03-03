"""Tests for GDB RSP command handlers."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock

from riscv_npu.cpu.cpu import CPU
from riscv_npu.gdb.commands import (
    GdbState,
    handle_continue,
    handle_packet,
    handle_step,
)
from riscv_npu.gdb.protocol import decode_reg32, encode_reg32
from riscv_npu.gdb.target_xml import TARGET_XML
from riscv_npu.memory.bus import MemoryBus
from riscv_npu.memory.ram import RAM

BASE = 0x80000000
RAM_SIZE = 1024 * 1024  # 1 MB

# NOP instruction: ADDI x0, x0, 0
NOP = 0x00000013

# ECALL: triggers halt via syscall handler
ECALL = 0x00000073

# EBREAK: environment break
EBREAK = 0x00100073


def _make_state() -> GdbState:
    """Create a GdbState with a CPU and RAM initialized with NOPs."""
    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    bus.register(BASE, RAM_SIZE, ram)
    cpu = CPU(bus)
    cpu.pc = BASE
    # Fill first 256 bytes with NOPs
    for i in range(64):
        cpu.memory.write32(BASE + i * 4, NOP)
    return GdbState(cpu=cpu, target_xml=TARGET_XML)


def _make_mock_socket() -> socket.socket:
    """Create a mock socket that returns no data (no interrupt)."""
    mock_sock = MagicMock(spec=socket.socket)
    mock_sock.recv = MagicMock(return_value=b"")
    mock_sock.fileno = MagicMock(return_value=-1)
    return mock_sock


def test_halt_reason() -> None:
    """'?' returns 'S05'."""
    state = _make_state()
    assert handle_packet(state, "?") == "S05"


def test_halt_reason_when_halted() -> None:
    """'?' returns exit code when CPU is halted."""
    state = _make_state()
    state.cpu.halted = True
    state.cpu.exit_code = 0
    assert handle_packet(state, "?") == "W00"


def test_read_regs_initial() -> None:
    """'g' returns 264 hex chars, all zero except PC."""
    state = _make_state()
    result = handle_packet(state, "g")
    assert result is not None
    assert len(result) == 264
    # x0-x31 are all 0 initially
    for i in range(32):
        reg_hex = result[i * 8 : (i + 1) * 8]
        assert reg_hex == "00000000", f"x{i} should be 0"
    # PC should be BASE (0x80000000)
    pc_hex = result[256:264]
    assert decode_reg32(pc_hex) == BASE


def test_read_regs_after_write() -> None:
    """'g' reflects modified register values."""
    state = _make_state()
    state.cpu.registers.write(1, 0x12345678)
    result = handle_packet(state, "g")
    assert result is not None
    # x1 is at offset 8..16
    assert decode_reg32(result[8:16]) == 0x12345678


def test_read_reg_pc() -> None:
    """'p20' returns PC value (register 32 = 0x20 in hex)."""
    state = _make_state()
    state.cpu.pc = BASE
    result = handle_packet(state, "p20")
    assert result is not None
    assert decode_reg32(result) == BASE


def test_read_reg_gpr() -> None:
    """'p01' returns x1 value."""
    state = _make_state()
    state.cpu.registers.write(1, 0xABCDEF01)
    result = handle_packet(state, "p1")
    assert result is not None
    assert decode_reg32(result) == 0xABCDEF01


def test_read_reg_fpr() -> None:
    """'p21' returns f0 value (register 33 = 0x21 in hex)."""
    state = _make_state()
    state.cpu.fpu_state.fregs.write_bits(0, 0x3F800000)  # 1.0f
    result = handle_packet(state, "p21")
    assert result is not None
    assert decode_reg32(result) == 0x3F800000


def test_read_reg_fflags() -> None:
    """'p41' returns fflags (register 65 = 0x41 in hex)."""
    state = _make_state()
    state.cpu.fpu_state.fflags = 0x1F
    result = handle_packet(state, "p41")
    assert result is not None
    assert decode_reg32(result) == 0x1F


def test_write_reg() -> None:
    """'P01=78563412' sets x1 to 0x12345678."""
    state = _make_state()
    result = handle_packet(state, "P1=78563412")
    assert result == "OK"
    assert state.cpu.registers.read(1) == 0x12345678


def test_write_reg_pc() -> None:
    """'P20=00000180' sets PC to 0x80010000."""
    state = _make_state()
    # 0x80010000 in LE is: 00 00 01 80 -> "00000180"
    result = handle_packet(state, "P20=00000180")
    assert result == "OK"
    assert state.cpu.pc == 0x80010000


def test_write_regs() -> None:
    """'G' with 264 hex chars writes all registers."""
    state = _make_state()
    # Build a register block: x0=0, x1=0x11, ..., x31=0, pc=BASE
    data = "00000000" * 32 + encode_reg32(BASE + 0x100)
    result = handle_packet(state, "G" + data)
    assert result == "OK"
    assert state.cpu.pc == BASE + 0x100


def test_read_mem() -> None:
    """'m80000000,4' returns 4 bytes from RAM."""
    state = _make_state()
    # NOP = 0x00000013, in LE bytes: 13 00 00 00
    result = handle_packet(state, "m80000000,4")
    assert result is not None
    assert result == "13000000"


def test_read_mem_unmapped() -> None:
    """'m00000000,4' returns E01 error for unmapped memory."""
    state = _make_state()
    result = handle_packet(state, "m00000000,4")
    assert result == "E01"


def test_write_mem() -> None:
    """'M80000000,4:deadbeef' writes to RAM."""
    state = _make_state()
    result = handle_packet(state, "M80000000,4:deadbeef")
    assert result == "OK"
    # Verify the write
    assert state.cpu.memory.read8(BASE) == 0xDE
    assert state.cpu.memory.read8(BASE + 1) == 0xAD
    assert state.cpu.memory.read8(BASE + 2) == 0xBE
    assert state.cpu.memory.read8(BASE + 3) == 0xEF


def test_write_mem_then_read() -> None:
    """Write via M, then read via m -- round-trip."""
    state = _make_state()
    handle_packet(state, "M80000000,4:cafebabe")
    result = handle_packet(state, "m80000000,4")
    assert result == "cafebabe"


def test_step() -> None:
    """'s' advances PC by 4 (on a NOP), returns 'S05'."""
    state = _make_state()
    initial_pc = state.cpu.pc
    result = handle_packet(state, "s")
    assert result == "S05"
    assert state.cpu.pc == initial_pc + 4


def test_step_halted() -> None:
    """'s' when halted returns 'W00'."""
    state = _make_state()
    state.cpu.halted = True
    state.cpu.exit_code = 0
    result = handle_packet(state, "s")
    assert result == "W00"


def test_continue_to_breakpoint() -> None:
    """'c' with breakpoint set stops at breakpoint."""
    state = _make_state()
    bp_addr = BASE + 8  # After 2 NOPs
    state.breakpoints.add(bp_addr)

    # Create a mock socket with select that returns empty (no interrupt)
    mock_sock = _make_mock_socket()
    result = handle_continue(state, mock_sock)
    assert result == "S05"
    assert state.cpu.pc == bp_addr


def test_continue_to_halt() -> None:
    """'c' with halt instruction runs to halt, returns 'W00'."""
    state = _make_state()
    # Write a program: NOP, NOP, then set halted via tohost CSR write
    # Use CSRRW to write to tohost CSR (0x51E)
    # CSRRW: imm[11:0]=0x51E, rs1=x1, funct3=001, rd=x0, opcode=0x73
    # First: ADDI x1, x0, 1 (load 1 into x1)
    addi_x1_1 = 0x00100093  # ADDI x1, x0, 1
    # CSRRW x0, 0x51E, x1
    csrrw_tohost = (0x51E << 20) | (1 << 15) | (0x001 << 12) | (0 << 7) | 0x73

    state.cpu.memory.write32(BASE, addi_x1_1)
    state.cpu.memory.write32(BASE + 4, csrrw_tohost)

    mock_sock = _make_mock_socket()
    result = handle_continue(state, mock_sock)
    assert result == "W00"


def test_insert_breakpoint() -> None:
    """'Z0,80000010,4' adds to breakpoint set."""
    state = _make_state()
    result = handle_packet(state, "Z0,80000010,4")
    assert result == "OK"
    assert 0x80000010 in state.breakpoints


def test_remove_breakpoint() -> None:
    """'z0,80000010,4' removes from breakpoint set."""
    state = _make_state()
    state.breakpoints.add(0x80000010)
    result = handle_packet(state, "z0,80000010,4")
    assert result == "OK"
    assert 0x80000010 not in state.breakpoints


def test_remove_nonexistent_breakpoint() -> None:
    """Removing a breakpoint that doesn't exist is OK."""
    state = _make_state()
    result = handle_packet(state, "z0,80000010,4")
    assert result == "OK"


def test_unsupported_returns_empty() -> None:
    """'Z1,...' (hardware breakpoint) returns empty string."""
    state = _make_state()
    assert handle_packet(state, "Z1,80000000,4") == ""
    assert handle_packet(state, "Z2,80000000,4") == ""
    assert handle_packet(state, "Z3,80000000,4") == ""
    assert handle_packet(state, "Z4,80000000,4") == ""


def test_query_supported() -> None:
    """'qSupported' returns feature list."""
    state = _make_state()
    result = handle_packet(state, "qSupported:multiprocess+")
    assert result is not None
    assert "PacketSize=4096" in result
    assert "QStartNoAckMode+" in result
    assert "qXfer:features:read+" in result


def test_no_ack_mode() -> None:
    """'QStartNoAckMode' sets state.no_ack = True."""
    state = _make_state()
    assert not state.no_ack
    result = handle_packet(state, "QStartNoAckMode")
    assert result == "OK"
    assert state.no_ack is True


def test_query_attached() -> None:
    """'qAttached' returns '1'."""
    state = _make_state()
    assert handle_packet(state, "qAttached") == "1"


def test_query_current_thread() -> None:
    """'qC' returns 'QC1'."""
    state = _make_state()
    assert handle_packet(state, "qC") == "QC1"


def test_set_thread() -> None:
    """'Hg0' returns 'OK'."""
    state = _make_state()
    assert handle_packet(state, "Hg0") == "OK"
    assert handle_packet(state, "Hc0") == "OK"
    assert handle_packet(state, "Hc-1") == "OK"


def test_xfer_target_xml() -> None:
    """'qXfer:features:read:target.xml:0,ffff' returns full XML."""
    state = _make_state()
    # Request enough bytes to get the entire XML (0xffff = 65535)
    result = handle_packet(state, "qXfer:features:read:target.xml:0,ffff")
    assert result is not None
    # Should start with 'l' (last chunk) since XML fits in one large read
    assert result[0] == "l"
    assert "riscv:rv32" in result
    assert "org.gnu.gdb.riscv.cpu" in result


def test_xfer_target_xml_partial() -> None:
    """Partial XML read returns 'm' prefix (more data available)."""
    state = _make_state()
    # Request just 10 bytes
    result = handle_packet(state, "qXfer:features:read:target.xml:0,a")
    assert result is not None
    assert result[0] == "m"  # More data available


def test_continue_returns_none() -> None:
    """'c' returns None from handle_packet (handled separately)."""
    state = _make_state()
    result = handle_packet(state, "c")
    assert result is None


def test_kill_returns_none() -> None:
    """'k' returns None from handle_packet (handled by server)."""
    state = _make_state()
    result = handle_packet(state, "k")
    assert result is None


def test_unknown_packet_returns_empty() -> None:
    """Unknown packets return empty string."""
    state = _make_state()
    assert handle_packet(state, "vMustReplyEmpty") == ""
