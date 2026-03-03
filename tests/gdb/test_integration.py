"""Integration tests for the GDB stub using a mock GDB client over TCP.

Each test starts the server in a daemon thread, connects via raw socket,
and exercises a real debug session. A short timeout prevents hangs.
"""

from __future__ import annotations

import socket
import threading
import time

from riscv_npu.cpu.cpu import CPU
from riscv_npu.gdb.protocol import build_packet, decode_reg32, parse_packet
from riscv_npu.gdb.server import _run_session
from riscv_npu.memory.bus import MemoryBus
from riscv_npu.memory.ram import RAM

BASE = 0x80000000
RAM_SIZE = 1024 * 1024  # 1 MB

# NOP instruction: ADDI x0, x0, 0
NOP = 0x00000013

# ADDI x1, x0, 1 (loads 1 into x1)
ADDI_X1_1 = 0x00100093

# ADDI x1, x1, 1 (increments x1)
ADDI_X1_X1_1 = 0x00108093

# CSRRW x0, tohost(0x51E), x1  -- halt the CPU
CSRRW_TOHOST = (0x51E << 20) | (1 << 15) | (0x001 << 12) | (0 << 7) | 0x73


def _make_cpu(program: list[int] | None = None) -> CPU:
    """Create a CPU with RAM and optionally load a program."""
    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    bus.register(BASE, RAM_SIZE, ram)
    cpu = CPU(bus)
    cpu.pc = BASE
    if program:
        for i, word in enumerate(program):
            cpu.memory.write32(BASE + i * 4, word)
    else:
        # Fill with NOPs by default
        for i in range(64):
            cpu.memory.write32(BASE + i * 4, NOP)
    return cpu


def _start_server(cpu: CPU) -> tuple[socket.socket, threading.Thread]:
    """Start the GDB server in a thread using a socket pair.

    Returns the client-side socket and the server thread.
    """
    # Use a socket pair for deterministic, port-free testing
    server_sock, client_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

    thread = threading.Thread(
        target=_run_session,
        args=(cpu, server_sock),
        daemon=True,
    )
    thread.start()

    return client_sock, thread


def _send_recv(sock: socket.socket, body: str, timeout: float = 2.0) -> str:
    """Send an RSP packet and receive the response.

    Handles ack bytes ('+') in the response by skipping them.

    Args:
        sock: Client socket.
        body: Packet body to send.
        timeout: Receive timeout in seconds.

    Returns:
        The response packet body string.
    """
    sock.settimeout(timeout)
    pkt = build_packet(body)
    sock.sendall(pkt)

    # Read response: may include ack '+' before the packet
    buf = b""
    while True:
        try:
            data = sock.recv(4096)
        except TimeoutError:
            raise TimeoutError(f"No response for packet '{body}', buf so far: {buf!r}")
        if not data:
            raise ConnectionError("Server closed connection")
        buf += data

        # Strip leading ack bytes
        while buf and buf[0:1] == b"+":
            buf = buf[1:]

        # Try to parse a complete packet from the buffer
        if b"$" in buf and b"#" in buf:
            dollar = buf.index(ord("$"))
            hash_pos = buf.index(ord("#"), dollar + 1)
            if hash_pos + 2 < len(buf):
                frame = buf[dollar : hash_pos + 3]
                result = parse_packet(frame)
                if result is not None:
                    return result


def _send_no_response(sock: socket.socket, body: str) -> None:
    """Send an RSP packet without waiting for a response."""
    pkt = build_packet(body)
    sock.sendall(pkt)


def test_gdb_connect_and_step() -> None:
    """Start server in thread, connect, step, read regs, disconnect."""
    cpu = _make_cpu([ADDI_X1_1, NOP, NOP, NOP])
    client, thread = _start_server(cpu)

    try:
        # Initial handshake
        resp = _send_recv(client, "qSupported:multiprocess+")
        assert "PacketSize" in resp

        # Halt reason
        resp = _send_recv(client, "?")
        assert resp == "S05"

        # Read initial registers
        resp = _send_recv(client, "g")
        assert len(resp) == 264

        # Step: executes ADDI x1, x0, 1
        resp = _send_recv(client, "s")
        assert resp == "S05"

        # Read registers again -- x1 should now be 1
        resp = _send_recv(client, "g")
        # x1 is at offset 8..16 in the hex string
        x1_val = decode_reg32(resp[8:16])
        assert x1_val == 1

        # PC should have advanced by 4
        pc_val = decode_reg32(resp[256:264])
        assert pc_val == BASE + 4

        # Kill
        _send_no_response(client, "k")

    finally:
        client.close()
        thread.join(timeout=2.0)


def test_gdb_breakpoint_hit() -> None:
    """Set breakpoint, continue, verify stop at breakpoint address."""
    # Program: NOP, NOP, NOP, NOP (4 instructions)
    cpu = _make_cpu([NOP, NOP, NOP, NOP])
    client, thread = _start_server(cpu)

    try:
        _send_recv(client, "qSupported:multiprocess+")
        _send_recv(client, "?")

        # Set breakpoint at BASE + 8 (after 2 NOPs)
        bp_addr = BASE + 8
        resp = _send_recv(client, f"Z0,{bp_addr:x},4")
        assert resp == "OK"

        # Continue -- should stop at breakpoint
        resp = _send_recv(client, "c")
        assert resp == "S05"

        # Read PC -- should be at breakpoint
        resp = _send_recv(client, "p20")
        pc_val = decode_reg32(resp)
        assert pc_val == bp_addr

        _send_no_response(client, "k")

    finally:
        client.close()
        thread.join(timeout=2.0)


def test_gdb_memory_read_write() -> None:
    """Write to memory via M packet, read back via m packet, verify match."""
    cpu = _make_cpu()
    client, thread = _start_server(cpu)

    try:
        _send_recv(client, "qSupported:multiprocess+")
        _send_recv(client, "?")

        # Write 4 bytes to RAM
        addr = BASE + 0x100
        resp = _send_recv(client, f"M{addr:x},4:cafebabe")
        assert resp == "OK"

        # Read them back
        resp = _send_recv(client, f"m{addr:x},4")
        assert resp == "cafebabe"

        _send_no_response(client, "k")

    finally:
        client.close()
        thread.join(timeout=2.0)


def test_gdb_continue_to_exit() -> None:
    """Continue a short program to completion, verify W00 reply."""
    # Program: ADDI x1, x0, 1; CSRRW x0, tohost, x1 (halts)
    cpu = _make_cpu([ADDI_X1_1, CSRRW_TOHOST])
    client, thread = _start_server(cpu)

    try:
        _send_recv(client, "qSupported:multiprocess+")
        _send_recv(client, "?")

        # Continue to completion
        resp = _send_recv(client, "c")
        assert resp == "W00"

    finally:
        client.close()
        thread.join(timeout=2.0)


def test_gdb_no_ack_mode() -> None:
    """Enable no-ack mode, verify subsequent packets still work."""
    cpu = _make_cpu()
    client, thread = _start_server(cpu)

    try:
        _send_recv(client, "qSupported:multiprocess+")

        # Enable no-ack mode
        resp = _send_recv(client, "QStartNoAckMode")
        assert resp == "OK"

        # After enabling no-ack, we still get packet responses
        resp = _send_recv(client, "?")
        assert resp == "S05"

        _send_no_response(client, "k")

    finally:
        client.close()
        thread.join(timeout=2.0)


def test_gdb_register_write_and_read() -> None:
    """Write a register via P packet, read it back via p, verify match."""
    cpu = _make_cpu()
    client, thread = _start_server(cpu)

    try:
        _send_recv(client, "qSupported:multiprocess+")
        _send_recv(client, "?")

        # Write x5 = 0xDEADBEEF (register 5)
        # 0xDEADBEEF in LE = efbeadde
        resp = _send_recv(client, "P5=efbeadde")
        assert resp == "OK"

        # Read it back
        resp = _send_recv(client, "p5")
        val = decode_reg32(resp)
        assert val == 0xDEADBEEF

        _send_no_response(client, "k")

    finally:
        client.close()
        thread.join(timeout=2.0)


def test_gdb_multiple_steps() -> None:
    """Step through multiple instructions and verify PC advances correctly."""
    cpu = _make_cpu([NOP, NOP, NOP, NOP])
    client, thread = _start_server(cpu)

    try:
        _send_recv(client, "qSupported:multiprocess+")
        _send_recv(client, "?")

        for i in range(3):
            resp = _send_recv(client, "s")
            assert resp == "S05"
            resp = _send_recv(client, "p20")
            pc = decode_reg32(resp)
            assert pc == BASE + (i + 1) * 4

        _send_no_response(client, "k")

    finally:
        client.close()
        thread.join(timeout=2.0)
