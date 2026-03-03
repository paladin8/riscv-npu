"""GDB Remote Serial Protocol TCP server.

Listens on a TCP port, accepts one GDB connection, and processes
RSP packets in a synchronous loop. The server is single-threaded:
it waits for a packet, processes it, sends the reply, and repeats.
"""

from __future__ import annotations

import socket
import sys
from typing import TYPE_CHECKING

from .commands import GdbState, handle_continue, handle_packet
from .protocol import build_packet, parse_packet
from .target_xml import TARGET_XML

if TYPE_CHECKING:
    from ..cpu.cpu import CPU


def serve(cpu: CPU, port: int = 1234) -> None:
    """Start the GDB RSP server and block until the session ends.

    Opens a TCP socket, waits for GDB to connect, then enters the
    packet-processing loop. Handles one connection at a time.

    Args:
        cpu: The CPU instance to debug.
        port: TCP port to listen on.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("", port))
        server_sock.listen(1)
        print(
            f"Listening on :{port}, waiting for GDB...",
            file=sys.stderr,
        )

        conn, addr = server_sock.accept()
        print(f"GDB connected from {addr[0]}:{addr[1]}.", file=sys.stderr)

    # Close the listening socket; we only handle one connection
    _run_session(cpu, conn)


def _run_session(cpu: CPU, conn: socket.socket) -> None:
    """Run the RSP packet-processing loop over an established connection.

    Args:
        cpu: The CPU instance to debug.
        conn: The connected client socket.
    """
    state = GdbState(cpu=cpu, target_xml=TARGET_XML)
    buf = b""

    try:
        while True:
            # Read data from GDB
            try:
                data = conn.recv(4096)
            except OSError:
                break

            if not data:
                # Connection closed
                break

            buf += data

            # Process all complete packets in the buffer
            while True:
                # Check for bare interrupt byte (0x03) before packet framing
                if buf and buf[0] == 0x03:
                    buf = buf[1:]
                    # Interrupt -- send SIGINT stop reply
                    _send_packet(conn, state, "S02")
                    continue

                # Find a complete packet: $...#xx
                dollar = buf.find(ord("$"))
                if dollar == -1:
                    # No packet start -- discard ack bytes and other noise
                    buf = b""
                    break

                # Strip any leading noise (e.g. ack '+' bytes)
                if dollar > 0:
                    buf = buf[dollar:]

                hash_pos = buf.find(ord("#"), 1)
                if hash_pos == -1 or hash_pos + 2 >= len(buf):
                    # Incomplete packet -- wait for more data
                    break

                # Extract the packet frame
                frame = buf[: hash_pos + 3]
                buf = buf[hash_pos + 3 :]

                body = parse_packet(frame)
                if body is None:
                    # Bad checksum -- NACK
                    if not state.no_ack:
                        conn.sendall(b"-")
                    continue

                # Valid packet -- ACK
                if not state.no_ack:
                    conn.sendall(b"+")

                # Check for kill command
                if body == "k":
                    return

                # Handle the packet
                response = handle_packet(state, body)

                if response is not None:
                    _send_packet(conn, state, response)
                else:
                    # 'c' (continue) -- run the CPU until it stops
                    if body.startswith("c"):
                        reply = handle_continue(state, conn)
                        _send_packet(conn, state, reply)

    except OSError:
        pass
    finally:
        _print_final_state(cpu)
        conn.close()


def _send_packet(conn: socket.socket, state: GdbState, body: str) -> None:
    """Send an RSP packet over the connection.

    Args:
        conn: The client socket.
        state: Current GDB session state.
        body: The packet body to send.
    """
    packet = build_packet(body)
    conn.sendall(packet)


def _print_final_state(cpu: CPU) -> None:
    """Print final CPU state to stderr on session end.

    Args:
        cpu: The CPU instance.
    """
    print(f"GDB session ended after {cpu.cycle_count} cycles.", file=sys.stderr)
    print(f"  x10 (a0) = {cpu.registers.read(10)}", file=sys.stderr)
    print(f"  x11 (a1) = {cpu.registers.read(11)}", file=sys.stderr)
