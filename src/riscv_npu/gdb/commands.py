"""GDB RSP command dispatch: map RSP packets to CPU operations.

Each handler takes a GdbState and (optionally) the packet body, and
returns the response body string. The dispatch function routes packets
to the appropriate handler.
"""

from __future__ import annotations

import select
import socket
from dataclasses import dataclass, field

from ..cpu.cpu import CPU
from .protocol import decode_reg32, encode_reg32, hex_decode, hex_encode


@dataclass
class GdbState:
    """State for the GDB debug session."""

    cpu: CPU
    breakpoints: set[int] = field(default_factory=set)
    no_ack: bool = False
    target_xml: str = ""


def handle_packet(state: GdbState, packet: str) -> str | None:
    """Dispatch an RSP packet and return the response body.

    Returns None for packets that should not send an immediate response
    (e.g., ``c`` for continue -- the response comes later when the CPU
    stops). Returns the response body string for all other packets.

    Args:
        state: Current GDB session state.
        packet: The RSP packet body string.

    Returns:
        Response body string, or None if no immediate response.
    """
    if not packet:
        return ""

    cmd = packet[0]

    if cmd == "?":
        return handle_question(state)
    elif cmd == "g":
        return handle_read_regs(state)
    elif cmd == "G":
        return handle_write_regs(state, packet[1:])
    elif cmd == "p":
        return handle_read_reg(state, packet[1:])
    elif cmd == "P":
        return handle_write_reg(state, packet[1:])
    elif cmd == "m":
        return handle_read_mem(state, packet[1:])
    elif cmd == "M":
        return handle_write_mem(state, packet[1:])
    elif cmd == "s":
        return handle_step(state)
    elif cmd == "c":
        # Continue is handled specially -- return None to signal
        # the server loop to run handle_continue with socket access
        return None
    elif cmd == "Z":
        return _handle_breakpoint_insert(state, packet[1:])
    elif cmd == "z":
        return _handle_breakpoint_remove(state, packet[1:])
    elif cmd == "k":
        return None  # Kill -- server handles connection close
    elif cmd == "q":
        return handle_query(state, packet)
    elif cmd == "Q":
        return handle_query(state, packet)
    elif cmd == "H":
        # Set thread -- single-threaded, always OK
        return "OK"
    elif cmd == "D":
        # Detach
        return "OK"
    else:
        # Unsupported packet -- empty response
        return ""


def handle_question(state: GdbState) -> str:
    """Handle ``?`` -- return halt reason.

    Always returns ``S05`` (SIGTRAP) on initial attach, indicating
    the target is stopped.

    Args:
        state: Current GDB session state.

    Returns:
        Stop reply string.
    """
    if state.cpu.halted:
        return f"W{state.cpu.exit_code:02x}"
    return "S05"


def handle_read_regs(state: GdbState) -> str:
    """Handle ``g`` -- read all registers as concatenated LE hex.

    Returns x0-x31 + pc as 264 hex characters (33 registers x 8 chars).

    Args:
        state: Current GDB session state.

    Returns:
        Hex string of all register values.
    """
    parts: list[str] = []
    for i in range(32):
        parts.append(encode_reg32(state.cpu.registers.read(i)))
    parts.append(encode_reg32(state.cpu.pc))
    return "".join(parts)


def handle_write_regs(state: GdbState, data: str) -> str:
    """Handle ``G`` -- write all registers from concatenated LE hex.

    Expects at least 264 hex chars for x0-x31 + pc.

    Args:
        state: Current GDB session state.
        data: Hex string of register values (after the ``G``).

    Returns:
        ``OK`` on success, ``E01`` on error.
    """
    if len(data) < 264:
        return "E01"

    for i in range(32):
        val = decode_reg32(data[i * 8 : (i + 1) * 8])
        state.cpu.registers.write(i, val)
    state.cpu.pc = decode_reg32(data[256:264])
    return "OK"


def handle_read_reg(state: GdbState, packet: str) -> str:
    """Handle ``p n`` -- read single register n.

    Register numbering:
    - 0-31: x0-x31 (GPR)
    - 32: pc
    - 33-64: f0-f31 (FPR)
    - 65: fflags
    - 66: frm
    - 67: fcsr

    Args:
        state: Current GDB session state.
        packet: Register number as hex string (after the ``p``).

    Returns:
        8 hex chars (LE) for the register value, or ``E01`` on error.
    """
    try:
        reg_num = int(packet, 16)
    except ValueError:
        return "E01"

    if 0 <= reg_num <= 31:
        return encode_reg32(state.cpu.registers.read(reg_num))
    elif reg_num == 32:
        return encode_reg32(state.cpu.pc)
    elif 33 <= reg_num <= 64:
        fpr_index = reg_num - 33
        return encode_reg32(state.cpu.fpu_state.fregs.read_bits(fpr_index))
    elif reg_num == 65:
        return encode_reg32(state.cpu.fpu_state.fflags)
    elif reg_num == 66:
        return encode_reg32(state.cpu.fpu_state.frm)
    elif reg_num == 67:
        return encode_reg32(state.cpu.fpu_state.fcsr)
    else:
        return "E01"


def handle_write_reg(state: GdbState, packet: str) -> str:
    """Handle ``P n=val`` -- write single register n.

    Args:
        state: Current GDB session state.
        packet: ``n=hexval`` string (after the ``P``).

    Returns:
        ``OK`` on success, ``E01`` on error.
    """
    if "=" not in packet:
        return "E01"

    reg_str, val_str = packet.split("=", 1)
    try:
        reg_num = int(reg_str, 16)
        val = decode_reg32(val_str)
    except (ValueError, IndexError):
        return "E01"

    if 0 <= reg_num <= 31:
        state.cpu.registers.write(reg_num, val)
    elif reg_num == 32:
        state.cpu.pc = val & 0xFFFFFFFF
    elif 33 <= reg_num <= 64:
        fpr_index = reg_num - 33
        state.cpu.fpu_state.fregs.write_bits(fpr_index, val)
    elif reg_num == 65:
        state.cpu.fpu_state.fflags = val & 0x1F
    elif reg_num == 66:
        state.cpu.fpu_state.frm = val & 0x7
    elif reg_num == 67:
        state.cpu.fpu_state.fcsr = val & 0xFF
    else:
        return "E01"

    return "OK"


def handle_read_mem(state: GdbState, packet: str) -> str:
    """Handle ``m addr,length`` -- read memory as hex.

    Args:
        state: Current GDB session state.
        packet: ``addr,length`` string (after the ``m``).

    Returns:
        Hex string of memory contents, or ``E01`` on error.
    """
    if "," not in packet:
        return "E01"

    addr_str, len_str = packet.split(",", 1)
    try:
        addr = int(addr_str, 16)
        length = int(len_str, 16)
    except ValueError:
        return "E01"

    try:
        result: list[str] = []
        for i in range(length):
            byte = state.cpu.memory.read8(addr + i)
            result.append(f"{byte:02x}")
        return "".join(result)
    except MemoryError:
        return "E01"


def handle_write_mem(state: GdbState, packet: str) -> str:
    """Handle ``M addr,length:data`` -- write hex data to memory.

    Args:
        state: Current GDB session state.
        packet: ``addr,length:hexdata`` string (after the ``M``).

    Returns:
        ``OK`` on success, ``E01`` on error.
    """
    if ":" not in packet:
        return "E01"

    header, hex_data = packet.split(":", 1)
    if "," not in header:
        return "E01"

    addr_str, len_str = header.split(",", 1)
    try:
        addr = int(addr_str, 16)
        length = int(len_str, 16)
    except ValueError:
        return "E01"

    try:
        data = hex_decode(hex_data)
    except ValueError:
        return "E01"

    if len(data) != length:
        return "E01"

    try:
        for i, byte in enumerate(data):
            state.cpu.memory.write8(addr + i, byte)
        return "OK"
    except MemoryError:
        return "E01"


def handle_step(state: GdbState) -> str:
    """Handle ``s`` -- single step, return stop reply.

    Executes one instruction. Returns ``S05`` (SIGTRAP) after
    stepping, or ``W00`` if the CPU has halted.

    Args:
        state: Current GDB session state.

    Returns:
        Stop reply string.
    """
    if state.cpu.halted:
        return f"W{state.cpu.exit_code:02x}"

    try:
        state.cpu.step()
    except MemoryError:
        return "S0b"  # SIGSEGV

    if state.cpu.halted:
        return f"W{state.cpu.exit_code:02x}"

    return "S05"


def handle_continue(state: GdbState, sock: socket.socket) -> str:
    """Handle ``c`` -- run until breakpoint, halt, or interrupt.

    Checks the socket for incoming ``0x03`` (Ctrl+C) between steps
    using ``select()`` with zero timeout.

    Args:
        state: Current GDB session state.
        sock: The client connection socket, for interrupt detection.

    Returns:
        Stop reply string.
    """
    if state.cpu.halted:
        return f"W{state.cpu.exit_code:02x}"

    while not state.cpu.halted:
        # Check for Ctrl+C interrupt (byte 0x03)
        try:
            readable, _, _ = select.select([sock], [], [], 0)
        except (ValueError, OSError):
            readable = []
        if readable:
            try:
                data = sock.recv(1, socket.MSG_PEEK)
                if data and data[0] == 0x03:
                    # Consume the interrupt byte
                    sock.recv(1)
                    return "S02"  # SIGINT
            except OSError:
                return "S02"

        try:
            state.cpu.step()
        except MemoryError:
            return "S0b"  # SIGSEGV

        # Check if we hit a breakpoint (check AFTER step, at new PC)
        if state.cpu.pc in state.breakpoints:
            return "S05"  # SIGTRAP

    return f"W{state.cpu.exit_code:02x}"


def handle_insert_bp(state: GdbState, packet: str) -> str:
    """Handle ``Z0,addr,kind`` -- insert software breakpoint.

    Args:
        state: Current GDB session state.
        packet: ``addr,kind`` string (after the ``Z0,``).

    Returns:
        ``OK`` on success, empty string for unsupported types.
    """
    parts = packet.split(",")
    if len(parts) < 2:
        return "E01"

    try:
        addr = int(parts[0], 16)
    except ValueError:
        return "E01"

    state.breakpoints.add(addr)
    return "OK"


def handle_remove_bp(state: GdbState, packet: str) -> str:
    """Handle ``z0,addr,kind`` -- remove software breakpoint.

    Args:
        state: Current GDB session state.
        packet: ``addr,kind`` string (after the ``z0,``).

    Returns:
        ``OK`` on success, empty string for unsupported types.
    """
    parts = packet.split(",")
    if len(parts) < 2:
        return "E01"

    try:
        addr = int(parts[0], 16)
    except ValueError:
        return "E01"

    state.breakpoints.discard(addr)
    return "OK"


def _handle_breakpoint_insert(state: GdbState, packet: str) -> str:
    """Route Z packets by breakpoint type.

    Only software breakpoints (type 0) are supported.

    Args:
        state: Current GDB session state.
        packet: Packet body after ``Z`` (e.g. ``0,addr,kind``).

    Returns:
        Response string.
    """
    if not packet or packet[0] != "0":
        return ""  # Unsupported breakpoint type

    # Skip "0,"
    if len(packet) > 2 and packet[1] == ",":
        return handle_insert_bp(state, packet[2:])
    return "E01"


def _handle_breakpoint_remove(state: GdbState, packet: str) -> str:
    """Route z packets by breakpoint type.

    Only software breakpoints (type 0) are supported.

    Args:
        state: Current GDB session state.
        packet: Packet body after ``z`` (e.g. ``0,addr,kind``).

    Returns:
        Response string.
    """
    if not packet or packet[0] != "0":
        return ""  # Unsupported breakpoint type

    # Skip "0,"
    if len(packet) > 2 and packet[1] == ",":
        return handle_remove_bp(state, packet[2:])
    return "E01"


def handle_query(state: GdbState, packet: str) -> str:
    """Route ``q...`` and ``Q...`` query packets.

    Args:
        state: Current GDB session state.
        packet: Full query packet string (including ``q``/``Q``).

    Returns:
        Response body string.
    """
    if packet.startswith("qSupported"):
        return "PacketSize=4096;QStartNoAckMode+;qXfer:features:read+"

    if packet == "QStartNoAckMode":
        state.no_ack = True
        return "OK"

    if packet.startswith("qXfer:features:read:target.xml:"):
        # Format: qXfer:features:read:target.xml:offset,length
        params = packet.split(":")[-1]
        if "," not in params:
            return "E01"
        offset_str, length_str = params.split(",", 1)
        try:
            offset = int(offset_str, 16)
            length = int(length_str, 16)
        except ValueError:
            return "E01"

        xml = state.target_xml
        chunk = xml[offset : offset + length]

        if offset + length >= len(xml):
            # Last chunk
            return "l" + chunk
        else:
            # More data available
            return "m" + chunk

    if packet == "qAttached":
        return "1"

    if packet == "qC":
        return "QC1"

    if packet.startswith("qOffsets"):
        return ""

    # Unknown query -- empty response
    return ""
