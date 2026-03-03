"""GDB Remote Serial Protocol packet framing, checksum, and encoding.

Implements the low-level wire format: ``$packet-data#checksum`` framing
with two-character hex checksums, plus helpers for encoding 32-bit
register values in the little-endian hex format GDB expects.
"""


def checksum(data: str) -> int:
    """Compute RSP checksum: sum of ASCII values mod 256.

    Args:
        data: The packet body string.

    Returns:
        Checksum value in range 0-255.
    """
    total = 0
    for ch in data:
        total += ord(ch)
    return total & 0xFF


def build_packet(body: str) -> bytes:
    """Build a framed RSP packet: ``$body#xx``.

    Args:
        body: The packet body string.

    Returns:
        The complete framed packet as bytes.
    """
    cs = checksum(body)
    return f"${body}#{cs:02x}".encode("ascii")


def parse_packet(data: bytes) -> str | None:
    """Extract packet body from ``$body#xx`` framing.

    Searches for the first complete ``$..#xx`` sequence in the data.
    Returns the body string if the checksum is valid, None otherwise.

    Args:
        data: Raw bytes that may contain an RSP packet.

    Returns:
        The packet body string if valid, None if checksum fails
        or no complete packet is found.
    """
    # Find the start marker
    try:
        start = data.index(ord("$"))
    except ValueError:
        return None

    # Find the checksum delimiter
    try:
        hash_pos = data.index(ord("#"), start + 1)
    except ValueError:
        return None

    # Need 2 hex chars after '#'
    if hash_pos + 2 >= len(data):
        return None

    body = data[start + 1 : hash_pos].decode("ascii")
    cs_hex = data[hash_pos + 1 : hash_pos + 3].decode("ascii")

    try:
        received_cs = int(cs_hex, 16)
    except ValueError:
        return None

    if checksum(body) != received_cs:
        return None

    return body


def hex_encode(data: bytes) -> str:
    """Encode raw bytes as a hex string.

    Args:
        data: Raw bytes to encode.

    Returns:
        Lowercase hex string (e.g. ``b'\\x80'`` -> ``'80'``).
    """
    return data.hex()


def hex_decode(hex_str: str) -> bytes:
    """Decode a hex string to raw bytes.

    Args:
        hex_str: Hex string to decode (e.g. ``'deadbeef'``).

    Returns:
        Raw bytes.
    """
    return bytes.fromhex(hex_str)


def encode_reg32(value: int) -> str:
    """Encode a 32-bit value as 8 little-endian hex chars.

    GDB expects register values in target byte order (little-endian
    for RISC-V). So ``0x12345678`` becomes ``"78563412"``.

    Args:
        value: 32-bit integer value to encode.

    Returns:
        8-character hex string in little-endian order.
    """
    value = value & 0xFFFFFFFF
    b = value.to_bytes(4, byteorder="little")
    return b.hex()


def decode_reg32(hex_str: str) -> int:
    """Decode 8 little-endian hex chars to a 32-bit integer.

    Args:
        hex_str: 8-character hex string in little-endian order.

    Returns:
        32-bit integer value.
    """
    b = bytes.fromhex(hex_str)
    return int.from_bytes(b, byteorder="little") & 0xFFFFFFFF
