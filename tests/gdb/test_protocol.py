"""Tests for GDB RSP protocol framing and encoding."""

from riscv_npu.gdb.protocol import (
    build_packet,
    checksum,
    decode_reg32,
    encode_reg32,
    hex_decode,
    hex_encode,
    parse_packet,
)


def test_checksum_empty() -> None:
    """Empty string checksum is 0."""
    assert checksum("") == 0


def test_checksum_known() -> None:
    """Known packet body checksum matches expected value."""
    # "OK" -> ord('O') + ord('K') = 79 + 75 = 154 = 0x9a
    assert checksum("OK") == 0x9A


def test_build_packet() -> None:
    """Returns $body#xx with correct checksum."""
    pkt = build_packet("OK")
    assert pkt == b"$OK#9a"


def test_build_packet_empty() -> None:
    """Building a packet with empty body produces $#00."""
    pkt = build_packet("")
    assert pkt == b"$#00"


def test_parse_packet_valid() -> None:
    """Extracts body from valid packet."""
    body = parse_packet(b"$OK#9a")
    assert body == "OK"


def test_parse_packet_bad_checksum() -> None:
    """Returns None for wrong checksum."""
    assert parse_packet(b"$OK#00") is None


def test_parse_packet_no_framing() -> None:
    """Returns None for data without $..#xx framing."""
    assert parse_packet(b"just some data") is None
    assert parse_packet(b"$incomplete") is None
    assert parse_packet(b"no dollar#9a") is None


def test_parse_packet_with_leading_noise() -> None:
    """Packet preceded by ack/nack bytes is still parsed."""
    body = parse_packet(b"+$OK#9a")
    assert body == "OK"


def test_hex_encode_decode_roundtrip() -> None:
    """Encode then decode returns original bytes."""
    original = b"\xde\xad\xbe\xef"
    encoded = hex_encode(original)
    decoded = hex_decode(encoded)
    assert decoded == original


def test_hex_encode() -> None:
    """Hex encode produces lowercase hex."""
    assert hex_encode(b"\x80\x00\x00\x00") == "80000000"


def test_hex_decode() -> None:
    """Hex decode handles various hex inputs."""
    assert hex_decode("deadbeef") == b"\xde\xad\xbe\xef"


def test_encode_reg32_little_endian() -> None:
    """0x80000000 encodes as '00000080' (little-endian)."""
    assert encode_reg32(0x80000000) == "00000080"


def test_encode_reg32_zero() -> None:
    """Zero encodes as 8 zero chars."""
    assert encode_reg32(0) == "00000000"


def test_encode_reg32_small() -> None:
    """Small value encodes in first byte position."""
    assert encode_reg32(0x01) == "01000000"


def test_decode_reg32_little_endian() -> None:
    """'00000080' decodes as 0x80000000."""
    assert decode_reg32("00000080") == 0x80000000


def test_encode_decode_reg32_roundtrip() -> None:
    """Encode then decode returns original value."""
    for val in [0, 1, 0x12345678, 0x80000000, 0xDEADBEEF, 0xFFFFFFFF]:
        assert decode_reg32(encode_reg32(val)) == val


def test_build_parse_roundtrip() -> None:
    """Build a packet, then parse it -- body matches."""
    body = "m80000000,10"
    pkt = build_packet(body)
    assert parse_packet(pkt) == body
