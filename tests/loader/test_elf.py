"""Tests for ELF loader."""

import struct
import tempfile

import pytest

from riscv_npu.loader.elf import ElfProgram, ElfSegment, load_elf, parse_elf
from riscv_npu.memory.ram import RAM


# --- Helpers to construct minimal ELF32 binaries ---

_ELF_MAGIC = b"\x7fELF"
_ELFCLASS32 = 1
_ELFDATA2LSB = 1
_EM_RISCV = 0xF3
_PT_LOAD = 1
_PT_NULL = 0
_PT_NOTE = 4

_ELF32_EHDR_SIZE = 52
_ELF32_PHDR_SIZE = 32


def _build_elf_header(
    *,
    magic: bytes = _ELF_MAGIC,
    ei_class: int = _ELFCLASS32,
    ei_data: int = _ELFDATA2LSB,
    e_machine: int = _EM_RISCV,
    e_entry: int = 0x80000000,
    e_phoff: int = _ELF32_EHDR_SIZE,
    e_phentsize: int = _ELF32_PHDR_SIZE,
    e_phnum: int = 0,
) -> bytes:
    """Build a minimal ELF32 header."""
    # e_ident: 16 bytes
    e_ident = bytearray(16)
    e_ident[0:4] = magic
    e_ident[4] = ei_class
    e_ident[5] = ei_data
    e_ident[6] = 1  # EV_CURRENT
    # Rest of e_ident is padding/zeros

    # ELF32 header after e_ident (36 bytes)
    rest = struct.pack(
        "<HHIIIIIHHHHHH",
        2,            # e_type: ET_EXEC
        e_machine,    # e_machine
        1,            # e_version: EV_CURRENT
        e_entry,      # e_entry
        e_phoff,      # e_phoff
        0,            # e_shoff (no section headers)
        0,            # e_flags
        _ELF32_EHDR_SIZE,  # e_ehsize
        e_phentsize,  # e_phentsize
        e_phnum,      # e_phnum
        0,            # e_shentsize
        0,            # e_shnum
        0,            # e_shstrndx
    )
    return bytes(e_ident) + rest


def _build_phdr(
    *,
    p_type: int = _PT_LOAD,
    p_offset: int = 0,
    p_vaddr: int = 0x80000000,
    p_paddr: int = 0,
    p_filesz: int = 0,
    p_memsz: int = 0,
    p_flags: int = 5,  # PF_R | PF_X
    p_align: int = 0x1000,
) -> bytes:
    """Build a single ELF32 program header."""
    return struct.pack(
        "<IIIIIIII",
        p_type, p_offset, p_vaddr, p_paddr,
        p_filesz, p_memsz, p_flags, p_align,
    )


def _build_elf_with_segments(
    segments: list[dict],
    entry: int = 0x80000000,
) -> bytes:
    """Build a complete ELF with the given segments.

    Each segment dict has: p_type, vaddr, data (bytes), memsz (optional, defaults to len(data)).
    """
    e_phoff = _ELF32_EHDR_SIZE
    e_phnum = len(segments)

    # Data starts after headers
    data_start = e_phoff + e_phnum * _ELF32_PHDR_SIZE

    header = _build_elf_header(
        e_entry=entry,
        e_phoff=e_phoff,
        e_phnum=e_phnum,
    )

    phdrs = bytearray()
    seg_data = bytearray()
    current_offset = data_start

    for seg in segments:
        seg_bytes = seg.get("data", b"")
        memsz = seg.get("memsz", len(seg_bytes))
        p_type = seg.get("p_type", _PT_LOAD)
        vaddr = seg.get("vaddr", 0x80000000)

        phdr = _build_phdr(
            p_type=p_type,
            p_offset=current_offset,
            p_vaddr=vaddr,
            p_filesz=len(seg_bytes),
            p_memsz=memsz,
        )
        phdrs.extend(phdr)
        seg_data.extend(seg_bytes)
        current_offset += len(seg_bytes)

    return header + bytes(phdrs) + bytes(seg_data)


# --- Tests ---

class TestParseValidElf:
    def test_single_segment(self) -> None:
        seg_data = b"\x13\x00\x00\x00"  # NOP instruction (ADDI x0, x0, 0)
        elf = _build_elf_with_segments(
            [{"vaddr": 0x80000000, "data": seg_data}],
            entry=0x80000000,
        )
        prog = parse_elf(elf)
        assert prog.entry == 0x80000000
        assert len(prog.segments) == 1
        assert prog.segments[0].vaddr == 0x80000000
        assert prog.segments[0].data == seg_data
        assert prog.segments[0].memsz == len(seg_data)

    def test_entry_point_preserved(self) -> None:
        elf = _build_elf_with_segments(
            [{"vaddr": 0x80000000, "data": b"\x00" * 4}],
            entry=0x80001000,
        )
        prog = parse_elf(elf)
        assert prog.entry == 0x80001000

    def test_returns_elfprogram_type(self) -> None:
        elf = _build_elf_with_segments(
            [{"vaddr": 0x80000000, "data": b"\x00" * 4}],
        )
        prog = parse_elf(elf)
        assert isinstance(prog, ElfProgram)
        assert isinstance(prog.segments[0], ElfSegment)


class TestBadMagic:
    def test_wrong_magic_bytes(self) -> None:
        elf = _build_elf_with_segments([{"data": b"\x00" * 4}])
        # Corrupt the magic
        bad_elf = b"\x00\x00\x00\x00" + elf[4:]
        with pytest.raises(ValueError, match="Bad ELF magic"):
            parse_elf(bad_elf)

    def test_truncated_file(self) -> None:
        with pytest.raises(ValueError, match="File too small"):
            parse_elf(b"\x7fELF")


class TestWrongClass:
    def test_elfclass64(self) -> None:
        header = _build_elf_header(ei_class=2)  # ELFCLASS64
        with pytest.raises(ValueError, match="Unsupported ELF class"):
            parse_elf(header)


class TestWrongEndian:
    def test_big_endian(self) -> None:
        header = _build_elf_header(ei_data=2)  # ELFDATA2MSB
        with pytest.raises(ValueError, match="Unsupported ELF endianness"):
            parse_elf(header)


class TestWrongMachine:
    def test_non_riscv(self) -> None:
        header = _build_elf_header(e_machine=0x03)  # EM_386
        with pytest.raises(ValueError, match="Unsupported machine type"):
            parse_elf(header)


class TestMultipleSegments:
    def test_two_load_segments(self) -> None:
        text_data = b"\x13\x00\x00\x00" * 4  # 4 NOPs
        data_data = b"\x01\x02\x03\x04"

        elf = _build_elf_with_segments([
            {"vaddr": 0x80000000, "data": text_data},
            {"vaddr": 0x80010000, "data": data_data},
        ])
        prog = parse_elf(elf)
        assert len(prog.segments) == 2
        assert prog.segments[0].vaddr == 0x80000000
        assert prog.segments[0].data == text_data
        assert prog.segments[1].vaddr == 0x80010000
        assert prog.segments[1].data == data_data


class TestBssSegment:
    def test_memsz_greater_than_filesz(self) -> None:
        """A .bss segment has memsz > filesz (uninitialized data)."""
        seg_data = b"\x01\x02\x03\x04"
        elf = _build_elf_with_segments([
            {"vaddr": 0x80000000, "data": seg_data, "memsz": 256},
        ])
        prog = parse_elf(elf)
        assert len(prog.segments) == 1
        seg = prog.segments[0]
        assert seg.data == seg_data
        assert seg.memsz == 256
        assert len(seg.data) == 4  # Only the file data, not zero-padded yet


class TestNonLoadSegmentsSkipped:
    def test_pt_null_skipped(self) -> None:
        elf = _build_elf_with_segments([
            {"p_type": _PT_NULL, "vaddr": 0, "data": b""},
            {"p_type": _PT_LOAD, "vaddr": 0x80000000, "data": b"\x13\x00\x00\x00"},
        ])
        prog = parse_elf(elf)
        assert len(prog.segments) == 1
        assert prog.segments[0].vaddr == 0x80000000

    def test_pt_note_skipped(self) -> None:
        elf = _build_elf_with_segments([
            {"p_type": _PT_NOTE, "vaddr": 0, "data": b"\x00" * 8},
            {"p_type": _PT_LOAD, "vaddr": 0x80000000, "data": b"\x13\x00\x00\x00"},
        ])
        prog = parse_elf(elf)
        assert len(prog.segments) == 1
        assert prog.segments[0].vaddr == 0x80000000

    def test_no_load_segments(self) -> None:
        elf = _build_elf_with_segments([
            {"p_type": _PT_NULL, "vaddr": 0, "data": b""},
        ])
        prog = parse_elf(elf)
        assert len(prog.segments) == 0


class TestLoadElf:
    def test_load_elf_integration(self, tmp_path) -> None:
        """load_elf reads a file, parses it, and loads segments into RAM."""
        seg_data = b"\x13\x00\x00\x00" * 4  # 4 NOP instructions
        elf_bytes = _build_elf_with_segments(
            [{"vaddr": 0x80000000, "data": seg_data}],
            entry=0x80000000,
        )
        elf_file = tmp_path / "test.elf"
        elf_file.write_bytes(elf_bytes)

        ram = RAM(0x80000000, 1024 * 1024)
        entry = load_elf(str(elf_file), ram)

        assert entry == 0x80000000
        # Verify the NOP instructions were loaded
        for i in range(4):
            assert ram.read32(0x80000000 + i * 4) == 0x00000013

    def test_load_elf_bss_zero_filled(self, tmp_path) -> None:
        """load_elf zero-fills .bss (memsz > filesz)."""
        seg_data = b"\xAA\xBB\xCC\xDD"
        elf_bytes = _build_elf_with_segments(
            [{"vaddr": 0x80000000, "data": seg_data, "memsz": 16}],
            entry=0x80000000,
        )
        elf_file = tmp_path / "test_bss.elf"
        elf_file.write_bytes(elf_bytes)

        ram = RAM(0x80000000, 1024 * 1024)
        load_elf(str(elf_file), ram)

        # First 4 bytes have data
        assert ram.read32(0x80000000) == 0xDDCCBBAA
        # Remaining 12 bytes are zero-filled
        assert ram.read32(0x80000004) == 0
        assert ram.read32(0x80000008) == 0
        assert ram.read32(0x8000000C) == 0

    def test_load_elf_multiple_segments(self, tmp_path) -> None:
        """load_elf loads multiple PT_LOAD segments."""
        text = b"\x01\x02\x03\x04"
        data_seg = b"\x05\x06\x07\x08"
        elf_bytes = _build_elf_with_segments([
            {"vaddr": 0x80000000, "data": text},
            {"vaddr": 0x80010000, "data": data_seg},
        ], entry=0x80000000)
        elf_file = tmp_path / "test_multi.elf"
        elf_file.write_bytes(elf_bytes)

        ram = RAM(0x80000000, 1024 * 1024)
        load_elf(str(elf_file), ram)

        assert ram.read32(0x80000000) == 0x04030201
        assert ram.read32(0x80010000) == 0x08070605
