"""ELF binary parser and loader: validates, extracts, and loads ELF32 RISC-V files."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..memory.ram import RAM


# ELF constants
_ELF_MAGIC = b"\x7fELF"
_ELFCLASS32 = 1
_ELFDATA2LSB = 1  # Little-endian
_EM_RISCV = 0xF3
_PT_LOAD = 1

# ELF32 header size and program header entry size
_ELF32_EHDR_SIZE = 52
_ELF32_PHDR_SIZE = 32


@dataclass(frozen=True)
class ElfSegment:
    """A loadable segment from an ELF file."""

    vaddr: int
    data: bytes
    memsz: int


@dataclass(frozen=True)
class ElfProgram:
    """Parsed ELF program: entry point and loadable segments."""

    entry: int
    segments: list[ElfSegment]


def parse_elf(data: bytes) -> ElfProgram:
    """Parse a 32-bit little-endian RISC-V ELF binary.

    Validates the ELF header, reads program headers, and extracts all
    PT_LOAD segments with their virtual addresses and contents.

    Args:
        data: Raw bytes of the ELF file.

    Returns:
        An ElfProgram with the entry point and list of loadable segments.

    Raises:
        ValueError: If the ELF header is invalid or unsupported.
    """
    if len(data) < _ELF32_EHDR_SIZE:
        raise ValueError(
            f"File too small for ELF header: {len(data)} bytes "
            f"(need at least {_ELF32_EHDR_SIZE})"
        )

    # Validate magic number
    magic = data[0:4]
    if magic != _ELF_MAGIC:
        raise ValueError(f"Bad ELF magic: {magic!r} (expected {_ELF_MAGIC!r})")

    # Validate class (32-bit)
    ei_class = data[4]
    if ei_class != _ELFCLASS32:
        raise ValueError(
            f"Unsupported ELF class: {ei_class} (expected {_ELFCLASS32} for 32-bit)"
        )

    # Validate endianness (little-endian)
    ei_data = data[5]
    if ei_data != _ELFDATA2LSB:
        raise ValueError(
            f"Unsupported ELF endianness: {ei_data} "
            f"(expected {_ELFDATA2LSB} for little-endian)"
        )

    # Parse ELF header fields (all little-endian)
    # e_machine at offset 18 (2 bytes)
    (e_machine,) = struct.unpack_from("<H", data, 18)
    if e_machine != _EM_RISCV:
        raise ValueError(
            f"Unsupported machine type: 0x{e_machine:04X} "
            f"(expected 0x{_EM_RISCV:04X} for RISC-V)"
        )

    # e_entry at offset 24 (4 bytes)
    (e_entry,) = struct.unpack_from("<I", data, 24)

    # e_phoff at offset 28 (4 bytes) - program header table offset
    (e_phoff,) = struct.unpack_from("<I", data, 28)

    # e_phentsize at offset 42 (2 bytes) - size of each program header entry
    (e_phentsize,) = struct.unpack_from("<H", data, 42)

    # e_phnum at offset 44 (2 bytes) - number of program header entries
    (e_phnum,) = struct.unpack_from("<H", data, 44)

    # Read program headers and extract PT_LOAD segments
    segments: list[ElfSegment] = []

    for i in range(e_phnum):
        ph_offset = e_phoff + i * e_phentsize

        if ph_offset + _ELF32_PHDR_SIZE > len(data):
            raise ValueError(
                f"Program header {i} extends beyond file "
                f"(offset {ph_offset}, file size {len(data)})"
            )

        # Parse program header: p_type, p_offset, p_vaddr, p_paddr,
        #                        p_filesz, p_memsz, p_flags, p_align
        (p_type, p_offset, p_vaddr, _p_paddr, p_filesz, p_memsz, _p_flags,
         _p_align) = struct.unpack_from("<IIIIIIII", data, ph_offset)

        if p_type != _PT_LOAD:
            continue

        # Validate segment data is within file bounds
        if p_offset + p_filesz > len(data):
            raise ValueError(
                f"Segment {i} data extends beyond file "
                f"(offset {p_offset}, filesz {p_filesz}, file size {len(data)})"
            )

        seg_data = data[p_offset : p_offset + p_filesz]
        segments.append(ElfSegment(vaddr=p_vaddr, data=seg_data, memsz=p_memsz))

    return ElfProgram(entry=e_entry, segments=segments)


def load_elf(path: str, ram: RAM) -> int:
    """Load an ELF binary into RAM.

    Reads the file at `path`, parses it as an ELF32 RISC-V binary,
    and loads all PT_LOAD segments into memory. Segments with memsz > filesz
    are zero-padded (handles .bss sections).

    Args:
        path: Path to the ELF file.
        ram: RAM instance to load segments into.

    Returns:
        The entry point address from the ELF header.

    Raises:
        ValueError: If the ELF file is invalid.
        MemoryError: If a segment doesn't fit in RAM.
    """
    with open(path, "rb") as f:
        data = f.read()

    prog = parse_elf(data)

    for seg in prog.segments:
        # Zero-pad to memsz (handles .bss)
        padded = seg.data + b"\x00" * (seg.memsz - len(seg.data))
        ram.load_segment(seg.vaddr, padded)

    return prog.entry
