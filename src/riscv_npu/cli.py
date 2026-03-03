"""Command-line interface for the RISC-V NPU emulator."""

import argparse
import sys
from pathlib import Path

from .cpu.cpu import CPU
from .devices.uart import UART, UART_BASE, UART_SIZE
from .loader.elf import find_symbol, load_elf, parse_elf
from .memory.bus import MemoryBus
from .memory.ram import RAM
from .syscall.handler import SyscallHandler

BASE = 0x80000000
RAM_SIZE = 4 * 1024 * 1024  # 4 MB
STACK_TOP = BASE + RAM_SIZE - 16  # Top of RAM, 16-byte aligned


def _parse_write_arg(value: str) -> tuple[str, str]:
    """Parse a --write SYMBOL:FILE argument.

    Args:
        value: String in the form "SYMBOL:FILE".

    Returns:
        Tuple of (symbol_name, file_path).

    Raises:
        argparse.ArgumentTypeError: If the format is invalid.
    """
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            f"invalid format '{value}', expected SYMBOL:FILE"
        )
    symbol, path = value.split(":", 1)
    if not symbol or not path:
        raise argparse.ArgumentTypeError(
            f"invalid format '{value}', expected SYMBOL:FILE"
        )
    return symbol, path


def main() -> None:
    """Entry point for the emulator CLI."""
    parser = argparse.ArgumentParser(description="RISC-V NPU Emulator")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run a binary or ELF file")
    run_parser.add_argument("binary", help="Path to binary or ELF file")
    run_parser.add_argument(
        "--write", action="append", type=_parse_write_arg, default=[],
        metavar="SYMBOL:FILE",
        help="Write FILE contents to ELF symbol address (repeatable)",
    )
    run_parser.add_argument(
        "--fs-root", type=Path, default=None,
        help="Sandbox guest file I/O to this directory",
    )

    debug_parser = sub.add_parser("debug", help="Run with TUI debugger")
    debug_parser.add_argument("binary", help="Path to ELF file to debug")
    debug_parser.add_argument(
        "--write", action="append", type=_parse_write_arg, default=[],
        metavar="SYMBOL:FILE",
        help="Write FILE contents to ELF symbol address (repeatable)",
    )
    debug_parser.add_argument(
        "--fs-root", type=Path, default=None,
        help="Sandbox guest file I/O to this directory",
    )

    gdb_parser = sub.add_parser("gdb", help="Start GDB remote debug server")
    gdb_parser.add_argument("binary", help="Path to ELF file to debug")
    gdb_parser.add_argument(
        "--port", type=int, default=1234,
        help="TCP port to listen on (default: 1234)",
    )
    gdb_parser.add_argument(
        "--write", action="append", type=_parse_write_arg, default=[],
        metavar="SYMBOL:FILE",
        help="Write FILE contents to ELF symbol address (repeatable)",
    )
    gdb_parser.add_argument(
        "--fs-root", type=Path, default=None,
        help="Sandbox guest file I/O to this directory",
    )

    args = parser.parse_args()

    if args.command == "run":
        run_binary(args.binary, args.write, fs_root=args.fs_root)
    elif args.command == "debug":
        from .tui import run_debugger
        run_debugger(args.binary, args.write, fs_root=args.fs_root)
    elif args.command == "gdb":
        run_gdb_server(args.binary, args.port, args.write, fs_root=args.fs_root)
    else:
        parser.print_help()
        sys.exit(1)


def _apply_writes(
    writes: list[tuple[str, str]], elf_data: bytes, ram: RAM,
) -> None:
    """Write file contents to ELF symbol addresses in RAM.

    Args:
        writes: List of (symbol_name, file_path) pairs.
        elf_data: Raw ELF file bytes for symbol lookup.
        ram: RAM device to write data into.

    Raises:
        SystemExit: If a symbol is not found or file cannot be read.
    """
    for symbol, file_path in writes:
        addr = find_symbol(elf_data, symbol)
        if addr is None:
            print(f"Error: symbol '{symbol}' not found in ELF", file=sys.stderr)
            sys.exit(1)
        try:
            with open(file_path, "rb") as f:
                data = f.read()
        except OSError as e:
            print(f"Error: cannot read '{file_path}': {e}", file=sys.stderr)
            sys.exit(1)
        ram.load_segment(addr, data)
        print(
            f"Loaded {len(data)} bytes from {file_path} "
            f"at {symbol} (0x{addr:08X})",
            file=sys.stderr,
        )


def run_binary(
    path: str,
    writes: list[tuple[str, str]] | None = None,
    fs_root: Path | None = None,
) -> None:
    """Load a binary or ELF file and run until halt or 1M cycles.

    Detects ELF files by magic bytes. For ELF files, loads segments to
    their virtual addresses, sets PC to the entry point, and sets SP
    to the top of RAM. For raw binaries, loads at 0x80000000.

    Args:
        path: Path to the binary or ELF file.
        writes: Optional list of (symbol, file_path) pairs to write into RAM.
        fs_root: Optional directory to sandbox guest file I/O.
    """
    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    uart = UART()
    bus.register(BASE, RAM_SIZE, ram)
    bus.register(UART_BASE, UART_SIZE, uart)

    cpu = CPU(bus)
    handler = SyscallHandler(fs_root=fs_root)
    cpu.syscall_handler = handler

    # Peek at the first 4 bytes to detect ELF
    with open(path, "rb") as f:
        magic = f.read(4)

    if magic == b"\x7fELF":
        entry = load_elf(path, ram)
        cpu.pc = entry
        cpu.registers.write(2, STACK_TOP)  # SP = x2
        # Set initial program break after loaded segments (16-byte aligned)
        with open(path, "rb") as f:
            elf_data = f.read()
        prog = parse_elf(elf_data)
        if prog.segments:
            end = max(s.vaddr + s.memsz for s in prog.segments)
            handler.brk = (end + 15) & ~15
        # Apply --write arguments
        if writes:
            _apply_writes(writes, elf_data, ram)
    else:
        if writes:
            print("Error: --write requires an ELF file", file=sys.stderr)
            sys.exit(1)
        # Raw binary: load at base address
        with open(path, "rb") as f:
            data = f.read()
        ram.load_segment(BASE, data)
        cpu.pc = BASE

    cpu.run()
    handler.close_all()

    print(f"Halted after {cpu.cycle_count} cycles.", file=sys.stderr)
    print(f"  x10 (a0) = {cpu.registers.read(10)}", file=sys.stderr)
    print(f"  x11 (a1) = {cpu.registers.read(11)}", file=sys.stderr)

    sys.exit(cpu.exit_code)


def run_gdb_server(
    path: str,
    port: int = 1234,
    writes: list[tuple[str, str]] | None = None,
    fs_root: Path | None = None,
) -> None:
    """Load an ELF file and start the GDB RSP debug server.

    Sets up the CPU with the loaded firmware, then hands control
    to the GDB stub which blocks until the debug session ends.

    Args:
        path: Path to the ELF file to debug.
        port: TCP port to listen on for GDB connections.
        writes: Optional list of (symbol, file_path) pairs to write into RAM.
        fs_root: Optional directory to sandbox guest file I/O.
    """
    from .gdb import serve

    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    uart = UART()
    bus.register(BASE, RAM_SIZE, ram)
    bus.register(UART_BASE, UART_SIZE, uart)

    cpu = CPU(bus)
    handler = SyscallHandler(fs_root=fs_root)
    cpu.syscall_handler = handler

    # Peek at the first 4 bytes to detect ELF
    with open(path, "rb") as f:
        magic = f.read(4)

    if magic != b"\x7fELF":
        print("Error: GDB server requires an ELF file", file=sys.stderr)
        sys.exit(1)

    entry = load_elf(path, ram)
    cpu.pc = entry
    cpu.registers.write(2, STACK_TOP)  # SP = x2

    # Set initial program break after loaded segments (16-byte aligned)
    with open(path, "rb") as f:
        elf_data = f.read()
    prog = parse_elf(elf_data)
    if prog.segments:
        end = max(s.vaddr + s.memsz for s in prog.segments)
        handler.brk = (end + 15) & ~15

    # Apply --write arguments
    if writes:
        _apply_writes(writes, elf_data, ram)

    serve(cpu, port=port)
