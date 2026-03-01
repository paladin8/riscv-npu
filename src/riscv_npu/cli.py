"""Command-line interface for the RISC-V NPU emulator."""

import argparse
import sys

from .cpu.cpu import CPU
from .devices.uart import UART, UART_BASE, UART_SIZE
from .loader.elf import load_elf, parse_elf
from .memory.bus import MemoryBus
from .memory.ram import RAM
from .syscall.handler import SyscallHandler

BASE = 0x80000000
RAM_SIZE = 1024 * 1024  # 1 MB
STACK_TOP = 0x80FFFFF0  # Top of RAM, 16-byte aligned


def main() -> None:
    """Entry point for the emulator CLI."""
    parser = argparse.ArgumentParser(description="RISC-V NPU Emulator")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run a binary or ELF file")
    run_parser.add_argument("binary", help="Path to binary or ELF file")

    debug_parser = sub.add_parser("debug", help="Run with TUI debugger")
    debug_parser.add_argument("binary", help="Path to ELF file to debug")

    args = parser.parse_args()

    if args.command == "run":
        run_binary(args.binary)
    elif args.command == "debug":
        from .tui import run_debugger
        run_debugger(args.binary)
    else:
        parser.print_help()
        sys.exit(1)


def run_binary(path: str) -> None:
    """Load a binary or ELF file and run until halt or 1M cycles.

    Detects ELF files by magic bytes. For ELF files, loads segments to
    their virtual addresses, sets PC to the entry point, and sets SP
    to the top of RAM. For raw binaries, loads at 0x80000000.
    """
    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    uart = UART()
    bus.register(BASE, RAM_SIZE, ram)
    bus.register(UART_BASE, UART_SIZE, uart)

    cpu = CPU(bus)
    handler = SyscallHandler()
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
            prog = parse_elf(f.read())
        if prog.segments:
            end = max(s.vaddr + s.memsz for s in prog.segments)
            handler.brk = (end + 15) & ~15
    else:
        # Raw binary: load at base address
        with open(path, "rb") as f:
            data = f.read()
        ram.load_segment(BASE, data)
        cpu.pc = BASE

    cpu.run()

    print(f"Halted after {cpu.cycle_count} cycles.", file=sys.stderr)
    print(f"  x10 (a0) = {cpu.registers.read(10)}", file=sys.stderr)
    print(f"  x11 (a1) = {cpu.registers.read(11)}", file=sys.stderr)

    sys.exit(cpu.exit_code)
