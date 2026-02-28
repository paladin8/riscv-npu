"""Command-line interface for the RISC-V NPU emulator."""

import argparse
import sys

from .cpu.cpu import CPU
from .memory.ram import RAM

BASE = 0x80000000
RAM_SIZE = 1024 * 1024  # 1 MB


def main() -> None:
    """Entry point for the emulator CLI."""
    parser = argparse.ArgumentParser(description="RISC-V NPU Emulator")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run a raw binary")
    run_parser.add_argument("binary", help="Path to raw binary file")

    # Placeholder for future 'debug' command
    sub.add_parser("debug", help="Run with TUI debugger (not yet implemented)")

    args = parser.parse_args()

    if args.command == "run":
        run_binary(args.binary)
    else:
        parser.print_help()
        sys.exit(1)


def run_binary(path: str) -> None:
    """Load a raw binary at 0x80000000 and run until halt or 1M cycles."""
    ram = RAM(BASE, RAM_SIZE)

    with open(path, "rb") as f:
        data = f.read()

    # Load binary into RAM
    for i, byte in enumerate(data):
        ram.write8(BASE + i, byte)

    cpu = CPU(ram)
    cpu.pc = BASE
    cpu.run()

    print(f"Halted after {cpu.cycle_count} cycles.")
    print(f"  x10 (a0) = {cpu.registers.read(10)}")
    print(f"  x11 (a1) = {cpu.registers.read(11)}")
