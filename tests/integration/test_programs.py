"""Integration tests for compiled firmware programs."""

import io
import pathlib
import shutil

import pytest

from riscv_npu.cpu.cpu import CPU
from riscv_npu.loader.elf import load_elf
from riscv_npu.memory.bus import MemoryBus
from riscv_npu.memory.ram import RAM
from riscv_npu.syscall.handler import SyscallHandler

BASE = 0x80000000
RAM_SIZE = 1024 * 1024  # 1 MB
STACK_TOP = 0x80FFFFF0

FIRMWARE_DIR = pathlib.Path(__file__).parent.parent.parent / "firmware"

_HAS_TOOLCHAIN = shutil.which("riscv64-unknown-elf-gcc") is not None


def _run_elf(elf_path: pathlib.Path, stdin_data: bytes = b"") -> tuple[bytes, int]:
    """Load and run an ELF, capturing stdout output and exit code.

    Returns:
        Tuple of (stdout_bytes, exit_code).
    """
    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    bus.register(BASE, RAM_SIZE, ram)

    cpu = CPU(bus)
    stdout = io.BytesIO()
    stdin = io.BytesIO(stdin_data)
    handler = SyscallHandler(stdout=stdout, stdin=stdin)
    cpu.syscall_handler = handler

    entry = load_elf(str(elf_path), ram)
    cpu.pc = entry
    cpu.registers.write(2, STACK_TOP)

    cpu.run()

    return stdout.getvalue(), cpu.exit_code


@pytest.mark.skipif(not _HAS_TOOLCHAIN, reason="riscv64 toolchain not installed")
class TestHelloWorld:
    """Integration test for firmware/hello."""

    def test_hello_world(self) -> None:
        """hello.elf prints 'Hello, World!\\n' and exits with code 0."""
        elf_path = FIRMWARE_DIR / "hello" / "hello.elf"
        if not elf_path.exists():
            pytest.skip("hello.elf not built (run: cd firmware/hello && make)")

        output, exit_code = _run_elf(elf_path)
        assert output == b"Hello, World!\n"
        assert exit_code == 0
