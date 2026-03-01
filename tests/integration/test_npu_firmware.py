"""Integration test for NPU firmware: runs npu_test.elf and verifies output."""

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


def _run_elf(elf_path: pathlib.Path) -> tuple[bytes, int]:
    """Load and run an ELF, capturing stdout output and exit code.

    Returns:
        Tuple of (stdout_bytes, exit_code).
    """
    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    bus.register(BASE, RAM_SIZE, ram)

    cpu = CPU(bus)
    stdout = io.BytesIO()
    handler = SyscallHandler(stdout=stdout)
    cpu.syscall_handler = handler

    entry = load_elf(str(elf_path), ram)
    cpu.pc = entry
    cpu.registers.write(2, STACK_TOP)

    cpu.run()

    return stdout.getvalue(), cpu.exit_code


@pytest.mark.skipif(not _HAS_TOOLCHAIN, reason="riscv64 toolchain not installed")
class TestNpuFirmware:
    """Integration tests for firmware/npu_test."""

    def test_npu_test_all_pass(self) -> None:
        """npu_test.elf prints PASS for every test and exits with code 0."""
        elf_path = FIRMWARE_DIR / "npu_test" / "npu_test.elf"
        if not elf_path.exists():
            pytest.skip("npu_test.elf not built (run: cd firmware/npu_test && make)")

        output, exit_code = _run_elf(elf_path)
        text = output.decode("utf-8")

        # Verify exit code
        assert exit_code == 0, f"npu_test.elf exited with code {exit_code}"

        # Verify ALL PASS is in output
        assert "ALL PASS" in text, f"Expected 'ALL PASS' in output:\n{text}"

        # Verify no FAIL lines
        lines = text.strip().split("\n")
        for line in lines:
            assert not line.startswith("FAIL"), f"Test failed: {line}"

    def test_npu_test_individual_results(self) -> None:
        """Each individual test line starts with PASS."""
        elf_path = FIRMWARE_DIR / "npu_test" / "npu_test.elf"
        if not elf_path.exists():
            pytest.skip("npu_test.elf not built (run: cd firmware/npu_test && make)")

        output, exit_code = _run_elf(elf_path)
        text = output.decode("utf-8")
        lines = text.strip().split("\n")

        # Filter test result lines (exclude "ALL PASS" summary)
        test_lines = [l for l in lines if l.startswith("PASS") or l.startswith("FAIL")]
        assert len(test_lines) >= 17, (
            f"Expected at least 17 test results, got {len(test_lines)}"
        )
        for line in test_lines:
            assert line.startswith("PASS"), f"Test failed: {line}"
