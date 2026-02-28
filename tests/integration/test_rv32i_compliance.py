"""RV32I compliance tests against riscv-tests suite.

Runs all rv32ui-p-* test binaries from the official RISC-V test suite.
Each test signals pass by writing 1 to the memory-mapped tohost address,
or failure by writing (test_case << 1 | 1).

These tests require prebuilt ELF binaries in tests/fixtures/riscv-tests/.
Run tests/fixtures/riscv-tests/build.sh to generate them.
"""

import pathlib

import pytest

from riscv_npu.cpu.cpu import CPU
from riscv_npu.loader.elf import find_symbol, parse_elf
from riscv_npu.memory.ram import RAM

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures" / "riscv-tests"
BASE = 0x80000000
RAM_SIZE = 4 * 1024 * 1024  # 4 MB


def _run_riscv_test(elf_path: pathlib.Path) -> None:
    """Load and run a riscv-test ELF, asserting it passes."""
    data = elf_path.read_bytes()
    prog = parse_elf(data)
    tohost_addr = find_symbol(data, "tohost") or 0

    ram = RAM(BASE, RAM_SIZE)
    for seg in prog.segments:
        padded = seg.data + b"\x00" * (seg.memsz - len(seg.data))
        ram.load_segment(seg.vaddr, padded)

    cpu = CPU(ram)
    cpu.pc = prog.entry
    cpu.tohost_addr = tohost_addr
    cpu.run(max_cycles=200_000)

    assert cpu.tohost == 1, (
        f"Test failed: tohost={cpu.tohost} "
        f"(test case {cpu.tohost >> 1}), "
        f"cycles={cpu.cycle_count}"
    )


# Collect all rv32ui-p-* test binaries
_RV32UI_TESTS = sorted(FIXTURES.glob("rv32ui-p-*"))


@pytest.mark.parametrize(
    "elf_path",
    _RV32UI_TESTS,
    ids=lambda p: p.name,
)
def test_rv32ui(elf_path: pathlib.Path) -> None:
    """Run a single rv32ui compliance test."""
    if not elf_path.exists():
        pytest.skip("riscv-tests not built (run tests/fixtures/riscv-tests/build.sh)")
    _run_riscv_test(elf_path)
