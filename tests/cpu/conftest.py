"""Shared fixtures for CPU tests."""

import pytest

from riscv_npu.cpu.cpu import CPU
from riscv_npu.memory.ram import RAM

BASE = 0x80000000
RAM_SIZE = 1024 * 1024  # 1 MB


@pytest.fixture
def make_cpu():
    """Factory fixture: returns a function that creates a fresh CPU."""
    def _make() -> CPU:
        ram = RAM(BASE, RAM_SIZE)
        cpu = CPU(ram)
        cpu.pc = BASE
        return cpu
    return _make


@pytest.fixture
def exec_instruction(make_cpu):
    """Decode and execute a single 32-bit instruction word, return the cpu."""
    def _exec(cpu: CPU | None = None, word: int = 0) -> CPU:
        if cpu is None:
            cpu = make_cpu()
        cpu.memory.write32(cpu.pc, word)
        cpu.step()
        return cpu
    return _exec


@pytest.fixture
def set_regs():
    """Set named registers (e.g., set_regs(cpu, x1=5, x2=10))."""
    def _set(cpu: CPU, **kwargs: int) -> None:
        for name, value in kwargs.items():
            if not name.startswith("x"):
                raise ValueError(f"Register name must start with 'x': {name}")
            index = int(name[1:])
            cpu.registers.write(index, value)
    return _set
