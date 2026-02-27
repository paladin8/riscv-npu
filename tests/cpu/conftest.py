"""Shared fixtures for CPU tests."""

import pytest


@pytest.fixture
def make_cpu():
    """Create a fresh CPU + Memory instance with default RAM at 0x80000000."""
    # TODO: implement in Phase 1
    pass


@pytest.fixture
def exec_instruction():
    """Decode and execute a single 32-bit instruction word, return cpu state."""
    # TODO: implement in Phase 1
    pass


@pytest.fixture
def set_regs():
    """Set named registers (e.g., set_regs(cpu, x1=5, x2=10))."""
    # TODO: implement in Phase 1
    pass
