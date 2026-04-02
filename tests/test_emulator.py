"""Tests for the Emulator library API."""

import pathlib

import numpy as np
import pytest

from riscv_npu import Emulator, RunResult

FIRMWARE_DIR = pathlib.Path(__file__).parent.parent / "firmware"
HELLO_ELF = FIRMWARE_DIR / "hello" / "hello.elf"


def _has_hello_elf() -> bool:
    return HELLO_ELF.exists()


@pytest.mark.skipif(not _has_hello_elf(), reason="hello.elf not built")
class TestLoadAndRun:
    """Tests that require hello.elf."""

    def test_load_and_run_hello(self) -> None:
        """Load hello.elf, run, check stdout and exit code."""
        emu = Emulator()
        emu.load_elf(str(HELLO_ELF))
        result = emu.run()

        assert result.exit_code == 0
        assert emu.stdout == b"Hello, World!\n"

    def test_run_result_stats(self) -> None:
        """Check cycles > 0 and stats dict has entries."""
        emu = Emulator()
        emu.load_elf(str(HELLO_ELF))
        result = emu.run()

        assert result.cycles > 0
        assert isinstance(result.stats, dict)
        assert len(result.stats) > 0

    def test_symbol_lookup(self) -> None:
        """Load ELF with known symbols, verify addresses."""
        emu = Emulator()
        emu.load_elf(str(HELLO_ELF))

        # 'main' should exist in any C program
        addr = emu.symbol("main")
        assert isinstance(addr, int)
        assert addr >= 0x80000000

    def test_stdout_capture(self) -> None:
        """Run hello, check emu.stdout matches expected output."""
        emu = Emulator()
        emu.load_elf(str(HELLO_ELF))
        emu.run()

        assert emu.stdout == b"Hello, World!\n"

    def test_reset(self) -> None:
        """Load, run, reset, run again — both runs produce correct output."""
        emu = Emulator()
        emu.load_elf(str(HELLO_ELF))

        result_1 = emu.run()
        assert result_1.exit_code == 0
        assert emu.stdout == b"Hello, World!\n"

        emu.reset()

        result_2 = emu.run()
        assert result_2.exit_code == 0
        assert emu.stdout == b"Hello, World!\n"


class TestSymbolErrors:
    """Symbol lookup error handling."""

    def test_symbol_not_found(self) -> None:
        """Raises KeyError for a symbol that doesn't exist."""
        if not _has_hello_elf():
            pytest.skip("hello.elf not built")

        emu = Emulator()
        emu.load_elf(str(HELLO_ELF))

        with pytest.raises(KeyError):
            emu.symbol("nonexistent_symbol_xyz_42")

    def test_symbol_no_elf_loaded(self) -> None:
        """Raises KeyError when no ELF is loaded."""
        emu = Emulator()

        with pytest.raises(KeyError):
            emu.symbol("main")


class TestMemoryAccess:
    """Read/write memory using raw addresses (no ELF needed)."""

    def test_write_read_f32(self) -> None:
        """Write float32 array, read back, compare."""
        emu = Emulator()
        addr = 0x80010000

        data = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        emu.write_f32(addr, data)
        result = emu.read_f32(addr, n=4)

        np.testing.assert_array_equal(result, data)
        assert result.dtype == np.float32

    def test_write_read_i32(self) -> None:
        """Write int32 array, read back, compare."""
        emu = Emulator()
        addr = 0x80010000

        data = np.array([10, -20, 30, -40], dtype=np.int32)
        emu.write_i32(addr, data)
        result = emu.read_i32(addr, n=4)

        np.testing.assert_array_equal(result, data)
        assert result.dtype == np.int32

    def test_write_read_bytes(self) -> None:
        """Write raw bytes, read back, compare."""
        emu = Emulator()
        addr = 0x80010000

        data = b"\xde\xad\xbe\xef\x01\x02\x03\x04"
        emu.write_bytes(addr, data)
        result = emu.read_bytes(addr, n=8)

        assert result == data

    def test_address_based_access(self) -> None:
        """Write/read by integer address, not symbol."""
        emu = Emulator()
        addr = 0x80020000

        data = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        emu.write_f32(addr, data)
        result = emu.read_f32(addr, n=3)

        np.testing.assert_array_equal(result, data)


class TestTimeout:
    """Execution timeout handling."""

    def test_timeout(self) -> None:
        """Run with max_cycles=1, verify TimeoutError."""
        if not _has_hello_elf():
            pytest.skip("hello.elf not built")

        emu = Emulator()
        emu.load_elf(str(HELLO_ELF))

        with pytest.raises(TimeoutError):
            emu.run(max_cycles=1)
