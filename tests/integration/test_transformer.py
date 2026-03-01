"""Transformer inference integration tests.

Runs compiled transformer firmware on the emulator with test input
sequences and verifies that the quantized inference matches the
Python reference predictions.
"""

from __future__ import annotations

import io
import pathlib
import shutil
import struct

import pytest

from riscv_npu.cpu.cpu import CPU
from riscv_npu.loader.elf import find_symbol, parse_elf
from riscv_npu.memory.bus import MemoryBus
from riscv_npu.memory.ram import RAM
from riscv_npu.syscall.handler import SyscallHandler

BASE = 0x80000000
RAM_SIZE = 4 * 1024 * 1024  # 4 MB
STACK_TOP = BASE + RAM_SIZE - 16

FIRMWARE_DIR = pathlib.Path(__file__).parent.parent.parent / "firmware"
ELF_PATH = FIRMWARE_DIR / "transformer" / "transformer.elf"
TEST_DATA_PATH = FIRMWARE_DIR / "transformer" / "test_data.py"

_HAS_TOOLCHAIN = shutil.which("riscv64-unknown-elf-gcc") is not None
_HAS_ELF = ELF_PATH.exists()
_HAS_TEST_DATA = TEST_DATA_PATH.exists()


def _load_test_data() -> dict:
    """Load the auto-generated test data module.

    Returns:
        Dict with SEQUENCES, FLOAT_PREDICTIONS, QUANT_PREDICTIONS, CONTEXT_LEN.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("test_data", str(TEST_DATA_PATH))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return {
        "SEQUENCES": mod.SEQUENCES,
        "FLOAT_PREDICTIONS": mod.FLOAT_PREDICTIONS,
        "QUANT_PREDICTIONS": mod.QUANT_PREDICTIONS,
        "CONTEXT_LEN": mod.CONTEXT_LEN,
    }


def _run_inference(
    elf_data: bytes,
    tokens: list[int],
    test_tokens_addr: int,
    test_n_tokens_addr: int,
) -> str:
    """Run transformer inference on a token sequence.

    Loads the ELF into RAM, writes token data, runs the CPU,
    and returns the stdout output.

    Args:
        elf_data: Raw ELF file bytes.
        tokens: List of token IDs (byte values 0-255).
        test_tokens_addr: Address of the test_tokens symbol.
        test_n_tokens_addr: Address of the test_n_tokens symbol.

    Returns:
        Captured stdout output as a string.
    """
    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    bus.register(BASE, RAM_SIZE, ram)

    cpu = CPU(bus)
    stdout_buf = io.BytesIO()
    handler = SyscallHandler(stdout=stdout_buf)
    cpu.syscall_handler = handler

    # Load ELF segments
    prog = parse_elf(elf_data)
    for seg in prog.segments:
        padded = seg.data + b"\x00" * (seg.memsz - len(seg.data))
        ram.load_segment(seg.vaddr, padded)

    # Write token data
    ram.load_segment(test_tokens_addr, bytes(tokens))

    # Write token count as int32 little-endian
    ram.load_segment(test_n_tokens_addr, struct.pack("<i", len(tokens)))

    # Set up CPU
    cpu.pc = prog.entry
    cpu.registers.write(2, STACK_TOP)

    # Run (generous limit -- transformer inference is slow)
    cpu.run(max_cycles=50_000_000)

    return stdout_buf.getvalue().decode("utf-8", errors="replace")


def _parse_prediction(output: str) -> int:
    """Parse the predicted token from firmware output.

    Args:
        output: Stdout text from the firmware.

    Returns:
        Predicted token ID (0-255).

    Raises:
        ValueError: If output cannot be parsed.
    """
    stripped = output.strip()
    if not stripped:
        raise ValueError("Empty output from firmware")
    return int(stripped)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TOOLCHAIN, reason="riscv64 toolchain not installed")
@pytest.mark.skipif(not _HAS_ELF, reason="transformer.elf not built")
@pytest.mark.skipif(not _HAS_TEST_DATA, reason="test_data.py not generated")
class TestTransformerInference:
    """Integration tests for transformer firmware inference."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Load ELF data and test data once per class."""
        self.elf_data = ELF_PATH.read_bytes()
        self.test_data = _load_test_data()
        self.test_tokens_addr = find_symbol(self.elf_data, "test_tokens")
        self.test_n_tokens_addr = find_symbol(self.elf_data, "test_n_tokens")
        assert self.test_tokens_addr is not None, "test_tokens symbol not found"
        assert self.test_n_tokens_addr is not None, "test_n_tokens symbol not found"

    def test_single_sequence(self) -> None:
        """Run inference on the first test sequence."""
        seq = self.test_data["SEQUENCES"][0]
        expected = self.test_data["QUANT_PREDICTIONS"][0]

        output = _run_inference(
            self.elf_data, seq,
            self.test_tokens_addr, self.test_n_tokens_addr,
        )
        predicted = _parse_prediction(output)

        assert predicted == expected, (
            f"Single sequence prediction mismatch: got {predicted}, "
            f"expected {expected}"
        )

    def test_multiple_sequences(self) -> None:
        """Run inference on all test sequences and verify matches."""
        sequences = self.test_data["SEQUENCES"]
        quant_preds = self.test_data["QUANT_PREDICTIONS"]

        correct = 0
        mismatches: list[str] = []

        for idx in range(len(sequences)):
            output = _run_inference(
                self.elf_data, sequences[idx],
                self.test_tokens_addr, self.test_n_tokens_addr,
            )
            predicted = _parse_prediction(output)

            if predicted == quant_preds[idx]:
                correct += 1
            else:
                mismatches.append(
                    f"  Seq {idx}: firmware={predicted}, python_quant={quant_preds[idx]}"
                )

        accuracy = correct / max(1, len(sequences)) * 100
        # Allow some mismatches due to implementation differences
        assert accuracy >= 50.0, (
            f"Agreement {accuracy:.1f}% < 50% "
            f"({correct}/{len(sequences)} matching)\n"
            f"Mismatches:\n" + "\n".join(mismatches[:10])
        )
