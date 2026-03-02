"""Transformer inference integration tests.

Runs compiled float32 transformer firmware on the emulator with test input
sequences and verifies that the firmware predictions match the Python
float reference predictions.
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
        Dict with SEQUENCES, FLOAT_PREDICTIONS, CONTEXT_LEN.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("test_data", str(TEST_DATA_PATH))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return {
        "SEQUENCES": mod.SEQUENCES,
        "FLOAT_PREDICTIONS": mod.FLOAT_PREDICTIONS,
        "CONTEXT_LEN": mod.CONTEXT_LEN,
    }


def _run_inference(
    elf_data: bytes,
    tokens: list[int],
    test_tokens_addr: int,
    test_n_tokens_addr: int,
    test_n_generate_addr: int,
    n_generate: int = 1,
) -> bytes:
    """Run transformer inference on a token sequence.

    Loads the ELF into RAM, writes token data and generation count,
    runs the CPU, and returns the raw stdout output bytes.

    Args:
        elf_data: Raw ELF file bytes.
        tokens: List of token IDs (byte values 0-255).
        test_tokens_addr: Address of the test_tokens symbol.
        test_n_tokens_addr: Address of the test_n_tokens symbol.
        test_n_generate_addr: Address of the test_n_generate symbol.
        n_generate: Number of tokens to generate.

    Returns:
        Captured stdout output as raw bytes.
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

    # Write prompt length and generation count as int32 little-endian
    ram.load_segment(test_n_tokens_addr, struct.pack("<i", len(tokens)))
    ram.load_segment(test_n_generate_addr, struct.pack("<i", n_generate))

    # Set up CPU
    cpu.pc = prog.entry
    cpu.registers.write(2, STACK_TOP)

    # Run (generous limit -- transformer inference is slow)
    cpu.run(max_cycles=50_000_000)

    # Output format: "<prompt>\n> <generated>\n"
    # Return only the generated tokens (after "> " on second line).
    raw = stdout_buf.getvalue()
    marker = b"\n> "
    idx = raw.find(marker)
    if idx == -1:
        return b""
    generated = raw[idx + len(marker) :]
    # Strip trailing newline
    if generated.endswith(b"\n"):
        generated = generated[:-1]
    return generated


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TOOLCHAIN, reason="riscv64 toolchain not installed")
@pytest.mark.skipif(not _HAS_ELF, reason="transformer.elf not built")
@pytest.mark.skipif(not _HAS_TEST_DATA, reason="test_data.py not generated")
class TestTransformerInference:
    """Integration tests for float32 transformer firmware inference."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Load ELF data and test data once per class."""
        self.elf_data = ELF_PATH.read_bytes()
        self.test_data = _load_test_data()
        self.test_tokens_addr = find_symbol(self.elf_data, "test_tokens")
        self.test_n_tokens_addr = find_symbol(self.elf_data, "test_n_tokens")
        self.test_n_generate_addr = find_symbol(self.elf_data, "test_n_generate")
        assert self.test_tokens_addr is not None, "test_tokens symbol not found"
        assert self.test_n_tokens_addr is not None, "test_n_tokens symbol not found"
        assert self.test_n_generate_addr is not None, "test_n_generate symbol not found"

    def test_single_sequence(self) -> None:
        """Run inference on the first test sequence and compare to float reference."""
        seq = self.test_data["SEQUENCES"][0]
        expected = self.test_data["FLOAT_PREDICTIONS"][0]

        output = _run_inference(
            self.elf_data, seq,
            self.test_tokens_addr, self.test_n_tokens_addr,
            self.test_n_generate_addr, n_generate=1,
        )
        assert len(output) >= 1, "No output from firmware"
        predicted = output[0]

        assert predicted == expected, (
            f"Single sequence prediction mismatch: got {predicted} ({chr(predicted)!r}), "
            f"expected {expected} ({chr(expected)!r})"
        )

    def test_multiple_sequences(self) -> None:
        """Run inference on all test sequences and verify majority match float reference."""
        sequences = self.test_data["SEQUENCES"]
        float_preds = self.test_data["FLOAT_PREDICTIONS"]

        correct = 0
        mismatches: list[str] = []

        for idx in range(len(sequences)):
            output = _run_inference(
                self.elf_data, sequences[idx],
                self.test_tokens_addr, self.test_n_tokens_addr,
                self.test_n_generate_addr, n_generate=1,
            )
            assert len(output) >= 1, f"No output for sequence {idx}"
            predicted = output[0]

            if predicted == float_preds[idx]:
                correct += 1
            else:
                mismatches.append(
                    f"  Seq {idx}: firmware={predicted} ({chr(predicted)!r}), "
                    f"python_float={float_preds[idx]} ({chr(float_preds[idx])!r})"
                )

        accuracy = correct / max(1, len(sequences)) * 100
        # Expect high agreement since both use float32
        assert accuracy >= 50.0, (
            f"Agreement {accuracy:.1f}% < 50% "
            f"({correct}/{len(sequences)} matching)\n"
            f"Mismatches:\n" + "\n".join(mismatches[:10])
        )
