"""MNIST inference integration tests.

Runs compiled MNIST firmware on the emulator with test images and verifies
that the quantized inference matches the Python reference predictions.
"""

from __future__ import annotations

import io
import pathlib
import shutil

import pytest

from riscv_npu.cpu.cpu import CPU
from riscv_npu.loader.elf import find_symbol, parse_elf
from riscv_npu.memory.bus import MemoryBus
from riscv_npu.memory.ram import RAM
from riscv_npu.syscall.handler import SyscallHandler

BASE = 0x80000000
RAM_SIZE = 4 * 1024 * 1024  # 4 MB (weights are ~100KB, .bss extends further)
STACK_TOP = BASE + RAM_SIZE - 16

FIRMWARE_DIR = pathlib.Path(__file__).parent.parent.parent / "firmware"
ELF_PATH = FIRMWARE_DIR / "mnist" / "mnist.elf"
TEST_DATA_PATH = FIRMWARE_DIR / "mnist" / "test_data.py"

_HAS_TOOLCHAIN = shutil.which("riscv64-unknown-elf-gcc") is not None
_HAS_ELF = ELF_PATH.exists()
_HAS_TEST_DATA = TEST_DATA_PATH.exists()


def _load_test_data() -> dict:
    """Load the auto-generated test data module.

    Returns:
        Dict with IMAGES, LABELS, PREDICTIONS, QUANT_PREDICTIONS, SHIFT1.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("test_data", str(TEST_DATA_PATH))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return {
        "IMAGES": mod.IMAGES,
        "LABELS": mod.LABELS,
        "PREDICTIONS": mod.PREDICTIONS,
        "QUANT_PREDICTIONS": mod.QUANT_PREDICTIONS,
        "SHIFT1": mod.SHIFT1,
    }


def _run_inference(elf_data: bytes, image_bytes: bytes, test_image_addr: int) -> str:
    """Run MNIST inference on a single image.

    Loads the ELF into RAM, writes image data to the test_image buffer,
    runs the CPU, and returns the stdout output.

    Args:
        elf_data: Raw ELF file bytes.
        image_bytes: 784 bytes of uint8 pixel data.
        test_image_addr: Address of the test_image symbol in the ELF.

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

    # Load ELF segments into RAM
    prog = parse_elf(elf_data)
    for seg in prog.segments:
        padded = seg.data + b"\x00" * (seg.memsz - len(seg.data))
        ram.load_segment(seg.vaddr, padded)

    # Write image data to the test_image buffer
    ram.load_segment(test_image_addr, image_bytes)

    # Set up CPU
    cpu.pc = prog.entry
    cpu.registers.write(2, STACK_TOP)

    # Run (generous limit -- inference takes ~600K cycles)
    cpu.run(max_cycles=2_000_000)

    return stdout_buf.getvalue().decode("utf-8", errors="replace")


def _parse_digit(output: str) -> int:
    """Parse the predicted digit from firmware output.

    Args:
        output: Stdout text from the firmware.

    Returns:
        Predicted digit 0-9.

    Raises:
        ValueError: If output cannot be parsed as a digit.
    """
    stripped = output.strip()
    if not stripped:
        raise ValueError("Empty output from firmware")
    return int(stripped)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TOOLCHAIN, reason="riscv64 toolchain not installed")
@pytest.mark.skipif(not _HAS_ELF, reason="mnist.elf not built (run: cd firmware/mnist && make)")
@pytest.mark.skipif(not _HAS_TEST_DATA, reason="test_data.py not generated (run: uv run --extra torch python -m riscv_npu.tools.export_mnist_weights)")
class TestMnistInference:
    """Integration tests for MNIST firmware inference."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Load ELF data and test data once per class."""
        self.elf_data = ELF_PATH.read_bytes()
        self.test_data = _load_test_data()
        self.test_image_addr = find_symbol(self.elf_data, "test_image")
        assert self.test_image_addr is not None, "test_image symbol not found in ELF"

    def test_mnist_single_image(self) -> None:
        """Run inference on the first test image and verify correct digit."""
        image = self.test_data["IMAGES"][0]
        expected = self.test_data["QUANT_PREDICTIONS"][0]

        image_bytes = bytes(image)
        output = _run_inference(self.elf_data, image_bytes, self.test_image_addr)
        predicted = _parse_digit(output)

        assert predicted == expected, (
            f"Single image prediction mismatch: got {predicted}, "
            f"expected {expected} (label={self.test_data['LABELS'][0]})"
        )

    def test_mnist_accuracy(self) -> None:
        """Run inference on 100 test images and verify >=95% accuracy.

        Compares firmware output to the Python quantized reference predictions.
        """
        images = self.test_data["IMAGES"]
        quant_preds = self.test_data["QUANT_PREDICTIONS"]
        labels = self.test_data["LABELS"]

        correct = 0
        mismatches: list[str] = []

        for idx in range(len(images)):
            image_bytes = bytes(images[idx])
            output = _run_inference(self.elf_data, image_bytes, self.test_image_addr)
            predicted = _parse_digit(output)

            if predicted == quant_preds[idx]:
                correct += 1
            else:
                mismatches.append(
                    f"  Image {idx}: firmware={predicted}, "
                    f"python_quant={quant_preds[idx]}, label={labels[idx]}"
                )

        accuracy = correct / len(images) * 100
        assert accuracy >= 95.0, (
            f"Accuracy {accuracy:.1f}% < 95% "
            f"({correct}/{len(images)} correct)\n"
            f"Mismatches:\n" + "\n".join(mismatches[:10])
        )

    def test_mnist_all_digits_represented(self) -> None:
        """Verify the test set covers all 10 digits."""
        labels = self.test_data["LABELS"]
        unique_digits = set(labels)
        assert unique_digits == set(range(10)), (
            f"Test data missing digits: {set(range(10)) - unique_digits}"
        )
