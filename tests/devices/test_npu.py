"""Tests for NPU memory-mapped device at 0x20000000."""

from riscv_npu.devices.npu import NpuDevice, NPU_BASE
from riscv_npu.npu.engine import NpuState, acc_set64


class TestNpuDevice:
    """Tests for NpuDevice read/write interface."""

    def test_read_acc_lo_zero(self) -> None:
        """Reading acc_lo bytes from a fresh device returns 0."""
        state = NpuState()
        dev = NpuDevice(state)
        for i in range(4):
            assert dev.read8(NPU_BASE + i) == 0

    def test_read_acc_lo_set(self) -> None:
        """Reading acc_lo bytes after setting the accumulator."""
        state = NpuState()
        state.acc_lo = 0x12345678
        dev = NpuDevice(state)
        assert dev.read8(NPU_BASE + 0) == 0x78  # little-endian byte 0
        assert dev.read8(NPU_BASE + 1) == 0x56
        assert dev.read8(NPU_BASE + 2) == 0x34
        assert dev.read8(NPU_BASE + 3) == 0x12

    def test_read_acc_hi(self) -> None:
        """Reading acc_hi bytes."""
        state = NpuState()
        state.acc_hi = 0xDEADBEEF
        dev = NpuDevice(state)
        assert dev.read8(NPU_BASE + 4) == 0xEF
        assert dev.read8(NPU_BASE + 5) == 0xBE
        assert dev.read8(NPU_BASE + 6) == 0xAD
        assert dev.read8(NPU_BASE + 7) == 0xDE

    def test_read_vreg(self) -> None:
        """Reading vector register bytes."""
        state = NpuState()
        state.vreg[0] = [10, -20, 127, -128]
        dev = NpuDevice(state)
        assert dev.read8(NPU_BASE + 8) == 10
        assert dev.read8(NPU_BASE + 9) == (-20) & 0xFF  # 0xEC
        assert dev.read8(NPU_BASE + 10) == 127
        assert dev.read8(NPU_BASE + 11) == (-128) & 0xFF  # 0x80

    def test_read_vreg_all(self) -> None:
        """Reading all 4 vector registers."""
        state = NpuState()
        for i in range(4):
            state.vreg[i] = [i * 4 + j for j in range(4)]
        dev = NpuDevice(state)
        for i in range(4):
            for j in range(4):
                assert dev.read8(NPU_BASE + 8 + i * 4 + j) == i * 4 + j

    def test_read_out_of_range(self) -> None:
        """Reading beyond defined registers returns 0."""
        state = NpuState()
        dev = NpuDevice(state)
        assert dev.read8(NPU_BASE + 24) == 0
        assert dev.read8(NPU_BASE + 100) == 0

    def test_write_offset_0_resets_acc(self) -> None:
        """Writing to offset 0 resets the accumulator."""
        state = NpuState()
        acc_set64(state, 0x123456789ABCDEF0)
        dev = NpuDevice(state)
        dev.write8(NPU_BASE, 0)
        assert state.acc_lo == 0
        assert state.acc_hi == 0

    def test_write_other_offsets_ignored(self) -> None:
        """Writing to non-zero offsets is a no-op."""
        state = NpuState()
        state.acc_lo = 0x42
        dev = NpuDevice(state)
        dev.write8(NPU_BASE + 1, 0xFF)
        assert state.acc_lo == 0x42  # Unchanged
