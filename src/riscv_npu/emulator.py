"""High-level library API for the riscv-npu emulator."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from .cpu.cpu import CPU
from .cpu.fpu import FpuState
from .devices.uart import UART, UART_BASE, UART_SIZE
from .loader.elf import find_symbol, parse_elf
from .memory.bus import MemoryBus
from .memory.ram import RAM
from .npu.engine import NpuState
from .syscall.handler import SyscallHandler

BASE = 0x80000000
DEFAULT_RAM_SIZE = 4 * 1024 * 1024  # 4 MB


@dataclass
class RunResult:
    """Result of an emulator execution."""

    exit_code: int
    cycles: int
    stats: dict[str, int] = field(default_factory=dict)


class Emulator:
    """High-level interface to the riscv-npu emulator."""

    def __init__(self, ram_size: int = DEFAULT_RAM_SIZE) -> None:
        """Create emulator with RAM at 0x80000000, UART, and NPU device.

        Args:
            ram_size: Size of RAM in bytes (default 4 MB).
        """
        self._ram_size = ram_size
        self._stack_top = BASE + ram_size - 16  # 16-byte aligned

        self._bus = MemoryBus()
        self._ram = RAM(BASE, ram_size)
        self._uart_stream = io.BytesIO()
        self._uart = UART(tx_stream=self._uart_stream)
        self._bus.register(BASE, ram_size, self._ram)
        self._bus.register(UART_BASE, UART_SIZE, self._uart)

        self._cpu = CPU(self._bus)
        self._handler = SyscallHandler(stdout=self._uart_stream)
        self._cpu.syscall_handler = self._handler

        self._elf_data: bytes | None = None
        self._entry: int = 0
        self._initial_brk: int = 0

    def load_elf(self, path: str | Path) -> None:
        """Load ELF binary. Sets PC to entry point, SP to stack top.

        Retains raw ELF data for symbol lookups.

        Args:
            path: Path to ELF binary file.
        """
        self._elf_data = Path(path).read_bytes()
        prog = parse_elf(self._elf_data)

        # Load segments into RAM
        for seg in prog.segments:
            padded = seg.data + b"\x00" * (seg.memsz - len(seg.data))
            self._ram.load_segment(seg.vaddr, padded)

        self._entry = prog.entry
        self._cpu.pc = prog.entry
        self._cpu.registers.write(2, self._stack_top)

        # Set initial program break (16-byte aligned, past last segment)
        if prog.segments:
            end = max(s.vaddr + s.memsz for s in prog.segments)
            brk = (end + 15) & ~15
            self._handler.brk = brk
            self._initial_brk = brk

    def symbol(self, name: str) -> int:
        """Look up a symbol address in the loaded ELF.

        Args:
            name: Symbol name to look up.

        Returns:
            The symbol's virtual address.

        Raises:
            KeyError: If the symbol is not found or no ELF is loaded.
        """
        if self._elf_data is None:
            raise KeyError(name)
        addr = find_symbol(self._elf_data, name)
        if addr is None:
            raise KeyError(name)
        return addr

    def _resolve_addr(self, symbol_or_addr: str | int) -> int:
        """Resolve a symbol name or raw address to an integer address."""
        if isinstance(symbol_or_addr, str):
            return self.symbol(symbol_or_addr)
        return symbol_or_addr

    def write_f32(self, symbol_or_addr: str | int, data: "np.ndarray") -> None:
        """Write a float32 array to memory at a symbol address or raw address.

        Args:
            symbol_or_addr: Symbol name or integer address.
            data: 1D float32 ndarray.
        """
        import numpy as np

        addr = self._resolve_addr(symbol_or_addr)
        arr = np.asarray(data, dtype=np.float32)
        self._ram.load_segment(addr, arr.tobytes())

    def read_f32(self, symbol_or_addr: str | int, n: int) -> "np.ndarray":
        """Read n float32 values from memory.

        Args:
            symbol_or_addr: Symbol name or integer address.
            n: Number of float32 values to read.

        Returns:
            1D float32 ndarray.
        """
        import numpy as np

        addr = self._resolve_addr(symbol_or_addr)
        nbytes = n * 4
        offset = self._ram._offset(addr, nbytes)
        raw = bytes(self._ram._data[offset : offset + nbytes])
        return np.frombuffer(raw, dtype=np.float32).copy()

    def write_i32(self, symbol_or_addr: str | int, data: "np.ndarray") -> None:
        """Write an int32 array to memory.

        Args:
            symbol_or_addr: Symbol name or integer address.
            data: 1D int32 ndarray.
        """
        import numpy as np

        addr = self._resolve_addr(symbol_or_addr)
        arr = np.asarray(data, dtype=np.int32)
        self._ram.load_segment(addr, arr.tobytes())

    def read_i32(self, symbol_or_addr: str | int, n: int) -> "np.ndarray":
        """Read n int32 values from memory.

        Args:
            symbol_or_addr: Symbol name or integer address.
            n: Number of int32 values to read.

        Returns:
            1D int32 ndarray.
        """
        import numpy as np

        addr = self._resolve_addr(symbol_or_addr)
        nbytes = n * 4
        offset = self._ram._offset(addr, nbytes)
        raw = bytes(self._ram._data[offset : offset + nbytes])
        return np.frombuffer(raw, dtype=np.int32).copy()

    def write_bytes(self, symbol_or_addr: str | int, data: bytes) -> None:
        """Write raw bytes to memory.

        Args:
            symbol_or_addr: Symbol name or integer address.
            data: Bytes to write.
        """
        addr = self._resolve_addr(symbol_or_addr)
        self._ram.load_segment(addr, data)

    def read_bytes(self, symbol_or_addr: str | int, n: int) -> bytes:
        """Read n raw bytes from memory.

        Args:
            symbol_or_addr: Symbol name or integer address.
            n: Number of bytes to read.

        Returns:
            Raw bytes.
        """
        addr = self._resolve_addr(symbol_or_addr)
        offset = self._ram._offset(addr, n)
        return bytes(self._ram._data[offset : offset + n])

    def run(self, max_cycles: int = 10_000_000) -> RunResult:
        """Execute until halt (ecall 93) or cycle limit.

        Args:
            max_cycles: Maximum number of cycles before timeout.

        Returns:
            RunResult with exit code, cycle count, and instruction stats.

        Raises:
            TimeoutError: If cycle limit reached without halting.
        """
        self._cpu.run(max_cycles)
        if not self._cpu.halted:
            raise TimeoutError(
                f"Execution did not halt within {max_cycles} cycles"
            )
        return RunResult(
            exit_code=self._cpu.exit_code,
            cycles=self._cpu.cycle_count,
            stats=dict(self._cpu.instruction_stats),
        )

    def reset(self) -> None:
        """Reset CPU state (PC, registers, FPU, NPU, halted flag, cycle count)
        but keep RAM contents and loaded ELF. Allows re-running with different
        inputs."""
        self._cpu.pc = self._entry
        for i in range(1, 32):
            self._cpu.registers.write(i, 0)
        self._cpu.registers.write(2, self._stack_top)
        self._cpu.halted = False
        self._cpu.cycle_count = 0
        self._cpu.instruction_stats.clear()
        self._cpu.fpu_state = FpuState()
        self._cpu.npu_state = NpuState()
        self._handler.brk = self._initial_brk
        self._uart_stream.seek(0)
        self._uart_stream.truncate(0)

    @property
    def stdout(self) -> bytes:
        """UART output captured during execution."""
        return self._uart_stream.getvalue()
