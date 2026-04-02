# Phase 12: Library API

## Goal

Expose a public Python API so external projects can use riscv-npu as an importable library, not just a CLI tool. A single `Emulator` class wraps the existing CPU/memory/device wiring and provides convenient methods for loading programs, accessing memory, and inspecting execution results.

## Background

All the building blocks already exist internally — `CPU`, `MemoryBus`, `RAM`, `UART`, `NpuDevice`, `SyscallHandler`, `load_elf`, `find_symbol`. The CLI (`python -m riscv_npu run`) wires them together in `cli.py:run_binary()`, and the integration tests do the same in `tests/integration/test_programs.py:_run_elf()`. But there is no public API — consumers must either shell out to the CLI or replicate the 20-line wiring boilerplate with internal imports.

This phase wraps that boilerplate in a clean `Emulator` class and adds typed array I/O helpers for numeric data.

## Usage Examples

### Basic: load and run an ELF

```python
from riscv_npu import Emulator

emu = Emulator()
emu.load_elf("firmware/hello/hello.elf")
result = emu.run()

print(result.exit_code)    # 0
print(result.cycles)       # 1423
print(emu.stdout)          # b'Hello, World!\n'
```

### Write data to memory, run, read results back

```python
import numpy as np
from riscv_npu import Emulator

emu = Emulator()
emu.load_elf("firmware/npu_add/npu_add.elf")

# Write input arrays at symbol addresses
A = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
B = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
emu.write_f32("A", A)
emu.write_f32("B", B)

result = emu.run()

# Read output array
out = emu.read_f32("out", n=4)
print(out)  # [6.0, 8.0, 10.0, 12.0]
```

### Address-based memory access (no symbols)

```python
emu.write_f32(0x80010000, data)
result_data = emu.read_f32(0x80010000, n=64)
```

### Inspect instruction statistics

```python
result = emu.run()
print(result.stats)
# {'ADD': 142, 'LW': 89, 'SW': 45, 'NPU.FVADD': 1, 'ECALL': 1, ...}
```

### Re-run with different inputs

```python
emu.load_elf("kernel.elf")
emu.write_f32("input", data_1)
result_1 = emu.run()
output_1 = emu.read_f32("output", n=64)

emu.reset()  # reset CPU state, keep RAM + ELF
emu.write_f32("input", data_2)
result_2 = emu.run()
output_2 = emu.read_f32("output", n=64)
```

### Install as a dependency

```bash
# Editable install for development alongside another project
uv add --editable ../riscv-npu

# Or via path dependency in pyproject.toml
# riscv-npu = { path = "../riscv-npu", editable = true }
```

## What to build

### 1. `Emulator` class

**File**: `src/riscv_npu/emulator.py`

```python
@dataclass
class RunResult:
    """Result of an emulator execution."""
    exit_code: int
    cycles: int
    stats: dict[str, int]  # instruction mnemonic -> execution count


class Emulator:
    """High-level interface to the riscv-npu emulator."""

    def __init__(self, ram_size: int = 4 * 1024 * 1024) -> None:
        """Create emulator with RAM at 0x80000000, UART, and NPU device."""

    def load_elf(self, path: str) -> None:
        """Load ELF binary. Sets PC to entry point, SP to stack top.
        Retains raw ELF data for symbol lookups."""

    def symbol(self, name: str) -> int:
        """Look up a symbol address in the loaded ELF.
        Raises KeyError if symbol not found."""

    def write_f32(self, symbol_or_addr: str | int, data: np.ndarray) -> None:
        """Write a float32 array to memory at a symbol address or raw address.
        data must be a 1D float32 ndarray."""

    def read_f32(self, symbol_or_addr: str | int, n: int) -> np.ndarray:
        """Read n float32 values from memory. Returns 1D float32 ndarray."""

    def write_i32(self, symbol_or_addr: str | int, data: np.ndarray) -> None:
        """Write an int32 array to memory."""

    def read_i32(self, symbol_or_addr: str | int, n: int) -> np.ndarray:
        """Read n int32 values from memory."""

    def write_bytes(self, symbol_or_addr: str | int, data: bytes) -> None:
        """Write raw bytes to memory."""

    def read_bytes(self, symbol_or_addr: str | int, n: int) -> bytes:
        """Read n raw bytes from memory."""

    def run(self, max_cycles: int = 10_000_000) -> RunResult:
        """Execute until halt (ecall 93) or cycle limit.
        Raises TimeoutError if cycle limit reached without halting."""

    def reset(self) -> None:
        """Reset CPU state (PC, registers, halted flag, cycle count) but keep
        RAM contents and loaded ELF. Allows re-running with different inputs."""

    @property
    def stdout(self) -> bytes:
        """UART output captured during execution."""
```

### Implementation notes

- `__init__` wires up `MemoryBus`, `RAM`, `UART(tx_stream=BytesIO())`, `NpuDevice`, `CPU`, `SyscallHandler` — same setup as `cli.py:run_binary()` but retaining references to all components.
- `load_elf` calls `loader.elf.load_elf()` and stores the raw ELF bytes (`Path(path).read_bytes()`) for `find_symbol()` calls.
- `symbol()` calls `loader.elf.find_symbol(self._elf_data, name)`, raises `KeyError` on `None`.
- `write_f32` resolves address (str → `symbol()`, int → use directly), then calls `ram.load_segment(addr, data.tobytes())`.
- `read_f32` resolves address, reads `n * 4` bytes from RAM's backing buffer, returns `np.frombuffer(..., dtype=np.float32).copy()`.
- `run()` calls `cpu.run(max_cycles)`, checks `cpu.halted`, returns `RunResult(cpu.exit_code, cpu.cycle_count, cpu.instruction_stats)`. Raises `TimeoutError` if not halted.
- `reset()` sets `cpu.pc` back to the ELF entry point, `cpu.registers.write(2, stack_top)`, clears `cpu.halted`, resets `cpu.cycle_count`, clears UART buffer.
- `stdout` returns `self._uart_stream.getvalue()`.
- numpy is used for `write_f32`/`read_f32` but is not a hard dependency — only imported when those methods are called. The `write_bytes`/`read_bytes` methods work without numpy.

### 2. Package exports

**File**: `src/riscv_npu/__init__.py`

```python
from riscv_npu.emulator import Emulator, RunResult

__all__ = ["Emulator", "RunResult"]
```

### 3. Tests

**File**: `tests/test_emulator.py`

- `test_load_and_run_hello` — load hello.elf, run, check stdout and exit code
- `test_run_result_stats` — check cycles > 0, stats dict has entries
- `test_symbol_lookup` — load ELF with known symbols, verify addresses
- `test_symbol_not_found` — raises KeyError
- `test_write_read_f32` — write array, read back, compare
- `test_write_read_i32` — same for int32
- `test_write_read_bytes` — same for raw bytes
- `test_address_based_access` — write/read by integer address, not symbol
- `test_reset` — load, run, reset, run again with different inputs
- `test_timeout` — run with max_cycles=1, verify TimeoutError
- `test_stdout_capture` — run hello, check emu.stdout matches expected output

### 4. Add numpy as optional dependency

In `pyproject.toml`:
```toml
[project.optional-dependencies]
numpy = ["numpy>=1.26.0"]
```

This keeps the core emulator lightweight (no numpy required for CLI usage) while allowing library consumers to use the array I/O methods.

## Scope

This phase is intentionally small:
- One new file (`emulator.py`) wrapping existing internals
- Updated `__init__.py` with exports
- Tests
- No changes to existing code — the `Emulator` class is a pure addition

The CLI continues to work as before. The `Emulator` class is an alternative interface for programmatic usage.
