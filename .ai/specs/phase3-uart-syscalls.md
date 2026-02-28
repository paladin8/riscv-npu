# Phase 3: UART + Syscalls

## Goal
Programs print to stdout, read from stdin, and exit with a code. Interactive terminal programs (ANSI escape codes, keyboard input) work.

## What to build
- MemoryBus: routes address ranges to devices. Interface: read8/16/32, write8/16/32. Devices register (base_addr, size, device).
- RAM device: wraps existing memory, mapped at 0x80000000
- UART device: mapped at 0x10000000.
  - TX (write): write byte to address -> appears on host stdout
  - RX (read): read byte from address -> returns next byte from host stdin (non-blocking, returns 0 if no input available)
  - LSR (Line Status Register) at offset 0x05: bit 0 = data ready (input available), bit 5 = THR empty (can write). This is standard 16550 UART behavior.
- SyscallHandler: on ECALL, read a7 for number:
  - 63 (read): a0=fd, a1=buf_ptr, a2=len -> read from stdin (fd=0 only), returns bytes read
  - 64 (write): a0=fd, a1=buf_ptr, a2=len -> write to stdout (fd=1 only)
  - 93 (exit): a0=code -> halt, report code
  - 214 (brk): a0=addr -> bump allocator
- firmware/common/syscalls.c: putchar, getchar, puts, exit wrappers using ecall
- firmware/hello/main.c: "Hello, World!" via write syscall

## UART implementation notes
Host stdin must be read non-blocking so the emulator doesn't stall waiting for input. Use select/poll or platform-appropriate non-blocking IO. Buffer incoming bytes -- the UART RX holds one byte at a time, the emulated program polls LSR then reads RX. ANSI escape codes from the emulated program pass through the UART TX to the host terminal unmodified -- the host terminal interprets them. This means emulated programs can do cursor movement, colors, screen clearing, etc. with raw escape codes.

---

## Design Decisions

### DD-1: MemoryBus as duck-type replacement for RAM
The CPU currently takes `memory: RAM` directly. We will make MemoryBus present the same read8/16/32, write8/16/32, and load_segment interface as RAM, so it can be used as a drop-in replacement. The CPU type hint changes from `RAM` to `MemoryBus`, but no behavioral changes are needed in CPU code beyond the type annotation. The `load_segment` method on MemoryBus delegates to the RAM device by calculating which device owns the target address range.

### DD-2: Device as a Protocol (structural typing)
Rather than using an ABC, define `Device` as a `typing.Protocol` with read8/write8 methods. This lets RAM satisfy the protocol without subclassing. Devices only need to implement byte-level read/write -- the bus handles multi-byte reads/writes by composing byte operations (little-endian). This simplifies device implementations (they only deal with bytes) while the bus handles width logic.

Actually, on reflection: RAM already has efficient read16/read32 via int.from_bytes. If the bus decomposes all accesses to byte-level, RAM loses that efficiency. But correctness is more important than performance for an emulator, and the simplicity of single-width device interfaces is worth it. Devices implement read8/write8 only. The bus composes multi-byte accesses.

### DD-3: UART uses injectable TX stream + push_rx for RX
The UART constructor takes a `tx_stream` parameter (defaulting to sys.stdout.buffer) and a `base` address parameter (defaulting to UART_BASE). For TX testing, inject `io.BytesIO`. For RX, the UART maintains an internal `deque[int]` buffer with a `push_rx(data: bytes)` method. External code (CLI stdin reader or test code) pushes bytes into this buffer. The UART itself never reads from stdin -- that responsibility lives in the CLI layer. This makes the UART a pure register-interface + buffer, keeping tests deterministic without mocking.

### DD-4: SyscallHandler intercepts ECALL before trap
Currently, ECALL either traps to mtvec or halts. With the SyscallHandler, we add a new path: if a SyscallHandler is installed on the CPU, ECALL first checks a7 for known syscall numbers. If recognized, the handler processes it and returns pc+4 (no trap). If unrecognized, it falls through to the existing trap/halt behavior. This means the CPU needs an optional `syscall_handler` attribute.

### DD-5: Separate exit_code from tohost
The SyscallHandler's exit syscall sets `cpu.halted = True` and stores the exit code in a new `cpu.exit_code: int` attribute (default 0). This is distinct from the tohost mechanism used by riscv-tests.

### DD-6: brk syscall uses simple bump allocator
The brk syscall manages a program break pointer. Initial break is set after all ELF segments are loaded (end of .bss, aligned to 16 bytes). brk(0) returns current break. brk(addr) with addr > current break extends it. The SyscallHandler tracks the break pointer internally.

---

## Deliverables List

### D1: Device Protocol + MemoryBus
Create the Device protocol and MemoryBus class that routes address ranges to registered devices.

### D2: RAM as a Bus Device + CPU Integration
Make RAM satisfy the Device protocol. Update CPU to accept MemoryBus. Update conftest fixtures.

### D3: UART Device
Implement the 16550-style UART with TX, RX, and LSR registers.

### D4: SyscallHandler
Implement ECALL dispatch for write, exit, read, and brk syscalls.

### D5: CPU + CLI Wiring
Wire SyscallHandler into CPU ECALL flow. Update CLI to build MemoryBus with RAM + UART.

### D6: Hello World Firmware
Create syscalls.c wrappers and firmware/hello/main.c that prints "Hello, World!".

---

## Implementation Details

### D1: Device Protocol + MemoryBus

**Files to create/modify:**
- `src/riscv_npu/memory/device.py` -- Define Device Protocol
- `src/riscv_npu/memory/bus.py` -- Implement MemoryBus
- `tests/memory/test_bus.py` -- Bus tests

**Device Protocol (`device.py`):**
```python
from typing import Protocol

class Device(Protocol):
    def read8(self, addr: int) -> int: ...
    def write8(self, addr: int, value: int) -> None: ...
```

Note: `addr` is an absolute address. The bus passes absolute addresses to devices; the device is responsible for translating to its internal offset. This keeps the bus simple (no offset math) and matches how RAM already works.

**MemoryBus (`bus.py`):**
```python
from dataclasses import dataclass, field
from .device import Device

@dataclass
class DeviceMapping:
    base: int
    size: int
    device: Device

class MemoryBus:
    def __init__(self) -> None:
        self._devices: list[DeviceMapping] = []

    def register(self, base: int, size: int, device: Device) -> None: ...
    def _find_device(self, addr: int, width: int) -> DeviceMapping: ...
    def read8(self, addr: int) -> int: ...
    def read16(self, addr: int) -> int: ...
    def read32(self, addr: int) -> int: ...
    def write8(self, addr: int, value: int) -> None: ...
    def write16(self, addr: int, value: int) -> None: ...
    def write32(self, addr: int, value: int) -> None: ...
    def load_segment(self, addr: int, data: bytes) -> None: ...
```

- `_find_device`: linear scan of device list; raises MemoryError if no device covers `[addr, addr+width)`.
- `read16/read32`: compose from device.read8 calls, little-endian (byte 0 = LSB).
- `write16/write32`: decompose into device.write8 calls, little-endian.
- `load_segment`: write each byte via write8.

### D2: RAM as a Bus Device + CPU Integration

**Files to modify:**
- `src/riscv_npu/cpu/cpu.py` -- Change memory type to MemoryBus (or keep duck-typed)
- `src/riscv_npu/cpu/execute.py` -- Update RAM type hints to be generic
- `tests/cpu/conftest.py` -- Update fixtures to use MemoryBus wrapping RAM

**Key changes:**
- RAM already has `read8(addr)` and `write8(addr, value)` with absolute addresses -- it already satisfies the Device protocol with no changes needed.
- CPU.__init__ type hint changes from `RAM` to `MemoryBus`. Update the import accordingly.
- execute.py: change `mem: RAM` type hints in `_exec_load` and `_exec_store` to `mem: MemoryBus`. Update the import from `..memory.ram import RAM` to `..memory.bus import MemoryBus`. Since MemoryBus provides the same read/write interface, this is a pure type-hint change with no behavioral impact.
- conftest.py: build MemoryBus, register RAM on it, pass bus to CPU. This ensures all existing CPU tests exercise the bus routing path.
- All existing 268 tests must pass unchanged.

### D3: UART Device

**Files to create/modify:**
- `src/riscv_npu/devices/uart.py` -- UART implementation
- `tests/devices/__init__.py` -- Create test package
- `tests/devices/test_uart.py` -- UART tests

**UART class:**
```python
from collections import deque
from typing import BinaryIO
import sys

UART_BASE = 0x10000000
UART_SIZE = 8  # 8 registers (standard 16550 has 8 byte-wide registers)

# Register offsets
_RBR = 0  # Receiver Buffer Register (read)
_THR = 0  # Transmitter Holding Register (write)
_LSR = 5  # Line Status Register (read)

# LSR bits
_LSR_DATA_READY = 0x01   # bit 0: data available in RBR
_LSR_THR_EMPTY  = 0x20   # bit 5: THR is empty (ready to write)

class UART:
    def __init__(
        self,
        base: int = UART_BASE,
        tx_stream: BinaryIO | None = None,
    ) -> None: ...

    def read8(self, addr: int) -> int: ...
    def write8(self, addr: int, value: int) -> None: ...
    def push_rx(self, data: bytes) -> None: ...
```

- Constructor: stores `self._base = base`, `self._tx = tx_stream or sys.stdout.buffer`, `self._rx_buf: deque[int] = deque()`.
- `read8`:
  - offset = addr - self._base
  - offset 0 (RBR): popleft from _rx_buf deque, return 0 if empty
  - offset 5 (LSR): return status bits (bit 5 always set = THR empty, bit 0 set if _rx_buf non-empty)
  - other offsets: return 0
- `write8`:
  - offset = addr - self._base
  - offset 0 (THR): write byte to tx_stream, flush
  - other offsets: no-op
- `push_rx(data)`: extend _rx_buf with each byte from data. Called by CLI stdin reader or test code.

The UART does NOT read from stdin itself. External code (CLI layer or tests) pushes bytes via `push_rx()`. This keeps the UART a pure register interface + buffer, making tests deterministic without mocking.

### D4: SyscallHandler

**Files to create/modify:**
- `src/riscv_npu/syscall/handler.py` -- SyscallHandler
- `tests/syscall/__init__.py` -- Create test package
- `tests/syscall/test_handler.py` -- Handler tests

**SyscallHandler class:**
```python
from __future__ import annotations
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from ..cpu.cpu import CPU

# Linux syscall numbers (RISC-V ABI)
SYS_READ  = 63
SYS_WRITE = 64
SYS_EXIT  = 93
SYS_BRK   = 214

class SyscallHandler:
    def __init__(
        self,
        stdout: BinaryIO | None = None,
        stdin: BinaryIO | None = None,
    ) -> None:
        self._stdout = stdout or sys.stdout.buffer
        self._stdin = stdin or sys.stdin.buffer
        self._brk: int = 0  # Current program break

    @property
    def brk(self) -> int: ...

    @brk.setter
    def brk(self, value: int) -> None: ...

    def handle(self, cpu: CPU) -> bool:
        """Handle a syscall. Returns True if handled, False to fall through."""
        ...

    def _sys_write(self, cpu: CPU) -> None: ...
    def _sys_read(self, cpu: CPU) -> None: ...
    def _sys_exit(self, cpu: CPU) -> None: ...
    def _sys_brk(self, cpu: CPU) -> None: ...
```

- `handle(cpu)`: reads a7 (x17). Dispatches to handler. Returns True if handled.
  - If handled: set a0 (x10) to return value, return True.
  - If not recognized: return False (let normal ECALL logic run).
- `_sys_write`: reads a0=fd, a1=buf_ptr, a2=len. If fd==1 or fd==2, read `len` bytes from memory at buf_ptr, write to stdout. Set a0=len (bytes written). If fd not 1 or 2, set a0 to `(-1) & 0xFFFFFFFF` (i.e. 0xFFFFFFFF, 32-bit -1).
- `_sys_read`: reads a0=fd, a1=buf_ptr, a2=len. If fd==0, read from stdin stream. Write bytes to memory at buf_ptr. Set a0=bytes_read. If fd not 0, set a0 to `(-1) & 0xFFFFFFFF`.
- `_sys_exit`: reads a0=code. Set cpu.exit_code=code, cpu.halted=True.
- `_sys_brk`: reads a0=addr. If addr==0, return current brk in a0. If addr >= current brk, set brk=addr, return addr in a0. If addr < current brk, ignore, return current brk.

### D5: CPU + CLI Wiring

**Files to modify:**
- `src/riscv_npu/cpu/cpu.py` -- Add optional syscall_handler
- `src/riscv_npu/cpu/execute.py` -- Check syscall_handler in _exec_ecall
- `src/riscv_npu/cli.py` -- Build MemoryBus, register UART, create SyscallHandler

**CPU changes:**
```python
class CPU:
    def __init__(self, memory: MemoryBus) -> None:
        ...
        self.exit_code: int = 0
        self.syscall_handler: SyscallHandler | None = None
```

**execute.py _exec_ecall change:**
```python
def _exec_ecall(cpu: CPU, pc: int) -> int:
    # Try syscall handler first
    if cpu.syscall_handler is not None:
        if cpu.syscall_handler.handle(cpu):
            return (pc + 4) & 0xFFFFFFFF
    # Fall through to existing trap/halt logic
    mtvec = cpu.csr_read(_CSR_MTVEC)
    ...
```

**CLI changes:**
```python
def run_binary(path: str) -> None:
    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    uart = UART()
    bus.register(BASE, RAM_SIZE, ram)
    bus.register(UART_BASE, UART_SIZE, uart)

    cpu = CPU(bus)
    handler = SyscallHandler()
    cpu.syscall_handler = handler

    # Load ELF, set entry, set SP, etc.
    ...
    cpu.run()
    sys.exit(cpu.exit_code)
```

### D6: Hello World Firmware

**Files to create:**
- `firmware/common/syscalls.c` -- Syscall wrappers
- `firmware/hello/main.c` -- Hello World program
- `firmware/hello/Makefile` -- Build rules

**syscalls.c:**
```c
/* Minimal syscall wrappers for the RISC-V NPU emulator. */

static inline long syscall3(long n, long a0, long a1, long a2) {
    register long a7_r __asm__("a7") = n;
    register long a0_r __asm__("a0") = a0;
    register long a1_r __asm__("a1") = a1;
    register long a2_r __asm__("a2") = a2;
    __asm__ volatile("ecall"
                     : "+r"(a0_r)
                     : "r"(a7_r), "r"(a1_r), "r"(a2_r)
                     : "memory");
    return a0_r;
}

long write(int fd, const void *buf, long len) {
    return syscall3(64, fd, (long)buf, len);
}

void _exit(int code) {
    register long a7_r __asm__("a7") = 93;
    register long a0_r __asm__("a0") = code;
    __asm__ volatile("ecall" : : "r"(a7_r), "r"(a0_r));
    __builtin_unreachable();
}

int putchar(int c) {
    char ch = (char)c;
    write(1, &ch, 1);
    return c;
}

long puts(const char *s) {
    const char *p = s;
    while (*p) p++;
    write(1, s, p - s);
    putchar('\n');
    return 0;
}
```

**hello/main.c:**
```c
long write(int fd, const void *buf, long len);

int main(void) {
    const char *msg = "Hello, World!\n";
    const char *p = msg;
    while (*p) p++;
    write(1, msg, p - msg);
    return 0;
}
```

**hello/Makefile:**
```makefile
TARGET := hello.elf

all: $(TARGET)

include ../common/Makefile

SRCS := main.c ../common/syscalls.c
OBJS := main.o syscalls.o

$(TARGET): $(START_OBJ) $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

main.o: main.c
	$(CC) $(CFLAGS) -c -o $@ $<

syscalls.o: ../common/syscalls.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o *.elf $(START_OBJ)

.PHONY: all clean
```

---

## Test Coverage Requirements

### D1: Device Protocol + MemoryBus (`tests/memory/test_bus.py`)

**Tests:**
- `TestMemoryBus::test_register_and_read8` -- register a mock device, read8 returns device data
- `TestMemoryBus::test_register_and_write8` -- register a mock device, write8 reaches device
- `TestMemoryBus::test_read16_little_endian` -- read16 composes two read8 calls correctly
- `TestMemoryBus::test_read32_little_endian` -- read32 composes four read8 calls correctly
- `TestMemoryBus::test_write16_little_endian` -- write16 decomposes into two write8 calls
- `TestMemoryBus::test_write32_little_endian` -- write32 decomposes into four write8 calls
- `TestMemoryBus::test_unmapped_address_raises` -- access to unregistered address raises MemoryError
- `TestMemoryBus::test_multiple_devices` -- two devices at different ranges, routing works
- `TestMemoryBus::test_load_segment` -- load_segment writes bytes via write8
- `TestMemoryBus::test_overlapping_registration_raises` -- registering overlapping ranges raises ValueError

### D2: RAM as Bus Device + CPU Integration

No new test file. Existing tests must all pass after refactoring. The conftest changes are verified by running the full suite.

### D3: UART Device (`tests/devices/test_uart.py`)

**Tests:**
- `TestUART::test_tx_write_byte` -- write to THR offset, byte appears in tx_stream
- `TestUART::test_tx_multiple_bytes` -- write multiple bytes, all appear in order
- `TestUART::test_rx_empty_returns_zero` -- read RBR with empty buffer returns 0
- `TestUART::test_rx_push_and_read` -- push_rx bytes, read RBR returns them in order
- `TestUART::test_lsr_thr_empty` -- LSR always has bit 5 set (THR empty)
- `TestUART::test_lsr_data_ready_empty` -- LSR bit 0 clear when rx buffer empty
- `TestUART::test_lsr_data_ready_has_data` -- LSR bit 0 set when rx buffer has data
- `TestUART::test_read_unknown_register` -- reading other offsets returns 0
- `TestUART::test_write_unknown_register` -- writing other offsets is no-op (no error)
- `TestUART::test_ansi_passthrough` -- ANSI escape sequence bytes pass through TX unmodified

### D4: SyscallHandler (`tests/syscall/test_handler.py`)

**Tests:**
- `TestSysWrite::test_write_stdout` -- write syscall sends bytes to stdout stream
- `TestSysWrite::test_write_stderr` -- write syscall with fd=2 also works
- `TestSysWrite::test_write_bad_fd` -- write with fd=3 returns -1 (0xFFFFFFFF in a0)
- `TestSysRead::test_read_stdin` -- read syscall reads bytes from stdin stream
- `TestSysRead::test_read_bad_fd` -- read with fd=1 returns -1
- `TestSysExit::test_exit_zero` -- exit(0) halts cpu with exit_code=0
- `TestSysExit::test_exit_nonzero` -- exit(1) halts cpu with exit_code=1
- `TestSysBrk::test_brk_query` -- brk(0) returns current break
- `TestSysBrk::test_brk_extend` -- brk(addr) with addr > current sets new break
- `TestSysBrk::test_brk_shrink_ignored` -- brk(addr) with addr < current returns current
- `TestSyscallHandler::test_unknown_syscall_returns_false` -- unknown a7 value returns False (fall through)
- `TestSyscallHandler::test_known_syscall_returns_true` -- known syscall returns True

### D5: CPU + CLI Wiring

Tested via integration: existing tests still pass + D6 firmware test.

Additional unit tests in `tests/cpu/test_cpu.py`:
- `TestCPUSyscall::test_ecall_with_handler_write` -- CPU with syscall handler, ECALL for write does not trap
- `TestCPUSyscall::test_ecall_with_handler_unknown_falls_through` -- Unknown syscall falls through to trap/halt

### D6: Hello World Firmware

Integration test in `tests/integration/test_programs.py`:
- `test_hello_world` -- compile and run hello.elf, capture stdout, assert "Hello, World!\n"

This test requires the cross-compiler. If not available, it should be skipped (pytest.mark.skipif).

---

## Acceptance Criteria

1. `uv run pytest tests/memory/test_bus.py -v` -- all pass
2. `uv run pytest tests/devices/test_uart.py -v` -- all pass
3. `uv run pytest tests/syscall/test_handler.py -v` -- all pass
4. `uv run pytest` -- all 268+ existing tests still pass, plus all new tests
5. `cd firmware/hello && make` -- compiles successfully
6. `uv run python -m riscv_npu run firmware/hello/hello.elf` -- prints "Hello, World!"
7. All new functions have type hints and docstrings
8. 32-bit masking applied where needed (syscall return values, memory addresses)
