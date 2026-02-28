"""Tests for the syscall dispatch handler."""

import io

from riscv_npu.cpu.cpu import CPU
from riscv_npu.memory.bus import MemoryBus
from riscv_npu.memory.ram import RAM
from riscv_npu.syscall.handler import SYS_BRK, SYS_EXIT, SYS_READ, SYS_WRITE, SyscallHandler

BASE = 0x80000000
RAM_SIZE = 1024 * 1024  # 1 MB


def _make_cpu() -> CPU:
    """Create a CPU with MemoryBus and RAM for syscall testing."""
    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    bus.register(BASE, RAM_SIZE, ram)
    cpu = CPU(bus)
    cpu.pc = BASE
    return cpu


class TestSysWrite:
    """Tests for the write(fd, buf, len) syscall."""

    def test_write_stdout(self) -> None:
        """Write syscall sends bytes from memory to stdout stream."""
        cpu = _make_cpu()
        stdout = io.BytesIO()
        handler = SyscallHandler(stdout=stdout)

        # Store "Hello" at BASE+0x1000
        msg = b"Hello"
        buf_addr = BASE + 0x1000
        for i, byte in enumerate(msg):
            cpu.memory.write8(buf_addr + i, byte)

        # Set syscall args: a7=64, a0=1(fd), a1=buf_addr, a2=5(len)
        cpu.registers.write(17, SYS_WRITE)
        cpu.registers.write(10, 1)
        cpu.registers.write(11, buf_addr)
        cpu.registers.write(12, len(msg))

        result = handler.handle(cpu)
        assert result is True
        assert stdout.getvalue() == b"Hello"
        assert cpu.registers.read(10) == 5  # bytes written

    def test_write_stderr(self) -> None:
        """Write syscall with fd=2 (stderr) also writes to stdout stream."""
        cpu = _make_cpu()
        stdout = io.BytesIO()
        handler = SyscallHandler(stdout=stdout)

        msg = b"err"
        buf_addr = BASE + 0x1000
        for i, byte in enumerate(msg):
            cpu.memory.write8(buf_addr + i, byte)

        cpu.registers.write(17, SYS_WRITE)
        cpu.registers.write(10, 2)  # fd=2 (stderr)
        cpu.registers.write(11, buf_addr)
        cpu.registers.write(12, len(msg))

        result = handler.handle(cpu)
        assert result is True
        assert stdout.getvalue() == b"err"

    def test_write_bad_fd(self) -> None:
        """Write with unsupported fd returns -1 (0xFFFFFFFF) in a0."""
        cpu = _make_cpu()
        handler = SyscallHandler(stdout=io.BytesIO())

        cpu.registers.write(17, SYS_WRITE)
        cpu.registers.write(10, 3)  # fd=3 (bad)
        cpu.registers.write(11, BASE + 0x1000)
        cpu.registers.write(12, 1)

        handler.handle(cpu)
        assert cpu.registers.read(10) == 0xFFFFFFFF


class TestSysRead:
    """Tests for the read(fd, buf, len) syscall."""

    def test_read_stdin(self) -> None:
        """Read syscall reads bytes from stdin into memory."""
        cpu = _make_cpu()
        stdin = io.BytesIO(b"ABC")
        handler = SyscallHandler(stdin=stdin)

        buf_addr = BASE + 0x2000
        cpu.registers.write(17, SYS_READ)
        cpu.registers.write(10, 0)  # fd=0 (stdin)
        cpu.registers.write(11, buf_addr)
        cpu.registers.write(12, 3)  # len=3

        result = handler.handle(cpu)
        assert result is True
        assert cpu.registers.read(10) == 3  # bytes read
        assert cpu.memory.read8(buf_addr) == ord("A")
        assert cpu.memory.read8(buf_addr + 1) == ord("B")
        assert cpu.memory.read8(buf_addr + 2) == ord("C")

    def test_read_bad_fd(self) -> None:
        """Read with unsupported fd returns -1 (0xFFFFFFFF) in a0."""
        cpu = _make_cpu()
        handler = SyscallHandler(stdin=io.BytesIO())

        cpu.registers.write(17, SYS_READ)
        cpu.registers.write(10, 1)  # fd=1 (bad for read)
        cpu.registers.write(11, BASE + 0x2000)
        cpu.registers.write(12, 1)

        handler.handle(cpu)
        assert cpu.registers.read(10) == 0xFFFFFFFF


class TestSysExit:
    """Tests for the exit(code) syscall."""

    def test_exit_zero(self) -> None:
        """exit(0) halts the CPU with exit_code=0."""
        cpu = _make_cpu()
        handler = SyscallHandler()

        cpu.registers.write(17, SYS_EXIT)
        cpu.registers.write(10, 0)

        handler.handle(cpu)
        assert cpu.halted is True
        assert cpu.exit_code == 0

    def test_exit_nonzero(self) -> None:
        """exit(1) halts the CPU with exit_code=1."""
        cpu = _make_cpu()
        handler = SyscallHandler()

        cpu.registers.write(17, SYS_EXIT)
        cpu.registers.write(10, 1)

        handler.handle(cpu)
        assert cpu.halted is True
        assert cpu.exit_code == 1


class TestSysBrk:
    """Tests for the brk(addr) syscall."""

    def test_brk_query(self) -> None:
        """brk(0) returns the current program break."""
        cpu = _make_cpu()
        handler = SyscallHandler()
        handler.brk = 0x80010000

        cpu.registers.write(17, SYS_BRK)
        cpu.registers.write(10, 0)

        handler.handle(cpu)
        assert cpu.registers.read(10) == 0x80010000

    def test_brk_extend(self) -> None:
        """brk(addr) with addr > current sets new break and returns it."""
        cpu = _make_cpu()
        handler = SyscallHandler()
        handler.brk = 0x80010000

        cpu.registers.write(17, SYS_BRK)
        cpu.registers.write(10, 0x80020000)

        handler.handle(cpu)
        assert cpu.registers.read(10) == 0x80020000
        assert handler.brk == 0x80020000

    def test_brk_shrink_ignored(self) -> None:
        """brk(addr) with addr < current returns current break unchanged."""
        cpu = _make_cpu()
        handler = SyscallHandler()
        handler.brk = 0x80020000

        cpu.registers.write(17, SYS_BRK)
        cpu.registers.write(10, 0x80010000)

        handler.handle(cpu)
        assert cpu.registers.read(10) == 0x80020000
        assert handler.brk == 0x80020000


class TestSyscallHandler:
    """Tests for syscall dispatch logic."""

    def test_unknown_syscall_returns_false(self) -> None:
        """Unrecognized syscall number returns False (fall through)."""
        cpu = _make_cpu()
        handler = SyscallHandler()

        cpu.registers.write(17, 999)  # unknown syscall

        result = handler.handle(cpu)
        assert result is False

    def test_known_syscall_returns_true(self) -> None:
        """Recognized syscall returns True."""
        cpu = _make_cpu()
        handler = SyscallHandler(stdout=io.BytesIO())

        # Set up a write syscall
        cpu.registers.write(17, SYS_WRITE)
        cpu.registers.write(10, 1)  # fd=1
        cpu.registers.write(11, BASE)  # buf
        cpu.registers.write(12, 0)  # len=0

        result = handler.handle(cpu)
        assert result is True
