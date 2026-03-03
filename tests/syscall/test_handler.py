"""Tests for the syscall dispatch handler."""

import errno
import io
from pathlib import Path

from riscv_npu.cpu.cpu import CPU
from riscv_npu.memory.bus import MemoryBus
from riscv_npu.memory.ram import RAM
from riscv_npu.syscall.handler import (
    SYS_BRK,
    SYS_CLOSE,
    SYS_EXIT,
    SYS_LSEEK,
    SYS_OPENAT,
    SYS_READ,
    SYS_WRITE,
    AT_FDCWD,
    O_CREAT,
    O_RDONLY,
    O_RDWR,
    O_TRUNC,
    O_WRONLY,
    SEEK_CUR,
    SEEK_END,
    SEEK_SET,
    SyscallHandler,
)

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
        """Write with unsupported fd returns -EBADF (0xFFFFFFF7) in a0."""
        cpu = _make_cpu()
        handler = SyscallHandler(stdout=io.BytesIO())

        cpu.registers.write(17, SYS_WRITE)
        cpu.registers.write(10, 99)  # fd=99 (bad, not in fd_table)
        cpu.registers.write(11, BASE + 0x1000)
        cpu.registers.write(12, 1)

        handler.handle(cpu)
        assert cpu.registers.read(10) == (-9) & 0xFFFFFFFF  # -EBADF


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
        """Read with unsupported fd returns -EBADF (0xFFFFFFF7) in a0."""
        cpu = _make_cpu()
        handler = SyscallHandler(stdin=io.BytesIO())

        cpu.registers.write(17, SYS_READ)
        cpu.registers.write(10, 99)  # fd=99 (bad, not in fd_table)
        cpu.registers.write(11, BASE + 0x2000)
        cpu.registers.write(12, 1)

        handler.handle(cpu)
        assert cpu.registers.read(10) == (-9) & 0xFFFFFFFF  # -EBADF


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


def _write_cstring(cpu: CPU, addr: int, s: str) -> None:
    """Write a null-terminated C string to guest memory."""
    for i, ch in enumerate(s.encode("utf-8")):
        cpu.memory.write8(addr + i, ch)
    cpu.memory.write8(addr + len(s.encode("utf-8")), 0)


class TestSysOpenat:
    """Tests for the openat(dirfd, pathname, flags, mode) syscall."""

    def test_create_new_file(self, tmp_path: Path) -> None:
        """openat with O_CREAT|O_WRONLY creates file and returns fd >= 3."""
        cpu = _make_cpu()
        handler = SyscallHandler(
            stdout=io.BytesIO(), fs_root=tmp_path,
        )

        path_addr = BASE + 0x3000
        _write_cstring(cpu, path_addr, "test.txt")

        cpu.registers.write(17, SYS_OPENAT)
        cpu.registers.write(10, AT_FDCWD)
        cpu.registers.write(11, path_addr)
        cpu.registers.write(12, O_WRONLY | O_CREAT | O_TRUNC)
        cpu.registers.write(13, 0o644)

        handler.handle(cpu)
        fd = cpu.registers.read(10)
        assert fd >= 3
        assert (tmp_path / "test.txt").exists()
        handler.close_all()

    def test_open_nonexistent_returns_enoent(self, tmp_path: Path) -> None:
        """openat for missing file without O_CREAT returns -ENOENT."""
        cpu = _make_cpu()
        handler = SyscallHandler(
            stdout=io.BytesIO(), fs_root=tmp_path,
        )

        path_addr = BASE + 0x3000
        _write_cstring(cpu, path_addr, "nope.txt")

        cpu.registers.write(17, SYS_OPENAT)
        cpu.registers.write(10, AT_FDCWD)
        cpu.registers.write(11, path_addr)
        cpu.registers.write(12, O_RDONLY)
        cpu.registers.write(13, 0)

        handler.handle(cpu)
        assert cpu.registers.read(10) == (-errno.ENOENT) & 0xFFFFFFFF

    def test_absolute_path_resolves_inside_sandbox(self, tmp_path: Path) -> None:
        """openat with absolute guest path resolves relative to fs_root."""
        cpu = _make_cpu()
        handler = SyscallHandler(
            stdout=io.BytesIO(), fs_root=tmp_path,
        )

        # Create a file at tmp_path/etc/test.txt
        (tmp_path / "etc").mkdir()
        (tmp_path / "etc" / "test.txt").write_text("inside sandbox")
        path_addr = BASE + 0x3000
        _write_cstring(cpu, path_addr, "/etc/test.txt")

        cpu.registers.write(17, SYS_OPENAT)
        cpu.registers.write(10, AT_FDCWD)
        cpu.registers.write(11, path_addr)
        cpu.registers.write(12, O_RDONLY)
        cpu.registers.write(13, 0)

        handler.handle(cpu)
        fd = cpu.registers.read(10)
        assert fd >= 3  # should succeed, not EACCES
        handler.close_all()

    def test_sandbox_escape_blocked(self, tmp_path: Path) -> None:
        """openat with '../' path escaping fs_root returns -EACCES."""
        cpu = _make_cpu()
        handler = SyscallHandler(
            stdout=io.BytesIO(), fs_root=tmp_path,
        )

        path_addr = BASE + 0x3000
        _write_cstring(cpu, path_addr, "../../../etc/passwd")

        cpu.registers.write(17, SYS_OPENAT)
        cpu.registers.write(10, AT_FDCWD)
        cpu.registers.write(11, path_addr)
        cpu.registers.write(12, O_RDONLY)
        cpu.registers.write(13, 0)

        handler.handle(cpu)
        assert cpu.registers.read(10) == (-errno.EACCES) & 0xFFFFFFFF

    def test_bad_dirfd_returns_ebadf(self, tmp_path: Path) -> None:
        """openat with dirfd != AT_FDCWD returns -EBADF."""
        cpu = _make_cpu()
        handler = SyscallHandler(
            stdout=io.BytesIO(), fs_root=tmp_path,
        )

        path_addr = BASE + 0x3000
        _write_cstring(cpu, path_addr, "test.txt")

        cpu.registers.write(17, SYS_OPENAT)
        cpu.registers.write(10, 5)  # not AT_FDCWD
        cpu.registers.write(11, path_addr)
        cpu.registers.write(12, O_RDONLY)
        cpu.registers.write(13, 0)

        handler.handle(cpu)
        assert cpu.registers.read(10) == (-errno.EBADF) & 0xFFFFFFFF

    def test_fd_limit_returns_emfile(self, tmp_path: Path) -> None:
        """openat returns -EMFILE when fd limit is reached."""
        cpu = _make_cpu()
        handler = SyscallHandler(
            stdout=io.BytesIO(), fs_root=tmp_path, max_fds=1,
        )

        # Create and open the first file to fill the limit
        (tmp_path / "a.txt").write_text("a")
        path_addr = BASE + 0x3000
        _write_cstring(cpu, path_addr, "a.txt")

        cpu.registers.write(17, SYS_OPENAT)
        cpu.registers.write(10, AT_FDCWD)
        cpu.registers.write(11, path_addr)
        cpu.registers.write(12, O_RDONLY)
        cpu.registers.write(13, 0)
        handler.handle(cpu)
        assert cpu.registers.read(10) >= 3  # first fd succeeds

        # Second open should fail
        (tmp_path / "b.txt").write_text("b")
        _write_cstring(cpu, path_addr, "b.txt")
        cpu.registers.write(17, SYS_OPENAT)
        cpu.registers.write(10, AT_FDCWD)
        cpu.registers.write(11, path_addr)
        cpu.registers.write(12, O_RDONLY)
        cpu.registers.write(13, 0)
        handler.handle(cpu)
        assert cpu.registers.read(10) == (-errno.EMFILE) & 0xFFFFFFFF
        handler.close_all()


class TestSysClose:
    """Tests for the close(fd) syscall."""

    def test_close_open_fd(self, tmp_path: Path) -> None:
        """close(fd) on an open file returns 0 and removes fd from table."""
        cpu = _make_cpu()
        handler = SyscallHandler(
            stdout=io.BytesIO(), fs_root=tmp_path,
        )

        (tmp_path / "f.txt").write_text("x")
        path_addr = BASE + 0x3000
        _write_cstring(cpu, path_addr, "f.txt")

        # Open
        cpu.registers.write(17, SYS_OPENAT)
        cpu.registers.write(10, AT_FDCWD)
        cpu.registers.write(11, path_addr)
        cpu.registers.write(12, O_RDONLY)
        cpu.registers.write(13, 0)
        handler.handle(cpu)
        fd = cpu.registers.read(10)
        assert fd >= 3

        # Close
        cpu.registers.write(17, SYS_CLOSE)
        cpu.registers.write(10, fd)
        handler.handle(cpu)
        assert cpu.registers.read(10) == 0
        assert fd not in handler._fd_table

    def test_close_bad_fd_returns_ebadf(self) -> None:
        """close(fd) on unknown fd returns -EBADF."""
        cpu = _make_cpu()
        handler = SyscallHandler(stdout=io.BytesIO())

        cpu.registers.write(17, SYS_CLOSE)
        cpu.registers.write(10, 42)

        handler.handle(cpu)
        assert cpu.registers.read(10) == (-errno.EBADF) & 0xFFFFFFFF

    def test_close_double_returns_ebadf(self, tmp_path: Path) -> None:
        """Closing an already-closed fd returns -EBADF."""
        cpu = _make_cpu()
        handler = SyscallHandler(
            stdout=io.BytesIO(), fs_root=tmp_path,
        )

        (tmp_path / "g.txt").write_text("y")
        path_addr = BASE + 0x3000
        _write_cstring(cpu, path_addr, "g.txt")

        # Open
        cpu.registers.write(17, SYS_OPENAT)
        cpu.registers.write(10, AT_FDCWD)
        cpu.registers.write(11, path_addr)
        cpu.registers.write(12, O_RDONLY)
        cpu.registers.write(13, 0)
        handler.handle(cpu)
        fd = cpu.registers.read(10)

        # Close once
        cpu.registers.write(17, SYS_CLOSE)
        cpu.registers.write(10, fd)
        handler.handle(cpu)
        assert cpu.registers.read(10) == 0

        # Close again
        cpu.registers.write(17, SYS_CLOSE)
        cpu.registers.write(10, fd)
        handler.handle(cpu)
        assert cpu.registers.read(10) == (-errno.EBADF) & 0xFFFFFFFF

    def test_close_stdio_is_noop(self) -> None:
        """close(0), close(1), close(2) succeed silently as no-ops."""
        cpu = _make_cpu()
        handler = SyscallHandler(stdout=io.BytesIO())

        for stdio_fd in (0, 1, 2):
            cpu.registers.write(17, SYS_CLOSE)
            cpu.registers.write(10, stdio_fd)
            handler.handle(cpu)
            assert cpu.registers.read(10) == 0


class TestSysLseek:
    """Tests for the lseek(fd, offset, whence) syscall."""

    def test_seek_set(self, tmp_path: Path) -> None:
        """lseek with SEEK_SET positions to the given offset."""
        cpu = _make_cpu()
        handler = SyscallHandler(
            stdout=io.BytesIO(), fs_root=tmp_path,
        )

        (tmp_path / "s.txt").write_bytes(b"ABCDEF")
        path_addr = BASE + 0x3000
        _write_cstring(cpu, path_addr, "s.txt")

        # Open
        cpu.registers.write(17, SYS_OPENAT)
        cpu.registers.write(10, AT_FDCWD)
        cpu.registers.write(11, path_addr)
        cpu.registers.write(12, O_RDONLY)
        cpu.registers.write(13, 0)
        handler.handle(cpu)
        fd = cpu.registers.read(10)

        # Seek to offset 3
        cpu.registers.write(17, SYS_LSEEK)
        cpu.registers.write(10, fd)
        cpu.registers.write(11, 3)
        cpu.registers.write(12, SEEK_SET)
        handler.handle(cpu)
        assert cpu.registers.read(10) == 3

        # Read should get "DEF"
        buf_addr = BASE + 0x4000
        cpu.registers.write(17, SYS_READ)
        cpu.registers.write(10, fd)
        cpu.registers.write(11, buf_addr)
        cpu.registers.write(12, 3)
        handler.handle(cpu)
        assert cpu.registers.read(10) == 3
        assert bytes(cpu.memory.read8(buf_addr + i) for i in range(3)) == b"DEF"
        handler.close_all()

    def test_seek_bad_fd_returns_ebadf(self) -> None:
        """lseek on unknown fd returns -EBADF."""
        cpu = _make_cpu()
        handler = SyscallHandler(stdout=io.BytesIO())

        cpu.registers.write(17, SYS_LSEEK)
        cpu.registers.write(10, 42)
        cpu.registers.write(11, 0)
        cpu.registers.write(12, SEEK_SET)
        handler.handle(cpu)
        assert cpu.registers.read(10) == (-errno.EBADF) & 0xFFFFFFFF

    def test_seek_stdio_returns_espipe(self) -> None:
        """lseek on stdin/stdout/stderr returns -ESPIPE."""
        cpu = _make_cpu()
        handler = SyscallHandler(stdout=io.BytesIO())

        for stdio_fd in (0, 1, 2):
            cpu.registers.write(17, SYS_LSEEK)
            cpu.registers.write(10, stdio_fd)
            cpu.registers.write(11, 0)
            cpu.registers.write(12, SEEK_SET)
            handler.handle(cpu)
            assert cpu.registers.read(10) == (-errno.ESPIPE) & 0xFFFFFFFF

    def test_seek_invalid_whence_returns_einval(self, tmp_path: Path) -> None:
        """lseek with invalid whence returns -EINVAL."""
        cpu = _make_cpu()
        handler = SyscallHandler(
            stdout=io.BytesIO(), fs_root=tmp_path,
        )

        (tmp_path / "w.txt").write_bytes(b"ABC")
        path_addr = BASE + 0x3000
        _write_cstring(cpu, path_addr, "w.txt")

        # Open
        cpu.registers.write(17, SYS_OPENAT)
        cpu.registers.write(10, AT_FDCWD)
        cpu.registers.write(11, path_addr)
        cpu.registers.write(12, O_RDONLY)
        cpu.registers.write(13, 0)
        handler.handle(cpu)
        fd = cpu.registers.read(10)

        # Invalid whence=99
        cpu.registers.write(17, SYS_LSEEK)
        cpu.registers.write(10, fd)
        cpu.registers.write(11, 0)
        cpu.registers.write(12, 99)
        handler.handle(cpu)
        assert cpu.registers.read(10) == (-errno.EINVAL) & 0xFFFFFFFF
        handler.close_all()


class TestFileReadWrite:
    """Tests for read/write on file descriptors."""

    def test_write_then_read_roundtrip(self, tmp_path: Path) -> None:
        """Write to file, seek back, read back same content."""
        cpu = _make_cpu()
        handler = SyscallHandler(
            stdout=io.BytesIO(), fs_root=tmp_path,
        )

        path_addr = BASE + 0x3000
        _write_cstring(cpu, path_addr, "rw.txt")

        # Open for read+write with create+truncate
        cpu.registers.write(17, SYS_OPENAT)
        cpu.registers.write(10, AT_FDCWD)
        cpu.registers.write(11, path_addr)
        cpu.registers.write(12, O_RDWR | O_CREAT | O_TRUNC)
        cpu.registers.write(13, 0o644)
        handler.handle(cpu)
        fd = cpu.registers.read(10)
        assert fd >= 3

        # Write "Hello"
        msg = b"Hello"
        buf_addr = BASE + 0x4000
        for i, byte in enumerate(msg):
            cpu.memory.write8(buf_addr + i, byte)

        cpu.registers.write(17, SYS_WRITE)
        cpu.registers.write(10, fd)
        cpu.registers.write(11, buf_addr)
        cpu.registers.write(12, len(msg))
        handler.handle(cpu)
        assert cpu.registers.read(10) == 5

        # Seek back to start
        cpu.registers.write(17, SYS_LSEEK)
        cpu.registers.write(10, fd)
        cpu.registers.write(11, 0)
        cpu.registers.write(12, SEEK_SET)
        handler.handle(cpu)
        assert cpu.registers.read(10) == 0

        # Read back
        read_buf = BASE + 0x5000
        cpu.registers.write(17, SYS_READ)
        cpu.registers.write(10, fd)
        cpu.registers.write(11, read_buf)
        cpu.registers.write(12, 10)
        handler.handle(cpu)
        assert cpu.registers.read(10) == 5
        assert bytes(cpu.memory.read8(read_buf + i) for i in range(5)) == b"Hello"

        handler.close_all()
        # Verify file persisted on host
        assert (tmp_path / "rw.txt").read_bytes() == b"Hello"

    def test_no_fs_root_uses_cwd(self, tmp_path: Path, monkeypatch: object) -> None:
        """Without fs_root, paths resolve relative to CWD."""
        monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]

        cpu = _make_cpu()
        handler = SyscallHandler(stdout=io.BytesIO())

        path_addr = BASE + 0x3000
        _write_cstring(cpu, path_addr, "cwd_test.txt")

        cpu.registers.write(17, SYS_OPENAT)
        cpu.registers.write(10, AT_FDCWD)
        cpu.registers.write(11, path_addr)
        cpu.registers.write(12, O_WRONLY | O_CREAT | O_TRUNC)
        cpu.registers.write(13, 0o644)
        handler.handle(cpu)
        fd = cpu.registers.read(10)
        assert fd >= 3
        assert (tmp_path / "cwd_test.txt").exists()
        handler.close_all()
