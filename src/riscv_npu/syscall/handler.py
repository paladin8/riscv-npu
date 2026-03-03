"""Syscall dispatch handler for ECALL instructions.

Implements Linux-style RISC-V syscalls: read, write, exit, brk,
openat, close, lseek. The handler intercepts ECALL before the CPU's
trap mechanism, checking register a7 for the syscall number and
dispatching accordingly.
"""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from ..cpu.cpu import CPU

# Linux syscall numbers (RISC-V ABI)
SYS_OPENAT = 56
SYS_CLOSE = 57
SYS_LSEEK = 62
SYS_READ = 63
SYS_WRITE = 64
SYS_EXIT = 93
SYS_BRK = 214

# openat constants
AT_FDCWD = (-100) & 0xFFFFFFFF  # unsigned form of -100

# Open flags (Linux RISC-V ABI values)
O_RDONLY = 0
O_WRONLY = 1
O_RDWR = 2
O_CREAT = 64
O_TRUNC = 512
O_APPEND = 1024

# Seek whence constants
SEEK_SET = 0
SEEK_CUR = 1
SEEK_END = 2


class SyscallHandler:
    """Dispatches Linux-style RISC-V syscalls.

    On ECALL, reads a7 (x17) for the syscall number. Supported:
    - 56  (openat): open file relative to directory fd
    - 57  (close):  close a file descriptor
    - 62  (lseek):  reposition file offset
    - 63  (read):   read from fd into memory buffer
    - 64  (write):  write memory buffer to fd
    - 93  (exit):   halt CPU with exit code
    - 214 (brk):    bump allocator for program break

    Returns True if the syscall was handled, False to fall through
    to the CPU's existing trap/halt logic.
    """

    def __init__(
        self,
        stdout: BinaryIO | None = None,
        stdin: BinaryIO | None = None,
        fs_root: Path | None = None,
        max_fds: int = 64,
    ) -> None:
        self._stdout = stdout if stdout is not None else sys.stdout.buffer
        self._stdin = stdin if stdin is not None else sys.stdin.buffer
        self._brk: int = 0
        self._fs_root = fs_root
        self._max_fds = max_fds
        self._fd_table: dict[int, BinaryIO] = {}
        self._next_fd: int = 3

    @property
    def brk(self) -> int:
        """Current program break address."""
        return self._brk

    @brk.setter
    def brk(self, value: int) -> None:
        """Set the program break address."""
        self._brk = value & 0xFFFFFFFF

    def _read_cstring(self, cpu: CPU, addr: int) -> str:
        """Read a null-terminated C string from guest memory.

        Args:
            cpu: The CPU instance for memory access.
            addr: Guest address of the string start.

        Returns:
            The decoded string (UTF-8, replacing errors).
        """
        chars: list[int] = []
        while True:
            byte = cpu.memory.read8(addr)
            if byte == 0:
                break
            chars.append(byte)
            addr = (addr + 1) & 0xFFFFFFFF
        return bytes(chars).decode("utf-8", errors="replace")

    def _alloc_fd(self) -> int | None:
        """Allocate the next available file descriptor.

        Returns:
            The new fd number, or None if the fd limit is reached.
        """
        if len(self._fd_table) >= self._max_fds:
            return None
        fd = self._next_fd
        self._next_fd += 1
        return fd

    def _resolve_path(self, raw_path: str) -> Path | None:
        """Resolve a guest path to a host path, enforcing sandbox.

        If fs_root is set, the path is resolved relative to it and
        must not escape via '..' traversal. If fs_root is None, the
        path is used relative to CWD.

        Args:
            raw_path: The path string from the guest.

        Returns:
            Resolved host Path, or None if the path escapes the sandbox.
        """
        if self._fs_root is not None:
            resolved = (self._fs_root / raw_path).resolve()
            fs_root_resolved = self._fs_root.resolve()
            if not str(resolved).startswith(str(fs_root_resolved) + os.sep) and \
               resolved != fs_root_resolved:
                return None
            return resolved
        return Path(raw_path)

    def close_all(self) -> None:
        """Close all open file descriptors in the fd table."""
        for f in self._fd_table.values():
            try:
                f.close()
            except OSError:
                pass
        self._fd_table.clear()

    def handle(self, cpu: CPU) -> bool:
        """Handle a syscall if recognized.

        Reads a7 (x17) for the syscall number. If recognized, processes
        the syscall, sets return value in a0 (x10), and returns True.
        If not recognized, returns False to let the CPU's normal ECALL
        logic (trap/halt) handle it.

        Args:
            cpu: The CPU instance (for register and memory access).

        Returns:
            True if the syscall was handled, False otherwise.
        """
        syscall_num = cpu.registers.read(17)  # a7

        if syscall_num == SYS_WRITE:
            self._sys_write(cpu)
            return True
        elif syscall_num == SYS_READ:
            self._sys_read(cpu)
            return True
        elif syscall_num == SYS_EXIT:
            self._sys_exit(cpu)
            return True
        elif syscall_num == SYS_BRK:
            self._sys_brk(cpu)
            return True
        elif syscall_num == SYS_OPENAT:
            self._sys_openat(cpu)
            return True
        elif syscall_num == SYS_CLOSE:
            self._sys_close(cpu)
            return True
        elif syscall_num == SYS_LSEEK:
            self._sys_lseek(cpu)
            return True
        return False

    def _sys_write(self, cpu: CPU) -> None:
        """Handle write(fd, buf, len) syscall.

        a0=fd, a1=buf_ptr, a2=len. Writes to stdout (fd=1/2),
        or to an open file fd. Sets a0 to bytes written, or -EBADF.
        """
        fd = cpu.registers.read(10)   # a0
        buf_ptr = cpu.registers.read(11)  # a1
        length = cpu.registers.read(12)   # a2

        if fd == 1 or fd == 2:
            data = bytes(
                cpu.memory.read8(buf_ptr + i) for i in range(length)
            )
            self._stdout.write(data)
            self._stdout.flush()
            cpu.registers.write(10, length & 0xFFFFFFFF)
        elif fd in self._fd_table:
            data = bytes(
                cpu.memory.read8(buf_ptr + i) for i in range(length)
            )
            written = self._fd_table[fd].write(data)
            self._fd_table[fd].flush()
            cpu.registers.write(10, (written if written else 0) & 0xFFFFFFFF)
        else:
            cpu.registers.write(10, (-errno.EBADF) & 0xFFFFFFFF)

    def _sys_read(self, cpu: CPU) -> None:
        """Handle read(fd, buf, len) syscall.

        a0=fd, a1=buf_ptr, a2=len. Reads from stdin (fd=0),
        or from an open file fd. Sets a0 to bytes read, or -EBADF.
        """
        fd = cpu.registers.read(10)
        buf_ptr = cpu.registers.read(11)
        length = cpu.registers.read(12)

        if fd == 0:
            data = self._stdin.read(length)
            if data is None:
                data = b""
            for i, byte in enumerate(data):
                cpu.memory.write8(buf_ptr + i, byte)
            cpu.registers.write(10, len(data) & 0xFFFFFFFF)
        elif fd in self._fd_table:
            data = self._fd_table[fd].read(length)
            if data is None:
                data = b""
            for i, byte in enumerate(data):
                cpu.memory.write8(buf_ptr + i, byte)
            cpu.registers.write(10, len(data) & 0xFFFFFFFF)
        else:
            cpu.registers.write(10, (-errno.EBADF) & 0xFFFFFFFF)

    def _sys_exit(self, cpu: CPU) -> None:
        """Handle exit(code) syscall.

        a0=exit_code. Halts the CPU and stores the exit code.
        """
        code = cpu.registers.read(10)
        cpu.exit_code = code & 0xFFFFFFFF
        cpu.halted = True

    def _sys_brk(self, cpu: CPU) -> None:
        """Handle brk(addr) syscall.

        a0=addr. If addr==0, returns current break. If addr >= current
        break, extends to addr. If addr < current, returns current break
        unchanged.
        """
        addr = cpu.registers.read(10)
        if addr == 0:
            cpu.registers.write(10, self._brk & 0xFFFFFFFF)
        elif addr >= self._brk:
            self._brk = addr & 0xFFFFFFFF
            cpu.registers.write(10, self._brk)
        else:
            cpu.registers.write(10, self._brk & 0xFFFFFFFF)

    def _sys_openat(self, cpu: CPU) -> None:
        """Handle openat(dirfd, pathname, flags, mode) syscall.

        a0=dirfd, a1=pathname_ptr, a2=flags, a3=mode.
        Only supports AT_FDCWD (-100) as dirfd.
        Sets a0 to new fd on success, or -errno on failure.
        """
        dirfd = cpu.registers.read(10)
        pathname_ptr = cpu.registers.read(11)
        flags = cpu.registers.read(12)
        # a3=mode (unused for now, permissions not enforced)

        # Only AT_FDCWD supported
        if dirfd != AT_FDCWD:
            cpu.registers.write(10, (-errno.EBADF) & 0xFFFFFFFF)
            return

        raw_path = self._read_cstring(cpu, pathname_ptr)

        resolved = self._resolve_path(raw_path)
        if resolved is None:
            cpu.registers.write(10, (-errno.EACCES) & 0xFFFFFFFF)
            return

        fd_num = self._alloc_fd()
        if fd_num is None:
            cpu.registers.write(10, (-errno.EMFILE) & 0xFFFFFFFF)
            return

        # Map Linux flags to Python open mode
        access = flags & 3  # O_RDONLY=0, O_WRONLY=1, O_RDWR=2
        if access == O_RDONLY:
            mode = "rb"
        elif access == O_WRONLY:
            mode = "wb" if (flags & O_TRUNC) else "r+b"
            if flags & O_APPEND:
                mode = "ab"
        elif access == O_RDWR:
            mode = "w+b" if (flags & O_TRUNC) else "r+b"
            if flags & O_APPEND:
                mode = "a+b"
        else:
            cpu.registers.write(10, (-errno.EINVAL) & 0xFFFFFFFF)
            return

        try:
            if (flags & O_CREAT) and not resolved.exists():
                # Create the file first if O_CREAT
                resolved.touch()
            f = open(resolved, mode)
        except FileNotFoundError:
            cpu.registers.write(10, (-errno.ENOENT) & 0xFFFFFFFF)
            return
        except PermissionError:
            cpu.registers.write(10, (-errno.EACCES) & 0xFFFFFFFF)
            return
        except IsADirectoryError:
            cpu.registers.write(10, (-errno.EISDIR) & 0xFFFFFFFF)
            return
        except OSError:
            cpu.registers.write(10, (-errno.EACCES) & 0xFFFFFFFF)
            return

        self._fd_table[fd_num] = f
        cpu.registers.write(10, fd_num & 0xFFFFFFFF)

    def _sys_close(self, cpu: CPU) -> None:
        """Handle close(fd) syscall.

        a0=fd. Closes the file descriptor and removes it from the table.
        Closing stdin/stdout/stderr (0-2) is a silent no-op.
        Sets a0 to 0 on success, or -EBADF for bad fd.
        """
        fd = cpu.registers.read(10)

        # Closing stdio fds is a silent no-op
        if fd <= 2:
            cpu.registers.write(10, 0)
            return

        if fd not in self._fd_table:
            cpu.registers.write(10, (-errno.EBADF) & 0xFFFFFFFF)
            return

        try:
            self._fd_table[fd].close()
        except OSError:
            pass
        del self._fd_table[fd]
        cpu.registers.write(10, 0)

    def _sys_lseek(self, cpu: CPU) -> None:
        """Handle lseek(fd, offset, whence) syscall.

        a0=fd, a1=offset (signed 32-bit), a2=whence.
        Sets a0 to new file position on success, or -errno on failure.
        """
        fd = cpu.registers.read(10)
        raw_offset = cpu.registers.read(11)
        whence = cpu.registers.read(12)

        # Interpret offset as signed 32-bit
        if raw_offset >= 0x80000000:
            offset = raw_offset - 0x100000000
        else:
            offset = raw_offset

        # lseek on stdio fds returns -ESPIPE
        if fd <= 2:
            cpu.registers.write(10, (-errno.ESPIPE) & 0xFFFFFFFF)
            return

        if fd not in self._fd_table:
            cpu.registers.write(10, (-errno.EBADF) & 0xFFFFFFFF)
            return

        if whence not in (SEEK_SET, SEEK_CUR, SEEK_END):
            cpu.registers.write(10, (-errno.EINVAL) & 0xFFFFFFFF)
            return

        try:
            pos = self._fd_table[fd].seek(offset, whence)
            cpu.registers.write(10, pos & 0xFFFFFFFF)
        except OSError:
            cpu.registers.write(10, (-errno.EINVAL) & 0xFFFFFFFF)
