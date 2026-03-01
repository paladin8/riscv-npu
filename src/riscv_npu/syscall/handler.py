"""Syscall dispatch handler for ECALL instructions.

Implements Linux-style RISC-V syscalls: read, write, exit, brk.
The handler intercepts ECALL before the CPU's trap mechanism, checking
register a7 for the syscall number and dispatching accordingly.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from ..cpu.cpu import CPU

# Linux syscall numbers (RISC-V ABI)
SYS_READ = 63
SYS_WRITE = 64
SYS_EXIT = 93
SYS_BRK = 214



class SyscallHandler:
    """Dispatches Linux-style RISC-V syscalls.

    On ECALL, reads a7 (x17) for the syscall number. Supported:
    - 63  (read):  read from stdin into memory buffer
    - 64  (write): write memory buffer to stdout
    - 93  (exit):  halt CPU with exit code
    - 214 (brk):   bump allocator for program break

    Returns True if the syscall was handled, False to fall through
    to the CPU's existing trap/halt logic.
    """

    def __init__(
        self,
        stdout: BinaryIO | None = None,
        stdin: BinaryIO | None = None,
    ) -> None:
        self._stdout = stdout if stdout is not None else sys.stdout.buffer
        self._stdin = stdin if stdin is not None else sys.stdin.buffer
        self._brk: int = 0

    @property
    def brk(self) -> int:
        """Current program break address."""
        return self._brk

    @brk.setter
    def brk(self, value: int) -> None:
        """Set the program break address."""
        self._brk = value & 0xFFFFFFFF

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
        return False

    def _sys_write(self, cpu: CPU) -> None:
        """Handle write(fd, buf, len) syscall.

        a0=fd, a1=buf_ptr, a2=len. Writes to stdout (fd=1) or
        stderr (fd=2). Sets a0 to bytes written, or -1 for bad fd.
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
        else:
            cpu.registers.write(10, (-1) & 0xFFFFFFFF)

    def _sys_read(self, cpu: CPU) -> None:
        """Handle read(fd, buf, len) syscall.

        a0=fd, a1=buf_ptr, a2=len. Reads from stdin (fd=0).
        Sets a0 to bytes read, or -1 for bad fd.
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
        else:
            cpu.registers.write(10, (-1) & 0xFFFFFFFF)

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
