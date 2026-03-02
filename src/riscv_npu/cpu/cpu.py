"""CPU core: fetch-decode-execute loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..memory.bus import MemoryBus
from ..npu.engine import NpuState
from .decode import decode, instruction_mnemonic
from .execute import execute
from .fpu import CSR_FCSR, CSR_FFLAGS, CSR_FRM, FpuState
from .registers import RegisterFile

if TYPE_CHECKING:
    from ..syscall.handler import SyscallHandler

# Well-known CSR addresses
CSR_MSTATUS = 0x300
CSR_MEDELEG = 0x302
CSR_MIDELEG = 0x303
CSR_MIE = 0x304
CSR_MTVEC = 0x305
CSR_MEPC = 0x341
CSR_MCAUSE = 0x342
CSR_SATP = 0x180
CSR_PMPADDR0 = 0x3B0
CSR_PMPCFG0 = 0x3A0
CSR_MHARTID = 0xF14
CSR_TOHOST = 0x51E


class CPU:
    """RISC-V CPU: fetch-decode-execute loop over a register file and memory."""

    def __init__(self, memory: MemoryBus) -> None:
        self.pc: int = 0
        self.registers = RegisterFile()
        self.memory = memory
        self.halted: bool = False
        self.exit_code: int = 0
        self.cycle_count: int = 0
        self.tohost: int = 0
        self.tohost_addr: int = 0  # Memory-mapped tohost address (set by test runner)
        self.npu_state = NpuState()
        self.fpu_state = FpuState()
        self.instruction_stats: dict[str, int] = {}
        self.syscall_handler: SyscallHandler | None = None
        self.csrs: dict[int, int] = {
            CSR_MHARTID: 0,  # Hart 0
        }

    def csr_read(self, addr: int) -> int:
        """Read a CSR value. Unknown CSRs return 0."""
        if addr == CSR_TOHOST:
            return self.tohost
        if addr == CSR_FFLAGS:
            return self.fpu_state.fflags
        if addr == CSR_FRM:
            return self.fpu_state.frm
        if addr == CSR_FCSR:
            return self.fpu_state.fcsr & 0xFF
        return self.csrs.get(addr, 0)

    def csr_write(self, addr: int, value: int) -> None:
        """Write a CSR value. Handles tohost and FPU CSRs specially."""
        value = value & 0xFFFFFFFF
        if addr == CSR_TOHOST:
            self.tohost = value
            if value != 0:
                self.halted = True
        elif addr == CSR_FFLAGS:
            self.fpu_state.fflags = value & 0x1F
        elif addr == CSR_FRM:
            self.fpu_state.frm = value & 0x7
        elif addr == CSR_FCSR:
            self.fpu_state.fcsr = value & 0xFF
        else:
            self.csrs[addr] = value

    def step(self) -> None:
        """Execute one instruction cycle: fetch, decode, execute.

        Also records the instruction mnemonic in ``instruction_stats``
        for per-instruction profiling.
        """
        word = self.memory.read32(self.pc)
        inst = decode(word)
        mnemonic = instruction_mnemonic(inst)
        self.instruction_stats[mnemonic] = self.instruction_stats.get(mnemonic, 0) + 1
        self.pc = execute(inst, self)
        self.cycle_count += 1

        # Check memory-mapped tohost (for riscv-tests compliance)
        if self.tohost_addr != 0:
            val = self.memory.read32(self.tohost_addr)
            if val != 0:
                self.tohost = val
                self.halted = True

    def run(self, max_cycles: int = 1_000_000) -> None:
        """Run until halted or cycle limit reached."""
        while not self.halted and self.cycle_count < max_cycles:
            self.step()
