"""CPU core: fetch-decode-execute loop."""

from ..memory.bus import MemoryBus
from .decode import decode
from .execute import execute
from .registers import RegisterFile

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
        self.cycle_count: int = 0
        self.tohost: int = 0
        self.tohost_addr: int = 0  # Memory-mapped tohost address (set by test runner)
        self.csrs: dict[int, int] = {
            CSR_MHARTID: 0,  # Hart 0
        }

    def csr_read(self, addr: int) -> int:
        """Read a CSR value. Unknown CSRs return 0."""
        if addr == CSR_TOHOST:
            return self.tohost
        return self.csrs.get(addr, 0)

    def csr_write(self, addr: int, value: int) -> None:
        """Write a CSR value. Handles tohost specially."""
        value = value & 0xFFFFFFFF
        if addr == CSR_TOHOST:
            self.tohost = value
            if value != 0:
                self.halted = True
        else:
            self.csrs[addr] = value

    def step(self) -> None:
        """Execute one instruction cycle: fetch, decode, execute."""
        word = self.memory.read32(self.pc)
        inst = decode(word)
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
