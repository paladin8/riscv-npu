"""CPU core: fetch-decode-execute loop."""

from ..memory.ram import RAM
from .decode import decode
from .execute import execute
from .registers import RegisterFile


class CPU:
    """RISC-V CPU: fetch-decode-execute loop over a register file and memory."""

    def __init__(self, memory: RAM) -> None:
        self.pc: int = 0
        self.registers = RegisterFile()
        self.memory = memory
        self.halted: bool = False
        self.cycle_count: int = 0

    def step(self) -> None:
        """Execute one instruction cycle: fetch, decode, execute."""
        word = self.memory.read32(self.pc)
        inst = decode(word)
        self.pc = execute(inst, self)
        self.cycle_count += 1

    def run(self, max_cycles: int = 1_000_000) -> None:
        """Run until halted or cycle limit reached."""
        while not self.halted and self.cycle_count < max_cycles:
            self.step()
