"""Register file: 32 x 32-bit general purpose registers."""


class RegisterFile:
    """32 general-purpose registers. x0 is hardwired to 0."""

    def __init__(self) -> None:
        self._regs: list[int] = [0] * 32

    def read(self, index: int) -> int:
        """Read register value. x0 always returns 0."""
        if index == 0:
            return 0
        return self._regs[index]

    def write(self, index: int, value: int) -> None:
        """Write register value. Writes to x0 are discarded. Value masked to 32 bits."""
        if index == 0:
            return
        self._regs[index] = value & 0xFFFFFFFF
