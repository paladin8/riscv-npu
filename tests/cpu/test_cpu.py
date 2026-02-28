"""Tests for CPU fetch-decode-execute loop."""

from riscv_npu.cpu.cpu import CPU
from riscv_npu.memory.ram import RAM

BASE = 0x80000000

# Encoding helpers
OP_R = 0x33
OP_I = 0x13
OP_JAL = 0x6F
OP_SYSTEM = 0x73


def _r(funct7: int, rs2: int, rs1: int, funct3: int, rd: int) -> int:
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | OP_R


def _i(imm12: int, rs1: int, funct3: int, rd: int, opcode: int = OP_I) -> int:
    return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def _j(imm21: int, rd: int) -> int:
    imm = imm21 & 0x1FFFFF
    bit_20 = (imm >> 20) & 1
    bits_10_1 = (imm >> 1) & 0x3FF
    bit_11 = (imm >> 11) & 1
    bits_19_12 = (imm >> 12) & 0xFF
    return (bit_20 << 31) | (bits_10_1 << 21) | (bit_11 << 20) | \
           (bits_19_12 << 12) | (rd << 7) | OP_JAL


def _make_cpu() -> CPU:
    ram = RAM(BASE, 1024 * 1024)
    cpu = CPU(ram)
    cpu.pc = BASE
    return cpu


def _write_program(cpu: CPU, words: list[int]) -> None:
    for i, word in enumerate(words):
        cpu.memory.write32(BASE + i * 4, word)


class TestCPUStep:
    def test_advances_pc(self) -> None:
        cpu = _make_cpu()
        # NOP = ADDI x0, x0, 0
        _write_program(cpu, [_i(0, 0, 0b000, 0)])
        cpu.step()
        assert cpu.pc == BASE + 4

    def test_increments_cycle(self) -> None:
        cpu = _make_cpu()
        _write_program(cpu, [_i(0, 0, 0b000, 0)])
        assert cpu.cycle_count == 0
        cpu.step()
        assert cpu.cycle_count == 1

    def test_multiple_steps(self) -> None:
        cpu = _make_cpu()
        # ADDI x1, x0, 5; ADDI x2, x0, 10; ADD x3, x1, x2
        _write_program(cpu, [
            _i(5, 0, 0b000, 1),      # ADDI x1, x0, 5
            _i(10, 0, 0b000, 2),     # ADDI x2, x0, 10
            _r(0, 2, 1, 0b000, 3),   # ADD x3, x1, x2
        ])
        cpu.step()
        cpu.step()
        cpu.step()
        assert cpu.registers.read(3) == 15


class TestCPURun:
    def test_stops_on_ecall(self) -> None:
        cpu = _make_cpu()
        _write_program(cpu, [
            _i(5, 0, 0b000, 1),               # ADDI x1, x0, 5
            _i(0, 0, 0b000, 0, OP_SYSTEM),    # ECALL
        ])
        cpu.run()
        assert cpu.halted is True
        assert cpu.cycle_count == 2
        assert cpu.registers.read(1) == 5

    def test_stops_at_max_cycles(self) -> None:
        cpu = _make_cpu()
        # Infinite loop: JAL x0, 0 (jump to self)
        _write_program(cpu, [_j(0, 0)])
        cpu.run(max_cycles=100)
        assert cpu.cycle_count == 100
        assert cpu.halted is False


class TestFibonacci:
    def test_fib_10(self) -> None:
        """Hand-encoded Fibonacci: fib(10) = 55, result in x10."""
        cpu = _make_cpu()
        # Program:
        #   ADDI x10, x0, 0     # a = 0
        #   ADDI x11, x0, 1     # b = 1
        #   ADDI x12, x0, 10    # n = 10
        #   ADDI x13, x0, 0     # i = 0
        # loop:                  # offset = 16 (4 instructions * 4)
        #   BEQ x13, x12, done  # if i == n, jump to done (+24 = 6 instr forward)
        #   ADD x14, x10, x11   # t = a + b
        #   ADD x10, x11, x0    # a = b  (ADD x10, x11, x0)
        #   ADD x11, x14, x0    # b = t  (ADD x11, x14, x0)
        #   ADDI x13, x13, 1    # i++
        #   JAL x0, loop        # jump back to loop (-20 bytes)
        # done:                  # offset = 40 (10 instructions * 4)
        #   ECALL                # halt

        # Encode BEQ x13, x12, +24 (6 instructions forward from BEQ = 24 bytes)
        # B-type: imm13=24
        def _b(imm13: int, rs2: int, rs1: int, funct3: int) -> int:
            imm = imm13 & 0x1FFF
            bit_12 = (imm >> 12) & 1
            bit_11 = (imm >> 11) & 1
            bits_10_5 = (imm >> 5) & 0x3F
            bits_4_1 = (imm >> 1) & 0xF
            return (bit_12 << 31) | (bits_10_5 << 25) | (rs2 << 20) | (rs1 << 15) | \
                   (funct3 << 12) | (bits_4_1 << 8) | (bit_11 << 7) | 0x63

        program = [
            _i(0, 0, 0b000, 10),             # ADDI x10, x0, 0
            _i(1, 0, 0b000, 11),             # ADDI x11, x0, 1
            _i(10, 0, 0b000, 12),            # ADDI x12, x0, 10
            _i(0, 0, 0b000, 13),             # ADDI x13, x0, 0
            _b(24, 12, 13, 0b000),           # BEQ x13, x12, +24
            _r(0, 11, 10, 0b000, 14),        # ADD x14, x10, x11
            _r(0, 0, 11, 0b000, 10),         # ADD x10, x11, x0
            _r(0, 0, 14, 0b000, 11),         # ADD x11, x14, x0
            _i(1, 13, 0b000, 13),            # ADDI x13, x13, 1
            _j((-20) & 0x1FFFFF, 0),         # JAL x0, -20
            _i(0, 0, 0b000, 0, OP_SYSTEM),   # ECALL
        ]
        _write_program(cpu, program)
        cpu.run()

        assert cpu.halted is True
        assert cpu.registers.read(10) == 55
        # 4 setup + 10*(1 BEQ + 4 body + 1 JAL) + 1 final BEQ + 1 ECALL = 4 + 60 + 2 = 66
        assert cpu.cycle_count == 66
