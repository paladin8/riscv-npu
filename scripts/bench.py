#!/usr/bin/env python3
"""Emulator performance profiler.

Runs micro-benchmarks of hot-path components and firmware workloads,
reporting throughput and instruction mix.

Usage:
    uv run python scripts/bench.py                 # all workloads
    uv run python scripts/bench.py fibonacci       # single workload
    uv run python scripts/bench.py --cprofile fib  # cProfile dump
    uv run python scripts/bench.py --micro-only    # just micro-benchmarks
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path
from typing import Any

# Add project to path so we can import without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from riscv_npu.cpu.cpu import CPU
from riscv_npu.devices.uart import UART, UART_BASE, UART_SIZE
from riscv_npu.loader.elf import load_elf, parse_elf
from riscv_npu.memory.bus import MemoryBus
from riscv_npu.memory.ram import RAM
from riscv_npu.syscall.handler import SyscallHandler

BASE = 0x8000_0000
RAM_SIZE = 4 * 1024 * 1024
STACK_TOP = BASE + RAM_SIZE - 16

FIRMWARE_DIR = Path(__file__).resolve().parent.parent / "firmware"

# Workloads: (name, elf_path, max_cycles, description)
WORKLOADS: list[tuple[str, Path, int, str]] = [
    ("fibonacci", FIRMWARE_DIR / "fibonacci" / "fibonacci.elf", 500_000,
     "Integer loop (ADDI/BEQ/JAL heavy)"),
    ("sort", FIRMWARE_DIR / "sort" / "sort.elf", 500_000,
     "Memory-bound (LW/SW heavy)"),
    ("newton", FIRMWARE_DIR / "newton" / "newton.elf", 500_000,
     "FP arithmetic (FMUL.S/FADD.S)"),
    ("fpu_test", FIRMWARE_DIR / "fpu_test" / "fpu_test.elf", 500_000,
     "FP instruction mix"),
    ("npu_test", FIRMWARE_DIR / "npu_test" / "npu_test.elf", 500_000,
     "NPU instruction mix"),
    ("mnist", FIRMWARE_DIR / "mnist" / "mnist.elf", 5_000_000,
     "Neural network inference (NPU-heavy)"),
    ("transformer", FIRMWARE_DIR / "transformer" / "transformer.elf", 5_000_000,
     "Transformer forward pass (FP NPU)"),
]


def setup_cpu(elf_path: Path) -> CPU:
    """Create a CPU with bus/ram/uart and load an ELF file."""
    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    uart = UART(tx_stream=io.BytesIO())  # suppress stdout
    bus.register(BASE, RAM_SIZE, ram)
    bus.register(UART_BASE, UART_SIZE, uart)

    cpu = CPU(bus)
    handler = SyscallHandler(stdout=io.BytesIO())  # suppress syscall output
    cpu.syscall_handler = handler

    entry = load_elf(str(elf_path), ram)
    cpu.pc = entry
    cpu.registers.write(2, STACK_TOP)

    with open(elf_path, "rb") as f:
        elf_data = f.read()
    prog = parse_elf(elf_data)
    if prog.segments:
        end = max(s.vaddr + s.memsz for s in prog.segments)
        handler.brk = (end + 15) & ~15

    return cpu


def run_timed(cpu: CPU, max_cycles: int) -> dict[str, Any]:
    """Run the CPU and return timing + stats."""
    start = time.perf_counter()
    cpu.run(max_cycles=max_cycles)
    elapsed = time.perf_counter() - start

    cycles = cpu.cycle_count
    ips = cycles / elapsed if elapsed > 0 else 0

    return {
        "cycles": cycles,
        "elapsed": elapsed,
        "ips": ips,
        "halted": cpu.halted,
        "stats": dict(cpu.instruction_stats),
    }


def run_cprofile(cpu: CPU, max_cycles: int) -> pstats.Stats:
    """Run under cProfile and return stats."""
    pr = cProfile.Profile()
    pr.enable()
    cpu.run(max_cycles=max_cycles)
    pr.disable()
    return pstats.Stats(pr)


# ---------------------------------------------------------------------------
# Hot-path micro-benchmarks (isolated from firmware)
# ---------------------------------------------------------------------------

MICRO_N = 500_000


def bench_bus_read32(n: int = MICRO_N) -> dict[str, Any]:
    """Benchmark bus.read32() in isolation (the instruction fetch path)."""
    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    bus.register(BASE, RAM_SIZE, ram)
    bus.register(UART_BASE, UART_SIZE, UART(tx_stream=io.BytesIO()))

    ram.write32(BASE, 0xDEADBEEF)
    addr = BASE

    start = time.perf_counter()
    for _ in range(n):
        bus.read32(addr)
    elapsed = time.perf_counter() - start
    return {"ops": n, "elapsed": elapsed, "ops_per_sec": n / elapsed}


def bench_bus_write32(n: int = MICRO_N) -> dict[str, Any]:
    """Benchmark bus.write32() in isolation."""
    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    bus.register(BASE, RAM_SIZE, ram)
    bus.register(UART_BASE, UART_SIZE, UART(tx_stream=io.BytesIO()))

    addr = BASE
    start = time.perf_counter()
    for _ in range(n):
        bus.write32(addr, 0xCAFEBABE)
    elapsed = time.perf_counter() - start
    return {"ops": n, "elapsed": elapsed, "ops_per_sec": n / elapsed}


def bench_decode(n: int = MICRO_N) -> dict[str, Any]:
    """Benchmark instruction decode + Instruction allocation."""
    from riscv_npu.cpu.decode import decode

    # Mix of common instruction words
    words = [
        0x00A00093,  # ADDI x1, x0, 10
        0x002081B3,  # ADD x3, x1, x2
        0x0020A023,  # SW x2, 0(x1)
        0xFE209EE3,  # BNE x1, x2, -4
        0x008000EF,  # JAL x1, 8
    ]
    nw = len(words)
    start = time.perf_counter()
    for i in range(n):
        decode(words[i % nw])
    elapsed = time.perf_counter() - start
    return {"ops": n, "elapsed": elapsed, "ops_per_sec": n / elapsed}


def bench_find_device(n: int = MICRO_N) -> dict[str, Any]:
    """Benchmark _find_device() lookup (cache-hit path)."""
    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    bus.register(BASE, RAM_SIZE, ram)
    bus.register(UART_BASE, UART_SIZE, UART(tx_stream=io.BytesIO()))

    addr = BASE
    start = time.perf_counter()
    for _ in range(n):
        bus._find_device(addr, 4)
    elapsed = time.perf_counter() - start
    return {"ops": n, "elapsed": elapsed, "ops_per_sec": n / elapsed}


def bench_step(n: int = 100_000) -> dict[str, Any]:
    """Benchmark a full cpu.step() cycle on a tight loop.

    Loads a 3-instruction loop: ADDI x1, x1, 1 / BNE x1, x2, -4
    so the hot path (fetch, decode, execute) is exercised end-to-end.
    """
    bus = MemoryBus()
    ram = RAM(BASE, RAM_SIZE)
    bus.register(BASE, RAM_SIZE, ram)
    bus.register(UART_BASE, UART_SIZE, UART(tx_stream=io.BytesIO()))
    cpu = CPU(bus)

    # li x2, n  (LUI + ADDI)
    # li x1, 0
    # loop: addi x1, x1, 1
    #        bne  x1, x2, loop
    #        ecall (halt)
    ram.write32(BASE + 0, 0x00000137 | ((n >> 12) << 12))        # LUI x2, n>>12
    ram.write32(BASE + 4, 0x00010113 | ((n & 0xFFF) << 20))      # ADDI x2, x2, n&0xFFF
    ram.write32(BASE + 8, 0x00000093)                              # ADDI x1, x0, 0
    ram.write32(BASE + 12, 0x00108093)                             # ADDI x1, x1, 1
    ram.write32(BASE + 16, 0xFE209EE3)                             # BNE x1, x2, -4
    ram.write32(BASE + 20, 0x00000073)                             # ECALL

    cpu.pc = BASE

    # Provide a syscall handler that halts on ecall
    handler = SyscallHandler()
    cpu.syscall_handler = handler

    start = time.perf_counter()
    cpu.run(max_cycles=n * 2 + 10)
    elapsed = time.perf_counter() - start
    cycles = cpu.cycle_count
    return {"ops": cycles, "elapsed": elapsed, "ops_per_sec": cycles / elapsed}


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def fmt_rate(ips: float) -> str:
    """Format instructions/operations per second."""
    if ips >= 1_000_000:
        return f"{ips / 1_000_000:.2f}M"
    if ips >= 1_000:
        return f"{ips / 1_000:.1f}K"
    return f"{ips:.0f}"


def print_workload_result(name: str, desc: str, result: dict[str, Any]) -> None:
    """Print results for a firmware workload."""
    status = "halted" if result["halted"] else f"hit {result['cycles']:,} cycle limit"
    print(f"  {name:<14} {result['elapsed']:7.3f}s  "
          f"{fmt_rate(result['ips']):>8}/s  "
          f"{result['cycles']:>10,} cycles  ({status})")

    # Top 5 instructions
    stats = result["stats"]
    total = sum(stats.values())
    if total == 0:
        return
    top5 = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:5]
    parts = []
    for mnemonic, count in top5:
        pct = count / total * 100
        parts.append(f"{mnemonic} {pct:.0f}%")
    print(f"  {'':14} mix: {', '.join(parts)}")


def print_micro_result(name: str, result: dict[str, Any]) -> None:
    """Print results for a micro-benchmark."""
    print(f"  {name:<20} {result['elapsed']:7.3f}s  "
          f"{fmt_rate(result['ops_per_sec']):>8}/s  "
          f"({result['ops']:,} ops)")


def print_cprofile_report(stats: pstats.Stats, top_n: int = 25) -> None:
    """Print a cProfile report focused on the hot path."""
    stream = io.StringIO()
    stats.stream = stream
    stats.sort_stats("tottime")
    stats.print_stats(top_n)
    print(stream.getvalue())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the RISC-V emulator")
    parser.add_argument("workload", nargs="?", default=None,
                        help="Run a specific workload (substring match)")
    parser.add_argument("--cprofile", action="store_true",
                        help="Run under cProfile and print hot functions")
    parser.add_argument("--micro-only", action="store_true",
                        help="Only run micro-benchmarks (no firmware)")
    parser.add_argument("--no-micro", action="store_true",
                        help="Skip micro-benchmarks")
    args = parser.parse_args()

    # Filter workloads
    if args.workload:
        selected = [(n, p, c, d) for n, p, c, d in WORKLOADS
                     if args.workload.lower() in n.lower()]
        if not selected:
            print(f"No workload matching '{args.workload}'")
            print(f"Available: {', '.join(n for n, *_ in WORKLOADS)}")
            sys.exit(1)
    else:
        selected = WORKLOADS

    # Micro-benchmarks
    if not args.no_micro:
        print("Micro-benchmarks (isolated hot-path components)")
        print("-" * 65)
        print_micro_result("bus.read32", bench_bus_read32())
        print_micro_result("bus.write32", bench_bus_write32())
        print_micro_result("decode", bench_decode())
        print_micro_result("_find_device", bench_find_device())
        print_micro_result("cpu.step (tight loop)", bench_step())
        print()

    if args.micro_only:
        return

    # Check firmware exists
    missing = [(n, p) for n, p, _, _ in selected if not p.exists()]
    if missing:
        for name, path in missing:
            print(f"  (skipping {name}: ELF not found)", file=sys.stderr)
        selected = [(n, p, c, d) for n, p, c, d in selected if p.exists()]

    if not selected:
        print("No firmware found. Compile with: cd firmware/<name> && make")
        sys.exit(1)

    # Firmware workloads
    print("Firmware workloads")
    print("-" * 65)

    for name, elf_path, max_cycles, desc in selected:
        if args.cprofile:
            print(f"\ncProfile: {name} ({desc})")
            print("=" * 65)
            cpu = setup_cpu(elf_path)
            ps = run_cprofile(cpu, max_cycles)
            print_cprofile_report(ps)
        else:
            cpu = setup_cpu(elf_path)
            result = run_timed(cpu, max_cycles)
            print_workload_result(name, desc, result)
    print()


if __name__ == "__main__":
    main()
