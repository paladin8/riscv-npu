# Phase 3: UART + Syscalls

## Goal
Programs print to stdout, read from stdin, and exit with a code. Interactive terminal programs (ANSI escape codes, keyboard input) work.

## What to build
- MemoryBus: routes address ranges to devices. Interface: read8/16/32, write8/16/32. Devices register (base_addr, size, device).
- RAM device: wraps existing memory, mapped at 0x80000000
- UART device: mapped at 0x10000000.
  - TX (write): write byte to address → appears on host stdout
  - RX (read): read byte from address → returns next byte from host stdin (non-blocking, returns 0 if no input available)
  - LSR (Line Status Register) at offset 0x05: bit 0 = data ready (input available), bit 5 = THR empty (can write). This is standard 16550 UART behavior.
- SyscallHandler: on ECALL, read a7 for number:
  - 63 (read): a0=fd, a1=buf_ptr, a2=len → read from stdin (fd=0 only), returns bytes read
  - 64 (write): a0=fd, a1=buf_ptr, a2=len → write to stdout (fd=1 only)
  - 93 (exit): a0=code → halt, report code
  - 214 (brk): a0=addr → bump allocator
- firmware/common/syscalls.c: putchar, getchar, puts, exit wrappers using ecall
- firmware/hello/main.c: "Hello, World!" via write syscall

## UART implementation notes
Host stdin must be read non-blocking so the emulator doesn't stall waiting for input. Use select/poll or platform-appropriate non-blocking IO. Buffer incoming bytes — the UART RX holds one byte at a time, the emulated program polls LSR then reads RX. ANSI escape codes from the emulated program pass through the UART TX to the host terminal unmodified — the host terminal interprets them. This means emulated programs can do cursor movement, colors, screen clearing, etc. with raw escape codes.

## Acceptance
uv run pytest tests/memory/test_bus.py -v — pass
uv run pytest tests/devices/ -v — pass
cd firmware/hello && make
uv run python -m riscv_npu run firmware/hello/hello.elf — prints "Hello, World!"
