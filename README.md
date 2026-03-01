# riscv-npu

RISC-V (RV32IM) emulator in Python with custom NPU instructions for neural network inference.

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- RISC-V cross-compiler (`riscv64-unknown-elf-gcc`) for building firmware

## Setup

```bash
uv sync
```

## Running programs

Run an ELF binary headless:

```bash
uv run python -m riscv_npu run firmware/fibonacci/fibonacci.elf
```

The emulator prints the cycle count and the values in registers `a0`/`a1` on exit. The process exit code matches the program's return value (`a0`).

## TUI debugger

Launch the interactive debugger:

```bash
uv run python -m riscv_npu debug firmware/uart-hello/uart-hello.elf
```

The debugger displays four panels: registers, disassembly, memory hex dump, and UART output. Changed registers are highlighted after each step.

### Debugger commands

| Command                        | Description                          |
| ------------------------------ | ------------------------------------ |
| `s` / `step`                   | Execute one instruction              |
| `c` / `r` / `continue` / `run` | Run until breakpoint or halt         |
| `b <hex_addr>`                 | Toggle breakpoint (e.g. `b 80000010`)|
| `g <hex_addr>`                 | Jump memory view to address          |
| `q` / `quit`                   | Exit debugger                        |

## Firmware

Firmware programs are C code that runs **on** the emulated CPU. Each program lives in `firmware/<name>/` with its own Makefile.

### Compiling firmware

```bash
cd firmware/hello && make
```

This requires a RISC-V cross-compiler. All firmware is compiled with `-march=rv32im -mabi=ilp32` — the emulator does **not** support the A (atomics) or C (compressed) extensions.

### Example programs

| Program      | Description                                                 |
| ------------ | ----------------------------------------------------------- |
| `fibonacci`  | Computes fib(10), returns result in `a0`                    |
| `hello`      | Prints "Hello, World!" via write syscall                    |
| `uart-hello` | Prints via direct UART register access (memory-mapped I/O)  |
| `sort`       | Insertion sort, returns 1 on success                        |

## Testing

```bash
uv run pytest              # all tests
uv run pytest tests/cpu/ -v  # CPU tests only
uv run pytest -x           # stop on first failure
```

## Architecture

```
src/riscv_npu/
  cpu/       — decode + execute (the core loop)
  memory/    — bus, RAM, device base class
  devices/   — UART, NPU (memory-mapped I/O)
  loader/    — ELF parser
  syscall/   — ecall dispatch
  npu/       — custom NPU instruction execution + compute engine
  tui/       — Rich-based terminal debugger
firmware/    — C programs that run on the emulator
tools/       — weight export, assembler utilities
```
