# riscv-npu

RISC-V (RV32IMF) emulator in Python with custom NPU instructions for neural network inference.

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
uv run python -m riscv_npu debug firmware/hello/hello.elf
```

The debugger displays panels for registers, disassembly, memory hex dump, UART output, FPU state, and NPU state (integer + float accumulators, vector registers). Changed values are highlighted after each step.

Use `--write SYMBOL:FILE` to load file contents into memory at an ELF symbol's address before execution. This is useful for injecting test data (e.g. weight files or input images) into firmware buffers.

### Debugger commands

| Command                | Description                                   |
|------------------------|-----------------------------------------------|
| `s` / `step`           | Execute one instruction                       |
| `c` / `continue`       | Run until breakpoint or halt                  |
| `r` / `run <hz> [max]` | Run at fixed speed with live display (Ctrl+C) |
| `b <hex_addr>`         | Toggle breakpoint (e.g. `b 80000010`)         |
| `g <hex_addr>`         | Jump memory view to address                   |
| `h` / `help`           | Show command help                             |
| `q` / `quit`           | Exit debugger                                 |

## Firmware

Firmware programs are C code that runs **on** the emulated CPU. Each program lives in `firmware/<name>/` with its own Makefile.

### Compiling firmware

```bash
cd firmware/hello && make
```

This requires a RISC-V cross-compiler. All firmware is compiled with `-march=rv32imf -mabi=ilp32f` — the emulator does **not** support the A (atomics) or C (compressed) extensions.

### Example programs

| Program       | Description                                                      |
|---------------|------------------------------------------------------------------|
| `fibonacci`   | Computes fib(10), returns result in `a0`                         |
| `hello`       | Prints "Hello, World!" via write syscall                         |
| `uart-hello`  | Prints via direct UART register access (memory-mapped I/O)       |
| `sort`        | Insertion sort, returns 1 on success                             |
| `newton`      | Square roots via Newton's method, verified against fsqrt         |
| `fpu_test`    | Tests all RV32F floating-point instructions (34 checks)          |
| `npu_test`    | Exercises all NPU instructions (MACC, RELU, QMUL, CLAMP, GELU)   |
| `mnist`       | Quantized 784->128->10 MLP, classifies handwritten digits            |
| `transformer` | Float32 char-level transformer LM, trained on Shakespeare (FP NPU)   |

### Generating weights

The `mnist` and `transformer` firmware require exported weight files before compiling.

```bash
# MNIST MLP weights (~100KB, int8 quantized)
uv run --extra torch python -m riscv_npu.tools.export_mnist_weights

# Transformer weights (~527KB, float32, trained on TinyShakespeare)
uv run --extra torch python -m riscv_npu.tools.export_transformer_weights
```

Then compile and run:

```bash
cd firmware/mnist && make
uv run python -m riscv_npu run firmware/mnist/mnist.elf

cd firmware/transformer && make
uv run python -m riscv_npu run firmware/transformer/transformer.elf
```

The transformer generates text autoregressively — it echoes the input prompt, then prints generated characters on a second line prefixed with `>`.

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
  tools/     — weight export, assembler utilities
  tui/       — Rich-based terminal debugger
firmware/    — C programs that run on the emulator
```
