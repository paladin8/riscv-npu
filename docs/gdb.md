# GDB Remote Debugging

The emulator includes a GDB remote stub that implements the GDB Remote Serial Protocol (RSP) over TCP. This lets you debug firmware using `gdb-multiarch` or `riscv64-unknown-elf-gdb` with the full GDB feature set: breakpoints, stepping, register and memory inspection, source-level debugging, and disassembly.

## Quick Start

Terminal 1 -- start the emulator:

```bash
uv run python -m riscv_npu gdb firmware/hello/hello.elf
# Listening on :1234, waiting for GDB...
```

Terminal 2 -- connect GDB:

```bash
gdb-multiarch firmware/hello/hello.elf -ex 'target remote :1234'
```

Or as a one-liner that breaks at main and runs to it:

```bash
gdb-multiarch firmware/hello/hello.elf \
  -ex 'target remote :1234' \
  -ex 'break main' \
  -ex 'continue'
```

## Command-Line Options

```
uv run python -m riscv_npu gdb <elf> [--port PORT] [--write SYMBOL:FILE ...]
```

| Option              | Default | Description                                          |
|---------------------|---------|------------------------------------------------------|
| `<elf>`             |         | Path to the ELF firmware binary (required)           |
| `--port PORT`       | 1234    | TCP port the stub listens on                         |
| `--write SYM:FILE`  |         | Load FILE into memory at ELF symbol SYM (repeatable) |

## What Works

### Stepping

| GDB command          | Description                                |
|----------------------|--------------------------------------------|
| `stepi` / `si`       | Execute one instruction                    |
| `continue` / `c`     | Run until breakpoint, halt, or Ctrl+C      |
| Ctrl+C               | Interrupt a running `continue`             |

`stepi` is the primary stepping command since the emulator operates at instruction granularity. `step` and `next` (source-level stepping) also work when firmware is compiled with `-g`, though variables may show as `<optimized out>` at `-O2`.

### Breakpoints

| GDB command          | Description                                |
|----------------------|--------------------------------------------|
| `break main`         | Break at a symbol (requires ELF symbols)   |
| `break main.c:15`    | Break at source line (requires `-g`)       |
| `break *0x80000010`  | Break at a raw address                     |
| `info breakpoints`   | List active breakpoints                    |
| `delete N`           | Remove breakpoint number N                 |

Only software breakpoints are supported. Hardware breakpoints and watchpoints are not implemented.

### Registers

| GDB command          | Description                                |
|----------------------|--------------------------------------------|
| `info registers`     | Show all GPRs (x0-x31) and pc             |
| `info float`         | Show FPU registers (f0-f31, fflags, frm, fcsr) |
| `print $a0`          | Print a single register by ABI name        |
| `print $f0`          | Print a float register                     |
| `print $pc`          | Print the program counter                  |
| `set $a0 = 42`       | Modify a register                          |

ABI names (`ra`, `sp`, `a0`-`a7`, `s0`-`s11`, `t0`-`t6`, `ft0`-`ft11`, `fa0`-`fa7`, `fs0`-`fs11`) are supported via the target description XML served to GDB on connection.

### Memory

| GDB command          | Description                                |
|----------------------|--------------------------------------------|
| `x/16xw 0x80000000`  | 16 words in hex starting at address        |
| `x/8xb $sp`          | 8 bytes in hex at stack pointer            |
| `x/10i $pc`          | Disassemble 10 instructions from PC        |
| `x/s 0x80000100`     | Print a null-terminated string             |
| `set {int}0x80000100 = 0xff` | Write to memory                    |

### Source-Level Debugging

When firmware is compiled with `-g` (enabled by default in `firmware/common/Makefile`):

| GDB command          | Description                                |
|----------------------|--------------------------------------------|
| `list`               | Show source around current PC              |
| `list main.c:1`      | Show source at a specific location         |
| `break main.c:15`    | Break at a source line                     |
| `print my_variable`  | Inspect a local variable                   |
| `info locals`        | Show all local variables                   |
| `backtrace` / `bt`   | Show call stack                            |

Since firmware is compiled with `-O2 -g`, the optimizer may inline functions, reorder code, or eliminate variables. For full debuggability of a specific program, recompile with `-O0`:

```bash
cd firmware/hello && make clean && make CFLAGS="-march=rv32imf -mabi=ilp32f -O0 -g -nostdlib -ffreestanding -Wall"
```

### Session Control

| GDB command          | Description                                |
|----------------------|--------------------------------------------|
| `kill`               | Terminate the debug session                |
| `detach`             | Disconnect from the emulator               |
| `quit`               | Exit GDB                                   |

## What's Not Supported

| Feature                     | Reason                                                |
|-----------------------------|-------------------------------------------------------|
| Hardware watchpoints        | `watch`/`rwatch`/`awatch` require memory trapping     |
| `vCont` protocol            | GDB falls back to `s`/`c` automatically               |
| Reverse debugging           | No execution history recorded                         |
| Multi-threading             | Single-hart emulator, one thread always               |
| Non-stop mode               | Single-threaded, all-stop only                        |
| `run` / `start`             | Target is pre-loaded; use `continue` after connecting |
| File I/O (`vFile`)          | No host filesystem access from firmware               |

## Register Layout

The stub exposes 68 registers to GDB via target description XML:

| Index | Register       | Group | Type           |
|-------|----------------|-------|----------------|
| 0-31  | x0-x31 (GPR)   | CPU   | int / ptr      |
| 32    | pc             | CPU   | code_ptr       |
| 33-64 | f0-f31 (FPR)   | FPU   | ieee_single    |
| 65    | fflags         | FPU   | int            |
| 66    | frm            | FPU   | int            |
| 67    | fcsr           | FPU   | int            |

GPR and PC are returned in the bulk register read (`g` packet). FPU registers are read individually as needed.

## Protocol Details

The stub implements the following RSP packets:

| Packet                                | Description                     |
|---------------------------------------|---------------------------------|
| `?`                                   | Halt reason (returns `T05`)     |
| `g` / `G`                             | Read / write all GPRs + PC      |
| `p` / `P`                             | Read / write single register    |
| `m` / `M`                             | Read / write memory             |
| `s`                                   | Single step                     |
| `c`                                   | Continue                        |
| `Z0` / `z0`                           | Insert / remove breakpoint      |
| `k`                                   | Kill                            |
| `D`                                   | Detach                          |
| `qSupported`                          | Feature negotiation             |
| `QStartNoAckMode`                     | Disable packet acknowledgement  |
| `qXfer:features:read:target.xml`      | Target description XML          |
| `qfThreadInfo` / `qsThreadInfo`       | Thread enumeration              |
| `qAttached` / `qC` / `Hg` / `Hc`      | Thread and attach queries       |

Unsupported packets receive an empty response, which tells GDB the feature is not available. No external dependencies are used -- the stub is built on Python's stdlib `socket` and `select` modules.

## Troubleshooting

**GDB says "Remote connection closed"** -- the emulator process may have crashed. Check the terminal running the emulator for error messages.

**Variables show `<optimized out>`** -- firmware is compiled with `-O2`. Recompile with `-O0 -g` for full variable visibility.

**`break main` fails with "Function not found"** -- the ELF may lack debug symbols. Verify with `riscv64-unknown-elf-objdump -h firmware.elf | grep debug`.

**Slow `continue` performance** -- the stub checks for Ctrl+C interrupts every 1024 instruction steps. This adds negligible overhead compared to the Python emulation loop itself.
