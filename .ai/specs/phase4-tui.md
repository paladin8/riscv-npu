# Phase 4: TUI Debugger

## Goal
Interactive Rich/Textual TUI for stepping through execution.

## Panels
- Registers: x0-x31 with ABI names + hex values, highlight changes since last step
- Disassembly: current PC ± 10 instructions, current line highlighted
- Memory: hex dump at configurable address
- Output: UART capture
- Status bar: PC, instruction count, run/pause state

## Controls
- step: execute one instruction
- continue: run until breakpoint or halt
- breakpoint <addr>: toggle breakpoint
- goto <addr>: move memory inspector

## Run modes
- uv run python -m riscv_npu run <elf> — headless
- uv run python -m riscv_npu debug <elf> — TUI

## Testing
Unit test formatting functions (register display, hex dump, disassembly text). Layout is manually verified. Expect human feedback and iteration on this phase.

## Acceptance
uv run python -m riscv_npu debug firmware/fibonacci/fibonacci.elf — opens, steps work, registers update
