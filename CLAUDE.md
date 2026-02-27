# riscv-npu

RISC-V (RV32IM) emulator in Python with custom NPU instructions for neural network inference.

## Commands
- uv run pytest                                    — run all tests
- uv run pytest tests/cpu/ -v                      — CPU tests only
- uv run pytest -x                                 — stop on first failure
- uv run python -m riscv_npu run <elf>             — run program headless
- uv run python -m riscv_npu debug <elf>           — run with TUI debugger
- cd firmware/<name> && make                — cross-compile firmware
- uv sync                                      — install/update dependencies
- uv add <pkg>                                 — add new dependency

## Session start
1. Read this file
2. Read .ai/memory.md
3. Read the active phase spec in .ai/specs/

## Session end
1. uv run pytest — confirm passing
2. Commit working changes (atomic, descriptive messages)
3. Update .ai/memory.md: what you did, what works, what's blocked

## Architecture
- src/riscv_npu/cpu/      — decode + execute (the core loop)
- src/riscv_npu/memory/   — bus, RAM, device base class
- src/riscv_npu/devices/  — UART, NPU (memory-mapped I/O)
- src/riscv_npu/loader/   — ELF parser
- src/riscv_npu/syscall/  — ecall dispatch
- src/riscv_npu/npu/      — custom NPU instruction execution + compute engine
- src/riscv_npu/tui/      — Rich-based terminal debugger
- firmware/               — C code that runs ON the emulator (cross-compiled)
- tools/                  — weight export, assembler utilities

## Conventions
- Python 3.14+, type hints on all signatures
- Dataclasses for structured data, functions for stateless ops
- snake_case everywhere
- Docstrings on all public functions
- One test file per module, pytest, descriptive names
- No external deps without logging rationale in .ai/memory.md. Add via uv add <pkg>.

## Testing rules
- One test minimum per instruction
- Run uv run pytest after every few functions, not just end of session
- Never modify tests to make them pass — fix the implementation
- Never skip or disable tests

## Git rules
- main is always passing
- Branch per phase: phase1-rv32i, phase2-elf, etc.
- Atomic commits after each working milestone
- Descriptive messages: "Implement R-type arithmetic (ADD, SUB, SLL...)" not "update file"

## 32-bit masking (critical)
Python ints are arbitrary precision. Mask to 32 bits (& 0xFFFFFFFF) after every arithmetic op.
Signed interpretation: if val >= 0x80000000, val is negative (val - 0x100000000).
This is the #1 source of subtle bugs. Be vigilant.

## Firmware compilation
All firmware MUST be compiled with: -march=rv32im -mabi=ilp32
This emulator does NOT support the A (atomics) or C (compressed) extensions. The C extension is especially dangerous — gcc enables it by default for rv32imac targets, producing 16-bit instructions that the decoder can't handle. Always pass -march=rv32im explicitly.
