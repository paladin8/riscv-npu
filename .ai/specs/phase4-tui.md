# Phase 4: TUI Debugger

## Goal
Interactive Rich-based TUI for stepping through RISC-V program execution.

## Design Decisions

1. **Rich Live display, not Textual**: The project already depends on Rich. Textual would add a large new dependency for a relatively simple debugger interface. Rich's `Live` context with `Layout` and `Panel` provides sufficient capability for a stepping debugger. The TUI is not an interactive application with widgets -- it's a display that refreshes on each debugger command.

2. **Separation of concerns**: Pure formatting functions (register display, hex dump, disassembly text) are separated from the TUI app. This allows thorough unit testing of the formatting logic without needing a terminal. The `DebuggerState` dataclass holds the debugging context (CPU, breakpoints, previous register values). The `app.py` module orchestrates Rich rendering and keyboard input.

3. **Disassembly from raw memory**: The disassembly panel reads instruction words directly from memory at PC +/- offsets and decodes them using the existing `decode()` function. A `disassemble_instruction()` function produces human-readable mnemonics from decoded `Instruction` objects. No separate disassembly file or ELF symbol table is needed for basic operation.

4. **UART output capture**: The TUI needs to capture UART TX output for display. The UART already accepts an injectable `tx_stream`. We pass an `io.BytesIO` buffer when running in debug mode, and display its contents in the Output panel.

5. **Input model**: The debugger uses Python's `input()` for command entry (blocking prompt at the bottom). Rich `Live` is updated after each command. This avoids the complexity of raw terminal key handling while keeping the interface responsive. Commands: `s`/`step`, `c`/`continue`, `b <addr>`/`breakpoint`, `g <addr>`/`goto`, `q`/`quit`.

6. **Register change tracking**: The debugger stores a snapshot of all 32 registers after each step. When rendering, registers whose values differ from the previous snapshot are highlighted (Rich markup `[bold yellow]`).

## Panels

- **Registers**: x0-x31 with ABI names + hex values, highlight changes since last step
- **Disassembly**: current PC +/- 10 instructions, current line highlighted
- **Memory**: hex dump at configurable address, 16 bytes per row, 16 rows
- **Output**: UART TX capture (last N lines)
- **Status bar**: PC, instruction count, run/pause state

## Controls

- `s` / `step`: execute one instruction
- `c` / `continue`: run until breakpoint, halt, or 10000 cycles
- `b <addr>`: toggle breakpoint at address (hex, e.g. `b 80000010`)
- `g <addr>`: move memory inspector to address
- `q` / `quit`: exit debugger
- `r` / `run`: alias for continue

## Run Modes

- `uv run python -m riscv_npu run <elf>` -- headless (existing)
- `uv run python -m riscv_npu debug <elf>` -- TUI debugger (new)

## Deliverables (ordered by dependency)

### Deliverable 1: Instruction Disassembler
Pure function that converts an `Instruction` dataclass into a human-readable mnemonic string.

**Files to create/modify:**
- `src/riscv_npu/tui/disasm.py` -- implement `disassemble_instruction(inst: Instruction) -> str` and `disassemble_region(memory: MemoryBus, center_pc: int, count: int) -> list[DisassemblyLine]`
- `tests/tui/test_disasm.py` -- unit tests

**Key functions:**
- `disassemble_instruction(inst: Instruction) -> str`: Returns mnemonic string like "ADD x1, x2, x3" or "ADDI x5, x0, 42"
- `disassemble_region(memory: MemoryBus, center_pc: int, count: int) -> list[DisassemblyLine]`: Reads `count` instructions centered on `center_pc` from memory, returns list of `DisassemblyLine(addr, word, text, is_current)`.

**Data structures:**
- `DisassemblyLine` dataclass: `addr: int, word: int, text: str, is_current: bool`

**Test coverage:**
- R-type instructions (ADD, SUB, AND, OR, XOR, SLL, SRL, SRA, SLT, SLTU)
- M-extension instructions (MUL, DIV, REM, etc.)
- I-type arithmetic (ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI)
- Load instructions (LB, LH, LW, LBU, LHU)
- Store instructions (SB, SH, SW)
- Branch instructions (BEQ, BNE, BLT, BGE, BLTU, BGEU)
- U-type (LUI, AUIPC)
- J-type (JAL, JALR)
- System (ECALL, EBREAK, MRET)
- CSR instructions (CSRRW, CSRRS, CSRRC)
- `disassemble_region` with a small memory region

### Deliverable 2: Register Formatting
Pure function that formats all 32 registers for display with ABI names and change highlighting.

**Files to create/modify:**
- `src/riscv_npu/tui/registers.py` -- implement `format_registers()`
- `tests/tui/test_registers.py` -- unit tests

**Key functions:**
- `format_registers(regs: RegisterFile, prev_values: list[int] | None) -> str`: Returns Rich-markup string showing all 32 registers in a 4-column layout. Registers with changed values are highlighted with `[bold yellow]`.

**Constants:**
- `ABI_NAMES: list[str]` -- 32-element list: ["zero", "ra", "sp", "gp", "tp", "t0", ..., "t6"]

**Test coverage:**
- All 32 registers displayed with correct ABI names
- x0 always shows 0
- Changed registers are highlighted with Rich markup
- No changes (prev_values == current) shows no highlighting
- None prev_values (first display) shows no highlighting

### Deliverable 3: Memory Hex Dump Formatting
Pure function that formats a memory region as a hex dump.

**Files to create/modify:**
- `src/riscv_npu/tui/memory.py` -- implement `format_hex_dump()`
- `tests/tui/test_memory.py` -- unit tests

**Key functions:**
- `format_hex_dump(memory: MemoryBus, start_addr: int, num_rows: int) -> str`: Returns a hex dump string with 16 bytes per row, showing address, hex bytes, and ASCII representation. Uses Rich markup for formatting.

**Test coverage:**
- Correct hex byte formatting
- Correct ASCII display (printable characters shown, non-printable as '.')
- Address alignment
- Multiple rows
- Handles unmapped addresses gracefully (shows '??' for unmapped bytes)

### Deliverable 4: DebuggerState and Controller
The debugger state management and command processing logic.

**Files to create/modify:**
- `src/riscv_npu/tui/debugger.py` -- implement `DebuggerState` and command processing
- `tests/tui/test_debugger.py` -- unit tests

**Key classes/functions:**
- `DebuggerState` dataclass:
  - `cpu: CPU`
  - `breakpoints: set[int]`
  - `prev_regs: list[int]` (snapshot of registers before last step)
  - `mem_view_addr: int` (address for memory hex dump panel)
  - `uart_capture: io.BytesIO` (UART TX output buffer)
  - `running: bool` (True = continue mode, False = paused)
  - `message: str` (status message for last command)

- `debugger_step(state: DebuggerState) -> None`: Execute one instruction, update prev_regs snapshot.
- `debugger_continue(state: DebuggerState, max_cycles: int) -> None`: Run until breakpoint, halt, or max_cycles.
- `process_command(state: DebuggerState, cmd: str) -> bool`: Parse and execute a debugger command. Returns False if the user wants to quit.

**Test coverage:**
- `debugger_step` advances PC and updates prev_regs
- `debugger_continue` stops at breakpoint
- `debugger_continue` stops at halt (EBREAK)
- `debugger_continue` stops at max_cycles limit
- `process_command` handles step, continue, breakpoint toggle, goto, quit
- Breakpoint toggle (add then remove)
- Invalid command handled gracefully

### Deliverable 5: TUI Layout and Rendering
Rich Layout composition that combines all panels into a full-screen display.

**Files to create/modify:**
- `src/riscv_npu/tui/app.py` -- implement `render_debugger()` and `run_debugger()`
- `tests/tui/test_app.py` -- basic smoke tests

**Key functions:**
- `render_debugger(state: DebuggerState) -> Layout`: Build the Rich Layout with all panels (Registers, Disassembly, Memory, Output, Status bar).
- `run_debugger(elf_path: str) -> None`: Main entry point. Loads the ELF, creates CPU+bus+UART, creates DebuggerState, runs the command loop with Rich Live display.

**Layout structure:**
```
+------------------+------------------+
|   Registers      |   Disassembly    |
|   (left-top)     |   (right-top)    |
+------------------+------------------+
|   Memory         |   Output (UART)  |
|   (left-bottom)  |   (right-bottom) |
+------------------+------------------+
|   Status bar (full width)           |
+---------+---------------------------+
```

**Test coverage:**
- `render_debugger` returns a Layout object (smoke test)
- Layout contains expected panel titles

### Deliverable 6: CLI Integration
Wire the `debug` subcommand in cli.py to launch the TUI.

**Files to create/modify:**
- `src/riscv_npu/cli.py` -- update `main()` to handle `debug` command
- `src/riscv_npu/tui/__init__.py` -- export `run_debugger`

**Test coverage:**
- CLI argument parsing includes `debug` subcommand (integration test, can be a simple smoke test)

## Testing

Unit test formatting functions (register display, hex dump, disassembly text). The debugger controller is tested with a CPU running small instruction sequences. Layout rendering is smoke-tested. Full TUI interaction requires manual verification.

## Acceptance Criteria

1. `uv run pytest` passes with all existing tests (303) plus new TUI tests
2. `uv run python -m riscv_npu debug firmware/fibonacci/fibonacci.elf` opens the debugger
3. Step command (`s`) executes one instruction, registers update visibly
4. Continue command (`c`) runs until halt
5. Breakpoint command (`b <addr>`) stops execution at the specified address
6. Memory view (`g <addr>`) updates the hex dump panel
7. Quit command (`q`) cleanly exits
8. UART output appears in the Output panel
9. Register changes are highlighted
10. Disassembly shows instructions around the current PC
