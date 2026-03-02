# Phase 8: Polish

## Goal
Project is presentable and documented.

## Deliverables
- README: architecture, screenshots, usage, build instructions
- docs/isa-reference.md: complete quick reference (should already exist, verify/update)
- TUI: instruction statistics
- Code cleanup: consistent formatting, complete docstrings, no dead code
- Performance profile: where is time spent, what would hardware accelerate

## Design Decisions

1. **Instruction statistics tracking**: Add an `InstructionStats` dataclass to the CPU that counts instructions by mnemonic (e.g. "ADD": 1234, "NPU.FVMAC": 56). The stats are populated during `CPU.step()` by classifying the decoded instruction via a lightweight mnemonic function in the decode module. This avoids modifying the execute path.

2. **TUI stats panel**: A new `tui/stats.py` module formats the instruction statistics as a Rich panel. The panel shows the top N instructions by count, plus totals for each category (RV32I, M-ext, F-ext, NPU-int, NPU-fp). The panel is added to the TUI layout in a new row below memory/output.

3. **Performance profile**: A standalone `docs/performance.md` document based on actual cProfile runs of `cpu.run()`. Identifies hot paths and recommends what a hardware NPU would accelerate.

4. **README enhancement**: Expand the existing README with a feature summary table, architecture diagram (ASCII), NPU instruction set overview, and links to docs/.

5. **ISA reference verification**: The existing docs/isa-reference.md is comprehensive. Verify column widths are ASCII-aligned per project conventions and that all 75 instructions are documented.

## Deliverables List (ordered, dependency-aware)

1. **D1: Instruction mnemonic classifier** -- `cpu/decode.py::instruction_mnemonic()` function
2. **D2: InstructionStats dataclass and CPU integration** -- `cpu/cpu.py` tracks per-instruction counts
3. **D3: TUI stats panel** -- `tui/stats.py::format_instruction_stats()` + layout integration
4. **D4: Code cleanup** -- docstrings, formatting, dead code removal across all source files
5. **D5: ISA reference update** -- verify/fix docs/isa-reference.md
6. **D6: Performance profile** -- docs/performance.md
7. **D7: README update** -- expanded README.md

## Implementation Details

### D1: Instruction mnemonic classifier
- **File**: `src/riscv_npu/cpu/decode.py`
- **Function**: `instruction_mnemonic(inst: Instruction) -> str`
- Returns a short human-readable name like "ADD", "ADDI", "NPU.FMACC", "flw", etc.
- Uses the same dispatch logic as `disasm.py` but returns only the mnemonic string, no operands
- Must be fast -- called on every instruction step

### D2: InstructionStats and CPU integration
- **File**: `src/riscv_npu/cpu/cpu.py`
- Add `instruction_stats: dict[str, int]` field to CPU (default empty dict)
- In `CPU.step()`, after decode, call `instruction_mnemonic(inst)` and increment the count
- Add `CPU.stats_summary() -> dict[str, int]` method that returns a copy of the stats

### D3: TUI stats panel
- **File**: `src/riscv_npu/tui/stats.py`
- **Function**: `format_instruction_stats(stats: dict[str, int], top_n: int = 15) -> str`
  - Groups instructions into categories: RV32I, M-ext, F-ext, NPU-int, NPU-fp
  - Shows top N instructions by count with percentage
  - Shows category totals
  - Returns Rich-markup string
- **File**: `src/riscv_npu/tui/app.py`
  - Add stats panel to layout, below the bottom row
  - Update `render_debugger()` to include the stats panel

### D4: Code cleanup
- Verify all public functions have docstrings
- Check for unused imports
- Ensure consistent formatting (snake_case, type hints)
- No changes expected -- codebase is already clean from prior phases

### D5: ISA reference update
- Verify all 75 instructions are documented
- ASCII-align all markdown table columns (pipe characters line up)
- Add any missing edge case notes

### D6: Performance profile
- **File**: `docs/performance.md`
- Profile `CPU.run()` on a realistic workload (fibonacci firmware in tests)
- Document hot paths: decode, memory access, execute dispatch
- Identify what NPU hardware would accelerate (VMAC inner loops, VEXP, etc.)
- Recommend potential optimizations (dispatch tables, memory layout)

### D7: README update
- Add feature summary table (ISA support, instruction count, NPU capabilities)
- Add ASCII architecture diagram showing data flow
- Add section on NPU instruction sets (integer + FP, with brief descriptions)
- Link to docs/isa-reference.md and docs/performance.md
- Keep existing content (prerequisites, usage, firmware, testing, debugger)

## Test Coverage Requirements

### D1: instruction_mnemonic tests
- **File**: `tests/cpu/test_decode.py` (add to existing)
- `test_instruction_mnemonic_r_type` -- ADD, SUB, SLL, etc.
- `test_instruction_mnemonic_i_type` -- ADDI, SLTI, SLLI, SRAI, etc.
- `test_instruction_mnemonic_load_store` -- LW, LB, SW, SH, etc.
- `test_instruction_mnemonic_branch` -- BEQ, BNE, BLT, etc.
- `test_instruction_mnemonic_upper_jump` -- LUI, AUIPC, JAL, JALR
- `test_instruction_mnemonic_system` -- ECALL, EBREAK, MRET, CSRRW
- `test_instruction_mnemonic_m_ext` -- MUL, DIV, REM, etc.
- `test_instruction_mnemonic_fpu` -- fadd.s, fsub.s, flw, fsw, fmadd.s, etc.
- `test_instruction_mnemonic_npu` -- NPU.MACC, NPU.RELU, NPU.VMAC, etc.
- `test_instruction_mnemonic_fp_npu` -- NPU.FMACC, NPU.FRELU, NPU.FVMAC, etc.

### D2: CPU stats integration tests
- **File**: `tests/cpu/test_cpu.py` (add to existing)
- `test_instruction_stats_counts_each_step` -- run a few instructions, verify stats dict
- `test_instruction_stats_empty_initially` -- fresh CPU has empty stats

### D3: TUI stats panel tests
- **File**: `tests/tui/test_stats.py` (new)
- `test_format_instruction_stats_empty` -- empty dict returns reasonable output
- `test_format_instruction_stats_top_n` -- only shows top N entries
- `test_format_instruction_stats_categories` -- verifies category grouping
- `test_format_instruction_stats_percentages` -- percentages sum correctly

## Acceptance Criteria

1. `uv run pytest -v` -- all tests pass, test count > 820
2. `uv run python -c "from riscv_npu.cpu.decode import instruction_mnemonic; print('OK')"` -- imports cleanly
3. README.md contains architecture diagram, NPU instruction overview, and links to docs
4. docs/isa-reference.md has ASCII-aligned tables for all instruction categories
5. docs/performance.md exists with profiling data and hardware acceleration analysis
6. TUI stats panel renders instruction statistics when debugging
7. No source files with missing docstrings on public functions
