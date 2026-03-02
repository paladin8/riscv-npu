"""TUI instruction statistics panel: formats per-instruction execution counts."""

from __future__ import annotations


# Instruction category definitions for grouping.
# Each set contains the mnemonics belonging to that category.
_RV32I_MNEMONICS: set[str] = {
    "ADD", "SUB", "SLL", "SLT", "SLTU", "XOR", "SRL", "SRA", "OR", "AND",
    "ADDI", "SLTI", "SLTIU", "XORI", "ORI", "ANDI", "SLLI", "SRLI", "SRAI",
    "LB", "LH", "LW", "LBU", "LHU", "SB", "SH", "SW",
    "BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU",
    "LUI", "AUIPC", "JAL", "JALR",
    "ECALL", "EBREAK", "MRET", "FENCE",
    "CSRRW", "CSRRS", "CSRRC", "CSRRWI", "CSRRSI", "CSRRCI",
}

_M_EXT_MNEMONICS: set[str] = {
    "MUL", "MULH", "MULHSU", "MULHU", "DIV", "DIVU", "REM", "REMU",
}

_F_EXT_MNEMONICS: set[str] = {
    "FADD.S", "FSUB.S", "FMUL.S", "FDIV.S", "FSQRT.S",
    "FMADD.S", "FMSUB.S", "FNMSUB.S", "FNMADD.S",
    "FSGNJ.S", "FSGNJN.S", "FSGNJX.S",
    "FMIN.S", "FMAX.S",
    "FEQ.S", "FLT.S", "FLE.S",
    "FCVT.W.S", "FCVT.WU.S", "FCVT.S.W", "FCVT.S.WU",
    "FMV.X.W", "FMV.W.X", "FCLASS.S",
    "FLW", "FSW",
}

_NPU_INT_MNEMONICS: set[str] = {
    "NPU.MACC", "NPU.VMAC", "NPU.VEXP", "NPU.VRSQRT", "NPU.VMUL",
    "NPU.VREDUCE", "NPU.VMAX",
    "NPU.RELU", "NPU.QMUL", "NPU.CLAMP", "NPU.GELU", "NPU.RSTACC",
    "NPU.LDVEC", "NPU.STVEC",
}

_NPU_FP_MNEMONICS: set[str] = {
    "NPU.FMACC", "NPU.FVMAC", "NPU.FVEXP", "NPU.FVRSQRT", "NPU.FVMUL",
    "NPU.FVREDUCE", "NPU.FVMAX",
    "NPU.FRELU", "NPU.FGELU", "NPU.FRSTACC",
}

# Category definitions: (label, mnemonic set)
_CATEGORIES: list[tuple[str, set[str]]] = [
    ("RV32I", _RV32I_MNEMONICS),
    ("M-ext", _M_EXT_MNEMONICS),
    ("F-ext", _F_EXT_MNEMONICS),
    ("NPU-int", _NPU_INT_MNEMONICS),
    ("NPU-fp", _NPU_FP_MNEMONICS),
]


def _categorize(mnemonic: str) -> str:
    """Return the category label for a given mnemonic.

    Args:
        mnemonic: The instruction mnemonic string.

    Returns:
        Category label (e.g. "RV32I", "M-ext"), or "Other" if not found.
    """
    for label, mnemonics in _CATEGORIES:
        if mnemonic in mnemonics:
            return label
    return "Other"


def format_instruction_stats(stats: dict[str, int], top_n: int = 15) -> str:
    """Format instruction execution statistics for display in a Rich panel.

    Shows the top N instructions by count with percentage of total,
    followed by category totals.

    Args:
        stats: Dict mapping instruction mnemonic to execution count.
        top_n: Maximum number of individual instructions to show.

    Returns:
        A multi-line string with Rich markup suitable for display.
    """
    if not stats:
        return "No instructions executed."

    total = sum(stats.values())
    lines: list[str] = []

    # Top N instructions by count
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    shown = sorted_stats[:top_n]

    # Find max mnemonic width for alignment
    max_name_len = max(len(name) for name, _ in shown)
    max_count_len = max(len(str(count)) for _, count in shown)

    lines.append(f"Top {min(top_n, len(sorted_stats))} instructions (total: {total:,})")
    lines.append("")

    # Format entries, then arrange in columns
    entries: list[str] = []
    for name, count in shown:
        pct = count / total * 100
        padded_name = name.ljust(max_name_len)
        padded_count = str(count).rjust(max_count_len)
        entries.append(f"{padded_name} {padded_count} ({pct:5.1f}%)")

    if len(sorted_stats) > top_n:
        rest_count = sum(c for _, c in sorted_stats[top_n:])
        rest_pct = rest_count / total * 100
        entries.append(f"{'... others'.ljust(max_name_len)} {str(rest_count).rjust(max_count_len)} ({rest_pct:5.1f}%)")

    # Lay out in 5 columns (3 rows for 15 entries)
    col_width = max(len(e) for e in entries) + 2
    n_cols = 5
    n_rows = (len(entries) + n_cols - 1) // n_cols
    for row in range(n_rows):
        parts: list[str] = []
        for col in range(n_cols):
            idx = col * n_rows + row
            if idx < len(entries):
                parts.append(entries[idx].ljust(col_width))
        lines.append("  " + "".join(parts).rstrip())

    # Category totals as a single row
    cat_totals: dict[str, int] = {}
    for name, count in stats.items():
        cat = _categorize(name)
        cat_totals[cat] = cat_totals.get(cat, 0) + count

    lines.append("")
    cat_parts: list[str] = []
    for cat in ["RV32I", "M-ext", "F-ext", "NPU-int", "NPU-fp", "Other"]:
        if cat in cat_totals:
            count = cat_totals[cat]
            pct = count / total * 100
            cat_parts.append(f"{cat}: {count:,} ({pct:.0f}%)")
    lines.append("  " + "  |  ".join(cat_parts))

    return "\n".join(lines)
