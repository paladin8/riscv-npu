"""Target description XML for GDB's RISC-V register layout.

Serves a minimal target description via ``qXfer:features:read:target.xml``
that declares the standard RISC-V CPU and FPU register features so GDB
can display registers with their ABI names and correct types.
"""

# Standard RISC-V ABI register names in order (x0-x31)
_GPR_NAMES: list[str] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
    "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
    "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",
]

# Standard RISC-V FPU ABI register names in order (f0-f31)
_FPR_NAMES: list[str] = [
    "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
    "fs0", "fs1", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5",
    "fa6", "fa7", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7",
    "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11",
]


def _build_target_xml() -> str:
    """Build the target description XML string.

    Returns:
        Complete XML target description with CPU and FPU features.
    """
    lines: list[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<!DOCTYPE target SYSTEM "gdb-target.dtd">')
    lines.append('<target version="1.0">')
    lines.append("  <architecture>riscv:rv32</architecture>")

    # CPU feature: x0-x31 + pc
    lines.append('  <feature name="org.gnu.gdb.riscv.cpu">')
    for i, name in enumerate(_GPR_NAMES):
        # ra and pc are code pointers; sp is a data pointer
        if name in ("ra", "pc"):
            reg_type = "code_ptr"
        elif name == "sp":
            reg_type = "data_ptr"
        else:
            reg_type = "int"
        lines.append(
            f'    <reg name="{name}" bitsize="32" regnum="{i}" type="{reg_type}" />'
        )
    # pc is register 32
    lines.append(
        '    <reg name="pc" bitsize="32" regnum="32" type="code_ptr" />'
    )
    lines.append("  </feature>")

    # FPU feature: f0-f31 + fflags + frm + fcsr
    lines.append('  <feature name="org.gnu.gdb.riscv.fpu">')
    for i, name in enumerate(_FPR_NAMES):
        regnum = 33 + i
        lines.append(
            f'    <reg name="{name}" bitsize="32" regnum="{regnum}" type="ieee_single" />'
        )
    # FPU CSRs
    lines.append(
        '    <reg name="fflags" bitsize="32" regnum="65" type="int" />'
    )
    lines.append(
        '    <reg name="frm" bitsize="32" regnum="66" type="int" />'
    )
    lines.append(
        '    <reg name="fcsr" bitsize="32" regnum="67" type="int" />'
    )
    lines.append("  </feature>")

    lines.append("</target>")
    return "\n".join(lines) + "\n"


TARGET_XML: str = _build_target_xml()
"""Complete target description XML for ``qXfer:features:read``."""
