"""Tests for GDB target description XML."""

import xml.etree.ElementTree as ET

from riscv_npu.gdb.target_xml import TARGET_XML


def _parse_xml() -> ET.Element:
    """Parse the target XML and return the root element."""
    return ET.fromstring(TARGET_XML)


def test_xml_is_well_formed() -> None:
    """Parses without errors."""
    root = _parse_xml()
    assert root is not None
    assert root.tag == "target"


def test_xml_has_architecture() -> None:
    """Contains <architecture>riscv:rv32</architecture>."""
    root = _parse_xml()
    arch = root.find("architecture")
    assert arch is not None
    assert arch.text == "riscv:rv32"


def test_xml_has_cpu_feature() -> None:
    """Has org.gnu.gdb.riscv.cpu feature."""
    root = _parse_xml()
    features = root.findall("feature")
    names = [f.get("name") for f in features]
    assert "org.gnu.gdb.riscv.cpu" in names


def test_xml_has_fpu_feature() -> None:
    """Has org.gnu.gdb.riscv.fpu feature."""
    root = _parse_xml()
    features = root.findall("feature")
    names = [f.get("name") for f in features]
    assert "org.gnu.gdb.riscv.fpu" in names


def test_xml_gpr_count() -> None:
    """CPU feature has 33 registers (x0-x31 + pc)."""
    root = _parse_xml()
    for feature in root.findall("feature"):
        if feature.get("name") == "org.gnu.gdb.riscv.cpu":
            regs = feature.findall("reg")
            assert len(regs) == 33
            return
    raise AssertionError("CPU feature not found")


def test_xml_fpr_count() -> None:
    """FPU feature has 35 registers (f0-f31 + fflags + frm + fcsr)."""
    root = _parse_xml()
    for feature in root.findall("feature"):
        if feature.get("name") == "org.gnu.gdb.riscv.fpu":
            regs = feature.findall("reg")
            assert len(regs) == 35
            return
    raise AssertionError("FPU feature not found")


def test_xml_pc_is_code_ptr() -> None:
    """PC register has type='code_ptr'."""
    root = _parse_xml()
    for feature in root.findall("feature"):
        if feature.get("name") == "org.gnu.gdb.riscv.cpu":
            for reg in feature.findall("reg"):
                if reg.get("name") == "pc":
                    assert reg.get("type") == "code_ptr"
                    return
    raise AssertionError("PC register not found")


def test_xml_regnums_contiguous() -> None:
    """Regnums go 0-67 without gaps."""
    root = _parse_xml()
    regnums: list[int] = []
    for feature in root.findall("feature"):
        for reg in feature.findall("reg"):
            regnum = reg.get("regnum")
            assert regnum is not None
            regnums.append(int(regnum))

    regnums.sort()
    assert regnums == list(range(68))


def test_xml_has_abi_names() -> None:
    """Registers use ABI names (ra, sp, a0, etc.)."""
    root = _parse_xml()
    reg_names: set[str] = set()
    for feature in root.findall("feature"):
        for reg in feature.findall("reg"):
            name = reg.get("name")
            assert name is not None
            reg_names.add(name)

    # Check key ABI names
    for expected in ["zero", "ra", "sp", "gp", "tp", "a0", "a7", "s0", "t0", "pc"]:
        assert expected in reg_names, f"Missing ABI name: {expected}"


def test_xml_fpu_register_types() -> None:
    """Float registers (f0-f31) have type='ieee_single'."""
    root = _parse_xml()
    fpr_names = {"ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
                 "fs0", "fs1", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5",
                 "fa6", "fa7", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7",
                 "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11"}
    for feature in root.findall("feature"):
        if feature.get("name") == "org.gnu.gdb.riscv.fpu":
            for reg in feature.findall("reg"):
                name = reg.get("name")
                if name in fpr_names:
                    assert reg.get("type") == "ieee_single", f"{name} should be ieee_single"
            return
    raise AssertionError("FPU feature not found")
