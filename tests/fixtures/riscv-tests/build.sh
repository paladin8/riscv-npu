#!/bin/bash
# Build riscv-tests ELF binaries for compliance testing.
#
# Prerequisites:
#   - riscv64-unknown-elf-gcc installed
#   - git
#
# Usage: ./build.sh
#
# This script clones the riscv-tests repo, builds rv32ui-p-*, rv32um-p-*,
# and rv32uf-p-* test binaries, and copies them to this directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="/tmp/riscv-tests-build-$$"

echo "Cloning riscv-tests..."
git clone --depth 1 https://github.com/riscv-software-src/riscv-tests.git "$BUILD_DIR"
cd "$BUILD_DIR"
git submodule update --init

echo "Building rv32ui-p-* tests..."
cd isa
RV32UI_TESTS="simple add addi and andi auipc beq bge bgeu blt bltu bne fence_i jal jalr lb lbu lh lhu lw ld_st lui ma_data or ori sb sh sw st_ld sll slli slt slti sltiu sltu sra srai srl srli sub xor xori"
for t in $RV32UI_TESTS; do
    make XLEN=32 "rv32ui-p-$t"
done

echo "Building rv32um-p-* tests..."
RV32UM_TESTS="div divu mul mulh mulhsu mulhu rem remu"
for t in $RV32UM_TESTS; do
    make XLEN=32 "rv32um-p-$t"
done

echo "Building rv32uf-p-* tests..."
RV32UF_TESTS="fadd fdiv fclass fcmp fcvt fcvt_w fmadd fmin ldst move recoding"
for t in $RV32UF_TESTS; do
    make XLEN=32 "rv32uf-p-$t"
done

echo "Copying binaries to $SCRIPT_DIR..."
cp rv32ui-p-* "$SCRIPT_DIR/" 2>/dev/null || true
cp rv32um-p-* "$SCRIPT_DIR/" 2>/dev/null || true
cp rv32uf-p-* "$SCRIPT_DIR/" 2>/dev/null || true
# Remove dump files if any
rm -f "$SCRIPT_DIR"/*.dump

echo "Cleaning up..."
rm -rf "$BUILD_DIR"

echo "Done. Test binaries are in $SCRIPT_DIR/"
ls "$SCRIPT_DIR"/rv32ui-p-* "$SCRIPT_DIR"/rv32um-p-* "$SCRIPT_DIR"/rv32uf-p-* 2>/dev/null | wc -l
echo "test binaries available."
