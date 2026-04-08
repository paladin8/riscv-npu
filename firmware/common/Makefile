# Common build settings for firmware targeting the RISC-V NPU emulator.
# Include this from individual firmware Makefiles.

COMMON_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

# Auto-detect toolchain prefix: riscv64-unknown-elf- or riscv64-elf-
CROSS_PREFIX := $(shell which riscv64-unknown-elf-gcc >/dev/null 2>&1 \
	&& echo riscv64-unknown-elf- || echo riscv64-elf-)

CC      := $(CROSS_PREFIX)gcc
AS      := $(CROSS_PREFIX)as
LD      := $(CROSS_PREFIX)ld
OBJCOPY := $(CROSS_PREFIX)objcopy

ARCH    := rv32imf
ABI     := ilp32f

CFLAGS  := -march=$(ARCH) -mabi=$(ABI) -O2 -g -nostdlib -ffreestanding -Wall
ASFLAGS := -march=$(ARCH) -mabi=$(ABI)
LDFLAGS := -T $(COMMON_DIR)linker.ld -nostdlib

START_OBJ := $(COMMON_DIR)start.o

$(COMMON_DIR)start.o: $(COMMON_DIR)start.s
	$(AS) $(ASFLAGS) -o $@ $<
