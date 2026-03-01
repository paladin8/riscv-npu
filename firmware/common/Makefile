# Common build settings for firmware targeting the RISC-V NPU emulator.
# Include this from individual firmware Makefiles.

COMMON_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

CC      := riscv64-unknown-elf-gcc
AS      := riscv64-unknown-elf-as
LD      := riscv64-unknown-elf-ld
OBJCOPY := riscv64-unknown-elf-objcopy

ARCH    := rv32imf
ABI     := ilp32f

CFLAGS  := -march=$(ARCH) -mabi=$(ABI) -O2 -nostdlib -ffreestanding -Wall
ASFLAGS := -march=$(ARCH) -mabi=$(ABI)
LDFLAGS := -T $(COMMON_DIR)linker.ld -nostdlib

START_OBJ := $(COMMON_DIR)start.o

$(COMMON_DIR)start.o: $(COMMON_DIR)start.s
	$(AS) $(ASFLAGS) -o $@ $<
