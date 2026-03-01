TARGET := mnist.elf

# Declare default target before including common (which defines its own targets)
all: $(TARGET)

include ../common/Makefile

OBJS := main.o nn_runtime.o syscalls.o

$(TARGET): $(START_OBJ) $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

main.o: main.c nn_runtime.h weights.h ../common/npu.h
	$(CC) $(CFLAGS) -c -o $@ $<

nn_runtime.o: nn_runtime.c nn_runtime.h weights.h ../common/npu.h
	$(CC) $(CFLAGS) -c -o $@ $<

syscalls.o: ../common/syscalls.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o *.elf $(START_OBJ)

.PHONY: all clean
