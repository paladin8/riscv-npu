TARGET := uart-hello.elf

all: $(TARGET)

include ../common/Makefile

OBJS := main.o syscalls.o

$(TARGET): $(START_OBJ) $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

main.o: main.c
	$(CC) $(CFLAGS) -c -o $@ $<

syscalls.o: ../common/syscalls.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o *.elf $(START_OBJ)

.PHONY: all clean
