TARGET := fibonacci.elf

# Declare default target before including common (which defines its own targets)
all: $(TARGET)

include ../common/Makefile

$(TARGET): $(START_OBJ) main.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

main.o: main.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o *.elf $(START_OBJ)

.PHONY: all clean
