/* MNIST digit recognition demo.
 *
 * Loads a 28x28 grayscale image from the test_image buffer (written by the
 * test harness), runs quantized inference using NPU instructions, and
 * prints the predicted digit to stdout.
 */

#include "nn_runtime.h"

/* Syscall wrappers (from common/syscalls.c) */
long write(int fd, const void *buf, long len);
void _exit(int code);

/* Image buffer: the test harness writes 784 bytes here before running.
 * Located in .bss so it has a known symbol address. */
uint8_t test_image[784];

/* Print a small non-negative integer (0-99) followed by newline. */
static void print_int(int n) {
    char buf[4];
    int len = 0;
    if (n >= 10) {
        buf[len++] = '0' + (n / 10);
    }
    buf[len++] = '0' + (n % 10);
    buf[len++] = '\n';
    write(1, buf, len);
}

int main(void) {
    int digit = inference(test_image);
    print_int(digit);
    return 0;
}
