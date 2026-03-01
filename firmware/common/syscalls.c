/* Minimal syscall wrappers for the RISC-V NPU emulator.
 *
 * Uses inline assembly to invoke ECALL with the Linux RISC-V ABI:
 *   a7 = syscall number
 *   a0-a2 = arguments
 *   a0 = return value
 */

static inline long syscall1(long n, long a0) {
    register long a7_r __asm__("a7") = n;
    register long a0_r __asm__("a0") = a0;
    __asm__ volatile("ecall"
                     : "+r"(a0_r)
                     : "r"(a7_r)
                     : "memory");
    return a0_r;
}

static inline long syscall3(long n, long a0, long a1, long a2) {
    register long a7_r __asm__("a7") = n;
    register long a0_r __asm__("a0") = a0;
    register long a1_r __asm__("a1") = a1;
    register long a2_r __asm__("a2") = a2;
    __asm__ volatile("ecall"
                     : "+r"(a0_r)
                     : "r"(a7_r), "r"(a1_r), "r"(a2_r)
                     : "memory");
    return a0_r;
}

long write(int fd, const void *buf, long len) {
    return syscall3(64, fd, (long)buf, len);
}

long read(int fd, void *buf, long len) {
    return syscall3(63, fd, (long)buf, len);
}

void _exit(int code) {
    syscall1(93, code);
    __builtin_unreachable();
}

int getchar(void) {
    char ch;
    long n = read(0, &ch, 1);
    if (n <= 0) return -1;
    return (unsigned char)ch;
}

int putchar(int c) {
    char ch = (char)c;
    write(1, &ch, 1);
    return c;
}

long puts(const char *s) {
    const char *p = s;
    while (*p) p++;
    write(1, s, p - s);
    putchar('\n');
    return 0;
}
