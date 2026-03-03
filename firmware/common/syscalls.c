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

static inline long syscall4(long n, long a0, long a1, long a2, long a3) {
    register long a7_r __asm__("a7") = n;
    register long a0_r __asm__("a0") = a0;
    register long a1_r __asm__("a1") = a1;
    register long a2_r __asm__("a2") = a2;
    register long a3_r __asm__("a3") = a3;
    __asm__ volatile("ecall"
                     : "+r"(a0_r)
                     : "r"(a7_r), "r"(a1_r), "r"(a2_r), "r"(a3_r)
                     : "memory");
    return a0_r;
}

/* Syscall numbers (Linux RISC-V ABI) */
#define SYS_openat 56
#define SYS_close  57
#define SYS_lseek  62

/* Open flags */
#define O_RDONLY   0
#define O_WRONLY   1
#define O_RDWR     2
#define O_CREAT    64
#define O_TRUNC    512
#define O_APPEND   1024

/* Special directory fd */
#define AT_FDCWD   (-100)

/* lseek whence */
#define SEEK_SET   0
#define SEEK_CUR   1
#define SEEK_END   2

long write(int fd, const void *buf, long len) {
    return syscall3(64, fd, (long)buf, len);
}

long read(int fd, void *buf, long len) {
    return syscall3(63, fd, (long)buf, len);
}

int openat(int dirfd, const char *pathname, int flags, int mode) {
    return (int)syscall4(SYS_openat, dirfd, (long)pathname, flags, mode);
}

int open(const char *pathname, int flags, int mode) {
    return openat(AT_FDCWD, pathname, flags, mode);
}

int close(int fd) {
    return (int)syscall1(SYS_close, fd);
}

long lseek(int fd, long offset, int whence) {
    return syscall3(SYS_lseek, fd, offset, whence);
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
