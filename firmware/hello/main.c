/* Hello World: prints "Hello, World!" via the write syscall. */

long write(int fd, const void *buf, long len);

int main(void) {
    const char *msg = "Hello, World!\n";
    /* Calculate length manually (no libc strlen) */
    const char *p = msg;
    while (*p) p++;
    write(1, msg, p - msg);
    return 0;
}
