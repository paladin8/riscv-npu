/* File I/O demo: create a file, write, seek, read back, and print.
 *
 * Uses the emulator's openat/close/lseek/read/write syscalls.
 */

/* Declarations from ../common/syscalls.c (included as .o) */
long write(int fd, const void *buf, long len);
long read(int fd, void *buf, long len);
void _exit(int code);
long puts(const char *s);
int putchar(int c);
int open(const char *pathname, int flags, int mode);
int close(int fd);
long lseek(int fd, long offset, int whence);

#define O_RDWR   2
#define O_CREAT  64
#define O_TRUNC  512
#define SEEK_SET 0

static long strlen(const char *s) {
    const char *p = s;
    while (*p) p++;
    return p - s;
}

int main(void) {
    const char *filename = "test_output.txt";
    const char *message = "Hello from RISC-V file I/O!";
    char buf[64];

    /* Open file for read+write, creating and truncating */
    int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        puts("ERROR: open failed");
        _exit(1);
    }

    /* Write message */
    long msg_len = strlen(message);
    long written = write(fd, message, msg_len);
    if (written != msg_len) {
        puts("ERROR: write failed");
        _exit(2);
    }

    /* Seek back to start */
    long pos = lseek(fd, 0, SEEK_SET);
    if (pos != 0) {
        puts("ERROR: lseek failed");
        _exit(3);
    }

    /* Read back */
    long nread = read(fd, buf, msg_len);
    if (nread != msg_len) {
        puts("ERROR: read failed");
        _exit(4);
    }

    /* Close file */
    close(fd);

    /* Print what we read */
    puts("Read back from file:");
    write(1, buf, nread);
    putchar('\n');

    return 0;
}
